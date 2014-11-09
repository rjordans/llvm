//===-- LoopPipelinePass.cpp - Loop pipelining pass -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements a high-level loop pipeliner.
//
// Coarsely based on
//
//     Towards a Source Level Compiler: Source Level Modulo Scheduling
//     Ben-Asher and Meisler,
//     Program Analysis and Compilation, Theory and Practice
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"

#include <unordered_map>

using namespace llvm;

#define DEBUG_TYPE "loop-pipeline"

STATISTIC(LoopsAnalyzed, "Number of loops analyzed for high-level software pipelining");
STATISTIC(LoopsPipelined, "Number of loops pipelined");

//***************************************************************************
// Pass registration and work list construction
//***************************************************************************
// Helper function (copied from LoopVectorize.cpp)
static void addInnerLoop(Loop &L, SmallVectorImpl<Loop *> &V) {
  if (L.empty())
    return V.push_back(&L);

  for (Loop *InnerL : L)
    addInnerLoop(*InnerL, V);
}

namespace {
  // Container for gathered information about the current loop body
  struct LoopPipelineInfo {
    BasicBlock *LoopBody;

    CodeMetrics CM;

    std::unordered_map<Instruction *, unsigned> ASAPtimes;
    std::unordered_map<Instruction *, unsigned> ALAPtimes;

    unsigned LoopLatency;
  };

  class LoopPipeline : public FunctionPass {
  public:
    static char ID; // Pass identification, replacement for typeid
    LoopPipeline() : FunctionPass(ID) {
      initializeLoopPipelinePass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override {
      DA = &getAnalysis<DependenceAnalysis>();
      LI = &getAnalysis<LoopInfo>();
      SE = &getAnalysis<ScalarEvolution>();
      TTI = &getAnalysis<TargetTransformInfo>();

      DEBUG(dbgs() << "\nLP: Hello from loop-pipeline: ";
            dbgs().write_escaped(F.getName()) << '\n');

      // Find inner loops, build worklist, copied from LoopVectorize.cpp
      SmallVector<Loop *, 8> Worklist;

      for (Loop *L : *LI)
        addInnerLoop(*L, Worklist);

      LoopsAnalyzed += Worklist.size();

      // Process each inner loop
      bool Changed = false;
      while (!Worklist.empty())
        Changed |= processLoop(Worklist.pop_back_val());

      return Changed;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<DependenceAnalysis>();
      AU.addRequired<LoopInfo>();
      AU.addRequired<ScalarEvolution>();
      AU.addRequired<TargetTransformInfo>();
    }

  private:
    DependenceAnalysis *DA;
    LoopInfo *LI;
    ScalarEvolution *SE;
    TargetTransformInfo *TTI;

    bool processLoop(Loop *L);
    bool canPipelineLoop(Loop *L, LoopPipelineInfo &LPI);
    unsigned computeRecurrenceMII(Loop *L, LoopPipelineInfo &LPI);
    unsigned computeResourceMII(LoopPipelineInfo &LPI);
    unsigned getInstructionCost(const Instruction *I) const;
    bool transformLoop(Loop *L, unsigned MII, LoopPipelineInfo &LPI);
  };
}

char LoopPipeline::ID = 0;
INITIALIZE_PASS(LoopPipeline, "loop-pipeline",
                "Software pipeline inner-loops", false, false)

FunctionPass *llvm::createLoopPipelinePass() {
  return new LoopPipeline();
}

//***************************************************************************
// Do it!
//***************************************************************************
//
// Process a single inner loop
//
bool LoopPipeline::processLoop(Loop *L) {
  assert(L->empty() && "Only process inner loops");

  BasicBlock *LoopBody = L->getBlocks()[0];
  DEBUG(dbgs() << "LP: processing '" << LoopBody->getName() << "'\n");

  // Check if loop is a candidate
  LoopPipelineInfo LPI;
  if( !canPipelineLoop(L, LPI) ) {
    DEBUG(dbgs() << "LP: failed to pipeline loop\n");
    return false;
  }

  // Estimate RecMII
  unsigned RecMII = computeRecurrenceMII(L, LPI);
  if( RecMII == 0 ) {
    DEBUG(dbgs() << "LP: failed to compute RecMII\n");
    return false;
  }
  DEBUG(dbgs() << "LP: Found recurrence MII of " << RecMII << '\n');

  // Estimate ResMII
  unsigned ResMII = computeResourceMII(LPI);
  DEBUG(dbgs() << "LP: Found resource MII of " << ResMII << '\n');


  // Decide MII
  unsigned MII = std::max(RecMII, ResMII);
  // Check if MII <= ASAP loop latency
  if( MII > LPI.LoopLatency) {
    DEBUG(dbgs() << "LP: MII larger than ASAP latency, no benefits expected for loop pipelining\n");
    return false;
  }

  // Perform actual software pipelining
  return transformLoop(L, MII, LPI);
}

//***************************************************************************
// Loop structure checks
//***************************************************************************
//
// Determine if a loop is a valid candidate for software pipelining
//
bool LoopPipeline::canPipelineLoop(Loop *L, LoopPipelineInfo &LPI) {
  // Check if loop body has no control flow (single BasicBlock)
  unsigned NumBlocks = L->getNumBlocks();
  if( NumBlocks != 1 ) {
    DEBUG(dbgs() << "LP: Can only software-pipeline simple loops\n");
    return false;
  }

  // Check if loop body includes function calls
  // CodeMetrics will ignore intrinsics that are expected to be lowered
  //  directly into operations
  // FIXME v3.6 updated interface, needs AssumptionTracker
  BasicBlock *B = L->getBlocks()[0];

  LPI.CM.analyzeBasicBlock(B, *TTI);
  if( LPI.CM.NumCalls > 0 ) {
    // NumCalls also includes inline assembly
    DEBUG(dbgs() << "LP: Can not software-pipeline loops with function calls\n");
    return false;
  }

  if( LPI.CM.notDuplicatable ) {
    DEBUG(dbgs() << "LP: Loop contains operations marked 'noduplicate'\n");
    return false;
  }
  // TODO: Add more checks
  // - Iteration count?
  // - Constant stride
  // - Loop optimization hints
  // - ...

  // Check loop for anti-dependencies through store-load combinations
  // TODO: add a dependency breaking pass
  LPI.LoopBody = L->getBlocks()[0];

  // Build lists of read and write operations for memory dependence checking
  SmallVector<Instruction*, 2> Writes;
  SmallVector<Instruction*, 4> Reads;
  for(auto I = LPI.LoopBody->begin(), E = LPI.LoopBody->end(); I != E; I++) {
    Instruction *Inst = cast<Instruction>(I);

    if( Inst->mayReadFromMemory() )
      Reads.push_back(Inst);
    if( Inst->mayWriteToMemory() )
      Writes.push_back(Inst);
  }

  // Find memory dependencies
  for(Instruction *ST : Writes) {
    for(Instruction *LD : Reads) {
      // Check for Store -> Load dependencies
      // Focus on dependencies inside the loop (3rd argument = false)
      Dependence *D = DA->depends(LD, ST, false);

      // Skip if no dependency is found
      if( !D )
        continue;

      // Skip loops where memory dependencies could not be determined
      if( D->isConfused() ) {
        DEBUG(dbgs() << "LP: Loop dependence checker confused, giving up.\n");
        return false;
      }

      // Check if we could compute a distance
      const SCEV *Distance = D->getDistance(D->getLevels());
      if( !Distance ) {
        DEBUG(dbgs() << "LP: Could not compute dependence distance, giving up.\n");
        return false;
      }

      // Check if distance is in the opposite direction of the loop increment
      if( !SE->isKnownNonNegative(Distance) ) {
        DEBUG(dbgs() << "LP: Loop carried anti-dependency found, giving up.\n");
        return false;
      }
    }
  }

  return true;
}

//***************************************************************************
// Computation of the recurrence constraint on Minimal Initiation Interval
//***************************************************************************
//
// Find the minimal initiation interval for the loop recurrences
//
unsigned LoopPipeline::computeRecurrenceMII(Loop *L, LoopPipelineInfo &LPI) {
  // At this point, all loop carried dependencies are modelled through phi nodes
  // Find the maximum length cycle through these phi nodes to get the RecMII
  BasicBlock *LoopBody = LPI.LoopBody;

  // Compute ASAP times for all operations in the loop body
  unsigned LoopLatency = 0;
  for(auto I = LoopBody->begin(), E = LoopBody->end(); I != E; I++) {
    unsigned OperandsASAP = 0;

    // find I's in-loop operands
    if( !isa<PHINode>(I) ) {
      for(auto &O : I->operands()) {
        Instruction *Op = dyn_cast<Instruction>(O);
        if( !Op ) continue;

        if( Op->getParent() == LoopBody ) {
          // get maximum schedule time
          OperandsASAP = std::max(OperandsASAP, LPI.ASAPtimes[Op] + getInstructionCost(Op));
        }
      }
    }

    LPI.ASAPtimes[I] = OperandsASAP;

    // keep track of loop latency for ALAPtimes computation
    LoopLatency = std::max(LoopLatency, OperandsASAP + getInstructionCost(I));
  }

  LPI.LoopLatency = LoopLatency;

  // Compute ALAP times for all operations in the loop body in reverse
  for(auto I = LoopBody->end(), E = LoopBody->begin(); I != E;) {
    Instruction *II = --I;
    unsigned DepsALAP = LoopLatency;

    // find I's in-loop users
    for(auto D : II->users()) {
      Instruction *Dep = dyn_cast<Instruction>(D);
      if( !Dep ) continue;

      if( Dep->getParent() == LoopBody && LPI.ALAPtimes.find(Dep) != LPI.ALAPtimes.end() ) {
        // get minimum schedule time
        DepsALAP = std::min(DepsALAP, LPI.ALAPtimes[Dep]-getInstructionCost(Dep));
      }
    }

    LPI.ALAPtimes[II] = DepsALAP;
  }

  // Find the phi node that depends on the highest latency operation
  unsigned MaxCycleLength = 0;
  for(auto I = LoopBody->begin(), E = LoopBody->end(); I != E; I++) {
    const PHINode *Phi = dyn_cast<PHINode>(I);
    if( !Phi )
      continue;

    for(unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
      Instruction *II = dyn_cast<Instruction>(Phi->getIncomingValue(i));
      if( !II ) continue;

      if( LPI.ALAPtimes[I] <= LPI.ASAPtimes[II] )
        MaxCycleLength = std::max(
            MaxCycleLength,
            LPI.ASAPtimes[II] - LPI.ALAPtimes[I] + getInstructionCost(II)
          );
    }
  }

  return MaxCycleLength;
}

//***************************************************************************
// Computation of the resource constraint on Minimal Initiation Interval
//***************************************************************************
//
// Find the minimal initiation interval given the processor resources as
// provided by TTI.
//
// Uses NumVectorInsts and NumInsts from CodeMetrics for FU utilization
// estimation
//
// FIXME: This has a very limited view of the processor resources
//
// - Vectorized IR code is assumed to be executed on vector function units
// - Default values of NoTTI impose no constraints on resources
// - No constraint for the number of parallel memory accesses for loops with
//   high memory bandwidth
//
// Make this conditional to observe the effect of adding resource constraints
// versus the approach taken in Ben-Asher & Meisler
//
// Extensions:
// - Use Number of cut edges (phi nodes) for new graph to estimate RF pressure
unsigned LoopPipeline::computeResourceMII(LoopPipelineInfo &LPI) {
  unsigned ResMII = 0;
  unsigned NumScalarInsts = LPI.CM.NumInsts - LPI.CM.NumVectorInsts;
  const unsigned ScalarFUCount = TTI->getScalarFunctionUnitCount();
  const unsigned VectorFUCount = TTI->getVectorFunctionUnitCount();

  DEBUG(dbgs() << "LP: NumInsts=" << LPI.CM.NumInsts
        << ", NumVectorInsts=" << LPI.CM.NumVectorInsts << '\n');

  DEBUG(dbgs() << "LP: ScalarFUs=" << ScalarFUCount
        << ", VectorFUs=" << VectorFUCount << '\n');

  if(ScalarFUCount) {
    ResMII = (NumScalarInsts + ScalarFUCount - 1) / ScalarFUCount;
  }

  if(VectorFUCount) {
    ResMII = std::max(
      ResMII,
      (LPI.CM.NumVectorInsts + VectorFUCount - 1) / VectorFUCount);
  }

  return ResMII;
}

//***************************************************************************
// Instruction cost model used during the different scheduling runs
//***************************************************************************
//
// FIXME: Large amounts of code duplication from lib/Analysis/CostModel.cpp
// A more generalized cost model could be very useful as loop vectorization
// also seems to implement its own version.
//
// - Needs a lot of improvement to better reflect the operation latencies and
//   possibly more hooks into TTI
static TargetTransformInfo::OperandValueKind getOperandInfo(Value *V) {
  TargetTransformInfo::OperandValueKind OpInfo =
    TargetTransformInfo::OK_AnyValue;

  // Check for a splat of a constant or for a non uniform vector of constants.
  if (isa<ConstantVector>(V) || isa<ConstantDataVector>(V)) {
    OpInfo = TargetTransformInfo::OK_NonUniformConstantValue;
    if (cast<Constant>(V)->getSplatValue() != nullptr)
      OpInfo = TargetTransformInfo::OK_UniformConstantValue;
  }

  return OpInfo;
}

unsigned LoopPipeline::getInstructionCost(const Instruction *I) const {

  switch (I->getOpcode()) {
  case Instruction::GetElementPtr:{
    Type *ValTy = I->getOperand(0)->getType()->getPointerElementType();
    return TTI->getAddressComputationCost(ValTy);
  }

  case Instruction::Ret:
  case Instruction::PHI:
  case Instruction::Br: {
    return TTI->getCFInstrCost(I->getOpcode());
  }
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    TargetTransformInfo::OperandValueKind Op1VK =
      getOperandInfo(I->getOperand(0));
    TargetTransformInfo::OperandValueKind Op2VK =
      getOperandInfo(I->getOperand(1));
    return TTI->getArithmeticInstrCost(I->getOpcode(), I->getType(), Op1VK,
                                       Op2VK);
  }
  case Instruction::Select: {
    const SelectInst *SI = cast<SelectInst>(I);
    Type *CondTy = SI->getCondition()->getType();
    return TTI->getCmpSelInstrCost(I->getOpcode(), I->getType(), CondTy);
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    Type *ValTy = I->getOperand(0)->getType();
    return TTI->getCmpSelInstrCost(I->getOpcode(), ValTy);
  }
  case Instruction::Store: {
    const StoreInst *SI = cast<StoreInst>(I);
    Type *ValTy = SI->getValueOperand()->getType();
    return TTI->getMemoryOpCost(I->getOpcode(), ValTy,
                                SI->getAlignment(),
                                SI->getPointerAddressSpace());
  }
  case Instruction::Load: {
    const LoadInst *LI = cast<LoadInst>(I);
    return TTI->getMemoryOpCost(I->getOpcode(), I->getType(),
                                LI->getAlignment(),
                                LI->getPointerAddressSpace());
  }
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::FPExt:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
  case Instruction::Trunc:
  case Instruction::FPTrunc:
  case Instruction::BitCast:
  case Instruction::AddrSpaceCast: {
    Type *SrcTy = I->getOperand(0)->getType();
    return TTI->getCastInstrCost(I->getOpcode(), I->getType(), SrcTy);
  }
  case Instruction::ExtractElement: {
    const ExtractElementInst * EEI = cast<ExtractElementInst>(I);
    ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1));
    unsigned Idx = -1;
    if (CI)
      Idx = CI->getZExtValue();

    return TTI->getVectorInstrCost(I->getOpcode(),
                                   EEI->getOperand(0)->getType(), Idx);
  }
  case Instruction::InsertElement: {
    const InsertElementInst * IE = cast<InsertElementInst>(I);
    ConstantInt *CI = dyn_cast<ConstantInt>(IE->getOperand(2));
    unsigned Idx = -1;
    if (CI)
      Idx = CI->getZExtValue();
    return TTI->getVectorInstrCost(I->getOpcode(),
                                   IE->getType(), Idx);
  }
  case Instruction::ShuffleVector: {
    // Randomly selected value TargetTransformInfo doesn't have a generic shuffle cost...
    return 4;
  }
  case Instruction::Call:
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
      SmallVector<Type*, 4> Tys;
      for (unsigned J = 0, JE = II->getNumArgOperands(); J != JE; ++J)
        Tys.push_back(II->getArgOperand(J)->getType());

      return TTI->getIntrinsicInstrCost(II->getIntrinsicID(), II->getType(),
                                        Tys);
    }
    return -1;
  default:
    // We don't have any information on this instruction.
    return -1;
  }
}

//***************************************************************************
// Schedule operations and generate transformed loop
//***************************************************************************
//
bool LoopPipeline::transformLoop(Loop *L, unsigned MII, LoopPipelineInfo &LPI) {
  DEBUG(dbgs() << "LP: software pipelining loop with MII=" << MII << '\n');
  LPI.LoopBody->dump();

  DEBUG(dbgs() << "LP: Schedule freedom (ASAP, ALAP, Mobility, Cost)\n  S L M C\n");
  for(auto I=LPI.LoopBody->begin(), E=LPI.LoopBody->end(); I != E; I++) {
    unsigned ASAP = LPI.ASAPtimes[I],
             ALAP = LPI.ALAPtimes[I],
             MOB = ALAP - ASAP;
    DEBUG(dbgs() << "  " << ASAP << " " << ALAP << " " << MOB << " " << getInstructionCost(I) << '\n');
  }
  // Construct node ordering
  // Schedule
  // Generate new inner loop
  // Construct prologue/epilogue

  // Done
  LoopsPipelined++;
  return true;
}

