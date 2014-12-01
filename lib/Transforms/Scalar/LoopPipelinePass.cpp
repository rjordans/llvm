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
#include <set>

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

static unsigned getInstructionCost(const Instruction *I, const TargetTransformInfo *TTI);

namespace {
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

    class InstructionTrace {
    public:
      InstructionTrace() : weight(0) {}
      InstructionTrace(const InstructionTrace &IT) : trace(IT.trace), weight(IT.weight) {}

      // Add instruction to trace
      void add(Instruction *I, TargetTransformInfo *TTI) {
        trace.push_back(I);
        weight += getInstructionCost(I, TTI);
      }

      // Access methods
      unsigned getWeight() const {
        return weight;
      }
      const SmallVector<Instruction *, 8> &data() const {
        return trace;
      }
      size_t size() const {
        return trace.size();
      }

      // Find
      const Instruction *find(Instruction *I) const {
		for(auto II=trace.begin(), E=trace.end(); II != E; II++)
		  if( *II == I )
			return *II;
		return nullptr;
      }

      // Compare for insertion into set
      bool operator< (const InstructionTrace &T) const {
        return weight <= T.weight && this != &T;
      }
    private:
      SmallVector<Instruction *, 8> trace;
      unsigned weight;
    };

    typedef std::unordered_map<Instruction *, unsigned> PartialSchedule;
    typedef std::set<InstructionTrace> CycleSet;

    bool processLoop(Loop *L);
    bool canPipelineLoop(Loop *L, CodeMetrics &CM);

    void getPhiCycles(Instruction *I, const PHINode *Phi, InstructionTrace trace, CycleSet &cycles);
    unsigned computeRecurrenceMII(Loop *L, CycleSet &cycles);
    unsigned computeResourceMII(CodeMetrics &CM);

    bool transformLoop(Loop *L, unsigned MII, CycleSet &cycles );

    unsigned scheduleASAP(BasicBlock *B, PartialSchedule &schedule);
    void scheduleALAP(BasicBlock *B, unsigned LastOperationStart, PartialSchedule &schedule);
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
  CodeMetrics CM;
  if( !canPipelineLoop(L, CM) ) {
    DEBUG(dbgs() << "LP: failed to pipeline loop\n");
    return false;
  }

  // Estimate RecMII and obtain a list of loop carried dependencies
  CycleSet cycles;
  unsigned RecMII = computeRecurrenceMII(L, cycles);
  if( RecMII == 0 ) {
    DEBUG(dbgs() << "LP: failed to compute RecMII\n");
    return false;
  }
  DEBUG(dbgs() << "LP: Found recurrence MII of " << RecMII << '\n');

  // Estimate ResMII
  unsigned ResMII = computeResourceMII(CM);
  DEBUG(dbgs() << "LP: Found resource MII of " << ResMII << '\n');

  // Decide MII
  unsigned MII = std::max(RecMII, ResMII);

  // Perform actual software pipelining
  return transformLoop(L, MII, cycles);
}

//***************************************************************************
// Loop structure checks
//***************************************************************************
//
// Determine if a loop is a valid candidate for software pipelining
//
bool LoopPipeline::canPipelineLoop(Loop *L, CodeMetrics &CM) {
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
  BasicBlock *LoopBody = L->getBlocks()[0];

  CM.analyzeBasicBlock(LoopBody, *TTI);
  if( CM.NumCalls > 0 ) {
    // NumCalls also includes inline assembly
    DEBUG(dbgs() << "LP: Can not software-pipeline loops with function calls\n");
    return false;
  }

  if( CM.notDuplicatable ) {
    DEBUG(dbgs() << "LP: Loop contains operations marked 'noduplicate'\n");
    return false;
  }
  // TODO: Add more checks
  // - Iteration count?
  // - Constant stride
#if 0
  // Analyzable loop - disabled, this seems to strong a limitation
  // Effectively limits us to forward loops with +1 increment
  if( !L->getCanonicalInductionVariable() ) {
    DEBUG(dbgs() << "LP: Loop not in analyzable form, try running Induction Variable Simplification first\n");
    return false;
  }
#endif

  // - Loop optimization hints
  // - ...

  // Check loop for anti-dependencies through store-load combinations
  // TODO: add a dependency breaking pass

  // Build lists of read and write operations for memory dependence checking
  SmallVector<Instruction*, 2> Writes;
  SmallVector<Instruction*, 4> Reads;
  for(auto I = LoopBody->begin(), E = LoopBody->end(); I != E; I++) {
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
// Enumerate loop recurrences through depth-first search over each back-edge
//

// Helper function to find loop dependency cycles through phi nodes
void LoopPipeline::getPhiCycles(Instruction *I, const PHINode *Phi,
                         InstructionTrace trace,
                         CycleSet &cycles) {
  // stay within the loop body
  if( I->getParent() != Phi->getParent() )
	return;

  // found a cycle when we end up at our start point
  if( I == Phi && trace.size() != 0 ) {
    cycles.insert(trace);
    return;
  }

  // found a cycle not passing through the currently considered phi-node
  // for example: a -> b -> c -> b, this can only happen if b is a phi-node
  if( isa<PHINode>(I) && trace.find(I) ) {
    return;
  }

  // add current instruction and check cycles for each operand of the instruction
  trace.add(I, TTI);
  if( isa<PHINode>(I) ) {
    PHINode *P = cast<PHINode>(I);

    for(unsigned i = 0; i < P->getNumIncomingValues(); i++) {
      Instruction *II = dyn_cast<Instruction>(P->getIncomingValue(i));
      if(II) getPhiCycles(II, Phi, trace, cycles);
    }
  } else {
    for(auto &O : I->operands() ) {
      Instruction *II = dyn_cast<Instruction>(O);
      if(II) getPhiCycles(II, Phi, trace, cycles);
    }
  }
}

unsigned LoopPipeline::computeRecurrenceMII(Loop *L, CycleSet &cycles) {
  // At this point, all loop carried dependencies are modelled through phi nodes
  // Find the maximum length cycle through these phi nodes to get the RecMII
  BasicBlock *LoopBody = L->getBlocks()[0];

  // Enumerate the loop carried dependency cycles
  for(auto I = LoopBody->begin(), E = LoopBody->end(); I != E; I++) {
    PHINode *Phi = dyn_cast<PHINode>(I);
    if( !Phi )
      continue;

    InstructionTrace trace;
    getPhiCycles(Phi, Phi, trace, cycles);
  }

  // cycles are sorted by weight, last cycle in the set has highest weight
  DEBUG(dbgs() << "LP: Found " << cycles.size() << " cycle(s)\n");
  return cycles.rbegin()->getWeight();
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
// This has a very limited view of the processor resources
// - Vectorized IR code is assumed to be executed on vector function units
// - Default values of NoTTI impose no constraints on resources
// - No constraint for the number of parallel memory accesses for loops with
//   high memory bandwidth
//
// TODO:
// Make this conditional to observe the effect of adding resource constraints
// versus the approach taken in Ben-Asher & Meisler
unsigned LoopPipeline::computeResourceMII(CodeMetrics &CM) {
  unsigned ResMII = 0;
  unsigned NumScalarInsts = CM.NumInsts - CM.NumVectorInsts;
  const unsigned ScalarFUCount = TTI->getScalarFunctionUnitCount();
  const unsigned VectorFUCount = TTI->getVectorFunctionUnitCount();

  DEBUG(dbgs() << "LP: NumInsts=" << CM.NumInsts
        << ", NumVectorInsts=" << CM.NumVectorInsts << '\n');

  DEBUG(dbgs() << "LP: ScalarFUs=" << ScalarFUCount
        << ", VectorFUs=" << VectorFUCount << '\n');

  if(ScalarFUCount) {
    ResMII = (NumScalarInsts + ScalarFUCount - 1) / ScalarFUCount;
  }

  if(VectorFUCount) {
    ResMII = std::max(
      ResMII,
      (CM.NumVectorInsts + VectorFUCount - 1) / VectorFUCount);
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

static unsigned getInstructionCost(const Instruction *I, const TargetTransformInfo *TTI) {

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
// Extensions:
// - Improved cycle detection algorithm for finding maximum length cycle
// - Use Number of phi nodes for new graph to estimate RF pressure
bool LoopPipeline::transformLoop(Loop *L, unsigned MII, CycleSet &cycles) {
  DEBUG(dbgs() << "LP: Software pipelining loop with MII=" << MII << '\n');
  BasicBlock *LoopBody = L->getBlocks()[0];

  PartialSchedule ASAPtimes;
  PartialSchedule ALAPtimes;

  // Compute ASAP times for all operations in the loop body
  unsigned LastOperationStart = scheduleASAP(LoopBody, ASAPtimes);

  // Compute ALAP times for all operations in the loop body in reverse
  scheduleALAP(LoopBody, LastOperationStart, ALAPtimes);

  // Debug print computed values
  DEBUG(dbgs() << "LP: Schedule freedom (ASAP, ALAP, Mobility, Cost, Depth, Height)\n  S L M D H C\n");
  for(auto I=LoopBody->begin(), E=LoopBody->end(); I != E; I++) {
    unsigned ASAP     = ASAPtimes[I],
             ALAP     = ALAPtimes[I],
             Mobility = ALAP - ASAP,
             Depth    = ASAP,
             Height   = LastOperationStart - ALAP,
             Cost     = getInstructionCost(I, TTI);

    DEBUG(
      dbgs() << "  " << ASAP << " " << ALAP << " " << Mobility << " " << Depth << " " << Height << " " << Cost;
      I->dump()
         );
  }

  // Construct node ordering
  //   - for each found cycle (long to short)
  //     - get length and insert into correct position in cycle set
  //     - prune nodes that are in the cycle from lower priority sets
  //     - find connected cycles through llvm node ordering and add connecting nodes to lower priority cycle
  //     - mark added nodes as done
  // - add remaining nodes as last set
  int i = cycles.size();
  for(auto cycle=cycles.rbegin(), E=cycles.rend(); cycle != E; cycle++) {
    DEBUG(dbgs() << "\nLP: Cycle " << --i << '\n');

    for(auto I : cycle->data()) {
      DEBUG(I->dump());
    }

    DEBUG(dbgs() << "LP: Length " << cycle->getWeight() << '\n');
  }

  // Schedule
  // Generate new inner loop
  // Construct prologue/epilogue

  // Done
  LoopsPipelined++;
  return true;
}

// Helper function - Compute ASAP schedule times
unsigned LoopPipeline::scheduleASAP(BasicBlock *B, PartialSchedule &schedule) {
  unsigned LastOperationStart = 0;
  for(auto I = B->begin(), E = B->end(); I != E; I++) {
    unsigned OperandsASAP = 0;

    // find I's in-loop operands
    if( !isa<PHINode>(I) ) {
      for(auto &O : I->operands()) {
        Instruction *Op = dyn_cast<Instruction>(O);
        if( !Op ) continue;

        if( Op->getParent() == B ) {
          // get maximum schedule time
          OperandsASAP = std::max(OperandsASAP, schedule[Op] + getInstructionCost(Op, TTI));
        }
      }
    }

    schedule[I] = OperandsASAP;

    // keep track of loop latency for ALAPtimes computation
    LastOperationStart = std::max(LastOperationStart, OperandsASAP);
  }

  return LastOperationStart;
}

// Helper function - Compute ALAP schedule times
void LoopPipeline::scheduleALAP(BasicBlock *B, unsigned LastOperationStart, PartialSchedule &schedule) {
  for(auto I = B->end(), E = B->begin(); I != E;) {
    Instruction *II = --I;
    unsigned DepsALAP = LastOperationStart;

    // find I's in-loop users
    for(auto D : II->users()) {
      Instruction *Dep = dyn_cast<Instruction>(D);
      if( !Dep ) continue;

      if( Dep->getParent() == B && schedule.find(Dep) != schedule.end() ) {
        // get minimum schedule time
        DepsALAP = std::min(DepsALAP, schedule[Dep]-getInstructionCost(I, TTI));
      }
    }
    schedule[II] = DepsALAP;
  }
}

