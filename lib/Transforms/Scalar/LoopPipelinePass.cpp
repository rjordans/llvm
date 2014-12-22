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
// and,
//
//     Swing Modulo Scheduling: A Lifetime-Sensitive Approach
//     Josep Llosa, Antonio Gonzalez, Eduard Ayguade, and Mateo Valero
//     Working Conference on Parallel Architectures and Compilation Techniques
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"

#include <set>

using namespace llvm;

#define DEBUG_TYPE "loop-pipeline"

STATISTIC(LoopsAnalyzed, "Number of loops analyzed for high-level software pipelining");
STATISTIC(LoopsPipelined, "Number of loops pipelined");

// FIXME:
// This is dependant on support in the back-end scheduler which is currently missing in the
// generic scheduler.  We should probably use a target hook for this information in stead of
// a user option.
static cl::opt<bool> AllowMultiIterationOperations("pipeline-allow-multi-iteration-ops",
    cl::init(false), cl::Hidden,
    cl::desc("Allow operations to cross over multiple itteration bounds"));

static cl::opt<bool> IgnoreResourceConstraints("pipeline-ignore-resources",
    cl::init(false), cl::Hidden,
    cl::desc("Ignore resource constraints during high-level software pipelining"));

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

    typedef DenseMap<Instruction *, unsigned> PartialSchedule;
    typedef std::set<InstructionTrace> CycleSet;

    bool processLoop(Loop *L);
    bool canPipelineLoop(Loop *L, CodeMetrics &CM);

    void getPhiCycles(Instruction *I, const PHINode *Phi, InstructionTrace trace, CycleSet &cycles);
    unsigned computeRecurrenceMII(Loop *L, CycleSet &cycles);
    unsigned computeResourceMII(CodeMetrics &CM);

    bool getConnectingNodes(Instruction *I,
        const BasicBlock *B,
        DenseMap<Instruction *, bool> &VisitedNodes,
        std::vector<Instruction *> &connectingNodes,
        bool direction);
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
    trace.add(I, TTI);
    cycles.insert(trace);
    return;
  }

  // found a cycle not passing through the currently considered phi-node
  // for example: a -> b -> c -> b, this can only happen if b is a phi-node
  if( isa<PHINode>(I) && trace.find(I) ) {
    return;
  }

  // Add current instruction and check cycles for each operand of the
  // instruction.  Don't add original Phi node until the cycle is completed
  // to preserve (reversed) ordering.
  if( I != Phi )
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
unsigned LoopPipeline::computeResourceMII(CodeMetrics &CM) {
  unsigned ResMII = 0;
  unsigned NumScalarInsts = CM.NumInsts - CM.NumVectorInsts;
  const unsigned ScalarFUCount = TTI->getScalarFunctionUnitCount();
  const unsigned VectorFUCount = TTI->getVectorFunctionUnitCount();

  if(IgnoreResourceConstraints) return 0;

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
// - Use Number of phi nodes for new graph to estimate RF pressure

// Helper function to find the nodes located at any path between the previous
// and current recurrence.
bool LoopPipeline::getConnectingNodes(Instruction *I,
    const BasicBlock *B,
    DenseMap<Instruction *, bool> &VisitedNodes,
    std::vector<Instruction *> &connectingNodes,
    bool direction)
{
  // Do not recurse over nodes outside of the current loop body
  if(I->getParent() != B) return false;

  // Recurse until a previously visited node is found
  if(VisitedNodes[I]) return true;

  // Recurse through operands/uses depending on direction
  bool found = false;
  if( direction ) {
    // avoid backedges
    if(isa<PHINode>(I)) return false;

    // search upwards
    for(auto &O : I->operands() ) {
      Instruction *II = dyn_cast<Instruction>(O);
      if(II)
        found |= getConnectingNodes(II, B, VisitedNodes, connectingNodes, direction);
    }
  } else {
    // search downwards
    for(auto U : I->users() ) {
      if(isa<PHINode>(U)) continue;
      Instruction *II = dyn_cast<Instruction>(U);
      if(II)
        found |= getConnectingNodes(II, B, VisitedNodes, connectingNodes, direction);
    }
  }

  // Add current node to the visited list and to the connecting nodes if a path was found
  if(found) {
    VisitedNodes[I] = true;
    connectingNodes.push_back(I);
  }

  return found;
}

bool LoopPipeline::transformLoop(Loop *L, unsigned MII, CycleSet &cycles) {
  DEBUG(dbgs() << "LP: Software pipelining loop with MII=" << MII << '\n');
  BasicBlock *LoopBody = L->getBlocks()[0];

  PartialSchedule ASAPtimes;
  PartialSchedule ALAPtimes;

  // Compute ASAP times for all operations in the loop body
  unsigned LastOperationStart = scheduleASAP(LoopBody, ASAPtimes);

  // Compute ALAP times for all operations in the loop body in reverse
  scheduleALAP(LoopBody, LastOperationStart, ALAPtimes);

#if 1
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
#endif

  // Construct priority sets based on the recurrences found earlier sorted by
  // their weight.  Add all nodes on paths between a previous recurrence and
  // the inserted one to avoid nodes which have both their ancestors and
  // successors scheduled.
  SmallVector<SmallVectorImpl<Instruction *>*, 4> PrioritySets;
  SmallVector<unsigned, 4> PrioritySetWeights;
  DenseMap<Instruction *, bool>  VisitedNodes;

  for(auto cycle=cycles.rbegin(), E=cycles.rend(); cycle != E; cycle++) {
    SmallVectorImpl<Instruction *>  *CurrentPrioritySet =
      new SmallVector<Instruction *, 8>();

    // Copy nodes from the current recurrence that haven't been visited before.
    for(auto I : cycle->data()) {
      if(VisitedNodes[I]) continue;

      CurrentPrioritySet->push_back(I);
      VisitedNodes[I] = true;
    }

    // Add nodes on paths between newly inserted set and the already visited nodes.
    if(!PrioritySets.empty()) {
      std::vector<Instruction *> connectingNodes;
      for(auto I : cycle->data()) {
        for(auto &O : I->operands() ) {
          Instruction *II = dyn_cast<Instruction>(O);
          if(II) getConnectingNodes(II, LoopBody, VisitedNodes, connectingNodes, true);
        }
        for(auto U : I->users() ) {
          Instruction *II = dyn_cast<Instruction>(U);
          if(II) getConnectingNodes(II, LoopBody, VisitedNodes, connectingNodes, false);
        }
      }

      for(auto I : connectingNodes)
        CurrentPrioritySet->push_back(I);
    }

    // Only add priority set to the worklist for the node ordering when
    // the set actually contains operations.
    if(!CurrentPrioritySet->empty()) {
      PrioritySetWeights.push_back(cycle->getWeight());
      PrioritySets.push_back(CurrentPrioritySet);
    } else {
      delete CurrentPrioritySet;
    }
  }

  // Copy the remaining operations into the final priority set.
  SmallVectorImpl<Instruction *> *AllNodesTrace =
    new SmallVector<Instruction *, 8>();

  for(auto I=LoopBody->begin(), E=LoopBody->end(); I != E; I++) {
    if( !VisitedNodes[I] )
      AllNodesTrace->push_back(I);
  }
  PrioritySetWeights.push_back(0);
  PrioritySets.push_back(AllNodesTrace);

  // Construct node ordering
  SmallVector<Instruction *, 16> OrderedNodes;
  DenseMap<Instruction *, bool>  AlreadyOrdered;
  SmallSet<Instruction *, 8>  PredecessorListO;
  SmallSet<Instruction *, 8>  SuccessorListO;

  // Compute node scheduling order
  for(auto CurrentPrioritySet: PrioritySets) {
    // Compute node ordering
    enum {BottomUp = 0, TopDown} order;
    SmallVector<Instruction *, 8> Ready;

    // Check if SuccessorListO or PredecessorListO is a subset of CurrentPrioritySet
    bool SuccLIsSubsetS = true, PredLIsSubsetS = true;
    for(auto I : SuccessorListO) {
      bool found = false;
      for(auto II : *CurrentPrioritySet)
        if(I==II) found = true;

      if(!found) {
        SuccLIsSubsetS = false;
        break;
      }
    }

    for(auto I : PredecessorListO) {
      bool found = false;
      for(auto II : *CurrentPrioritySet)
        if(I==II) found = true;

      if(!found) {
        PredLIsSubsetS = false;
        break;
      }
    }

    if(!PredecessorListO.empty() && PredLIsSubsetS ) {
      for(auto I : PredecessorListO)
        Ready.push_back(I);

      order = BottomUp;
    } else if(!SuccessorListO.empty() && SuccLIsSubsetS) {
      for(auto I : SuccessorListO)
        Ready.push_back(I);

      order = TopDown;
    } else {
      // Start with node that has the highest ASAP value in S
      auto v = CurrentPrioritySet->begin();
      for(auto I=v, E=CurrentPrioritySet->end(); I != E; I++)
        if(ASAPtimes[*v] < ASAPtimes[*I] )
          v = I;

      Ready.push_back(*v);
      order = BottomUp;
    }

    while(!Ready.empty()) {
      if(order == TopDown) {
        DEBUG(dbgs() << "LP: Swung (v)\n");
        // Top-down ordering
        while(!Ready.empty()) {
          auto v = Ready.begin();
          // Select operation with the highest height (smallest ALAP)
          // if more than one, choose node with lowest mobility
          for(auto I=v, E=Ready.end(); I != E; I++) {
            unsigned tV = ALAPtimes[*v],
                     tI = ALAPtimes[*I],
                     mV = tV - ASAPtimes[*v],
                     mI = tI - ASAPtimes[*I];

            if(tV > tI || (tV == tI && mV > mI) ) v = I;
          }
          OrderedNodes.push_back(*v);
          AlreadyOrdered[*v] = true;

          // Update R as R := R - {v} \union (Pred(v) \intersect S)
          for(auto I : *CurrentPrioritySet) {
            if(AlreadyOrdered[I])
              continue;

            for(auto U : (*v)->users()) {
              Instruction *II = dyn_cast<Instruction>(U);
              if(!II) continue;

              bool alreadyReady = false;
              for(auto III : Ready)
                if( III == I ) alreadyReady = true;

              if( !alreadyReady && I == II )
                Ready.push_back(I);
            }
          }

          // Update PredecessorListO and SuccessorListO
          SuccessorListO.erase(*v);
          PredecessorListO.erase(*v);
          for(auto I : OrderedNodes) {
            for(auto &O : I->operands()) {
              Instruction *II = dyn_cast<Instruction>(O);
              if(!II) continue;

              if( II->getParent() == LoopBody && !AlreadyOrdered[II])
                PredecessorListO.insert(II);
            }

            for(auto U : I->users()) {
              Instruction *II = dyn_cast<Instruction>(U);
              if(!II) continue;

              if( II->getParent() == LoopBody && !AlreadyOrdered[II])
                SuccessorListO.insert(II);
            }
          }

          Ready.erase(v);
        }

        // add intersection of pred_L(O) with S to R
        for(auto I : PredecessorListO)
          for(auto II : *CurrentPrioritySet)
            if(I == II && !AlreadyOrdered[I])
              Ready.push_back(I);

        order = BottomUp;
      } else {
        DEBUG(dbgs() << "LP: Swing (^)\n");
        // Bottom-up ordering
        while(!Ready.empty()) {
          auto v = Ready.begin();
          // Select operation with the highest depth (heighest ASAP)
          // if more than one, choose node with lowest mobility
          for(auto I=v, E=Ready.end(); I != E; I++) {
            unsigned tV = ASAPtimes[*v],
                     tI = ASAPtimes[*I],
                     mV = ALAPtimes[*v] - tV,
                     mI = ALAPtimes[*I] - tI;

            if(tV < tI || (tV == tI && mV > mI) ) v = I;
          }
          OrderedNodes.push_back(*v);
          AlreadyOrdered[*v] = true;

          // Update R as R := R - {v} \union (Suc(v) \intersect S)
          for(auto I : *CurrentPrioritySet) {
            if(AlreadyOrdered[I])
              continue;

            for(auto &O : (*v)->operands()) {
              Instruction *II = dyn_cast<Instruction>(O);
              if(!II) continue;

              bool alreadyReady = false;
              for(auto III : Ready)
                if( III == I ) alreadyReady = true;

              if( !alreadyReady && I == II )
                Ready.push_back(I);
            }
          }

          // Update PredecessorListO and SuccessorListO
          SuccessorListO.erase(*v);
          PredecessorListO.erase(*v);
          for(auto I : OrderedNodes) {
            for(auto &O : I->operands()) {
              Instruction *II = dyn_cast<Instruction>(O);
              if(!II) continue;

              if( II->getParent() == LoopBody && !AlreadyOrdered[II])
                PredecessorListO.insert(II);
            }

            for(auto U : I->users()) {
              Instruction *II = dyn_cast<Instruction>(U);
              if(!II) continue;

              if( II->getParent() == LoopBody && !AlreadyOrdered[II])
                SuccessorListO.insert(II);
            }
          }
          Ready.erase(v);
        }
        // add intersection of suc_L(O) with S to R
        for(auto I : SuccessorListO)
          for(auto II : *CurrentPrioritySet)
            if(I == II && !AlreadyOrdered[I])
              Ready.push_back(I);

        order = TopDown;
      }
    }

    delete CurrentPrioritySet;
  }

#if 0
  DEBUG(dbgs() << "LP: Node scheduling order\n");
  for(auto I : OrderedNodes) {
    DEBUG(I->dump());
  }
#endif

  assert(PredecessorListO.empty() && "Missed some nodes...");
  assert(SuccessorListO.empty() && "Missed some nodes...");

  // Schedule
  DenseMap<Instruction *, unsigned> ScheduledNodes;
  DenseMap<Instruction *, bool> AlreadyScheduled;

  // Resource information
  const unsigned ScalarFUCount = TTI->getScalarFunctionUnitCount();
  const unsigned VectorFUCount = TTI->getVectorFunctionUnitCount();

  // Retry with higher MII until either a schedule was found or the schedule
  // is no-longer a pipelined schedule
  unsigned II = MII;
  bool SchedulingDone = false;
  bool ScheduleHasFold;
  for( ; !SchedulingDone; II++) {
    DEBUG(dbgs() << "LP: Scheduling with II=" << II << "\n");

    // Resource allocation tables
    std::vector<unsigned> ScalarSlotsUsed;
    std::vector<unsigned> VectorSlotsUsed;

    // Initialize resource allocation
    ScalarSlotsUsed.resize(II, 0);
    VectorSlotsUsed.resize(II, 0);

    // Clear previous scheduling results
    ScheduledNodes.clear();
    AlreadyScheduled.clear();

    // Assume that scheduling will work this time (unset on failure)
    SchedulingDone = true;
    ScheduleHasFold = false;

    enum {Up = 0, Down} ScheduleDirection;

    // Schedule nodes using the previously obtained order
    for(auto I : OrderedNodes) {
      unsigned EarlyStart = ASAPtimes[I],
               LateStart = LastOperationStart + II - 1;

      // Default scheduling direction
      ScheduleDirection = Down;

      // if successors in scheduled nodes
      //  -> Update LateStart accordingly
      for(auto U : I->users() ) {
        Instruction *II = dyn_cast<Instruction>(U);
        // TODO test for back-edges?
        if(II && AlreadyScheduled[II]) {
          LateStart = std::min(LateStart, ScheduledNodes[II] - getInstructionCost(I, TTI));
          ScheduleDirection = Up;
        }
      }

      // if predecessors in scheduled nodes
      //  -> Update EarlyStart accordingly
      for(auto &O : I->operands() ) {
        Instruction *II = dyn_cast<Instruction>(O);
        // TODO test for back-edges?
        if(II && AlreadyScheduled[II]) {
          EarlyStart = std::max(EarlyStart, ScheduledNodes[II] + getInstructionCost(II, TTI));
          ScheduleDirection = Down;
        }
      }

      // Compute schedule range (EarlyStart has already been set)
      LateStart = std::min(LateStart, EarlyStart + II - 1);
#if 0
      DEBUG(dbgs() << "LP: Scheduling in range [" << EarlyStart << ", " << LateStart << "]:"; I->dump());
#endif

      // Unschedulable window found
      if(LateStart + getInstructionCost(I, TTI) < EarlyStart) {
        SchedulingDone = false;
        break;
      }

      // Classify operation type, distinguish between Vector, Scalar, and Free operations
      // Also count vector operations as scalar operations when no separate vector units are available
      bool isVectorOperation = VectorFUCount != 0 && (isa<ExtractElementInst>(I) || I->getType()->isVectorTy());
      bool isFreeOperation = getInstructionCost(I, TTI) == 0;

      // Find free slot for scheduling
      unsigned ScheduleAt;
      if(ScheduleDirection == Down) {
        ScheduleAt = EarlyStart;

        // Check resource availability
        if(!isFreeOperation && !IgnoreResourceConstraints) {
          for( ; ScheduleAt <= LateStart; ScheduleAt++) {
            bool ResourceAvailable;

            // Skip time slots for which the current operation would cross multiple iteration bounds
            if(!AllowMultiIterationOperations && ((ScheduleAt + getInstructionCost(I, TTI))/II - ScheduleAt/II) > 1) {
              DEBUG(dbgs() << "LP: Failed: Could not fold operation over more than one iteration\n");
              continue;
            }

            if(isVectorOperation)
              ResourceAvailable = !VectorFUCount ? true : VectorSlotsUsed[ScheduleAt % II] < VectorFUCount;
            else
              ResourceAvailable = !ScalarFUCount ? true : ScalarSlotsUsed[ScheduleAt % II] < ScalarFUCount;
            if(ResourceAvailable) break;
          }

          // Fail if no free slot is found (and unset SchedulingDone)
          if(ScheduleAt > LateStart) {
            SchedulingDone = false;
            break;
          }
        }
      } else {
        ScheduleAt = LateStart;

        // Check resource availability
        if(!isFreeOperation && !IgnoreResourceConstraints) {
          for( ; ScheduleAt <= EarlyStart; ScheduleAt--) {
            bool ResourceAvailable;

            // Skip time slots for which the current operation would cross multiple iteration bounds
            if(!AllowMultiIterationOperations && ((ScheduleAt + getInstructionCost(I, TTI))/II - ScheduleAt/II) > 1) {
              DEBUG(dbgs() << "LP: Failed: Could not fold operation over more than one iteration\n");
              continue;
            }

            if(isVectorOperation)
              ResourceAvailable = !VectorFUCount ? true : VectorSlotsUsed[ScheduleAt % II] < VectorFUCount;
            else
              ResourceAvailable = !ScalarFUCount ? true : ScalarSlotsUsed[ScheduleAt % II] < ScalarFUCount;
            if(ResourceAvailable) break;
          }

          // Fail if no free slot is found (and unset SchedulingDone)
          if(ScheduleAt < EarlyStart) {
            SchedulingDone = false;
            break;
          }
        }
      }
      DEBUG(dbgs() << "LP: Scheduling at " << ScheduleAt << ":"; I->dump());
      ScheduledNodes[I] = ScheduleAt;
      if(!isFreeOperation) {
        if(isVectorOperation)
          VectorSlotsUsed[ScheduleAt % II]++;
        else
          ScalarSlotsUsed[ScheduleAt % II]++;
      }

      // Check if schedule is still a modulo schedule that spans multiple iterations
      if(ScheduleAt >= II) {
        ScheduleHasFold = true;
      }
    }
  }
  // Undo last increment of for loop
  II--;

  // Check if schedule is still a modulo schedule that spans multiple iterations
  if( !ScheduleHasFold ) {
    DEBUG(dbgs() << "LP: Pipelined schedule has no benefit over non-pipelined version\n");
    return false;
  }

  DEBUG(dbgs() << "LP: Found schedule with II of " << II << '\n');

  // Generate new inner loop
  // Construct prologue/epilogue
  // TODO finish...

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

