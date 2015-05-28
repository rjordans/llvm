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
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/CostModelAnalysis.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <set>

using namespace llvm;

#define DEBUG_TYPE "loop-pipeline"

STATISTIC(LoopsAnalyzed, "Number of loops analyzed for high-level software pipelining");
STATISTIC(LoopsPipelined, "Number of loops pipelined");

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

namespace {
  class LoopPipeline : public FunctionPass {
  public:
    static char ID; // Pass identification, replacement for typeid
    LoopPipeline() : FunctionPass(ID) {
      initializeLoopPipelinePass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override {
      CMA = &getAnalysis<CostModelAnalysis>();
      DA = &getAnalysis<DependenceAnalysis>();
      LI = &getAnalysis<LoopInfo>();
      SE = &getAnalysis<ScalarEvolution>();
      TTI = &getAnalysis<TargetTransformInfo>();
      TLI = &getAnalysis<TargetLibraryInfo>();

      DataLayoutPass *DLP = getAnalysisIfAvailable<DataLayoutPass>();
      DL = DLP ? &DLP->getDataLayout() : nullptr;


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
      AU.addRequired<CostModelAnalysis>();
      AU.addRequired<DependenceAnalysis>();
      AU.addRequired<LoopInfo>();
      AU.addRequired<ScalarEvolution>();
      AU.addRequired<TargetTransformInfo>();
      AU.addRequired<TargetLibraryInfo>();

      AU.addPreserved<CostModelAnalysis>();
      AU.addPreserved<TargetLibraryInfo>();
      AU.addPreserved<TargetTransformInfo>();
    }

  private:
    const CostModelAnalysis *CMA;
    const DataLayout *DL;
    DependenceAnalysis *DA;
    LoopInfo *LI;
    ScalarEvolution *SE;
    TargetTransformInfo *TTI;
    TargetLibraryInfo *TLI;

    class InstructionTrace {
    public:
      InstructionTrace(const CostModelAnalysis *CMA) : CMA(CMA), weight(0) {}
      InstructionTrace(const InstructionTrace &IT) : CMA(IT.CMA), trace(IT.trace), weight(IT.weight) {}

      // Add instruction to trace
      void add(Instruction *I, TargetTransformInfo *TTI) {
        trace.push_back(I);
        weight += CMA->getInstructionCost(I);
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
      const CostModelAnalysis *CMA;
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
INITIALIZE_PASS_BEGIN(LoopPipeline, "loop-pipeline",
                "Software pipeline inner-loops", false, false)
INITIALIZE_PASS_DEPENDENCY(CostModelAnalysis)
INITIALIZE_PASS_DEPENDENCY(DependenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_AG_DEPENDENCY(TargetTransformInfo)
INITIALIZE_PASS_END(LoopPipeline, "loop-pipeline",
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
  // TODO: Consider adding a dependency breaking pass

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

    InstructionTrace trace(CMA);
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

  /*
   * Construct priority sets based on the recurrences found earlier sorted by
   * their weight.  Add all nodes on paths between a previous recurrence and
   * the inserted one to avoid nodes which have both their ancestors and
   * successors scheduled.
   */
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

  PartialSchedule ASAPtimes;
  PartialSchedule ALAPtimes;

  // Compute ASAP times for all operations in the loop body
  unsigned LastOperationStart = scheduleASAP(LoopBody, ASAPtimes);

  // Compute ALAP times for all operations in the loop body in reverse
  scheduleALAP(LoopBody, LastOperationStart, ALAPtimes);

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
          auto SelectedOp = *v;
          OrderedNodes.push_back(SelectedOp);
          AlreadyOrdered[SelectedOp] = true;

          // Update R as R := R - {v} \union (Pred(v) \intersect S)
          Ready.erase(v);
          for(auto I : *CurrentPrioritySet) {
            if(AlreadyOrdered[I])
              continue;

            for(auto U : SelectedOp->users()) {
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
          SuccessorListO.erase(SelectedOp);
          PredecessorListO.erase(SelectedOp);
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
          auto SelectedOp = *v;
          OrderedNodes.push_back(SelectedOp);
          AlreadyOrdered[SelectedOp] = true;

          // Update R as R := R - {v} \union (Suc(v) \intersect S)
          Ready.erase(v);
          for(auto I : *CurrentPrioritySet) {
            if(AlreadyOrdered[I])
              continue;

            for(auto &O : SelectedOp->operands()) {
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
          SuccessorListO.erase(SelectedOp);
          PredecessorListO.erase(SelectedOp);
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

  /*
   * Schedule operations according to the prioritized ordering
   */
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
  unsigned LoopLatency = 0;
  for( ; !SchedulingDone; II++) {
    // Abort scheduling when the II grows too much
    if(LastOperationStart < II) break;

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

#if 1
    // Debug print computed values
    DEBUG(dbgs() << "LP: Schedule freedom (ASAP, ALAP, Mobility, Cost, Depth, Height)\n  S L M D H C\n");
    for(auto I=LoopBody->begin(), E=LoopBody->end(); I != E; I++) {
      unsigned ASAP     = ASAPtimes[I],
               ALAP     = ALAPtimes[I],
               Mobility = ALAP - ASAP,
               Depth    = ASAP,
               Height   = LastOperationStart - ALAP,
               Cost     = CMA->getInstructionCost(I);

      DEBUG(
        dbgs() << "  " << ASAP << " " << ALAP << " " << Mobility << " " << Depth << " " << Height << " " << Cost;
        I->dump()
           );
    }
#endif

    // Schedule nodes using the previously obtained order
    for(auto I : OrderedNodes) {
      unsigned EarlyStart = ASAPtimes[I],
               LateStart = LastOperationStart + II - 1;

      // Default scheduling direction
      ScheduleDirection = Down;

      // if successors in scheduled nodes
      //  -> Update LateStart accordingly
      for(auto U : I->users() ) {
        Instruction *dep = dyn_cast<Instruction>(U);

        if(dep && AlreadyScheduled[dep]) {
          unsigned userStart;
          if(isa<PHINode>(dep)) {
            userStart = std::max(ScheduledNodes[dep] - CMA->getInstructionCost(I), II) - II;
          } else {
            userStart = ScheduledNodes[dep] - CMA->getInstructionCost(I);
          }
          LateStart = std::min(LateStart, userStart);
          ScheduleDirection = Up;
        }
      }

      // if predecessors in scheduled nodes
      //  -> Update EarlyStart accordingly
      for(auto &O : I->operands() ) {
        Instruction *op = dyn_cast<Instruction>(O);

        if(op && AlreadyScheduled[op]) {
          unsigned operandFinish = ScheduledNodes[op]; + CMA->getInstructionCost(op);
          if(isa<PHINode>(I)) {
            operandFinish = std::min(operandFinish - II, operandFinish);
          }
          EarlyStart = std::max(EarlyStart, operandFinish);
          ScheduleDirection = Down;
        }
      }

      // Compute schedule range (EarlyStart has already been set)
      LateStart = std::min(LateStart, EarlyStart + II - 1);
#if 0
      DEBUG(dbgs() << "LP: Scheduling in range [" << EarlyStart << ", " << LateStart << "]:"; I->dump());
#endif

      // Unschedulable window found
      if(LateStart < EarlyStart) {
        SchedulingDone = false;
        break;
      }

      // Classify operation type, distinguish between Vector, Scalar, and Free operations
      // Also count vector operations as scalar operations when no separate vector units are available
      bool isVectorOperation = VectorFUCount != 0 && (isa<ExtractElementInst>(I) || I->getType()->isVectorTy());
      bool isFreeOperation = CMA->getInstructionCost(I) == 0;

      // Find free slot for scheduling
      unsigned ScheduleAt;
      if(ScheduleDirection == Down) {
        ScheduleAt = EarlyStart;

        // Check resource availability
        if(!isFreeOperation && !IgnoreResourceConstraints) {
          for( ; ScheduleAt <= LateStart; ScheduleAt++) {
            bool ResourceAvailable;

            // Skip time slots for which the current operation would cross multiple iteration bounds
            if(((ScheduleAt + CMA->getInstructionCost(I))/II - ScheduleAt/II) > 1) {
              DEBUG(dbgs() << "LP: Could not fold operation over more than one iteration\n");
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
          for( ; ScheduleAt >= EarlyStart; ScheduleAt--) {
            bool ResourceAvailable;

            // Skip time slots for which the current operation would cross multiple iteration bounds
            if(((ScheduleAt + CMA->getInstructionCost(I))/II - ScheduleAt/II) > 1) {
              DEBUG(dbgs() << "LP: Could not fold operation over more than one iteration\n");
              // Special case ScheduleAt == 0 to avoid an infinite loop due to integer
              // wraparound during scheduling
              if(ScheduleAt == 0) {
                SchedulingDone = false;
                break;
              } else continue;
            }

            if(isVectorOperation)
              ResourceAvailable = !VectorFUCount ? true : VectorSlotsUsed[ScheduleAt % II] < VectorFUCount;
            else
              ResourceAvailable = !ScalarFUCount ? true : ScalarSlotsUsed[ScheduleAt % II] < ScalarFUCount;
            if(ResourceAvailable) break;
          }

          // Fail if no free slot is found (and unset SchedulingDone)
          if(!SchedulingDone || ScheduleAt < EarlyStart) {
            SchedulingDone = false;
            break;
          }
        }
      }
#if 0
      DEBUG(dbgs() << "LP: Scheduling at " << ScheduleAt << ":"; I->dump());
#endif
      ScheduledNodes[I] = ScheduleAt;
      AlreadyScheduled[I] = true;
      if(!isFreeOperation) {
        if(isVectorOperation)
          VectorSlotsUsed[ScheduleAt % II]++;
        else
          ScalarSlotsUsed[ScheduleAt % II]++;
      }

      // Keep track of loop latency (last operation start)
      LoopLatency = std::max(LoopLatency, ScheduleAt);

      // Check if schedule is still a modulo schedule that spans multiple iterations
      if(ScheduleAt >= II && !isFreeOperation) {
        ScheduleHasFold = true;
      }
    }

    if(!SchedulingDone) {
      DEBUG(dbgs() << "LP: Failed to find a schedule with II=" << II << '\n');
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
  DEBUG(dbgs() << "LP: Schedule with latency "<< LoopLatency << "\n");
  for(auto I=LoopBody->begin(), E=LoopBody->end(); I != E; I++) {
    DEBUG(dbgs() << "  c" << ScheduledNodes[I]; I->dump());
  }

  // Compute number of overlapping kernels to determine block count for prologue/epilogue
  const unsigned NumberOfInterleavedIterations = LoopLatency / II + 1;
  DEBUG(dbgs() << "LP: Interleaving " << NumberOfInterleavedIterations << " loop iterations\n");

  /* Generate alternative inner loop and select pipelined alternative when there is sufficient data

               [  ] <------- back-edge taken count check
                /\
               /  \
              /    v
             /    [ ] <----- prologue
            /      |
            v      v
   old --> [ ] \ [   ] \ <-- pipelined loop
   loop    [ ]_| [   ]_|
            \      |
             \     v
              \   [ ] <----- epilogue
               \   /
                \ /
                 v
                [ ] <------- exit block
   ...
   */
  BasicBlock *OldPreheaderBlock = L->getLoopPreheader();
  BasicBlock *OldExitBlock = L->getExitBlock();
  Loop *ParentLoop = L->getParentLoop();

  if(!OldPreheaderBlock || !OldExitBlock) {
    DEBUG(dbgs() << "LP: Failed to create loop preheader or exit block for prologue and epilogue creation\n");
    return false;
  }

  LLVMContext &C = OldPreheaderBlock->getContext();

  // Create a new loop structure for pipelined loop so that we can keep the original loop as a backup...
  BasicBlock *Prologue = BasicBlock::Create(C, LoopBody->getName()+".lp.prologue", OldPreheaderBlock->getParent(), OldExitBlock);
  TerminatorInst *PrologueTerm = BranchInst::Create(OldExitBlock, Prologue);
  PrologueTerm->setDebugLoc(OldPreheaderBlock->getTerminator()->getDebugLoc());

  BasicBlock *PipelinedBody = Prologue->splitBasicBlock(Prologue->getTerminator(), LoopBody->getName()+".lp.kernel");
  BasicBlock *Epilogue = PipelinedBody->splitBasicBlock(PipelinedBody->getTerminator(), LoopBody->getName()+".lp.epilogue");

  Loop *Lp = new Loop();
  if(ParentLoop) {
    ParentLoop->addChildLoop(Lp);
    ParentLoop->addBasicBlockToLoop(Prologue, LI->getBase());
    ParentLoop->addBasicBlockToLoop(Epilogue, LI->getBase());
  } else {
    LI->addTopLevelLoop(Lp);
  }
  Lp->addBasicBlockToLoop(PipelinedBody, LI->getBase());

  // TODO: Split critical edges from OldPreheaderBlock when finished?
//  SplitCriticalEdge(LoopBody->getTerminator(), 0, this, false, false, true);
//  SplitCriticalEdge(LoopBody->getTerminator(), 1, this, false, false, true);

  // Construct worklists
  SmallVector<SmallVectorImpl<Instruction*>*,4> worklists;
  for(unsigned interval = 0; interval < NumberOfInterleavedIterations; interval++) {
    // Construct worklist for each interval
    SmallVectorImpl<Instruction*> *worklist = new SmallVector<Instruction *, 8>();

    for( auto I=LoopBody->begin(), E=LoopBody->end(); I != E; I++) {
      if( (ScheduledNodes[I] / II) == interval )
        worklist->push_back(I);
    }
    worklists.push_back(worklist);
  }

  /*
   * Construct the prologue
   */
  // Instruction translation maps
  SmallVector<DenseMap<Value *, Value *>*, 4> TranslationMaps;

  unsigned stage = 0;
  for(; stage < NumberOfInterleavedIterations-1; stage++) {
    // Create new translation map
    DenseMap<Value *, Value *> *TranslationMap = new DenseMap<Value *, Value *>();

    // Store translation map
    TranslationMaps.push_back(TranslationMap);

    // Insert operations
    for(unsigned block = stage+1; block;) {
      for(auto I : *worklists[--block]) {
        switch(I->getOpcode()) {
        case Instruction::Br:
          // No control-flow is needed during prologue
          break;
        case Instruction::PHI: {
          // Translate phi nodes either into the original inputs or into
          // results from the previous stage
          PHINode *Phi = dyn_cast<PHINode>(I);
          if(stage == 0) {
            // First iteration, values come from outside the loop
            (*TranslationMap)[Phi] =
                  Phi->getIncomingValueForBlock(OldPreheaderBlock);
          } else {
            // Value is served over back-edge of original loop
            Value *V = Phi->getIncomingValueForBlock(LoopBody);

            // Try to find the incoming value in the previous stages
            Value *OldV = (*TranslationMaps[stage-1])[V];

            // If OldV is a nullptr we didn't schedule the operation for the
            // prologue yet.  In this case, fall back to the other input value
            // for the original phi operation.
            if(OldV == nullptr) {
              OldV = Phi->getIncomingValueForBlock(OldPreheaderBlock);
            }

            // Find definition in previous stage and insert
            (*TranslationMap)[Phi] = OldV;
          }
          break; }
        default: {
          // Construct instruction from original loop body and register it in
          // current stage.
          Instruction *Inst = dyn_cast<Instruction>(I);
          // Clone instruction
          Instruction *NewNode = Inst->clone();
          if(Inst->hasName())
            NewNode->setName(Inst->getName() + ".lp" + Twine(stage));

          // Rewrite operands with previously cloned versions
          for(unsigned OpIdx = 0; OpIdx < NewNode->getNumOperands(); OpIdx++) {
            Instruction *Op = dyn_cast<Instruction>(NewNode->getOperand(OpIdx));

            if(Op && Op->getParent() == LoopBody) {
              // Replace operand with cloned version
              // Find stage distance between Operand and NewNode
              unsigned distance = (ScheduledNodes[Inst]/II - ScheduledNodes[Op]/II);

              // Get replacement operand
              Value *NewOp = (*TranslationMaps[stage-distance])[Op];
              assert(NewOp && "Undefined node encountered during rewriting");

              // Replace operand
              NewNode->setOperand(OpIdx, NewOp);
            }
          }

          // Add new node to CurrentStage
          NewNode->insertBefore(Prologue->getTerminator());

          // Try to constant fold new operation
          if(Constant *C = ConstantFoldInstruction(NewNode, DL, TLI)) {
            // Add the constant value to the list when successful
            (*TranslationMap)[I] = C;

            DEBUG(dbgs() << "LP: Constant folding"; I->dump(); dbgs() << "  to"; C->dump());

            // Discard the newly inserted operation
            NewNode->eraseFromParent();
          } else {
            // Add new node to translation map and keep it in the current stage
            (*TranslationMap)[I] = NewNode;
          }
          break; }
        }
      }
    }
  }

#if 1
  DEBUG(Prologue->dump());
#endif

  /*
   * Construct the kernel
   */
  // Create new translation map
  DenseMap<Value *, Value *> *TranslationMap = new DenseMap<Value *, Value *>();
  DenseMap<PHINode *, Value *> PhiNodeMap;
  // Store translation map
  TranslationMaps.push_back(TranslationMap);

  // Insert operations
  for(unsigned block = NumberOfInterleavedIterations; block; ) {
    for(auto I : *worklists[--block]) {
      DEBUG(I->dump());
      switch(I->getOpcode()) {
        case Instruction::Br:
          // The loop branch instruction is inserted after constructing the kernel
          break;
        case Instruction::PHI: {
          // Translate phi nodes into results from the previous stage
          PHINode *Phi = dyn_cast<PHINode>(I);

          // Value is served over back-edge of original loop
          Value *V = Phi->getIncomingValueForBlock(LoopBody);

          // Find definition in previous stage and create Phi node
          PHINode *NewPhi = PHINode::Create(V->getType(), 2, "", PipelinedBody->getFirstNonPHI());
          if(V->hasName()) NewPhi->setName(V->getName()+".lp.kernel.phi");

          // Try to find the incoming value in the previous stages
          Value *OldV = (*TranslationMaps[stage-1])[V];

          // If OldV is a nullptr we didn't schedule the operation for the
          // kernel yet.  In this case, fall back to the other input value for
          // the original phi operation.
          if(OldV == nullptr) {
            OldV = Phi->getIncomingValueForBlock(OldPreheaderBlock);
          }

          // Add edge from prologue
          NewPhi->addIncoming(OldV, Prologue);
          PhiNodeMap[NewPhi] = V;

          // Add the newly created Phi node to the translation map
          (*TranslationMap)[Phi] = NewPhi;
          break; }
        default: {
          // Construct instruction from original loop body and register it in
          // current stage.
          Instruction *Inst = dyn_cast<Instruction>(I);
          // Clone instruction
          Instruction *NewNode = Inst->clone();
          if(Inst->hasName())
            NewNode->setName(Inst->getName() + ".lp.kernel");

          // Rewrite operands with previously cloned versions
          for(unsigned OpIdx = 0; OpIdx < Inst->getNumOperands(); OpIdx++) {
            Instruction *Op = dyn_cast<Instruction>(Inst->getOperand(OpIdx));

            if(Op && Op->getParent() == LoopBody) {
              // Replace operand with cloned version
              // Find stage distance between Operand and NewNode
              unsigned distance = (ScheduledNodes[Inst]/II - ScheduledNodes[Op]/II);

              // Get replacement operand
              Value *NewOp;
              if(distance) {
                // TODO: Check for values crossing multiple stage boundaries
                // FIXME: This has problems with several of the test loops
                //
                // These need extra phi nodes in the kernel construction which
                // are currently not generated
#if 1
                DEBUG(dbgs() << "\n  Edge from:"; Op->dump(); dbgs() << "  To:"; Inst->dump());
#endif
//                assert(distance == 1 && "LP: FATAL unsupported dependency length");

                // Construct new phi node and copy the value name from the parent
                PHINode *NewPhi = PHINode::Create(
                    Op->getType(), 2, "", PipelinedBody->getFirstNonPHI());

                if(Op->hasName()) NewPhi->setName(Op->getName()+".lp.kernel.phi");

                // Add edge from prologue
                Value *OldV = (*TranslationMaps[stage-1])[Op];
                assert(OldV);
#if 1
                DEBUG(dbgs() << "  Becomes from:"; OldV->dump());
#endif
                NewPhi->addIncoming(OldV,Prologue);
                PhiNodeMap[NewPhi] = Op;

                NewOp = NewPhi;
              } else {
                NewOp = (*TranslationMap)[Op];
              }
              assert(NewOp && "Undefined node encountered during rewriting");

              // Replace operand
              NewNode->setOperand(OpIdx, NewOp);
            }
          }

          // Add new node to CurrentStage
          NewNode->insertBefore(PipelinedBody->getTerminator());

          // Try to constant fold new operation
          if(Constant *C = ConstantFoldInstruction(NewNode, DL, TLI)) {
            // Add the constant value to the list when successful
            (*TranslationMap)[I] = C;

            DEBUG(dbgs() << "LP: Constant folding"; I->dump(); dbgs() << "  to"; C->dump());

            // Discard the newly inserted operation
            NewNode->eraseFromParent();
          } else {
            // Add new node to translation map and keep it in the current stage
            (*TranslationMap)[I] = NewNode;
          }
          break; }
      }
    }
  }

  // Add kernel back-edges to newly create Phi nodes
  for(auto PNMapping : PhiNodeMap) {
    PHINode *PN = PNMapping.first;
    Value *V = (*TranslationMap)[PNMapping.second];
    PN->addIncoming(V, PipelinedBody);
  }

  // Insert new loop condition
  BranchInst *Term = cast<BranchInst>(LoopBody->getTerminator());
  BranchInst *OldTerm = cast<BranchInst>(PipelinedBody->getTerminator());
  Value *Cond = (*TranslationMap)[Term->getCondition()];

  // Make sure that the true and false destinations match the original loop
  if(Term->getSuccessor(0) == LoopBody) {
    BranchInst::Create(PipelinedBody, Epilogue, Cond, PipelinedBody);
  } else {
    BranchInst::Create(Epilogue, PipelinedBody, Cond, PipelinedBody);
  }
  OldTerm->eraseFromParent();

#if 1
  DEBUG(PipelinedBody->dump());
#endif

  /*
   * Construct the epilogue
   */
  for(stage++; stage < 2 * NumberOfInterleavedIterations - 1; stage++) {
    // Create new translation map
    TranslationMap = new DenseMap<Value *, Value *>();

    // Store translation map
    TranslationMaps.push_back(TranslationMap);

    // Insert operations
    for(unsigned block = stage - NumberOfInterleavedIterations + 1;
                 block < NumberOfInterleavedIterations;
                 block++) {
      for(auto I : *worklists[block]) {
        switch(I->getOpcode()) {
        case Instruction::Br:
          // No control-flow is needed during the epilogue
          break;
        case Instruction::PHI: {
          // Translate phi nodes either into the original inputs or into
          // results from the previous stage
          PHINode *Phi = dyn_cast<PHINode>(I);

          // Value is served over back-edge of original loop
          Value *V = Phi->getIncomingValueForBlock(LoopBody);

          // Find definition in previous stage and insert
          (*TranslationMap)[Phi] = (*TranslationMaps[stage-1])[V];
          break; }
        default: {
          // Construct instruction from original loop body and register it in
          // current stage.
          Instruction *Inst = dyn_cast<Instruction>(I);
          // Clone instruction
          Instruction *NewNode = Inst->clone();
          if(Inst->hasName())
            NewNode->setName(Inst->getName() + ".lp" + Twine(stage));

          // Rewrite operands with previously cloned versions
          for(unsigned OpIdx = 0; OpIdx < NewNode->getNumOperands(); OpIdx++) {
            Instruction *Op = dyn_cast<Instruction>(NewNode->getOperand(OpIdx));

            if(Op && Op->getParent() == LoopBody) {
              // Replace operand with cloned version
              // Find stage distance between Operand and NewNode
              unsigned distance = (ScheduledNodes[Inst]/II - ScheduledNodes[Op]/II);

              // Get replacement operand
              Value *NewOp = (*TranslationMaps[stage-distance])[Op];
              if(distance) {
                // If distance > 0 then the incomming edge is in the kernel and
                // we need a phi node to select between the kernel value and
                // the one from the prologue

                // These need extra phi nodes in the kernel construction which
                // are currently not generated
                assert(distance == 1 && "LP: FATAL unsupported dependency length");

                // Find phi node from kernel and clone it into the epilogue
                PHINode *OldPhi;
                for(auto U : NewOp->users()) {
                  if((OldPhi = dyn_cast<PHINode>(U)))
                    break;
                }

                // Construct new phi node and copy the value name from the parent
                PHINode *NewPhi = cast<PHINode>(OldPhi->clone());

                if(Op->hasName()) NewPhi->setName(Op->getName()+".lp.epilogue.phi");

                // Insert into epilogue
                Epilogue->getInstList().push_front(NewPhi);

                PhiNodeMap[NewPhi] = Op;
                NewOp = NewPhi;
              } else {
                NewOp = (*TranslationMaps[stage-distance])[Op];
              }
              assert(NewOp && "Undefined node encountered during rewriting");

              // Replace operand
              NewNode->setOperand(OpIdx, NewOp);
            }
          }

          // Add new node to CurrentStage
          NewNode->insertBefore(Epilogue->getTerminator());

          // Try to constant fold new operation
          if(Constant *C = ConstantFoldInstruction(NewNode, DL, TLI)) {
            // Add the constant value to the list when successful
            (*TranslationMap)[I] = C;

            DEBUG(dbgs() << "LP: Constant folding"; I->dump(); dbgs() << "  to"; C->dump());

            // Discard the newly inserted operation
            NewNode->eraseFromParent();
          } else {
            // Add new node to translation map and keep it in the current stage
            (*TranslationMap)[I] = NewNode;
          }
          break; }
        }
      }
    }
  }

#if 1
  DEBUG(Epilogue->dump());
#endif

  /*
   * Successfully constructed pipelined loop contents.
   *
   * Now connect the live-out variables of both loop versions using phi nodes
   * in the merge block and create the latch block that guards the pipelined
   * version of the loop.
   */
  // Connect live-out variables of both loop versions
  //
  // For each live out variable of the LoopBody, add a phi-node to the
  // OldExitBlock and replace all uses of the original value within the old
  // exit block with the new phi node.

  // Fist construct a the set of new phi nodes
  for(auto I = LoopBody->begin(), E = LoopBody->end(); I != E; I++) {
    PHINode *NewPhi = nullptr;

    // See if I has uses outside of the original loop and construct a new phi
    // node
    for(auto *U : I->users()) {
      Instruction *Dep = dyn_cast<Instruction>(U);
      if(Dep && Dep->getParent() != LoopBody) {
        // Add phi-node to OldExitBlock to replace this dependency with a
        // phi(I, (*TranslationMap)[I])

        NewPhi = PHINode::Create( I->getType(), 2, "",
            OldExitBlock->getFirstNonPHI());

        // Set name
        if(I->hasName()) NewPhi->setName(I->getName()+".lp.loopmerge.phi");

        // Set edges
        NewPhi->addIncoming(I, LoopBody);
        NewPhi->addIncoming((*TranslationMap)[I], Epilogue);

        // We don't need to check the other dependencies, we don't care if
        // there are more uses outside of the loop, we've already created the
        // new phi node...
        break;
      }
    }

    // If we constructed a new phi node, replace all uses of I outside of the
    // original loop body with the newly created phi node
    // (except for the new phi node itself)
    if(NewPhi) {
      auto UI = I->use_begin(), UE = I->use_end();
      for(; UI != UE;) {
        Use &U = *UI;
        ++UI;
        auto *Usr = dyn_cast<Instruction>(U.getUser());
        if (Usr && (Usr == NewPhi || Usr->getParent() != LoopBody))
          continue;
        U.set(NewPhi);
      }
    }
  }

  // Get branch condition from prologue to construct selector block
  BranchInst *LoopBr = cast<BranchInst>(LoopBody->getTerminator());

  // There are N-1 blocks in the prologue and we start counting from 0 so use
  // -2 as index to reach the branch condition for the last iteration within
  // the prologue and -3 to find the one-before-last branch condition in the
  // prologue.
  //
  // The -3 version should tell us if we actually have sufficient data to
  // finish the prologue so that's the one we'll use when there are more than
  // 2 interleaved iterations.  Otherwise we always branch into the pipelined
  // loop as it should behave exactly like the original and this allows to
  // remove the original loop completely.
  Value *PrologueBranchValue;
  if(NumberOfInterleavedIterations == 2) {
    Type *ConditionType = LoopBr->getCondition()->getType();

    if(LoopBr->getSuccessor(0) == LoopBody) {
      PrologueBranchValue = ConstantInt::getFalse(ConditionType);
    } else {
      PrologueBranchValue = ConstantInt::getTrue(ConditionType);
    }
  } else {
    PrologueBranchValue =
      (*TranslationMaps[NumberOfInterleavedIterations-3])[LoopBr->getCondition()];
  }

  // Check if the branch condition actually is an Instruction or if it's a Constant
  if(Instruction *PrologueBranchCond = dyn_cast<Instruction>(PrologueBranchValue)) {
    assert(PrologueBranchCond && "Could not find branch condition for prologue");

    // Move the branch condition with all of its dependencies into the selector
    // block.
    //
    // Also remove the unused versions from the prologue and replce the
    // occurences in the prologue with the operation inserted into the selector
    // block.
    //
    // FIXME: This currently does not check if the branch condition computation
    // contains operations which may trap.  For example, loops which have
    // multiple iterations in the prologue and for which an operation in the
    // second iteration causes a trap if executed while there is only sufficient
    // data for a single operation.
    SmallVector<Instruction *,4> MoveList;
    std::set<Instruction *> MoveSet;
    std::set<Instruction *> WorkSet;
    WorkSet.insert(PrologueBranchCond);

    while(!WorkSet.empty()) {
      // Get first element from WorkSet
      Instruction *Inst = *WorkSet.begin();
      WorkSet.erase(WorkSet.begin());

      DEBUG(dbgs() << "LP: Considering to move"; Inst->dump());

      // Add to MoveSet for our administration
      MoveSet.insert(Inst);

      // Add the operation for moving
      MoveList.push_back(Inst);

      // Consider the operands of the added operation for addition
      for(auto &op : Inst->operands()) {
        Instruction *I = dyn_cast<Instruction>(op);

        // Only add instructions that are within the current block and have not
        // been considered yet
        if(I && I->getParent() == Prologue
             && MoveSet.find(I) == MoveSet.end()) {
          DEBUG(dbgs() << "LP: Adding "; I->dump());
          WorkSet.insert(I);
        }
      }
    }

    // Reverse the move list to get the right insertion order
    std::reverse(MoveList.begin(), MoveList.end());

    // Do the move!
    for(auto I : MoveList) {
      I->removeFromParent();
      I->insertBefore(OldPreheaderBlock->getTerminator());
    }

    // Replace branch in selector block with a conditional branch
    TerminatorInst *OldBranch = OldPreheaderBlock->getTerminator();
    if(LoopBr->getSuccessor(0) == LoopBody) {
      BranchInst::Create(Prologue, LoopBody, PrologueBranchCond, OldBranch);
    } else {
      BranchInst::Create(LoopBody, Prologue, PrologueBranchCond, OldBranch);
    }
    OldBranch->eraseFromParent();

  } else {
    // Got a constant? value as loop condition which means we can do some
    // cleanup.  Either we never enter the kernel loop and can drop the loop
    // from existence.  Or we never need the backup original loop and can
    // remove that one
    ConstantInt *BranchCondition = dyn_cast<ConstantInt>(PrologueBranchValue);

    assert(BranchCondition
           && "Expected a constant branch condition in the prologue");

    DEBUG(dbgs() << "LP: Found constant loop entry condition ";
      BranchCondition->dump());

    // New loop gets executed when:
    // - Either BranchCondition == true & LoopBr->getSuccessor(0) == LoopBody
    // - Or BranchCondition == false & LoopBr->getSuccessor(1) == LoopBody
    if(BranchCondition->equalsInt(0) ^ (LoopBr->getSuccessor(0) == LoopBody)) {
      // TODO: Maybe we could figure this out earlier and save some trouble ;)
      //
      // The new loop won't get executed, just keep the old one and undo our
      // changes
      DEBUG(dbgs() << "LP: Discarding pipelined loop as unused\n");

      // Keep old branch (do nothing), the new loop is unreachable and will be
      // removed during cleanup
    } else {
      // The old loop won't get executed, remove it
      DEBUG(dbgs() << "LP: Discarding original loop as unused\n");

      // Insert new branch
      TerminatorInst *OldBranch = OldPreheaderBlock->getTerminator();
      BranchInst::Create(Prologue, OldBranch);
      OldBranch->eraseFromParent();

      // The old loop won't get executed and is unreachable now.  Cleanup is
      // performed after this pass finishes and will remove the old loop.
    }
  }

  /*
   * Clean-up allocated structures and generated code
   */
  for(auto worklist : worklists) {
    delete worklist;
  }
  for(auto translationmap : TranslationMaps) {
    delete translationmap;
  }

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
          OperandsASAP = std::max(OperandsASAP, schedule[Op] + CMA->getInstructionCost(Op));
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
        DepsALAP = std::min(DepsALAP, schedule[Dep]-CMA->getInstructionCost(I));
      }
    }
    schedule[II] = DepsALAP;
  }
}
