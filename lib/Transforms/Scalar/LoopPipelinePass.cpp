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
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

#define DEBUG_TYPE "loop-pipeline"

STATISTIC(LoopsAnalyzed, "Number of loops analyzed for high-level software pipelining");
STATISTIC(LoopsPipelined, "Number of loops pipelined");

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
    bool canPipelineLoop(Loop *L, CodeMetrics &CM);
    unsigned computeRecurrenceMII(Loop *L);
    unsigned computeResourceMII(CodeMetrics &CM);
  };
}

char LoopPipeline::ID = 0;
INITIALIZE_PASS(LoopPipeline, "loop-pipeline",
                "Software pipeline inner-loops", false, false)

FunctionPass *llvm::createLoopPipelinePass() {
  return new LoopPipeline();
}

//
// Process a single inner loop
//
bool LoopPipeline::processLoop(Loop *L) {
  assert(L->empty() && "Only process inner loops");

  // Check if loop is a candidate
  CodeMetrics CM;
  if( !canPipelineLoop(L, CM) ) {
    DEBUG(dbgs() << "LP: failed to pipeline loop\n");
    return false;
  }

  // Estimate RecMII
  unsigned RecMII = computeRecurrenceMII(L);
  if( RecMII == 0 ) {
    DEBUG(dbgs() << "LP: failed to compute RecMII\n");
    return false;
  }
  DEBUG(dbgs() << "LP: Found recurrence MII of " << RecMII << '\n');

  // FIXME: Estimate ResMII
  // Make this conditional to observe the effect of adding resource constraints
  // versus the approach taken in Ben-Asher & Meisler
  //
  // Uses NumVectorInsts and NumInsts from CodeMetrics for FU utilization
  // estimation
  //
  // Options
  // - Use Number of cut edges (phi nodes) for new graph to estimate RF pressure
  unsigned ResMII = computeResourceMII(CM);
  DEBUG(dbgs() << "LP: Found resource MII of " << ResMII << '\n');


  // Decide MII
  unsigned MII = std::max(RecMII, ResMII);
  DEBUG(dbgs() << "LP: software pipelining loop with MII=" << MII << '\n');

  // TODO: Perform actual software pipelining
  // - 'Schedule'
  // - Generate new inner loop

  // Done
  LoopsPipelined++;
  return true;
}

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
  BasicBlock *B = L->getBlocks()[0];

  CM.analyzeBasicBlock(B, *TTI);
  if( CM.NumCalls > 0 ) {
    DEBUG(dbgs() << "LP: Can not software-pipeline loops with function calls\n");
    return false;
  }

  if( CM.notDuplicatable ) {
    DEBUG(dbgs() << "LP: Loop contains operations marked 'noduplicate'\n");
    return false;
  }
  // TODO: Add more checks
  // - Itteration count?
  // - Constant stride
  // - Loop optimization hints
  // - ...


  return true;
}

//
// Find the minimal initiation interval for the loop recurrences
//
unsigned LoopPipeline::computeRecurrenceMII(Loop *L) {
  BasicBlock *LoopBody = L->getBlocks()[0];
  DEBUG(dbgs() << "LP: processing '" << LoopBody->getName() << "'\n");

  // Build list of memory operations as candidates for backedges
  SmallVector<Instruction*, 8> MemOps;
  for(auto I = LoopBody->begin(), E = LoopBody->end(); I != E; I++) {
    Instruction *Inst = cast<Instruction>(I);

    if( !Inst->mayReadFromMemory() && !Inst->mayWriteToMemory() )
      continue;

    // Add to list of memory operations
    MemOps.push_back(Inst);
  }

  // Find memory dependencies and their lengths
  SmallVector<const Dependence*, 8> LoopDeps;
  while( !MemOps.empty()) {
    Instruction *Inst = MemOps.pop_back_val();

    for(Instruction *II : MemOps) {
      // TODO figure out in which direction to compute these dependencies...
      // Load -> Store, Store -> Load, Load -> Load, ...
      // Focus on dependencies inside the loop (3rd argument = false)
      Dependence *D = DA->depends(II, Inst, false);

      // Skip if no dependency is found
      if( !D )
        continue;

      // Skip loops where memory dependencies could not be determined
      if( D->isConfused() ) {
        DEBUG(dbgs() << "LP: Loop depdence checker confused, giving up.\n");
        return 0;
      }

      // Check if we could compute a distance
      const SCEV *Distance = D->getDistance(D->getLevels());
      if( !Distance ) {
        DEBUG(dbgs() << "LP: Could not compute dependence distance, giving up.\n");
        return 0;
      }

      LoopDeps.push_back(D);

      // TODO
      // - Also consider phi nodes?
      // - Ignore induction variables?
    }
 }

  if( LoopDeps.empty() ) {
    // No loop carried dependencies found, set RecMII to 1
    DEBUG(dbgs() << "LP: No loop carried dependencies found\n");
    return 1;
  }

  while( !LoopDeps.empty() ) {
    const Dependence *D = LoopDeps.pop_back_val();

    if( isa<StoreInst>(D->getSrc()) || isa<StoreInst>(D->getDst()) ) {
      // Found a loop carried dependency, which not supported for now...
//      return 0;
      DEBUG(
        dbgs() << "LP: Loop carried dependency found: ";
        D->dump(dbgs())
        );
    }
//    DEBUG(
//      D->dump(dbgs());
//      dbgs() << " From\n";
//      D->getSrc()->dump();
//      dbgs() << " To\n";
//      D->getDst()->dump()
//    );
//
    // Get back-edge length
    //const SCEV *Distance = D->getDistance(D->getLevels());
  }
//  LoopBody->dump();

  return 0;
}

//
// Find the minimal initiation interval given the processor resources as
// provided by TTI.
//
// FIXME: This has a very limited view of the processor resources
//
// - Vectorized IR code is assumed to be executed on vector function units
// - Default values of NoTTI impose no constraints on resources
// - No constraint for the number of parallel memory accesses for loops with
//   high memory bandwitdh
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
