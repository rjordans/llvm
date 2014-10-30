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
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
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
      AA = &getAnalysis<AliasAnalysis>();
      DA = &getAnalysis<DependenceAnalysis>();
      LI = &getAnalysis<LoopInfo>();
      SE = &getAnalysis<ScalarEvolution>();

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
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<DependenceAnalysis>();
      AU.addRequired<LoopInfo>();
      AU.addRequired<ScalarEvolution>();
    }

  private:
    AliasAnalysis *AA;
    DependenceAnalysis *DA;
    LoopInfo *LI;
    ScalarEvolution *SE;

    bool processLoop(Loop *L);
    bool canPipelineLoop(Loop *L);
    unsigned computeRecurrenceMII(Loop *L);
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
  assert(L->empty() && "Only process inner loops.");

  // Check if loop is a candidate
  if( !canPipelineLoop(L) ) {
    DEBUG(dbgs() << "LP: failed to pipeline loop.\n");
    return false;
  }

  // Estimate RecMII
  unsigned RecMII = computeRecurrenceMII(L);
  if( RecMII == 0 ) {
    DEBUG(dbgs() << "LP: failed to compute RecMII.\n");
    return false;
  }

  // FIXME: Estimate ResMII
  // Make this conditional to observe the effect of adding resource constraints
  // versus the approach taken in Ben-Asher & Meisler
  unsigned ResMII = 0;

  // Decide MII
  unsigned MII = std::max(RecMII, ResMII);
  DEBUG(dbgs() << "LP: software pipelining loop with MII=" << MII << ".\n");

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
bool LoopPipeline::canPipelineLoop(Loop *L) {
  unsigned NumBlocks = L->getNumBlocks();
  if( NumBlocks != 1 ) {
    DEBUG(dbgs() << "LP: Can only software-pipeline simple loops.\n");
    return false;
  }

  // TODO: Add more checks
  // - No control flow
  // - Doesn't call functions?  How about intrinsics...
  // - Right loop shape?
  // - Itteration count?
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

  // Find back-edges and their lengths
  SmallVector<const SCEV*, 8> MemPatterns;
  while( !MemOps.empty()) {
    Instruction *Inst = MemOps.pop_back_val();

    DEBUG(dbgs() << "LP: Analyzing memory op "; Inst->dump());

    // Find memory access recurrence
    const SCEV *scev;
    if( LoadInst *LI = dyn_cast<LoadInst>(Inst) ) {
      scev = SE->getSCEV(LI->getPointerOperand());
    } else if( StoreInst *SI = dyn_cast<StoreInst>(Inst) ) {
      scev = SE->getSCEV(SI->getPointerOperand());
    } else {
      DEBUG(dbgs() << "LP: Skipping loop containing unhandled memory operation type.\n");
      return 0;
    }

    if( !scev ) {
      DEBUG(dbgs() << "LP: Skipping loop containing un-analyzable memory access pattern.\n");
      return 0;
    }

    // Skip loop invariant load/store operation
    if( SE->isLoopInvariant(scev, L) )
      continue;

    MemPatterns.push_back(scev);
  }

  while( !MemPatterns.empty() ) {
    const SCEV *scev = MemPatterns.pop_back_val();
    scev->dump();
    // - Alias analysis on load/store combinations
    // - Also consider phi nodes?  Ignore induction variables
    // - Dependency analysis on MayAlias and DoAlias results, check NoAlias for loop dependence...
    // - Substract SCEV's to get back-edge length?
  }
//  LoopBody->dump();

  return 0;
}

