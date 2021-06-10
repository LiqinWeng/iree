// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/Common/Passes.h"
#include "iree/compiler/Conversion/Common/Transforms.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

static const unsigned kNumMaxParallelDims = 3;

namespace mlir {
namespace iree_compiler {

namespace {
class SetNumWorkgroupsPass
    : public PassWrapper<SetNumWorkgroupsPass,
                         OperationPass<IREE::HAL::ExecutableTargetOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AffineDialect, IREE::HAL::HALDialect, linalg::LinalgDialect>();
  }

  SetNumWorkgroupsPass(ArrayRef<int64_t> ws = {})
      : workgroupSize(ws.begin(), ws.end()) {}
  SetNumWorkgroupsPass(const SetNumWorkgroupsPass &pass)
      : workgroupSize(pass.workgroupSize) {}

  void runOnOperation() override;

 private:
  SmallVector<int64_t> workgroupSize;
};
}  // namespace

void SetNumWorkgroupsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IREE::HAL::ExecutableTargetOp targetOp = getOperation();
  ModuleOp module = targetOp.getInnerModule();

  if (workgroupSize.empty()) {
    // If no workgroup size is specified, leave the workgroup size as is, just
    // set the number of workgroups to be 1, 1, 1 to have a single invocation.
    WorkgroupCountRegionBuilder regionBuilder =
        [](OpBuilder &b, Location loc,
           std::array<Value, 3> workload) -> std::array<Value, 3> {
      Value one = b.create<ConstantIndexOp>(loc, 1);
      return {one, one, one};
    };
    OpBuilder builder(context);
    for (auto funcOp : module.getOps<FuncOp>()) {
      if (failed(defineWorkgroupCountRegion(builder, funcOp, regionBuilder))) {
        return signalPassFailure();
      }
    }
    return;
  }

  auto entryPointFn = getSingleEntryPointFunction(module);
  if (failed(entryPointFn)) {
    return signalPassFailure();
  }
  auto funcOp = entryPointFn.getValue();

  if (failed(materializeStaticLaunchInformation(funcOp, workgroupSize))) {
    funcOp.emitError("failed to materialize constant workgroup size");
    return signalPassFailure();
  }

  // Apply post distribution canonicalization passes.
  OwningRewritePatternList canonicalization(context);
  AffineMinOp::getCanonicalizationPatterns(canonicalization, context);
  populateAffineMinSCFCanonicalizationPattern(canonicalization);
  IREE::Flow::populateFlowDispatchCanonicalizationPatterns(canonicalization,
                                                           context);
  (void)applyPatternsAndFoldGreedily(module, std::move(canonicalization));
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createSetNumWorkgroupsPass(ArrayRef<int64_t> workgroupSize) {
  return std::make_unique<SetNumWorkgroupsPass>(workgroupSize);
}

static PassRegistration<SetNumWorkgroupsPass> pass(
    "iree-set-num-workgroups",
    "Set the number of workgroups to use for every entry point function in the "
    "dispatch region");

}  // namespace iree_compiler
}  // namespace mlir
