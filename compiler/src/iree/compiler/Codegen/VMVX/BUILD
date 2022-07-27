# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_compiler_cc_library(
    name = "VMVX",
    srcs = [
        "LowerLinalgMicrokernels.cpp",
    ],
    deps = [
        "//compiler/src/iree/compiler/Codegen:PassHeaders",
        "//compiler/src/iree/compiler/Dialect/VMVX/IR",
        "//compiler/src/iree/compiler/Dialect/VMVX/IR:VMVXDialect",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)