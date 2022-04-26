# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:cc_binary_benchmark.bzl", "cc_binary_benchmark")
load("//iree:build_defs.oss.bzl", "iree_runtime_cc_library", "iree_runtime_cc_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_runtime_cc_library(
    name = "buffer_transfer",
    srcs = ["buffer_transfer.c"],
    hdrs = ["buffer_transfer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/hal",
    ],
)

iree_runtime_cc_library(
    name = "deferred_command_buffer",
    srcs = ["deferred_command_buffer.c"],
    hdrs = ["deferred_command_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":resource_set",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal:arena",
        "//runtime/src/iree/hal",
    ],
)

iree_runtime_cc_library(
    name = "resource_set",
    srcs = ["resource_set.c"],
    hdrs = ["resource_set.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal:arena",
        "//runtime/src/iree/hal",
    ],
)

cc_binary_benchmark(
    name = "resource_set_benchmark",
    srcs = ["resource_set_benchmark.c"],
    deps = [
        ":resource_set",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:prng",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/testing:benchmark",
    ],
)

iree_runtime_cc_test(
    name = "resource_set_test",
    srcs = ["resource_set_test.cc"],
    deps = [
        ":resource_set",
        "//runtime/src/iree/base",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)