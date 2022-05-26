# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# End-to-end matrix multiplication tests.

load("//build_tools/bazel:iree_trace_runner_test.bzl", "iree_generated_trace_runner_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

py_binary(
    name = "generate_e2e_matmul_tests",
    srcs = ["generate_e2e_matmul_tests.py"],
)

[iree_generated_trace_runner_test(
    name = "e2e_matmul_direct_%s_small" % lhs_rhs_type,
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=small",
    ],
    target_backends_and_drivers = [
        ("dylib-llvm-aot", "dylib"),
        ("vmvx", "vmvx"),
    ],
    trace_runner = "//tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "i8",
    "f32",
]]

# Test asm
[iree_generated_trace_runner_test(
    name = "e2e_matmul_mmt4d_%s_small" % lhs_rhs_type,
    compiler_flags = [
        "--iree-flow-mmt4d-target-options=enable_generic_slow #pass_options_variant#",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=small",
    ],
    target_backends_and_drivers = [
        ("dylib-llvm-aot", "dylib"),
    ],
    target_cpu_features_variants = ["default"] +
                                   ([
                                       "aarch64:+dotprod",
                                       "aarch64:+i8mm",
                                   ] if lhs_rhs_type == "i8" else []),
    trace_runner = "//tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "i8",
    "f32",
]]

[iree_generated_trace_runner_test(
    name = "e2e_matmul_mmt4d_%s_large" % lhs_rhs_type,
    compiler_flags = [
        "--iree-flow-mmt4d-target-options=enable_generic_slow #pass_options_variant#",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=large",
    ],
    target_backends_and_drivers = [
        ("dylib-llvm-aot", "dylib"),
    ],
    target_cpu_features_variants = ["default"] +
                                   ([
                                       "aarch64:+dotprod",
                                       "aarch64:+i8mm",
                                   ] if lhs_rhs_type == "i8" else []),
    trace_runner = "//tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "i8",
    "f32",
]]

# Test intrinsics. No need to run vmvx again, since it isn't affected by this
# codegen flag. No need to run "large" sizes, since this only differs from other
# tests in ways that are orthogonal to problem sizes.
[iree_generated_trace_runner_test(
    name = "e2e_matmul_mmt4d_%s_intrinsics_%s" % (lhs_rhs_type, size),
    compiler_flags = [
        "--iree-codegen-mmt4d-use-intrinsics",
        "--iree-flow-mmt4d-target-options=enable_generic_slow #pass_options_variant#",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=%s" % size,
    ],
    target_backends_and_drivers = [
        ("dylib-llvm-aot", "dylib"),
    ],
    target_cpu_features_variants = ["default"] +
                                   ([
                                       "aarch64:+dotprod",
                                       "aarch64:+i8mm",
                                   ] if lhs_rhs_type == "i8" else []),
    trace_runner = "//tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "i8",
    "f32",
] for size in [
    "small",
]]

[iree_generated_trace_runner_test(
    name = "e2e_matmul_direct_f32_gpu_large_%s" % compilation_info,
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=f32",
        "--shapes=gpu_large",
        "--compilation_info=%s" % compilation_info,
    ],
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backends_and_drivers = [
        ("cuda", "cuda"),
    ],
    trace_runner = "//tools:iree-e2e-matmul-test",
) for compilation_info in [
    "LLVMGPUMatmulSimt",
]]

# Testing Ampere+ tensorcore path.
[iree_generated_trace_runner_test(
    name = "e2e_matmul_direct_f32_gpu_large_%s" % compilation_info,
    compiler_flags = [
        "--iree-hal-cuda-llvm-target-arch=sm_80",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=f32",
        "--shapes=gpu_large",
        "--compilation_info=%s" % compilation_info,
    ],
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backends_and_drivers = [
        ("cuda", "cuda"),
    ],
    trace_runner = "//tools:iree-e2e-matmul-test",
) for compilation_info in [
    "LLVMGPUMatmulTensorCore",
]]

[iree_generated_trace_runner_test(
    name = "e2e_matmul_direct_%s_large_split_k" % lhs_rhs_type,
    compiler_flags = [
        "--iree-flow-split-matmul-reduction=4",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=large",
    ],
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backends_and_drivers = [
        ("cuda", "cuda"),
        ("dylib-llvm-aot", "dylib"),
    ],
    trace_runner = "//tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "f32",
]]

[iree_generated_trace_runner_test(
    name = "e2e_matmul_direct_f32_gpu_large_%s" % vulkan_target,
    compiler_flags = [
        "--iree-vulkan-target-triple=%s" % vulkan_target,
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=f32",
        "--shapes=gpu_large",
        "--compilation_info=SPIRVVectorize",
    ],
    tags = [
        "requires-gpu-nvidia",
    ],
    target_backends_and_drivers = [
        ("vulkan-spirv", "vulkan"),
    ],
    trace_runner = "//tools:iree-e2e-matmul-test",
) for vulkan_target in [
    "valhall-unknown-android11",
    "ampere-unknown-linux",
]]