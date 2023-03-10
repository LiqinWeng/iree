# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Generates runner variants for the tf math tests.

import generate_runner

# These layers were selected by:
#   1. Getting all subclasses of `tf.keras.layers.Layer`
#   2. Removing deperacated layers based on the tf.keras docs
#   3. Removing irrelevant layers
#   4. Removing layers that don't fit in the testing framework (Wrappers, DenseFeatures, ...)
LAYERS = [
    "Activation",
    "ActivityRegularization",
    "Add",
    "AdditiveAttention",
    "AlphaDropout",
    "Attention",
    "Average",
    "AveragePooling1D",
    "AveragePooling2D",
    "AveragePooling3D",
    "BatchNormalization",
    "Concatenate",
    "Conv1D",
    "Conv1DTranspose",
    "Conv2D",
    "Conv2DTranspose",
    "Conv3D",
    "Conv3DTranspose",
    # "ConvLSTM2D",  # TODO(meadowlark): Debug flakiness.
    "Cropping1D",
    "Cropping2D",
    "Cropping3D",
    "Dense",
    "DepthwiseConv2D",
    "Dot",
    "Dropout",
    "ELU",
    "Embedding",
    "Flatten",
    "GRU",
    "GaussianDropout",
    "GaussianNoise",
    "GlobalAveragePooling1D",
    "GlobalAveragePooling2D",
    "GlobalAveragePooling3D",
    "GlobalMaxPool1D",
    "GlobalMaxPool2D",
    "GlobalMaxPool3D",
    "InputLayer",
    "LSTM",
    "Lambda",
    "LayerNormalization",
    "LeakyReLU",
    "LocallyConnected1D",
    "LocallyConnected2D",
    "Masking",
    "MaxPool1D",
    "MaxPool2D",
    "MaxPool3D",
    "Maximum",
    "Minimum",
    "MultiHeadAttention",
    "Multiply",
    "PReLU",
    "Permute",
    "ReLU",
    "RepeatVector",
    "Reshape",
    "SeparableConv1D",
    "SeparableConv2D",
    # "SimpleRNN",  # TODO(meadowlark): Debug flakiness.
    "Softmax",
    "SpatialDropout1D",
    "SpatialDropout2D",
    "SpatialDropout3D",
    "Subtract",
    "ThresholdedReLU",
    "UpSampling1D",
    "UpSampling2D",
    "UpSampling3D",
    "ZeroPadding1D",
    "ZeroPadding2D",
    "ZeroPadding3D",
]

# A list of all layers with non-default api tests can be generated by running:
#   bazel run integrations/tensorflow/e2e/keras/layers:layers_test_manual -- \
#     --list_layers_with_full_api_tests
LAYERS_WITH_FULL_API_TESTS = [
    "ActivityRegularization",
    "AdditiveAttention",
    "Attention",
    "AveragePooling1D",
    "AveragePooling2D",
    "AveragePooling3D",
    "BatchNormalization",
    "Concatenate",
    "Conv1D",
    "Conv1DTranspose",
    "Conv2D",
    "Conv2DTranspose",
    "Conv3D",
    "Conv3DTranspose",
    # "ConvLSTM2D",  # TODO(meadowlark): Debug flakiness.
    "Cropping1D",
    "Cropping2D",
    "Cropping3D",
    "DepthwiseConv2D",
    "GRU",
    "LSTM",
    "LocallyConnected1D",
    "LocallyConnected2D",
    "MaxPool1D",
    "MaxPool2D",
    "MaxPool3D",
    "SeparableConv1D",
    "SeparableConv2D",
    "SimpleRNN",
    # "SimpleRNN",  # TODO(meadowlark): Debug flakiness.
]

# Layers that mention a training kwarg in their doc.
LAYERS_WITH_TRAINING_BEHAVIOR = [
    "AdditiveAttention",
    "AlphaDropout",
    "Attention",
    "BatchNormalization",
    # "ConvLSTM2D",  # TODO(meadowlark): Debug flakiness.
    "Dropout",
    "GRU",
    "GaussianDropout",
    "GaussianNoise",
    "LSTM",
    "MultiHeadAttention",
    # "SimpleRNN",  # TODO(meadowlark): Debug flakiness.
    "SpatialDropout1D",
    "SpatialDropout2D",
    "SpatialDropout3D",
]

BACKENDS = [
    ("llvmcpu", "--target_backends=iree_llvmcpu"),
    ("vulkan", "--target_backends=iree_vulkan"),
]

# Non dynamic dim tests.
for variant, flags in BACKENDS:
  for layer in LAYERS:
    # Static.
    generate_runner.main([
        variant,
        (f"{flags} --dynamic_dims=false --training=false "
         f"--test_default_kwargs_only=true --layer={layer} --artifacts_dir=%t"),
        f"iree_tf_tests/layers/layers_test.py:{layer}"
    ])
    # Dynamic.
    generate_runner.main([
        variant,
        (f"{flags} --dynamic_dims=true --training=false "
         f"--test_default_kwargs_only=true --layer={layer} --artifacts_dir=%t"),
        f"iree_tf_tests/layers/layers_test.py:dynamic_dims_{layer}"
    ])

  # Test with test_default_kwargs_only=false
  for layer in LAYERS_WITH_FULL_API_TESTS:
    generate_runner.main([
        variant,
        (f"{flags} --dynamic_dims=false --training=false "
         f"--test_default_kwargs_only=false --layer={layer} --artifacts_dir=%t"
        ), f"iree_tf_tests/layers/layers_test.py:full_api_{layer}"
    ])

  # Test with training flags.
  for layer in LAYERS_WITH_TRAINING_BEHAVIOR:
    generate_runner.main([
        variant,
        (f"{flags} --dynamic_dims=false --training=true "
         f"--test_default_kwargs_only=true --layer={layer} --artifacts_dir=%t"),
        f"iree_tf_tests/layers/layers_test.py:training_{layer}"
    ])
