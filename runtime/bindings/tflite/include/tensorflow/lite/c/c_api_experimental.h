/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_C_C_API_EXPERIMENTAL_H_
#define TENSORFLOW_LITE_C_C_API_EXPERIMENTAL_H_

#if defined(IREE_BINDINGS_TFLITE_INCLUDE_UNSUPPORTED_APIS)
#include "tensorflow/lite/builtin_ops.h"
#endif  // IREE_BINDINGS_TFLITE_INCLUDE_UNSUPPORTED_APIS

#include "c_api.h"
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Resets all variable tensors to zero.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterResetVariableTensors(
    TfLiteInterpreter* interpreter);

#if defined(IREE_BINDINGS_TFLITE_INCLUDE_UNSUPPORTED_APIS)

/// Adds an op registration for a builtin operator.
///
/// Op registrations are used to map ops referenced in the FlatBuffer model
/// to executable function pointers (`TfLiteRegistration`s).
///
/// NOTE: The interpreter will make a shallow copy of `registration` internally,
/// so the caller should ensure that its contents (function pointers, etc...)
/// remain valid for the duration of the interpreter's lifetime. A common
/// practice is making the provided `TfLiteRegistration` instance static.
///
/// Code that uses this function should NOT call
/// `TfLiteInterpreterOptionsSetOpResolver' on the same options object.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT void TfLiteInterpreterOptionsAddBuiltinOp(
    TfLiteInterpreterOptions* options, TfLiteBuiltinOperator op,
    const TfLiteRegistration* registration, int32_t min_version,
    int32_t max_version);

/// Adds an op registration for a custom operator.
///
/// Op registrations are used to map ops referenced in the FlatBuffer model
/// to executable function pointers (`TfLiteRegistration`s).
///
/// NOTE: The interpreter will make a shallow copy of `registration` internally,
/// so the caller should ensure that its contents (function pointers, etc...)
/// remain valid for the duration of any created interpreter's lifetime. A
/// common practice is making the provided `TfLiteRegistration` instance static.
///
/// Code that uses this function should NOT call
/// `TfLiteInterpreterOptionsSetOpResolver' on the same options object.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT void TfLiteInterpreterOptionsAddCustomOp(
    TfLiteInterpreterOptions* options, const char* name,
    const TfLiteRegistration* registration, int32_t min_version,
    int32_t max_version);

/// Registers callbacks for resolving builtin or custom operators.
///
/// The `TfLiteInterpreterOptionsSetOpResolver` function provides an alternative
/// method for registering builtin ops and/or custom ops, by providing operator
/// resolver callbacks.  Unlike using `TfLiteInterpreterOptionsAddBuiltinOp`
/// and/or `TfLiteInterpreterOptionsAddAddCustomOp`, these let you register all
/// the operators in a single call.
///
/// Code that uses this function should NOT call
/// `TfLiteInterpreterOptionsAddBuiltin' or
/// `TfLiteInterpreterOptionsAddCustomOp' on the same options object.
///
/// WARNING: This is an experimental API and subject to change.
void TfLiteInterpreterOptionsSetOpResolver(
    TfLiteInterpreterOptions* options,
    const TfLiteRegistration* (*find_builtin_op)(void* user_data,
                                                 TfLiteBuiltinOperator op,
                                                 int version),
    const TfLiteRegistration* (*find_custom_op)(void* user_data,
                                                const char* custom_op,
                                                int version),
    void* op_resolver_user_data);

#endif  // IREE_BINDINGS_TFLITE_INCLUDE_UNSUPPORTED_APIS

/// Returns a new interpreter using the provided model and options, or null on
/// failure, where the model uses only the operators explicitly added to the
/// options.  This is the same as `TFLiteInterpreterCreate` from `c_api.h`,
/// except that the only operators that are supported are the ones registered
/// in `options` via calls to `TfLiteInterpreterOptionsSetOpResolver`,
/// `TfLiteInterpreterOptionsAddBuiltinOp`, and/or
/// `TfLiteInterpreterOptionsAddCustomOp`.
///
/// * `model` must be a valid model instance. The caller retains ownership of
///   the object, and can destroy it immediately after creating the interpreter;
///   the interpreter will maintain its own reference to the underlying model
///   data.
/// * `options` should not be null. The caller retains ownership of the object,
///   and can safely destroy it immediately after creating the interpreter.
///
/// NOTE: The client *must* explicitly allocate tensors before attempting to
/// access input tensor data or invoke the interpreter.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteInterpreter*
TfLiteInterpreterCreateWithSelectedOps(const TfLiteModel* model,
                                       const TfLiteInterpreterOptions* options);

/// Enable or disable the NN API for the interpreter (true to enable).
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsSetUseNNAPI(
    TfLiteInterpreterOptions* options, bool enable);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_C_C_API_EXPERIMENTAL_H_
