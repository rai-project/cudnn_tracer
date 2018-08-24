#include "cudnn.h"
#include "json.hpp"
#include "recorder.h"
#include "utils.hpp"
#include <dlfcn.h>

using json = nlohmann::json;

// 1
cudnnStatus_t CUDNNWINAPI cudnnQueryRuntimeError(cudnnHandle_t handle,
                                                 cudnnStatus_t *rstatus,
                                                 cudnnErrQueryMode_t mode,
                                                 cudnnRuntimeTag_t *tag) {
  using fun_t = decltype(cudnnQueryRuntimeError);
  static const std::string funName{"cudnnQueryRuntimeError"};
  static fun_t *orig_cudnnQueryRuntimeError =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnQueryRuntimeError");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnQueryRuntimeError(handle, rstatus, mode, tag);
  }
  const auto tic = now();
  const auto res = orig_cudnnQueryRuntimeError(handle, rstatus, mode, tag);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 2
cudnnStatus_t CUDNNWINAPI cudnnGetProperty(libraryPropertyType type, int *value) {
  using fun_t = decltype(cudnnGetProperty);
  static const std::string funName{"cudnnGetProperty"};
  static fun_t *orig_cudnnGetProperty = (fun_t *) dlsym(RTLD_NEXT, "cudnnGetProperty");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetProperty(type, value);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetProperty(type, value);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"type", type}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 3
cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t *handle) {
  using fun_t = decltype(cudnnCreate);
  static const std::string funName{"cudnnCreate"};
  static fun_t *orig_cudnnCreate = (fun_t *) dlsym(RTLD_NEXT, "cudnnCreate");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreate(handle);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreate(handle);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 4
cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t handle) {
  using fun_t = decltype(cudnnDestroy);
  static const std::string funName{"cudnnDestroy"};
  static fun_t *orig_cudnnDestroy = (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroy");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroy(handle);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroy(handle);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 5
cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
  using fun_t = decltype(cudnnSetStream);
  static const std::string funName{"cudnnSetStream"};
  static fun_t *orig_cudnnSetStream = (fun_t *) dlsym(RTLD_NEXT, "cudnnSetStream");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetStream(handle, streamId);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetStream(handle, streamId);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 6
cudnnStatus_t CUDNNWINAPI cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId) {
  using fun_t = decltype(cudnnGetStream);
  static const std::string funName{"cudnnGetStream"};
  static fun_t *orig_cudnnGetStream = (fun_t *) dlsym(RTLD_NEXT, "cudnnGetStream");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetStream(handle, streamId);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetStream(handle, streamId);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 7
cudnnStatus_t CUDNNWINAPI
    cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
  using fun_t = decltype(cudnnCreateTensorDescriptor);
  static const std::string funName{"cudnnCreateTensorDescriptor"};
  static fun_t *orig_cudnnCreateTensorDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCreateTensorDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreateTensorDescriptor(tensorDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreateTensorDescriptor(tensorDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 8
cudnnStatus_t CUDNNWINAPI
    cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                               cudnnTensorFormat_t format,
                               cudnnDataType_t dataType, /* image data type */
                               int n, /* number of inputs (batch size) */
                               int c, /* number of input feature maps */
                               int h, /* height of input section */
                               int w) {
  using fun_t = decltype(cudnnSetTensor4dDescriptor);
  static const std::string funName{"cudnnSetTensor4dDescriptor"};
  static fun_t *orig_cudnnSetTensor4dDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetTensor4dDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnSetTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"c", c}, {"w", w}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 9
cudnnStatus_t CUDNNWINAPI
    cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                                 cudnnDataType_t dataType, /* image data type */
                                 int n, /* number of inputs (batch size) */
                                 int c, /* number of input feature maps */
                                 int h, /* height of input section */
                                 int w, /* width of input section */
                                 int nStride,
                                 int cStride,
                                 int hStride,
                                 int wStride) {
  using fun_t = decltype(cudnnSetTensor4dDescriptorEx);
  static const std::string funName{"cudnnSetTensor4dDescriptorEx"};
  static fun_t *orig_cudnnSetTensor4dDescriptorEx =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetTensor4dDescriptorEx");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetTensor4dDescriptorEx(
        tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetTensor4dDescriptorEx(
      tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"c", c},
                                               {"nStride", nStride},
                                               {"cStride", cStride},
                                               {"hStride", hStride},
                                               {"wStride", wStride}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 10
cudnnStatus_t CUDNNWINAPI
    cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                               cudnnDataType_t *dataType, /* image data type */
                               int *n, /* number of inputs (batch size) */
                               int *c, /* number of input feature maps  */
                               int *h, /* height of input section */
                               int *w, /* width of input section */
                               int *nStride,
                               int *cStride,
                               int *hStride,
                               int *wStride) {
  using fun_t = decltype(cudnnGetTensor4dDescriptor);
  static const std::string funName{"cudnnGetTensor4dDescriptor"};
  static fun_t *orig_cudnnGetTensor4dDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetTensor4dDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetTensor4dDescriptor(
        tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetTensor4dDescriptor(
      tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 11
cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                                     cudnnDataType_t dataType,
                                                     int nbDims,
                                                     const int dimA[],
                                                     const int strideA[]) {
  using fun_t = decltype(cudnnSetTensorNdDescriptor);
  static const std::string funName{"cudnnSetTensorNdDescriptor"};
  static fun_t *orig_cudnnSetTensorNdDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetTensorNdDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"nbDims", nbDims}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 12
cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                                                       cudnnTensorFormat_t format,
                                                       cudnnDataType_t dataType,
                                                       int nbDims,
                                                       const int dimA[]) {
  using fun_t = decltype(cudnnSetTensorNdDescriptorEx);
  static const std::string funName{"cudnnSetTensorNdDescriptorEx"};
  static fun_t *orig_cudnnSetTensorNdDescriptorEx =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetTensorNdDescriptorEx");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetTensorNdDescriptorEx(tensorDesc, format, dataType, nbDims, dimA);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnSetTensorNdDescriptorEx(tensorDesc, format, dataType, nbDims, dimA);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"nbDims", nbDims}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 13
cudnnStatus_t CUDNNWINAPI
    cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                               int nbDimsRequested,
                               cudnnDataType_t *dataType,
                               int *nbDims,
                               int dimA[],
                               int strideA[]) {
  using fun_t = decltype(cudnnGetTensorNdDescriptor);
  static const std::string funName{"cudnnGetTensorNdDescriptor"};
  static fun_t *orig_cudnnGetTensorNdDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetTensorNdDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetTensorNdDescriptor(
        tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetTensorNdDescriptor(
      tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"nbDimsRequested", nbDimsRequested}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 14
cudnnStatus_t CUDNNWINAPI
    cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc, size_t *size) {
  using fun_t = decltype(cudnnGetTensorSizeInBytes);
  static const std::string funName{"cudnnGetTensorSizeInBytes"};
  static fun_t *orig_cudnnGetTensorSizeInBytes =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetTensorSizeInBytes");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetTensorSizeInBytes(tensorDesc, size);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetTensorSizeInBytes(tensorDesc, size);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 15
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
  using fun_t = decltype(cudnnDestroyTensorDescriptor);
  static const std::string funName{"cudnnDestroyTensorDescriptor"};
  static fun_t *orig_cudnnDestroyTensorDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroyTensorDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroyTensorDescriptor(tensorDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroyTensorDescriptor(tensorDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 16
cudnnStatus_t CUDNNWINAPI cudnnTransformTensor(cudnnHandle_t handle,
                                               const void *alpha,
                                               const cudnnTensorDescriptor_t xDesc,
                                               const void *x,
                                               const void *beta,
                                               const cudnnTensorDescriptor_t yDesc,
                                               void *y) {
  using fun_t = decltype(cudnnTransformTensor);
  static const std::string funName{"cudnnTransformTensor"};
  static fun_t *orig_cudnnTransformTensor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnTransformTensor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y);
  }
  const auto tic = now();
  const auto res = orig_cudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 17
cudnnStatus_t CUDNNWINAPI cudnnAddTensor(cudnnHandle_t handle,
                                         const void *alpha,
                                         const cudnnTensorDescriptor_t aDesc,
                                         const void *A,
                                         const void *beta,
                                         const cudnnTensorDescriptor_t cDesc,
                                         void *C) {
  using fun_t = decltype(cudnnAddTensor);
  static const std::string funName{"cudnnAddTensor"};
  static fun_t *orig_cudnnAddTensor = (fun_t *) dlsym(RTLD_NEXT, "cudnnAddTensor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);
  }
  const auto tic = now();
  const auto res = orig_cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 18
cudnnStatus_t CUDNNWINAPI
    cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t *opTensorDesc) {
  using fun_t = decltype(cudnnCreateOpTensorDescriptor);
  static const std::string funName{"cudnnCreateOpTensorDescriptor"};
  static fun_t *orig_cudnnCreateOpTensorDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCreateOpTensorDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreateOpTensorDescriptor(opTensorDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreateOpTensorDescriptor(opTensorDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 19
cudnnStatus_t CUDNNWINAPI
    cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc,
                               cudnnOpTensorOp_t opTensorOp,
                               cudnnDataType_t opTensorCompType,
                               cudnnNanPropagation_t opTensorNanOpt) {
  using fun_t = decltype(cudnnSetOpTensorDescriptor);
  static const std::string funName{"cudnnSetOpTensorDescriptor"};
  static fun_t *orig_cudnnSetOpTensorDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetOpTensorDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetOpTensorDescriptor(
        opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetOpTensorDescriptor(
      opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 20
cudnnStatus_t CUDNNWINAPI
    cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t opTensorDesc,
                               cudnnOpTensorOp_t *opTensorOp,
                               cudnnDataType_t *opTensorCompType,
                               cudnnNanPropagation_t *opTensorNanOpt) {
  using fun_t = decltype(cudnnGetOpTensorDescriptor);
  static const std::string funName{"cudnnGetOpTensorDescriptor"};
  static fun_t *orig_cudnnGetOpTensorDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetOpTensorDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetOpTensorDescriptor(
        opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetOpTensorDescriptor(
      opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 21
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc) {
  using fun_t = decltype(cudnnDestroyOpTensorDescriptor);
  static const std::string funName{"cudnnDestroyOpTensorDescriptor"};
  static fun_t *orig_cudnnDestroyOpTensorDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroyOpTensorDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroyOpTensorDescriptor(opTensorDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroyOpTensorDescriptor(opTensorDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 22
cudnnStatus_t CUDNNWINAPI cudnnOpTensor(cudnnHandle_t handle,
                                        const cudnnOpTensorDescriptor_t opTensorDesc,
                                        const void *alpha1,
                                        const cudnnTensorDescriptor_t aDesc,
                                        const void *A,
                                        const void *alpha2,
                                        const cudnnTensorDescriptor_t bDesc,
                                        const void *B,
                                        const void *beta,
                                        const cudnnTensorDescriptor_t cDesc,
                                        void *C) {
  using fun_t = decltype(cudnnOpTensor);
  static const std::string funName{"cudnnOpTensor"};
  static fun_t *orig_cudnnOpTensor = (fun_t *) dlsym(RTLD_NEXT, "cudnnOpTensor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnOpTensor(
        handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
  }
  const auto tic = now();
  const auto res = orig_cudnnOpTensor(
      handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 23
cudnnStatus_t CUDNNWINAPI
    cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc) {
  using fun_t = decltype(cudnnCreateReduceTensorDescriptor);
  static const std::string funName{"cudnnCreateReduceTensorDescriptor"};
  static fun_t *orig_cudnnCreateReduceTensorDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCreateReduceTensorDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreateReduceTensorDescriptor(reduceTensorDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreateReduceTensorDescriptor(reduceTensorDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 24
cudnnStatus_t CUDNNWINAPI
    cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                   cudnnReduceTensorOp_t reduceTensorOp,
                                   cudnnDataType_t reduceTensorCompType,
                                   cudnnNanPropagation_t reduceTensorNanOpt,
                                   cudnnReduceTensorIndices_t reduceTensorIndices,
                                   cudnnIndicesType_t reduceTensorIndicesType) {
  using fun_t = decltype(cudnnSetReduceTensorDescriptor);
  static const std::string funName{"cudnnSetReduceTensorDescriptor"};
  static fun_t *orig_cudnnSetReduceTensorDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetReduceTensorDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetReduceTensorDescriptor(reduceTensorDesc,
                                               reduceTensorOp,
                                               reduceTensorCompType,
                                               reduceTensorNanOpt,
                                               reduceTensorIndices,
                                               reduceTensorIndicesType);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetReduceTensorDescriptor(reduceTensorDesc,
                                                       reduceTensorOp,
                                                       reduceTensorCompType,
                                                       reduceTensorNanOpt,
                                                       reduceTensorIndices,
                                                       reduceTensorIndicesType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 25
cudnnStatus_t CUDNNWINAPI
    cudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                   cudnnReduceTensorOp_t *reduceTensorOp,
                                   cudnnDataType_t *reduceTensorCompType,
                                   cudnnNanPropagation_t *reduceTensorNanOpt,
                                   cudnnReduceTensorIndices_t *reduceTensorIndices,
                                   cudnnIndicesType_t *reduceTensorIndicesType) {
  using fun_t = decltype(cudnnGetReduceTensorDescriptor);
  static const std::string funName{"cudnnGetReduceTensorDescriptor"};
  static fun_t *orig_cudnnGetReduceTensorDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetReduceTensorDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetReduceTensorDescriptor(reduceTensorDesc,
                                               reduceTensorOp,
                                               reduceTensorCompType,
                                               reduceTensorNanOpt,
                                               reduceTensorIndices,
                                               reduceTensorIndicesType);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetReduceTensorDescriptor(reduceTensorDesc,
                                                       reduceTensorOp,
                                                       reduceTensorCompType,
                                                       reduceTensorNanOpt,
                                                       reduceTensorIndices,
                                                       reduceTensorIndicesType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 26
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc) {
  using fun_t = decltype(cudnnDestroyReduceTensorDescriptor);
  static const std::string funName{"cudnnDestroyReduceTensorDescriptor"};
  static fun_t *orig_cudnnDestroyReduceTensorDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroyReduceTensorDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 27
cudnnStatus_t CUDNNWINAPI
    cudnnGetReductionIndicesSize(cudnnHandle_t handle,
                                 const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                 const cudnnTensorDescriptor_t aDesc,
                                 const cudnnTensorDescriptor_t cDesc,
                                 size_t *sizeInBytes) {
  using fun_t = decltype(cudnnGetReductionIndicesSize);
  static const std::string funName{"cudnnGetReductionIndicesSize"};
  static fun_t *orig_cudnnGetReductionIndicesSize =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetReductionIndicesSize");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetReductionIndicesSize(
        handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetReductionIndicesSize(
      handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 28
cudnnStatus_t CUDNNWINAPI
    cudnnGetReductionWorkspaceSize(cudnnHandle_t handle,
                                   const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                   const cudnnTensorDescriptor_t aDesc,
                                   const cudnnTensorDescriptor_t cDesc,
                                   size_t *sizeInBytes) {
  using fun_t = decltype(cudnnGetReductionWorkspaceSize);
  static const std::string funName{"cudnnGetReductionWorkspaceSize"};
  static fun_t *orig_cudnnGetReductionWorkspaceSize =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetReductionWorkspaceSize");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetReductionWorkspaceSize(
        handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetReductionWorkspaceSize(
      handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 29
cudnnStatus_t CUDNNWINAPI
    cudnnReduceTensor(cudnnHandle_t handle,
                      const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                      void *indices,
                      size_t indicesSizeInBytes,
                      void *workspace,
                      size_t workspaceSizeInBytes,
                      const void *alpha,
                      const cudnnTensorDescriptor_t aDesc,
                      const void *A,
                      const void *beta,
                      const cudnnTensorDescriptor_t cDesc,
                      void *C) {
  using fun_t = decltype(cudnnReduceTensor);
  static const std::string funName{"cudnnReduceTensor"};
  static fun_t *orig_cudnnReduceTensor = (fun_t *) dlsym(RTLD_NEXT, "cudnnReduceTensor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnReduceTensor(handle,
                                  reduceTensorDesc,
                                  indices,
                                  indicesSizeInBytes,
                                  workspace,
                                  workspaceSizeInBytes,
                                  alpha,
                                  aDesc,
                                  A,
                                  beta,
                                  cDesc,
                                  C);
  }
  const auto tic = now();
  const auto res = orig_cudnnReduceTensor(handle,
                                          reduceTensorDesc,
                                          indices,
                                          indicesSizeInBytes,
                                          workspace,
                                          workspaceSizeInBytes,
                                          alpha,
                                          aDesc,
                                          A,
                                          beta,
                                          cDesc,
                                          C);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"indicesSizeInBytes", indicesSizeInBytes},
                                  {"workspaceSizeInBytes", workspaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 30
cudnnStatus_t CUDNNWINAPI cudnnSetTensor(cudnnHandle_t handle,
                                         const cudnnTensorDescriptor_t yDesc,
                                         void *y,
                                         const void *valuePtr) {
  using fun_t = decltype(cudnnSetTensor);
  static const std::string funName{"cudnnSetTensor"};
  static fun_t *orig_cudnnSetTensor = (fun_t *) dlsym(RTLD_NEXT, "cudnnSetTensor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetTensor(handle, yDesc, y, valuePtr);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetTensor(handle, yDesc, y, valuePtr);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 31
cudnnStatus_t CUDNNWINAPI cudnnScaleTensor(cudnnHandle_t handle,
                                           const cudnnTensorDescriptor_t yDesc,
                                           void *y,
                                           const void *alpha) {
  using fun_t = decltype(cudnnScaleTensor);
  static const std::string funName{"cudnnScaleTensor"};
  static fun_t *orig_cudnnScaleTensor = (fun_t *) dlsym(RTLD_NEXT, "cudnnScaleTensor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnScaleTensor(handle, yDesc, y, alpha);
  }
  const auto tic = now();
  const auto res = orig_cudnnScaleTensor(handle, yDesc, y, alpha);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 32
cudnnStatus_t CUDNNWINAPI
    cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
  using fun_t = decltype(cudnnCreateFilterDescriptor);
  static const std::string funName{"cudnnCreateFilterDescriptor"};
  static fun_t *orig_cudnnCreateFilterDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCreateFilterDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreateFilterDescriptor(filterDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreateFilterDescriptor(filterDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 33
cudnnStatus_t CUDNNWINAPI
    cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                               cudnnDataType_t dataType, /* image data type */
                               cudnnTensorFormat_t format,
                               int k, /* number of output feature maps */
                               int c, /* number of input feature maps */
                               int h, /* height of each input filter */
                               int w) {
  using fun_t = decltype(cudnnSetFilter4dDescriptor);
  static const std::string funName{"cudnnSetFilter4dDescriptor"};
  static fun_t *orig_cudnnSetFilter4dDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetFilter4dDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"k", k}, {"c", c}, {"w", w}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 34
cudnnStatus_t CUDNNWINAPI
    cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t filterDesc,
                               cudnnDataType_t *dataType, /* image data type */
                               cudnnTensorFormat_t *format,
                               int *k, /* number of output feature maps */
                               int *c, /* number of input feature maps */
                               int *h, /* height of each input filter */
                               int *w) {
  using fun_t = decltype(cudnnGetFilter4dDescriptor);
  static const std::string funName{"cudnnGetFilter4dDescriptor"};
  static fun_t *orig_cudnnGetFilter4dDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetFilter4dDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnGetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 35
cudnnStatus_t CUDNNWINAPI
    cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,
                               cudnnDataType_t dataType, /* image data type */
                               cudnnTensorFormat_t format,
                               int nbDims,
                               const int filterDimA[]) {
  using fun_t = decltype(cudnnSetFilterNdDescriptor);
  static const std::string funName{"cudnnSetFilterNdDescriptor"};
  static fun_t *orig_cudnnSetFilterNdDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetFilterNdDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetFilterNdDescriptor(
        filterDesc, dataType, format, nbDims, filterDimA);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnSetFilterNdDescriptor(filterDesc, dataType, format, nbDims, filterDimA);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"nbDims", nbDims}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 36
cudnnStatus_t CUDNNWINAPI
    cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t filterDesc,
                               int nbDimsRequested,
                               cudnnDataType_t *dataType, /* image data type */
                               cudnnTensorFormat_t *format,
                               int *nbDims,
                               int filterDimA[]) {
  using fun_t = decltype(cudnnGetFilterNdDescriptor);
  static const std::string funName{"cudnnGetFilterNdDescriptor"};
  static fun_t *orig_cudnnGetFilterNdDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetFilterNdDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetFilterNdDescriptor(
        filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetFilterNdDescriptor(
      filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"nbDimsRequested", nbDimsRequested}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 37
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
  using fun_t = decltype(cudnnDestroyFilterDescriptor);
  static const std::string funName{"cudnnDestroyFilterDescriptor"};
  static fun_t *orig_cudnnDestroyFilterDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroyFilterDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroyFilterDescriptor(filterDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroyFilterDescriptor(filterDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 38
cudnnStatus_t CUDNNWINAPI
    cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
  using fun_t = decltype(cudnnCreateConvolutionDescriptor);
  static const std::string funName{"cudnnCreateConvolutionDescriptor"};
  static fun_t *orig_cudnnCreateConvolutionDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCreateConvolutionDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreateConvolutionDescriptor(convDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreateConvolutionDescriptor(convDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 39
cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionMathType(
    cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType) {
  using fun_t = decltype(cudnnSetConvolutionMathType);
  static const std::string funName{"cudnnSetConvolutionMathType"};
  static fun_t *orig_cudnnSetConvolutionMathType =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetConvolutionMathType");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetConvolutionMathType(convDesc, mathType);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetConvolutionMathType(convDesc, mathType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 40
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionMathType(
    cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType) {
  using fun_t = decltype(cudnnGetConvolutionMathType);
  static const std::string funName{"cudnnGetConvolutionMathType"};
  static fun_t *orig_cudnnGetConvolutionMathType =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionMathType");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionMathType(convDesc, mathType);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionMathType(convDesc, mathType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 41
cudnnStatus_t CUDNNWINAPI
    cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount) {
  using fun_t = decltype(cudnnSetConvolutionGroupCount);
  static const std::string funName{"cudnnSetConvolutionGroupCount"};
  static fun_t *orig_cudnnSetConvolutionGroupCount =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetConvolutionGroupCount");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetConvolutionGroupCount(convDesc, groupCount);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetConvolutionGroupCount(convDesc, groupCount);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"groupCount", groupCount}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 42
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionGroupCount(
    cudnnConvolutionDescriptor_t convDesc, int *groupCount) {
  using fun_t = decltype(cudnnGetConvolutionGroupCount);
  static const std::string funName{"cudnnGetConvolutionGroupCount"};
  static fun_t *orig_cudnnGetConvolutionGroupCount =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionGroupCount");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionGroupCount(convDesc, groupCount);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionGroupCount(convDesc, groupCount);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 43
cudnnStatus_t CUDNNWINAPI cudnnSetConvolution2dDescriptor(
    cudnnConvolutionDescriptor_t convDesc,
    int pad_h,      /* zero-padding height */
    int pad_w,      /* zero-padding width */
    int u,          /* vertical filter stride */
    int v,          /* horizontal filter stride */
    int dilation_h, /* filter dilation in the vertical dimension */
    int dilation_w, /* filter dilation in the horizontal dimension */
    cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType) {
  using fun_t = decltype(cudnnSetConvolution2dDescriptor);
  static const std::string funName{"cudnnSetConvolution2dDescriptor"};
  static fun_t *orig_cudnnSetConvolution2dDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetConvolution2dDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetConvolution2dDescriptor(
        convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetConvolution2dDescriptor(
      convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"pad_h", pad_h},
                                               {"pad_w", pad_w},
                                               {"u", u},
                                               {"dilation_h", dilation_h},
                                               {"dilation_w", dilation_w}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 44
cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dDescriptor(
    const cudnnConvolutionDescriptor_t convDesc,
    int *pad_h,      /* zero-padding height */
    int *pad_w,      /* zero-padding width */
    int *u,          /* vertical filter stride */
    int *v,          /* horizontal filter stride */
    int *dilation_h, /* filter dilation in the vertical dimension */
    int *dilation_w, /* filter dilation in the horizontal dimension */
    cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType) {
  using fun_t = decltype(cudnnGetConvolution2dDescriptor);
  static const std::string funName{"cudnnGetConvolution2dDescriptor"};
  static fun_t *orig_cudnnGetConvolution2dDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolution2dDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolution2dDescriptor(
        convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolution2dDescriptor(
      convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 45
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                          const cudnnTensorDescriptor_t inputTensorDesc,
                                          const cudnnFilterDescriptor_t filterDesc,
                                          int *n,
                                          int *c,
                                          int *h,
                                          int *w) {
  using fun_t = decltype(cudnnGetConvolution2dForwardOutputDim);
  static const std::string funName{"cudnnGetConvolution2dForwardOutputDim"};
  static fun_t *orig_cudnnGetConvolution2dForwardOutputDim =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolution2dForwardOutputDim");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolution2dForwardOutputDim(
        convDesc, inputTensorDesc, filterDesc, n, c, h, w);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolution2dForwardOutputDim(
      convDesc, inputTensorDesc, filterDesc, n, c, h, w);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 46
cudnnStatus_t CUDNNWINAPI
    cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                    int arrayLength, /* nbDims-2 size */
                                    const int padA[],
                                    const int filterStrideA[],
                                    const int dilationA[],
                                    cudnnConvolutionMode_t mode,
                                    cudnnDataType_t computeType) {
  using fun_t = decltype(cudnnSetConvolutionNdDescriptor);
  static const std::string funName{"cudnnSetConvolutionNdDescriptor"};
  static fun_t *orig_cudnnSetConvolutionNdDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetConvolutionNdDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetConvolutionNdDescriptor(
        convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetConvolutionNdDescriptor(
      convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"arrayLength", arrayLength}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 47
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t convDesc,
                                    int arrayLengthRequested,
                                    int *arrayLength,
                                    int padA[],
                                    int strideA[],
                                    int dilationA[],
                                    cudnnConvolutionMode_t *mode,
                                    cudnnDataType_t *computeType) {
  using fun_t = decltype(cudnnGetConvolutionNdDescriptor);
  static const std::string funName{"cudnnGetConvolutionNdDescriptor"};
  static fun_t *orig_cudnnGetConvolutionNdDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionNdDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionNdDescriptor(convDesc,
                                                arrayLengthRequested,
                                                arrayLength,
                                                padA,
                                                strideA,
                                                dilationA,
                                                mode,
                                                computeType);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionNdDescriptor(convDesc,
                                                        arrayLengthRequested,
                                                        arrayLength,
                                                        padA,
                                                        strideA,
                                                        dilationA,
                                                        mode,
                                                        computeType);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"arrayLengthRequested", arrayLengthRequested}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 48
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                          const cudnnTensorDescriptor_t inputTensorDesc,
                                          const cudnnFilterDescriptor_t filterDesc,
                                          int nbDims,
                                          int tensorOuputDimA[]) {
  using fun_t = decltype(cudnnGetConvolutionNdForwardOutputDim);
  static const std::string funName{"cudnnGetConvolutionNdForwardOutputDim"};
  static fun_t *orig_cudnnGetConvolutionNdForwardOutputDim =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionNdForwardOutputDim");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionNdForwardOutputDim(
        convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionNdForwardOutputDim(
      convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"nbDims", nbDims}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 49
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
  using fun_t = decltype(cudnnDestroyConvolutionDescriptor);
  static const std::string funName{"cudnnDestroyConvolutionDescriptor"};
  static fun_t *orig_cudnnDestroyConvolutionDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroyConvolutionDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroyConvolutionDescriptor(convDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroyConvolutionDescriptor(convDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 50
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
  using fun_t = decltype(cudnnGetConvolutionForwardAlgorithmMaxCount);
  static const std::string funName{"cudnnGetConvolutionForwardAlgorithmMaxCount"};
  static fun_t *orig_cudnnGetConvolutionForwardAlgorithmMaxCount =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionForwardAlgorithmMaxCount");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionForwardAlgorithmMaxCount(handle, count);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionForwardAlgorithmMaxCount(handle, count);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 51
cudnnStatus_t CUDNNWINAPI
    cudnnFindConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                         const cudnnTensorDescriptor_t xDesc,
                                         const cudnnFilterDescriptor_t wDesc,
                                         const cudnnConvolutionDescriptor_t convDesc,
                                         const cudnnTensorDescriptor_t yDesc,
                                         const int requestedAlgoCount,
                                         int *returnedAlgoCount,
                                         cudnnConvolutionFwdAlgoPerf_t *perfResults) {
  using fun_t = decltype(cudnnFindConvolutionForwardAlgorithm);
  static const std::string funName{"cudnnFindConvolutionForwardAlgorithm"};
  static fun_t *orig_cudnnFindConvolutionForwardAlgorithm =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnFindConvolutionForwardAlgorithm");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnFindConvolutionForwardAlgorithm(handle,
                                                     xDesc,
                                                     wDesc,
                                                     convDesc,
                                                     yDesc,
                                                     requestedAlgoCount,
                                                     returnedAlgoCount,
                                                     perfResults);
  }
  const auto tic = now();
  const auto res = orig_cudnnFindConvolutionForwardAlgorithm(handle,
                                                             xDesc,
                                                             wDesc,
                                                             convDesc,
                                                             yDesc,
                                                             requestedAlgoCount,
                                                             returnedAlgoCount,
                                                             perfResults);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"requestedAlgoCount", requestedAlgoCount}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 52
cudnnStatus_t CUDNNWINAPI
    cudnnFindConvolutionForwardAlgorithmEx(cudnnHandle_t handle,
                                           const cudnnTensorDescriptor_t xDesc,
                                           const void *x,
                                           const cudnnFilterDescriptor_t wDesc,
                                           const void *w,
                                           const cudnnConvolutionDescriptor_t convDesc,
                                           const cudnnTensorDescriptor_t yDesc,
                                           void *y,
                                           const int requestedAlgoCount,
                                           int *returnedAlgoCount,
                                           cudnnConvolutionFwdAlgoPerf_t *perfResults,
                                           void *workSpace,
                                           size_t workSpaceSizeInBytes) {
  using fun_t = decltype(cudnnFindConvolutionForwardAlgorithmEx);
  static const std::string funName{"cudnnFindConvolutionForwardAlgorithmEx"};
  static fun_t *orig_cudnnFindConvolutionForwardAlgorithmEx =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnFindConvolutionForwardAlgorithmEx");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnFindConvolutionForwardAlgorithmEx(handle,
                                                       xDesc,
                                                       x,
                                                       wDesc,
                                                       w,
                                                       convDesc,
                                                       yDesc,
                                                       y,
                                                       requestedAlgoCount,
                                                       returnedAlgoCount,
                                                       perfResults,
                                                       workSpace,
                                                       workSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnFindConvolutionForwardAlgorithmEx(handle,
                                                               xDesc,
                                                               x,
                                                               wDesc,
                                                               w,
                                                               convDesc,
                                                               yDesc,
                                                               y,
                                                               requestedAlgoCount,
                                                               returnedAlgoCount,
                                                               perfResults,
                                                               workSpace,
                                                               workSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"requestedAlgoCount", requestedAlgoCount},
                                  {"workSpaceSizeInBytes", workSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 53
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                        const cudnnTensorDescriptor_t xDesc,
                                        const cudnnFilterDescriptor_t wDesc,
                                        const cudnnConvolutionDescriptor_t convDesc,
                                        const cudnnTensorDescriptor_t yDesc,
                                        cudnnConvolutionFwdPreference_t preference,
                                        size_t memoryLimitInBytes,
                                        cudnnConvolutionFwdAlgo_t *algo) {
  using fun_t = decltype(cudnnGetConvolutionForwardAlgorithm);
  static const std::string funName{"cudnnGetConvolutionForwardAlgorithm"};
  static fun_t *orig_cudnnGetConvolutionForwardAlgorithm =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionForwardAlgorithm");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionForwardAlgorithm(
        handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, algo);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionForwardAlgorithm(
      handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, algo);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"memoryLimitInBytes", memoryLimitInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 54
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t handle,
                                           const cudnnTensorDescriptor_t srcDesc,
                                           const cudnnFilterDescriptor_t filterDesc,
                                           const cudnnConvolutionDescriptor_t convDesc,
                                           const cudnnTensorDescriptor_t destDesc,
                                           const int requestedAlgoCount,
                                           int *returnedAlgoCount,
                                           cudnnConvolutionFwdAlgoPerf_t *perfResults) {
  using fun_t = decltype(cudnnGetConvolutionForwardAlgorithm_v7);
  static const std::string funName{"cudnnGetConvolutionForwardAlgorithm_v7"};
  static fun_t *orig_cudnnGetConvolutionForwardAlgorithm_v7 =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionForwardAlgorithm_v7");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionForwardAlgorithm_v7(handle,
                                                       srcDesc,
                                                       filterDesc,
                                                       convDesc,
                                                       destDesc,
                                                       requestedAlgoCount,
                                                       returnedAlgoCount,
                                                       perfResults);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionForwardAlgorithm_v7(handle,
                                                               srcDesc,
                                                               filterDesc,
                                                               convDesc,
                                                               destDesc,
                                                               requestedAlgoCount,
                                                               returnedAlgoCount,
                                                               perfResults);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"requestedAlgoCount", requestedAlgoCount}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 55
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,
                                            const cudnnTensorDescriptor_t xDesc,
                                            const cudnnFilterDescriptor_t wDesc,
                                            const cudnnConvolutionDescriptor_t convDesc,
                                            const cudnnTensorDescriptor_t yDesc,
                                            cudnnConvolutionFwdAlgo_t algo,
                                            size_t *sizeInBytes) {
  using fun_t = decltype(cudnnGetConvolutionForwardWorkspaceSize);
  static const std::string funName{"cudnnGetConvolutionForwardWorkspaceSize"};
  static fun_t *orig_cudnnGetConvolutionForwardWorkspaceSize =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionForwardWorkspaceSize");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionForwardWorkspaceSize(
        handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionForwardWorkspaceSize(
      handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 56
cudnnStatus_t CUDNNWINAPI
    cudnnConvolutionForward(cudnnHandle_t handle,
                            const void *alpha,
                            const cudnnTensorDescriptor_t xDesc,
                            const void *x,
                            const cudnnFilterDescriptor_t wDesc,
                            const void *w,
                            const cudnnConvolutionDescriptor_t convDesc,
                            cudnnConvolutionFwdAlgo_t algo,
                            void *workSpace,
                            size_t workSpaceSizeInBytes,
                            const void *beta,
                            const cudnnTensorDescriptor_t yDesc,
                            void *y) {
  using fun_t = decltype(cudnnConvolutionForward);
  static const std::string funName{"cudnnConvolutionForward"};
  static fun_t *orig_cudnnConvolutionForward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnConvolutionForward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnConvolutionForward(handle,
                                        alpha,
                                        xDesc,
                                        x,
                                        wDesc,
                                        w,
                                        convDesc,
                                        algo,
                                        workSpace,
                                        workSpaceSizeInBytes,
                                        beta,
                                        yDesc,
                                        y);
  }
  const auto tic = now();
  const auto res = orig_cudnnConvolutionForward(handle,
                                                alpha,
                                                xDesc,
                                                x,
                                                wDesc,
                                                w,
                                                convDesc,
                                                algo,
                                                workSpace,
                                                workSpaceSizeInBytes,
                                                beta,
                                                yDesc,
                                                y);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"workSpaceSizeInBytes", workSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 57
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBiasActivationForward(
    cudnnHandle_t handle,
    const void *alpha1,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *alpha2,
    const cudnnTensorDescriptor_t zDesc,
    const void *z,
    const cudnnTensorDescriptor_t biasDesc,
    const void *bias,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc,
    void *y) {
  using fun_t = decltype(cudnnConvolutionBiasActivationForward);
  static const std::string funName{"cudnnConvolutionBiasActivationForward"};
  static fun_t *orig_cudnnConvolutionBiasActivationForward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnConvolutionBiasActivationForward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnConvolutionBiasActivationForward(handle,
                                                      alpha1,
                                                      xDesc,
                                                      x,
                                                      wDesc,
                                                      w,
                                                      convDesc,
                                                      algo,
                                                      workSpace,
                                                      workSpaceSizeInBytes,
                                                      alpha2,
                                                      zDesc,
                                                      z,
                                                      biasDesc,
                                                      bias,
                                                      activationDesc,
                                                      yDesc,
                                                      y);
  }
  const auto tic = now();
  const auto res = orig_cudnnConvolutionBiasActivationForward(handle,
                                                              alpha1,
                                                              xDesc,
                                                              x,
                                                              wDesc,
                                                              w,
                                                              convDesc,
                                                              algo,
                                                              workSpace,
                                                              workSpaceSizeInBytes,
                                                              alpha2,
                                                              zDesc,
                                                              z,
                                                              biasDesc,
                                                              bias,
                                                              activationDesc,
                                                              yDesc,
                                                              y);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"workSpaceSizeInBytes", workSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 58
cudnnStatus_t CUDNNWINAPI
    cudnnConvolutionBackwardBias(cudnnHandle_t handle,
                                 const void *alpha,
                                 const cudnnTensorDescriptor_t dyDesc,
                                 const void *dy,
                                 const void *beta,
                                 const cudnnTensorDescriptor_t dbDesc,
                                 void *db) {
  using fun_t = decltype(cudnnConvolutionBackwardBias);
  static const std::string funName{"cudnnConvolutionBackwardBias"};
  static fun_t *orig_cudnnConvolutionBackwardBias =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnConvolutionBackwardBias");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, dbDesc, db);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, dbDesc, db);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 59
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
  using fun_t = decltype(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount);
  static const std::string funName{"cudnnGetConvolutionBackwardFilterAlgorithmMaxCount"};
  static fun_t *orig_cudnnGetConvolutionBackwardFilterAlgorithmMaxCount =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, count);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, count);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 60
cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
  using fun_t = decltype(cudnnFindConvolutionBackwardFilterAlgorithm);
  static const std::string funName{"cudnnFindConvolutionBackwardFilterAlgorithm"};
  static fun_t *orig_cudnnFindConvolutionBackwardFilterAlgorithm =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnFindConvolutionBackwardFilterAlgorithm");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnFindConvolutionBackwardFilterAlgorithm(handle,
                                                            xDesc,
                                                            dyDesc,
                                                            convDesc,
                                                            dwDesc,
                                                            requestedAlgoCount,
                                                            returnedAlgoCount,
                                                            perfResults);
  }
  const auto tic = now();
  const auto res = orig_cudnnFindConvolutionBackwardFilterAlgorithm(handle,
                                                                    xDesc,
                                                                    dyDesc,
                                                                    convDesc,
                                                                    dwDesc,
                                                                    requestedAlgoCount,
                                                                    returnedAlgoCount,
                                                                    perfResults);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"requestedAlgoCount", requestedAlgoCount}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 61
cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithmEx(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *y,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
    void *workSpace,
    size_t workSpaceSizeInBytes) {
  using fun_t = decltype(cudnnFindConvolutionBackwardFilterAlgorithmEx);
  static const std::string funName{"cudnnFindConvolutionBackwardFilterAlgorithmEx"};
  static fun_t *orig_cudnnFindConvolutionBackwardFilterAlgorithmEx =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnFindConvolutionBackwardFilterAlgorithmEx");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnFindConvolutionBackwardFilterAlgorithmEx(handle,
                                                              xDesc,
                                                              x,
                                                              dyDesc,
                                                              y,
                                                              convDesc,
                                                              dwDesc,
                                                              dw,
                                                              requestedAlgoCount,
                                                              returnedAlgoCount,
                                                              perfResults,
                                                              workSpace,
                                                              workSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnFindConvolutionBackwardFilterAlgorithmEx(handle,
                                                         xDesc,
                                                         x,
                                                         dyDesc,
                                                         y,
                                                         convDesc,
                                                         dwDesc,
                                                         dw,
                                                         requestedAlgoCount,
                                                         returnedAlgoCount,
                                                         perfResults,
                                                         workSpace,
                                                         workSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"requestedAlgoCount", requestedAlgoCount},
                                  {"workSpaceSizeInBytes", workSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 62
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    cudnnConvolutionBwdFilterPreference_t preference,
    size_t memoryLimitInBytes,
    cudnnConvolutionBwdFilterAlgo_t *algo) {
  using fun_t = decltype(cudnnGetConvolutionBackwardFilterAlgorithm);
  static const std::string funName{"cudnnGetConvolutionBackwardFilterAlgorithm"};
  static fun_t *orig_cudnnGetConvolutionBackwardFilterAlgorithm =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionBackwardFilterAlgorithm");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionBackwardFilterAlgorithm(
        handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, algo);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardFilterAlgorithm(
      handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, algo);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"memoryLimitInBytes", memoryLimitInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 63
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t srcDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
  using fun_t = decltype(cudnnGetConvolutionBackwardFilterAlgorithm_v7);
  static const std::string funName{"cudnnGetConvolutionBackwardFilterAlgorithm_v7"};
  static fun_t *orig_cudnnGetConvolutionBackwardFilterAlgorithm_v7 =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionBackwardFilterAlgorithm_v7");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle,
                                                              srcDesc,
                                                              diffDesc,
                                                              convDesc,
                                                              gradDesc,
                                                              requestedAlgoCount,
                                                              returnedAlgoCount,
                                                              perfResults);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle,
                                                                      srcDesc,
                                                                      diffDesc,
                                                                      convDesc,
                                                                      gradDesc,
                                                                      requestedAlgoCount,
                                                                      returnedAlgoCount,
                                                                      perfResults);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"requestedAlgoCount", requestedAlgoCount}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 64
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    size_t *sizeInBytes) {
  using fun_t = decltype(cudnnGetConvolutionBackwardFilterWorkspaceSize);
  static const std::string funName{"cudnnGetConvolutionBackwardFilterWorkspaceSize"};
  static fun_t *orig_cudnnGetConvolutionBackwardFilterWorkspaceSize =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionBackwardFilterWorkspaceSize");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardFilterWorkspaceSize(
      handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 65
cudnnStatus_t CUDNNWINAPI
    cudnnConvolutionBackwardFilter(cudnnHandle_t handle,
                                   const void *alpha,
                                   const cudnnTensorDescriptor_t xDesc,
                                   const void *x,
                                   const cudnnTensorDescriptor_t dyDesc,
                                   const void *dy,
                                   const cudnnConvolutionDescriptor_t convDesc,
                                   cudnnConvolutionBwdFilterAlgo_t algo,
                                   void *workSpace,
                                   size_t workSpaceSizeInBytes,
                                   const void *beta,
                                   const cudnnFilterDescriptor_t dwDesc,
                                   void *dw) {
  using fun_t = decltype(cudnnConvolutionBackwardFilter);
  static const std::string funName{"cudnnConvolutionBackwardFilter"};
  static fun_t *orig_cudnnConvolutionBackwardFilter =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnConvolutionBackwardFilter");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnConvolutionBackwardFilter(handle,
                                               alpha,
                                               xDesc,
                                               x,
                                               dyDesc,
                                               dy,
                                               convDesc,
                                               algo,
                                               workSpace,
                                               workSpaceSizeInBytes,
                                               beta,
                                               dwDesc,
                                               dw);
  }
  const auto tic = now();
  const auto res = orig_cudnnConvolutionBackwardFilter(handle,
                                                       alpha,
                                                       xDesc,
                                                       x,
                                                       dyDesc,
                                                       dy,
                                                       convDesc,
                                                       algo,
                                                       workSpace,
                                                       workSpaceSizeInBytes,
                                                       beta,
                                                       dwDesc,
                                                       dw);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"workSpaceSizeInBytes", workSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 66
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
  using fun_t = decltype(cudnnGetConvolutionBackwardDataAlgorithmMaxCount);
  static const std::string funName{"cudnnGetConvolutionBackwardDataAlgorithmMaxCount"};
  static fun_t *orig_cudnnGetConvolutionBackwardDataAlgorithmMaxCount =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionBackwardDataAlgorithmMaxCount");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, count);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, count);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 67
cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
  using fun_t = decltype(cudnnFindConvolutionBackwardDataAlgorithm);
  static const std::string funName{"cudnnFindConvolutionBackwardDataAlgorithm"};
  static fun_t *orig_cudnnFindConvolutionBackwardDataAlgorithm =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnFindConvolutionBackwardDataAlgorithm");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnFindConvolutionBackwardDataAlgorithm(handle,
                                                          wDesc,
                                                          dyDesc,
                                                          convDesc,
                                                          dxDesc,
                                                          requestedAlgoCount,
                                                          returnedAlgoCount,
                                                          perfResults);
  }
  const auto tic = now();
  const auto res = orig_cudnnFindConvolutionBackwardDataAlgorithm(handle,
                                                                  wDesc,
                                                                  dyDesc,
                                                                  convDesc,
                                                                  dxDesc,
                                                                  requestedAlgoCount,
                                                                  returnedAlgoCount,
                                                                  perfResults);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"requestedAlgoCount", requestedAlgoCount}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 68
cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithmEx(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults,
    void *workSpace,
    size_t workSpaceSizeInBytes) {
  using fun_t = decltype(cudnnFindConvolutionBackwardDataAlgorithmEx);
  static const std::string funName{"cudnnFindConvolutionBackwardDataAlgorithmEx"};
  static fun_t *orig_cudnnFindConvolutionBackwardDataAlgorithmEx =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnFindConvolutionBackwardDataAlgorithmEx");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnFindConvolutionBackwardDataAlgorithmEx(handle,
                                                            wDesc,
                                                            w,
                                                            dyDesc,
                                                            dy,
                                                            convDesc,
                                                            dxDesc,
                                                            dx,
                                                            requestedAlgoCount,
                                                            returnedAlgoCount,
                                                            perfResults,
                                                            workSpace,
                                                            workSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnFindConvolutionBackwardDataAlgorithmEx(handle,
                                                                    wDesc,
                                                                    w,
                                                                    dyDesc,
                                                                    dy,
                                                                    convDesc,
                                                                    dxDesc,
                                                                    dx,
                                                                    requestedAlgoCount,
                                                                    returnedAlgoCount,
                                                                    perfResults,
                                                                    workSpace,
                                                                    workSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"requestedAlgoCount", requestedAlgoCount},
                                  {"workSpaceSizeInBytes", workSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 69
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataPreference_t preference,
    size_t memoryLimitInBytes,
    cudnnConvolutionBwdDataAlgo_t *algo) {
  using fun_t = decltype(cudnnGetConvolutionBackwardDataAlgorithm);
  static const std::string funName{"cudnnGetConvolutionBackwardDataAlgorithm"};
  static fun_t *orig_cudnnGetConvolutionBackwardDataAlgorithm =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionBackwardDataAlgorithm");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionBackwardDataAlgorithm(
        handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, algo);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardDataAlgorithm(
      handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, algo);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"memoryLimitInBytes", memoryLimitInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 70
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
  using fun_t = decltype(cudnnGetConvolutionBackwardDataAlgorithm_v7);
  static const std::string funName{"cudnnGetConvolutionBackwardDataAlgorithm_v7"};
  static fun_t *orig_cudnnGetConvolutionBackwardDataAlgorithm_v7 =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionBackwardDataAlgorithm_v7");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionBackwardDataAlgorithm_v7(handle,
                                                            filterDesc,
                                                            diffDesc,
                                                            convDesc,
                                                            gradDesc,
                                                            requestedAlgoCount,
                                                            returnedAlgoCount,
                                                            perfResults);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardDataAlgorithm_v7(handle,
                                                                    filterDesc,
                                                                    diffDesc,
                                                                    convDesc,
                                                                    gradDesc,
                                                                    requestedAlgoCount,
                                                                    returnedAlgoCount,
                                                                    perfResults);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"requestedAlgoCount", requestedAlgoCount}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 71
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes) {
  using fun_t = decltype(cudnnGetConvolutionBackwardDataWorkspaceSize);
  static const std::string funName{"cudnnGetConvolutionBackwardDataWorkspaceSize"};
  static fun_t *orig_cudnnGetConvolutionBackwardDataWorkspaceSize =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetConvolutionBackwardDataWorkspaceSize");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardDataWorkspaceSize(
      handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 72
cudnnStatus_t CUDNNWINAPI
    cudnnConvolutionBackwardData(cudnnHandle_t handle,
                                 const void *alpha,
                                 const cudnnFilterDescriptor_t wDesc,
                                 const void *w,
                                 const cudnnTensorDescriptor_t dyDesc,
                                 const void *dy,
                                 const cudnnConvolutionDescriptor_t convDesc,
                                 cudnnConvolutionBwdDataAlgo_t algo,
                                 void *workSpace,
                                 size_t workSpaceSizeInBytes,
                                 const void *beta,
                                 const cudnnTensorDescriptor_t dxDesc,
                                 void *dx) {
  using fun_t = decltype(cudnnConvolutionBackwardData);
  static const std::string funName{"cudnnConvolutionBackwardData"};
  static fun_t *orig_cudnnConvolutionBackwardData =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnConvolutionBackwardData");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnConvolutionBackwardData(handle,
                                             alpha,
                                             wDesc,
                                             w,
                                             dyDesc,
                                             dy,
                                             convDesc,
                                             algo,
                                             workSpace,
                                             workSpaceSizeInBytes,
                                             beta,
                                             dxDesc,
                                             dx);
  }
  const auto tic = now();
  const auto res = orig_cudnnConvolutionBackwardData(handle,
                                                     alpha,
                                                     wDesc,
                                                     w,
                                                     dyDesc,
                                                     dy,
                                                     convDesc,
                                                     algo,
                                                     workSpace,
                                                     workSpaceSizeInBytes,
                                                     beta,
                                                     dxDesc,
                                                     dx);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"workSpaceSizeInBytes", workSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 73
cudnnStatus_t CUDNNWINAPI cudnnIm2Col(cudnnHandle_t handle,
                                      const cudnnTensorDescriptor_t xDesc,
                                      const void *x,
                                      const cudnnFilterDescriptor_t wDesc,
                                      const cudnnConvolutionDescriptor_t convDesc,
                                      void *colBuffer) {
  using fun_t = decltype(cudnnIm2Col);
  static const std::string funName{"cudnnIm2Col"};
  static fun_t *orig_cudnnIm2Col = (fun_t *) dlsym(RTLD_NEXT, "cudnnIm2Col");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnIm2Col(handle, xDesc, x, wDesc, convDesc, colBuffer);
  }
  const auto tic = now();
  const auto res = orig_cudnnIm2Col(handle, xDesc, x, wDesc, convDesc, colBuffer);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 74
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxForward(cudnnHandle_t handle,
                                              cudnnSoftmaxAlgorithm_t algo,
                                              cudnnSoftmaxMode_t mode,
                                              const void *alpha,
                                              const cudnnTensorDescriptor_t xDesc,
                                              const void *x,
                                              const void *beta,
                                              const cudnnTensorDescriptor_t yDesc,
                                              void *y) {
  using fun_t = decltype(cudnnSoftmaxForward);
  static const std::string funName{"cudnnSoftmaxForward"};
  static fun_t *orig_cudnnSoftmaxForward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSoftmaxForward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 75
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxBackward(cudnnHandle_t handle,
                                               cudnnSoftmaxAlgorithm_t algo,
                                               cudnnSoftmaxMode_t mode,
                                               const void *alpha,
                                               const cudnnTensorDescriptor_t yDesc,
                                               const void *y,
                                               const cudnnTensorDescriptor_t dyDesc,
                                               const void *dy,
                                               const void *beta,
                                               const cudnnTensorDescriptor_t dxDesc,
                                               void *dx) {
  using fun_t = decltype(cudnnSoftmaxBackward);
  static const std::string funName{"cudnnSoftmaxBackward"};
  static fun_t *orig_cudnnSoftmaxBackward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSoftmaxBackward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSoftmaxBackward(
        handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);
  }
  const auto tic = now();
  const auto res = orig_cudnnSoftmaxBackward(
      handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 76
cudnnStatus_t CUDNNWINAPI
    cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc) {
  using fun_t = decltype(cudnnCreatePoolingDescriptor);
  static const std::string funName{"cudnnCreatePoolingDescriptor"};
  static fun_t *orig_cudnnCreatePoolingDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCreatePoolingDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreatePoolingDescriptor(poolingDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreatePoolingDescriptor(poolingDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 77
cudnnStatus_t CUDNNWINAPI
    cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                cudnnPoolingMode_t mode,
                                cudnnNanPropagation_t maxpoolingNanOpt,
                                int windowHeight,
                                int windowWidth,
                                int verticalPadding,
                                int horizontalPadding,
                                int verticalStride,
                                int horizontalStride) {
  using fun_t = decltype(cudnnSetPooling2dDescriptor);
  static const std::string funName{"cudnnSetPooling2dDescriptor"};
  static fun_t *orig_cudnnSetPooling2dDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetPooling2dDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetPooling2dDescriptor(poolingDesc,
                                            mode,
                                            maxpoolingNanOpt,
                                            windowHeight,
                                            windowWidth,
                                            verticalPadding,
                                            horizontalPadding,
                                            verticalStride,
                                            horizontalStride);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetPooling2dDescriptor(poolingDesc,
                                                    mode,
                                                    maxpoolingNanOpt,
                                                    windowHeight,
                                                    windowWidth,
                                                    verticalPadding,
                                                    horizontalPadding,
                                                    verticalStride,
                                                    horizontalStride);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"windowHeight", windowHeight},
                                               {"windowWidth", windowWidth},
                                               {"verticalPadding", verticalPadding},
                                               {"horizontalPadding", horizontalPadding},
                                               {"verticalStride", verticalStride},
                                               {"horizontalStride", horizontalStride}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 78
cudnnStatus_t CUDNNWINAPI
    cudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t poolingDesc,
                                cudnnPoolingMode_t *mode,
                                cudnnNanPropagation_t *maxpoolingNanOpt,
                                int *windowHeight,
                                int *windowWidth,
                                int *verticalPadding,
                                int *horizontalPadding,
                                int *verticalStride,
                                int *horizontalStride) {
  using fun_t = decltype(cudnnGetPooling2dDescriptor);
  static const std::string funName{"cudnnGetPooling2dDescriptor"};
  static fun_t *orig_cudnnGetPooling2dDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetPooling2dDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetPooling2dDescriptor(poolingDesc,
                                            mode,
                                            maxpoolingNanOpt,
                                            windowHeight,
                                            windowWidth,
                                            verticalPadding,
                                            horizontalPadding,
                                            verticalStride,
                                            horizontalStride);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetPooling2dDescriptor(poolingDesc,
                                                    mode,
                                                    maxpoolingNanOpt,
                                                    windowHeight,
                                                    windowWidth,
                                                    verticalPadding,
                                                    horizontalPadding,
                                                    verticalStride,
                                                    horizontalStride);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 79
cudnnStatus_t CUDNNWINAPI
    cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                const cudnnPoolingMode_t mode,
                                const cudnnNanPropagation_t maxpoolingNanOpt,
                                int nbDims,
                                const int windowDimA[],
                                const int paddingA[],
                                const int strideA[]) {
  using fun_t = decltype(cudnnSetPoolingNdDescriptor);
  static const std::string funName{"cudnnSetPoolingNdDescriptor"};
  static fun_t *orig_cudnnSetPoolingNdDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetPoolingNdDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetPoolingNdDescriptor(
        poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetPoolingNdDescriptor(
      poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"nbDims", nbDims}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 80
cudnnStatus_t CUDNNWINAPI
    cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t poolingDesc,
                                int nbDimsRequested,
                                cudnnPoolingMode_t *mode,
                                cudnnNanPropagation_t *maxpoolingNanOpt,
                                int *nbDims,
                                int windowDimA[],
                                int paddingA[],
                                int strideA[]) {
  using fun_t = decltype(cudnnGetPoolingNdDescriptor);
  static const std::string funName{"cudnnGetPoolingNdDescriptor"};
  static fun_t *orig_cudnnGetPoolingNdDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetPoolingNdDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetPoolingNdDescriptor(poolingDesc,
                                            nbDimsRequested,
                                            mode,
                                            maxpoolingNanOpt,
                                            nbDims,
                                            windowDimA,
                                            paddingA,
                                            strideA);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetPoolingNdDescriptor(poolingDesc,
                                                    nbDimsRequested,
                                                    mode,
                                                    maxpoolingNanOpt,
                                                    nbDims,
                                                    windowDimA,
                                                    paddingA,
                                                    strideA);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"nbDimsRequested", nbDimsRequested}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 81
cudnnStatus_t CUDNNWINAPI
    cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                      const cudnnTensorDescriptor_t inputTensorDesc,
                                      int nbDims,
                                      int outputTensorDimA[]) {
  using fun_t = decltype(cudnnGetPoolingNdForwardOutputDim);
  static const std::string funName{"cudnnGetPoolingNdForwardOutputDim"};
  static fun_t *orig_cudnnGetPoolingNdForwardOutputDim =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetPoolingNdForwardOutputDim");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetPoolingNdForwardOutputDim(
        poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetPoolingNdForwardOutputDim(
      poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"nbDims", nbDims}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 82
cudnnStatus_t CUDNNWINAPI
    cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                      const cudnnTensorDescriptor_t inputTensorDesc,
                                      int *n,
                                      int *c,
                                      int *h,
                                      int *w) {
  using fun_t = decltype(cudnnGetPooling2dForwardOutputDim);
  static const std::string funName{"cudnnGetPooling2dForwardOutputDim"};
  static fun_t *orig_cudnnGetPooling2dForwardOutputDim =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetPooling2dForwardOutputDim");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetPooling2dForwardOutputDim(
        poolingDesc, inputTensorDesc, n, c, h, w);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnGetPooling2dForwardOutputDim(poolingDesc, inputTensorDesc, n, c, h, w);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 83
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
  using fun_t = decltype(cudnnDestroyPoolingDescriptor);
  static const std::string funName{"cudnnDestroyPoolingDescriptor"};
  static fun_t *orig_cudnnDestroyPoolingDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroyPoolingDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroyPoolingDescriptor(poolingDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroyPoolingDescriptor(poolingDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 84
cudnnStatus_t CUDNNWINAPI cudnnPoolingForward(cudnnHandle_t handle,
                                              const cudnnPoolingDescriptor_t poolingDesc,
                                              const void *alpha,
                                              const cudnnTensorDescriptor_t xDesc,
                                              const void *x,
                                              const void *beta,
                                              const cudnnTensorDescriptor_t yDesc,
                                              void *y) {
  using fun_t = decltype(cudnnPoolingForward);
  static const std::string funName{"cudnnPoolingForward"};
  static fun_t *orig_cudnnPoolingForward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnPoolingForward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 85
cudnnStatus_t CUDNNWINAPI cudnnPoolingBackward(cudnnHandle_t handle,
                                               const cudnnPoolingDescriptor_t poolingDesc,
                                               const void *alpha,
                                               const cudnnTensorDescriptor_t yDesc,
                                               const void *y,
                                               const cudnnTensorDescriptor_t dyDesc,
                                               const void *dy,
                                               const cudnnTensorDescriptor_t xDesc,
                                               const void *x,
                                               const void *beta,
                                               const cudnnTensorDescriptor_t dxDesc,
                                               void *dx) {
  using fun_t = decltype(cudnnPoolingBackward);
  static const std::string funName{"cudnnPoolingBackward"};
  static fun_t *orig_cudnnPoolingBackward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnPoolingBackward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnPoolingBackward(
        handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
  }
  const auto tic = now();
  const auto res = orig_cudnnPoolingBackward(
      handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 86
cudnnStatus_t CUDNNWINAPI
    cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc) {
  using fun_t = decltype(cudnnCreateActivationDescriptor);
  static const std::string funName{"cudnnCreateActivationDescriptor"};
  static fun_t *orig_cudnnCreateActivationDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCreateActivationDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreateActivationDescriptor(activationDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreateActivationDescriptor(activationDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 87
cudnnStatus_t CUDNNWINAPI
    cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                                 cudnnActivationMode_t mode,
                                 cudnnNanPropagation_t reluNanOpt,
                                 double coef) {
  using fun_t = decltype(cudnnSetActivationDescriptor);
  static const std::string funName{"cudnnSetActivationDescriptor"};
  static fun_t *orig_cudnnSetActivationDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetActivationDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnSetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"coef", coef}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 88
cudnnStatus_t CUDNNWINAPI
    cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,
                                 cudnnActivationMode_t *mode,
                                 cudnnNanPropagation_t *reluNanOpt,
                                 double *coef) {
  using fun_t = decltype(cudnnGetActivationDescriptor);
  static const std::string funName{"cudnnGetActivationDescriptor"};
  static fun_t *orig_cudnnGetActivationDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetActivationDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnGetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 89
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
  using fun_t = decltype(cudnnDestroyActivationDescriptor);
  static const std::string funName{"cudnnDestroyActivationDescriptor"};
  static fun_t *orig_cudnnDestroyActivationDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroyActivationDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroyActivationDescriptor(activationDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroyActivationDescriptor(activationDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 90
cudnnStatus_t CUDNNWINAPI
    cudnnActivationForward(cudnnHandle_t handle,
                           cudnnActivationDescriptor_t activationDesc,
                           const void *alpha,
                           const cudnnTensorDescriptor_t xDesc,
                           const void *x,
                           const void *beta,
                           const cudnnTensorDescriptor_t yDesc,
                           void *y) {
  using fun_t = decltype(cudnnActivationForward);
  static const std::string funName{"cudnnActivationForward"};
  static fun_t *orig_cudnnActivationForward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnActivationForward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnActivationForward(
        handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
  }
  const auto tic = now();
  const auto res = orig_cudnnActivationForward(
      handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 91
cudnnStatus_t CUDNNWINAPI
    cudnnActivationBackward(cudnnHandle_t handle,
                            cudnnActivationDescriptor_t activationDesc,
                            const void *alpha,
                            const cudnnTensorDescriptor_t yDesc,
                            const void *y,
                            const cudnnTensorDescriptor_t dyDesc,
                            const void *dy,
                            const cudnnTensorDescriptor_t xDesc,
                            const void *x,
                            const void *beta,
                            const cudnnTensorDescriptor_t dxDesc,
                            void *dx) {
  using fun_t = decltype(cudnnActivationBackward);
  static const std::string funName{"cudnnActivationBackward"};
  static fun_t *orig_cudnnActivationBackward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnActivationBackward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnActivationBackward(
        handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
  }
  const auto tic = now();
  const auto res = orig_cudnnActivationBackward(
      handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 92
cudnnStatus_t CUDNNWINAPI cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc) {
  using fun_t = decltype(cudnnCreateLRNDescriptor);
  static const std::string funName{"cudnnCreateLRNDescriptor"};
  static fun_t *orig_cudnnCreateLRNDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCreateLRNDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreateLRNDescriptor(normDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreateLRNDescriptor(normDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 93
cudnnStatus_t CUDNNWINAPI cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
                                                unsigned lrnN,
                                                double lrnAlpha,
                                                double lrnBeta,
                                                double lrnK) {
  using fun_t = decltype(cudnnSetLRNDescriptor);
  static const std::string funName{"cudnnSetLRNDescriptor"};
  static fun_t *orig_cudnnSetLRNDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetLRNDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"lrnN", lrnN},
                                               {"lrnAlpha", lrnAlpha},
                                               {"lrnBeta", lrnBeta},
                                               {"lrnK", lrnK}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 94
cudnnStatus_t CUDNNWINAPI cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
                                                unsigned *lrnN,
                                                double *lrnAlpha,
                                                double *lrnBeta,
                                                double *lrnK) {
  using fun_t = decltype(cudnnGetLRNDescriptor);
  static const std::string funName{"cudnnGetLRNDescriptor"};
  static fun_t *orig_cudnnGetLRNDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetLRNDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 95
cudnnStatus_t CUDNNWINAPI cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc) {
  using fun_t = decltype(cudnnDestroyLRNDescriptor);
  static const std::string funName{"cudnnDestroyLRNDescriptor"};
  static fun_t *orig_cudnnDestroyLRNDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroyLRNDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroyLRNDescriptor(lrnDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroyLRNDescriptor(lrnDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 96
cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelForward(cudnnHandle_t handle,
                                                      cudnnLRNDescriptor_t normDesc,
                                                      cudnnLRNMode_t lrnMode,
                                                      const void *alpha,
                                                      const cudnnTensorDescriptor_t xDesc,
                                                      const void *x,
                                                      const void *beta,
                                                      const cudnnTensorDescriptor_t yDesc,
                                                      void *y) {
  using fun_t = decltype(cudnnLRNCrossChannelForward);
  static const std::string funName{"cudnnLRNCrossChannelForward"};
  static fun_t *orig_cudnnLRNCrossChannelForward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnLRNCrossChannelForward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnLRNCrossChannelForward(
        handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);
  }
  const auto tic = now();
  const auto res = orig_cudnnLRNCrossChannelForward(
      handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 97
cudnnStatus_t CUDNNWINAPI
    cudnnLRNCrossChannelBackward(cudnnHandle_t handle,
                                 cudnnLRNDescriptor_t normDesc,
                                 cudnnLRNMode_t lrnMode,
                                 const void *alpha,
                                 const cudnnTensorDescriptor_t yDesc,
                                 const void *y,
                                 const cudnnTensorDescriptor_t dyDesc,
                                 const void *dy,
                                 const cudnnTensorDescriptor_t xDesc,
                                 const void *x,
                                 const void *beta,
                                 const cudnnTensorDescriptor_t dxDesc,
                                 void *dx) {
  using fun_t = decltype(cudnnLRNCrossChannelBackward);
  static const std::string funName{"cudnnLRNCrossChannelBackward"};
  static fun_t *orig_cudnnLRNCrossChannelBackward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnLRNCrossChannelBackward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnLRNCrossChannelBackward(handle,
                                             normDesc,
                                             lrnMode,
                                             alpha,
                                             yDesc,
                                             y,
                                             dyDesc,
                                             dy,
                                             xDesc,
                                             x,
                                             beta,
                                             dxDesc,
                                             dx);
  }
  const auto tic = now();
  const auto res = orig_cudnnLRNCrossChannelBackward(
      handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 98
cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationForward(
    cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc, /* same desc for means, temp, temp2 */
    const void *x,
    const void *means, /* if NULL, means are assumed to be zero */
    void *temp,
    void *temp2,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y) {
  using fun_t = decltype(cudnnDivisiveNormalizationForward);
  static const std::string funName{"cudnnDivisiveNormalizationForward"};
  static fun_t *orig_cudnnDivisiveNormalizationForward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDivisiveNormalizationForward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDivisiveNormalizationForward(
        handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y);
  }
  const auto tic = now();
  const auto res = orig_cudnnDivisiveNormalizationForward(
      handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 99
cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationBackward(
    cudnnHandle_t handle,
    cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc, /* same desc for x, means, dy, temp, temp2 */
    const void *x,
    const void *means, /* if NULL, means are assumed to be zero */
    const void *dy,
    void *temp,
    void *temp2,
    const void *beta,
    const cudnnTensorDescriptor_t dXdMeansDesc, /* same desc for dx, dMeans */
    void *dx,                                   /* output x differential */
    void *dMeans) {
  using fun_t = decltype(cudnnDivisiveNormalizationBackward);
  static const std::string funName{"cudnnDivisiveNormalizationBackward"};
  static fun_t *orig_cudnnDivisiveNormalizationBackward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDivisiveNormalizationBackward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDivisiveNormalizationBackward(handle,
                                                   normDesc,
                                                   mode,
                                                   alpha,
                                                   xDesc,
                                                   x,
                                                   means,
                                                   dy,
                                                   temp,
                                                   temp2,
                                                   beta,
                                                   dXdMeansDesc,
                                                   dx,
                                                   dMeans);
  }
  const auto tic = now();
  const auto res = orig_cudnnDivisiveNormalizationBackward(handle,
                                                           normDesc,
                                                           mode,
                                                           alpha,
                                                           xDesc,
                                                           x,
                                                           means,
                                                           dy,
                                                           temp,
                                                           temp2,
                                                           beta,
                                                           dXdMeansDesc,
                                                           dx,
                                                           dMeans);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 100
cudnnStatus_t CUDNNWINAPI
    cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,
                                  const cudnnTensorDescriptor_t xDesc,
                                  cudnnBatchNormMode_t mode) {
  using fun_t = decltype(cudnnDeriveBNTensorDescriptor);
  static const std::string funName{"cudnnDeriveBNTensorDescriptor"};
  static fun_t *orig_cudnnDeriveBNTensorDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDeriveBNTensorDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDeriveBNTensorDescriptor(derivedBnDesc, xDesc, mode);
  }
  const auto tic = now();
  const auto res = orig_cudnnDeriveBNTensorDescriptor(derivedBnDesc, xDesc, mode);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 101
cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,

    const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,  /* beta[0] = dest layer blend factor */

    const cudnnTensorDescriptor_t xDesc,
    const void *x, /* NxCxHxW */
    const cudnnTensorDescriptor_t yDesc,
    void *y, /* NxCxHxW */

    /* Shared desc for the next 6 tensors in the argument list.
       Data type to be set as follows:
       type = (typeOf(x) == double) ? double : float
       Dimensions for this descriptor depend on normalization mode
       - Spatial Normalization : tensors are expected to have dims 1xCx1x1
        (normalization is performed across NxHxW)
       - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW
        (normalization is performed across N) */
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,

    /* 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation */
    const void *bnScale,
    const void *bnBias,

    /* MUST use factor=1 in the very first call of a complete training cycle.
       Use a factor=1/(1+n) at N-th call to the function to get
       Cumulative Moving Average (CMA) behavior
       CMA[n] = (x[1]+...+x[n])/n
       Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
       ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
       CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1) */
    double exponentialAverageFactor,

    /* Used in Training phase only.
       runningMean = newMean*factor + runningMean*(1-factor) */
    void *resultRunningMean,
    /* Output in training mode, input in inference. Is the moving average
       of  variance[x] (factor is applied in the same way as for runningMean) */
    void *resultRunningVariance,

    /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward
       functions. */
    double epsilon,

    /* Optionally save intermediate results from the forward pass here
       - can be reused to speed up backward pass. NULL if unused */
    void *resultSaveMean,
    void *resultSaveInvVariance) {
  using fun_t = decltype(cudnnBatchNormalizationForwardTraining);
  static const std::string funName{"cudnnBatchNormalizationForwardTraining"};
  static fun_t *orig_cudnnBatchNormalizationForwardTraining =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnBatchNormalizationForwardTraining");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnBatchNormalizationForwardTraining(handle,
                                                       mode,
                                                       alpha,
                                                       beta,
                                                       xDesc,
                                                       x,
                                                       yDesc,
                                                       y,
                                                       bnScaleBiasMeanVarDesc,
                                                       bnScale,
                                                       bnBias,
                                                       exponentialAverageFactor,
                                                       resultRunningMean,
                                                       resultRunningVariance,
                                                       epsilon,
                                                       resultSaveMean,
                                                       resultSaveInvVariance);
  }
  const auto tic = now();
  const auto res = orig_cudnnBatchNormalizationForwardTraining(handle,
                                                               mode,
                                                               alpha,
                                                               beta,
                                                               xDesc,
                                                               x,
                                                               yDesc,
                                                               y,
                                                               bnScaleBiasMeanVarDesc,
                                                               bnScale,
                                                               bnBias,
                                                               exponentialAverageFactor,
                                                               resultRunningMean,
                                                               resultRunningVariance,
                                                               epsilon,
                                                               resultSaveMean,
                                                               resultSaveInvVariance);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"exponentialAverageFactor", exponentialAverageFactor},
                                  {"epsilon", epsilon}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 102
cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,  /* beta[0] = dest layer blend factor */
    const cudnnTensorDescriptor_t xDesc,
    const void *x, /* NxCxHxW */
    const cudnnTensorDescriptor_t yDesc,
    void *y, /* NxCxHxW */
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale,
    const void *bnBias,
    const void *estimatedMean,
    const void *estimatedVariance,
    double epsilon) {
  using fun_t = decltype(cudnnBatchNormalizationForwardInference);
  static const std::string funName{"cudnnBatchNormalizationForwardInference"};
  static fun_t *orig_cudnnBatchNormalizationForwardInference =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnBatchNormalizationForwardInference");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnBatchNormalizationForwardInference(handle,
                                                        mode,
                                                        alpha,
                                                        beta,
                                                        xDesc,
                                                        x,
                                                        yDesc,
                                                        y,
                                                        bnScaleBiasMeanVarDesc,
                                                        bnScale,
                                                        bnBias,
                                                        estimatedMean,
                                                        estimatedVariance,
                                                        epsilon);
  }
  const auto tic = now();
  const auto res = orig_cudnnBatchNormalizationForwardInference(handle,
                                                                mode,
                                                                alpha,
                                                                beta,
                                                                xDesc,
                                                                x,
                                                                yDesc,
                                                                y,
                                                                bnScaleBiasMeanVarDesc,
                                                                bnScale,
                                                                bnBias,
                                                                estimatedMean,
                                                                estimatedVariance,
                                                                epsilon);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"epsilon", epsilon}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 103
cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackward(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, /* same desc for x, dx, dy */
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    /* Shared tensor desc for the 4 tensors below */
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScale, /* bnBias doesn't affect backpropagation */
    /* scale and bias diff are not backpropagated below this layer */
    void *dBnScaleResult,
    void *dBnBiasResult,
    /* Same epsilon as forward pass */
    double epsilon,

    /* Optionally cached intermediate results from
       forward pass */
    const void *savedMean,
    const void *savedInvVariance) {
  using fun_t = decltype(cudnnBatchNormalizationBackward);
  static const std::string funName{"cudnnBatchNormalizationBackward"};
  static fun_t *orig_cudnnBatchNormalizationBackward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnBatchNormalizationBackward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnBatchNormalizationBackward(handle,
                                                mode,
                                                alphaDataDiff,
                                                betaDataDiff,
                                                alphaParamDiff,
                                                betaParamDiff,
                                                xDesc,
                                                x,
                                                dyDesc,
                                                dy,
                                                dxDesc,
                                                dx,
                                                dBnScaleBiasDesc,
                                                bnScale,
                                                dBnScaleResult,
                                                dBnBiasResult,
                                                epsilon,
                                                savedMean,
                                                savedInvVariance);
  }
  const auto tic = now();
  const auto res = orig_cudnnBatchNormalizationBackward(handle,
                                                        mode,
                                                        alphaDataDiff,
                                                        betaDataDiff,
                                                        alphaParamDiff,
                                                        betaParamDiff,
                                                        xDesc,
                                                        x,
                                                        dyDesc,
                                                        dy,
                                                        dxDesc,
                                                        dx,
                                                        dBnScaleBiasDesc,
                                                        bnScale,
                                                        dBnScaleResult,
                                                        dBnBiasResult,
                                                        epsilon,
                                                        savedMean,
                                                        savedInvVariance);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"epsilon", epsilon}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 104
cudnnStatus_t CUDNNWINAPI
    cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t *stDesc) {
  using fun_t = decltype(cudnnCreateSpatialTransformerDescriptor);
  static const std::string funName{"cudnnCreateSpatialTransformerDescriptor"};
  static fun_t *orig_cudnnCreateSpatialTransformerDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCreateSpatialTransformerDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreateSpatialTransformerDescriptor(stDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreateSpatialTransformerDescriptor(stDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 105
cudnnStatus_t CUDNNWINAPI
    cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t stDesc,
                                           cudnnSamplerType_t samplerType,
                                           cudnnDataType_t dataType,
                                           const int nbDims,
                                           const int dimA[]) {
  using fun_t = decltype(cudnnSetSpatialTransformerNdDescriptor);
  static const std::string funName{"cudnnSetSpatialTransformerNdDescriptor"};
  static fun_t *orig_cudnnSetSpatialTransformerNdDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetSpatialTransformerNdDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetSpatialTransformerNdDescriptor(
        stDesc, samplerType, dataType, nbDims, dimA);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetSpatialTransformerNdDescriptor(
      stDesc, samplerType, dataType, nbDims, dimA);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"nbDims", nbDims}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 106
cudnnStatus_t CUDNNWINAPI
    cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t stDesc) {
  using fun_t = decltype(cudnnDestroySpatialTransformerDescriptor);
  static const std::string funName{"cudnnDestroySpatialTransformerDescriptor"};
  static fun_t *orig_cudnnDestroySpatialTransformerDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroySpatialTransformerDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroySpatialTransformerDescriptor(stDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroySpatialTransformerDescriptor(stDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 107
cudnnStatus_t CUDNNWINAPI
    cudnnSpatialTfGridGeneratorForward(cudnnHandle_t handle,
                                       const cudnnSpatialTransformerDescriptor_t stDesc,
                                       const void *theta,
                                       void *grid) {
  using fun_t = decltype(cudnnSpatialTfGridGeneratorForward);
  static const std::string funName{"cudnnSpatialTfGridGeneratorForward"};
  static fun_t *orig_cudnnSpatialTfGridGeneratorForward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSpatialTfGridGeneratorForward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSpatialTfGridGeneratorForward(handle, stDesc, theta, grid);
  }
  const auto tic = now();
  const auto res = orig_cudnnSpatialTfGridGeneratorForward(handle, stDesc, theta, grid);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 108
cudnnStatus_t CUDNNWINAPI
    cudnnSpatialTfGridGeneratorBackward(cudnnHandle_t handle,
                                        const cudnnSpatialTransformerDescriptor_t stDesc,
                                        const void *dgrid,
                                        void *dtheta) {
  using fun_t = decltype(cudnnSpatialTfGridGeneratorBackward);
  static const std::string funName{"cudnnSpatialTfGridGeneratorBackward"};
  static fun_t *orig_cudnnSpatialTfGridGeneratorBackward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSpatialTfGridGeneratorBackward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSpatialTfGridGeneratorBackward(handle, stDesc, dgrid, dtheta);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnSpatialTfGridGeneratorBackward(handle, stDesc, dgrid, dtheta);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 109
cudnnStatus_t CUDNNWINAPI
    cudnnSpatialTfSamplerForward(cudnnHandle_t handle,
                                 cudnnSpatialTransformerDescriptor_t stDesc,
                                 const void *alpha,
                                 const cudnnTensorDescriptor_t xDesc,
                                 const void *x,
                                 const void *grid,
                                 const void *beta,
                                 cudnnTensorDescriptor_t yDesc,
                                 void *y) {
  using fun_t = decltype(cudnnSpatialTfSamplerForward);
  static const std::string funName{"cudnnSpatialTfSamplerForward"};
  static fun_t *orig_cudnnSpatialTfSamplerForward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSpatialTfSamplerForward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSpatialTfSamplerForward(
        handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y);
  }
  const auto tic = now();
  const auto res = orig_cudnnSpatialTfSamplerForward(
      handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 110
cudnnStatus_t CUDNNWINAPI
    cudnnSpatialTfSamplerBackward(cudnnHandle_t handle,
                                  cudnnSpatialTransformerDescriptor_t stDesc,
                                  const void *alpha,
                                  const cudnnTensorDescriptor_t xDesc,
                                  const void *x,
                                  const void *beta,
                                  const cudnnTensorDescriptor_t dxDesc,
                                  void *dx,
                                  const void *alphaDgrid,
                                  const cudnnTensorDescriptor_t dyDesc,
                                  const void *dy,
                                  const void *grid,
                                  const void *betaDgrid,
                                  void *dgrid) {
  using fun_t = decltype(cudnnSpatialTfSamplerBackward);
  static const std::string funName{"cudnnSpatialTfSamplerBackward"};
  static fun_t *orig_cudnnSpatialTfSamplerBackward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSpatialTfSamplerBackward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSpatialTfSamplerBackward(handle,
                                              stDesc,
                                              alpha,
                                              xDesc,
                                              x,
                                              beta,
                                              dxDesc,
                                              dx,
                                              alphaDgrid,
                                              dyDesc,
                                              dy,
                                              grid,
                                              betaDgrid,
                                              dgrid);
  }
  const auto tic = now();
  const auto res = orig_cudnnSpatialTfSamplerBackward(handle,
                                                      stDesc,
                                                      alpha,
                                                      xDesc,
                                                      x,
                                                      beta,
                                                      dxDesc,
                                                      dx,
                                                      alphaDgrid,
                                                      dyDesc,
                                                      dy,
                                                      grid,
                                                      betaDgrid,
                                                      dgrid);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 111
cudnnStatus_t CUDNNWINAPI
    cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc) {
  using fun_t = decltype(cudnnCreateDropoutDescriptor);
  static const std::string funName{"cudnnCreateDropoutDescriptor"};
  static fun_t *orig_cudnnCreateDropoutDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCreateDropoutDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreateDropoutDescriptor(dropoutDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreateDropoutDescriptor(dropoutDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 112
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) {
  using fun_t = decltype(cudnnDestroyDropoutDescriptor);
  static const std::string funName{"cudnnDestroyDropoutDescriptor"};
  static fun_t *orig_cudnnDestroyDropoutDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroyDropoutDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroyDropoutDescriptor(dropoutDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroyDropoutDescriptor(dropoutDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 113
cudnnStatus_t CUDNNWINAPI cudnnDropoutGetStatesSize(cudnnHandle_t handle,
                                                    size_t *sizeInBytes) {
  using fun_t = decltype(cudnnDropoutGetStatesSize);
  static const std::string funName{"cudnnDropoutGetStatesSize"};
  static fun_t *orig_cudnnDropoutGetStatesSize =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDropoutGetStatesSize");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDropoutGetStatesSize(handle, sizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnDropoutGetStatesSize(handle, sizeInBytes);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 114
cudnnStatus_t CUDNNWINAPI cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc,
                                                          size_t *sizeInBytes) {
  using fun_t = decltype(cudnnDropoutGetReserveSpaceSize);
  static const std::string funName{"cudnnDropoutGetReserveSpaceSize"};
  static fun_t *orig_cudnnDropoutGetReserveSpaceSize =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDropoutGetReserveSpaceSize");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDropoutGetReserveSpaceSize(xdesc, sizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnDropoutGetReserveSpaceSize(xdesc, sizeInBytes);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 115
cudnnStatus_t CUDNNWINAPI cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                                    cudnnHandle_t handle,
                                                    float dropout,
                                                    void *states,
                                                    size_t stateSizeInBytes,
                                                    unsigned long long seed) {
  using fun_t = decltype(cudnnSetDropoutDescriptor);
  static const std::string funName{"cudnnSetDropoutDescriptor"};
  static fun_t *orig_cudnnSetDropoutDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetDropoutDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetDropoutDescriptor(
        dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetDropoutDescriptor(
      dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"stateSizeInBytes", stateSizeInBytes},
                                               {"seed", seed}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 116
cudnnStatus_t CUDNNWINAPI
    cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                  cudnnHandle_t handle,
                                  float dropout,
                                  void *states,
                                  size_t stateSizeInBytes,
                                  unsigned long long seed) {
  using fun_t = decltype(cudnnRestoreDropoutDescriptor);
  static const std::string funName{"cudnnRestoreDropoutDescriptor"};
  static fun_t *orig_cudnnRestoreDropoutDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnRestoreDropoutDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnRestoreDropoutDescriptor(
        dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
  }
  const auto tic = now();
  const auto res = orig_cudnnRestoreDropoutDescriptor(
      dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"stateSizeInBytes", stateSizeInBytes},
                                               {"seed", seed}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 117
cudnnStatus_t CUDNNWINAPI cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                                    cudnnHandle_t handle,
                                                    float *dropout,
                                                    void **states,
                                                    unsigned long long *seed) {
  using fun_t = decltype(cudnnGetDropoutDescriptor);
  static const std::string funName{"cudnnGetDropoutDescriptor"};
  static fun_t *orig_cudnnGetDropoutDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetDropoutDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetDropoutDescriptor(dropoutDesc, handle, dropout, states, seed);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnGetDropoutDescriptor(dropoutDesc, handle, dropout, states, seed);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 118
cudnnStatus_t CUDNNWINAPI cudnnDropoutForward(cudnnHandle_t handle,
                                              const cudnnDropoutDescriptor_t dropoutDesc,
                                              const cudnnTensorDescriptor_t xdesc,
                                              const void *x,
                                              const cudnnTensorDescriptor_t ydesc,
                                              void *y,
                                              void *reserveSpace,
                                              size_t reserveSpaceSizeInBytes) {
  using fun_t = decltype(cudnnDropoutForward);
  static const std::string funName{"cudnnDropoutForward"};
  static fun_t *orig_cudnnDropoutForward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDropoutForward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDropoutForward(
        handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnDropoutForward(
      handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"reserveSpaceSizeInBytes",
                                                reserveSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 119
cudnnStatus_t CUDNNWINAPI cudnnDropoutBackward(cudnnHandle_t handle,
                                               const cudnnDropoutDescriptor_t dropoutDesc,
                                               const cudnnTensorDescriptor_t dydesc,
                                               const void *dy,
                                               const cudnnTensorDescriptor_t dxdesc,
                                               void *dx,
                                               void *reserveSpace,
                                               size_t reserveSpaceSizeInBytes) {
  using fun_t = decltype(cudnnDropoutBackward);
  static const std::string funName{"cudnnDropoutBackward"};
  static fun_t *orig_cudnnDropoutBackward =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDropoutBackward");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDropoutBackward(handle,
                                     dropoutDesc,
                                     dydesc,
                                     dy,
                                     dxdesc,
                                     dx,
                                     reserveSpace,
                                     reserveSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnDropoutBackward(
      handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"reserveSpaceSizeInBytes",
                                                reserveSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 120
cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc) {
  using fun_t = decltype(cudnnCreateRNNDescriptor);
  static const std::string funName{"cudnnCreateRNNDescriptor"};
  static fun_t *orig_cudnnCreateRNNDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCreateRNNDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreateRNNDescriptor(rnnDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreateRNNDescriptor(rnnDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 121
cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
  using fun_t = decltype(cudnnDestroyRNNDescriptor);
  static const std::string funName{"cudnnDestroyRNNDescriptor"};
  static fun_t *orig_cudnnDestroyRNNDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroyRNNDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroyRNNDescriptor(rnnDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroyRNNDescriptor(rnnDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 122
cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardInferenceAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  using fun_t = decltype(cudnnGetRNNForwardInferenceAlgorithmMaxCount);
  static const std::string funName{"cudnnGetRNNForwardInferenceAlgorithmMaxCount"};
  static fun_t *orig_cudnnGetRNNForwardInferenceAlgorithmMaxCount =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetRNNForwardInferenceAlgorithmMaxCount");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetRNNForwardInferenceAlgorithmMaxCount(handle, rnnDesc, count);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNForwardInferenceAlgorithmMaxCount(handle, rnnDesc, count);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 123
cudnnStatus_t CUDNNWINAPI
    cudnnFindRNNForwardInferenceAlgorithmEx(cudnnHandle_t handle,
                                            const cudnnRNNDescriptor_t rnnDesc,
                                            const int seqLength,
                                            const cudnnTensorDescriptor_t *xDesc,
                                            const void *x,
                                            const cudnnTensorDescriptor_t hxDesc,
                                            const void *hx,
                                            const cudnnTensorDescriptor_t cxDesc,
                                            const void *cx,
                                            const cudnnFilterDescriptor_t wDesc,
                                            const void *w,
                                            const cudnnTensorDescriptor_t *yDesc,
                                            void *y,
                                            const cudnnTensorDescriptor_t hyDesc,
                                            void *hy,
                                            const cudnnTensorDescriptor_t cyDesc,
                                            void *cy,
                                            const float findIntensity,
                                            const int requestedAlgoCount,
                                            int *returnedAlgoCount,
                                            cudnnAlgorithmPerformance_t *perfResults,
                                            void *workspace,
                                            size_t workSpaceSizeInBytes) {
  using fun_t = decltype(cudnnFindRNNForwardInferenceAlgorithmEx);
  static const std::string funName{"cudnnFindRNNForwardInferenceAlgorithmEx"};
  static fun_t *orig_cudnnFindRNNForwardInferenceAlgorithmEx =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnFindRNNForwardInferenceAlgorithmEx");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnFindRNNForwardInferenceAlgorithmEx(handle,
                                                        rnnDesc,
                                                        seqLength,
                                                        xDesc,
                                                        x,
                                                        hxDesc,
                                                        hx,
                                                        cxDesc,
                                                        cx,
                                                        wDesc,
                                                        w,
                                                        yDesc,
                                                        y,
                                                        hyDesc,
                                                        hy,
                                                        cyDesc,
                                                        cy,
                                                        findIntensity,
                                                        requestedAlgoCount,
                                                        returnedAlgoCount,
                                                        perfResults,
                                                        workspace,
                                                        workSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnFindRNNForwardInferenceAlgorithmEx(handle,
                                                                rnnDesc,
                                                                seqLength,
                                                                xDesc,
                                                                x,
                                                                hxDesc,
                                                                hx,
                                                                cxDesc,
                                                                cx,
                                                                wDesc,
                                                                w,
                                                                yDesc,
                                                                y,
                                                                hyDesc,
                                                                hy,
                                                                cyDesc,
                                                                cy,
                                                                findIntensity,
                                                                requestedAlgoCount,
                                                                returnedAlgoCount,
                                                                perfResults,
                                                                workspace,
                                                                workSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"seqLength", seqLength},
                                  {"findIntensity", findIntensity},
                                  {"requestedAlgoCount", requestedAlgoCount},
                                  {"workSpaceSizeInBytes", workSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 124
cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardTrainingAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  using fun_t = decltype(cudnnGetRNNForwardTrainingAlgorithmMaxCount);
  static const std::string funName{"cudnnGetRNNForwardTrainingAlgorithmMaxCount"};
  static fun_t *orig_cudnnGetRNNForwardTrainingAlgorithmMaxCount =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetRNNForwardTrainingAlgorithmMaxCount");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetRNNForwardTrainingAlgorithmMaxCount(handle, rnnDesc, count);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNForwardTrainingAlgorithmMaxCount(handle, rnnDesc, count);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 125
cudnnStatus_t CUDNNWINAPI
    cudnnFindRNNForwardTrainingAlgorithmEx(cudnnHandle_t handle,
                                           const cudnnRNNDescriptor_t rnnDesc,
                                           const int seqLength,
                                           const cudnnTensorDescriptor_t *xDesc,
                                           const void *x,
                                           const cudnnTensorDescriptor_t hxDesc,
                                           const void *hx,
                                           const cudnnTensorDescriptor_t cxDesc,
                                           const void *cx,
                                           const cudnnFilterDescriptor_t wDesc,
                                           const void *w,
                                           const cudnnTensorDescriptor_t *yDesc,
                                           void *y,
                                           const cudnnTensorDescriptor_t hyDesc,
                                           void *hy,
                                           const cudnnTensorDescriptor_t cyDesc,
                                           void *cy,
                                           const float findIntensity,
                                           const int requestedAlgoCount,
                                           int *returnedAlgoCount,
                                           cudnnAlgorithmPerformance_t *perfResults,
                                           void *workspace,
                                           size_t workSpaceSizeInBytes,
                                           void *reserveSpace,
                                           size_t reserveSpaceSizeInBytes) {
  using fun_t = decltype(cudnnFindRNNForwardTrainingAlgorithmEx);
  static const std::string funName{"cudnnFindRNNForwardTrainingAlgorithmEx"};
  static fun_t *orig_cudnnFindRNNForwardTrainingAlgorithmEx =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnFindRNNForwardTrainingAlgorithmEx");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnFindRNNForwardTrainingAlgorithmEx(handle,
                                                       rnnDesc,
                                                       seqLength,
                                                       xDesc,
                                                       x,
                                                       hxDesc,
                                                       hx,
                                                       cxDesc,
                                                       cx,
                                                       wDesc,
                                                       w,
                                                       yDesc,
                                                       y,
                                                       hyDesc,
                                                       hy,
                                                       cyDesc,
                                                       cy,
                                                       findIntensity,
                                                       requestedAlgoCount,
                                                       returnedAlgoCount,
                                                       perfResults,
                                                       workspace,
                                                       workSpaceSizeInBytes,
                                                       reserveSpace,
                                                       reserveSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnFindRNNForwardTrainingAlgorithmEx(handle,
                                                               rnnDesc,
                                                               seqLength,
                                                               xDesc,
                                                               x,
                                                               hxDesc,
                                                               hx,
                                                               cxDesc,
                                                               cx,
                                                               wDesc,
                                                               w,
                                                               yDesc,
                                                               y,
                                                               hyDesc,
                                                               hy,
                                                               cyDesc,
                                                               cy,
                                                               findIntensity,
                                                               requestedAlgoCount,
                                                               returnedAlgoCount,
                                                               perfResults,
                                                               workspace,
                                                               workSpaceSizeInBytes,
                                                               reserveSpace,
                                                               reserveSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments",
       json::object({{"seqLength", seqLength},
                     {"findIntensity", findIntensity},
                     {"requestedAlgoCount", requestedAlgoCount},
                     {"workSpaceSizeInBytes", workSpaceSizeInBytes},
                     {"reserveSpaceSizeInBytes", reserveSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 126
cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardDataAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  using fun_t = decltype(cudnnGetRNNBackwardDataAlgorithmMaxCount);
  static const std::string funName{"cudnnGetRNNBackwardDataAlgorithmMaxCount"};
  static fun_t *orig_cudnnGetRNNBackwardDataAlgorithmMaxCount =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetRNNBackwardDataAlgorithmMaxCount");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetRNNBackwardDataAlgorithmMaxCount(handle, rnnDesc, count);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetRNNBackwardDataAlgorithmMaxCount(handle, rnnDesc, count);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 127
cudnnStatus_t CUDNNWINAPI
    cudnnFindRNNBackwardDataAlgorithmEx(cudnnHandle_t handle,
                                        const cudnnRNNDescriptor_t rnnDesc,
                                        const int seqLength,
                                        const cudnnTensorDescriptor_t *yDesc,
                                        const void *y,
                                        const cudnnTensorDescriptor_t *dyDesc,
                                        const void *dy,
                                        const cudnnTensorDescriptor_t dhyDesc,
                                        const void *dhy,
                                        const cudnnTensorDescriptor_t dcyDesc,
                                        const void *dcy,
                                        const cudnnFilterDescriptor_t wDesc,
                                        const void *w,
                                        const cudnnTensorDescriptor_t hxDesc,
                                        const void *hx,
                                        const cudnnTensorDescriptor_t cxDesc,
                                        const void *cx,
                                        const cudnnTensorDescriptor_t *dxDesc,
                                        void *dx,
                                        const cudnnTensorDescriptor_t dhxDesc,
                                        void *dhx,
                                        const cudnnTensorDescriptor_t dcxDesc,
                                        void *dcx,
                                        const float findIntensity,
                                        const int requestedAlgoCount,
                                        int *returnedAlgoCount,
                                        cudnnAlgorithmPerformance_t *perfResults,
                                        void *workspace,
                                        size_t workSpaceSizeInBytes,
                                        void *reserveSpace,
                                        size_t reserveSpaceSizeInBytes) {
  using fun_t = decltype(cudnnFindRNNBackwardDataAlgorithmEx);
  static const std::string funName{"cudnnFindRNNBackwardDataAlgorithmEx"};
  static fun_t *orig_cudnnFindRNNBackwardDataAlgorithmEx =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnFindRNNBackwardDataAlgorithmEx");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnFindRNNBackwardDataAlgorithmEx(handle,
                                                    rnnDesc,
                                                    seqLength,
                                                    yDesc,
                                                    y,
                                                    dyDesc,
                                                    dy,
                                                    dhyDesc,
                                                    dhy,
                                                    dcyDesc,
                                                    dcy,
                                                    wDesc,
                                                    w,
                                                    hxDesc,
                                                    hx,
                                                    cxDesc,
                                                    cx,
                                                    dxDesc,
                                                    dx,
                                                    dhxDesc,
                                                    dhx,
                                                    dcxDesc,
                                                    dcx,
                                                    findIntensity,
                                                    requestedAlgoCount,
                                                    returnedAlgoCount,
                                                    perfResults,
                                                    workspace,
                                                    workSpaceSizeInBytes,
                                                    reserveSpace,
                                                    reserveSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnFindRNNBackwardDataAlgorithmEx(handle,
                                                            rnnDesc,
                                                            seqLength,
                                                            yDesc,
                                                            y,
                                                            dyDesc,
                                                            dy,
                                                            dhyDesc,
                                                            dhy,
                                                            dcyDesc,
                                                            dcy,
                                                            wDesc,
                                                            w,
                                                            hxDesc,
                                                            hx,
                                                            cxDesc,
                                                            cx,
                                                            dxDesc,
                                                            dx,
                                                            dhxDesc,
                                                            dhx,
                                                            dcxDesc,
                                                            dcx,
                                                            findIntensity,
                                                            requestedAlgoCount,
                                                            returnedAlgoCount,
                                                            perfResults,
                                                            workspace,
                                                            workSpaceSizeInBytes,
                                                            reserveSpace,
                                                            reserveSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments",
       json::object({{"seqLength", seqLength},
                     {"findIntensity", findIntensity},
                     {"requestedAlgoCount", requestedAlgoCount},
                     {"workSpaceSizeInBytes", workSpaceSizeInBytes},
                     {"reserveSpaceSizeInBytes", reserveSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 128
cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardWeightsAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  using fun_t = decltype(cudnnGetRNNBackwardWeightsAlgorithmMaxCount);
  static const std::string funName{"cudnnGetRNNBackwardWeightsAlgorithmMaxCount"};
  static fun_t *orig_cudnnGetRNNBackwardWeightsAlgorithmMaxCount =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetRNNBackwardWeightsAlgorithmMaxCount");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle, rnnDesc, count);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle, rnnDesc, count);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 129
cudnnStatus_t CUDNNWINAPI
    cudnnFindRNNBackwardWeightsAlgorithmEx(cudnnHandle_t handle,
                                           const cudnnRNNDescriptor_t rnnDesc,
                                           const int seqLength,
                                           const cudnnTensorDescriptor_t *xDesc,
                                           const void *x,
                                           const cudnnTensorDescriptor_t hxDesc,
                                           const void *hx,
                                           const cudnnTensorDescriptor_t *yDesc,
                                           const void *y,
                                           const float findIntensity,
                                           const int requestedAlgoCount,
                                           int *returnedAlgoCount,
                                           cudnnAlgorithmPerformance_t *perfResults,
                                           const void *workspace,
                                           size_t workSpaceSizeInBytes,
                                           const cudnnFilterDescriptor_t dwDesc,
                                           void *dw,
                                           const void *reserveSpace,
                                           size_t reserveSpaceSizeInBytes) {
  using fun_t = decltype(cudnnFindRNNBackwardWeightsAlgorithmEx);
  static const std::string funName{"cudnnFindRNNBackwardWeightsAlgorithmEx"};
  static fun_t *orig_cudnnFindRNNBackwardWeightsAlgorithmEx =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnFindRNNBackwardWeightsAlgorithmEx");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnFindRNNBackwardWeightsAlgorithmEx(handle,
                                                       rnnDesc,
                                                       seqLength,
                                                       xDesc,
                                                       x,
                                                       hxDesc,
                                                       hx,
                                                       yDesc,
                                                       y,
                                                       findIntensity,
                                                       requestedAlgoCount,
                                                       returnedAlgoCount,
                                                       perfResults,
                                                       workspace,
                                                       workSpaceSizeInBytes,
                                                       dwDesc,
                                                       dw,
                                                       reserveSpace,
                                                       reserveSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnFindRNNBackwardWeightsAlgorithmEx(handle,
                                                               rnnDesc,
                                                               seqLength,
                                                               xDesc,
                                                               x,
                                                               hxDesc,
                                                               hx,
                                                               yDesc,
                                                               y,
                                                               findIntensity,
                                                               requestedAlgoCount,
                                                               returnedAlgoCount,
                                                               perfResults,
                                                               workspace,
                                                               workSpaceSizeInBytes,
                                                               dwDesc,
                                                               dw,
                                                               reserveSpace,
                                                               reserveSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments",
       json::object({{"seqLength", seqLength},
                     {"findIntensity", findIntensity},
                     {"requestedAlgoCount", requestedAlgoCount},
                     {"workSpaceSizeInBytes", workSpaceSizeInBytes},
                     {"reserveSpaceSizeInBytes", reserveSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 130
cudnnStatus_t CUDNNWINAPI cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
                                                       const int minibatch,
                                                       const cudnnDataType_t dataType,
                                                       cudnnPersistentRNNPlan_t *plan) {
  using fun_t = decltype(cudnnCreatePersistentRNNPlan);
  static const std::string funName{"cudnnCreatePersistentRNNPlan"};
  static fun_t *orig_cudnnCreatePersistentRNNPlan =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCreatePersistentRNNPlan");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreatePersistentRNNPlan(rnnDesc, minibatch, dataType, plan);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreatePersistentRNNPlan(rnnDesc, minibatch, dataType, plan);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"minibatch", minibatch}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 131
cudnnStatus_t CUDNNWINAPI cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
                                                    cudnnPersistentRNNPlan_t plan) {
  using fun_t = decltype(cudnnSetPersistentRNNPlan);
  static const std::string funName{"cudnnSetPersistentRNNPlan"};
  static fun_t *orig_cudnnSetPersistentRNNPlan =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetPersistentRNNPlan");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetPersistentRNNPlan(rnnDesc, plan);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetPersistentRNNPlan(rnnDesc, plan);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 132
cudnnStatus_t CUDNNWINAPI cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan) {
  using fun_t = decltype(cudnnDestroyPersistentRNNPlan);
  static const std::string funName{"cudnnDestroyPersistentRNNPlan"};
  static fun_t *orig_cudnnDestroyPersistentRNNPlan =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroyPersistentRNNPlan");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroyPersistentRNNPlan(plan);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroyPersistentRNNPlan(plan);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 133
cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    const int hiddenSize,
    const int numLayers,
    cudnnDropoutDescriptor_t
        dropoutDesc, /* Between layers, not between recurrent steps. */
    cudnnRNNInputMode_t inputMode,
    cudnnDirectionMode_t direction,
    cudnnRNNMode_t mode,
    cudnnRNNAlgo_t algo,
    cudnnDataType_t dataType) {
  using fun_t = decltype(cudnnSetRNNDescriptor);
  static const std::string funName{"cudnnSetRNNDescriptor"};
  static fun_t *orig_cudnnSetRNNDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetRNNDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetRNNDescriptor(handle,
                                      rnnDesc,
                                      hiddenSize,
                                      numLayers,
                                      dropoutDesc,
                                      inputMode,
                                      direction,
                                      mode,
                                      algo,
                                      dataType);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetRNNDescriptor(handle,
                                              rnnDesc,
                                              hiddenSize,
                                              numLayers,
                                              dropoutDesc,
                                              inputMode,
                                              direction,
                                              mode,
                                              algo,
                                              dataType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"hiddenSize", hiddenSize},
                                               {"numLayers", numLayers}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 134
cudnnStatus_t CUDNNWINAPI cudnnSetRNNProjectionLayers(cudnnHandle_t handle,
                                                      cudnnRNNDescriptor_t rnnDesc,
                                                      const int recProjSize,
                                                      const int outProjSize) {
  using fun_t = decltype(cudnnSetRNNProjectionLayers);
  static const std::string funName{"cudnnSetRNNProjectionLayers"};
  static fun_t *orig_cudnnSetRNNProjectionLayers =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetRNNProjectionLayers");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnSetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"recProjSize", recProjSize},
                                               {"outProjSize", outProjSize}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 135
cudnnStatus_t CUDNNWINAPI cudnnGetRNNProjectionLayers(cudnnHandle_t handle,
                                                      const cudnnRNNDescriptor_t rnnDesc,
                                                      int *recProjSize,
                                                      int *outProjSize) {
  using fun_t = decltype(cudnnGetRNNProjectionLayers);
  static const std::string funName{"cudnnGetRNNProjectionLayers"};
  static fun_t *orig_cudnnGetRNNProjectionLayers =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetRNNProjectionLayers");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 136
cudnnStatus_t CUDNNWINAPI
    cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t handle,
                                   cudnnRNNDescriptor_t rnnDesc,
                                   cudnnAlgorithmDescriptor_t algoDesc) {
  using fun_t = decltype(cudnnSetRNNAlgorithmDescriptor);
  static const std::string funName{"cudnnSetRNNAlgorithmDescriptor"};
  static fun_t *orig_cudnnSetRNNAlgorithmDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetRNNAlgorithmDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetRNNAlgorithmDescriptor(handle, rnnDesc, algoDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetRNNAlgorithmDescriptor(handle, rnnDesc, algoDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 137
cudnnStatus_t CUDNNWINAPI cudnnGetRNNDescriptor(cudnnHandle_t handle,
                                                cudnnRNNDescriptor_t rnnDesc,
                                                int *hiddenSize,
                                                int *numLayers,
                                                cudnnDropoutDescriptor_t *dropoutDesc,
                                                cudnnRNNInputMode_t *inputMode,
                                                cudnnDirectionMode_t *direction,
                                                cudnnRNNMode_t *mode,
                                                cudnnRNNAlgo_t *algo,
                                                cudnnDataType_t *dataType) {
  using fun_t = decltype(cudnnGetRNNDescriptor);
  static const std::string funName{"cudnnGetRNNDescriptor"};
  static fun_t *orig_cudnnGetRNNDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetRNNDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetRNNDescriptor(handle,
                                      rnnDesc,
                                      hiddenSize,
                                      numLayers,
                                      dropoutDesc,
                                      inputMode,
                                      direction,
                                      mode,
                                      algo,
                                      dataType);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetRNNDescriptor(handle,
                                              rnnDesc,
                                              hiddenSize,
                                              numLayers,
                                              dropoutDesc,
                                              inputMode,
                                              direction,
                                              mode,
                                              algo,
                                              dataType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 138
cudnnStatus_t CUDNNWINAPI cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc,
                                                    cudnnMathType_t mType) {
  using fun_t = decltype(cudnnSetRNNMatrixMathType);
  static const std::string funName{"cudnnSetRNNMatrixMathType"};
  static fun_t *orig_cudnnSetRNNMatrixMathType =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetRNNMatrixMathType");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetRNNMatrixMathType(rnnDesc, mType);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetRNNMatrixMathType(rnnDesc, mType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 139
cudnnStatus_t CUDNNWINAPI cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc,
                                                    cudnnMathType_t *mType) {
  using fun_t = decltype(cudnnGetRNNMatrixMathType);
  static const std::string funName{"cudnnGetRNNMatrixMathType"};
  static fun_t *orig_cudnnGetRNNMatrixMathType =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetRNNMatrixMathType");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetRNNMatrixMathType(rnnDesc, mType);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetRNNMatrixMathType(rnnDesc, mType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 140
cudnnStatus_t CUDNNWINAPI cudnnGetRNNWorkspaceSize(cudnnHandle_t handle,
                                                   const cudnnRNNDescriptor_t rnnDesc,
                                                   const int seqLength,
                                                   const cudnnTensorDescriptor_t *xDesc,
                                                   size_t *sizeInBytes) {
  using fun_t = decltype(cudnnGetRNNWorkspaceSize);
  static const std::string funName{"cudnnGetRNNWorkspaceSize"};
  static fun_t *orig_cudnnGetRNNWorkspaceSize =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetRNNWorkspaceSize");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"seqLength", seqLength}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 141
cudnnStatus_t CUDNNWINAPI
    cudnnGetRNNTrainingReserveSize(cudnnHandle_t handle,
                                   const cudnnRNNDescriptor_t rnnDesc,
                                   const int seqLength,
                                   const cudnnTensorDescriptor_t *xDesc,
                                   size_t *sizeInBytes) {
  using fun_t = decltype(cudnnGetRNNTrainingReserveSize);
  static const std::string funName{"cudnnGetRNNTrainingReserveSize"};
  static fun_t *orig_cudnnGetRNNTrainingReserveSize =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetRNNTrainingReserveSize");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetRNNTrainingReserveSize(
        handle, rnnDesc, seqLength, xDesc, sizeInBytes);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNTrainingReserveSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"seqLength", seqLength}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 142
cudnnStatus_t CUDNNWINAPI cudnnGetRNNParamsSize(cudnnHandle_t handle,
                                                const cudnnRNNDescriptor_t rnnDesc,
                                                const cudnnTensorDescriptor_t xDesc,
                                                size_t *sizeInBytes,
                                                cudnnDataType_t dataType) {
  using fun_t = decltype(cudnnGetRNNParamsSize);
  static const std::string funName{"cudnnGetRNNParamsSize"};
  static fun_t *orig_cudnnGetRNNParamsSize =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetRNNParamsSize");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetRNNParamsSize(handle, rnnDesc, xDesc, sizeInBytes, dataType);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNParamsSize(handle, rnnDesc, xDesc, sizeInBytes, dataType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 143
cudnnStatus_t CUDNNWINAPI
    cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t handle,
                                    const cudnnRNNDescriptor_t rnnDesc,
                                    const int pseudoLayer,
                                    const cudnnTensorDescriptor_t xDesc,
                                    const cudnnFilterDescriptor_t wDesc,
                                    const void *w,
                                    const int linLayerID,
                                    cudnnFilterDescriptor_t linLayerMatDesc,
                                    void **linLayerMat) {
  using fun_t = decltype(cudnnGetRNNLinLayerMatrixParams);
  static const std::string funName{"cudnnGetRNNLinLayerMatrixParams"};
  static fun_t *orig_cudnnGetRNNLinLayerMatrixParams =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetRNNLinLayerMatrixParams");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetRNNLinLayerMatrixParams(handle,
                                                rnnDesc,
                                                pseudoLayer,
                                                xDesc,
                                                wDesc,
                                                w,
                                                linLayerID,
                                                linLayerMatDesc,
                                                linLayerMat);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetRNNLinLayerMatrixParams(handle,
                                                        rnnDesc,
                                                        pseudoLayer,
                                                        xDesc,
                                                        wDesc,
                                                        w,
                                                        linLayerID,
                                                        linLayerMatDesc,
                                                        linLayerMat);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"pseudoLayer", pseudoLayer},
                                               {"linLayerID", linLayerID}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 144
cudnnStatus_t CUDNNWINAPI
    cudnnGetRNNLinLayerBiasParams(cudnnHandle_t handle,
                                  const cudnnRNNDescriptor_t rnnDesc,
                                  const int pseudoLayer,
                                  const cudnnTensorDescriptor_t xDesc,
                                  const cudnnFilterDescriptor_t wDesc,
                                  const void *w,
                                  const int linLayerID,
                                  cudnnFilterDescriptor_t linLayerBiasDesc,
                                  void **linLayerBias) {
  using fun_t = decltype(cudnnGetRNNLinLayerBiasParams);
  static const std::string funName{"cudnnGetRNNLinLayerBiasParams"};
  static fun_t *orig_cudnnGetRNNLinLayerBiasParams =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetRNNLinLayerBiasParams");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetRNNLinLayerBiasParams(handle,
                                              rnnDesc,
                                              pseudoLayer,
                                              xDesc,
                                              wDesc,
                                              w,
                                              linLayerID,
                                              linLayerBiasDesc,
                                              linLayerBias);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetRNNLinLayerBiasParams(handle,
                                                      rnnDesc,
                                                      pseudoLayer,
                                                      xDesc,
                                                      wDesc,
                                                      w,
                                                      linLayerID,
                                                      linLayerBiasDesc,
                                                      linLayerBias);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"pseudoLayer", pseudoLayer},
                                               {"linLayerID", linLayerID}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 145
cudnnStatus_t CUDNNWINAPI cudnnRNNForwardInference(cudnnHandle_t handle,
                                                   const cudnnRNNDescriptor_t rnnDesc,
                                                   const int seqLength,
                                                   const cudnnTensorDescriptor_t *xDesc,
                                                   const void *x,
                                                   const cudnnTensorDescriptor_t hxDesc,
                                                   const void *hx,
                                                   const cudnnTensorDescriptor_t cxDesc,
                                                   const void *cx,
                                                   const cudnnFilterDescriptor_t wDesc,
                                                   const void *w,
                                                   const cudnnTensorDescriptor_t *yDesc,
                                                   void *y,
                                                   const cudnnTensorDescriptor_t hyDesc,
                                                   void *hy,
                                                   const cudnnTensorDescriptor_t cyDesc,
                                                   void *cy,
                                                   void *workspace,
                                                   size_t workSpaceSizeInBytes) {
  using fun_t = decltype(cudnnRNNForwardInference);
  static const std::string funName{"cudnnRNNForwardInference"};
  static fun_t *orig_cudnnRNNForwardInference =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnRNNForwardInference");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnRNNForwardInference(handle,
                                         rnnDesc,
                                         seqLength,
                                         xDesc,
                                         x,
                                         hxDesc,
                                         hx,
                                         cxDesc,
                                         cx,
                                         wDesc,
                                         w,
                                         yDesc,
                                         y,
                                         hyDesc,
                                         hy,
                                         cyDesc,
                                         cy,
                                         workspace,
                                         workSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnRNNForwardInference(handle,
                                                 rnnDesc,
                                                 seqLength,
                                                 xDesc,
                                                 x,
                                                 hxDesc,
                                                 hx,
                                                 cxDesc,
                                                 cx,
                                                 wDesc,
                                                 w,
                                                 yDesc,
                                                 y,
                                                 hyDesc,
                                                 hy,
                                                 cyDesc,
                                                 cy,
                                                 workspace,
                                                 workSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"seqLength", seqLength},
                                  {"workSpaceSizeInBytes", workSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 146
cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTraining(cudnnHandle_t handle,
                                                  const cudnnRNNDescriptor_t rnnDesc,
                                                  const int seqLength,
                                                  const cudnnTensorDescriptor_t *xDesc,
                                                  const void *x,
                                                  const cudnnTensorDescriptor_t hxDesc,
                                                  const void *hx,
                                                  const cudnnTensorDescriptor_t cxDesc,
                                                  const void *cx,
                                                  const cudnnFilterDescriptor_t wDesc,
                                                  const void *w,
                                                  const cudnnTensorDescriptor_t *yDesc,
                                                  void *y,
                                                  const cudnnTensorDescriptor_t hyDesc,
                                                  void *hy,
                                                  const cudnnTensorDescriptor_t cyDesc,
                                                  void *cy,
                                                  void *workspace,
                                                  size_t workSpaceSizeInBytes,
                                                  void *reserveSpace,
                                                  size_t reserveSpaceSizeInBytes) {
  using fun_t = decltype(cudnnRNNForwardTraining);
  static const std::string funName{"cudnnRNNForwardTraining"};
  static fun_t *orig_cudnnRNNForwardTraining =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnRNNForwardTraining");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnRNNForwardTraining(handle,
                                        rnnDesc,
                                        seqLength,
                                        xDesc,
                                        x,
                                        hxDesc,
                                        hx,
                                        cxDesc,
                                        cx,
                                        wDesc,
                                        w,
                                        yDesc,
                                        y,
                                        hyDesc,
                                        hy,
                                        cyDesc,
                                        cy,
                                        workspace,
                                        workSpaceSizeInBytes,
                                        reserveSpace,
                                        reserveSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnRNNForwardTraining(handle,
                                                rnnDesc,
                                                seqLength,
                                                xDesc,
                                                x,
                                                hxDesc,
                                                hx,
                                                cxDesc,
                                                cx,
                                                wDesc,
                                                w,
                                                yDesc,
                                                y,
                                                hyDesc,
                                                hy,
                                                cyDesc,
                                                cy,
                                                workspace,
                                                workSpaceSizeInBytes,
                                                reserveSpace,
                                                reserveSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments",
       json::object({{"seqLength", seqLength},
                     {"workSpaceSizeInBytes", workSpaceSizeInBytes},
                     {"reserveSpaceSizeInBytes", reserveSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 147
cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardData(cudnnHandle_t handle,
                                               const cudnnRNNDescriptor_t rnnDesc,
                                               const int seqLength,
                                               const cudnnTensorDescriptor_t *yDesc,
                                               const void *y,
                                               const cudnnTensorDescriptor_t *dyDesc,
                                               const void *dy,
                                               const cudnnTensorDescriptor_t dhyDesc,
                                               const void *dhy,
                                               const cudnnTensorDescriptor_t dcyDesc,
                                               const void *dcy,
                                               const cudnnFilterDescriptor_t wDesc,
                                               const void *w,
                                               const cudnnTensorDescriptor_t hxDesc,
                                               const void *hx,
                                               const cudnnTensorDescriptor_t cxDesc,
                                               const void *cx,
                                               const cudnnTensorDescriptor_t *dxDesc,
                                               void *dx,
                                               const cudnnTensorDescriptor_t dhxDesc,
                                               void *dhx,
                                               const cudnnTensorDescriptor_t dcxDesc,
                                               void *dcx,
                                               void *workspace,
                                               size_t workSpaceSizeInBytes,
                                               void *reserveSpace,
                                               size_t reserveSpaceSizeInBytes) {
  using fun_t = decltype(cudnnRNNBackwardData);
  static const std::string funName{"cudnnRNNBackwardData"};
  static fun_t *orig_cudnnRNNBackwardData =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnRNNBackwardData");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnRNNBackwardData(handle,
                                     rnnDesc,
                                     seqLength,
                                     yDesc,
                                     y,
                                     dyDesc,
                                     dy,
                                     dhyDesc,
                                     dhy,
                                     dcyDesc,
                                     dcy,
                                     wDesc,
                                     w,
                                     hxDesc,
                                     hx,
                                     cxDesc,
                                     cx,
                                     dxDesc,
                                     dx,
                                     dhxDesc,
                                     dhx,
                                     dcxDesc,
                                     dcx,
                                     workspace,
                                     workSpaceSizeInBytes,
                                     reserveSpace,
                                     reserveSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnRNNBackwardData(handle,
                                             rnnDesc,
                                             seqLength,
                                             yDesc,
                                             y,
                                             dyDesc,
                                             dy,
                                             dhyDesc,
                                             dhy,
                                             dcyDesc,
                                             dcy,
                                             wDesc,
                                             w,
                                             hxDesc,
                                             hx,
                                             cxDesc,
                                             cx,
                                             dxDesc,
                                             dx,
                                             dhxDesc,
                                             dhx,
                                             dcxDesc,
                                             dcx,
                                             workspace,
                                             workSpaceSizeInBytes,
                                             reserveSpace,
                                             reserveSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments",
       json::object({{"seqLength", seqLength},
                     {"workSpaceSizeInBytes", workSpaceSizeInBytes},
                     {"reserveSpaceSizeInBytes", reserveSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 148
cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights(cudnnHandle_t handle,
                                                  const cudnnRNNDescriptor_t rnnDesc,
                                                  const int seqLength,
                                                  const cudnnTensorDescriptor_t *xDesc,
                                                  const void *x,
                                                  const cudnnTensorDescriptor_t hxDesc,
                                                  const void *hx,
                                                  const cudnnTensorDescriptor_t *yDesc,
                                                  const void *y,
                                                  const void *workspace,
                                                  size_t workSpaceSizeInBytes,
                                                  const cudnnFilterDescriptor_t dwDesc,
                                                  void *dw,
                                                  const void *reserveSpace,
                                                  size_t reserveSpaceSizeInBytes) {
  using fun_t = decltype(cudnnRNNBackwardWeights);
  static const std::string funName{"cudnnRNNBackwardWeights"};
  static fun_t *orig_cudnnRNNBackwardWeights =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnRNNBackwardWeights");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnRNNBackwardWeights(handle,
                                        rnnDesc,
                                        seqLength,
                                        xDesc,
                                        x,
                                        hxDesc,
                                        hx,
                                        yDesc,
                                        y,
                                        workspace,
                                        workSpaceSizeInBytes,
                                        dwDesc,
                                        dw,
                                        reserveSpace,
                                        reserveSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnRNNBackwardWeights(handle,
                                                rnnDesc,
                                                seqLength,
                                                xDesc,
                                                x,
                                                hxDesc,
                                                hx,
                                                yDesc,
                                                y,
                                                workspace,
                                                workSpaceSizeInBytes,
                                                dwDesc,
                                                dw,
                                                reserveSpace,
                                                reserveSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments",
       json::object({{"seqLength", seqLength},
                     {"workSpaceSizeInBytes", workSpaceSizeInBytes},
                     {"reserveSpaceSizeInBytes", reserveSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 149
cudnnStatus_t CUDNNWINAPI
    cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t *ctcLossDesc) {
  using fun_t = decltype(cudnnCreateCTCLossDescriptor);
  static const std::string funName{"cudnnCreateCTCLossDescriptor"};
  static fun_t *orig_cudnnCreateCTCLossDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCreateCTCLossDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreateCTCLossDescriptor(ctcLossDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreateCTCLossDescriptor(ctcLossDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 150
cudnnStatus_t CUDNNWINAPI cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc,
                                                    cudnnDataType_t compType) {
  using fun_t = decltype(cudnnSetCTCLossDescriptor);
  static const std::string funName{"cudnnSetCTCLossDescriptor"};
  static fun_t *orig_cudnnSetCTCLossDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetCTCLossDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetCTCLossDescriptor(ctcLossDesc, compType);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetCTCLossDescriptor(ctcLossDesc, compType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 151
cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc,
                                                    cudnnDataType_t *compType) {
  using fun_t = decltype(cudnnGetCTCLossDescriptor);
  static const std::string funName{"cudnnGetCTCLossDescriptor"};
  static fun_t *orig_cudnnGetCTCLossDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetCTCLossDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetCTCLossDescriptor(ctcLossDesc, compType);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetCTCLossDescriptor(ctcLossDesc, compType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 152
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc) {
  using fun_t = decltype(cudnnDestroyCTCLossDescriptor);
  static const std::string funName{"cudnnDestroyCTCLossDescriptor"};
  static fun_t *orig_cudnnDestroyCTCLossDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroyCTCLossDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroyCTCLossDescriptor(ctcLossDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroyCTCLossDescriptor(ctcLossDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 153
cudnnStatus_t CUDNNWINAPI cudnnCTCLoss(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t
        probsDesc, /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is
                      the timing steps, N is the mini batch size, A is the alphabet size)
                    */
    const void *probs,       /* probabilities after softmax, in GPU memory */
    const int *labels,       /* labels, in CPU memory */
    const int *labelLengths, /* the length of each label, in CPU memory */
    const int
        *inputLengths, /* the lengths of timing steps in each batch, in CPU memory */
    void *costs,       /* the returned costs of CTC, in GPU memory */
    const cudnnTensorDescriptor_t
        gradientsDesc,     /* Tensor descriptor for gradients, the dimensions are T,N,A */
    const void *gradients, /* the returned CTC gradients, in GPU memory, to compute costs
                              only, set it to NULL */
    cudnnCTCLossAlgo_t algo, /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc,
    void *workspace, /* pointer to the workspace, in GPU memory */
    size_t workSpaceSizeInBytes) {
  using fun_t = decltype(cudnnCTCLoss);
  static const std::string funName{"cudnnCTCLoss"};
  static fun_t *orig_cudnnCTCLoss = (fun_t *) dlsym(RTLD_NEXT, "cudnnCTCLoss");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCTCLoss(handle,
                             probsDesc,
                             probs,
                             labels,
                             labelLengths,
                             inputLengths,
                             costs,
                             gradientsDesc,
                             gradients,
                             algo,
                             ctcLossDesc,
                             workspace,
                             workSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnCTCLoss(handle,
                                     probsDesc,
                                     probs,
                                     labels,
                                     labelLengths,
                                     inputLengths,
                                     costs,
                                     gradientsDesc,
                                     gradients,
                                     algo,
                                     ctcLossDesc,
                                     workspace,
                                     workSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"workSpaceSizeInBytes", workSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 154
cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossWorkspaceSize(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t
        probsDesc, /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is
                      the timing steps, N is the mini batch size, A is the alphabet size)
                    */
    const cudnnTensorDescriptor_t
        gradientsDesc, /* Tensor descriptor for gradients, the dimensions are T,N,A. To
                          compute costs only, set it to NULL */
    const int *labels, /* labels, in CPU memory */
    const int *labelLengths, /* the length of each label, in CPU memory */
    const int
        *inputLengths, /* the lengths of timing steps in each batch, in CPU memory */
    cudnnCTCLossAlgo_t algo, /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc,
    size_t *sizeInBytes) {
  using fun_t = decltype(cudnnGetCTCLossWorkspaceSize);
  static const std::string funName{"cudnnGetCTCLossWorkspaceSize"};
  static fun_t *orig_cudnnGetCTCLossWorkspaceSize =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetCTCLossWorkspaceSize");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetCTCLossWorkspaceSize(handle,
                                             probsDesc,
                                             gradientsDesc,
                                             labels,
                                             labelLengths,
                                             inputLengths,
                                             algo,
                                             ctcLossDesc,
                                             sizeInBytes);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetCTCLossWorkspaceSize(handle,
                                                     probsDesc,
                                                     gradientsDesc,
                                                     labels,
                                                     labelLengths,
                                                     inputLengths,
                                                     algo,
                                                     ctcLossDesc,
                                                     sizeInBytes);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 155
cudnnStatus_t CUDNNWINAPI
    cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t *algoDesc) {
  using fun_t = decltype(cudnnCreateAlgorithmDescriptor);
  static const std::string funName{"cudnnCreateAlgorithmDescriptor"};
  static fun_t *orig_cudnnCreateAlgorithmDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCreateAlgorithmDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreateAlgorithmDescriptor(algoDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreateAlgorithmDescriptor(algoDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 156
cudnnStatus_t CUDNNWINAPI cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc,
                                                      cudnnAlgorithm_t algorithm) {
  using fun_t = decltype(cudnnSetAlgorithmDescriptor);
  static const std::string funName{"cudnnSetAlgorithmDescriptor"};
  static fun_t *orig_cudnnSetAlgorithmDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetAlgorithmDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetAlgorithmDescriptor(algoDesc, algorithm);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetAlgorithmDescriptor(algoDesc, algorithm);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 157
cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmDescriptor(
    const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algorithm) {
  using fun_t = decltype(cudnnGetAlgorithmDescriptor);
  static const std::string funName{"cudnnGetAlgorithmDescriptor"};
  static fun_t *orig_cudnnGetAlgorithmDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetAlgorithmDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetAlgorithmDescriptor(algoDesc, algorithm);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetAlgorithmDescriptor(algoDesc, algorithm);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 158
cudnnStatus_t CUDNNWINAPI cudnnCopyAlgorithmDescriptor(
    const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest) {
  using fun_t = decltype(cudnnCopyAlgorithmDescriptor);
  static const std::string funName{"cudnnCopyAlgorithmDescriptor"};
  static fun_t *orig_cudnnCopyAlgorithmDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCopyAlgorithmDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCopyAlgorithmDescriptor(src, dest);
  }
  const auto tic = now();
  const auto res = orig_cudnnCopyAlgorithmDescriptor(src, dest);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 159
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc) {
  using fun_t = decltype(cudnnDestroyAlgorithmDescriptor);
  static const std::string funName{"cudnnDestroyAlgorithmDescriptor"};
  static fun_t *orig_cudnnDestroyAlgorithmDescriptor =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroyAlgorithmDescriptor");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroyAlgorithmDescriptor(algoDesc);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroyAlgorithmDescriptor(algoDesc);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 160
cudnnStatus_t CUDNNWINAPI cudnnCreateAlgorithmPerformance(
    cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate) {
  using fun_t = decltype(cudnnCreateAlgorithmPerformance);
  static const std::string funName{"cudnnCreateAlgorithmPerformance"};
  static fun_t *orig_cudnnCreateAlgorithmPerformance =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnCreateAlgorithmPerformance");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnCreateAlgorithmPerformance(algoPerf, numberToCreate);
  }
  const auto tic = now();
  const auto res = orig_cudnnCreateAlgorithmPerformance(algoPerf, numberToCreate);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"numberToCreate", numberToCreate}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 161
cudnnStatus_t CUDNNWINAPI
    cudnnSetAlgorithmPerformance(cudnnAlgorithmPerformance_t algoPerf,
                                 cudnnAlgorithmDescriptor_t algoDesc,
                                 cudnnStatus_t status,
                                 float time,
                                 size_t memory) {
  using fun_t = decltype(cudnnSetAlgorithmPerformance);
  static const std::string funName{"cudnnSetAlgorithmPerformance"};
  static fun_t *orig_cudnnSetAlgorithmPerformance =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetAlgorithmPerformance");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnSetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"time", time}, {"memory", memory}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 162
cudnnStatus_t CUDNNWINAPI
    cudnnGetAlgorithmPerformance(const cudnnAlgorithmPerformance_t algoPerf,
                                 cudnnAlgorithmDescriptor_t *algoDesc,
                                 cudnnStatus_t *status,
                                 float *time,
                                 size_t *memory) {
  using fun_t = decltype(cudnnGetAlgorithmPerformance);
  static const std::string funName{"cudnnGetAlgorithmPerformance"};
  static fun_t *orig_cudnnGetAlgorithmPerformance =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetAlgorithmPerformance");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnGetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 163
cudnnStatus_t CUDNNWINAPI cudnnDestroyAlgorithmPerformance(
    cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy) {
  using fun_t = decltype(cudnnDestroyAlgorithmPerformance);
  static const std::string funName{"cudnnDestroyAlgorithmPerformance"};
  static fun_t *orig_cudnnDestroyAlgorithmPerformance =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnDestroyAlgorithmPerformance");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnDestroyAlgorithmPerformance(algoPerf, numberToDestroy);
  }
  const auto tic = now();
  const auto res = orig_cudnnDestroyAlgorithmPerformance(algoPerf, numberToDestroy);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"numberToDestroy", numberToDestroy}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 164
cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmSpaceSize(cudnnHandle_t handle,
                                                     cudnnAlgorithmDescriptor_t algoDesc,
                                                     size_t *algoSpaceSizeInBytes) {
  using fun_t = decltype(cudnnGetAlgorithmSpaceSize);
  static const std::string funName{"cudnnGetAlgorithmSpaceSize"};
  static fun_t *orig_cudnnGetAlgorithmSpaceSize =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnGetAlgorithmSpaceSize");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetAlgorithmSpaceSize(handle, algoDesc, algoSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnGetAlgorithmSpaceSize(handle, algoDesc, algoSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 165
cudnnStatus_t CUDNNWINAPI cudnnSaveAlgorithm(cudnnHandle_t handle,
                                             cudnnAlgorithmDescriptor_t algoDesc,
                                             void *algoSpace,
                                             size_t algoSpaceSizeInBytes) {
  using fun_t = decltype(cudnnSaveAlgorithm);
  static const std::string funName{"cudnnSaveAlgorithm"};
  static fun_t *orig_cudnnSaveAlgorithm =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSaveAlgorithm");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSaveAlgorithm(handle, algoDesc, algoSpace, algoSpaceSizeInBytes);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnSaveAlgorithm(handle, algoDesc, algoSpace, algoSpaceSizeInBytes);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"algoSpaceSizeInBytes", algoSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 166
cudnnStatus_t CUDNNWINAPI cudnnRestoreAlgorithm(cudnnHandle_t handle,
                                                void *algoSpace,
                                                size_t algoSpaceSizeInBytes,
                                                cudnnAlgorithmDescriptor_t algoDesc) {
  using fun_t = decltype(cudnnRestoreAlgorithm);
  static const std::string funName{"cudnnRestoreAlgorithm"};
  static fun_t *orig_cudnnRestoreAlgorithm =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnRestoreAlgorithm");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnRestoreAlgorithm(handle, algoSpace, algoSpaceSizeInBytes, algoDesc);
  }
  const auto tic = now();
  const auto res =
      orig_cudnnRestoreAlgorithm(handle, algoSpace, algoSpaceSizeInBytes, algoDesc);
  const auto toc = now();
  const json js  = {
      {"function_ame", funName},
      {"time_unit", "ns"},
      {"start", to_nanoseconds(tic)},
      {"end", to_nanoseconds(toc)},
      {"arguments", json::object({{"algoSpaceSizeInBytes", algoSpaceSizeInBytes}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 167
cudnnStatus_t CUDNNWINAPI cudnnSetCallback(unsigned mask,
                                           void *udata,
                                           cudnnCallback_t fptr) {
  using fun_t = decltype(cudnnSetCallback);
  static const std::string funName{"cudnnSetCallback"};
  static fun_t *orig_cudnnSetCallback = (fun_t *) dlsym(RTLD_NEXT, "cudnnSetCallback");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetCallback(mask, udata, fptr);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetCallback(mask, udata, fptr);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"mask", mask}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 168
cudnnStatus_t CUDNNWINAPI cudnnGetCallback(unsigned *mask,
                                           void **udata,
                                           cudnnCallback_t *fptr) {
  using fun_t = decltype(cudnnGetCallback);
  static const std::string funName{"cudnnGetCallback"};
  static fun_t *orig_cudnnGetCallback = (fun_t *) dlsym(RTLD_NEXT, "cudnnGetCallback");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnGetCallback(mask, udata, fptr);
  }
  const auto tic = now();
  const auto res = orig_cudnnGetCallback(mask, udata, fptr);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({

                                 })}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 169
cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v6(cudnnHandle_t handle,
                                                   cudnnRNNDescriptor_t rnnDesc,
                                                   const int hiddenSize,
                                                   const int numLayers,
                                                   cudnnDropoutDescriptor_t dropoutDesc,
                                                   cudnnRNNInputMode_t inputMode,
                                                   cudnnDirectionMode_t direction,
                                                   cudnnRNNMode_t mode,
                                                   cudnnRNNAlgo_t algo,
                                                   cudnnDataType_t dataType) {
  using fun_t = decltype(cudnnSetRNNDescriptor_v6);
  static const std::string funName{"cudnnSetRNNDescriptor_v6"};
  static fun_t *orig_cudnnSetRNNDescriptor_v6 =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetRNNDescriptor_v6");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetRNNDescriptor_v6(handle,
                                         rnnDesc,
                                         hiddenSize,
                                         numLayers,
                                         dropoutDesc,
                                         inputMode,
                                         direction,
                                         mode,
                                         algo,
                                         dataType);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetRNNDescriptor_v6(handle,
                                                 rnnDesc,
                                                 hiddenSize,
                                                 numLayers,
                                                 dropoutDesc,
                                                 inputMode,
                                                 direction,
                                                 mode,
                                                 algo,
                                                 dataType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"hiddenSize", hiddenSize},
                                               {"numLayers", numLayers}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}

// 170
cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v5(cudnnRNNDescriptor_t rnnDesc,
                                                   int hiddenSize,
                                                   int numLayers,
                                                   cudnnDropoutDescriptor_t dropoutDesc,
                                                   cudnnRNNInputMode_t inputMode,
                                                   cudnnDirectionMode_t direction,
                                                   cudnnRNNMode_t mode,
                                                   cudnnDataType_t dataType) {
  using fun_t = decltype(cudnnSetRNNDescriptor_v5);
  static const std::string funName{"cudnnSetRNNDescriptor_v5"};
  static fun_t *orig_cudnnSetRNNDescriptor_v5 =
      (fun_t *) dlsym(RTLD_NEXT, "cudnnSetRNNDescriptor_v5");

  if (!is_record_time_enabled_q()) {
    return orig_cudnnSetRNNDescriptor_v5(rnnDesc,
                                         hiddenSize,
                                         numLayers,
                                         dropoutDesc,
                                         inputMode,
                                         direction,
                                         mode,
                                         dataType);
  }
  const auto tic = now();
  const auto res = orig_cudnnSetRNNDescriptor_v5(
      rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, dataType);
  const auto toc = now();
  const json js  = {{"function_ame", funName},
                   {"time_unit", "ns"},
                   {"start", to_nanoseconds(tic)},
                   {"end", to_nanoseconds(toc)},
                   {"arguments", json::object({{"hiddenSize", hiddenSize},
                                               {"numLayers", numLayers}})}};
  const auto str = js.dump();
  record_cudnn_time(str.c_str());
  return res;
}
