#include "cudnn.h"
#include "utils.hpp"
#include <dlfcn.h>

// 0
cudnnStatus_t CUDNNWINAPI cudnnQueryRuntimeError(cudnnHandle_t handle,
                                                 cudnnStatus_t *rstatus,
                                                 cudnnErrQueryMode_t mode,
                                                 cudnnRuntimeTag_t *tag) {
  static const std::string funName{"cudnnQueryRuntimeError"};
  static auto orig_cudnnQueryRuntimeError =
      (decltype(cudnnQueryRuntimeError)) dlsym(RTLD_NEXT, "cudnnQueryRuntimeError");
  const auto tic = now();
  const auto res = orig_cudnnQueryRuntimeError(handle, rstatus, mode, tag);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"mode" : to_json(mode)}
    })
  };
  return res;
}

// 1
cudnnStatus_t CUDNNWINAPI cudnnGetProperty(libraryPropertyType type, int *value) {
  static const std::string funName{"cudnnGetProperty"};
  static auto orig_cudnnGetProperty =
      (decltype(cudnnGetProperty)) dlsym(RTLD_NEXT, "cudnnGetProperty");
  const auto tic = now();
  const auto res = orig_cudnnGetProperty(value);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 2
cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t *handle) {
  static const std::string funName{"cudnnCreate"};
  static auto orig_cudnnCreate = (decltype(cudnnCreate)) dlsym(RTLD_NEXT, "cudnnCreate");
  const auto tic               = now();
  const auto res               = orig_cudnnCreate(handle);
  const auto toc               = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 3
cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t handle) {
  static const std::string funName{"cudnnDestroy"};
  static auto orig_cudnnDestroy =
      (decltype(cudnnDestroy)) dlsym(RTLD_NEXT, "cudnnDestroy");
  const auto tic = now();
  const auto res = orig_cudnnDestroy(handle);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 4
cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
  static const std::string funName{"cudnnSetStream"};
  static auto orig_cudnnSetStream =
      (decltype(cudnnSetStream)) dlsym(RTLD_NEXT, "cudnnSetStream");
  const auto tic = now();
  const auto res = orig_cudnnSetStream(streamId);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"streamId" : to_json(streamId)}
    })
  };
  return res;
}

// 5
cudnnStatus_t CUDNNWINAPI cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId) {
  static const std::string funName{"cudnnGetStream"};
  static auto orig_cudnnGetStream =
      (decltype(cudnnGetStream)) dlsym(RTLD_NEXT, "cudnnGetStream");
  const auto tic = now();
  const auto res = orig_cudnnGetStream(streamId);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 6
cudnnStatus_t CUDNNWINAPI
    cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
  static const std::string funName{"cudnnCreateTensorDescriptor"};
  static auto orig_cudnnCreateTensorDescriptor =
      (decltype(cudnnCreateTensorDescriptor)) dlsym(RTLD_NEXT,
                                                    "cudnnCreateTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateTensorDescriptor(tensorDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 7
cudnnStatus_t CUDNNWINAPI
    cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                               cudnnTensorFormat_t format,
                               cudnnDataType_t dataType, /* image data type */
                               int n, /* number of inputs (batch size) */
                               int c, /* number of input feature maps */
                               int h, /* height of input section */
                               int w) {
  static const std::string funName{"cudnnSetTensor4dDescriptor"};
  static auto orig_cudnnSetTensor4dDescriptor =
      (decltype(cudnnSetTensor4dDescriptor)) dlsym(RTLD_NEXT,
                                                   "cudnnSetTensor4dDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnSetTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "format" : to_json(format),
        "dataType" : to_json(dataType),
        "c" : to_json(c),
        "w" : to_json(w)
      }
    })
  };
  return res;
}

// 8
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
  static const std::string funName{"cudnnSetTensor4dDescriptorEx"};
  static auto orig_cudnnSetTensor4dDescriptorEx =
      (decltype(cudnnSetTensor4dDescriptorEx)) dlsym(RTLD_NEXT,
                                                     "cudnnSetTensor4dDescriptorEx");
  const auto tic = now();
  const auto res = orig_cudnnSetTensor4dDescriptorEx(
      tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "dataType" : to_json(dataType),
        "c" : to_json(c),
        "nStride" : to_json(nStride),
        "cStride" : to_json(cStride),
        "hStride" : to_json(hStride),
        "wStride" : to_json(wStride)
      }
    })
  };
  return res;
}

// 9
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
  static const std::string funName{"cudnnGetTensor4dDescriptor"};
  static auto orig_cudnnGetTensor4dDescriptor =
      (decltype(cudnnGetTensor4dDescriptor)) dlsym(RTLD_NEXT,
                                                   "cudnnGetTensor4dDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetTensor4dDescriptor(
      tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 10
cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                                     cudnnDataType_t dataType,
                                                     int nbDims,
                                                     const int dimA[],
                                                     const int strideA[]) {
  static const std::string funName{"cudnnSetTensorNdDescriptor"};
  static auto orig_cudnnSetTensorNdDescriptor =
      (decltype(cudnnSetTensorNdDescriptor)) dlsym(RTLD_NEXT,
                                                   "cudnnSetTensorNdDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "dataType" : to_json(dataType),
        "nbDims" : to_json(nbDims),
        "dimA" : to_json(dimA),
        "strideA" : to_json(strideA)
      }
    })
  };
  return res;
}

// 11
cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                                                       cudnnTensorFormat_t format,
                                                       cudnnDataType_t dataType,
                                                       int nbDims,
                                                       const int dimA[]) {
  static const std::string funName{"cudnnSetTensorNdDescriptorEx"};
  static auto orig_cudnnSetTensorNdDescriptorEx =
      (decltype(cudnnSetTensorNdDescriptorEx)) dlsym(RTLD_NEXT,
                                                     "cudnnSetTensorNdDescriptorEx");
  const auto tic = now();
  const auto res =
      orig_cudnnSetTensorNdDescriptorEx(tensorDesc, format, dataType, nbDims, dimA);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "format" : to_json(format),
        "dataType" : to_json(dataType),
        "nbDims" : to_json(nbDims),
        "dimA" : to_json(dimA)
      }
    })
  };
  return res;
}

// 12
cudnnStatus_t CUDNNWINAPI
    cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                               int nbDimsRequested,
                               cudnnDataType_t *dataType,
                               int *nbDims,
                               int dimA[],
                               int strideA[]) {
  static const std::string funName{"cudnnGetTensorNdDescriptor"};
  static auto orig_cudnnGetTensorNdDescriptor =
      (decltype(cudnnGetTensorNdDescriptor)) dlsym(RTLD_NEXT,
                                                   "cudnnGetTensorNdDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetTensorNdDescriptor(
      tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "nbDimsRequested" : to_json(nbDimsRequested),
        "dimA" : to_json(dimA),
        "strideA" : to_json(strideA)
      }
    })
  };
  return res;
}

// 13
cudnnStatus_t CUDNNWINAPI
    cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc, size_t *size) {
  static const std::string funName{"cudnnGetTensorSizeInBytes"};
  static auto orig_cudnnGetTensorSizeInBytes =
      (decltype(cudnnGetTensorSizeInBytes)) dlsym(RTLD_NEXT, "cudnnGetTensorSizeInBytes");
  const auto tic = now();
  const auto res = orig_cudnnGetTensorSizeInBytes(tensorDesc, size);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 14
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
  static const std::string funName{"cudnnDestroyTensorDescriptor"};
  static auto orig_cudnnDestroyTensorDescriptor =
      (decltype(cudnnDestroyTensorDescriptor)) dlsym(RTLD_NEXT,
                                                     "cudnnDestroyTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyTensorDescriptor(tensorDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 15
cudnnStatus_t CUDNNWINAPI cudnnTransformTensor(cudnnHandle_t handle,
                                               const void *alpha,
                                               const cudnnTensorDescriptor_t xDesc,
                                               const void *x,
                                               const void *beta,
                                               const cudnnTensorDescriptor_t yDesc,
                                               void *y) {
  static const std::string funName{"cudnnTransformTensor"};
  static auto orig_cudnnTransformTensor =
      (decltype(cudnnTransformTensor)) dlsym(RTLD_NEXT, "cudnnTransformTensor");
  const auto tic = now();
  const auto res = orig_cudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 16
cudnnStatus_t CUDNNWINAPI cudnnAddTensor(cudnnHandle_t handle,
                                         const void *alpha,
                                         const cudnnTensorDescriptor_t aDesc,
                                         const void *A,
                                         const void *beta,
                                         const cudnnTensorDescriptor_t cDesc,
                                         void *C) {
  static const std::string funName{"cudnnAddTensor"};
  static auto orig_cudnnAddTensor =
      (decltype(cudnnAddTensor)) dlsym(RTLD_NEXT, "cudnnAddTensor");
  const auto tic = now();
  const auto res = orig_cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 17
cudnnStatus_t CUDNNWINAPI
    cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t *opTensorDesc) {
  static const std::string funName{"cudnnCreateOpTensorDescriptor"};
  static auto orig_cudnnCreateOpTensorDescriptor =
      (decltype(cudnnCreateOpTensorDescriptor)) dlsym(RTLD_NEXT,
                                                      "cudnnCreateOpTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateOpTensorDescriptor(opTensorDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 18
cudnnStatus_t CUDNNWINAPI
    cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc,
                               cudnnOpTensorOp_t opTensorOp,
                               cudnnDataType_t opTensorCompType,
                               cudnnNanPropagation_t opTensorNanOpt) {
  static const std::string funName{"cudnnSetOpTensorDescriptor"};
  static auto orig_cudnnSetOpTensorDescriptor =
      (decltype(cudnnSetOpTensorDescriptor)) dlsym(RTLD_NEXT,
                                                   "cudnnSetOpTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetOpTensorDescriptor(
      opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "opTensorOp" : to_json(opTensorOp),
        "opTensorCompType" : to_json(opTensorCompType),
        "opTensorNanOpt" : to_json(opTensorNanOpt)
      }
    })
  };
  return res;
}

// 19
cudnnStatus_t CUDNNWINAPI
    cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t opTensorDesc,
                               cudnnOpTensorOp_t *opTensorOp,
                               cudnnDataType_t *opTensorCompType,
                               cudnnNanPropagation_t *opTensorNanOpt) {
  static const std::string funName{"cudnnGetOpTensorDescriptor"};
  static auto orig_cudnnGetOpTensorDescriptor =
      (decltype(cudnnGetOpTensorDescriptor)) dlsym(RTLD_NEXT,
                                                   "cudnnGetOpTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetOpTensorDescriptor(
      opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 20
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc) {
  static const std::string funName{"cudnnDestroyOpTensorDescriptor"};
  static auto orig_cudnnDestroyOpTensorDescriptor =
      (decltype(cudnnDestroyOpTensorDescriptor)) dlsym(RTLD_NEXT,
                                                       "cudnnDestroyOpTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyOpTensorDescriptor(opTensorDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 21
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
  static const std::string funName{"cudnnOpTensor"};
  static auto orig_cudnnOpTensor =
      (decltype(cudnnOpTensor)) dlsym(RTLD_NEXT, "cudnnOpTensor");
  const auto tic = now();
  const auto res = orig_cudnnOpTensor(
      handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 22
cudnnStatus_t CUDNNWINAPI
    cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc) {
  static const std::string funName{"cudnnCreateReduceTensorDescriptor"};
  static auto orig_cudnnCreateReduceTensorDescriptor =
      (decltype(cudnnCreateReduceTensorDescriptor)) dlsym(
          RTLD_NEXT, "cudnnCreateReduceTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateReduceTensorDescriptor(reduceTensorDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 23
cudnnStatus_t CUDNNWINAPI
    cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                   cudnnReduceTensorOp_t reduceTensorOp,
                                   cudnnDataType_t reduceTensorCompType,
                                   cudnnNanPropagation_t reduceTensorNanOpt,
                                   cudnnReduceTensorIndices_t reduceTensorIndices,
                                   cudnnIndicesType_t reduceTensorIndicesType) {
  static const std::string funName{"cudnnSetReduceTensorDescriptor"};
  static auto orig_cudnnSetReduceTensorDescriptor =
      (decltype(cudnnSetReduceTensorDescriptor)) dlsym(RTLD_NEXT,
                                                       "cudnnSetReduceTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetReduceTensorDescriptor(reduceTensorDesc,
                                                       reduceTensorOp,
                                                       reduceTensorCompType,
                                                       reduceTensorNanOpt,
                                                       reduceTensorIndices,
                                                       reduceTensorIndicesType);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "reduceTensorOp" : to_json(reduceTensorOp),
        "reduceTensorCompType" : to_json(reduceTensorCompType),
        "reduceTensorNanOpt" : to_json(reduceTensorNanOpt),
        "reduceTensorIndices" : to_json(reduceTensorIndices),
        "reduceTensorIndicesType" : to_json(reduceTensorIndicesType)
      }
    })
  };
  return res;
}

// 24
cudnnStatus_t CUDNNWINAPI
    cudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                   cudnnReduceTensorOp_t *reduceTensorOp,
                                   cudnnDataType_t *reduceTensorCompType,
                                   cudnnNanPropagation_t *reduceTensorNanOpt,
                                   cudnnReduceTensorIndices_t *reduceTensorIndices,
                                   cudnnIndicesType_t *reduceTensorIndicesType) {
  static const std::string funName{"cudnnGetReduceTensorDescriptor"};
  static auto orig_cudnnGetReduceTensorDescriptor =
      (decltype(cudnnGetReduceTensorDescriptor)) dlsym(RTLD_NEXT,
                                                       "cudnnGetReduceTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetReduceTensorDescriptor(reduceTensorDesc,
                                                       reduceTensorOp,
                                                       reduceTensorCompType,
                                                       reduceTensorNanOpt,
                                                       reduceTensorIndices,
                                                       reduceTensorIndicesType);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 25
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc) {
  static const std::string funName{"cudnnDestroyReduceTensorDescriptor"};
  static auto orig_cudnnDestroyReduceTensorDescriptor =
      (decltype(cudnnDestroyReduceTensorDescriptor)) dlsym(
          RTLD_NEXT, "cudnnDestroyReduceTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 26
cudnnStatus_t CUDNNWINAPI
    cudnnGetReductionIndicesSize(cudnnHandle_t handle,
                                 const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                 const cudnnTensorDescriptor_t aDesc,
                                 const cudnnTensorDescriptor_t cDesc,
                                 size_t *sizeInBytes) {
  static const std::string funName{"cudnnGetReductionIndicesSize"};
  static auto orig_cudnnGetReductionIndicesSize =
      (decltype(cudnnGetReductionIndicesSize)) dlsym(RTLD_NEXT,
                                                     "cudnnGetReductionIndicesSize");
  const auto tic = now();
  const auto res = orig_cudnnGetReductionIndicesSize(
      handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 27
cudnnStatus_t CUDNNWINAPI
    cudnnGetReductionWorkspaceSize(cudnnHandle_t handle,
                                   const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                   const cudnnTensorDescriptor_t aDesc,
                                   const cudnnTensorDescriptor_t cDesc,
                                   size_t *sizeInBytes) {
  static const std::string funName{"cudnnGetReductionWorkspaceSize"};
  static auto orig_cudnnGetReductionWorkspaceSize =
      (decltype(cudnnGetReductionWorkspaceSize)) dlsym(RTLD_NEXT,
                                                       "cudnnGetReductionWorkspaceSize");
  const auto tic = now();
  const auto res = orig_cudnnGetReductionWorkspaceSize(
      handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 28
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
  static const std::string funName{"cudnnReduceTensor"};
  static auto orig_cudnnReduceTensor =
      (decltype(cudnnReduceTensor)) dlsym(RTLD_NEXT, "cudnnReduceTensor");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "indicesSizeInBytes" : to_json(indicesSizeInBytes),
        "workspaceSizeInBytes" : to_json(workspaceSizeInBytes)
      }
    })
  };
  return res;
}

// 29
cudnnStatus_t CUDNNWINAPI cudnnSetTensor(cudnnHandle_t handle,
                                         const cudnnTensorDescriptor_t yDesc,
                                         void *y,
                                         const void *valuePtr) {
  static const std::string funName{"cudnnSetTensor"};
  static auto orig_cudnnSetTensor =
      (decltype(cudnnSetTensor)) dlsym(RTLD_NEXT, "cudnnSetTensor");
  const auto tic = now();
  const auto res = orig_cudnnSetTensor(handle, yDesc, y, valuePtr);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 30
cudnnStatus_t CUDNNWINAPI cudnnScaleTensor(cudnnHandle_t handle,
                                           const cudnnTensorDescriptor_t yDesc,
                                           void *y,
                                           const void *alpha) {
  static const std::string funName{"cudnnScaleTensor"};
  static auto orig_cudnnScaleTensor =
      (decltype(cudnnScaleTensor)) dlsym(RTLD_NEXT, "cudnnScaleTensor");
  const auto tic = now();
  const auto res = orig_cudnnScaleTensor(handle, yDesc, y, alpha);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 31
cudnnStatus_t CUDNNWINAPI
    cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
  static const std::string funName{"cudnnCreateFilterDescriptor"};
  static auto orig_cudnnCreateFilterDescriptor =
      (decltype(cudnnCreateFilterDescriptor)) dlsym(RTLD_NEXT,
                                                    "cudnnCreateFilterDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateFilterDescriptor(filterDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 32
cudnnStatus_t CUDNNWINAPI
    cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                               cudnnDataType_t dataType, /* image data type */
                               cudnnTensorFormat_t format,
                               int k, /* number of output feature maps */
                               int c, /* number of input feature maps */
                               int h, /* height of each input filter */
                               int w) {
  static const std::string funName{"cudnnSetFilter4dDescriptor"};
  static auto orig_cudnnSetFilter4dDescriptor =
      (decltype(cudnnSetFilter4dDescriptor)) dlsym(RTLD_NEXT,
                                                   "cudnnSetFilter4dDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "dataType" : to_json(dataType),
        "format" : to_json(format),
        "k" : to_json(k),
        "c" : to_json(c),
        "w" : to_json(w)
      }
    })
  };
  return res;
}

// 33
cudnnStatus_t CUDNNWINAPI
    cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t filterDesc,
                               cudnnDataType_t *dataType, /* image data type */
                               cudnnTensorFormat_t *format,
                               int *k, /* number of output feature maps */
                               int *c, /* number of input feature maps */
                               int *h, /* height of each input filter */
                               int *w) {
  static const std::string funName{"cudnnGetFilter4dDescriptor"};
  static auto orig_cudnnGetFilter4dDescriptor =
      (decltype(cudnnGetFilter4dDescriptor)) dlsym(RTLD_NEXT,
                                                   "cudnnGetFilter4dDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnGetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 34
cudnnStatus_t CUDNNWINAPI
    cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,
                               cudnnDataType_t dataType, /* image data type */
                               cudnnTensorFormat_t format,
                               int nbDims,
                               const int filterDimA[]) {
  static const std::string funName{"cudnnSetFilterNdDescriptor"};
  static auto orig_cudnnSetFilterNdDescriptor =
      (decltype(cudnnSetFilterNdDescriptor)) dlsym(RTLD_NEXT,
                                                   "cudnnSetFilterNdDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnSetFilterNdDescriptor(filterDesc, dataType, format, nbDims, filterDimA);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "dataType" : to_json(dataType),
        "format" : to_json(format),
        "nbDims" : to_json(nbDims),
        "filterDimA" : to_json(filterDimA)
      }
    })
  };
  return res;
}

// 35
cudnnStatus_t CUDNNWINAPI
    cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t filterDesc,
                               int nbDimsRequested,
                               cudnnDataType_t *dataType, /* image data type */
                               cudnnTensorFormat_t *format,
                               int *nbDims,
                               int filterDimA[]) {
  static const std::string funName{"cudnnGetFilterNdDescriptor"};
  static auto orig_cudnnGetFilterNdDescriptor =
      (decltype(cudnnGetFilterNdDescriptor)) dlsym(RTLD_NEXT,
                                                   "cudnnGetFilterNdDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetFilterNdDescriptor(
      filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "nbDimsRequested" : to_json(nbDimsRequested),
        "filterDimA" : to_json(filterDimA)
      }
    })
  };
  return res;
}

// 36
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
  static const std::string funName{"cudnnDestroyFilterDescriptor"};
  static auto orig_cudnnDestroyFilterDescriptor =
      (decltype(cudnnDestroyFilterDescriptor)) dlsym(RTLD_NEXT,
                                                     "cudnnDestroyFilterDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyFilterDescriptor(filterDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 37
cudnnStatus_t CUDNNWINAPI
    cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
  static const std::string funName{"cudnnCreateConvolutionDescriptor"};
  static auto orig_cudnnCreateConvolutionDescriptor =
      (decltype(cudnnCreateConvolutionDescriptor)) dlsym(
          RTLD_NEXT, "cudnnCreateConvolutionDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateConvolutionDescriptor(convDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 38
cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionMathType(
    cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType) {
  static const std::string funName{"cudnnSetConvolutionMathType"};
  static auto orig_cudnnSetConvolutionMathType =
      (decltype(cudnnSetConvolutionMathType)) dlsym(RTLD_NEXT,
                                                    "cudnnSetConvolutionMathType");
  const auto tic = now();
  const auto res = orig_cudnnSetConvolutionMathType(convDesc, mathType);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"mathType" : to_json(mathType)}
    })
  };
  return res;
}

// 39
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionMathType(
    cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType) {
  static const std::string funName{"cudnnGetConvolutionMathType"};
  static auto orig_cudnnGetConvolutionMathType =
      (decltype(cudnnGetConvolutionMathType)) dlsym(RTLD_NEXT,
                                                    "cudnnGetConvolutionMathType");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionMathType(convDesc, mathType);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 40
cudnnStatus_t CUDNNWINAPI
    cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount) {
  static const std::string funName{"cudnnSetConvolutionGroupCount"};
  static auto orig_cudnnSetConvolutionGroupCount =
      (decltype(cudnnSetConvolutionGroupCount)) dlsym(RTLD_NEXT,
                                                      "cudnnSetConvolutionGroupCount");
  const auto tic = now();
  const auto res = orig_cudnnSetConvolutionGroupCount(convDesc, groupCount);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"groupCount" : to_json(groupCount)}
    })
  };
  return res;
}

// 41
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionGroupCount(
    cudnnConvolutionDescriptor_t convDesc, int *groupCount) {
  static const std::string funName{"cudnnGetConvolutionGroupCount"};
  static auto orig_cudnnGetConvolutionGroupCount =
      (decltype(cudnnGetConvolutionGroupCount)) dlsym(RTLD_NEXT,
                                                      "cudnnGetConvolutionGroupCount");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionGroupCount(convDesc, groupCount);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 42
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
  static const std::string funName{"cudnnSetConvolution2dDescriptor"};
  static auto orig_cudnnSetConvolution2dDescriptor =
      (decltype(cudnnSetConvolution2dDescriptor)) dlsym(
          RTLD_NEXT, "cudnnSetConvolution2dDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetConvolution2dDescriptor(
      convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "pad_h" : to_json(pad_h),
        "pad_w" : to_json(pad_w),
        "u" : to_json(u),
        "dilation_h" : to_json(dilation_h),
        "dilation_w" : to_json(dilation_w),
        "mode" : to_json(mode),
        "computeType" : to_json(computeType)
      }
    })
  };
  return res;
}

// 43
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
  static const std::string funName{"cudnnGetConvolution2dDescriptor"};
  static auto orig_cudnnGetConvolution2dDescriptor =
      (decltype(cudnnGetConvolution2dDescriptor)) dlsym(
          RTLD_NEXT, "cudnnGetConvolution2dDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolution2dDescriptor(
      convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 44
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                          const cudnnTensorDescriptor_t inputTensorDesc,
                                          const cudnnFilterDescriptor_t filterDesc,
                                          int *n,
                                          int *c,
                                          int *h,
                                          int *w) {
  static const std::string funName{"cudnnGetConvolution2dForwardOutputDim"};
  static auto orig_cudnnGetConvolution2dForwardOutputDim =
      (decltype(cudnnGetConvolution2dForwardOutputDim)) dlsym(
          RTLD_NEXT, "cudnnGetConvolution2dForwardOutputDim");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolution2dForwardOutputDim(
      convDesc, inputTensorDesc, filterDesc, n, c, h, w);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 45
cudnnStatus_t CUDNNWINAPI
    cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                    int arrayLength, /* nbDims-2 size */
                                    const int padA[],
                                    const int filterStrideA[],
                                    const int dilationA[],
                                    cudnnConvolutionMode_t mode,
                                    cudnnDataType_t computeType) {
  static const std::string funName{"cudnnSetConvolutionNdDescriptor"};
  static auto orig_cudnnSetConvolutionNdDescriptor =
      (decltype(cudnnSetConvolutionNdDescriptor)) dlsym(
          RTLD_NEXT, "cudnnSetConvolutionNdDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetConvolutionNdDescriptor(
      convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "arrayLength" : to_json(arrayLength),
        "padA" : to_json(padA),
        "filterStrideA" : to_json(filterStrideA),
        "dilationA" : to_json(dilationA),
        "mode" : to_json(mode),
        "computeType" : to_json(computeType)
      }
    })
  };
  return res;
}

// 46
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t convDesc,
                                    int arrayLengthRequested,
                                    int *arrayLength,
                                    int padA[],
                                    int strideA[],
                                    int dilationA[],
                                    cudnnConvolutionMode_t *mode,
                                    cudnnDataType_t *computeType) {
  static const std::string funName{"cudnnGetConvolutionNdDescriptor"};
  static auto orig_cudnnGetConvolutionNdDescriptor =
      (decltype(cudnnGetConvolutionNdDescriptor)) dlsym(
          RTLD_NEXT, "cudnnGetConvolutionNdDescriptor");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "arrayLengthRequested" : to_json(arrayLengthRequested),
        "padA" : to_json(padA),
        "strideA" : to_json(strideA),
        "dilationA" : to_json(dilationA)
      }
    })
  };
  return res;
}

// 47
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                          const cudnnTensorDescriptor_t inputTensorDesc,
                                          const cudnnFilterDescriptor_t filterDesc,
                                          int nbDims,
                                          int tensorOuputDimA[]) {
  static const std::string funName{"cudnnGetConvolutionNdForwardOutputDim"};
  static auto orig_cudnnGetConvolutionNdForwardOutputDim =
      (decltype(cudnnGetConvolutionNdForwardOutputDim)) dlsym(
          RTLD_NEXT, "cudnnGetConvolutionNdForwardOutputDim");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionNdForwardOutputDim(
      convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" :
          {"nbDims" : to_json(nbDims), "tensorOuputDimA" : to_json(tensorOuputDimA)}
    })
  };
  return res;
}

// 48
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
  static const std::string funName{"cudnnDestroyConvolutionDescriptor"};
  static auto orig_cudnnDestroyConvolutionDescriptor =
      (decltype(cudnnDestroyConvolutionDescriptor)) dlsym(
          RTLD_NEXT, "cudnnDestroyConvolutionDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyConvolutionDescriptor(convDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 49
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
  static const std::string funName{"cudnnGetConvolutionForwardAlgorithmMaxCount"};
  static auto orig_cudnnGetConvolutionForwardAlgorithmMaxCount =
      (decltype(cudnnGetConvolutionForwardAlgorithmMaxCount)) dlsym(
          RTLD_NEXT, "cudnnGetConvolutionForwardAlgorithmMaxCount");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionForwardAlgorithmMaxCount(handle, count);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 50
cudnnStatus_t CUDNNWINAPI
    cudnnFindConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                         const cudnnTensorDescriptor_t xDesc,
                                         const cudnnFilterDescriptor_t wDesc,
                                         const cudnnConvolutionDescriptor_t convDesc,
                                         const cudnnTensorDescriptor_t yDesc,
                                         const int requestedAlgoCount,
                                         int *returnedAlgoCount,
                                         cudnnConvolutionFwdAlgoPerf_t *perfResults) {
  static const std::string funName{"cudnnFindConvolutionForwardAlgorithm"};
  static auto orig_cudnnFindConvolutionForwardAlgorithm =
      (decltype(cudnnFindConvolutionForwardAlgorithm)) dlsym(
          RTLD_NEXT, "cudnnFindConvolutionForwardAlgorithm");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"requestedAlgoCount" : to_json(requestedAlgoCount)}
    })
  };
  return res;
}

// 51
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
  static const std::string funName{"cudnnFindConvolutionForwardAlgorithmEx"};
  static auto orig_cudnnFindConvolutionForwardAlgorithmEx =
      (decltype(cudnnFindConvolutionForwardAlgorithmEx)) dlsym(
          RTLD_NEXT, "cudnnFindConvolutionForwardAlgorithmEx");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "requestedAlgoCount" : to_json(requestedAlgoCount),
        "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes)
      }
    })
  };
  return res;
}

// 52
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                        const cudnnTensorDescriptor_t xDesc,
                                        const cudnnFilterDescriptor_t wDesc,
                                        const cudnnConvolutionDescriptor_t convDesc,
                                        const cudnnTensorDescriptor_t yDesc,
                                        cudnnConvolutionFwdPreference_t preference,
                                        size_t memoryLimitInBytes,
                                        cudnnConvolutionFwdAlgo_t *algo) {
  static const std::string funName{"cudnnGetConvolutionForwardAlgorithm"};
  static auto orig_cudnnGetConvolutionForwardAlgorithm =
      (decltype(cudnnGetConvolutionForwardAlgorithm)) dlsym(
          RTLD_NEXT, "cudnnGetConvolutionForwardAlgorithm");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionForwardAlgorithm(
      handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, algo);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "preference" : to_json(preference),
        "memoryLimitInBytes" : to_json(memoryLimitInBytes)
      }
    })
  };
  return res;
}

// 53
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t handle,
                                           const cudnnTensorDescriptor_t srcDesc,
                                           const cudnnFilterDescriptor_t filterDesc,
                                           const cudnnConvolutionDescriptor_t convDesc,
                                           const cudnnTensorDescriptor_t destDesc,
                                           const int requestedAlgoCount,
                                           int *returnedAlgoCount,
                                           cudnnConvolutionFwdAlgoPerf_t *perfResults) {
  static const std::string funName{"cudnnGetConvolutionForwardAlgorithm_v7"};
  static auto orig_cudnnGetConvolutionForwardAlgorithm_v7 =
      (decltype(cudnnGetConvolutionForwardAlgorithm_v7)) dlsym(
          RTLD_NEXT, "cudnnGetConvolutionForwardAlgorithm_v7");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"requestedAlgoCount" : to_json(requestedAlgoCount)}
    })
  };
  return res;
}

// 54
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,
                                            const cudnnTensorDescriptor_t xDesc,
                                            const cudnnFilterDescriptor_t wDesc,
                                            const cudnnConvolutionDescriptor_t convDesc,
                                            const cudnnTensorDescriptor_t yDesc,
                                            cudnnConvolutionFwdAlgo_t algo,
                                            size_t *sizeInBytes) {
  static const std::string funName{"cudnnGetConvolutionForwardWorkspaceSize"};
  static auto orig_cudnnGetConvolutionForwardWorkspaceSize =
      (decltype(cudnnGetConvolutionForwardWorkspaceSize)) dlsym(
          RTLD_NEXT, "cudnnGetConvolutionForwardWorkspaceSize");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionForwardWorkspaceSize(
      handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"algo" : to_json(algo)}
    })
  };
  return res;
}

// 55
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
  static const std::string funName{"cudnnConvolutionForward"};
  static auto orig_cudnnConvolutionForward =
      (decltype(cudnnConvolutionForward)) dlsym(RTLD_NEXT, "cudnnConvolutionForward");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "algo" : to_json(algo),
        "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes)
      }
    })
  };
  return res;
}

// 56
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
  static const std::string funName{"cudnnConvolutionBiasActivationForward"};
  static auto orig_cudnnConvolutionBiasActivationForward =
      (decltype(cudnnConvolutionBiasActivationForward)) dlsym(
          RTLD_NEXT, "cudnnConvolutionBiasActivationForward");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "algo" : to_json(algo),
        "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes)
      }
    })
  };
  return res;
}

// 57
cudnnStatus_t CUDNNWINAPI
    cudnnConvolutionBackwardBias(cudnnHandle_t handle,
                                 const void *alpha,
                                 const cudnnTensorDescriptor_t dyDesc,
                                 const void *dy,
                                 const void *beta,
                                 const cudnnTensorDescriptor_t dbDesc,
                                 void *db) {
  static const std::string funName{"cudnnConvolutionBackwardBias"};
  static auto orig_cudnnConvolutionBackwardBias =
      (decltype(cudnnConvolutionBackwardBias)) dlsym(RTLD_NEXT,
                                                     "cudnnConvolutionBackwardBias");
  const auto tic = now();
  const auto res =
      orig_cudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, dbDesc, db);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 58
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
  static const std::string funName{"cudnnGetConvolutionBackwardFilterAlgorithmMaxCount"};
  static auto orig_cudnnGetConvolutionBackwardFilterAlgorithmMaxCount =
      (decltype(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount)) dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, count);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 59
cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
  static const std::string funName{"cudnnFindConvolutionBackwardFilterAlgorithm"};
  static auto orig_cudnnFindConvolutionBackwardFilterAlgorithm =
      (decltype(cudnnFindConvolutionBackwardFilterAlgorithm)) dlsym(
          RTLD_NEXT, "cudnnFindConvolutionBackwardFilterAlgorithm");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"requestedAlgoCount" : to_json(requestedAlgoCount)}
    })
  };
  return res;
}

// 60
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
  static const std::string funName{"cudnnFindConvolutionBackwardFilterAlgorithmEx"};
  static auto orig_cudnnFindConvolutionBackwardFilterAlgorithmEx =
      (decltype(cudnnFindConvolutionBackwardFilterAlgorithmEx)) dlsym(
          RTLD_NEXT, "cudnnFindConvolutionBackwardFilterAlgorithmEx");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "requestedAlgoCount" : to_json(requestedAlgoCount),
        "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes)
      }
    })
  };
  return res;
}

// 61
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    cudnnConvolutionBwdFilterPreference_t preference,
    size_t memoryLimitInBytes,
    cudnnConvolutionBwdFilterAlgo_t *algo) {
  static const std::string funName{"cudnnGetConvolutionBackwardFilterAlgorithm"};
  static auto orig_cudnnGetConvolutionBackwardFilterAlgorithm =
      (decltype(cudnnGetConvolutionBackwardFilterAlgorithm)) dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardFilterAlgorithm");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardFilterAlgorithm(
      handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, algo);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "preference" : to_json(preference),
        "memoryLimitInBytes" : to_json(memoryLimitInBytes)
      }
    })
  };
  return res;
}

// 62
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t srcDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
  static const std::string funName{"cudnnGetConvolutionBackwardFilterAlgorithm_v7"};
  static auto orig_cudnnGetConvolutionBackwardFilterAlgorithm_v7 =
      (decltype(cudnnGetConvolutionBackwardFilterAlgorithm_v7)) dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardFilterAlgorithm_v7");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"requestedAlgoCount" : to_json(requestedAlgoCount)}
    })
  };
  return res;
}

// 63
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    size_t *sizeInBytes) {
  static const std::string funName{"cudnnGetConvolutionBackwardFilterWorkspaceSize"};
  static auto orig_cudnnGetConvolutionBackwardFilterWorkspaceSize =
      (decltype(cudnnGetConvolutionBackwardFilterWorkspaceSize)) dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardFilterWorkspaceSize");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardFilterWorkspaceSize(
      handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"algo" : to_json(algo)}
    })
  };
  return res;
}

// 64
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
  static const std::string funName{"cudnnConvolutionBackwardFilter"};
  static auto orig_cudnnConvolutionBackwardFilter =
      (decltype(cudnnConvolutionBackwardFilter)) dlsym(RTLD_NEXT,
                                                       "cudnnConvolutionBackwardFilter");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "algo" : to_json(algo),
        "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes)
      }
    })
  };
  return res;
}

// 65
cudnnStatus_t CUDNNWINAPI
    cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
  static const std::string funName{"cudnnGetConvolutionBackwardDataAlgorithmMaxCount"};
  static auto orig_cudnnGetConvolutionBackwardDataAlgorithmMaxCount =
      (decltype(cudnnGetConvolutionBackwardDataAlgorithmMaxCount)) dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardDataAlgorithmMaxCount");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, count);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 66
cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
  static const std::string funName{"cudnnFindConvolutionBackwardDataAlgorithm"};
  static auto orig_cudnnFindConvolutionBackwardDataAlgorithm =
      (decltype(cudnnFindConvolutionBackwardDataAlgorithm)) dlsym(
          RTLD_NEXT, "cudnnFindConvolutionBackwardDataAlgorithm");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"requestedAlgoCount" : to_json(requestedAlgoCount)}
    })
  };
  return res;
}

// 67
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
  static const std::string funName{"cudnnFindConvolutionBackwardDataAlgorithmEx"};
  static auto orig_cudnnFindConvolutionBackwardDataAlgorithmEx =
      (decltype(cudnnFindConvolutionBackwardDataAlgorithmEx)) dlsym(
          RTLD_NEXT, "cudnnFindConvolutionBackwardDataAlgorithmEx");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "requestedAlgoCount" : to_json(requestedAlgoCount),
        "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes)
      }
    })
  };
  return res;
}

// 68
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataPreference_t preference,
    size_t memoryLimitInBytes,
    cudnnConvolutionBwdDataAlgo_t *algo) {
  static const std::string funName{"cudnnGetConvolutionBackwardDataAlgorithm"};
  static auto orig_cudnnGetConvolutionBackwardDataAlgorithm =
      (decltype(cudnnGetConvolutionBackwardDataAlgorithm)) dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardDataAlgorithm");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardDataAlgorithm(
      handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, algo);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "preference" : to_json(preference),
        "memoryLimitInBytes" : to_json(memoryLimitInBytes)
      }
    })
  };
  return res;
}

// 69
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
  static const std::string funName{"cudnnGetConvolutionBackwardDataAlgorithm_v7"};
  static auto orig_cudnnGetConvolutionBackwardDataAlgorithm_v7 =
      (decltype(cudnnGetConvolutionBackwardDataAlgorithm_v7)) dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardDataAlgorithm_v7");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"requestedAlgoCount" : to_json(requestedAlgoCount)}
    })
  };
  return res;
}

// 70
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes) {
  static const std::string funName{"cudnnGetConvolutionBackwardDataWorkspaceSize"};
  static auto orig_cudnnGetConvolutionBackwardDataWorkspaceSize =
      (decltype(cudnnGetConvolutionBackwardDataWorkspaceSize)) dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardDataWorkspaceSize");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardDataWorkspaceSize(
      handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"algo" : to_json(algo)}
    })
  };
  return res;
}

// 71
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
  static const std::string funName{"cudnnConvolutionBackwardData"};
  static auto orig_cudnnConvolutionBackwardData =
      (decltype(cudnnConvolutionBackwardData)) dlsym(RTLD_NEXT,
                                                     "cudnnConvolutionBackwardData");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "algo" : to_json(algo),
        "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes)
      }
    })
  };
  return res;
}

// 72
cudnnStatus_t CUDNNWINAPI cudnnIm2Col(cudnnHandle_t handle,
                                      const cudnnTensorDescriptor_t xDesc,
                                      const void *x,
                                      const cudnnFilterDescriptor_t wDesc,
                                      const cudnnConvolutionDescriptor_t convDesc,
                                      void *colBuffer) {
  static const std::string funName{"cudnnIm2Col"};
  static auto orig_cudnnIm2Col = (decltype(cudnnIm2Col)) dlsym(RTLD_NEXT, "cudnnIm2Col");
  const auto tic               = now();
  const auto res = orig_cudnnIm2Col(handle, xDesc, x, wDesc, convDesc, colBuffer);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 73
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxForward(cudnnHandle_t handle,
                                              cudnnSoftmaxAlgorithm_t algo,
                                              cudnnSoftmaxMode_t mode,
                                              const void *alpha,
                                              const cudnnTensorDescriptor_t xDesc,
                                              const void *x,
                                              const void *beta,
                                              const cudnnTensorDescriptor_t yDesc,
                                              void *y) {
  static const std::string funName{"cudnnSoftmaxForward"};
  static auto orig_cudnnSoftmaxForward =
      (decltype(cudnnSoftmaxForward)) dlsym(RTLD_NEXT, "cudnnSoftmaxForward");
  const auto tic = now();
  const auto res =
      orig_cudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"algo" : to_json(algo), "mode" : to_json(mode)}
    })
  };
  return res;
}

// 74
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
  static const std::string funName{"cudnnSoftmaxBackward"};
  static auto orig_cudnnSoftmaxBackward =
      (decltype(cudnnSoftmaxBackward)) dlsym(RTLD_NEXT, "cudnnSoftmaxBackward");
  const auto tic = now();
  const auto res = orig_cudnnSoftmaxBackward(
      handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"algo" : to_json(algo), "mode" : to_json(mode)}
    })
  };
  return res;
}

// 75
cudnnStatus_t CUDNNWINAPI
    cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc) {
  static const std::string funName{"cudnnCreatePoolingDescriptor"};
  static auto orig_cudnnCreatePoolingDescriptor =
      (decltype(cudnnCreatePoolingDescriptor)) dlsym(RTLD_NEXT,
                                                     "cudnnCreatePoolingDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreatePoolingDescriptor(poolingDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 76
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
  static const std::string funName{"cudnnSetPooling2dDescriptor"};
  static auto orig_cudnnSetPooling2dDescriptor =
      (decltype(cudnnSetPooling2dDescriptor)) dlsym(RTLD_NEXT,
                                                    "cudnnSetPooling2dDescriptor");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "mode" : to_json(mode),
        "maxpoolingNanOpt" : to_json(maxpoolingNanOpt),
        "windowHeight" : to_json(windowHeight),
        "windowWidth" : to_json(windowWidth),
        "verticalPadding" : to_json(verticalPadding),
        "horizontalPadding" : to_json(horizontalPadding),
        "verticalStride" : to_json(verticalStride),
        "horizontalStride" : to_json(horizontalStride)
      }
    })
  };
  return res;
}

// 77
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
  static const std::string funName{"cudnnGetPooling2dDescriptor"};
  static auto orig_cudnnGetPooling2dDescriptor =
      (decltype(cudnnGetPooling2dDescriptor)) dlsym(RTLD_NEXT,
                                                    "cudnnGetPooling2dDescriptor");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 78
cudnnStatus_t CUDNNWINAPI
    cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                const cudnnPoolingMode_t mode,
                                const cudnnNanPropagation_t maxpoolingNanOpt,
                                int nbDims,
                                const int windowDimA[],
                                const int paddingA[],
                                const int strideA[]) {
  static const std::string funName{"cudnnSetPoolingNdDescriptor"};
  static auto orig_cudnnSetPoolingNdDescriptor =
      (decltype(cudnnSetPoolingNdDescriptor)) dlsym(RTLD_NEXT,
                                                    "cudnnSetPoolingNdDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetPoolingNdDescriptor(
      poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "mode" : to_json(mode),
        "maxpoolingNanOpt" : to_json(maxpoolingNanOpt),
        "nbDims" : to_json(nbDims),
        "windowDimA" : to_json(windowDimA),
        "paddingA" : to_json(paddingA),
        "strideA" : to_json(strideA)
      }
    })
  };
  return res;
}

// 79
cudnnStatus_t CUDNNWINAPI
    cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t poolingDesc,
                                int nbDimsRequested,
                                cudnnPoolingMode_t *mode,
                                cudnnNanPropagation_t *maxpoolingNanOpt,
                                int *nbDims,
                                int windowDimA[],
                                int paddingA[],
                                int strideA[]) {
  static const std::string funName{"cudnnGetPoolingNdDescriptor"};
  static auto orig_cudnnGetPoolingNdDescriptor =
      (decltype(cudnnGetPoolingNdDescriptor)) dlsym(RTLD_NEXT,
                                                    "cudnnGetPoolingNdDescriptor");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "nbDimsRequested" : to_json(nbDimsRequested),
        "windowDimA" : to_json(windowDimA),
        "paddingA" : to_json(paddingA),
        "strideA" : to_json(strideA)
      }
    })
  };
  return res;
}

// 80
cudnnStatus_t CUDNNWINAPI
    cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                      const cudnnTensorDescriptor_t inputTensorDesc,
                                      int nbDims,
                                      int outputTensorDimA[]) {
  static const std::string funName{"cudnnGetPoolingNdForwardOutputDim"};
  static auto orig_cudnnGetPoolingNdForwardOutputDim =
      (decltype(cudnnGetPoolingNdForwardOutputDim)) dlsym(
          RTLD_NEXT, "cudnnGetPoolingNdForwardOutputDim");
  const auto tic = now();
  const auto res = orig_cudnnGetPoolingNdForwardOutputDim(
      poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" :
          {"nbDims" : to_json(nbDims), "outputTensorDimA" : to_json(outputTensorDimA)}
    })
  };
  return res;
}

// 81
cudnnStatus_t CUDNNWINAPI
    cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                      const cudnnTensorDescriptor_t inputTensorDesc,
                                      int *n,
                                      int *c,
                                      int *h,
                                      int *w) {
  static const std::string funName{"cudnnGetPooling2dForwardOutputDim"};
  static auto orig_cudnnGetPooling2dForwardOutputDim =
      (decltype(cudnnGetPooling2dForwardOutputDim)) dlsym(
          RTLD_NEXT, "cudnnGetPooling2dForwardOutputDim");
  const auto tic = now();
  const auto res =
      orig_cudnnGetPooling2dForwardOutputDim(poolingDesc, inputTensorDesc, n, c, h, w);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 82
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
  static const std::string funName{"cudnnDestroyPoolingDescriptor"};
  static auto orig_cudnnDestroyPoolingDescriptor =
      (decltype(cudnnDestroyPoolingDescriptor)) dlsym(RTLD_NEXT,
                                                      "cudnnDestroyPoolingDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyPoolingDescriptor(poolingDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 83
cudnnStatus_t CUDNNWINAPI cudnnPoolingForward(cudnnHandle_t handle,
                                              const cudnnPoolingDescriptor_t poolingDesc,
                                              const void *alpha,
                                              const cudnnTensorDescriptor_t xDesc,
                                              const void *x,
                                              const void *beta,
                                              const cudnnTensorDescriptor_t yDesc,
                                              void *y) {
  static const std::string funName{"cudnnPoolingForward"};
  static auto orig_cudnnPoolingForward =
      (decltype(cudnnPoolingForward)) dlsym(RTLD_NEXT, "cudnnPoolingForward");
  const auto tic = now();
  const auto res =
      orig_cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 84
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
  static const std::string funName{"cudnnPoolingBackward"};
  static auto orig_cudnnPoolingBackward =
      (decltype(cudnnPoolingBackward)) dlsym(RTLD_NEXT, "cudnnPoolingBackward");
  const auto tic = now();
  const auto res = orig_cudnnPoolingBackward(
      handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 85
cudnnStatus_t CUDNNWINAPI
    cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc) {
  static const std::string funName{"cudnnCreateActivationDescriptor"};
  static auto orig_cudnnCreateActivationDescriptor =
      (decltype(cudnnCreateActivationDescriptor)) dlsym(
          RTLD_NEXT, "cudnnCreateActivationDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateActivationDescriptor(activationDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 86
cudnnStatus_t CUDNNWINAPI
    cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                                 cudnnActivationMode_t mode,
                                 cudnnNanPropagation_t reluNanOpt,
                                 double coef) {
  static const std::string funName{"cudnnSetActivationDescriptor"};
  static auto orig_cudnnSetActivationDescriptor =
      (decltype(cudnnSetActivationDescriptor)) dlsym(RTLD_NEXT,
                                                     "cudnnSetActivationDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnSetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "mode" : to_json(mode),
        "reluNanOpt" : to_json(reluNanOpt),
        "coef" : to_json(coef)
      }
    })
  };
  return res;
}

// 87
cudnnStatus_t CUDNNWINAPI
    cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,
                                 cudnnActivationMode_t *mode,
                                 cudnnNanPropagation_t *reluNanOpt,
                                 double *coef) {
  static const std::string funName{"cudnnGetActivationDescriptor"};
  static auto orig_cudnnGetActivationDescriptor =
      (decltype(cudnnGetActivationDescriptor)) dlsym(RTLD_NEXT,
                                                     "cudnnGetActivationDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnGetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 88
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
  static const std::string funName{"cudnnDestroyActivationDescriptor"};
  static auto orig_cudnnDestroyActivationDescriptor =
      (decltype(cudnnDestroyActivationDescriptor)) dlsym(
          RTLD_NEXT, "cudnnDestroyActivationDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyActivationDescriptor(activationDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 89
cudnnStatus_t CUDNNWINAPI
    cudnnActivationForward(cudnnHandle_t handle,
                           cudnnActivationDescriptor_t activationDesc,
                           const void *alpha,
                           const cudnnTensorDescriptor_t xDesc,
                           const void *x,
                           const void *beta,
                           const cudnnTensorDescriptor_t yDesc,
                           void *y) {
  static const std::string funName{"cudnnActivationForward"};
  static auto orig_cudnnActivationForward =
      (decltype(cudnnActivationForward)) dlsym(RTLD_NEXT, "cudnnActivationForward");
  const auto tic = now();
  const auto res = orig_cudnnActivationForward(
      handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 90
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
  static const std::string funName{"cudnnActivationBackward"};
  static auto orig_cudnnActivationBackward =
      (decltype(cudnnActivationBackward)) dlsym(RTLD_NEXT, "cudnnActivationBackward");
  const auto tic = now();
  const auto res = orig_cudnnActivationBackward(
      handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 91
cudnnStatus_t CUDNNWINAPI cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc) {
  static const std::string funName{"cudnnCreateLRNDescriptor"};
  static auto orig_cudnnCreateLRNDescriptor =
      (decltype(cudnnCreateLRNDescriptor)) dlsym(RTLD_NEXT, "cudnnCreateLRNDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateLRNDescriptor(normDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 92
cudnnStatus_t CUDNNWINAPI cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
                                                unsigned lrnN,
                                                double lrnAlpha,
                                                double lrnBeta,
                                                double lrnK) {
  static const std::string funName{"cudnnSetLRNDescriptor"};
  static auto orig_cudnnSetLRNDescriptor =
      (decltype(cudnnSetLRNDescriptor)) dlsym(RTLD_NEXT, "cudnnSetLRNDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "lrnN" : to_json(lrnN),
        "lrnAlpha" : to_json(lrnAlpha),
        "lrnBeta" : to_json(lrnBeta),
        "lrnK" : to_json(lrnK)
      }
    })
  };
  return res;
}

// 93
cudnnStatus_t CUDNNWINAPI cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
                                                unsigned *lrnN,
                                                double *lrnAlpha,
                                                double *lrnBeta,
                                                double *lrnK) {
  static const std::string funName{"cudnnGetLRNDescriptor"};
  static auto orig_cudnnGetLRNDescriptor =
      (decltype(cudnnGetLRNDescriptor)) dlsym(RTLD_NEXT, "cudnnGetLRNDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 94
cudnnStatus_t CUDNNWINAPI cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc) {
  static const std::string funName{"cudnnDestroyLRNDescriptor"};
  static auto orig_cudnnDestroyLRNDescriptor =
      (decltype(cudnnDestroyLRNDescriptor)) dlsym(RTLD_NEXT, "cudnnDestroyLRNDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyLRNDescriptor(lrnDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 95
cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelForward(cudnnHandle_t handle,
                                                      cudnnLRNDescriptor_t normDesc,
                                                      cudnnLRNMode_t lrnMode,
                                                      const void *alpha,
                                                      const cudnnTensorDescriptor_t xDesc,
                                                      const void *x,
                                                      const void *beta,
                                                      const cudnnTensorDescriptor_t yDesc,
                                                      void *y) {
  static const std::string funName{"cudnnLRNCrossChannelForward"};
  static auto orig_cudnnLRNCrossChannelForward =
      (decltype(cudnnLRNCrossChannelForward)) dlsym(RTLD_NEXT,
                                                    "cudnnLRNCrossChannelForward");
  const auto tic = now();
  const auto res = orig_cudnnLRNCrossChannelForward(
      handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"lrnMode" : to_json(lrnMode)}
    })
  };
  return res;
}

// 96
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
  static const std::string funName{"cudnnLRNCrossChannelBackward"};
  static auto orig_cudnnLRNCrossChannelBackward =
      (decltype(cudnnLRNCrossChannelBackward)) dlsym(RTLD_NEXT,
                                                     "cudnnLRNCrossChannelBackward");
  const auto tic = now();
  const auto res = orig_cudnnLRNCrossChannelBackward(
      handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"lrnMode" : to_json(lrnMode)}
    })
  };
  return res;
}

// 97
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
  static const std::string funName{"cudnnDivisiveNormalizationForward"};
  static auto orig_cudnnDivisiveNormalizationForward =
      (decltype(cudnnDivisiveNormalizationForward)) dlsym(
          RTLD_NEXT, "cudnnDivisiveNormalizationForward");
  const auto tic = now();
  const auto res = orig_cudnnDivisiveNormalizationForward(
      handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"mode" : to_json(mode)}
    })
  };
  return res;
}

// 98
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
  static const std::string funName{"cudnnDivisiveNormalizationBackward"};
  static auto orig_cudnnDivisiveNormalizationBackward =
      (decltype(cudnnDivisiveNormalizationBackward)) dlsym(
          RTLD_NEXT, "cudnnDivisiveNormalizationBackward");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"mode" : to_json(mode)}
    })
  };
  return res;
}

// 99
cudnnStatus_t CUDNNWINAPI
    cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,
                                  const cudnnTensorDescriptor_t xDesc,
                                  cudnnBatchNormMode_t mode) {
  static const std::string funName{"cudnnDeriveBNTensorDescriptor"};
  static auto orig_cudnnDeriveBNTensorDescriptor =
      (decltype(cudnnDeriveBNTensorDescriptor)) dlsym(RTLD_NEXT,
                                                      "cudnnDeriveBNTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDeriveBNTensorDescriptor(derivedBnDesc, xDesc, mode);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"mode" : to_json(mode)}
    })
  };
  return res;
}

// 100
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
  static const std::string funName{"cudnnBatchNormalizationForwardTraining"};
  static auto orig_cudnnBatchNormalizationForwardTraining =
      (decltype(cudnnBatchNormalizationForwardTraining)) dlsym(
          RTLD_NEXT, "cudnnBatchNormalizationForwardTraining");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "mode" : to_json(mode),
        "exponentialAverageFactor" : to_json(exponentialAverageFactor),
        "epsilon" : to_json(epsilon)
      }
    })
  };
  return res;
}

// 101
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
  static const std::string funName{"cudnnBatchNormalizationForwardInference"};
  static auto orig_cudnnBatchNormalizationForwardInference =
      (decltype(cudnnBatchNormalizationForwardInference)) dlsym(
          RTLD_NEXT, "cudnnBatchNormalizationForwardInference");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"mode" : to_json(mode), "epsilon" : to_json(epsilon)}
    })
  };
  return res;
}

// 102
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
  static const std::string funName{"cudnnBatchNormalizationBackward"};
  static auto orig_cudnnBatchNormalizationBackward =
      (decltype(cudnnBatchNormalizationBackward)) dlsym(
          RTLD_NEXT, "cudnnBatchNormalizationBackward");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"mode" : to_json(mode), "epsilon" : to_json(epsilon)}
    })
  };
  return res;
}

// 103
cudnnStatus_t CUDNNWINAPI
    cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t *stDesc) {
  static const std::string funName{"cudnnCreateSpatialTransformerDescriptor"};
  static auto orig_cudnnCreateSpatialTransformerDescriptor =
      (decltype(cudnnCreateSpatialTransformerDescriptor)) dlsym(
          RTLD_NEXT, "cudnnCreateSpatialTransformerDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateSpatialTransformerDescriptor(stDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 104
cudnnStatus_t CUDNNWINAPI
    cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t stDesc,
                                           cudnnSamplerType_t samplerType,
                                           cudnnDataType_t dataType,
                                           const int nbDims,
                                           const int dimA[]) {
  static const std::string funName{"cudnnSetSpatialTransformerNdDescriptor"};
  static auto orig_cudnnSetSpatialTransformerNdDescriptor =
      (decltype(cudnnSetSpatialTransformerNdDescriptor)) dlsym(
          RTLD_NEXT, "cudnnSetSpatialTransformerNdDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetSpatialTransformerNdDescriptor(
      stDesc, samplerType, dataType, nbDims, dimA);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "samplerType" : to_json(samplerType),
        "dataType" : to_json(dataType),
        "nbDims" : to_json(nbDims),
        "dimA" : to_json(dimA)
      }
    })
  };
  return res;
}

// 105
cudnnStatus_t CUDNNWINAPI
    cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t stDesc) {
  static const std::string funName{"cudnnDestroySpatialTransformerDescriptor"};
  static auto orig_cudnnDestroySpatialTransformerDescriptor =
      (decltype(cudnnDestroySpatialTransformerDescriptor)) dlsym(
          RTLD_NEXT, "cudnnDestroySpatialTransformerDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroySpatialTransformerDescriptor(stDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 106
cudnnStatus_t CUDNNWINAPI
    cudnnSpatialTfGridGeneratorForward(cudnnHandle_t handle,
                                       const cudnnSpatialTransformerDescriptor_t stDesc,
                                       const void *theta,
                                       void *grid) {
  static const std::string funName{"cudnnSpatialTfGridGeneratorForward"};
  static auto orig_cudnnSpatialTfGridGeneratorForward =
      (decltype(cudnnSpatialTfGridGeneratorForward)) dlsym(
          RTLD_NEXT, "cudnnSpatialTfGridGeneratorForward");
  const auto tic = now();
  const auto res = orig_cudnnSpatialTfGridGeneratorForward(handle, stDesc, theta, grid);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 107
cudnnStatus_t CUDNNWINAPI
    cudnnSpatialTfGridGeneratorBackward(cudnnHandle_t handle,
                                        const cudnnSpatialTransformerDescriptor_t stDesc,
                                        const void *dgrid,
                                        void *dtheta) {
  static const std::string funName{"cudnnSpatialTfGridGeneratorBackward"};
  static auto orig_cudnnSpatialTfGridGeneratorBackward =
      (decltype(cudnnSpatialTfGridGeneratorBackward)) dlsym(
          RTLD_NEXT, "cudnnSpatialTfGridGeneratorBackward");
  const auto tic = now();
  const auto res =
      orig_cudnnSpatialTfGridGeneratorBackward(handle, stDesc, dgrid, dtheta);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 108
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
  static const std::string funName{"cudnnSpatialTfSamplerForward"};
  static auto orig_cudnnSpatialTfSamplerForward =
      (decltype(cudnnSpatialTfSamplerForward)) dlsym(RTLD_NEXT,
                                                     "cudnnSpatialTfSamplerForward");
  const auto tic = now();
  const auto res = orig_cudnnSpatialTfSamplerForward(
      handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 109
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
  static const std::string funName{"cudnnSpatialTfSamplerBackward"};
  static auto orig_cudnnSpatialTfSamplerBackward =
      (decltype(cudnnSpatialTfSamplerBackward)) dlsym(RTLD_NEXT,
                                                      "cudnnSpatialTfSamplerBackward");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 110
cudnnStatus_t CUDNNWINAPI
    cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc) {
  static const std::string funName{"cudnnCreateDropoutDescriptor"};
  static auto orig_cudnnCreateDropoutDescriptor =
      (decltype(cudnnCreateDropoutDescriptor)) dlsym(RTLD_NEXT,
                                                     "cudnnCreateDropoutDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateDropoutDescriptor(dropoutDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 111
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) {
  static const std::string funName{"cudnnDestroyDropoutDescriptor"};
  static auto orig_cudnnDestroyDropoutDescriptor =
      (decltype(cudnnDestroyDropoutDescriptor)) dlsym(RTLD_NEXT,
                                                      "cudnnDestroyDropoutDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyDropoutDescriptor(dropoutDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 112
cudnnStatus_t CUDNNWINAPI cudnnDropoutGetStatesSize(cudnnHandle_t handle,
                                                    size_t *sizeInBytes) {
  static const std::string funName{"cudnnDropoutGetStatesSize"};
  static auto orig_cudnnDropoutGetStatesSize =
      (decltype(cudnnDropoutGetStatesSize)) dlsym(RTLD_NEXT, "cudnnDropoutGetStatesSize");
  const auto tic = now();
  const auto res = orig_cudnnDropoutGetStatesSize(sizeInBytes);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 113
cudnnStatus_t CUDNNWINAPI cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc,
                                                          size_t *sizeInBytes) {
  static const std::string funName{"cudnnDropoutGetReserveSpaceSize"};
  static auto orig_cudnnDropoutGetReserveSpaceSize =
      (decltype(cudnnDropoutGetReserveSpaceSize)) dlsym(
          RTLD_NEXT, "cudnnDropoutGetReserveSpaceSize");
  const auto tic = now();
  const auto res = orig_cudnnDropoutGetReserveSpaceSize(sizeInBytes);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 114
cudnnStatus_t CUDNNWINAPI cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                                    cudnnHandle_t handle,
                                                    float dropout,
                                                    void *states,
                                                    size_t stateSizeInBytes,
                                                    unsigned long long seed) {
  static const std::string funName{"cudnnSetDropoutDescriptor"};
  static auto orig_cudnnSetDropoutDescriptor =
      (decltype(cudnnSetDropoutDescriptor)) dlsym(RTLD_NEXT, "cudnnSetDropoutDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetDropoutDescriptor(
      dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" :
          {"stateSizeInBytes" : to_json(stateSizeInBytes), "seed" : to_json(seed)}
    })
  };
  return res;
}

// 115
cudnnStatus_t CUDNNWINAPI
    cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                  cudnnHandle_t handle,
                                  float dropout,
                                  void *states,
                                  size_t stateSizeInBytes,
                                  unsigned long long seed) {
  static const std::string funName{"cudnnRestoreDropoutDescriptor"};
  static auto orig_cudnnRestoreDropoutDescriptor =
      (decltype(cudnnRestoreDropoutDescriptor)) dlsym(RTLD_NEXT,
                                                      "cudnnRestoreDropoutDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnRestoreDropoutDescriptor(
      dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" :
          {"stateSizeInBytes" : to_json(stateSizeInBytes), "seed" : to_json(seed)}
    })
  };
  return res;
}

// 116
cudnnStatus_t CUDNNWINAPI cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                                    cudnnHandle_t handle,
                                                    float *dropout,
                                                    void **states,
                                                    unsigned long long *seed) {
  static const std::string funName{"cudnnGetDropoutDescriptor"};
  static auto orig_cudnnGetDropoutDescriptor =
      (decltype(cudnnGetDropoutDescriptor)) dlsym(RTLD_NEXT, "cudnnGetDropoutDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnGetDropoutDescriptor(dropoutDesc, handle, dropout, states, seed);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 117
cudnnStatus_t CUDNNWINAPI cudnnDropoutForward(cudnnHandle_t handle,
                                              const cudnnDropoutDescriptor_t dropoutDesc,
                                              const cudnnTensorDescriptor_t xdesc,
                                              const void *x,
                                              const cudnnTensorDescriptor_t ydesc,
                                              void *y,
                                              void *reserveSpace,
                                              size_t reserveSpaceSizeInBytes) {
  static const std::string funName{"cudnnDropoutForward"};
  static auto orig_cudnnDropoutForward =
      (decltype(cudnnDropoutForward)) dlsym(RTLD_NEXT, "cudnnDropoutForward");
  const auto tic = now();
  const auto res = orig_cudnnDropoutForward(
      handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)}
    })
  };
  return res;
}

// 118
cudnnStatus_t CUDNNWINAPI cudnnDropoutBackward(cudnnHandle_t handle,
                                               const cudnnDropoutDescriptor_t dropoutDesc,
                                               const cudnnTensorDescriptor_t dydesc,
                                               const void *dy,
                                               const cudnnTensorDescriptor_t dxdesc,
                                               void *dx,
                                               void *reserveSpace,
                                               size_t reserveSpaceSizeInBytes) {
  static const std::string funName{"cudnnDropoutBackward"};
  static auto orig_cudnnDropoutBackward =
      (decltype(cudnnDropoutBackward)) dlsym(RTLD_NEXT, "cudnnDropoutBackward");
  const auto tic = now();
  const auto res = orig_cudnnDropoutBackward(
      handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)}
    })
  };
  return res;
}

// 119
cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc) {
  static const std::string funName{"cudnnCreateRNNDescriptor"};
  static auto orig_cudnnCreateRNNDescriptor =
      (decltype(cudnnCreateRNNDescriptor)) dlsym(RTLD_NEXT, "cudnnCreateRNNDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateRNNDescriptor(rnnDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 120
cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
  static const std::string funName{"cudnnDestroyRNNDescriptor"};
  static auto orig_cudnnDestroyRNNDescriptor =
      (decltype(cudnnDestroyRNNDescriptor)) dlsym(RTLD_NEXT, "cudnnDestroyRNNDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyRNNDescriptor(rnnDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 121
cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardInferenceAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  static const std::string funName{"cudnnGetRNNForwardInferenceAlgorithmMaxCount"};
  static auto orig_cudnnGetRNNForwardInferenceAlgorithmMaxCount =
      (decltype(cudnnGetRNNForwardInferenceAlgorithmMaxCount)) dlsym(
          RTLD_NEXT, "cudnnGetRNNForwardInferenceAlgorithmMaxCount");
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNForwardInferenceAlgorithmMaxCount(handle, rnnDesc, count);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 122
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
  static const std::string funName{"cudnnFindRNNForwardInferenceAlgorithmEx"};
  static auto orig_cudnnFindRNNForwardInferenceAlgorithmEx =
      (decltype(cudnnFindRNNForwardInferenceAlgorithmEx)) dlsym(
          RTLD_NEXT, "cudnnFindRNNForwardInferenceAlgorithmEx");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "seqLength" : to_json(seqLength),
        "findIntensity" : to_json(findIntensity),
        "requestedAlgoCount" : to_json(requestedAlgoCount),
        "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes)
      }
    })
  };
  return res;
}

// 123
cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardTrainingAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  static const std::string funName{"cudnnGetRNNForwardTrainingAlgorithmMaxCount"};
  static auto orig_cudnnGetRNNForwardTrainingAlgorithmMaxCount =
      (decltype(cudnnGetRNNForwardTrainingAlgorithmMaxCount)) dlsym(
          RTLD_NEXT, "cudnnGetRNNForwardTrainingAlgorithmMaxCount");
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNForwardTrainingAlgorithmMaxCount(handle, rnnDesc, count);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 124
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
  static const std::string funName{"cudnnFindRNNForwardTrainingAlgorithmEx"};
  static auto orig_cudnnFindRNNForwardTrainingAlgorithmEx =
      (decltype(cudnnFindRNNForwardTrainingAlgorithmEx)) dlsym(
          RTLD_NEXT, "cudnnFindRNNForwardTrainingAlgorithmEx");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "seqLength" : to_json(seqLength),
        "findIntensity" : to_json(findIntensity),
        "requestedAlgoCount" : to_json(requestedAlgoCount),
        "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
        "reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)
      }
    })
  };
  return res;
}

// 125
cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardDataAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  static const std::string funName{"cudnnGetRNNBackwardDataAlgorithmMaxCount"};
  static auto orig_cudnnGetRNNBackwardDataAlgorithmMaxCount =
      (decltype(cudnnGetRNNBackwardDataAlgorithmMaxCount)) dlsym(
          RTLD_NEXT, "cudnnGetRNNBackwardDataAlgorithmMaxCount");
  const auto tic = now();
  const auto res = orig_cudnnGetRNNBackwardDataAlgorithmMaxCount(handle, rnnDesc, count);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 126
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
  static const std::string funName{"cudnnFindRNNBackwardDataAlgorithmEx"};
  static auto orig_cudnnFindRNNBackwardDataAlgorithmEx =
      (decltype(cudnnFindRNNBackwardDataAlgorithmEx)) dlsym(
          RTLD_NEXT, "cudnnFindRNNBackwardDataAlgorithmEx");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "seqLength" : to_json(seqLength),
        "findIntensity" : to_json(findIntensity),
        "requestedAlgoCount" : to_json(requestedAlgoCount),
        "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
        "reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)
      }
    })
  };
  return res;
}

// 127
cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardWeightsAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  static const std::string funName{"cudnnGetRNNBackwardWeightsAlgorithmMaxCount"};
  static auto orig_cudnnGetRNNBackwardWeightsAlgorithmMaxCount =
      (decltype(cudnnGetRNNBackwardWeightsAlgorithmMaxCount)) dlsym(
          RTLD_NEXT, "cudnnGetRNNBackwardWeightsAlgorithmMaxCount");
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle, rnnDesc, count);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 128
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
  static const std::string funName{"cudnnFindRNNBackwardWeightsAlgorithmEx"};
  static auto orig_cudnnFindRNNBackwardWeightsAlgorithmEx =
      (decltype(cudnnFindRNNBackwardWeightsAlgorithmEx)) dlsym(
          RTLD_NEXT, "cudnnFindRNNBackwardWeightsAlgorithmEx");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "seqLength" : to_json(seqLength),
        "findIntensity" : to_json(findIntensity),
        "requestedAlgoCount" : to_json(requestedAlgoCount),
        "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
        "reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)
      }
    })
  };
  return res;
}

// 129
cudnnStatus_t CUDNNWINAPI cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
                                                       const int minibatch,
                                                       const cudnnDataType_t dataType,
                                                       cudnnPersistentRNNPlan_t *plan) {
  static const std::string funName{"cudnnCreatePersistentRNNPlan"};
  static auto orig_cudnnCreatePersistentRNNPlan =
      (decltype(cudnnCreatePersistentRNNPlan)) dlsym(RTLD_NEXT,
                                                     "cudnnCreatePersistentRNNPlan");
  const auto tic = now();
  const auto res = orig_cudnnCreatePersistentRNNPlan(rnnDesc, minibatch, dataType, plan);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"minibatch" : to_json(minibatch), "dataType" : to_json(dataType)}
    })
  };
  return res;
}

// 130
cudnnStatus_t CUDNNWINAPI cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
                                                    cudnnPersistentRNNPlan_t plan) {
  static const std::string funName{"cudnnSetPersistentRNNPlan"};
  static auto orig_cudnnSetPersistentRNNPlan =
      (decltype(cudnnSetPersistentRNNPlan)) dlsym(RTLD_NEXT, "cudnnSetPersistentRNNPlan");
  const auto tic = now();
  const auto res = orig_cudnnSetPersistentRNNPlan(rnnDesc, plan);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"plan" : to_json(plan)}
    })
  };
  return res;
}

// 131
cudnnStatus_t CUDNNWINAPI cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan) {
  static const std::string funName{"cudnnDestroyPersistentRNNPlan"};
  static auto orig_cudnnDestroyPersistentRNNPlan =
      (decltype(cudnnDestroyPersistentRNNPlan)) dlsym(RTLD_NEXT,
                                                      "cudnnDestroyPersistentRNNPlan");
  const auto tic = now();
  const auto res = orig_cudnnDestroyPersistentRNNPlan(plan);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"plan" : to_json(plan)}
    })
  };
  return res;
}

// 132
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
  static const std::string funName{"cudnnSetRNNDescriptor"};
  static auto orig_cudnnSetRNNDescriptor =
      (decltype(cudnnSetRNNDescriptor)) dlsym(RTLD_NEXT, "cudnnSetRNNDescriptor");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "hiddenSize" : to_json(hiddenSize),
        "numLayers" : to_json(numLayers),
        "inputMode" : to_json(inputMode),
        "direction" : to_json(direction),
        "mode" : to_json(mode),
        "algo" : to_json(algo),
        "dataType" : to_json(dataType)
      }
    })
  };
  return res;
}

// 133
cudnnStatus_t CUDNNWINAPI cudnnSetRNNProjectionLayers(cudnnHandle_t handle,
                                                      cudnnRNNDescriptor_t rnnDesc,
                                                      const int recProjSize,
                                                      const int outProjSize) {
  static const std::string funName{"cudnnSetRNNProjectionLayers"};
  static auto orig_cudnnSetRNNProjectionLayers =
      (decltype(cudnnSetRNNProjectionLayers)) dlsym(RTLD_NEXT,
                                                    "cudnnSetRNNProjectionLayers");
  const auto tic = now();
  const auto res =
      orig_cudnnSetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" :
          {"recProjSize" : to_json(recProjSize), "outProjSize" : to_json(outProjSize)}
    })
  };
  return res;
}

// 134
cudnnStatus_t CUDNNWINAPI cudnnGetRNNProjectionLayers(cudnnHandle_t handle,
                                                      const cudnnRNNDescriptor_t rnnDesc,
                                                      int *recProjSize,
                                                      int *outProjSize) {
  static const std::string funName{"cudnnGetRNNProjectionLayers"};
  static auto orig_cudnnGetRNNProjectionLayers =
      (decltype(cudnnGetRNNProjectionLayers)) dlsym(RTLD_NEXT,
                                                    "cudnnGetRNNProjectionLayers");
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 135
cudnnStatus_t CUDNNWINAPI
    cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t handle,
                                   cudnnRNNDescriptor_t rnnDesc,
                                   cudnnAlgorithmDescriptor_t algoDesc) {
  static const std::string funName{"cudnnSetRNNAlgorithmDescriptor"};
  static auto orig_cudnnSetRNNAlgorithmDescriptor =
      (decltype(cudnnSetRNNAlgorithmDescriptor)) dlsym(RTLD_NEXT,
                                                       "cudnnSetRNNAlgorithmDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetRNNAlgorithmDescriptor(handle, rnnDesc, algoDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 136
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
  static const std::string funName{"cudnnGetRNNDescriptor"};
  static auto orig_cudnnGetRNNDescriptor =
      (decltype(cudnnGetRNNDescriptor)) dlsym(RTLD_NEXT, "cudnnGetRNNDescriptor");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 137
cudnnStatus_t CUDNNWINAPI cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc,
                                                    cudnnMathType_t mType) {
  static const std::string funName{"cudnnSetRNNMatrixMathType"};
  static auto orig_cudnnSetRNNMatrixMathType =
      (decltype(cudnnSetRNNMatrixMathType)) dlsym(RTLD_NEXT, "cudnnSetRNNMatrixMathType");
  const auto tic = now();
  const auto res = orig_cudnnSetRNNMatrixMathType(mType);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"mType" : to_json(mType)}
    })
  };
  return res;
}

// 138
cudnnStatus_t CUDNNWINAPI cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc,
                                                    cudnnMathType_t *mType) {
  static const std::string funName{"cudnnGetRNNMatrixMathType"};
  static auto orig_cudnnGetRNNMatrixMathType =
      (decltype(cudnnGetRNNMatrixMathType)) dlsym(RTLD_NEXT, "cudnnGetRNNMatrixMathType");
  const auto tic = now();
  const auto res = orig_cudnnGetRNNMatrixMathType(mType);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 139
cudnnStatus_t CUDNNWINAPI cudnnGetRNNWorkspaceSize(cudnnHandle_t handle,
                                                   const cudnnRNNDescriptor_t rnnDesc,
                                                   const int seqLength,
                                                   const cudnnTensorDescriptor_t *xDesc,
                                                   size_t *sizeInBytes) {
  static const std::string funName{"cudnnGetRNNWorkspaceSize"};
  static auto orig_cudnnGetRNNWorkspaceSize =
      (decltype(cudnnGetRNNWorkspaceSize)) dlsym(RTLD_NEXT, "cudnnGetRNNWorkspaceSize");
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"seqLength" : to_json(seqLength)}
    })
  };
  return res;
}

// 140
cudnnStatus_t CUDNNWINAPI
    cudnnGetRNNTrainingReserveSize(cudnnHandle_t handle,
                                   const cudnnRNNDescriptor_t rnnDesc,
                                   const int seqLength,
                                   const cudnnTensorDescriptor_t *xDesc,
                                   size_t *sizeInBytes) {
  static const std::string funName{"cudnnGetRNNTrainingReserveSize"};
  static auto orig_cudnnGetRNNTrainingReserveSize =
      (decltype(cudnnGetRNNTrainingReserveSize)) dlsym(RTLD_NEXT,
                                                       "cudnnGetRNNTrainingReserveSize");
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNTrainingReserveSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"seqLength" : to_json(seqLength)}
    })
  };
  return res;
}

// 141
cudnnStatus_t CUDNNWINAPI cudnnGetRNNParamsSize(cudnnHandle_t handle,
                                                const cudnnRNNDescriptor_t rnnDesc,
                                                const cudnnTensorDescriptor_t xDesc,
                                                size_t *sizeInBytes,
                                                cudnnDataType_t dataType) {
  static const std::string funName{"cudnnGetRNNParamsSize"};
  static auto orig_cudnnGetRNNParamsSize =
      (decltype(cudnnGetRNNParamsSize)) dlsym(RTLD_NEXT, "cudnnGetRNNParamsSize");
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNParamsSize(handle, rnnDesc, xDesc, sizeInBytes, dataType);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"dataType" : to_json(dataType)}
    })
  };
  return res;
}

// 142
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
  static const std::string funName{"cudnnGetRNNLinLayerMatrixParams"};
  static auto orig_cudnnGetRNNLinLayerMatrixParams =
      (decltype(cudnnGetRNNLinLayerMatrixParams)) dlsym(
          RTLD_NEXT, "cudnnGetRNNLinLayerMatrixParams");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" :
          {"pseudoLayer" : to_json(pseudoLayer), "linLayerID" : to_json(linLayerID)}
    })
  };
  return res;
}

// 143
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
  static const std::string funName{"cudnnGetRNNLinLayerBiasParams"};
  static auto orig_cudnnGetRNNLinLayerBiasParams =
      (decltype(cudnnGetRNNLinLayerBiasParams)) dlsym(RTLD_NEXT,
                                                      "cudnnGetRNNLinLayerBiasParams");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" :
          {"pseudoLayer" : to_json(pseudoLayer), "linLayerID" : to_json(linLayerID)}
    })
  };
  return res;
}

// 144
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
  static const std::string funName{"cudnnRNNForwardInference"};
  static auto orig_cudnnRNNForwardInference =
      (decltype(cudnnRNNForwardInference)) dlsym(RTLD_NEXT, "cudnnRNNForwardInference");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "seqLength" : to_json(seqLength),
        "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes)
      }
    })
  };
  return res;
}

// 145
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
  static const std::string funName{"cudnnRNNForwardTraining"};
  static auto orig_cudnnRNNForwardTraining =
      (decltype(cudnnRNNForwardTraining)) dlsym(RTLD_NEXT, "cudnnRNNForwardTraining");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "seqLength" : to_json(seqLength),
        "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
        "reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)
      }
    })
  };
  return res;
}

// 146
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
  static const std::string funName{"cudnnRNNBackwardData"};
  static auto orig_cudnnRNNBackwardData =
      (decltype(cudnnRNNBackwardData)) dlsym(RTLD_NEXT, "cudnnRNNBackwardData");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "seqLength" : to_json(seqLength),
        "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
        "reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)
      }
    })
  };
  return res;
}

// 147
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
  static const std::string funName{"cudnnRNNBackwardWeights"};
  static auto orig_cudnnRNNBackwardWeights =
      (decltype(cudnnRNNBackwardWeights)) dlsym(RTLD_NEXT, "cudnnRNNBackwardWeights");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "seqLength" : to_json(seqLength),
        "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
        "reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)
      }
    })
  };
  return res;
}

// 148
cudnnStatus_t CUDNNWINAPI
    cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t *ctcLossDesc) {
  static const std::string funName{"cudnnCreateCTCLossDescriptor"};
  static auto orig_cudnnCreateCTCLossDescriptor =
      (decltype(cudnnCreateCTCLossDescriptor)) dlsym(RTLD_NEXT,
                                                     "cudnnCreateCTCLossDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateCTCLossDescriptor(ctcLossDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 149
cudnnStatus_t CUDNNWINAPI cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc,
                                                    cudnnDataType_t compType) {
  static const std::string funName{"cudnnSetCTCLossDescriptor"};
  static auto orig_cudnnSetCTCLossDescriptor =
      (decltype(cudnnSetCTCLossDescriptor)) dlsym(RTLD_NEXT, "cudnnSetCTCLossDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetCTCLossDescriptor(ctcLossDesc, compType);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"compType" : to_json(compType)}
    })
  };
  return res;
}

// 150
cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc,
                                                    cudnnDataType_t *compType) {
  static const std::string funName{"cudnnGetCTCLossDescriptor"};
  static auto orig_cudnnGetCTCLossDescriptor =
      (decltype(cudnnGetCTCLossDescriptor)) dlsym(RTLD_NEXT, "cudnnGetCTCLossDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetCTCLossDescriptor(ctcLossDesc, compType);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 151
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc) {
  static const std::string funName{"cudnnDestroyCTCLossDescriptor"};
  static auto orig_cudnnDestroyCTCLossDescriptor =
      (decltype(cudnnDestroyCTCLossDescriptor)) dlsym(RTLD_NEXT,
                                                      "cudnnDestroyCTCLossDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyCTCLossDescriptor(ctcLossDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 152
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
  static const std::string funName{"cudnnCTCLoss"};
  static auto orig_cudnnCTCLoss =
      (decltype(cudnnCTCLoss)) dlsym(RTLD_NEXT, "cudnnCTCLoss");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes)}
    })
  };
  return res;
}

// 153
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
  static const std::string funName{"cudnnGetCTCLossWorkspaceSize"};
  static auto orig_cudnnGetCTCLossWorkspaceSize =
      (decltype(cudnnGetCTCLossWorkspaceSize)) dlsym(RTLD_NEXT,
                                                     "cudnnGetCTCLossWorkspaceSize");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 154
cudnnStatus_t CUDNNWINAPI
    cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t *algoDesc) {
  static const std::string funName{"cudnnCreateAlgorithmDescriptor"};
  static auto orig_cudnnCreateAlgorithmDescriptor =
      (decltype(cudnnCreateAlgorithmDescriptor)) dlsym(RTLD_NEXT,
                                                       "cudnnCreateAlgorithmDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateAlgorithmDescriptor(algoDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 155
cudnnStatus_t CUDNNWINAPI cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc,
                                                      cudnnAlgorithm_t algorithm) {
  static const std::string funName{"cudnnSetAlgorithmDescriptor"};
  static auto orig_cudnnSetAlgorithmDescriptor =
      (decltype(cudnnSetAlgorithmDescriptor)) dlsym(RTLD_NEXT,
                                                    "cudnnSetAlgorithmDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetAlgorithmDescriptor(algoDesc, algorithm);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"algorithm" : to_json(algorithm)}
    })
  };
  return res;
}

// 156
cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmDescriptor(
    const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algorithm) {
  static const std::string funName{"cudnnGetAlgorithmDescriptor"};
  static auto orig_cudnnGetAlgorithmDescriptor =
      (decltype(cudnnGetAlgorithmDescriptor)) dlsym(RTLD_NEXT,
                                                    "cudnnGetAlgorithmDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetAlgorithmDescriptor(algoDesc, algorithm);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 157
cudnnStatus_t CUDNNWINAPI cudnnCopyAlgorithmDescriptor(
    const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest) {
  static const std::string funName{"cudnnCopyAlgorithmDescriptor"};
  static auto orig_cudnnCopyAlgorithmDescriptor =
      (decltype(cudnnCopyAlgorithmDescriptor)) dlsym(RTLD_NEXT,
                                                     "cudnnCopyAlgorithmDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCopyAlgorithmDescriptor(src, dest);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 158
cudnnStatus_t CUDNNWINAPI
    cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc) {
  static const std::string funName{"cudnnDestroyAlgorithmDescriptor"};
  static auto orig_cudnnDestroyAlgorithmDescriptor =
      (decltype(cudnnDestroyAlgorithmDescriptor)) dlsym(
          RTLD_NEXT, "cudnnDestroyAlgorithmDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyAlgorithmDescriptor(algoDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 159
cudnnStatus_t CUDNNWINAPI cudnnCreateAlgorithmPerformance(
    cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate) {
  static const std::string funName{"cudnnCreateAlgorithmPerformance"};
  static auto orig_cudnnCreateAlgorithmPerformance =
      (decltype(cudnnCreateAlgorithmPerformance)) dlsym(
          RTLD_NEXT, "cudnnCreateAlgorithmPerformance");
  const auto tic = now();
  const auto res = orig_cudnnCreateAlgorithmPerformance(algoPerf, numberToCreate);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"numberToCreate" : to_json(numberToCreate)}
    })
  };
  return res;
}

// 160
cudnnStatus_t CUDNNWINAPI
    cudnnSetAlgorithmPerformance(cudnnAlgorithmPerformance_t algoPerf,
                                 cudnnAlgorithmDescriptor_t algoDesc,
                                 cudnnStatus_t status,
                                 float time,
                                 size_t memory) {
  static const std::string funName{"cudnnSetAlgorithmPerformance"};
  static auto orig_cudnnSetAlgorithmPerformance =
      (decltype(cudnnSetAlgorithmPerformance)) dlsym(RTLD_NEXT,
                                                     "cudnnSetAlgorithmPerformance");
  const auto tic = now();
  const auto res =
      orig_cudnnSetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "algoPerf" : to_json(algoPerf),
        "status" : to_json(status),
        "time" : to_json(time),
        "memory" : to_json(memory)
      }
    })
  };
  return res;
}

// 161
cudnnStatus_t CUDNNWINAPI
    cudnnGetAlgorithmPerformance(const cudnnAlgorithmPerformance_t algoPerf,
                                 cudnnAlgorithmDescriptor_t *algoDesc,
                                 cudnnStatus_t *status,
                                 float *time,
                                 size_t *memory) {
  static const std::string funName{"cudnnGetAlgorithmPerformance"};
  static auto orig_cudnnGetAlgorithmPerformance =
      (decltype(cudnnGetAlgorithmPerformance)) dlsym(RTLD_NEXT,
                                                     "cudnnGetAlgorithmPerformance");
  const auto tic = now();
  const auto res =
      orig_cudnnGetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"algoPerf" : to_json(algoPerf)}
    })
  };
  return res;
}

// 162
cudnnStatus_t CUDNNWINAPI cudnnDestroyAlgorithmPerformance(
    cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy) {
  static const std::string funName{"cudnnDestroyAlgorithmPerformance"};
  static auto orig_cudnnDestroyAlgorithmPerformance =
      (decltype(cudnnDestroyAlgorithmPerformance)) dlsym(
          RTLD_NEXT, "cudnnDestroyAlgorithmPerformance");
  const auto tic = now();
  const auto res = orig_cudnnDestroyAlgorithmPerformance(algoPerf, numberToDestroy);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"numberToDestroy" : to_json(numberToDestroy)}
    })
  };
  return res;
}

// 163
cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmSpaceSize(cudnnHandle_t handle,
                                                     cudnnAlgorithmDescriptor_t algoDesc,
                                                     size_t *algoSpaceSizeInBytes) {
  static const std::string funName{"cudnnGetAlgorithmSpaceSize"};
  static auto orig_cudnnGetAlgorithmSpaceSize =
      (decltype(cudnnGetAlgorithmSpaceSize)) dlsym(RTLD_NEXT,
                                                   "cudnnGetAlgorithmSpaceSize");
  const auto tic = now();
  const auto res =
      orig_cudnnGetAlgorithmSpaceSize(handle, algoDesc, algoSpaceSizeInBytes);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 164
cudnnStatus_t CUDNNWINAPI cudnnSaveAlgorithm(cudnnHandle_t handle,
                                             cudnnAlgorithmDescriptor_t algoDesc,
                                             void *algoSpace,
                                             size_t algoSpaceSizeInBytes) {
  static const std::string funName{"cudnnSaveAlgorithm"};
  static auto orig_cudnnSaveAlgorithm =
      (decltype(cudnnSaveAlgorithm)) dlsym(RTLD_NEXT, "cudnnSaveAlgorithm");
  const auto tic = now();
  const auto res =
      orig_cudnnSaveAlgorithm(handle, algoDesc, algoSpace, algoSpaceSizeInBytes);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"algoSpaceSizeInBytes" : to_json(algoSpaceSizeInBytes)}
    })
  };
  return res;
}

// 165
cudnnStatus_t CUDNNWINAPI cudnnRestoreAlgorithm(cudnnHandle_t handle,
                                                void *algoSpace,
                                                size_t algoSpaceSizeInBytes,
                                                cudnnAlgorithmDescriptor_t algoDesc) {
  static const std::string funName{"cudnnRestoreAlgorithm"};
  static auto orig_cudnnRestoreAlgorithm =
      (decltype(cudnnRestoreAlgorithm)) dlsym(RTLD_NEXT, "cudnnRestoreAlgorithm");
  const auto tic = now();
  const auto res =
      orig_cudnnRestoreAlgorithm(handle, algoSpace, algoSpaceSizeInBytes, algoDesc);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"algoSpaceSizeInBytes" : to_json(algoSpaceSizeInBytes)}
    })
  };
  return res;
}

// 166
cudnnStatus_t CUDNNWINAPI cudnnSetCallback(unsigned mask,
                                           void *udata,
                                           cudnnCallback_t fptr) {
  static const std::string funName{"cudnnSetCallback"};
  static auto orig_cudnnSetCallback =
      (decltype(cudnnSetCallback)) dlsym(RTLD_NEXT, "cudnnSetCallback");
  const auto tic = now();
  const auto res = orig_cudnnSetCallback(mask, udata, fptr);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {"mask" : to_json(mask), "fptr" : to_json(fptr)}
    })
  };
  return res;
}

// 167
cudnnStatus_t CUDNNWINAPI cudnnGetCallback(unsigned *mask,
                                           void **udata,
                                           cudnnCallback_t *fptr) {
  static const std::string funName{"cudnnGetCallback"};
  static auto orig_cudnnGetCallback =
      (decltype(cudnnGetCallback)) dlsym(RTLD_NEXT, "cudnnGetCallback");
  const auto tic = now();
  const auto res = orig_cudnnGetCallback(mask, udata, fptr);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {

      }
    })
  };
  return res;
}

// 168
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
  static const std::string funName{"cudnnSetRNNDescriptor_v6"};
  static auto orig_cudnnSetRNNDescriptor_v6 =
      (decltype(cudnnSetRNNDescriptor_v6)) dlsym(RTLD_NEXT, "cudnnSetRNNDescriptor_v6");
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

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "hiddenSize" : to_json(hiddenSize),
        "numLayers" : to_json(numLayers),
        "inputMode" : to_json(inputMode),
        "direction" : to_json(direction),
        "mode" : to_json(mode),
        "algo" : to_json(algo),
        "dataType" : to_json(dataType)
      }
    })
  };
  return res;
}

// 169
cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v5(cudnnRNNDescriptor_t rnnDesc,
                                                   int hiddenSize,
                                                   int numLayers,
                                                   cudnnDropoutDescriptor_t dropoutDesc,
                                                   cudnnRNNInputMode_t inputMode,
                                                   cudnnDirectionMode_t direction,
                                                   cudnnRNNMode_t mode,
                                                   cudnnDataType_t dataType) {
  static const std::string funName{"cudnnSetRNNDescriptor_v5"};
  static auto orig_cudnnSetRNNDescriptor_v5 =
      (decltype(cudnnSetRNNDescriptor_v5)) dlsym(RTLD_NEXT, "cudnnSetRNNDescriptor_v5");
  const auto tic = now();
  const auto res = orig_cudnnSetRNNDescriptor_v5(
      rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, dataType);
  const auto toc = now();

  if (record_time_enabled()) {
    record_cudnn_time({
      "funName" : funName,
      "start" : tic,
      "end" : toc,
      "arguments" : {
        "hiddenSize" : to_json(hiddenSize),
        "numLayers" : to_json(numLayers),
        "inputMode" : to_json(inputMode),
        "direction" : to_json(direction),
        "mode" : to_json(mode),
        "dataType" : to_json(dataType)
      }
    })
  };
  return res;
}
