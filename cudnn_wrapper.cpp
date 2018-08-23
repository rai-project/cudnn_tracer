#include "cudnn.h"
#include "utils.hpp"
#include <dlfcn.h>

// 0
cudnnStatus_t CUDNNWINAPI cudnnQueryRuntimeError(cudnnHandle_t handle,
                                                 cudnnStatus_t *rstatus,
                                                 cudnnErrQueryMode_t mode,
                                                 cudnnRuntimeTag_t *tag) {
  static auto orig_cudnnQueryRuntimeError =
      (decltype(cudnnQueryRuntimeError))dlsym(RTLD_NEXT,
                                              "cudnnQueryRuntimeError");
  const auto tic = now();
  const auto res = orig_cudnnQueryRuntimeError(handle, rstatus, mode, tag);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnQueryRuntimeError"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rstatus" : to_json(rstatus),
    "mode" : to_json(mode),
    "tag" : to_json(tag)
  });
  return res;
}

// 1
cudnnStatus_t CUDNNWINAPI cudnnGetProperty(libraryPropertyType type,
                                           int *value) {
  static auto orig_cudnnGetProperty =
      (decltype(cudnnGetProperty))dlsym(RTLD_NEXT, "cudnnGetProperty");
  const auto tic = now();
  const auto res = orig_cudnnGetProperty(value);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetProperty"},
    "start" : tic,
    "end" : toc,
    "value" : to_json(value)
  });
  return res;
}

// 2
cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t *handle) {
  static auto orig_cudnnCreate =
      (decltype(cudnnCreate))dlsym(RTLD_NEXT, "cudnnCreate");
  const auto tic = now();
  const auto res = orig_cudnnCreate(handle);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreate"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle)
  });
  return res;
}

// 3
cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t handle) {
  static auto orig_cudnnDestroy =
      (decltype(cudnnDestroy))dlsym(RTLD_NEXT, "cudnnDestroy");
  const auto tic = now();
  const auto res = orig_cudnnDestroy(handle);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroy"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle)
  });
  return res;
}

// 4
cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t handle,
                                         cudaStream_t streamId) {
  static auto orig_cudnnSetStream =
      (decltype(cudnnSetStream))dlsym(RTLD_NEXT, "cudnnSetStream");
  const auto tic = now();
  const auto res = orig_cudnnSetStream(streamId);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetStream"},
    "start" : tic,
    "end" : toc,
    "streamId" : to_json(streamId)
  });
  return res;
}

// 5
cudnnStatus_t CUDNNWINAPI cudnnGetStream(cudnnHandle_t handle,
                                         cudaStream_t *streamId) {
  static auto orig_cudnnGetStream =
      (decltype(cudnnGetStream))dlsym(RTLD_NEXT, "cudnnGetStream");
  const auto tic = now();
  const auto res = orig_cudnnGetStream(streamId);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetStream"},
    "start" : tic,
    "end" : toc,
    "streamId" : to_json(streamId)
  });
  return res;
}

// 6
cudnnStatus_t CUDNNWINAPI
cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
  static auto orig_cudnnCreateTensorDescriptor =
      (decltype(cudnnCreateTensorDescriptor))dlsym(
          RTLD_NEXT, "cudnnCreateTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateTensorDescriptor(tensorDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreateTensorDescriptor"},
    "start" : tic,
    "end" : toc,
    "tensorDesc" : to_json(tensorDesc)
  });
  return res;
}

// 7
cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor(
    cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, /* image data type */
    int n,                    /* number of inputs (batch size) */
    int c,                    /* number of input feature maps */
    int h,                    /* height of input section */
    int w) {
  static auto orig_cudnnSetTensor4dDescriptor =
      (decltype(cudnnSetTensor4dDescriptor))dlsym(RTLD_NEXT,
                                                  "cudnnSetTensor4dDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetTensor4dDescriptor(size);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetTensor4dDescriptor"},
    "start" : tic,
    "end" : toc,
    "size" : to_json(size)
  });
  return res;
}

// 8
cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptorEx(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t dataType, /* image data type */
    int n,                    /* number of inputs (batch size) */
    int c,                    /* number of input feature maps */
    int h,                    /* height of input section */
    int w,                    /* width of input section */
    int nStride, int cStride, int hStride, int wStride) {
  static auto orig_cudnnSetTensor4dDescriptorEx =
      (decltype(cudnnSetTensor4dDescriptorEx))dlsym(
          RTLD_NEXT, "cudnnSetTensor4dDescriptorEx");
  const auto tic = now();
  const auto res = orig_cudnnSetTensor4dDescriptorEx(size);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetTensor4dDescriptorEx"},
    "start" : tic,
    "end" : toc,
    "size" : to_json(size)
  });
  return res;
}

// 9
cudnnStatus_t CUDNNWINAPI cudnnGetTensor4dDescriptor(
    const cudnnTensorDescriptor_t tensorDesc,
    cudnnDataType_t *dataType, /* image data type */
    int *n,                    /* number of inputs (batch size) */
    int *c,                    /* number of input feature maps  */
    int *h,                    /* height of input section */
    int *w,                    /* width of input section */
    int *nStride, int *cStride, int *hStride, int *wStride) {
  static auto orig_cudnnGetTensor4dDescriptor =
      (decltype(cudnnGetTensor4dDescriptor))dlsym(RTLD_NEXT,
                                                  "cudnnGetTensor4dDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetTensor4dDescriptor(size);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetTensor4dDescriptor"},
    "start" : tic,
    "end" : toc,
    "size" : to_json(size)
  });
  return res;
}

// 10
cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(
    cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int nbDims,
    const int dimA[], const int strideA[]) {
  static auto orig_cudnnSetTensorNdDescriptor =
      (decltype(cudnnSetTensorNdDescriptor))dlsym(RTLD_NEXT,
                                                  "cudnnSetTensorNdDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims,
                                                   dimA, strideA);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetTensorNdDescriptor"},
    "start" : tic,
    "end" : toc,
    "tensorDesc" : to_json(tensorDesc),
    "dataType" : to_json(dataType),
    "nbDims" : to_json(nbDims),
    "dimA" : to_json(dimA),
    "strideA" : to_json(strideA)
  });
  return res;
}

// 11
cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptorEx(
    cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, int nbDims, const int dimA[]) {
  static auto orig_cudnnSetTensorNdDescriptorEx =
      (decltype(cudnnSetTensorNdDescriptorEx))dlsym(
          RTLD_NEXT, "cudnnSetTensorNdDescriptorEx");
  const auto tic = now();
  const auto res = orig_cudnnSetTensorNdDescriptorEx(tensorDesc, format,
                                                     dataType, nbDims, dimA);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetTensorNdDescriptorEx"},
    "start" : tic,
    "end" : toc,
    "tensorDesc" : to_json(tensorDesc),
    "format" : to_json(format),
    "dataType" : to_json(dataType),
    "nbDims" : to_json(nbDims),
    "dimA" : to_json(dimA)
  });
  return res;
}

// 12
cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(
    const cudnnTensorDescriptor_t tensorDesc, int nbDimsRequested,
    cudnnDataType_t *dataType, int *nbDims, int dimA[], int strideA[]) {
  static auto orig_cudnnGetTensorNdDescriptor =
      (decltype(cudnnGetTensorNdDescriptor))dlsym(RTLD_NEXT,
                                                  "cudnnGetTensorNdDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetTensorNdDescriptor(
      tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetTensorNdDescriptor"},
    "start" : tic,
    "end" : toc,
    "tensorDesc" : to_json(tensorDesc),
    "nbDimsRequested" : to_json(nbDimsRequested),
    "dataType" : to_json(dataType),
    "nbDims" : to_json(nbDims),
    "dimA" : to_json(dimA),
    "strideA" : to_json(strideA)
  });
  return res;
}

// 13
cudnnStatus_t CUDNNWINAPI cudnnGetTensorSizeInBytes(
    const cudnnTensorDescriptor_t tensorDesc, size_t *size) {
  static auto orig_cudnnGetTensorSizeInBytes =
      (decltype(cudnnGetTensorSizeInBytes))dlsym(RTLD_NEXT,
                                                 "cudnnGetTensorSizeInBytes");
  const auto tic = now();
  const auto res = orig_cudnnGetTensorSizeInBytes(tensorDesc, size);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetTensorSizeInBytes"},
    "start" : tic,
    "end" : toc,
    "tensorDesc" : to_json(tensorDesc),
    "size" : to_json(size)
  });
  return res;
}

// 14
cudnnStatus_t CUDNNWINAPI
cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
  static auto orig_cudnnDestroyTensorDescriptor =
      (decltype(cudnnDestroyTensorDescriptor))dlsym(
          RTLD_NEXT, "cudnnDestroyTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyTensorDescriptor(tensorDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroyTensorDescriptor"},
    "start" : tic,
    "end" : toc,
    "tensorDesc" : to_json(tensorDesc)
  });
  return res;
}

// 15
cudnnStatus_t CUDNNWINAPI cudnnTransformTensor(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y) {
  static auto orig_cudnnTransformTensor =
      (decltype(cudnnTransformTensor))dlsym(RTLD_NEXT, "cudnnTransformTensor");
  const auto tic = now();
  const auto res =
      orig_cudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnTransformTensor"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "alpha" : to_json(alpha),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "beta" : to_json(beta),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y)
  });
  return res;
}

// 16
cudnnStatus_t CUDNNWINAPI cudnnAddTensor(cudnnHandle_t handle,
                                         const void *alpha,
                                         const cudnnTensorDescriptor_t aDesc,
                                         const void *A, const void *beta,
                                         const cudnnTensorDescriptor_t cDesc,
                                         void *C) {
  static auto orig_cudnnAddTensor =
      (decltype(cudnnAddTensor))dlsym(RTLD_NEXT, "cudnnAddTensor");
  const auto tic = now();
  const auto res = orig_cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnAddTensor"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "alpha" : to_json(alpha),
    "aDesc" : to_json(aDesc),
    "A" : to_json(A),
    "beta" : to_json(beta),
    "cDesc" : to_json(cDesc),
    "C" : to_json(C)
  });
  return res;
}

// 17
cudnnStatus_t CUDNNWINAPI
cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t *opTensorDesc) {
  static auto orig_cudnnCreateOpTensorDescriptor =
      (decltype(cudnnCreateOpTensorDescriptor))dlsym(
          RTLD_NEXT, "cudnnCreateOpTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateOpTensorDescriptor(opTensorDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreateOpTensorDescriptor"},
    "start" : tic,
    "end" : toc,
    "opTensorDesc" : to_json(opTensorDesc)
  });
  return res;
}

// 18
cudnnStatus_t CUDNNWINAPI cudnnSetOpTensorDescriptor(
    cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t opTensorOp,
    cudnnDataType_t opTensorCompType, cudnnNanPropagation_t opTensorNanOpt) {
  static auto orig_cudnnSetOpTensorDescriptor =
      (decltype(cudnnSetOpTensorDescriptor))dlsym(RTLD_NEXT,
                                                  "cudnnSetOpTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetOpTensorDescriptor(
      opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetOpTensorDescriptor"},
    "start" : tic,
    "end" : toc,
    "opTensorDesc" : to_json(opTensorDesc),
    "opTensorOp" : to_json(opTensorOp),
    "opTensorCompType" : to_json(opTensorCompType),
    "opTensorNanOpt" : to_json(opTensorNanOpt)
  });
  return res;
}

// 19
cudnnStatus_t CUDNNWINAPI cudnnGetOpTensorDescriptor(
    const cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t *opTensorOp,
    cudnnDataType_t *opTensorCompType, cudnnNanPropagation_t *opTensorNanOpt) {
  static auto orig_cudnnGetOpTensorDescriptor =
      (decltype(cudnnGetOpTensorDescriptor))dlsym(RTLD_NEXT,
                                                  "cudnnGetOpTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetOpTensorDescriptor(
      opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetOpTensorDescriptor"},
    "start" : tic,
    "end" : toc,
    "opTensorDesc" : to_json(opTensorDesc),
    "opTensorOp" : to_json(opTensorOp),
    "opTensorCompType" : to_json(opTensorCompType),
    "opTensorNanOpt" : to_json(opTensorNanOpt)
  });
  return res;
}

// 20
cudnnStatus_t CUDNNWINAPI
cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc) {
  static auto orig_cudnnDestroyOpTensorDescriptor =
      (decltype(cudnnDestroyOpTensorDescriptor))dlsym(
          RTLD_NEXT, "cudnnDestroyOpTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyOpTensorDescriptor(opTensorDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroyOpTensorDescriptor"},
    "start" : tic,
    "end" : toc,
    "opTensorDesc" : to_json(opTensorDesc)
  });
  return res;
}

// 21
cudnnStatus_t CUDNNWINAPI cudnnOpTensor(
    cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc,
    const void *alpha1, const cudnnTensorDescriptor_t aDesc, const void *A,
    const void *alpha2, const cudnnTensorDescriptor_t bDesc, const void *B,
    const void *beta, const cudnnTensorDescriptor_t cDesc, void *C) {
  static auto orig_cudnnOpTensor =
      (decltype(cudnnOpTensor))dlsym(RTLD_NEXT, "cudnnOpTensor");
  const auto tic = now();
  const auto res = orig_cudnnOpTensor(handle, opTensorDesc, alpha1, aDesc, A,
                                      alpha2, bDesc, B, beta, cDesc, C);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnOpTensor"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "opTensorDesc" : to_json(opTensorDesc),
    "alpha1" : to_json(alpha1),
    "aDesc" : to_json(aDesc),
    "A" : to_json(A),
    "alpha2" : to_json(alpha2),
    "bDesc" : to_json(bDesc),
    "B" : to_json(B),
    "beta" : to_json(beta),
    "cDesc" : to_json(cDesc),
    "C" : to_json(C)
  });
  return res;
}

// 22
cudnnStatus_t CUDNNWINAPI cudnnCreateReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t *reduceTensorDesc) {
  static auto orig_cudnnCreateReduceTensorDescriptor =
      (decltype(cudnnCreateReduceTensorDescriptor))dlsym(
          RTLD_NEXT, "cudnnCreateReduceTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateReduceTensorDescriptor(reduceTensorDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreateReduceTensorDescriptor"},
    "start" : tic,
    "end" : toc,
    "reduceTensorDesc" : to_json(reduceTensorDesc)
  });
  return res;
}

// 23
cudnnStatus_t CUDNNWINAPI cudnnSetReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t reduceTensorDesc,
    cudnnReduceTensorOp_t reduceTensorOp, cudnnDataType_t reduceTensorCompType,
    cudnnNanPropagation_t reduceTensorNanOpt,
    cudnnReduceTensorIndices_t reduceTensorIndices,
    cudnnIndicesType_t reduceTensorIndicesType) {
  static auto orig_cudnnSetReduceTensorDescriptor =
      (decltype(cudnnSetReduceTensorDescriptor))dlsym(
          RTLD_NEXT, "cudnnSetReduceTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetReduceTensorDescriptor(
      reduceTensorDesc, reduceTensorOp, reduceTensorCompType,
      reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetReduceTensorDescriptor"},
    "start" : tic,
    "end" : toc,
    "reduceTensorDesc" : to_json(reduceTensorDesc),
    "reduceTensorOp" : to_json(reduceTensorOp),
    "reduceTensorCompType" : to_json(reduceTensorCompType),
    "reduceTensorNanOpt" : to_json(reduceTensorNanOpt),
    "reduceTensorIndices" : to_json(reduceTensorIndices),
    "reduceTensorIndicesType" : to_json(reduceTensorIndicesType)
  });
  return res;
}

// 24
cudnnStatus_t CUDNNWINAPI cudnnGetReduceTensorDescriptor(
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    cudnnReduceTensorOp_t *reduceTensorOp,
    cudnnDataType_t *reduceTensorCompType,
    cudnnNanPropagation_t *reduceTensorNanOpt,
    cudnnReduceTensorIndices_t *reduceTensorIndices,
    cudnnIndicesType_t *reduceTensorIndicesType) {
  static auto orig_cudnnGetReduceTensorDescriptor =
      (decltype(cudnnGetReduceTensorDescriptor))dlsym(
          RTLD_NEXT, "cudnnGetReduceTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetReduceTensorDescriptor(
      reduceTensorDesc, reduceTensorOp, reduceTensorCompType,
      reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetReduceTensorDescriptor"},
    "start" : tic,
    "end" : toc,
    "reduceTensorDesc" : to_json(reduceTensorDesc),
    "reduceTensorOp" : to_json(reduceTensorOp),
    "reduceTensorCompType" : to_json(reduceTensorCompType),
    "reduceTensorNanOpt" : to_json(reduceTensorNanOpt),
    "reduceTensorIndices" : to_json(reduceTensorIndices),
    "reduceTensorIndicesType" : to_json(reduceTensorIndicesType)
  });
  return res;
}

// 25
cudnnStatus_t CUDNNWINAPI cudnnDestroyReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t reduceTensorDesc) {
  static auto orig_cudnnDestroyReduceTensorDescriptor =
      (decltype(cudnnDestroyReduceTensorDescriptor))dlsym(
          RTLD_NEXT, "cudnnDestroyReduceTensorDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroyReduceTensorDescriptor"},
    "start" : tic,
    "end" : toc,
    "reduceTensorDesc" : to_json(reduceTensorDesc)
  });
  return res;
}

// 26
cudnnStatus_t CUDNNWINAPI cudnnGetReductionIndicesSize(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes) {
  static auto orig_cudnnGetReductionIndicesSize =
      (decltype(cudnnGetReductionIndicesSize))dlsym(
          RTLD_NEXT, "cudnnGetReductionIndicesSize");
  const auto tic = now();
  const auto res = orig_cudnnGetReductionIndicesSize(handle, reduceTensorDesc,
                                                     aDesc, cDesc, sizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetReductionIndicesSize"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "reduceTensorDesc" : to_json(reduceTensorDesc),
    "aDesc" : to_json(aDesc),
    "cDesc" : to_json(cDesc),
    "sizeInBytes" : to_json(sizeInBytes)
  });
  return res;
}

// 27
cudnnStatus_t CUDNNWINAPI cudnnGetReductionWorkspaceSize(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes) {
  static auto orig_cudnnGetReductionWorkspaceSize =
      (decltype(cudnnGetReductionWorkspaceSize))dlsym(
          RTLD_NEXT, "cudnnGetReductionWorkspaceSize");
  const auto tic = now();
  const auto res = orig_cudnnGetReductionWorkspaceSize(
      handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetReductionWorkspaceSize"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "reduceTensorDesc" : to_json(reduceTensorDesc),
    "aDesc" : to_json(aDesc),
    "cDesc" : to_json(cDesc),
    "sizeInBytes" : to_json(sizeInBytes)
  });
  return res;
}

// 28
cudnnStatus_t CUDNNWINAPI cudnnReduceTensor(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    void *indices, size_t indicesSizeInBytes, void *workspace,
    size_t workspaceSizeInBytes, const void *alpha,
    const cudnnTensorDescriptor_t aDesc, const void *A, const void *beta,
    const cudnnTensorDescriptor_t cDesc, void *C) {
  static auto orig_cudnnReduceTensor =
      (decltype(cudnnReduceTensor))dlsym(RTLD_NEXT, "cudnnReduceTensor");
  const auto tic = now();
  const auto res = orig_cudnnReduceTensor(
      handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace,
      workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnReduceTensor"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "reduceTensorDesc" : to_json(reduceTensorDesc),
    "indices" : to_json(indices),
    "indicesSizeInBytes" : to_json(indicesSizeInBytes),
    "workspace" : to_json(workspace),
    "workspaceSizeInBytes" : to_json(workspaceSizeInBytes),
    "alpha" : to_json(alpha),
    "aDesc" : to_json(aDesc),
    "A" : to_json(A),
    "beta" : to_json(beta),
    "cDesc" : to_json(cDesc),
    "C" : to_json(C)
  });
  return res;
}

// 29
cudnnStatus_t CUDNNWINAPI cudnnSetTensor(cudnnHandle_t handle,
                                         const cudnnTensorDescriptor_t yDesc,
                                         void *y, const void *valuePtr) {
  static auto orig_cudnnSetTensor =
      (decltype(cudnnSetTensor))dlsym(RTLD_NEXT, "cudnnSetTensor");
  const auto tic = now();
  const auto res = orig_cudnnSetTensor(handle, yDesc, y, valuePtr);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetTensor"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y),
    "valuePtr" : to_json(valuePtr)
  });
  return res;
}

// 30
cudnnStatus_t CUDNNWINAPI cudnnScaleTensor(cudnnHandle_t handle,
                                           const cudnnTensorDescriptor_t yDesc,
                                           void *y, const void *alpha) {
  static auto orig_cudnnScaleTensor =
      (decltype(cudnnScaleTensor))dlsym(RTLD_NEXT, "cudnnScaleTensor");
  const auto tic = now();
  const auto res = orig_cudnnScaleTensor(handle, yDesc, y, alpha);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnScaleTensor"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y),
    "alpha" : to_json(alpha)
  });
  return res;
}

// 31
cudnnStatus_t CUDNNWINAPI
cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
  static auto orig_cudnnCreateFilterDescriptor =
      (decltype(cudnnCreateFilterDescriptor))dlsym(
          RTLD_NEXT, "cudnnCreateFilterDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateFilterDescriptor(filterDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreateFilterDescriptor"},
    "start" : tic,
    "end" : toc,
    "filterDesc" : to_json(filterDesc)
  });
  return res;
}

// 32
cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor(
    cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType,          /* image data type */
    cudnnTensorFormat_t format, int k, /* number of output feature maps */
    int c,                             /* number of input feature maps */
    int h,                             /* height of each input filter */
    int w) {
  static auto orig_cudnnSetFilter4dDescriptor =
      (decltype(cudnnSetFilter4dDescriptor))dlsym(RTLD_NEXT,
                                                  "cudnnSetFilter4dDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnSetFilter4dDescriptor(filterDesc, /, format, /, /, /, w);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetFilter4dDescriptor"},
    "start" : tic,
    "end" : toc,
    "filterDesc" : to_json(filterDesc),
    "/" : to_json(/),
    "format" : to_json(format),
    "/" : to_json(/),
    "/" : to_json(/),
    "/" : to_json(/),
    "w" : to_json(w)
  });
  return res;
}

// 33
cudnnStatus_t CUDNNWINAPI cudnnGetFilter4dDescriptor(
    const cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t *dataType,           /* image data type */
    cudnnTensorFormat_t *format, int *k, /* number of output feature maps */
    int *c,                              /* number of input feature maps */
    int *h,                              /* height of each input filter */
    int *w) {
  static auto orig_cudnnGetFilter4dDescriptor =
      (decltype(cudnnGetFilter4dDescriptor))dlsym(RTLD_NEXT,
                                                  "cudnnGetFilter4dDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnGetFilter4dDescriptor(filterDesc, /, format, /, /, /, w);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetFilter4dDescriptor"},
    "start" : tic,
    "end" : toc,
    "filterDesc" : to_json(filterDesc),
    "/" : to_json(/),
    "format" : to_json(format),
    "/" : to_json(/),
    "/" : to_json(/),
    "/" : to_json(/),
    "w" : to_json(w)
  });
  return res;
}

// 34
cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor(
    cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType, /* image data type */
    cudnnTensorFormat_t format, int nbDims, const int filterDimA[]) {
  static auto orig_cudnnSetFilterNdDescriptor =
      (decltype(cudnnSetFilterNdDescriptor))dlsym(RTLD_NEXT,
                                                  "cudnnSetFilterNdDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetFilterNdDescriptor(filterDesc, /, format,
                                                   nbDims, filterDimA);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetFilterNdDescriptor"},
    "start" : tic,
    "end" : toc,
    "filterDesc" : to_json(filterDesc),
    "/" : to_json(/),
    "format" : to_json(format),
    "nbDims" : to_json(nbDims),
    "filterDimA" : to_json(filterDimA)
  });
  return res;
}

// 35
cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor(
    const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested,
    cudnnDataType_t *dataType, /* image data type */
    cudnnTensorFormat_t *format, int *nbDims, int filterDimA[]) {
  static auto orig_cudnnGetFilterNdDescriptor =
      (decltype(cudnnGetFilterNdDescriptor))dlsym(RTLD_NEXT,
                                                  "cudnnGetFilterNdDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetFilterNdDescriptor(
      filterDesc, nbDimsRequested, /, format, nbDims, filterDimA);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetFilterNdDescriptor"},
    "start" : tic,
    "end" : toc,
    "filterDesc" : to_json(filterDesc),
    "nbDimsRequested" : to_json(nbDimsRequested),
    "/" : to_json(/),
    "format" : to_json(format),
    "nbDims" : to_json(nbDims),
    "filterDimA" : to_json(filterDimA)
  });
  return res;
}

// 36
cudnnStatus_t CUDNNWINAPI
cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
  static auto orig_cudnnDestroyFilterDescriptor =
      (decltype(cudnnDestroyFilterDescriptor))dlsym(
          RTLD_NEXT, "cudnnDestroyFilterDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyFilterDescriptor(filterDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroyFilterDescriptor"},
    "start" : tic,
    "end" : toc,
    "filterDesc" : to_json(filterDesc)
  });
  return res;
}

// 37
cudnnStatus_t CUDNNWINAPI
cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
  static auto orig_cudnnCreateConvolutionDescriptor =
      (decltype(cudnnCreateConvolutionDescriptor))dlsym(
          RTLD_NEXT, "cudnnCreateConvolutionDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateConvolutionDescriptor(convDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreateConvolutionDescriptor"},
    "start" : tic,
    "end" : toc,
    "convDesc" : to_json(convDesc)
  });
  return res;
}

// 38
cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionMathType(
    cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType) {
  static auto orig_cudnnSetConvolutionMathType =
      (decltype(cudnnSetConvolutionMathType))dlsym(
          RTLD_NEXT, "cudnnSetConvolutionMathType");
  const auto tic = now();
  const auto res = orig_cudnnSetConvolutionMathType(convDesc, mathType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetConvolutionMathType"},
    "start" : tic,
    "end" : toc,
    "convDesc" : to_json(convDesc),
    "mathType" : to_json(mathType)
  });
  return res;
}

// 39
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionMathType(
    cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType) {
  static auto orig_cudnnGetConvolutionMathType =
      (decltype(cudnnGetConvolutionMathType))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionMathType");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionMathType(convDesc, mathType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolutionMathType"},
    "start" : tic,
    "end" : toc,
    "convDesc" : to_json(convDesc),
    "mathType" : to_json(mathType)
  });
  return res;
}

// 40
cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionGroupCount(
    cudnnConvolutionDescriptor_t convDesc, int groupCount) {
  static auto orig_cudnnSetConvolutionGroupCount =
      (decltype(cudnnSetConvolutionGroupCount))dlsym(
          RTLD_NEXT, "cudnnSetConvolutionGroupCount");
  const auto tic = now();
  const auto res = orig_cudnnSetConvolutionGroupCount(convDesc, groupCount);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetConvolutionGroupCount"},
    "start" : tic,
    "end" : toc,
    "convDesc" : to_json(convDesc),
    "groupCount" : to_json(groupCount)
  });
  return res;
}

// 41
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionGroupCount(
    cudnnConvolutionDescriptor_t convDesc, int *groupCount) {
  static auto orig_cudnnGetConvolutionGroupCount =
      (decltype(cudnnGetConvolutionGroupCount))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionGroupCount");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionGroupCount(convDesc, groupCount);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolutionGroupCount"},
    "start" : tic,
    "end" : toc,
    "convDesc" : to_json(convDesc),
    "groupCount" : to_json(groupCount)
  });
  return res;
}

// 42
cudnnStatus_t CUDNNWINAPI cudnnSetConvolution2dDescriptor(
    cudnnConvolutionDescriptor_t convDesc, int pad_h, /* zero-padding height */
    int pad_w,                                        /* zero-padding width */
    int u,          /* vertical filter stride */
    int v,          /* horizontal filter stride */
    int dilation_h, /* filter dilation in the vertical dimension */
    int dilation_w, /* filter dilation in the horizontal dimension */
    cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {
  static auto orig_cudnnSetConvolution2dDescriptor =
      (decltype(cudnnSetConvolution2dDescriptor))dlsym(
          RTLD_NEXT, "cudnnSetConvolution2dDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetConvolution2dDescriptor(convDesc, /, /, /, /, /,
                                                        /, mode, computeType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetConvolution2dDescriptor"},
    "start" : tic,
    "end" : toc,
    "convDesc" : to_json(convDesc),
    "/" : to_json(/),
    "/" : to_json(/),
    "/" : to_json(/),
    "/" : to_json(/),
    "/" : to_json(/),
    "/" : to_json(/),
    "mode" : to_json(mode),
    "computeType" : to_json(computeType)
  });
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
    cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType) {
  static auto orig_cudnnGetConvolution2dDescriptor =
      (decltype(cudnnGetConvolution2dDescriptor))dlsym(
          RTLD_NEXT, "cudnnGetConvolution2dDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolution2dDescriptor(convDesc, /, /, /, /, /,
                                                        /, mode, computeType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolution2dDescriptor"},
    "start" : tic,
    "end" : toc,
    "convDesc" : to_json(convDesc),
    "/" : to_json(/),
    "/" : to_json(/),
    "/" : to_json(/),
    "/" : to_json(/),
    "/" : to_json(/),
    "/" : to_json(/),
    "mode" : to_json(mode),
    "computeType" : to_json(computeType)
  });
  return res;
}

// 44
cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int *n, int *c, int *h, int *w) {
  static auto orig_cudnnGetConvolution2dForwardOutputDim =
      (decltype(cudnnGetConvolution2dForwardOutputDim))dlsym(
          RTLD_NEXT, "cudnnGetConvolution2dForwardOutputDim");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolution2dForwardOutputDim(
      convDesc, inputTensorDesc, filterDesc, n, c, h, w);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolution2dForwardOutputDim"},
    "start" : tic,
    "end" : toc,
    "convDesc" : to_json(convDesc),
    "inputTensorDesc" : to_json(inputTensorDesc),
    "filterDesc" : to_json(filterDesc),
    "n" : to_json(n),
    "c" : to_json(c),
    "h" : to_json(h),
    "w" : to_json(w)
  });
  return res;
}

// 45
cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionNdDescriptor(
    cudnnConvolutionDescriptor_t convDesc, int arrayLength, /* nbDims-2 size */
    const int padA[], const int filterStrideA[], const int dilationA[],
    cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {
  static auto orig_cudnnSetConvolutionNdDescriptor =
      (decltype(cudnnSetConvolutionNdDescriptor))dlsym(
          RTLD_NEXT, "cudnnSetConvolutionNdDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetConvolutionNdDescriptor(
      convDesc, /, padA, filterStrideA, dilationA, mode, computeType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetConvolutionNdDescriptor"},
    "start" : tic,
    "end" : toc,
    "convDesc" : to_json(convDesc),
    "/" : to_json(/),
    "padA" : to_json(padA),
    "filterStrideA" : to_json(filterStrideA),
    "dilationA" : to_json(dilationA),
    "mode" : to_json(mode),
    "computeType" : to_json(computeType)
  });
  return res;
}

// 46
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdDescriptor(
    const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested,
    int *arrayLength, int padA[], int strideA[], int dilationA[],
    cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType) {
  static auto orig_cudnnGetConvolutionNdDescriptor =
      (decltype(cudnnGetConvolutionNdDescriptor))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionNdDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionNdDescriptor(
      convDesc, arrayLengthRequested, arrayLength, padA, strideA, dilationA,
      mode, computeType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolutionNdDescriptor"},
    "start" : tic,
    "end" : toc,
    "convDesc" : to_json(convDesc),
    "arrayLengthRequested" : to_json(arrayLengthRequested),
    "arrayLength" : to_json(arrayLength),
    "padA" : to_json(padA),
    "strideA" : to_json(strideA),
    "dilationA" : to_json(dilationA),
    "mode" : to_json(mode),
    "computeType" : to_json(computeType)
  });
  return res;
}

// 47
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int nbDims,
    int tensorOuputDimA[]) {
  static auto orig_cudnnGetConvolutionNdForwardOutputDim =
      (decltype(cudnnGetConvolutionNdForwardOutputDim))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionNdForwardOutputDim");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionNdForwardOutputDim(
      convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolutionNdForwardOutputDim"},
    "start" : tic,
    "end" : toc,
    "convDesc" : to_json(convDesc),
    "inputTensorDesc" : to_json(inputTensorDesc),
    "filterDesc" : to_json(filterDesc),
    "nbDims" : to_json(nbDims),
    "tensorOuputDimA" : to_json(tensorOuputDimA)
  });
  return res;
}

// 48
cudnnStatus_t CUDNNWINAPI
cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
  static auto orig_cudnnDestroyConvolutionDescriptor =
      (decltype(cudnnDestroyConvolutionDescriptor))dlsym(
          RTLD_NEXT, "cudnnDestroyConvolutionDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyConvolutionDescriptor(convDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroyConvolutionDescriptor"},
    "start" : tic,
    "end" : toc,
    "convDesc" : to_json(convDesc)
  });
  return res;
}

// 49
cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
  static auto orig_cudnnGetConvolutionForwardAlgorithmMaxCount =
      (decltype(cudnnGetConvolutionForwardAlgorithmMaxCount))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionForwardAlgorithmMaxCount");
  const auto tic = now();
  const auto res =
      orig_cudnnGetConvolutionForwardAlgorithmMaxCount(handle, count);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolutionForwardAlgorithmMaxCount"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "count" : to_json(count)
  });
  return res;
}

// 50
cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults) {
  static auto orig_cudnnFindConvolutionForwardAlgorithm =
      (decltype(cudnnFindConvolutionForwardAlgorithm))dlsym(
          RTLD_NEXT, "cudnnFindConvolutionForwardAlgorithm");
  const auto tic = now();
  const auto res = orig_cudnnFindConvolutionForwardAlgorithm(
      handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount,
      returnedAlgoCount, perfResults);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnFindConvolutionForwardAlgorithm"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "xDesc" : to_json(xDesc),
    "wDesc" : to_json(wDesc),
    "convDesc" : to_json(convDesc),
    "yDesc" : to_json(yDesc),
    "requestedAlgoCount" : to_json(requestedAlgoCount),
    "returnedAlgoCount" : to_json(returnedAlgoCount),
    "perfResults" : to_json(perfResults)
  });
  return res;
}

// 51
cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, void *y, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults,
    void *workSpace, size_t workSpaceSizeInBytes) {
  static auto orig_cudnnFindConvolutionForwardAlgorithmEx =
      (decltype(cudnnFindConvolutionForwardAlgorithmEx))dlsym(
          RTLD_NEXT, "cudnnFindConvolutionForwardAlgorithmEx");
  const auto tic = now();
  const auto res = orig_cudnnFindConvolutionForwardAlgorithmEx(
      handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount,
      returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnFindConvolutionForwardAlgorithmEx"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "wDesc" : to_json(wDesc),
    "w" : to_json(w),
    "convDesc" : to_json(convDesc),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y),
    "requestedAlgoCount" : to_json(requestedAlgoCount),
    "returnedAlgoCount" : to_json(returnedAlgoCount),
    "perfResults" : to_json(perfResults),
    "workSpace" : to_json(workSpace),
    "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes)
  });
  return res;
}

// 52
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionFwdAlgo_t *algo) {
  static auto orig_cudnnGetConvolutionForwardAlgorithm =
      (decltype(cudnnGetConvolutionForwardAlgorithm))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionForwardAlgorithm");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionForwardAlgorithm(
      handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes,
      algo);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolutionForwardAlgorithm"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "xDesc" : to_json(xDesc),
    "wDesc" : to_json(wDesc),
    "convDesc" : to_json(convDesc),
    "yDesc" : to_json(yDesc),
    "preference" : to_json(preference),
    "memoryLimitInBytes" : to_json(memoryLimitInBytes),
    "algo" : to_json(algo)
  });
  return res;
}

// 53
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t destDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults) {
  static auto orig_cudnnGetConvolutionForwardAlgorithm_v7 =
      (decltype(cudnnGetConvolutionForwardAlgorithm_v7))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionForwardAlgorithm_v7");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionForwardAlgorithm_v7(
      handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount,
      returnedAlgoCount, perfResults);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolutionForwardAlgorithm_v7"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "srcDesc" : to_json(srcDesc),
    "filterDesc" : to_json(filterDesc),
    "convDesc" : to_json(convDesc),
    "destDesc" : to_json(destDesc),
    "requestedAlgoCount" : to_json(requestedAlgoCount),
    "returnedAlgoCount" : to_json(returnedAlgoCount),
    "perfResults" : to_json(perfResults)
  });
  return res;
}

// 54
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes) {
  static auto orig_cudnnGetConvolutionForwardWorkspaceSize =
      (decltype(cudnnGetConvolutionForwardWorkspaceSize))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionForwardWorkspaceSize");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionForwardWorkspaceSize(
      handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolutionForwardWorkspaceSize"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "xDesc" : to_json(xDesc),
    "wDesc" : to_json(wDesc),
    "convDesc" : to_json(convDesc),
    "yDesc" : to_json(yDesc),
    "algo" : to_json(algo),
    "sizeInBytes" : to_json(sizeInBytes)
  });
  return res;
}

// 55
cudnnStatus_t CUDNNWINAPI cudnnConvolutionForward(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y) {
  static auto orig_cudnnConvolutionForward =
      (decltype(cudnnConvolutionForward))dlsym(RTLD_NEXT,
                                               "cudnnConvolutionForward");
  const auto tic = now();
  const auto res = orig_cudnnConvolutionForward(
      handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace,
      workSpaceSizeInBytes, beta, yDesc, y);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnConvolutionForward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "alpha" : to_json(alpha),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "wDesc" : to_json(wDesc),
    "w" : to_json(w),
    "convDesc" : to_json(convDesc),
    "algo" : to_json(algo),
    "workSpace" : to_json(workSpace),
    "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
    "beta" : to_json(beta),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y)
  });
  return res;
}

// 56
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBiasActivationForward(
    cudnnHandle_t handle, const void *alpha1,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *alpha2,
    const cudnnTensorDescriptor_t zDesc, const void *z,
    const cudnnTensorDescriptor_t biasDesc, const void *bias,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc, void *y) {
  static auto orig_cudnnConvolutionBiasActivationForward =
      (decltype(cudnnConvolutionBiasActivationForward))dlsym(
          RTLD_NEXT, "cudnnConvolutionBiasActivationForward");
  const auto tic = now();
  const auto res = orig_cudnnConvolutionBiasActivationForward(
      handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace,
      workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc,
      yDesc, y);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnConvolutionBiasActivationForward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "alpha1" : to_json(alpha1),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "wDesc" : to_json(wDesc),
    "w" : to_json(w),
    "convDesc" : to_json(convDesc),
    "algo" : to_json(algo),
    "workSpace" : to_json(workSpace),
    "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
    "alpha2" : to_json(alpha2),
    "zDesc" : to_json(zDesc),
    "z" : to_json(z),
    "biasDesc" : to_json(biasDesc),
    "bias" : to_json(bias),
    "activationDesc" : to_json(activationDesc),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y)
  });
  return res;
}

// 57
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardBias(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *beta,
    const cudnnTensorDescriptor_t dbDesc, void *db) {
  static auto orig_cudnnConvolutionBackwardBias =
      (decltype(cudnnConvolutionBackwardBias))dlsym(
          RTLD_NEXT, "cudnnConvolutionBackwardBias");
  const auto tic = now();
  const auto res = orig_cudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy,
                                                     beta, dbDesc, db);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnConvolutionBackwardBias"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "alpha" : to_json(alpha),
    "dyDesc" : to_json(dyDesc),
    "dy" : to_json(dy),
    "beta" : to_json(beta),
    "dbDesc" : to_json(dbDesc),
    "db" : to_json(db)
  });
  return res;
}

// 58
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
    cudnnHandle_t handle, int *count) {
  static auto orig_cudnnGetConvolutionBackwardFilterAlgorithmMaxCount =
      (decltype(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");
  const auto tic = now();
  const auto res =
      orig_cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, count);
  const auto toc = now();

  callback({
    "funName" :
        std::string{"cudnnGetConvolutionBackwardFilterAlgorithmMaxCount"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "count" : to_json(count)
  });
  return res;
}

// 59
cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
  static auto orig_cudnnFindConvolutionBackwardFilterAlgorithm =
      (decltype(cudnnFindConvolutionBackwardFilterAlgorithm))dlsym(
          RTLD_NEXT, "cudnnFindConvolutionBackwardFilterAlgorithm");
  const auto tic = now();
  const auto res = orig_cudnnFindConvolutionBackwardFilterAlgorithm(
      handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount,
      returnedAlgoCount, perfResults);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnFindConvolutionBackwardFilterAlgorithm"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "xDesc" : to_json(xDesc),
    "dyDesc" : to_json(dyDesc),
    "convDesc" : to_json(convDesc),
    "dwDesc" : to_json(dwDesc),
    "requestedAlgoCount" : to_json(requestedAlgoCount),
    "returnedAlgoCount" : to_json(returnedAlgoCount),
    "perfResults" : to_json(perfResults)
  });
  return res;
}

// 60
cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *y,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, void *dw,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes) {
  static auto orig_cudnnFindConvolutionBackwardFilterAlgorithmEx =
      (decltype(cudnnFindConvolutionBackwardFilterAlgorithmEx))dlsym(
          RTLD_NEXT, "cudnnFindConvolutionBackwardFilterAlgorithmEx");
  const auto tic = now();
  const auto res = orig_cudnnFindConvolutionBackwardFilterAlgorithmEx(
      handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw, requestedAlgoCount,
      returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnFindConvolutionBackwardFilterAlgorithmEx"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "dyDesc" : to_json(dyDesc),
    "y" : to_json(y),
    "convDesc" : to_json(convDesc),
    "dwDesc" : to_json(dwDesc),
    "dw" : to_json(dw),
    "requestedAlgoCount" : to_json(requestedAlgoCount),
    "returnedAlgoCount" : to_json(returnedAlgoCount),
    "perfResults" : to_json(perfResults),
    "workSpace" : to_json(workSpace),
    "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes)
  });
  return res;
}

// 61
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    cudnnConvolutionBwdFilterPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionBwdFilterAlgo_t *algo) {
  static auto orig_cudnnGetConvolutionBackwardFilterAlgorithm =
      (decltype(cudnnGetConvolutionBackwardFilterAlgorithm))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardFilterAlgorithm");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardFilterAlgorithm(
      handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes,
      algo);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolutionBackwardFilterAlgorithm"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "xDesc" : to_json(xDesc),
    "dyDesc" : to_json(dyDesc),
    "convDesc" : to_json(convDesc),
    "dwDesc" : to_json(dwDesc),
    "preference" : to_json(preference),
    "memoryLimitInBytes" : to_json(memoryLimitInBytes),
    "algo" : to_json(algo)
  });
  return res;
}

// 62
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
  static auto orig_cudnnGetConvolutionBackwardFilterAlgorithm_v7 =
      (decltype(cudnnGetConvolutionBackwardFilterAlgorithm_v7))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardFilterAlgorithm_v7");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardFilterAlgorithm_v7(
      handle, srcDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount,
      returnedAlgoCount, perfResults);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolutionBackwardFilterAlgorithm_v7"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "srcDesc" : to_json(srcDesc),
    "diffDesc" : to_json(diffDesc),
    "convDesc" : to_json(convDesc),
    "gradDesc" : to_json(gradDesc),
    "requestedAlgoCount" : to_json(requestedAlgoCount),
    "returnedAlgoCount" : to_json(returnedAlgoCount),
    "perfResults" : to_json(perfResults)
  });
  return res;
}

// 63
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes) {
  static auto orig_cudnnGetConvolutionBackwardFilterWorkspaceSize =
      (decltype(cudnnGetConvolutionBackwardFilterWorkspaceSize))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardFilterWorkspaceSize");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardFilterWorkspaceSize(
      handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolutionBackwardFilterWorkspaceSize"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "xDesc" : to_json(xDesc),
    "dyDesc" : to_json(dyDesc),
    "convDesc" : to_json(convDesc),
    "gradDesc" : to_json(gradDesc),
    "algo" : to_json(algo),
    "sizeInBytes" : to_json(sizeInBytes)
  });
  return res;
}

// 64
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnFilterDescriptor_t dwDesc, void *dw) {
  static auto orig_cudnnConvolutionBackwardFilter =
      (decltype(cudnnConvolutionBackwardFilter))dlsym(
          RTLD_NEXT, "cudnnConvolutionBackwardFilter");
  const auto tic = now();
  const auto res = orig_cudnnConvolutionBackwardFilter(
      handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace,
      workSpaceSizeInBytes, beta, dwDesc, dw);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnConvolutionBackwardFilter"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "alpha" : to_json(alpha),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "dyDesc" : to_json(dyDesc),
    "dy" : to_json(dy),
    "convDesc" : to_json(convDesc),
    "algo" : to_json(algo),
    "workSpace" : to_json(workSpace),
    "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
    "beta" : to_json(beta),
    "dwDesc" : to_json(dwDesc),
    "dw" : to_json(dw)
  });
  return res;
}

// 65
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
    cudnnHandle_t handle, int *count) {
  static auto orig_cudnnGetConvolutionBackwardDataAlgorithmMaxCount =
      (decltype(cudnnGetConvolutionBackwardDataAlgorithmMaxCount))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardDataAlgorithmMaxCount");
  const auto tic = now();
  const auto res =
      orig_cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, count);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolutionBackwardDataAlgorithmMaxCount"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "count" : to_json(count)
  });
  return res;
}

// 66
cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
  static auto orig_cudnnFindConvolutionBackwardDataAlgorithm =
      (decltype(cudnnFindConvolutionBackwardDataAlgorithm))dlsym(
          RTLD_NEXT, "cudnnFindConvolutionBackwardDataAlgorithm");
  const auto tic = now();
  const auto res = orig_cudnnFindConvolutionBackwardDataAlgorithm(
      handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount,
      returnedAlgoCount, perfResults);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnFindConvolutionBackwardDataAlgorithm"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "wDesc" : to_json(wDesc),
    "dyDesc" : to_json(dyDesc),
    "convDesc" : to_json(convDesc),
    "dxDesc" : to_json(dxDesc),
    "requestedAlgoCount" : to_json(requestedAlgoCount),
    "returnedAlgoCount" : to_json(returnedAlgoCount),
    "perfResults" : to_json(perfResults)
  });
  return res;
}

// 67
cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes) {
  static auto orig_cudnnFindConvolutionBackwardDataAlgorithmEx =
      (decltype(cudnnFindConvolutionBackwardDataAlgorithmEx))dlsym(
          RTLD_NEXT, "cudnnFindConvolutionBackwardDataAlgorithmEx");
  const auto tic = now();
  const auto res = orig_cudnnFindConvolutionBackwardDataAlgorithmEx(
      handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, requestedAlgoCount,
      returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnFindConvolutionBackwardDataAlgorithmEx"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "wDesc" : to_json(wDesc),
    "w" : to_json(w),
    "dyDesc" : to_json(dyDesc),
    "dy" : to_json(dy),
    "convDesc" : to_json(convDesc),
    "dxDesc" : to_json(dxDesc),
    "dx" : to_json(dx),
    "requestedAlgoCount" : to_json(requestedAlgoCount),
    "returnedAlgoCount" : to_json(returnedAlgoCount),
    "perfResults" : to_json(perfResults),
    "workSpace" : to_json(workSpace),
    "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes)
  });
  return res;
}

// 68
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionBwdDataAlgo_t *algo) {
  static auto orig_cudnnGetConvolutionBackwardDataAlgorithm =
      (decltype(cudnnGetConvolutionBackwardDataAlgorithm))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardDataAlgorithm");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardDataAlgorithm(
      handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes,
      algo);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolutionBackwardDataAlgorithm"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "wDesc" : to_json(wDesc),
    "dyDesc" : to_json(dyDesc),
    "convDesc" : to_json(convDesc),
    "dxDesc" : to_json(dxDesc),
    "preference" : to_json(preference),
    "memoryLimitInBytes" : to_json(memoryLimitInBytes),
    "algo" : to_json(algo)
  });
  return res;
}

// 69
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
  static auto orig_cudnnGetConvolutionBackwardDataAlgorithm_v7 =
      (decltype(cudnnGetConvolutionBackwardDataAlgorithm_v7))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardDataAlgorithm_v7");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardDataAlgorithm_v7(
      handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount,
      returnedAlgoCount, perfResults);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolutionBackwardDataAlgorithm_v7"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "filterDesc" : to_json(filterDesc),
    "diffDesc" : to_json(diffDesc),
    "convDesc" : to_json(convDesc),
    "gradDesc" : to_json(gradDesc),
    "requestedAlgoCount" : to_json(requestedAlgoCount),
    "returnedAlgoCount" : to_json(returnedAlgoCount),
    "perfResults" : to_json(perfResults)
  });
  return res;
}

// 70
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes) {
  static auto orig_cudnnGetConvolutionBackwardDataWorkspaceSize =
      (decltype(cudnnGetConvolutionBackwardDataWorkspaceSize))dlsym(
          RTLD_NEXT, "cudnnGetConvolutionBackwardDataWorkspaceSize");
  const auto tic = now();
  const auto res = orig_cudnnGetConvolutionBackwardDataWorkspaceSize(
      handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetConvolutionBackwardDataWorkspaceSize"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "wDesc" : to_json(wDesc),
    "dyDesc" : to_json(dyDesc),
    "convDesc" : to_json(convDesc),
    "dxDesc" : to_json(dxDesc),
    "algo" : to_json(algo),
    "sizeInBytes" : to_json(sizeInBytes)
  });
  return res;
}

// 71
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  static auto orig_cudnnConvolutionBackwardData =
      (decltype(cudnnConvolutionBackwardData))dlsym(
          RTLD_NEXT, "cudnnConvolutionBackwardData");
  const auto tic = now();
  const auto res = orig_cudnnConvolutionBackwardData(
      handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace,
      workSpaceSizeInBytes, beta, dxDesc, dx);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnConvolutionBackwardData"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "alpha" : to_json(alpha),
    "wDesc" : to_json(wDesc),
    "w" : to_json(w),
    "dyDesc" : to_json(dyDesc),
    "dy" : to_json(dy),
    "convDesc" : to_json(convDesc),
    "algo" : to_json(algo),
    "workSpace" : to_json(workSpace),
    "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
    "beta" : to_json(beta),
    "dxDesc" : to_json(dxDesc),
    "dx" : to_json(dx)
  });
  return res;
}

// 72
cudnnStatus_t CUDNNWINAPI
cudnnIm2Col(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
            const void *x, const cudnnFilterDescriptor_t wDesc,
            const cudnnConvolutionDescriptor_t convDesc, void *colBuffer) {
  static auto orig_cudnnIm2Col =
      (decltype(cudnnIm2Col))dlsym(RTLD_NEXT, "cudnnIm2Col");
  const auto tic = now();
  const auto res =
      orig_cudnnIm2Col(handle, xDesc, x, wDesc, convDesc, colBuffer);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnIm2Col"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "wDesc" : to_json(wDesc),
    "convDesc" : to_json(convDesc),
    "colBuffer" : to_json(colBuffer)
  });
  return res;
}

// 73
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxForward(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  static auto orig_cudnnSoftmaxForward =
      (decltype(cudnnSoftmaxForward))dlsym(RTLD_NEXT, "cudnnSoftmaxForward");
  const auto tic = now();
  const auto res = orig_cudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x,
                                            beta, yDesc, y);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSoftmaxForward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "algo" : to_json(algo),
    "mode" : to_json(mode),
    "alpha" : to_json(alpha),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "beta" : to_json(beta),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y)
  });
  return res;
}

// 74
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxBackward(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  static auto orig_cudnnSoftmaxBackward =
      (decltype(cudnnSoftmaxBackward))dlsym(RTLD_NEXT, "cudnnSoftmaxBackward");
  const auto tic = now();
  const auto res = orig_cudnnSoftmaxBackward(handle, algo, mode, alpha, yDesc,
                                             y, dyDesc, dy, beta, dxDesc, dx);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSoftmaxBackward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "algo" : to_json(algo),
    "mode" : to_json(mode),
    "alpha" : to_json(alpha),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y),
    "dyDesc" : to_json(dyDesc),
    "dy" : to_json(dy),
    "beta" : to_json(beta),
    "dxDesc" : to_json(dxDesc),
    "dx" : to_json(dx)
  });
  return res;
}

// 75
cudnnStatus_t CUDNNWINAPI
cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc) {
  static auto orig_cudnnCreatePoolingDescriptor =
      (decltype(cudnnCreatePoolingDescriptor))dlsym(
          RTLD_NEXT, "cudnnCreatePoolingDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreatePoolingDescriptor(poolingDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreatePoolingDescriptor"},
    "start" : tic,
    "end" : toc,
    "poolingDesc" : to_json(poolingDesc)
  });
  return res;
}

// 76
cudnnStatus_t CUDNNWINAPI cudnnSetPooling2dDescriptor(
    cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode,
    cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight, int windowWidth,
    int verticalPadding, int horizontalPadding, int verticalStride,
    int horizontalStride) {
  static auto orig_cudnnSetPooling2dDescriptor =
      (decltype(cudnnSetPooling2dDescriptor))dlsym(
          RTLD_NEXT, "cudnnSetPooling2dDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetPooling2dDescriptor(
      poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth,
      verticalPadding, horizontalPadding, verticalStride, horizontalStride);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetPooling2dDescriptor"},
    "start" : tic,
    "end" : toc,
    "poolingDesc" : to_json(poolingDesc),
    "mode" : to_json(mode),
    "maxpoolingNanOpt" : to_json(maxpoolingNanOpt),
    "windowHeight" : to_json(windowHeight),
    "windowWidth" : to_json(windowWidth),
    "verticalPadding" : to_json(verticalPadding),
    "horizontalPadding" : to_json(horizontalPadding),
    "verticalStride" : to_json(verticalStride),
    "horizontalStride" : to_json(horizontalStride)
  });
  return res;
}

// 77
cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dDescriptor(
    const cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt, int *windowHeight,
    int *windowWidth, int *verticalPadding, int *horizontalPadding,
    int *verticalStride, int *horizontalStride) {
  static auto orig_cudnnGetPooling2dDescriptor =
      (decltype(cudnnGetPooling2dDescriptor))dlsym(
          RTLD_NEXT, "cudnnGetPooling2dDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetPooling2dDescriptor(
      poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth,
      verticalPadding, horizontalPadding, verticalStride, horizontalStride);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetPooling2dDescriptor"},
    "start" : tic,
    "end" : toc,
    "poolingDesc" : to_json(poolingDesc),
    "mode" : to_json(mode),
    "maxpoolingNanOpt" : to_json(maxpoolingNanOpt),
    "windowHeight" : to_json(windowHeight),
    "windowWidth" : to_json(windowWidth),
    "verticalPadding" : to_json(verticalPadding),
    "horizontalPadding" : to_json(horizontalPadding),
    "verticalStride" : to_json(verticalStride),
    "horizontalStride" : to_json(horizontalStride)
  });
  return res;
}

// 78
cudnnStatus_t CUDNNWINAPI cudnnSetPoolingNdDescriptor(
    cudnnPoolingDescriptor_t poolingDesc, const cudnnPoolingMode_t mode,
    const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims,
    const int windowDimA[], const int paddingA[], const int strideA[]) {
  static auto orig_cudnnSetPoolingNdDescriptor =
      (decltype(cudnnSetPoolingNdDescriptor))dlsym(
          RTLD_NEXT, "cudnnSetPoolingNdDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnSetPoolingNdDescriptor(poolingDesc, mode, maxpoolingNanOpt,
                                       nbDims, windowDimA, paddingA, strideA);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetPoolingNdDescriptor"},
    "start" : tic,
    "end" : toc,
    "poolingDesc" : to_json(poolingDesc),
    "mode" : to_json(mode),
    "maxpoolingNanOpt" : to_json(maxpoolingNanOpt),
    "nbDims" : to_json(nbDims),
    "windowDimA" : to_json(windowDimA),
    "paddingA" : to_json(paddingA),
    "strideA" : to_json(strideA)
  });
  return res;
}

// 79
cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdDescriptor(
    const cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested,
    cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt,
    int *nbDims, int windowDimA[], int paddingA[], int strideA[]) {
  static auto orig_cudnnGetPoolingNdDescriptor =
      (decltype(cudnnGetPoolingNdDescriptor))dlsym(
          RTLD_NEXT, "cudnnGetPoolingNdDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetPoolingNdDescriptor(
      poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA,
      paddingA, strideA);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetPoolingNdDescriptor"},
    "start" : tic,
    "end" : toc,
    "poolingDesc" : to_json(poolingDesc),
    "nbDimsRequested" : to_json(nbDimsRequested),
    "mode" : to_json(mode),
    "maxpoolingNanOpt" : to_json(maxpoolingNanOpt),
    "nbDims" : to_json(nbDims),
    "windowDimA" : to_json(windowDimA),
    "paddingA" : to_json(paddingA),
    "strideA" : to_json(strideA)
  });
  return res;
}

// 80
cudnnStatus_t CUDNNWINAPI
cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                  const cudnnTensorDescriptor_t inputTensorDesc,
                                  int nbDims, int outputTensorDimA[]) {
  static auto orig_cudnnGetPoolingNdForwardOutputDim =
      (decltype(cudnnGetPoolingNdForwardOutputDim))dlsym(
          RTLD_NEXT, "cudnnGetPoolingNdForwardOutputDim");
  const auto tic = now();
  const auto res = orig_cudnnGetPoolingNdForwardOutputDim(
      poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetPoolingNdForwardOutputDim"},
    "start" : tic,
    "end" : toc,
    "poolingDesc" : to_json(poolingDesc),
    "inputTensorDesc" : to_json(inputTensorDesc),
    "nbDims" : to_json(nbDims),
    "outputTensorDimA" : to_json(outputTensorDimA)
  });
  return res;
}

// 81
cudnnStatus_t CUDNNWINAPI
cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                  const cudnnTensorDescriptor_t inputTensorDesc,
                                  int *n, int *c, int *h, int *w) {
  static auto orig_cudnnGetPooling2dForwardOutputDim =
      (decltype(cudnnGetPooling2dForwardOutputDim))dlsym(
          RTLD_NEXT, "cudnnGetPooling2dForwardOutputDim");
  const auto tic = now();
  const auto res = orig_cudnnGetPooling2dForwardOutputDim(
      poolingDesc, inputTensorDesc, n, c, h, w);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetPooling2dForwardOutputDim"},
    "start" : tic,
    "end" : toc,
    "poolingDesc" : to_json(poolingDesc),
    "inputTensorDesc" : to_json(inputTensorDesc),
    "n" : to_json(n),
    "c" : to_json(c),
    "h" : to_json(h),
    "w" : to_json(w)
  });
  return res;
}

// 82
cudnnStatus_t CUDNNWINAPI
cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
  static auto orig_cudnnDestroyPoolingDescriptor =
      (decltype(cudnnDestroyPoolingDescriptor))dlsym(
          RTLD_NEXT, "cudnnDestroyPoolingDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyPoolingDescriptor(poolingDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroyPoolingDescriptor"},
    "start" : tic,
    "end" : toc,
    "poolingDesc" : to_json(poolingDesc)
  });
  return res;
}

// 83
cudnnStatus_t CUDNNWINAPI cudnnPoolingForward(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  static auto orig_cudnnPoolingForward =
      (decltype(cudnnPoolingForward))dlsym(RTLD_NEXT, "cudnnPoolingForward");
  const auto tic = now();
  const auto res = orig_cudnnPoolingForward(handle, poolingDesc, alpha, xDesc,
                                            x, beta, yDesc, y);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnPoolingForward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "poolingDesc" : to_json(poolingDesc),
    "alpha" : to_json(alpha),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "beta" : to_json(beta),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y)
  });
  return res;
}

// 84
cudnnStatus_t CUDNNWINAPI cudnnPoolingBackward(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  static auto orig_cudnnPoolingBackward =
      (decltype(cudnnPoolingBackward))dlsym(RTLD_NEXT, "cudnnPoolingBackward");
  const auto tic = now();
  const auto res =
      orig_cudnnPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc,
                                dy, xDesc, x, beta, dxDesc, dx);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnPoolingBackward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "poolingDesc" : to_json(poolingDesc),
    "alpha" : to_json(alpha),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y),
    "dyDesc" : to_json(dyDesc),
    "dy" : to_json(dy),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "beta" : to_json(beta),
    "dxDesc" : to_json(dxDesc),
    "dx" : to_json(dx)
  });
  return res;
}

// 85
cudnnStatus_t CUDNNWINAPI
cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc) {
  static auto orig_cudnnCreateActivationDescriptor =
      (decltype(cudnnCreateActivationDescriptor))dlsym(
          RTLD_NEXT, "cudnnCreateActivationDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateActivationDescriptor(activationDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreateActivationDescriptor"},
    "start" : tic,
    "end" : toc,
    "activationDesc" : to_json(activationDesc)
  });
  return res;
}

// 86
cudnnStatus_t CUDNNWINAPI cudnnSetActivationDescriptor(
    cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t mode,
    cudnnNanPropagation_t reluNanOpt, double coef) {
  static auto orig_cudnnSetActivationDescriptor =
      (decltype(cudnnSetActivationDescriptor))dlsym(
          RTLD_NEXT, "cudnnSetActivationDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnSetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetActivationDescriptor"},
    "start" : tic,
    "end" : toc,
    "activationDesc" : to_json(activationDesc),
    "mode" : to_json(mode),
    "reluNanOpt" : to_json(reluNanOpt),
    "coef" : to_json(coef)
  });
  return res;
}

// 87
cudnnStatus_t CUDNNWINAPI
cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,
                             cudnnActivationMode_t *mode,
                             cudnnNanPropagation_t *reluNanOpt, double *coef) {
  static auto orig_cudnnGetActivationDescriptor =
      (decltype(cudnnGetActivationDescriptor))dlsym(
          RTLD_NEXT, "cudnnGetActivationDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnGetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetActivationDescriptor"},
    "start" : tic,
    "end" : toc,
    "activationDesc" : to_json(activationDesc),
    "mode" : to_json(mode),
    "reluNanOpt" : to_json(reluNanOpt),
    "coef" : to_json(coef)
  });
  return res;
}

// 88
cudnnStatus_t CUDNNWINAPI
cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
  static auto orig_cudnnDestroyActivationDescriptor =
      (decltype(cudnnDestroyActivationDescriptor))dlsym(
          RTLD_NEXT, "cudnnDestroyActivationDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyActivationDescriptor(activationDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroyActivationDescriptor"},
    "start" : tic,
    "end" : toc,
    "activationDesc" : to_json(activationDesc)
  });
  return res;
}

// 89
cudnnStatus_t CUDNNWINAPI cudnnActivationForward(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  static auto orig_cudnnActivationForward =
      (decltype(cudnnActivationForward))dlsym(RTLD_NEXT,
                                              "cudnnActivationForward");
  const auto tic = now();
  const auto res = orig_cudnnActivationForward(handle, activationDesc, alpha,
                                               xDesc, x, beta, yDesc, y);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnActivationForward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "activationDesc" : to_json(activationDesc),
    "alpha" : to_json(alpha),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "beta" : to_json(beta),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y)
  });
  return res;
}

// 90
cudnnStatus_t CUDNNWINAPI cudnnActivationBackward(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  static auto orig_cudnnActivationBackward =
      (decltype(cudnnActivationBackward))dlsym(RTLD_NEXT,
                                               "cudnnActivationBackward");
  const auto tic = now();
  const auto res =
      orig_cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y,
                                   dyDesc, dy, xDesc, x, beta, dxDesc, dx);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnActivationBackward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "activationDesc" : to_json(activationDesc),
    "alpha" : to_json(alpha),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y),
    "dyDesc" : to_json(dyDesc),
    "dy" : to_json(dy),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "beta" : to_json(beta),
    "dxDesc" : to_json(dxDesc),
    "dx" : to_json(dx)
  });
  return res;
}

// 91
cudnnStatus_t CUDNNWINAPI
cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc) {
  static auto orig_cudnnCreateLRNDescriptor =
      (decltype(cudnnCreateLRNDescriptor))dlsym(RTLD_NEXT,
                                                "cudnnCreateLRNDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateLRNDescriptor(normDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreateLRNDescriptor"},
    "start" : tic,
    "end" : toc,
    "normDesc" : to_json(normDesc)
  });
  return res;
}

// 92
cudnnStatus_t CUDNNWINAPI cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
                                                unsigned lrnN, double lrnAlpha,
                                                double lrnBeta, double lrnK) {
  static auto orig_cudnnSetLRNDescriptor =
      (decltype(cudnnSetLRNDescriptor))dlsym(RTLD_NEXT,
                                             "cudnnSetLRNDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetLRNDescriptor"},
    "start" : tic,
    "end" : toc,
    "normDesc" : to_json(normDesc),
    "lrnN" : to_json(lrnN),
    "lrnAlpha" : to_json(lrnAlpha),
    "lrnBeta" : to_json(lrnBeta),
    "lrnK" : to_json(lrnK)
  });
  return res;
}

// 93
cudnnStatus_t CUDNNWINAPI cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
                                                unsigned *lrnN,
                                                double *lrnAlpha,
                                                double *lrnBeta, double *lrnK) {
  static auto orig_cudnnGetLRNDescriptor =
      (decltype(cudnnGetLRNDescriptor))dlsym(RTLD_NEXT,
                                             "cudnnGetLRNDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnGetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetLRNDescriptor"},
    "start" : tic,
    "end" : toc,
    "normDesc" : to_json(normDesc),
    "lrnN" : to_json(lrnN),
    "lrnAlpha" : to_json(lrnAlpha),
    "lrnBeta" : to_json(lrnBeta),
    "lrnK" : to_json(lrnK)
  });
  return res;
}

// 94
cudnnStatus_t CUDNNWINAPI
cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc) {
  static auto orig_cudnnDestroyLRNDescriptor =
      (decltype(cudnnDestroyLRNDescriptor))dlsym(RTLD_NEXT,
                                                 "cudnnDestroyLRNDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyLRNDescriptor(lrnDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroyLRNDescriptor"},
    "start" : tic,
    "end" : toc,
    "lrnDesc" : to_json(lrnDesc)
  });
  return res;
}

// 95
cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelForward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  static auto orig_cudnnLRNCrossChannelForward =
      (decltype(cudnnLRNCrossChannelForward))dlsym(
          RTLD_NEXT, "cudnnLRNCrossChannelForward");
  const auto tic = now();
  const auto res = orig_cudnnLRNCrossChannelForward(
      handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnLRNCrossChannelForward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "normDesc" : to_json(normDesc),
    "lrnMode" : to_json(lrnMode),
    "alpha" : to_json(alpha),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "beta" : to_json(beta),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y)
  });
  return res;
}

// 96
cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelBackward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  static auto orig_cudnnLRNCrossChannelBackward =
      (decltype(cudnnLRNCrossChannelBackward))dlsym(
          RTLD_NEXT, "cudnnLRNCrossChannelBackward");
  const auto tic = now();
  const auto res = orig_cudnnLRNCrossChannelBackward(
      handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta,
      dxDesc, dx);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnLRNCrossChannelBackward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "normDesc" : to_json(normDesc),
    "lrnMode" : to_json(lrnMode),
    "alpha" : to_json(alpha),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y),
    "dyDesc" : to_json(dyDesc),
    "dy" : to_json(dy),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "beta" : to_json(beta),
    "dxDesc" : to_json(dxDesc),
    "dx" : to_json(dx)
  });
  return res;
}

// 97
cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationForward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, /* same desc for means, temp, temp2 */
    const void *x,
    const void *means, /* if NULL, means are assumed to be zero */
    void *temp, void *temp2, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y) {
  static auto orig_cudnnDivisiveNormalizationForward =
      (decltype(cudnnDivisiveNormalizationForward))dlsym(
          RTLD_NEXT, "cudnnDivisiveNormalizationForward");
  const auto tic = now();
  const auto res = orig_cudnnDivisiveNormalizationForward(
      handle, normDesc, mode, alpha, /, x, /, temp, temp2, beta, yDesc, y);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDivisiveNormalizationForward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "normDesc" : to_json(normDesc),
    "mode" : to_json(mode),
    "alpha" : to_json(alpha),
    "/" : to_json(/),
    "x" : to_json(x),
    "/" : to_json(/),
    "temp" : to_json(temp),
    "temp2" : to_json(temp2),
    "beta" : to_json(beta),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y)
  });
  return res;
}

// 98
cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationBackward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t
        xDesc, /* same desc for x, means, dy, temp, temp2 */
    const void *x,
    const void *means, /* if NULL, means are assumed to be zero */
    const void *dy, void *temp, void *temp2, const void *beta,
    const cudnnTensorDescriptor_t dXdMeansDesc, /* same desc for dx, dMeans */
    void *dx,                                   /* output x differential */
    void *dMeans) {
  static auto orig_cudnnDivisiveNormalizationBackward =
      (decltype(cudnnDivisiveNormalizationBackward))dlsym(
          RTLD_NEXT, "cudnnDivisiveNormalizationBackward");
  const auto tic = now();
  const auto res = orig_cudnnDivisiveNormalizationBackward(
      handle, normDesc, mode, alpha, /, x, /, dy, temp, temp2, beta, /, /,
      dMeans);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDivisiveNormalizationBackward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "normDesc" : to_json(normDesc),
    "mode" : to_json(mode),
    "alpha" : to_json(alpha),
    "/" : to_json(/),
    "x" : to_json(x),
    "/" : to_json(/),
    "dy" : to_json(dy),
    "temp" : to_json(temp),
    "temp2" : to_json(temp2),
    "beta" : to_json(beta),
    "/" : to_json(/),
    "/" : to_json(/),
    "dMeans" : to_json(dMeans)
  });
  return res;
}

// 99
cudnnStatus_t CUDNNWINAPI cudnnDeriveBNTensorDescriptor(
    cudnnTensorDescriptor_t derivedBnDesc, const cudnnTensorDescriptor_t xDesc,
    cudnnBatchNormMode_t mode) {
  static auto orig_cudnnDeriveBNTensorDescriptor =
      (decltype(cudnnDeriveBNTensorDescriptor))dlsym(
          RTLD_NEXT, "cudnnDeriveBNTensorDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnDeriveBNTensorDescriptor(derivedBnDesc, xDesc, mode);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDeriveBNTensorDescriptor"},
    "start" : tic,
    "end" : toc,
    "derivedBnDesc" : to_json(derivedBnDesc),
    "xDesc" : to_json(xDesc),
    "mode" : to_json(mode)
  });
  return res;
}

// 100
cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,

    const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,  /* beta[0] = dest layer blend factor */

    const cudnnTensorDescriptor_t xDesc, const void *x, /* NxCxHxW */
    const cudnnTensorDescriptor_t yDesc, void *y,       /* NxCxHxW */

    /* Shared desc for the next 6 tensors in the argument list.
       Data type to be set as follows:
       type = (typeOf(x) == double) ? double : float
       Dimensions for this descriptor depend on normalization mode
       - Spatial Normalization : tensors are expected to have dims 1xCx1x1
        (normalization is performed across NxHxW)
       - Per-Activation Normalization : tensors are expected to have dims of
       1xCxHxW (normalization is performed across N) */
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,

    /* 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation
     */
    const void *bnScale, const void *bnBias,

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

    /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and
       backward functions. */
    double epsilon,

    /* Optionally save intermediate results from the forward pass here
       - can be reused to speed up backward pass. NULL if unused */
    void *resultSaveMean, void *resultSaveInvVariance) {
  static auto orig_cudnnBatchNormalizationForwardTraining =
      (decltype(cudnnBatchNormalizationForwardTraining))dlsym(
          RTLD_NEXT, "cudnnBatchNormalizationForwardTraining");
  const auto tic = now();
  const auto res = orig_cudnnBatchNormalizationForwardTraining(runningMean);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnBatchNormalizationForwardTraining"},
    "start" : tic,
    "end" : toc,
    "runningMean" : to_json(runningMean)
  });
  return res;
}

// 101
cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode,
    const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,  /* beta[0] = dest layer blend factor */
    const cudnnTensorDescriptor_t xDesc, const void *x, /* NxCxHxW */
    const cudnnTensorDescriptor_t yDesc, void *y,       /* NxCxHxW */
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon) {
  static auto orig_cudnnBatchNormalizationForwardInference =
      (decltype(cudnnBatchNormalizationForwardInference))dlsym(
          RTLD_NEXT, "cudnnBatchNormalizationForwardInference");
  const auto tic = now();
  const auto res = orig_cudnnBatchNormalizationForwardInference(
      handle, mode, /, /, xDesc, /, yDesc, /, bnScaleBiasMeanVarDesc, bnScale,
      bnBias, estimatedMean, estimatedVariance, epsilon);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnBatchNormalizationForwardInference"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "mode" : to_json(mode),
    "/" : to_json(/),
    "/" : to_json(/),
    "xDesc" : to_json(xDesc),
    "/" : to_json(/),
    "yDesc" : to_json(yDesc),
    "/" : to_json(/),
    "bnScaleBiasMeanVarDesc" : to_json(bnScaleBiasMeanVarDesc),
    "bnScale" : to_json(bnScale),
    "bnBias" : to_json(bnBias),
    "estimatedMean" : to_json(estimatedMean),
    "estimatedVariance" : to_json(estimatedVariance),
    "epsilon" : to_json(epsilon)
  });
  return res;
}

// 102
cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackward(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alphaDataDiff,
    const void *betaDataDiff, const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, /* same desc for x, dx, dy */
    const void *x, const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    /* Shared tensor desc for the 4 tensors below */
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScale, /* bnBias doesn't affect backpropagation */
    /* scale and bias diff are not backpropagated below this layer */
    void *dBnScaleResult, void *dBnBiasResult,
    /* Same epsilon as forward pass */
    double epsilon,

    /* Optionally cached intermediate results from
       forward pass */
    const void *savedMean, const void *savedInvVariance) {
  static auto orig_cudnnBatchNormalizationBackward =
      (decltype(cudnnBatchNormalizationBackward))dlsym(
          RTLD_NEXT, "cudnnBatchNormalizationBackward");
  const auto tic = now();
  const auto res = orig_cudnnBatchNormalizationBackward(
      handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff,
      /, x, dyDesc, dy, dxDesc, dx, /, dBnScaleBiasDesc, /, /, dBnScaleResult,
      dBnBiasResult, /, epsilon, from, /, savedMean, savedInvVariance);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnBatchNormalizationBackward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "mode" : to_json(mode),
    "alphaDataDiff" : to_json(alphaDataDiff),
    "betaDataDiff" : to_json(betaDataDiff),
    "alphaParamDiff" : to_json(alphaParamDiff),
    "betaParamDiff" : to_json(betaParamDiff),
    "/" : to_json(/),
    "x" : to_json(x),
    "dyDesc" : to_json(dyDesc),
    "dy" : to_json(dy),
    "dxDesc" : to_json(dxDesc),
    "dx" : to_json(dx),
    "/" : to_json(/),
    "dBnScaleBiasDesc" : to_json(dBnScaleBiasDesc),
    "/" : to_json(/),
    "/" : to_json(/),
    "dBnScaleResult" : to_json(dBnScaleResult),
    "dBnBiasResult" : to_json(dBnBiasResult),
    "/" : to_json(/),
    "epsilon" : to_json(epsilon),
    "from" : to_json(from),
    "/" : to_json(/),
    "savedMean" : to_json(savedMean),
    "savedInvVariance" : to_json(savedInvVariance)
  });
  return res;
}

// 103
cudnnStatus_t CUDNNWINAPI cudnnCreateSpatialTransformerDescriptor(
    cudnnSpatialTransformerDescriptor_t *stDesc) {
  static auto orig_cudnnCreateSpatialTransformerDescriptor =
      (decltype(cudnnCreateSpatialTransformerDescriptor))dlsym(
          RTLD_NEXT, "cudnnCreateSpatialTransformerDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateSpatialTransformerDescriptor(stDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreateSpatialTransformerDescriptor"},
    "start" : tic,
    "end" : toc,
    "stDesc" : to_json(stDesc)
  });
  return res;
}

// 104
cudnnStatus_t CUDNNWINAPI cudnnSetSpatialTransformerNdDescriptor(
    cudnnSpatialTransformerDescriptor_t stDesc, cudnnSamplerType_t samplerType,
    cudnnDataType_t dataType, const int nbDims, const int dimA[]) {
  static auto orig_cudnnSetSpatialTransformerNdDescriptor =
      (decltype(cudnnSetSpatialTransformerNdDescriptor))dlsym(
          RTLD_NEXT, "cudnnSetSpatialTransformerNdDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetSpatialTransformerNdDescriptor(
      stDesc, samplerType, dataType, nbDims, dimA);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetSpatialTransformerNdDescriptor"},
    "start" : tic,
    "end" : toc,
    "stDesc" : to_json(stDesc),
    "samplerType" : to_json(samplerType),
    "dataType" : to_json(dataType),
    "nbDims" : to_json(nbDims),
    "dimA" : to_json(dimA)
  });
  return res;
}

// 105
cudnnStatus_t CUDNNWINAPI cudnnDestroySpatialTransformerDescriptor(
    cudnnSpatialTransformerDescriptor_t stDesc) {
  static auto orig_cudnnDestroySpatialTransformerDescriptor =
      (decltype(cudnnDestroySpatialTransformerDescriptor))dlsym(
          RTLD_NEXT, "cudnnDestroySpatialTransformerDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroySpatialTransformerDescriptor(stDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroySpatialTransformerDescriptor"},
    "start" : tic,
    "end" : toc,
    "stDesc" : to_json(stDesc)
  });
  return res;
}

// 106
cudnnStatus_t CUDNNWINAPI cudnnSpatialTfGridGeneratorForward(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *theta, void *grid) {
  static auto orig_cudnnSpatialTfGridGeneratorForward =
      (decltype(cudnnSpatialTfGridGeneratorForward))dlsym(
          RTLD_NEXT, "cudnnSpatialTfGridGeneratorForward");
  const auto tic = now();
  const auto res =
      orig_cudnnSpatialTfGridGeneratorForward(handle, stDesc, theta, grid);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSpatialTfGridGeneratorForward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "stDesc" : to_json(stDesc),
    "theta" : to_json(theta),
    "grid" : to_json(grid)
  });
  return res;
}

// 107
cudnnStatus_t CUDNNWINAPI cudnnSpatialTfGridGeneratorBackward(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *dgrid, void *dtheta) {
  static auto orig_cudnnSpatialTfGridGeneratorBackward =
      (decltype(cudnnSpatialTfGridGeneratorBackward))dlsym(
          RTLD_NEXT, "cudnnSpatialTfGridGeneratorBackward");
  const auto tic = now();
  const auto res =
      orig_cudnnSpatialTfGridGeneratorBackward(handle, stDesc, dgrid, dtheta);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSpatialTfGridGeneratorBackward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "stDesc" : to_json(stDesc),
    "dgrid" : to_json(dgrid),
    "dtheta" : to_json(dtheta)
  });
  return res;
}

// 108
cudnnStatus_t CUDNNWINAPI cudnnSpatialTfSamplerForward(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *grid, const void *beta, cudnnTensorDescriptor_t yDesc,
    void *y) {
  static auto orig_cudnnSpatialTfSamplerForward =
      (decltype(cudnnSpatialTfSamplerForward))dlsym(
          RTLD_NEXT, "cudnnSpatialTfSamplerForward");
  const auto tic = now();
  const auto res = orig_cudnnSpatialTfSamplerForward(
      handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSpatialTfSamplerForward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "stDesc" : to_json(stDesc),
    "alpha" : to_json(alpha),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "grid" : to_json(grid),
    "beta" : to_json(beta),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y)
  });
  return res;
}

// 109
cudnnStatus_t CUDNNWINAPI cudnnSpatialTfSamplerBackward(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx,
    const void *alphaDgrid, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const void *grid, const void *betaDgrid, void *dgrid) {
  static auto orig_cudnnSpatialTfSamplerBackward =
      (decltype(cudnnSpatialTfSamplerBackward))dlsym(
          RTLD_NEXT, "cudnnSpatialTfSamplerBackward");
  const auto tic = now();
  const auto res = orig_cudnnSpatialTfSamplerBackward(
      handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc, dy,
      grid, betaDgrid, dgrid);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSpatialTfSamplerBackward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "stDesc" : to_json(stDesc),
    "alpha" : to_json(alpha),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "beta" : to_json(beta),
    "dxDesc" : to_json(dxDesc),
    "dx" : to_json(dx),
    "alphaDgrid" : to_json(alphaDgrid),
    "dyDesc" : to_json(dyDesc),
    "dy" : to_json(dy),
    "grid" : to_json(grid),
    "betaDgrid" : to_json(betaDgrid),
    "dgrid" : to_json(dgrid)
  });
  return res;
}

// 110
cudnnStatus_t CUDNNWINAPI
cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc) {
  static auto orig_cudnnCreateDropoutDescriptor =
      (decltype(cudnnCreateDropoutDescriptor))dlsym(
          RTLD_NEXT, "cudnnCreateDropoutDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateDropoutDescriptor(dropoutDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreateDropoutDescriptor"},
    "start" : tic,
    "end" : toc,
    "dropoutDesc" : to_json(dropoutDesc)
  });
  return res;
}

// 111
cudnnStatus_t CUDNNWINAPI
cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) {
  static auto orig_cudnnDestroyDropoutDescriptor =
      (decltype(cudnnDestroyDropoutDescriptor))dlsym(
          RTLD_NEXT, "cudnnDestroyDropoutDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyDropoutDescriptor(dropoutDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroyDropoutDescriptor"},
    "start" : tic,
    "end" : toc,
    "dropoutDesc" : to_json(dropoutDesc)
  });
  return res;
}

// 112
cudnnStatus_t CUDNNWINAPI cudnnDropoutGetStatesSize(cudnnHandle_t handle,
                                                    size_t *sizeInBytes) {
  static auto orig_cudnnDropoutGetStatesSize =
      (decltype(cudnnDropoutGetStatesSize))dlsym(RTLD_NEXT,
                                                 "cudnnDropoutGetStatesSize");
  const auto tic = now();
  const auto res = orig_cudnnDropoutGetStatesSize(sizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDropoutGetStatesSize"},
    "start" : tic,
    "end" : toc,
    "sizeInBytes" : to_json(sizeInBytes)
  });
  return res;
}

// 113
cudnnStatus_t CUDNNWINAPI cudnnDropoutGetReserveSpaceSize(
    cudnnTensorDescriptor_t xdesc, size_t *sizeInBytes) {
  static auto orig_cudnnDropoutGetReserveSpaceSize =
      (decltype(cudnnDropoutGetReserveSpaceSize))dlsym(
          RTLD_NEXT, "cudnnDropoutGetReserveSpaceSize");
  const auto tic = now();
  const auto res = orig_cudnnDropoutGetReserveSpaceSize(sizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDropoutGetReserveSpaceSize"},
    "start" : tic,
    "end" : toc,
    "sizeInBytes" : to_json(sizeInBytes)
  });
  return res;
}

// 114
cudnnStatus_t CUDNNWINAPI cudnnSetDropoutDescriptor(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout,
    void *states, size_t stateSizeInBytes, unsigned long long seed) {
  static auto orig_cudnnSetDropoutDescriptor =
      (decltype(cudnnSetDropoutDescriptor))dlsym(RTLD_NEXT,
                                                 "cudnnSetDropoutDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetDropoutDescriptor(
      dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetDropoutDescriptor"},
    "start" : tic,
    "end" : toc,
    "dropoutDesc" : to_json(dropoutDesc),
    "handle" : to_json(handle),
    "dropout" : to_json(dropout),
    "states" : to_json(states),
    "stateSizeInBytes" : to_json(stateSizeInBytes),
    "seed" : to_json(seed)
  });
  return res;
}

// 115
cudnnStatus_t CUDNNWINAPI cudnnRestoreDropoutDescriptor(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout,
    void *states, size_t stateSizeInBytes, unsigned long long seed) {
  static auto orig_cudnnRestoreDropoutDescriptor =
      (decltype(cudnnRestoreDropoutDescriptor))dlsym(
          RTLD_NEXT, "cudnnRestoreDropoutDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnRestoreDropoutDescriptor(
      dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnRestoreDropoutDescriptor"},
    "start" : tic,
    "end" : toc,
    "dropoutDesc" : to_json(dropoutDesc),
    "handle" : to_json(handle),
    "dropout" : to_json(dropout),
    "states" : to_json(states),
    "stateSizeInBytes" : to_json(stateSizeInBytes),
    "seed" : to_json(seed)
  });
  return res;
}

// 116
cudnnStatus_t CUDNNWINAPI cudnnGetDropoutDescriptor(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float *dropout,
    void **states, unsigned long long *seed) {
  static auto orig_cudnnGetDropoutDescriptor =
      (decltype(cudnnGetDropoutDescriptor))dlsym(RTLD_NEXT,
                                                 "cudnnGetDropoutDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetDropoutDescriptor(dropoutDesc, handle, dropout,
                                                  states, seed);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetDropoutDescriptor"},
    "start" : tic,
    "end" : toc,
    "dropoutDesc" : to_json(dropoutDesc),
    "handle" : to_json(handle),
    "dropout" : to_json(dropout),
    "states" : to_json(states),
    "seed" : to_json(seed)
  });
  return res;
}

// 117
cudnnStatus_t CUDNNWINAPI cudnnDropoutForward(
    cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t xdesc, const void *x,
    const cudnnTensorDescriptor_t ydesc, void *y, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  static auto orig_cudnnDropoutForward =
      (decltype(cudnnDropoutForward))dlsym(RTLD_NEXT, "cudnnDropoutForward");
  const auto tic = now();
  const auto res =
      orig_cudnnDropoutForward(handle, dropoutDesc, xdesc, x, ydesc, y,
                               reserveSpace, reserveSpaceSizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDropoutForward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "dropoutDesc" : to_json(dropoutDesc),
    "xdesc" : to_json(xdesc),
    "x" : to_json(x),
    "ydesc" : to_json(ydesc),
    "y" : to_json(y),
    "reserveSpace" : to_json(reserveSpace),
    "reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)
  });
  return res;
}

// 118
cudnnStatus_t CUDNNWINAPI cudnnDropoutBackward(
    cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t dydesc, const void *dy,
    const cudnnTensorDescriptor_t dxdesc, void *dx, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  static auto orig_cudnnDropoutBackward =
      (decltype(cudnnDropoutBackward))dlsym(RTLD_NEXT, "cudnnDropoutBackward");
  const auto tic = now();
  const auto res =
      orig_cudnnDropoutBackward(handle, dropoutDesc, dydesc, dy, dxdesc, dx,
                                reserveSpace, reserveSpaceSizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDropoutBackward"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "dropoutDesc" : to_json(dropoutDesc),
    "dydesc" : to_json(dydesc),
    "dy" : to_json(dy),
    "dxdesc" : to_json(dxdesc),
    "dx" : to_json(dx),
    "reserveSpace" : to_json(reserveSpace),
    "reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)
  });
  return res;
}

// 119
cudnnStatus_t CUDNNWINAPI
cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc) {
  static auto orig_cudnnCreateRNNDescriptor =
      (decltype(cudnnCreateRNNDescriptor))dlsym(RTLD_NEXT,
                                                "cudnnCreateRNNDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateRNNDescriptor(rnnDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreateRNNDescriptor"},
    "start" : tic,
    "end" : toc,
    "rnnDesc" : to_json(rnnDesc)
  });
  return res;
}

// 120
cudnnStatus_t CUDNNWINAPI
cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
  static auto orig_cudnnDestroyRNNDescriptor =
      (decltype(cudnnDestroyRNNDescriptor))dlsym(RTLD_NEXT,
                                                 "cudnnDestroyRNNDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyRNNDescriptor(rnnDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroyRNNDescriptor"},
    "start" : tic,
    "end" : toc,
    "rnnDesc" : to_json(rnnDesc)
  });
  return res;
}

// 121
cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardInferenceAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  static auto orig_cudnnGetRNNForwardInferenceAlgorithmMaxCount =
      (decltype(cudnnGetRNNForwardInferenceAlgorithmMaxCount))dlsym(
          RTLD_NEXT, "cudnnGetRNNForwardInferenceAlgorithmMaxCount");
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNForwardInferenceAlgorithmMaxCount(handle, rnnDesc, count);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetRNNForwardInferenceAlgorithmMaxCount"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "count" : to_json(count)
  });
  return res;
}

// 122
cudnnStatus_t CUDNNWINAPI cudnnFindRNNForwardInferenceAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes) {
  static auto orig_cudnnFindRNNForwardInferenceAlgorithmEx =
      (decltype(cudnnFindRNNForwardInferenceAlgorithmEx))dlsym(
          RTLD_NEXT, "cudnnFindRNNForwardInferenceAlgorithmEx");
  const auto tic = now();
  const auto res = orig_cudnnFindRNNForwardInferenceAlgorithmEx(
      handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w,
      yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount,
      returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnFindRNNForwardInferenceAlgorithmEx"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "seqLength" : to_json(seqLength),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "hxDesc" : to_json(hxDesc),
    "hx" : to_json(hx),
    "cxDesc" : to_json(cxDesc),
    "cx" : to_json(cx),
    "wDesc" : to_json(wDesc),
    "w" : to_json(w),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y),
    "hyDesc" : to_json(hyDesc),
    "hy" : to_json(hy),
    "cyDesc" : to_json(cyDesc),
    "cy" : to_json(cy),
    "findIntensity" : to_json(findIntensity),
    "requestedAlgoCount" : to_json(requestedAlgoCount),
    "returnedAlgoCount" : to_json(returnedAlgoCount),
    "perfResults" : to_json(perfResults),
    "workspace" : to_json(workspace),
    "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes)
  });
  return res;
}

// 123
cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardTrainingAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  static auto orig_cudnnGetRNNForwardTrainingAlgorithmMaxCount =
      (decltype(cudnnGetRNNForwardTrainingAlgorithmMaxCount))dlsym(
          RTLD_NEXT, "cudnnGetRNNForwardTrainingAlgorithmMaxCount");
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNForwardTrainingAlgorithmMaxCount(handle, rnnDesc, count);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetRNNForwardTrainingAlgorithmMaxCount"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "count" : to_json(count)
  });
  return res;
}

// 124
cudnnStatus_t CUDNNWINAPI cudnnFindRNNForwardTrainingAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  static auto orig_cudnnFindRNNForwardTrainingAlgorithmEx =
      (decltype(cudnnFindRNNForwardTrainingAlgorithmEx))dlsym(
          RTLD_NEXT, "cudnnFindRNNForwardTrainingAlgorithmEx");
  const auto tic = now();
  const auto res = orig_cudnnFindRNNForwardTrainingAlgorithmEx(
      handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w,
      yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount,
      returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes,
      reserveSpace, reserveSpaceSizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnFindRNNForwardTrainingAlgorithmEx"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "seqLength" : to_json(seqLength),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "hxDesc" : to_json(hxDesc),
    "hx" : to_json(hx),
    "cxDesc" : to_json(cxDesc),
    "cx" : to_json(cx),
    "wDesc" : to_json(wDesc),
    "w" : to_json(w),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y),
    "hyDesc" : to_json(hyDesc),
    "hy" : to_json(hy),
    "cyDesc" : to_json(cyDesc),
    "cy" : to_json(cy),
    "findIntensity" : to_json(findIntensity),
    "requestedAlgoCount" : to_json(requestedAlgoCount),
    "returnedAlgoCount" : to_json(returnedAlgoCount),
    "perfResults" : to_json(perfResults),
    "workspace" : to_json(workspace),
    "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
    "reserveSpace" : to_json(reserveSpace),
    "reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)
  });
  return res;
}

// 125
cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardDataAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  static auto orig_cudnnGetRNNBackwardDataAlgorithmMaxCount =
      (decltype(cudnnGetRNNBackwardDataAlgorithmMaxCount))dlsym(
          RTLD_NEXT, "cudnnGetRNNBackwardDataAlgorithmMaxCount");
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNBackwardDataAlgorithmMaxCount(handle, rnnDesc, count);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetRNNBackwardDataAlgorithmMaxCount"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "count" : to_json(count)
  });
  return res;
}

// 126
cudnnStatus_t CUDNNWINAPI cudnnFindRNNBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *yDesc, const void *y,
    const cudnnTensorDescriptor_t *dyDesc, const void *dy,
    const cudnnTensorDescriptor_t dhyDesc, const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc, const void *dcy,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnTensorDescriptor_t *dxDesc, void *dx,
    const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  static auto orig_cudnnFindRNNBackwardDataAlgorithmEx =
      (decltype(cudnnFindRNNBackwardDataAlgorithmEx))dlsym(
          RTLD_NEXT, "cudnnFindRNNBackwardDataAlgorithmEx");
  const auto tic = now();
  const auto res = orig_cudnnFindRNNBackwardDataAlgorithmEx(
      handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc,
      dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc,
      dcx, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults,
      workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnFindRNNBackwardDataAlgorithmEx"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "seqLength" : to_json(seqLength),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y),
    "dyDesc" : to_json(dyDesc),
    "dy" : to_json(dy),
    "dhyDesc" : to_json(dhyDesc),
    "dhy" : to_json(dhy),
    "dcyDesc" : to_json(dcyDesc),
    "dcy" : to_json(dcy),
    "wDesc" : to_json(wDesc),
    "w" : to_json(w),
    "hxDesc" : to_json(hxDesc),
    "hx" : to_json(hx),
    "cxDesc" : to_json(cxDesc),
    "cx" : to_json(cx),
    "dxDesc" : to_json(dxDesc),
    "dx" : to_json(dx),
    "dhxDesc" : to_json(dhxDesc),
    "dhx" : to_json(dhx),
    "dcxDesc" : to_json(dcxDesc),
    "dcx" : to_json(dcx),
    "findIntensity" : to_json(findIntensity),
    "requestedAlgoCount" : to_json(requestedAlgoCount),
    "returnedAlgoCount" : to_json(returnedAlgoCount),
    "perfResults" : to_json(perfResults),
    "workspace" : to_json(workspace),
    "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
    "reserveSpace" : to_json(reserveSpace),
    "reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)
  });
  return res;
}

// 127
cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardWeightsAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  static auto orig_cudnnGetRNNBackwardWeightsAlgorithmMaxCount =
      (decltype(cudnnGetRNNBackwardWeightsAlgorithmMaxCount))dlsym(
          RTLD_NEXT, "cudnnGetRNNBackwardWeightsAlgorithmMaxCount");
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle, rnnDesc, count);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetRNNBackwardWeightsAlgorithmMaxCount"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "count" : to_json(count)
  });
  return res;
}

// 128
cudnnStatus_t CUDNNWINAPI cudnnFindRNNBackwardWeightsAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t *yDesc, const void *y,
    const float findIntensity, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnAlgorithmPerformance_t *perfResults,
    const void *workspace, size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc, void *dw, const void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  static auto orig_cudnnFindRNNBackwardWeightsAlgorithmEx =
      (decltype(cudnnFindRNNBackwardWeightsAlgorithmEx))dlsym(
          RTLD_NEXT, "cudnnFindRNNBackwardWeightsAlgorithmEx");
  const auto tic = now();
  const auto res = orig_cudnnFindRNNBackwardWeightsAlgorithmEx(
      handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, findIntensity,
      requestedAlgoCount, returnedAlgoCount, perfResults, workspace,
      workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnFindRNNBackwardWeightsAlgorithmEx"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "seqLength" : to_json(seqLength),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "hxDesc" : to_json(hxDesc),
    "hx" : to_json(hx),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y),
    "findIntensity" : to_json(findIntensity),
    "requestedAlgoCount" : to_json(requestedAlgoCount),
    "returnedAlgoCount" : to_json(returnedAlgoCount),
    "perfResults" : to_json(perfResults),
    "workspace" : to_json(workspace),
    "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
    "dwDesc" : to_json(dwDesc),
    "dw" : to_json(dw),
    "reserveSpace" : to_json(reserveSpace),
    "reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)
  });
  return res;
}

// 129
cudnnStatus_t CUDNNWINAPI cudnnCreatePersistentRNNPlan(
    cudnnRNNDescriptor_t rnnDesc, const int minibatch,
    const cudnnDataType_t dataType, cudnnPersistentRNNPlan_t *plan) {
  static auto orig_cudnnCreatePersistentRNNPlan =
      (decltype(cudnnCreatePersistentRNNPlan))dlsym(
          RTLD_NEXT, "cudnnCreatePersistentRNNPlan");
  const auto tic = now();
  const auto res =
      orig_cudnnCreatePersistentRNNPlan(rnnDesc, minibatch, dataType, plan);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreatePersistentRNNPlan"},
    "start" : tic,
    "end" : toc,
    "rnnDesc" : to_json(rnnDesc),
    "minibatch" : to_json(minibatch),
    "dataType" : to_json(dataType),
    "plan" : to_json(plan)
  });
  return res;
}

// 130
cudnnStatus_t CUDNNWINAPI cudnnSetPersistentRNNPlan(
    cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan) {
  static auto orig_cudnnSetPersistentRNNPlan =
      (decltype(cudnnSetPersistentRNNPlan))dlsym(RTLD_NEXT,
                                                 "cudnnSetPersistentRNNPlan");
  const auto tic = now();
  const auto res = orig_cudnnSetPersistentRNNPlan(rnnDesc, plan);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetPersistentRNNPlan"},
    "start" : tic,
    "end" : toc,
    "rnnDesc" : to_json(rnnDesc),
    "plan" : to_json(plan)
  });
  return res;
}

// 131
cudnnStatus_t CUDNNWINAPI
cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan) {
  static auto orig_cudnnDestroyPersistentRNNPlan =
      (decltype(cudnnDestroyPersistentRNNPlan))dlsym(
          RTLD_NEXT, "cudnnDestroyPersistentRNNPlan");
  const auto tic = now();
  const auto res = orig_cudnnDestroyPersistentRNNPlan(plan);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroyPersistentRNNPlan"},
    "start" : tic,
    "end" : toc,
    "plan" : to_json(plan)
  });
  return res;
}

// 132
cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int hiddenSize,
    const int numLayers,
    cudnnDropoutDescriptor_t
        dropoutDesc, /* Between layers, not between recurrent steps. */
    cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction,
    cudnnRNNMode_t mode, cudnnRNNAlgo_t algo, cudnnDataType_t dataType) {
  static auto orig_cudnnSetRNNDescriptor =
      (decltype(cudnnSetRNNDescriptor))dlsym(RTLD_NEXT,
                                             "cudnnSetRNNDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnSetRNNDescriptor(handle, rnnDesc, hiddenSize, numLayers, /,
                                 inputMode, direction, mode, algo, dataType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetRNNDescriptor"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "hiddenSize" : to_json(hiddenSize),
    "numLayers" : to_json(numLayers),
    "/" : to_json(/),
    "inputMode" : to_json(inputMode),
    "direction" : to_json(direction),
    "mode" : to_json(mode),
    "algo" : to_json(algo),
    "dataType" : to_json(dataType)
  });
  return res;
}

// 133
cudnnStatus_t CUDNNWINAPI
cudnnSetRNNProjectionLayers(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                            const int recProjSize, const int outProjSize) {
  static auto orig_cudnnSetRNNProjectionLayers =
      (decltype(cudnnSetRNNProjectionLayers))dlsym(
          RTLD_NEXT, "cudnnSetRNNProjectionLayers");
  const auto tic = now();
  const auto res = orig_cudnnSetRNNProjectionLayers(handle, rnnDesc,
                                                    recProjSize, outProjSize);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetRNNProjectionLayers"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "recProjSize" : to_json(recProjSize),
    "outProjSize" : to_json(outProjSize)
  });
  return res;
}

// 134
cudnnStatus_t CUDNNWINAPI cudnnGetRNNProjectionLayers(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *recProjSize,
    int *outProjSize) {
  static auto orig_cudnnGetRNNProjectionLayers =
      (decltype(cudnnGetRNNProjectionLayers))dlsym(
          RTLD_NEXT, "cudnnGetRNNProjectionLayers");
  const auto tic = now();
  const auto res = orig_cudnnGetRNNProjectionLayers(handle, rnnDesc,
                                                    recProjSize, outProjSize);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetRNNProjectionLayers"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "recProjSize" : to_json(recProjSize),
    "outProjSize" : to_json(outProjSize)
  });
  return res;
}

// 135
cudnnStatus_t CUDNNWINAPI cudnnSetRNNAlgorithmDescriptor(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnAlgorithmDescriptor_t algoDesc) {
  static auto orig_cudnnSetRNNAlgorithmDescriptor =
      (decltype(cudnnSetRNNAlgorithmDescriptor))dlsym(
          RTLD_NEXT, "cudnnSetRNNAlgorithmDescriptor");
  const auto tic = now();
  const auto res =
      orig_cudnnSetRNNAlgorithmDescriptor(handle, rnnDesc, algoDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetRNNAlgorithmDescriptor"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "algoDesc" : to_json(algoDesc)
  });
  return res;
}

// 136
cudnnStatus_t CUDNNWINAPI cudnnGetRNNDescriptor(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int *hiddenSize,
    int *numLayers, cudnnDropoutDescriptor_t *dropoutDesc,
    cudnnRNNInputMode_t *inputMode, cudnnDirectionMode_t *direction,
    cudnnRNNMode_t *mode, cudnnRNNAlgo_t *algo, cudnnDataType_t *dataType) {
  static auto orig_cudnnGetRNNDescriptor =
      (decltype(cudnnGetRNNDescriptor))dlsym(RTLD_NEXT,
                                             "cudnnGetRNNDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetRNNDescriptor(handle, rnnDesc, hiddenSize,
                                              numLayers, dropoutDesc, inputMode,
                                              direction, mode, algo, dataType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetRNNDescriptor"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "hiddenSize" : to_json(hiddenSize),
    "numLayers" : to_json(numLayers),
    "dropoutDesc" : to_json(dropoutDesc),
    "inputMode" : to_json(inputMode),
    "direction" : to_json(direction),
    "mode" : to_json(mode),
    "algo" : to_json(algo),
    "dataType" : to_json(dataType)
  });
  return res;
}

// 137
cudnnStatus_t CUDNNWINAPI
cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType) {
  static auto orig_cudnnSetRNNMatrixMathType =
      (decltype(cudnnSetRNNMatrixMathType))dlsym(RTLD_NEXT,
                                                 "cudnnSetRNNMatrixMathType");
  const auto tic = now();
  const auto res = orig_cudnnSetRNNMatrixMathType(mType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetRNNMatrixMathType"},
    "start" : tic,
    "end" : toc,
    "mType" : to_json(mType)
  });
  return res;
}

// 138
cudnnStatus_t CUDNNWINAPI cudnnGetRNNMatrixMathType(
    cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t *mType) {
  static auto orig_cudnnGetRNNMatrixMathType =
      (decltype(cudnnGetRNNMatrixMathType))dlsym(RTLD_NEXT,
                                                 "cudnnGetRNNMatrixMathType");
  const auto tic = now();
  const auto res = orig_cudnnGetRNNMatrixMathType(mType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetRNNMatrixMathType"},
    "start" : tic,
    "end" : toc,
    "mType" : to_json(mType)
  });
  return res;
}

// 139
cudnnStatus_t CUDNNWINAPI cudnnGetRNNWorkspaceSize(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes) {
  static auto orig_cudnnGetRNNWorkspaceSize =
      (decltype(cudnnGetRNNWorkspaceSize))dlsym(RTLD_NEXT,
                                                "cudnnGetRNNWorkspaceSize");
  const auto tic = now();
  const auto res = orig_cudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength,
                                                 xDesc, sizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetRNNWorkspaceSize"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "seqLength" : to_json(seqLength),
    "xDesc" : to_json(xDesc),
    "sizeInBytes" : to_json(sizeInBytes)
  });
  return res;
}

// 140
cudnnStatus_t CUDNNWINAPI cudnnGetRNNTrainingReserveSize(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes) {
  static auto orig_cudnnGetRNNTrainingReserveSize =
      (decltype(cudnnGetRNNTrainingReserveSize))dlsym(
          RTLD_NEXT, "cudnnGetRNNTrainingReserveSize");
  const auto tic = now();
  const auto res = orig_cudnnGetRNNTrainingReserveSize(
      handle, rnnDesc, seqLength, xDesc, sizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetRNNTrainingReserveSize"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "seqLength" : to_json(seqLength),
    "xDesc" : to_json(xDesc),
    "sizeInBytes" : to_json(sizeInBytes)
  });
  return res;
}

// 141
cudnnStatus_t CUDNNWINAPI
cudnnGetRNNParamsSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                      const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes,
                      cudnnDataType_t dataType) {
  static auto orig_cudnnGetRNNParamsSize =
      (decltype(cudnnGetRNNParamsSize))dlsym(RTLD_NEXT,
                                             "cudnnGetRNNParamsSize");
  const auto tic = now();
  const auto res =
      orig_cudnnGetRNNParamsSize(handle, rnnDesc, xDesc, sizeInBytes, dataType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetRNNParamsSize"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "xDesc" : to_json(xDesc),
    "sizeInBytes" : to_json(sizeInBytes),
    "dataType" : to_json(dataType)
  });
  return res;
}

// 142
cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerMatrixParams(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const void *w, const int linLayerID,
    cudnnFilterDescriptor_t linLayerMatDesc, void **linLayerMat) {
  static auto orig_cudnnGetRNNLinLayerMatrixParams =
      (decltype(cudnnGetRNNLinLayerMatrixParams))dlsym(
          RTLD_NEXT, "cudnnGetRNNLinLayerMatrixParams");
  const auto tic = now();
  const auto res = orig_cudnnGetRNNLinLayerMatrixParams(
      handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID,
      linLayerMatDesc, linLayerMat);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetRNNLinLayerMatrixParams"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "pseudoLayer" : to_json(pseudoLayer),
    "xDesc" : to_json(xDesc),
    "wDesc" : to_json(wDesc),
    "w" : to_json(w),
    "linLayerID" : to_json(linLayerID),
    "linLayerMatDesc" : to_json(linLayerMatDesc),
    "linLayerMat" : to_json(linLayerMat)
  });
  return res;
}

// 143
cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerBiasParams(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const void *w, const int linLayerID,
    cudnnFilterDescriptor_t linLayerBiasDesc, void **linLayerBias) {
  static auto orig_cudnnGetRNNLinLayerBiasParams =
      (decltype(cudnnGetRNNLinLayerBiasParams))dlsym(
          RTLD_NEXT, "cudnnGetRNNLinLayerBiasParams");
  const auto tic = now();
  const auto res = orig_cudnnGetRNNLinLayerBiasParams(
      handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID,
      linLayerBiasDesc, linLayerBias);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetRNNLinLayerBiasParams"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "pseudoLayer" : to_json(pseudoLayer),
    "xDesc" : to_json(xDesc),
    "wDesc" : to_json(wDesc),
    "w" : to_json(w),
    "linLayerID" : to_json(linLayerID),
    "linLayerBiasDesc" : to_json(linLayerBiasDesc),
    "linLayerBias" : to_json(linLayerBias)
  });
  return res;
}

// 144
cudnnStatus_t CUDNNWINAPI cudnnRNNForwardInference(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes) {
  static auto orig_cudnnRNNForwardInference =
      (decltype(cudnnRNNForwardInference))dlsym(RTLD_NEXT,
                                                "cudnnRNNForwardInference");
  const auto tic = now();
  const auto res = orig_cudnnRNNForwardInference(
      handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w,
      yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnRNNForwardInference"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "seqLength" : to_json(seqLength),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "hxDesc" : to_json(hxDesc),
    "hx" : to_json(hx),
    "cxDesc" : to_json(cxDesc),
    "cx" : to_json(cx),
    "wDesc" : to_json(wDesc),
    "w" : to_json(w),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y),
    "hyDesc" : to_json(hyDesc),
    "hy" : to_json(hy),
    "cyDesc" : to_json(cyDesc),
    "cy" : to_json(cy),
    "workspace" : to_json(workspace),
    "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes)
  });
  return res;
}

// 145
cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTraining(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  static auto orig_cudnnRNNForwardTraining =
      (decltype(cudnnRNNForwardTraining))dlsym(RTLD_NEXT,
                                               "cudnnRNNForwardTraining");
  const auto tic = now();
  const auto res = orig_cudnnRNNForwardTraining(
      handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w,
      yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes,
      reserveSpace, reserveSpaceSizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnRNNForwardTraining"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "seqLength" : to_json(seqLength),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "hxDesc" : to_json(hxDesc),
    "hx" : to_json(hx),
    "cxDesc" : to_json(cxDesc),
    "cx" : to_json(cx),
    "wDesc" : to_json(wDesc),
    "w" : to_json(w),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y),
    "hyDesc" : to_json(hyDesc),
    "hy" : to_json(hy),
    "cyDesc" : to_json(cyDesc),
    "cy" : to_json(cy),
    "workspace" : to_json(workspace),
    "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
    "reserveSpace" : to_json(reserveSpace),
    "reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)
  });
  return res;
}

// 146
cudnnStatus_t CUDNNWINAPI
cudnnRNNBackwardData(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                     const int seqLength, const cudnnTensorDescriptor_t *yDesc,
                     const void *y, const cudnnTensorDescriptor_t *dyDesc,
                     const void *dy, const cudnnTensorDescriptor_t dhyDesc,
                     const void *dhy, const cudnnTensorDescriptor_t dcyDesc,
                     const void *dcy, const cudnnFilterDescriptor_t wDesc,
                     const void *w, const cudnnTensorDescriptor_t hxDesc,
                     const void *hx, const cudnnTensorDescriptor_t cxDesc,
                     const void *cx, const cudnnTensorDescriptor_t *dxDesc,
                     void *dx, const cudnnTensorDescriptor_t dhxDesc, void *dhx,
                     const cudnnTensorDescriptor_t dcxDesc, void *dcx,
                     void *workspace, size_t workSpaceSizeInBytes,
                     void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  static auto orig_cudnnRNNBackwardData =
      (decltype(cudnnRNNBackwardData))dlsym(RTLD_NEXT, "cudnnRNNBackwardData");
  const auto tic = now();
  const auto res = orig_cudnnRNNBackwardData(
      handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc,
      dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc,
      dcx, workspace, workSpaceSizeInBytes, reserveSpace,
      reserveSpaceSizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnRNNBackwardData"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "seqLength" : to_json(seqLength),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y),
    "dyDesc" : to_json(dyDesc),
    "dy" : to_json(dy),
    "dhyDesc" : to_json(dhyDesc),
    "dhy" : to_json(dhy),
    "dcyDesc" : to_json(dcyDesc),
    "dcy" : to_json(dcy),
    "wDesc" : to_json(wDesc),
    "w" : to_json(w),
    "hxDesc" : to_json(hxDesc),
    "hx" : to_json(hx),
    "cxDesc" : to_json(cxDesc),
    "cx" : to_json(cx),
    "dxDesc" : to_json(dxDesc),
    "dx" : to_json(dx),
    "dhxDesc" : to_json(dhxDesc),
    "dhx" : to_json(dhx),
    "dcxDesc" : to_json(dcxDesc),
    "dcx" : to_json(dcx),
    "workspace" : to_json(workspace),
    "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
    "reserveSpace" : to_json(reserveSpace),
    "reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)
  });
  return res;
}

// 147
cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t *yDesc, const void *y, const void *workspace,
    size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void *dw,
    const void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  static auto orig_cudnnRNNBackwardWeights =
      (decltype(cudnnRNNBackwardWeights))dlsym(RTLD_NEXT,
                                               "cudnnRNNBackwardWeights");
  const auto tic = now();
  const auto res = orig_cudnnRNNBackwardWeights(
      handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workspace,
      workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnRNNBackwardWeights"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "seqLength" : to_json(seqLength),
    "xDesc" : to_json(xDesc),
    "x" : to_json(x),
    "hxDesc" : to_json(hxDesc),
    "hx" : to_json(hx),
    "yDesc" : to_json(yDesc),
    "y" : to_json(y),
    "workspace" : to_json(workspace),
    "workSpaceSizeInBytes" : to_json(workSpaceSizeInBytes),
    "dwDesc" : to_json(dwDesc),
    "dw" : to_json(dw),
    "reserveSpace" : to_json(reserveSpace),
    "reserveSpaceSizeInBytes" : to_json(reserveSpaceSizeInBytes)
  });
  return res;
}

// 148
cudnnStatus_t CUDNNWINAPI
cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t *ctcLossDesc) {
  static auto orig_cudnnCreateCTCLossDescriptor =
      (decltype(cudnnCreateCTCLossDescriptor))dlsym(
          RTLD_NEXT, "cudnnCreateCTCLossDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateCTCLossDescriptor(ctcLossDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreateCTCLossDescriptor"},
    "start" : tic,
    "end" : toc,
    "ctcLossDesc" : to_json(ctcLossDesc)
  });
  return res;
}

// 149
cudnnStatus_t CUDNNWINAPI cudnnSetCTCLossDescriptor(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType) {
  static auto orig_cudnnSetCTCLossDescriptor =
      (decltype(cudnnSetCTCLossDescriptor))dlsym(RTLD_NEXT,
                                                 "cudnnSetCTCLossDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetCTCLossDescriptor(ctcLossDesc, compType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetCTCLossDescriptor"},
    "start" : tic,
    "end" : toc,
    "ctcLossDesc" : to_json(ctcLossDesc),
    "compType" : to_json(compType)
  });
  return res;
}

// 150
cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossDescriptor(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType) {
  static auto orig_cudnnGetCTCLossDescriptor =
      (decltype(cudnnGetCTCLossDescriptor))dlsym(RTLD_NEXT,
                                                 "cudnnGetCTCLossDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetCTCLossDescriptor(ctcLossDesc, compType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetCTCLossDescriptor"},
    "start" : tic,
    "end" : toc,
    "ctcLossDesc" : to_json(ctcLossDesc),
    "compType" : to_json(compType)
  });
  return res;
}

// 151
cudnnStatus_t CUDNNWINAPI
cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc) {
  static auto orig_cudnnDestroyCTCLossDescriptor =
      (decltype(cudnnDestroyCTCLossDescriptor))dlsym(
          RTLD_NEXT, "cudnnDestroyCTCLossDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyCTCLossDescriptor(ctcLossDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroyCTCLossDescriptor"},
    "start" : tic,
    "end" : toc,
    "ctcLossDesc" : to_json(ctcLossDesc)
  });
  return res;
}

// 152
cudnnStatus_t CUDNNWINAPI cudnnCTCLoss(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t
        probsDesc, /* Tensor descriptor for probabilities, the dimensions are
                      T,N,A (T is the timing steps, N is the mini batch size, A
                      is the alphabet size)  */
    const void *probs,       /* probabilities after softmax, in GPU memory */
    const int *labels,       /* labels, in CPU memory */
    const int *labelLengths, /* the length of each label, in CPU memory */
    const int *inputLengths, /* the lengths of timing steps in each batch, in
                                CPU memory */
    void *costs,             /* the returned costs of CTC, in GPU memory */
    const cudnnTensorDescriptor_t
        gradientsDesc, /* Tensor descriptor for gradients, the dimensions are
                          T,N,A */
    const void *gradients,   /* the returned CTC gradients, in GPU memory, to
                                compute costs only, set it to NULL */
    cudnnCTCLossAlgo_t algo, /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc,
    void *workspace, /* pointer to the workspace, in GPU memory */
    size_t workSpaceSizeInBytes) {
  static auto orig_cudnnCTCLoss =
      (decltype(cudnnCTCLoss))dlsym(RTLD_NEXT, "cudnnCTCLoss");
  const auto tic = now();
  const auto res = orig_cudnnCTCLoss(size);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCTCLoss"},
    "start" : tic,
    "end" : toc,
    "size" : to_json(size)
  });
  return res;
}

// 153
cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossWorkspaceSize(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t
        probsDesc, /* Tensor descriptor for probabilities, the dimensions are
                      T,N,A (T is the timing steps, N is the mini batch size, A
                      is the alphabet size) */
    const cudnnTensorDescriptor_t
        gradientsDesc, /* Tensor descriptor for gradients, the dimensions are
                          T,N,A. To compute costs only, set it to NULL */
    const int *labels, /* labels, in CPU memory */
    const int *labelLengths, /* the length of each label, in CPU memory */
    const int *inputLengths, /* the lengths of timing steps in each batch, in
                                CPU memory */
    cudnnCTCLossAlgo_t algo, /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc, size_t *sizeInBytes) {
  static auto orig_cudnnGetCTCLossWorkspaceSize =
      (decltype(cudnnGetCTCLossWorkspaceSize))dlsym(
          RTLD_NEXT, "cudnnGetCTCLossWorkspaceSize");
  const auto tic = now();
  const auto res = orig_cudnnGetCTCLossWorkspaceSize(size);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetCTCLossWorkspaceSize"},
    "start" : tic,
    "end" : toc,
    "size" : to_json(size)
  });
  return res;
}

// 154
cudnnStatus_t CUDNNWINAPI
cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t *algoDesc) {
  static auto orig_cudnnCreateAlgorithmDescriptor =
      (decltype(cudnnCreateAlgorithmDescriptor))dlsym(
          RTLD_NEXT, "cudnnCreateAlgorithmDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCreateAlgorithmDescriptor(algoDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreateAlgorithmDescriptor"},
    "start" : tic,
    "end" : toc,
    "algoDesc" : to_json(algoDesc)
  });
  return res;
}

// 155
cudnnStatus_t CUDNNWINAPI cudnnSetAlgorithmDescriptor(
    cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm) {
  static auto orig_cudnnSetAlgorithmDescriptor =
      (decltype(cudnnSetAlgorithmDescriptor))dlsym(
          RTLD_NEXT, "cudnnSetAlgorithmDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnSetAlgorithmDescriptor(algoDesc, algorithm);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetAlgorithmDescriptor"},
    "start" : tic,
    "end" : toc,
    "algoDesc" : to_json(algoDesc),
    "algorithm" : to_json(algorithm)
  });
  return res;
}

// 156
cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmDescriptor(
    const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algorithm) {
  static auto orig_cudnnGetAlgorithmDescriptor =
      (decltype(cudnnGetAlgorithmDescriptor))dlsym(
          RTLD_NEXT, "cudnnGetAlgorithmDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnGetAlgorithmDescriptor(algoDesc, algorithm);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetAlgorithmDescriptor"},
    "start" : tic,
    "end" : toc,
    "algoDesc" : to_json(algoDesc),
    "algorithm" : to_json(algorithm)
  });
  return res;
}

// 157
cudnnStatus_t CUDNNWINAPI cudnnCopyAlgorithmDescriptor(
    const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest) {
  static auto orig_cudnnCopyAlgorithmDescriptor =
      (decltype(cudnnCopyAlgorithmDescriptor))dlsym(
          RTLD_NEXT, "cudnnCopyAlgorithmDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnCopyAlgorithmDescriptor(src, dest);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCopyAlgorithmDescriptor"},
    "start" : tic,
    "end" : toc,
    "src" : to_json(src),
    "dest" : to_json(dest)
  });
  return res;
}

// 158
cudnnStatus_t CUDNNWINAPI
cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc) {
  static auto orig_cudnnDestroyAlgorithmDescriptor =
      (decltype(cudnnDestroyAlgorithmDescriptor))dlsym(
          RTLD_NEXT, "cudnnDestroyAlgorithmDescriptor");
  const auto tic = now();
  const auto res = orig_cudnnDestroyAlgorithmDescriptor(algoDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroyAlgorithmDescriptor"},
    "start" : tic,
    "end" : toc,
    "algoDesc" : to_json(algoDesc)
  });
  return res;
}

// 159
cudnnStatus_t CUDNNWINAPI cudnnCreateAlgorithmPerformance(
    cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate) {
  static auto orig_cudnnCreateAlgorithmPerformance =
      (decltype(cudnnCreateAlgorithmPerformance))dlsym(
          RTLD_NEXT, "cudnnCreateAlgorithmPerformance");
  const auto tic = now();
  const auto res =
      orig_cudnnCreateAlgorithmPerformance(algoPerf, numberToCreate);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnCreateAlgorithmPerformance"},
    "start" : tic,
    "end" : toc,
    "algoPerf" : to_json(algoPerf),
    "numberToCreate" : to_json(numberToCreate)
  });
  return res;
}

// 160
cudnnStatus_t CUDNNWINAPI cudnnSetAlgorithmPerformance(
    cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t algoDesc,
    cudnnStatus_t status, float time, size_t memory) {
  static auto orig_cudnnSetAlgorithmPerformance =
      (decltype(cudnnSetAlgorithmPerformance))dlsym(
          RTLD_NEXT, "cudnnSetAlgorithmPerformance");
  const auto tic = now();
  const auto res = orig_cudnnSetAlgorithmPerformance(algoPerf, algoDesc, status,
                                                     time, memory);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetAlgorithmPerformance"},
    "start" : tic,
    "end" : toc,
    "algoPerf" : to_json(algoPerf),
    "algoDesc" : to_json(algoDesc),
    "status" : to_json(status),
    "time" : to_json(time),
    "memory" : to_json(memory)
  });
  return res;
}

// 161
cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmPerformance(
    const cudnnAlgorithmPerformance_t algoPerf,
    cudnnAlgorithmDescriptor_t *algoDesc, cudnnStatus_t *status, float *time,
    size_t *memory) {
  static auto orig_cudnnGetAlgorithmPerformance =
      (decltype(cudnnGetAlgorithmPerformance))dlsym(
          RTLD_NEXT, "cudnnGetAlgorithmPerformance");
  const auto tic = now();
  const auto res = orig_cudnnGetAlgorithmPerformance(algoPerf, algoDesc, status,
                                                     time, memory);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetAlgorithmPerformance"},
    "start" : tic,
    "end" : toc,
    "algoPerf" : to_json(algoPerf),
    "algoDesc" : to_json(algoDesc),
    "status" : to_json(status),
    "time" : to_json(time),
    "memory" : to_json(memory)
  });
  return res;
}

// 162
cudnnStatus_t CUDNNWINAPI cudnnDestroyAlgorithmPerformance(
    cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy) {
  static auto orig_cudnnDestroyAlgorithmPerformance =
      (decltype(cudnnDestroyAlgorithmPerformance))dlsym(
          RTLD_NEXT, "cudnnDestroyAlgorithmPerformance");
  const auto tic = now();
  const auto res =
      orig_cudnnDestroyAlgorithmPerformance(algoPerf, numberToDestroy);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnDestroyAlgorithmPerformance"},
    "start" : tic,
    "end" : toc,
    "algoPerf" : to_json(algoPerf),
    "numberToDestroy" : to_json(numberToDestroy)
  });
  return res;
}

// 163
cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmSpaceSize(
    cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc,
    size_t *algoSpaceSizeInBytes) {
  static auto orig_cudnnGetAlgorithmSpaceSize =
      (decltype(cudnnGetAlgorithmSpaceSize))dlsym(RTLD_NEXT,
                                                  "cudnnGetAlgorithmSpaceSize");
  const auto tic = now();
  const auto res =
      orig_cudnnGetAlgorithmSpaceSize(handle, algoDesc, algoSpaceSizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetAlgorithmSpaceSize"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "algoDesc" : to_json(algoDesc),
    "algoSpaceSizeInBytes" : to_json(algoSpaceSizeInBytes)
  });
  return res;
}

// 164
cudnnStatus_t CUDNNWINAPI
cudnnSaveAlgorithm(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc,
                   void *algoSpace, size_t algoSpaceSizeInBytes) {
  static auto orig_cudnnSaveAlgorithm =
      (decltype(cudnnSaveAlgorithm))dlsym(RTLD_NEXT, "cudnnSaveAlgorithm");
  const auto tic = now();
  const auto res = orig_cudnnSaveAlgorithm(handle, algoDesc, algoSpace,
                                           algoSpaceSizeInBytes);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSaveAlgorithm"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "algoDesc" : to_json(algoDesc),
    "algoSpace" : to_json(algoSpace),
    "algoSpaceSizeInBytes" : to_json(algoSpaceSizeInBytes)
  });
  return res;
}

// 165
cudnnStatus_t CUDNNWINAPI cudnnRestoreAlgorithm(
    cudnnHandle_t handle, void *algoSpace, size_t algoSpaceSizeInBytes,
    cudnnAlgorithmDescriptor_t algoDesc) {
  static auto orig_cudnnRestoreAlgorithm =
      (decltype(cudnnRestoreAlgorithm))dlsym(RTLD_NEXT,
                                             "cudnnRestoreAlgorithm");
  const auto tic = now();
  const auto res = orig_cudnnRestoreAlgorithm(handle, algoSpace,
                                              algoSpaceSizeInBytes, algoDesc);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnRestoreAlgorithm"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "algoSpace" : to_json(algoSpace),
    "algoSpaceSizeInBytes" : to_json(algoSpaceSizeInBytes),
    "algoDesc" : to_json(algoDesc)
  });
  return res;
}

// 166
cudnnStatus_t CUDNNWINAPI cudnnSetCallback(unsigned mask, void *udata,
                                           cudnnCallback_t fptr) {
  static auto orig_cudnnSetCallback =
      (decltype(cudnnSetCallback))dlsym(RTLD_NEXT, "cudnnSetCallback");
  const auto tic = now();
  const auto res = orig_cudnnSetCallback(mask, udata, fptr);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetCallback"},
    "start" : tic,
    "end" : toc,
    "mask" : to_json(mask),
    "udata" : to_json(udata),
    "fptr" : to_json(fptr)
  });
  return res;
}

// 167
cudnnStatus_t CUDNNWINAPI cudnnGetCallback(unsigned *mask, void **udata,
                                           cudnnCallback_t *fptr) {
  static auto orig_cudnnGetCallback =
      (decltype(cudnnGetCallback))dlsym(RTLD_NEXT, "cudnnGetCallback");
  const auto tic = now();
  const auto res = orig_cudnnGetCallback(mask, udata, fptr);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnGetCallback"},
    "start" : tic,
    "end" : toc,
    "mask" : to_json(mask),
    "udata" : to_json(udata),
    "fptr" : to_json(fptr)
  });
  return res;
}

// 168
cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v6(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int hiddenSize,
    const int numLayers, cudnnDropoutDescriptor_t dropoutDesc,
    cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction,
    cudnnRNNMode_t mode, cudnnRNNAlgo_t algo, cudnnDataType_t dataType) {
  static auto orig_cudnnSetRNNDescriptor_v6 =
      (decltype(cudnnSetRNNDescriptor_v6))dlsym(RTLD_NEXT,
                                                "cudnnSetRNNDescriptor_v6");
  const auto tic = now();
  const auto res = orig_cudnnSetRNNDescriptor_v6(
      handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction,
      mode, algo, dataType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetRNNDescriptor_v6"},
    "start" : tic,
    "end" : toc,
    "handle" : to_json(handle),
    "rnnDesc" : to_json(rnnDesc),
    "hiddenSize" : to_json(hiddenSize),
    "numLayers" : to_json(numLayers),
    "dropoutDesc" : to_json(dropoutDesc),
    "inputMode" : to_json(inputMode),
    "direction" : to_json(direction),
    "mode" : to_json(mode),
    "algo" : to_json(algo),
    "dataType" : to_json(dataType)
  });
  return res;
}

// 169
cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v5(
    cudnnRNNDescriptor_t rnnDesc, int hiddenSize, int numLayers,
    cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
    cudnnDirectionMode_t direction, cudnnRNNMode_t mode,
    cudnnDataType_t dataType) {
  static auto orig_cudnnSetRNNDescriptor_v5 =
      (decltype(cudnnSetRNNDescriptor_v5))dlsym(RTLD_NEXT,
                                                "cudnnSetRNNDescriptor_v5");
  const auto tic = now();
  const auto res =
      orig_cudnnSetRNNDescriptor_v5(rnnDesc, hiddenSize, numLayers, dropoutDesc,
                                    inputMode, direction, mode, dataType);
  const auto toc = now();

  callback({
    "funName" : std::string{"cudnnSetRNNDescriptor_v5"},
    "start" : tic,
    "end" : toc,
    "rnnDesc" : to_json(rnnDesc),
    "hiddenSize" : to_json(hiddenSize),
    "numLayers" : to_json(numLayers),
    "dropoutDesc" : to_json(dropoutDesc),
    "inputMode" : to_json(inputMode),
    "direction" : to_json(direction),
    "mode" : to_json(mode),
    "dataType" : to_json(dataType)
  });
  return res;
}
