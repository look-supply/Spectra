#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include "GrainParams.h"

// ============================================================
// CUDA dispatch interface — mirrors MetalKernels.h
// On macOS (no CUDA), every function is a no-op / returns false.
// ============================================================

bool cudaInit();
void cudaRelease();
void* cudaUploadLut(const float* data, int lutSize);
void  cudaFreeLut(void* lutDevPtr);

void cudaDispatchGrain(void* streamPtr,
                       void* srcPtr, void* dstPtr,
                       void* lutPtr,
                       void* outputCstPtr,
                       void* inputCstPtr,
                       const GrainParams& params);

#endif // CUDA_KERNELS_H
