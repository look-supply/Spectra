// ============================================================
// CUDA kernel stubs
//
// On macOS there is no CUDA runtime, so these are all no-ops.
// On Linux / Windows, replace this file with a real .cu
// implementation that mirrors the Metal kernels.
// ============================================================

#include "CudaKernels.h"

#ifdef __APPLE__
// ---- macOS stubs (CUDA not available) -----------------------

bool  cudaInit()                                   { return false; }
void  cudaRelease()                                {}
void* cudaUploadLut(const float*, int)             { return nullptr; }
void  cudaFreeLut(void*)                           {}

void cudaDispatchGrain(void*, void*, void*,
                       void*, void*, void*,
                       const GrainParams&) {}

#else
// ---- Linux / Windows real implementation --------------------
// TODO: Implement CUDA kernels matching Metal shader logic.
//       Use the same GrainParams struct and algorithm ports
//       from the shared FilmKernelCore.inc implementation.
//
// A skeleton is provided below for the build to succeed.

#include <cstdio>

bool cudaInit() {
    fprintf(stderr, "LookSupplySpectra: CUDA kernels not yet implemented\n");
    return false;
}
void  cudaRelease()                                {}
void* cudaUploadLut(const float*, int)             { return nullptr; }
void  cudaFreeLut(void*)                           {}

void cudaDispatchGrain(void*, void*, void*,
                       void*, void*, void*,
                       const GrainParams&) {}

#endif
