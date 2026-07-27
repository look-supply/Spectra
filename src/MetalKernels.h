#ifndef METAL_KERNELS_H
#define METAL_KERNELS_H

#include "GrainParams.h"

#ifdef __APPLE__

// Initialize Metal resources (compile shaders). Call once at plugin load.
// cmdQueuePtr: id<MTLCommandQueue> as void* (device is extracted from queue).
// Returns true on success.
bool metalInit(void* cmdQueuePtr);

// Release Metal resources. Call at plugin unload.
void metalRelease();

// Returns a self-hosted id<MTLCommandQueue> as void* — created on the system
// default device if the host does not provide one. Never nullptr on Apple
// Silicon; nullptr only if no Metal device exists on the machine.
void* metalGetSelfHostedQueue(void);

// Upload look data to a Metal buffer.
// cmdQueuePtr: id<MTLCommandQueue> as void* (device is extracted from queue).
// Returns an opaque handle (id<MTLBuffer> as void*), or nullptr on failure.
void* metalUploadLut(void* cmdQueuePtr, const float* data, int lutSize);

// Free a previously uploaded look buffer.
void metalFreeLut(void* lutBufferPtr);

// Dispatch the grain kernel.
//   cmdQueuePtr:       id<MTLCommandQueue> as void*
//   srcBufPtr:         id<MTLBuffer> (source image) as void*
//   dstBufPtr:         id<MTLBuffer> (output image) as void*
//   lutBufPtr:         id<MTLBuffer> (look data) as void*, or nullptr
//   outputCstBufPtr:   id<MTLBuffer> (output CST LUT) as void*, or nullptr
//   inputCstBufPtr:    id<MTLBuffer> (input CST LUT) as void*, or nullptr
//   params:            filled GrainParams struct
void metalDispatchGrain(void* cmdQueuePtr,
                        void* srcBufPtr, void* dstBufPtr,
                        void* lutBufPtr,
                        void* outputCstBufPtr,
                        void* inputCstBufPtr,
                        const GrainParams& params,
                        const float* curveData = nullptr,
                        bool waitForCompletion = false);

// Self-hosted grain dispatch: srcCpuPtr/dstCpuPtr are plain CPU float* buffers
// (host did NOT enable Metal). Wraps them in shared MTLBuffers, runs the full
// grain kernel, copies result back to dstCpuPtr. Returns true on success.
// lutBufPtr / outputCstBufPtr / inputCstBufPtr may be host-side Metal buffers
// (id<MTLBuffer> as void*) from a prior metalUploadLut call, or nullptr.
bool metalDispatchGrainFromCpu(void* srcCpuPtr, void* dstCpuPtr,
                               void* lutBufPtr,
                               void* outputCstBufPtr,
                               void* inputCstBufPtr,
                               const GrainParams& params,
                               const float* curveData = nullptr);

#endif // __APPLE__
#endif // METAL_KERNELS_H
