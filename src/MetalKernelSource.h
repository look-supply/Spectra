#ifndef METAL_KERNEL_SOURCE_H
#define METAL_KERNEL_SOURCE_H

// ============================================================
// Look Supply Film — Metal runtime shader source.
//
// ALL Film shader math lives in FilmKernelCore.inc and is shared
// with the (future) CUDA path. This file:
//   1. Pulls in the core text as a C++ string literal embedded by
//      CMake via configure_file from FilmKernelCore.inc.
//   2. Prepends a Metal-specific PREFIX that #includes <metal_stdlib>
//      and defines the platform-abstraction macros the core uses.
//   3. Appends a Metal-specific SUFFIX with the `kernel void` entry
//      points that forward to the DEVFN body functions in the core.
//
// To keep Metal and CUDA in sync, edit FilmKernelCore.inc ONLY.
// Only touch this file if you are genuinely adding a new platform
// macro, kernel entry point, or buffer binding.
// ============================================================

#include <string>
#include "FilmKernelCoreText.h"   // generated: `filmKernelCoreSource`

namespace {

static const char* kFilmMetalPrefix = R"METAL_PREFIX(
#include <metal_stdlib>
using namespace metal;

#define DEVFN             inline
#define GLOBAL_RO(T)      device const T*
#define GLOBAL_RW(T)      device T*
#define CONSTANT_REF(T)   constant T&
#define READ_TEX          texture2d<float, access::sample>
#define WRITE_TEX         texture2d<float, access::write>
#define MAKE_FLOAT2(a,b)        float2((a),(b))
#define MAKE_FLOAT3(a,b,c)      float3((a),(b),(c))
#define MAKE_FLOAT4(a,b,c,d)    float4((a),(b),(c),(d))
#define MAKE_INT2(a,b)          int2((a),(b))
#define MAKE_INT3(a,b,c)        int3((a),(b),(c))
#define MAKE_INT4(a,b,c,d)      int4((a),(b),(c),(d))
#define EXP2      precise::exp2
#define LOG2      precise::log2
#define FAST_EXP  fast::exp
#define FAST_SQRT fast::sqrt
#define FAST_POW  fast::pow
#define FAST_RSQRT(x) rsqrt(x)

constexpr sampler film_linear_sampler(filter::linear, address::clamp_to_edge);

#define TEX_SAMPLE(t,u,v)  (t).sample(film_linear_sampler, float2((u),(v)))
#define TEX_WRITE(t,x,y,v) (t).write((v), uint2(uint(x), uint(y)))
)METAL_PREFIX";

static const char* kFilmMetalSuffix = R"METAL_SUFFIX(
kernel void cwLinearizeKernel(
    device const float* input                                  [[buffer(0)]],
    texture2d<float, access::write> linOut                     [[texture(0)]],
    constant GrainParams& p                                    [[buffer(2)]],
    device const float* inputCstLutData                        [[buffer(3)]],
    uint2 gid                                                  [[thread_position_in_grid]])
{
    int x = int(gid.x), y = int(gid.y);
    if (x >= p.width || y >= p.height) return;
    cwLinearizeKernel_body(x, y, input, linOut, p, inputCstLutData);
}

kernel void cwHalationPrepassKernel(
    texture2d<float, access::sample> linIn     [[texture(0)]],
    texture2d<float, access::sample> halBlurIn [[texture(2)]],
    texture2d<float, access::write>  halLinOut [[texture(1)]],
    constant GrainParams& p                    [[buffer(2)]],
    uint2 gid                                  [[thread_position_in_grid]])
{
    cwHalationPrepassKernel_body(int(gid.x), int(gid.y), linIn, halBlurIn, halLinOut, p);
}

kernel void cwHalationBlurHKernel(
    texture2d<float, access::sample> linIn       [[texture(0)]],
    texture2d<float, access::write>  halBlurHOut [[texture(1)]],
    constant GrainParams& p                      [[buffer(2)]],
    uint2 gid                                    [[thread_position_in_grid]])
{
    cwHalationBlurHKernel_body(int(gid.x), int(gid.y), linIn, halBlurHOut, p);
}

kernel void cwHalationBlurVKernel(
    texture2d<float, access::sample> halBlurHIn [[texture(0)]],
    texture2d<float, access::write>  halBlurOut [[texture(1)]],
    constant GrainParams& p                     [[buffer(2)]],
    uint2 gid                                   [[thread_position_in_grid]])
{
    cwHalationBlurVKernel_body(int(gid.x), int(gid.y), halBlurHIn, halBlurOut, p);
}

kernel void cwDiffusionBlurHKernel(
    texture2d<float, access::sample> scatterIn  [[texture(0)]],
    texture2d<float, access::write>  scatterOut [[texture(1)]],
    constant GrainParams& p                     [[buffer(2)]],
    uint2 gid                                   [[thread_position_in_grid]])
{
    cwDiffusionBlurHKernel_body(int(gid.x), int(gid.y), scatterIn, scatterOut, p);
}

kernel void cwDiffusionBlurVKernel(
    texture2d<float, access::sample> scatterIn  [[texture(0)]],
    texture2d<float, access::write>  scatterOut [[texture(1)]],
    constant GrainParams& p                     [[buffer(2)]],
    uint2 gid                                   [[thread_position_in_grid]])
{
    cwDiffusionBlurVKernel_body(int(gid.x), int(gid.y), scatterIn, scatterOut, p);
}

kernel void cwDiffusionScatterKernel(
    texture2d<float, access::sample> linInput                  [[texture(0)]],
    texture2d<float, access::write> diffusionBlurOut           [[texture(1)]],
    texture2d<float, access::write> diffusionGlowOut           [[texture(2)]],
    constant GrainParams& p                                    [[buffer(2)]],
    device const float* diffusionSampleTable                   [[buffer(7)]],
    uint2 gid                                                  [[thread_position_in_grid]])
{
    int x = int(gid.x), y = int(gid.y);
    cwDiffusionScatterKernel_body(x, y, linInput, diffusionBlurOut, diffusionGlowOut, p,
                                  diffusionSampleTable);
}

kernel void cwFilmBlurHKernel(
    device const float* input                              [[buffer(0)]],
    texture2d<float, access::write> hOut                   [[texture(0)]],
    constant GrainParams& p                                [[buffer(2)]],
    uint2 gid                                              [[thread_position_in_grid]])
{
    cwFilmBlurHKernel_body(input, hOut, int(gid.x), int(gid.y), p);
}

kernel void cwFilmBlurVKernel(
    texture2d<float, access::sample> hTex                  [[texture(0)]],
    texture2d<float, access::write>  vOut                  [[texture(1)]],
    constant GrainParams& p                                [[buffer(2)]],
    uint2 gid                                              [[thread_position_in_grid]])
{
    cwFilmBlurVKernel_body(hTex, vOut, int(gid.x), int(gid.y), p);
}

kernel void cwGrainKernel(
    device const float* input                            [[buffer(0)]],
    device float*       output                           [[buffer(1)]],
    constant GrainParams& p                              [[buffer(2)]],
    device const float* lutData                          [[buffer(3)]],
    device const float* cstLutData                       [[buffer(4)]],
    device const float* inputCstLutData                  [[buffer(5)]],
    device const float* curveData                        [[buffer(6)]],
    texture2d<float, access::sample> linInput            [[texture(0)]],
    texture2d<float, access::sample> diffusionBlurInput  [[texture(1)]],
    texture2d<float, access::sample> diffusionGlowInput  [[texture(2)]],
    texture2d<float, access::sample> filmBlurInput       [[texture(3)]],
    texture2d<float, access::sample> halBlurInput        [[texture(4)]],
    uint2 gid                                            [[thread_position_in_grid]])
{
    int x = int(gid.x), y = int(gid.y);
    if (x >= p.width || y >= p.height) return;
    cwGrainKernel_body(x, y, input, output, p, lutData, cstLutData, inputCstLutData, curveData,
                       linInput, diffusionBlurInput, diffusionGlowInput, filmBlurInput, halBlurInput);
}

)METAL_SUFFIX";

inline const std::string& getMetalKernelSource() {
    static const std::string s =
        std::string(kFilmMetalPrefix) +
        std::string(filmKernelCoreSource) +
        std::string(kFilmMetalSuffix);
    return s;
}

} // namespace

#endif // METAL_KERNEL_SOURCE_H
