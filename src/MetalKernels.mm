#ifdef __APPLE__

#import <Metal/Metal.h>
#include <cmath>
#include <string>
#include "MetalKernels.h"
#include "MetalKernelSource.h"

// ============================================================
// Module-level Metal state (shared across all instances)
// ============================================================
static id<MTLDevice>               s_selfDevice      = nil;
static id<MTLCommandQueue>         s_selfQueue       = nil;
static id<MTLLibrary>              s_metalLibrary    = nil;
static id<MTLComputePipelineState> s_cwPipeline      = nil;
static id<MTLComputePipelineState> s_linearizePipeline = nil;
static id<MTLComputePipelineState> s_diffusionScatterPipeline = nil;
static id<MTLComputePipelineState> s_diffusionBlurHPipeline   = nil;
static id<MTLComputePipelineState> s_diffusionBlurVPipeline   = nil;
static id<MTLComputePipelineState> s_halationPrepassPipeline  = nil;
static id<MTLComputePipelineState> s_filmBlurHPipeline = nil;
static id<MTLComputePipelineState> s_filmBlurVPipeline = nil;
static id<MTLComputePipelineState> s_halBlurHPipeline  = nil;
static id<MTLComputePipelineState> s_halBlurVPipeline  = nil;

// Reusable scratch texture for the linearize prepass (rgba16Float, sized
// to the current render frame). Shared across all instances; safe
// because Resolve serialises render calls per host queue.
static id<MTLTexture>              s_linearizeTex    = nil;
static int                         s_linearizeTexW   = 0;
static int                         s_linearizeTexH   = 0;
static id<MTLTexture>              s_diffusionBlurTex = nil;
static id<MTLTexture>              s_diffusionGlowTex = nil;
static id<MTLTexture>              s_diffusionScatterHTex = nil;  // ping-pong for scatter blur
static int                         s_diffusionScatterTexW = 0;
static int                         s_diffusionScatterTexH = 0;
static id<MTLTexture>              s_halationLinTex   = nil;
static int                         s_halationLinTexW  = 0;
static int                         s_halationLinTexH  = 0;
static id<MTLTexture>              s_halBlurHTex      = nil;
static int                         s_halBlurHTexW     = 0;
static int                         s_halBlurHTexH     = 0;
static id<MTLTexture>              s_halBlurTex       = nil;
static int                         s_halBlurTexW      = 0;
static int                         s_halBlurTexH      = 0;
static id<MTLTexture>              s_filmBlurHTex     = nil;
static id<MTLTexture>              s_filmBlurTex      = nil;
static int                         s_filmBlurTexW     = 0;
static int                         s_filmBlurTexH     = 0;

static inline float cwClampFloat(float value, float lo, float hi) {
    return fminf(fmaxf(value, lo), hi);
}

static inline float cwDiToLinearCPU(float value) {
    constexpr float kA = 0.0075f;
    constexpr float kB = 7.0f;
    constexpr float kC = 0.07329248f;
    constexpr float kM = 10.44426855f;
    constexpr float kLogCut = 0.02740668f;
    return (value > kLogCut) ? exp2f(value / kC - kB) - kA : value / kM;
}

// ============================================================
// Diffusion PSF — sum of isotropic 2D exponentials
// PSF(r, λ) ∝ exp(-r/λ)   (λ² norm cancels with table normalisation)
// ============================================================

struct DiffusionPsfFamily {
    float core_lambda_um;
    float halo_lambda_um;
    float bloom_lambda_um;
    float core_spread;    // geometric spread factor
    float halo_spread;
    float bloom_spread;
    int   core_n;         // number of components in each group
    int   halo_n;
    int   bloom_n;
    float bloom_alpha;    // power-law exponent for bloom weights
    float w_core;         // group energy fractions (sum = 1)
    float w_halo;
    float w_bloom;
    float halo_warmth_base; // family-intrinsic warmth level
};

// warmth_axis for R, G, B channels (positive = warm/red, negative = cool/blue)
static const float kWarmthAxis[3] = { +1.30f, +0.15f, -1.45f };

static const DiffusionPsfFamily kDiffusionPsfFamilies[4] = {
    // core_λ  halo_λ  bloom_λ  cSpr hSpr bSpr cn hn bn  α     wc    wh    wb    warmth_base
    {  10.0f,  50.0f,  260.0f, 1.5f, 2.0f, 2.5f, 2, 3, 4, 3.2f, 0.60f, 0.30f, 0.10f, 0.0f  }, // glimmerglass
    {  16.0f,  95.0f,  380.0f, 1.5f, 2.0f, 2.5f, 2, 3, 4, 3.5f, 0.40f, 0.47f, 0.13f, 0.65f }, // black_pro_mist
    {  14.0f, 150.0f,  650.0f, 1.5f, 2.0f, 2.5f, 2, 3, 4, 2.9f, 0.28f, 0.42f, 0.30f, 0.40f }, // pro_mist
    {  20.0f, 200.0f, 1000.0f, 1.5f, 2.0f, 2.5f, 2, 3, 4, 2.5f, 0.22f, 0.30f, 0.48f, 0.85f }, // cinebloom
};

// Expand geometric lambda progression into lambdas_px[n]:
//   λ_k = center_px * spread^(2*k/(n-1) - 1),  k = 0..n-1
static void expandLambdas(float center_px, float spread, int n, float* lambdas_px) {
    for (int k = 0; k < n; ++k) {
        float exp_arg = (n > 1) ? (2.0f * k / (float)(n - 1) - 1.0f) : 0.0f;
        lambdas_px[k] = center_px * powf(spread, exp_arg);
    }
}

// Compute bloom component weights: w_k ∝ λ_k^(2 - alpha), normalised
static void bloomWeights(const float* lambdas_px, int n, float alpha, float* weights) {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        weights[k] = powf(lambdas_px[k], 2.0f - alpha);
        sum += weights[k];
    }
    float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    for (int k = 0; k < n; ++k) weights[k] *= inv;
}

// Compute per-channel, per-component halo warmth weights.
// positions = linspace(-1, 1, n_halo), raw_w_c_k = exp(pos_k * axis_c * warmth)
// Output: [channel][component], normalised per channel.
static void haloWarmthWeights(int n, float total_warmth, float weights[3][4]) {
    for (int c = 0; c < 3; ++c) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            float pos = (n > 1) ? (2.0f * k / (float)(n - 1) - 1.0f) : 0.0f;
            weights[c][k] = expf(pos * kWarmthAxis[c] * total_warmth);
            sum += weights[c][k];
        }
        float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
        for (int k = 0; k < n; ++k) weights[c][k] *= inv;
    }
}

// Evaluate the full per-channel PSF at radius r_px for one sample.
static void diffusionPsfSample(float r_px,
                               const DiffusionPsfFamily& fam,
                               float scale_px_per_um,
                               const float halo_w[3][4],
                               const float* core_lambdas_px, const float* halo_lambdas_px,
                               const float* bloom_lambdas_px, const float* bloom_w,
                               float* out_r, float* out_g, float* out_b) {
    // Core group (uniform weights, same for all channels)
    float core_val = 0.0f;
    for (int k = 0; k < fam.core_n; ++k)
        core_val += expf(-r_px / fmaxf(core_lambdas_px[k], 0.01f));
    core_val /= (float)fam.core_n;

    // Bloom group (power-law weights, same for all channels)
    float bloom_val = 0.0f;
    for (int k = 0; k < fam.bloom_n; ++k)
        bloom_val += bloom_w[k] * expf(-r_px / fmaxf(bloom_lambdas_px[k], 0.01f));

    // Halo group (per-channel warmth weights)
    float halo_r = 0.0f, halo_g = 0.0f, halo_b = 0.0f;
    for (int k = 0; k < fam.halo_n; ++k) {
        float e = expf(-r_px / fmaxf(halo_lambdas_px[k], 0.01f));
        halo_r += halo_w[0][k] * e;
        halo_g += halo_w[1][k] * e;
        halo_b += halo_w[2][k] * e;
    }

    *out_r = fam.w_core * core_val + fam.w_halo * halo_r + fam.w_bloom * bloom_val;
    *out_g = fam.w_core * core_val + fam.w_halo * halo_g + fam.w_bloom * bloom_val;
    *out_b = fam.w_core * core_val + fam.w_halo * halo_b + fam.w_bloom * bloom_val;
}

static size_t fillDiffusionSampleTable(const GrainParams& params, float* table) {
    constexpr int kMaxSamples = 192;

    int family = params.diffusionPsfFamily;
    if (family < 0 || family >= 4) family = 0;
    const DiffusionPsfFamily& fam = kDiffusionPsfFamilies[family];

    float haloRadius = fmaxf(params.diffusionSize, 1.0f);
    int sampleCount = (int)cwClampFloat(96.0f + haloRadius * 0.55f, 112.0f, (float)kMaxSamples);

    // Spatial scale: bloom center at diffusionSize/2 pixels
    float scale_px_per_um = haloRadius / (2.0f * fam.bloom_lambda_um);

    // Expand lambda arrays (max 4 components each)
    float core_lambdas_px[2], halo_lambdas_px[3], bloom_lambdas_px[4];
    expandLambdas(fam.core_lambda_um  * scale_px_per_um, fam.core_spread,  fam.core_n,  core_lambdas_px);
    expandLambdas(fam.halo_lambda_um  * scale_px_per_um, fam.halo_spread,  fam.halo_n,  halo_lambdas_px);
    expandLambdas(fam.bloom_lambda_um * scale_px_per_um, fam.bloom_spread, fam.bloom_n, bloom_lambdas_px);

    // Bloom power-law weights
    float bloom_w[4];
    bloomWeights(bloom_lambdas_px, fam.bloom_n, fam.bloom_alpha, bloom_w);

    // Halo warmth per-channel weights
    // total_warmth = family base + user warmth offset (diffusionHaloWarmth stores this sum)
    float halo_w[3][4] = {};
    haloWarmthWeights(fam.halo_n, params.diffusionHaloWarmth, halo_w);

    // Threshold in scene-linear
    float thresholdLin = fmaxf(cwDiToLinearCPU(cwClampFloat(params.diffusionHighlightGate, 0.0f, 1.0f)), 0.0f);

    // --- Build center sample (r=0) with 1/64 density scaling ---
    float cR, cG, cB;
    diffusionPsfSample(0.0f, fam, scale_px_per_um, halo_w,
                       core_lambdas_px, halo_lambdas_px, bloom_lambdas_px, bloom_w,
                       &cR, &cG, &cB);
    cR *= 0.015625f; cG *= 0.015625f; cB *= 0.015625f;

    // New header: [0]=count [1]=cW_r [2]=cW_g [3]=cW_b [4]=threshold [5]=pad
    table[0] = (float)sampleCount;
    table[1] = cR;
    table[2] = cG;
    table[3] = cB;
    table[4] = thresholdLin;
    table[5] = 0.0f;

    float wsumR = cR, wsumG = cG, wsumB = cB;

    for (int s = 0; s < sampleCount; ++s) {
        float fi = (float)s;
        float sampleRadius = sqrtf((fi + 0.5f) / (float)sampleCount) * haloRadius;
        float wr, wg, wb;
        diffusionPsfSample(sampleRadius, fam, scale_px_per_um, halo_w,
                           core_lambdas_px, halo_lambdas_px, bloom_lambdas_px, bloom_w,
                           &wr, &wg, &wb);
        int ti = 6 + s * 4;
        table[ti]     = sampleRadius;
        table[ti + 1] = wr;
        table[ti + 2] = wg;
        table[ti + 3] = wb;
        wsumR += wr; wsumG += wg; wsumB += wb;
    }

    // Normalise per channel so weights sum to 1
    float invR = (wsumR > 0.0f) ? 1.0f / wsumR : 0.0f;
    float invG = (wsumG > 0.0f) ? 1.0f / wsumG : 0.0f;
    float invB = (wsumB > 0.0f) ? 1.0f / wsumB : 0.0f;
    table[1] *= invR; table[2] *= invG; table[3] *= invB;
    for (int s = 0; s < sampleCount; ++s) {
        int ti = 6 + s * 4;
        table[ti + 1] *= invR;
        table[ti + 2] *= invG;
        table[ti + 3] *= invB;
    }

    return (size_t)(6 + sampleCount * 4) * sizeof(float);
}

static inline int diffusionScatterScaleForRadius(float radius) {
    if (radius >= 56.0f) return 4;
    if (radius >= 32.0f) return 3;
    return 2;
}

// ============================================================
// metalInit — compile shaders from embedded source
// ============================================================
bool metalInit(void* cmdQueuePtr) {
    @autoreleasepool {
        // If fully initialised, return fast
        if (s_metalLibrary && s_cwPipeline) return true;

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)cmdQueuePtr;
        if (!queue) { NSLog(@"LookSupplySpectra: metalInit called with nil queue"); return false; }
        id<MTLDevice> device = queue.device;
        if (!device) { NSLog(@"LookSupplySpectra: queue.device is nil"); return false; }

        NSError* error = nil;

        if (!s_metalLibrary) {
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            if (@available(macOS 15.0, *)) {
                opts.mathMode = MTLMathModeFast;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
            }

            const std::string& src = getMetalKernelSource();
            s_metalLibrary = [device newLibraryWithSource:
                                [NSString stringWithUTF8String:src.c_str()]
                              options:opts
                              error:&error];
            if (!s_metalLibrary) {
                NSLog(@"LookSupplySpectra: Metal library compile failed: %@", error);
                return false;
            }
            NSLog(@"LookSupplySpectra: Metal library compiled successfully");
        }

        // CW pipeline
        if (!s_cwPipeline) {
            id<MTLFunction> cwFunc = [s_metalLibrary newFunctionWithName:@"cwGrainKernel"];
            if (cwFunc)
                s_cwPipeline = [device newComputePipelineStateWithFunction:cwFunc error:&error];
            if (!s_cwPipeline)
                NSLog(@"LookSupplySpectra: CW pipeline creation failed: %@", error);
        }

        // Linearize prepass pipeline
        if (!s_linearizePipeline) {
            id<MTLFunction> linFunc = [s_metalLibrary newFunctionWithName:@"cwLinearizeKernel"];
            if (linFunc)
                s_linearizePipeline = [device newComputePipelineStateWithFunction:linFunc error:&error];
            if (!s_linearizePipeline)
                NSLog(@"LookSupplySpectra: Linearize pipeline creation failed: %@", error);
        }

        // Half-resolution diffusion scatter prepass pipeline
        if (!s_diffusionScatterPipeline) {
            id<MTLFunction> diffusionFunc = [s_metalLibrary newFunctionWithName:@"cwDiffusionScatterKernel"];
            if (diffusionFunc)
                s_diffusionScatterPipeline = [device newComputePipelineStateWithFunction:diffusionFunc error:&error];
            if (!s_diffusionScatterPipeline)
                NSLog(@"LookSupplySpectra: Diffusion scatter pipeline creation failed: %@", error);
        }

        // Scatter texture smoothing pipelines (H and V separable Gaussian)
        if (!s_diffusionBlurHPipeline) {
            id<MTLFunction> f = [s_metalLibrary newFunctionWithName:@"cwDiffusionBlurHKernel"];
            if (f)
                s_diffusionBlurHPipeline = [device newComputePipelineStateWithFunction:f error:&error];
            if (!s_diffusionBlurHPipeline)
                NSLog(@"LookSupplySpectra: Diffusion blur H pipeline creation failed: %@", error);
        }
        if (!s_diffusionBlurVPipeline) {
            id<MTLFunction> f = [s_metalLibrary newFunctionWithName:@"cwDiffusionBlurVKernel"];
            if (f)
                s_diffusionBlurVPipeline = [device newComputePipelineStateWithFunction:f error:&error];
            if (!s_diffusionBlurVPipeline)
                NSLog(@"LookSupplySpectra: Diffusion blur V pipeline creation failed: %@", error);
        }

        // Full-resolution halation prepass pipeline
        if (!s_halationPrepassPipeline) {
            id<MTLFunction> halFunc = [s_metalLibrary newFunctionWithName:@"cwHalationPrepassKernel"];
            if (halFunc)
                s_halationPrepassPipeline = [device newComputePipelineStateWithFunction:halFunc error:&error];
            if (!s_halationPrepassPipeline)
                NSLog(@"LookSupplySpectra: Halation prepass pipeline creation failed: %@", error);
        }

        // Film blur separable prepass pipelines (H and V passes)
        if (!s_filmBlurHPipeline) {
            id<MTLFunction> fbhFunc = [s_metalLibrary newFunctionWithName:@"cwFilmBlurHKernel"];
            if (fbhFunc)
                s_filmBlurHPipeline = [device newComputePipelineStateWithFunction:fbhFunc error:&error];
            if (!s_filmBlurHPipeline)
                NSLog(@"LookSupplySpectra: Film blur H pipeline creation failed: %@", error);
        }
        if (!s_filmBlurVPipeline) {
            id<MTLFunction> fbvFunc = [s_metalLibrary newFunctionWithName:@"cwFilmBlurVKernel"];
            if (fbvFunc)
                s_filmBlurVPipeline = [device newComputePipelineStateWithFunction:fbvFunc error:&error];
            if (!s_filmBlurVPipeline)
                NSLog(@"LookSupplySpectra: Film blur V pipeline creation failed: %@", error);
        }

        // Halation blur separable prepass pipelines (H then V — O(radius) each)
        if (!s_halBlurHPipeline) {
            id<MTLFunction> hbhFunc = [s_metalLibrary newFunctionWithName:@"cwHalationBlurHKernel"];
            if (hbhFunc)
                s_halBlurHPipeline = [device newComputePipelineStateWithFunction:hbhFunc error:&error];
            if (!s_halBlurHPipeline)
                NSLog(@"LookSupplySpectra: Halation blur H pipeline creation failed: %@", error);
        }
        if (!s_halBlurVPipeline) {
            id<MTLFunction> hbvFunc = [s_metalLibrary newFunctionWithName:@"cwHalationBlurVKernel"];
            if (hbvFunc)
                s_halBlurVPipeline = [device newComputePipelineStateWithFunction:hbvFunc error:&error];
            if (!s_halBlurVPipeline)
                NSLog(@"LookSupplySpectra: Halation blur V pipeline creation failed: %@", error);
        }

        return (s_cwPipeline != nil);
    }
}

// ============================================================
// metalRelease — tear down pipelines
// ============================================================
void metalRelease() {
    @autoreleasepool {
        s_cwPipeline   = nil;
        s_linearizePipeline = nil;
        s_diffusionScatterPipeline = nil;
        s_diffusionBlurHPipeline   = nil;
        s_diffusionBlurVPipeline   = nil;
        s_halationPrepassPipeline = nil;
        s_filmBlurHPipeline = nil;
        s_filmBlurVPipeline = nil;
        s_halBlurHPipeline = nil;
        s_halBlurVPipeline = nil;
        s_linearizeTex = nil;
        s_linearizeTexW = 0;
        s_linearizeTexH = 0;
        s_diffusionBlurTex = nil;
        s_diffusionGlowTex = nil;
        s_diffusionScatterHTex = nil;
        s_diffusionScatterTexW = 0;
        s_diffusionScatterTexH = 0;
        s_halationLinTex = nil;
        s_halationLinTexW = 0;
        s_halationLinTexH = 0;
        s_halBlurHTex = nil;
        s_halBlurHTexW = 0;
        s_halBlurHTexH = 0;
        s_halBlurTex = nil;
        s_halBlurTexW = 0;
        s_halBlurTexH = 0;
        s_filmBlurHTex = nil;
        s_filmBlurTex  = nil;
        s_filmBlurTexW = 0;
        s_filmBlurTexH = 0;
        s_metalLibrary = nil;
    }
}

// ============================================================
// LUT upload / free
// ============================================================
void* metalUploadLut(void* cmdQueuePtr, const float* data, int lutSize) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)cmdQueuePtr;
        if (!queue || !data || lutSize <= 0) return nullptr;
        id<MTLDevice> device = queue.device;

        size_t bytes = (size_t)lutSize * lutSize * lutSize * 3 * sizeof(float);
        id<MTLBuffer> buf = [device newBufferWithBytes:data
                                    length:bytes
                                    options:MTLResourceStorageModeShared];
        if (!buf) return nullptr;
        // Transfer ownership to the caller (bridging retain)
        return (void*)CFBridgingRetain(buf);
    }
}

void metalFreeLut(void* lutBufferPtr) {
    if (lutBufferPtr) {
        // Release the retained reference
        CFBridgingRelease(lutBufferPtr);
    }
}

// ============================================================
// Kernel dispatch helper
// ============================================================
static void metalDispatch(id<MTLComputePipelineState> pipeline,
                          void* cmdQueuePtr,
                          void* srcBufPtr, void* dstBufPtr,
                          void* lutBufPtr,
                          void* outputCstBufPtr,
                          void* inputCstBufPtr,
                          const GrainParams& params,
                          const float* curveData = nullptr,
                          bool waitForCompletion = false)
{
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)cmdQueuePtr;
        id<MTLBuffer> srcBuf = (__bridge id<MTLBuffer>)srcBufPtr;
        id<MTLBuffer> dstBuf = (__bridge id<MTLBuffer>)dstBufPtr;

        id<MTLDevice> device = queue.device;

        // LUT buffer — use the uploaded one, or a tiny dummy
        id<MTLBuffer> lutBuf;
        if (lutBufPtr) {
            lutBuf = (__bridge id<MTLBuffer>)lutBufPtr;
        } else {
            // Minimal 4-byte dummy buffer so the kernel binding is valid
            static id<MTLBuffer> sDummy = nil;
            if (!sDummy)
                sDummy = [device newBufferWithLength:16
                                 options:MTLResourceStorageModeShared];
            lutBuf = sDummy;
        }

        // CST LUT buffer — use the uploaded one, or a tiny dummy
        id<MTLBuffer> cstLutBuf;
        if (outputCstBufPtr) {
            cstLutBuf = (__bridge id<MTLBuffer>)outputCstBufPtr;
        } else {
            static id<MTLBuffer> sCstDummy = nil;
            if (!sCstDummy)
                sCstDummy = [device newBufferWithLength:16
                                    options:MTLResourceStorageModeShared];
            cstLutBuf = sCstDummy;
        }

        // Input CST LUT buffer — use the uploaded one, or a tiny dummy
        id<MTLBuffer> inputCstBuf;
        if (inputCstBufPtr) {
            inputCstBuf = (__bridge id<MTLBuffer>)inputCstBufPtr;
        } else {
            static id<MTLBuffer> sInputCstDummy = nil;
            if (!sInputCstDummy)
                sInputCstDummy = [device newBufferWithLength:16
                                         options:MTLResourceStorageModeShared];
            inputCstBuf = sInputCstDummy;
        }

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        GrainParams dispatchParams = params;
        dispatchParams.diffusionScatterMode = 0;
        dispatchParams.diffusionScatterWidth = 0;
        dispatchParams.diffusionScatterHeight = 0;
        dispatchParams.filmBlurPrepassEnabled = 0;
        dispatchParams.halBlurPrepassEnabled = 0;

        static id<MTLTexture> sDummyTex = nil;
        if (!sDummyTex) {
            MTLTextureDescriptor* d =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                   width:1
                                                                  height:1
                                                               mipmapped:NO];
            d.usage = MTLTextureUsageShaderRead;
            d.storageMode = MTLStorageModePrivate;
            sDummyTex = [device newTextureWithDescriptor:d];
        }

        // ---- Linearize prepass (only when diffusion/halation actually contributes) ----
        // Writes (CST → di_to_linear)(input) into an rgba16Float texture so
        // diffusion/halation taps can use hardware-bilinear texture sampling.
        bool needDiffusionPrepass = params.diffusionEnabled &&
                               (params.diffusionAmount > 1e-6f || params.diffusionVeil > 1e-6f);
        bool needPrepass = s_linearizePipeline &&
                           (needDiffusionPrepass ||
                            (params.halEnabled &&
                             (params.halOpacity > 1e-6f)));
        id<MTLTexture> linTex = nil;
        if (needPrepass) {
            if (!s_linearizeTex ||
                s_linearizeTexW != params.width ||
                s_linearizeTexH != params.height) {
                MTLTextureDescriptor* desc =
                    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                       width:(NSUInteger)params.width
                                                                      height:(NSUInteger)params.height
                                                                   mipmapped:NO];
                desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
                desc.storageMode = MTLStorageModePrivate;
                s_linearizeTex = [device newTextureWithDescriptor:desc];
                s_linearizeTexW = params.width;
                s_linearizeTexH = params.height;
            }
            linTex = s_linearizeTex;

            id<MTLComputeCommandEncoder> linEnc = [cmdBuf computeCommandEncoder];
            [linEnc setComputePipelineState:s_linearizePipeline];
            [linEnc setBuffer:srcBuf offset:0 atIndex:0];
            [linEnc setTexture:linTex atIndex:0];
            [linEnc setBytes:&dispatchParams length:sizeof(GrainParams) atIndex:2];
            [linEnc setBuffer:inputCstBuf offset:0 atIndex:3];

            NSUInteger linTw = s_linearizePipeline.threadExecutionWidth;
            NSUInteger linTh = s_linearizePipeline.maxTotalThreadsPerThreadgroup / linTw;
            MTLSize linTg = MTLSizeMake(linTw, linTh, 1);
            MTLSize linGrid = MTLSizeMake((NSUInteger)params.width,
                                          (NSUInteger)params.height, 1);
            [linEnc dispatchThreads:linGrid threadsPerThreadgroup:linTg];
            [linEnc endEncoding];
        } else {
            // Bind a tiny dummy texture so texture(0) has a valid resource.
            linTex = sDummyTex;
        }

        id<MTLTexture> diffusionBlurTex = sDummyTex;
        id<MTLTexture> filmBlurOutTex = sDummyTex;

        // ---- Film blur separable prepass (H then V) ----
        // Runs when film blur is active; produces a blurred DI texture the main
        // kernel samples in one tap, replacing the former 49-tap in-kernel loop.
        bool needFilmBlurPrepass = s_filmBlurHPipeline &&
                                   s_filmBlurVPipeline &&
                                   params.image_blur >= 0.01f;
        if (needFilmBlurPrepass) {
            if (!s_filmBlurHTex || !s_filmBlurTex ||
                s_filmBlurTexW != params.width ||
                s_filmBlurTexH != params.height) {
                MTLTextureDescriptor* desc =
                    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                       width:(NSUInteger)params.width
                                                                      height:(NSUInteger)params.height
                                                                   mipmapped:NO];
                desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
                desc.storageMode = MTLStorageModePrivate;
                s_filmBlurHTex = [device newTextureWithDescriptor:desc];
                s_filmBlurTex  = [device newTextureWithDescriptor:desc];
                s_filmBlurTexW = params.width;
                s_filmBlurTexH = params.height;
            }

            MTLSize fbGrid = MTLSizeMake((NSUInteger)params.width, (NSUInteger)params.height, 1);

            // H pass: input buffer → s_filmBlurHTex
            id<MTLComputeCommandEncoder> fbhEnc = [cmdBuf computeCommandEncoder];
            [fbhEnc setComputePipelineState:s_filmBlurHPipeline];
            [fbhEnc setBuffer:srcBuf           offset:0 atIndex:0];
            [fbhEnc setTexture:s_filmBlurHTex  atIndex:0];
            [fbhEnc setBytes:&dispatchParams   length:sizeof(GrainParams) atIndex:2];
            NSUInteger fbhTw = s_filmBlurHPipeline.threadExecutionWidth;
            NSUInteger fbhTh = s_filmBlurHPipeline.maxTotalThreadsPerThreadgroup / fbhTw;
            [fbhEnc dispatchThreads:fbGrid threadsPerThreadgroup:MTLSizeMake(fbhTw, fbhTh, 1)];
            [fbhEnc endEncoding];

            // V pass: s_filmBlurHTex → s_filmBlurTex
            id<MTLComputeCommandEncoder> fbvEnc = [cmdBuf computeCommandEncoder];
            [fbvEnc setComputePipelineState:s_filmBlurVPipeline];
            [fbvEnc setTexture:s_filmBlurHTex  atIndex:0];
            [fbvEnc setTexture:s_filmBlurTex   atIndex:1];
            [fbvEnc setBytes:&dispatchParams   length:sizeof(GrainParams) atIndex:2];
            NSUInteger fbvTw = s_filmBlurVPipeline.threadExecutionWidth;
            NSUInteger fbvTh = s_filmBlurVPipeline.maxTotalThreadsPerThreadgroup / fbvTw;
            [fbvEnc dispatchThreads:fbGrid threadsPerThreadgroup:MTLSizeMake(fbvTw, fbvTh, 1)];
            [fbvEnc endEncoding];

            filmBlurOutTex = s_filmBlurTex;
            dispatchParams.filmBlurPrepassEnabled = 1;
        }
        id<MTLTexture> diffusionGlowTex = sDummyTex;

        // ---- Halation blur separable prepass (H pass then V pass) ----
        // Replaces the O(radius²) 2D exponential blur with two O(radius) 1D passes.
        // At halSize=1.5 (radius≈7): 15+15=30 taps instead of 225. ~7x speedup.
        bool needHalBlurPrepass = s_halBlurHPipeline && s_halBlurVPipeline &&
                                  params.halEnabled &&
                                  params.halOpacity > 1e-6f &&
                                  linTex != sDummyTex;
        id<MTLTexture> halBlurTex = sDummyTex;

        if (needHalBlurPrepass) {
            // Allocate H intermediate and final textures if needed
            if (!s_halBlurHTex ||
                s_halBlurHTexW != params.width ||
                s_halBlurHTexH != params.height) {
                MTLTextureDescriptor* desc =
                    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                       width:(NSUInteger)params.width
                                                                      height:(NSUInteger)params.height
                                                                   mipmapped:NO];
                desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
                desc.storageMode = MTLStorageModePrivate;
                s_halBlurHTex  = [device newTextureWithDescriptor:desc];
                s_halBlurHTexW = params.width;
                s_halBlurHTexH = params.height;
            }
            if (!s_halBlurTex ||
                s_halBlurTexW != params.width ||
                s_halBlurTexH != params.height) {
                MTLTextureDescriptor* desc =
                    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                       width:(NSUInteger)params.width
                                                                      height:(NSUInteger)params.height
                                                                   mipmapped:NO];
                desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
                desc.storageMode = MTLStorageModePrivate;
                s_halBlurTex  = [device newTextureWithDescriptor:desc];
                s_halBlurTexW = params.width;
                s_halBlurTexH = params.height;
            }

            MTLSize halBlurGrid = MTLSizeMake((NSUInteger)params.width, (NSUInteger)params.height, 1);

            // H pass: linTex → s_halBlurHTex
            id<MTLComputeCommandEncoder> hbhEnc = [cmdBuf computeCommandEncoder];
            [hbhEnc setComputePipelineState:s_halBlurHPipeline];
            [hbhEnc setTexture:linTex        atIndex:0];
            [hbhEnc setTexture:s_halBlurHTex atIndex:1];
            [hbhEnc setBytes:&dispatchParams length:sizeof(GrainParams) atIndex:2];
            NSUInteger hbhTw = s_halBlurHPipeline.threadExecutionWidth;
            NSUInteger hbhTh = s_halBlurHPipeline.maxTotalThreadsPerThreadgroup / hbhTw;
            [hbhEnc dispatchThreads:halBlurGrid threadsPerThreadgroup:MTLSizeMake(hbhTw, hbhTh, 1)];
            [hbhEnc endEncoding];

            // V pass: s_halBlurHTex → s_halBlurTex
            id<MTLComputeCommandEncoder> hbvEnc = [cmdBuf computeCommandEncoder];
            [hbvEnc setComputePipelineState:s_halBlurVPipeline];
            [hbvEnc setTexture:s_halBlurHTex atIndex:0];
            [hbvEnc setTexture:s_halBlurTex  atIndex:1];
            [hbvEnc setBytes:&dispatchParams length:sizeof(GrainParams) atIndex:2];
            NSUInteger hbvTw = s_halBlurVPipeline.threadExecutionWidth;
            NSUInteger hbvTh = s_halBlurVPipeline.maxTotalThreadsPerThreadgroup / hbvTw;
            [hbvEnc dispatchThreads:halBlurGrid threadsPerThreadgroup:MTLSizeMake(hbvTw, hbvTh, 1)];
            [hbvEnc endEncoding];

            halBlurTex = s_halBlurTex;
            dispatchParams.halBlurPrepassEnabled = 1;
        }

        // ---- Halation prepass (when both halation and diffusion are active) ----
        // The scatter prepass computes PSF(halated_image) so the diffusion glow
        // captures halation-boosted highlights. Diffusion is still applied before
        // halation in the main kernel (lens filter before film artifact), but the
        // scatter is seeded from the halated brightness to preserve visual strength.
        bool needHalationPrepass = needDiffusionPrepass &&
                                   s_halationPrepassPipeline &&
                                   params.halEnabled &&
                                   params.halOpacity > 1e-6f;
        id<MTLTexture> scatterInputTex = linTex;  // default: original scene-linear

        if (needHalationPrepass) {
            if (!s_halationLinTex ||
                s_halationLinTexW != params.width ||
                s_halationLinTexH != params.height) {
                MTLTextureDescriptor* desc =
                    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                       width:(NSUInteger)params.width
                                                                      height:(NSUInteger)params.height
                                                                   mipmapped:NO];
                desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
                desc.storageMode = MTLStorageModePrivate;
                s_halationLinTex = [device newTextureWithDescriptor:desc];
                s_halationLinTexW = params.width;
                s_halationLinTexH = params.height;
            }

            id<MTLComputeCommandEncoder> halEnc = [cmdBuf computeCommandEncoder];
            [halEnc setComputePipelineState:s_halationPrepassPipeline];
            [halEnc setTexture:linTex          atIndex:0];
            [halEnc setTexture:s_halationLinTex atIndex:1];
            [halEnc setTexture:halBlurTex       atIndex:2];
            [halEnc setBytes:&dispatchParams length:sizeof(GrainParams) atIndex:2];

            NSUInteger halTw = s_halationPrepassPipeline.threadExecutionWidth;
            NSUInteger halTh = s_halationPrepassPipeline.maxTotalThreadsPerThreadgroup / halTw;
            [halEnc dispatchThreads:MTLSizeMake((NSUInteger)params.width, (NSUInteger)params.height, 1)
                threadsPerThreadgroup:MTLSizeMake(halTw, halTh, 1)];
            [halEnc endEncoding];

            scatterInputTex = s_halationLinTex;
        }

        bool useDiffusionScatterPass = needDiffusionPrepass &&
                                       s_diffusionScatterPipeline &&
                                       params.diffusionSize >= 12.0f;
        if (useDiffusionScatterPass) {
            int scatterScale = diffusionScatterScaleForRadius(params.diffusionSize);
            int scatterW = (params.width + scatterScale - 1) / scatterScale;
            int scatterH = (params.height + scatterScale - 1) / scatterScale;
            if (scatterW < 1) scatterW = 1;
            if (scatterH < 1) scatterH = 1;

            if (!s_diffusionBlurTex || !s_diffusionGlowTex || !s_diffusionScatterHTex ||
                s_diffusionScatterTexW != scatterW ||
                s_diffusionScatterTexH != scatterH) {
                MTLTextureDescriptor* desc =
                    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                       width:(NSUInteger)scatterW
                                                                      height:(NSUInteger)scatterH
                                                                   mipmapped:NO];
                desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
                desc.storageMode = MTLStorageModePrivate;
                s_diffusionBlurTex     = [device newTextureWithDescriptor:desc];
                s_diffusionGlowTex     = [device newTextureWithDescriptor:desc];
                s_diffusionScatterHTex = [device newTextureWithDescriptor:desc];
                s_diffusionScatterTexW = scatterW;
                s_diffusionScatterTexH = scatterH;
            }

            dispatchParams.diffusionScatterMode = 1;
            dispatchParams.diffusionScatterWidth = scatterW;
            dispatchParams.diffusionScatterHeight = scatterH;

            float diffusionSampleTable[6 + 192 * 4];  // per-channel: 6 header + 4 floats/sample
            size_t diffusionSampleTableBytes = fillDiffusionSampleTable(dispatchParams, diffusionSampleTable);

            id<MTLComputeCommandEncoder> diffusionEnc = [cmdBuf computeCommandEncoder];
            [diffusionEnc setComputePipelineState:s_diffusionScatterPipeline];
            [diffusionEnc setTexture:scatterInputTex  atIndex:0];
            [diffusionEnc setTexture:s_diffusionBlurTex atIndex:1];
            [diffusionEnc setTexture:s_diffusionGlowTex atIndex:2];
            [diffusionEnc setBytes:&dispatchParams length:sizeof(GrainParams) atIndex:2];
            [diffusionEnc setBytes:diffusionSampleTable length:diffusionSampleTableBytes atIndex:7];

            NSUInteger diffusionTw = s_diffusionScatterPipeline.threadExecutionWidth;
            NSUInteger diffusionTh = s_diffusionScatterPipeline.maxTotalThreadsPerThreadgroup / diffusionTw;
            MTLSize diffusionTg = MTLSizeMake(diffusionTw, diffusionTh, 1);
            MTLSize diffusionGrid = MTLSizeMake((NSUInteger)scatterW, (NSUInteger)scatterH, 1);
            [diffusionEnc dispatchThreads:diffusionGrid threadsPerThreadgroup:diffusionTg];
            [diffusionEnc endEncoding];

            // Smooth the scatter textures with a separable Gaussian (H then V)
            // to eliminate Monte Carlo block-boundary artifacts before upsampling.
            if (s_diffusionBlurHPipeline && s_diffusionBlurVPipeline && s_diffusionScatterHTex) {
                auto dispatchScatterBlur = [&](id<MTLTexture> srcTex,
                                               id<MTLTexture> dstTex) {
                    // H pass: srcTex → s_diffusionScatterHTex
                    id<MTLComputeCommandEncoder> hEnc = [cmdBuf computeCommandEncoder];
                    [hEnc setComputePipelineState:s_diffusionBlurHPipeline];
                    [hEnc setTexture:srcTex                atIndex:0];
                    [hEnc setTexture:s_diffusionScatterHTex atIndex:1];
                    [hEnc setBytes:&dispatchParams length:sizeof(GrainParams) atIndex:2];
                    NSUInteger htw = s_diffusionBlurHPipeline.threadExecutionWidth;
                    NSUInteger hth = s_diffusionBlurHPipeline.maxTotalThreadsPerThreadgroup / htw;
                    [hEnc dispatchThreads:diffusionGrid threadsPerThreadgroup:MTLSizeMake(htw, hth, 1)];
                    [hEnc endEncoding];
                    // V pass: s_diffusionScatterHTex → dstTex
                    id<MTLComputeCommandEncoder> vEnc = [cmdBuf computeCommandEncoder];
                    [vEnc setComputePipelineState:s_diffusionBlurVPipeline];
                    [vEnc setTexture:s_diffusionScatterHTex atIndex:0];
                    [vEnc setTexture:dstTex                atIndex:1];
                    [vEnc setBytes:&dispatchParams length:sizeof(GrainParams) atIndex:2];
                    NSUInteger vtw = s_diffusionBlurVPipeline.threadExecutionWidth;
                    NSUInteger vth = s_diffusionBlurVPipeline.maxTotalThreadsPerThreadgroup / vtw;
                    [vEnc dispatchThreads:diffusionGrid threadsPerThreadgroup:MTLSizeMake(vtw, vth, 1)];
                    [vEnc endEncoding];
                };
                dispatchScatterBlur(s_diffusionBlurTex, s_diffusionBlurTex);
                dispatchScatterBlur(s_diffusionGlowTex, s_diffusionGlowTex);
            }

            diffusionBlurTex = s_diffusionBlurTex;
            diffusionGlowTex = s_diffusionGlowTex;
        }

        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:srcBuf offset:0 atIndex:0];
        [enc setBuffer:dstBuf offset:0 atIndex:1];
        [enc setBytes:&dispatchParams length:sizeof(GrainParams) atIndex:2];
        [enc setBuffer:lutBuf offset:0 atIndex:3];
        [enc setBuffer:cstLutBuf offset:0 atIndex:4];
        [enc setBuffer:inputCstBuf offset:0 atIndex:5];

        // Curve data — 768 floats (256 R + 256 G + 256 B) = 3072 bytes (< 4KB setBytes limit)
        static const float zeroCurve[768] = {0};
        const float* curveSrc = curveData ? curveData : zeroCurve;
        [enc setBytes:curveSrc length:768 * sizeof(float) atIndex:6];

        [enc setTexture:linTex atIndex:0];
        [enc setTexture:diffusionBlurTex atIndex:1];
        [enc setTexture:diffusionGlowTex atIndex:2];
        [enc setTexture:filmBlurOutTex   atIndex:3];
        [enc setTexture:halBlurTex       atIndex:4];

        NSUInteger tw = pipeline.threadExecutionWidth;
        NSUInteger th = pipeline.maxTotalThreadsPerThreadgroup / tw;
        MTLSize threadGroupSize = MTLSizeMake(tw, th, 1);
        MTLSize gridSize = MTLSizeMake((NSUInteger)params.width,
                                       (NSUInteger)params.height, 1);

        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [enc endEncoding];
        [cmdBuf commit];
        if (waitForCompletion) {
            [cmdBuf waitUntilCompleted];
        }
    }
}

void metalDispatchGrain(void* cmdQueuePtr,
                        void* srcBufPtr, void* dstBufPtr,
                        void* lutBufPtr,
                        void* outputCstBufPtr,
                        void* inputCstBufPtr,
                        const GrainParams& params,
                        const float* curveData,
                        bool waitForCompletion) {
    if (s_cwPipeline)
        metalDispatch(s_cwPipeline, cmdQueuePtr, srcBufPtr, dstBufPtr,
                      lutBufPtr, outputCstBufPtr, inputCstBufPtr, params,
                      curveData, waitForCompletion);
}

// ============================================================
// Self-hosted Metal queue — created on the system default device
// when the host does not provide a Metal context (metalEnabled=0).
// ============================================================
void* metalGetSelfHostedQueue(void) {
    @autoreleasepool {
        if (s_selfQueue) return (__bridge void*)s_selfQueue;
        s_selfDevice = MTLCreateSystemDefaultDevice();
        if (!s_selfDevice) {
            NSLog(@"LookSupplySpectra: no Metal device found for self-hosted queue");
            return nullptr;
        }
        s_selfQueue = [s_selfDevice newCommandQueue];
        if (!s_selfQueue) {
            NSLog(@"LookSupplySpectra: failed to create self-hosted command queue");
            return nullptr;
        }
        return (__bridge void*)s_selfQueue;
    }
}

// ============================================================
// metalDispatchGrainFromCpu
// Wraps plain CPU float* buffers in shared MTLBuffers, runs the
// full grain kernel, then copies the result back to dstCpuPtr.
// Called when the OFX host does not enable Metal for this render.
// ============================================================
bool metalDispatchGrainFromCpu(void* srcCpuPtr, void* dstCpuPtr,
                               void* lutBufPtr,
                               void* outputCstBufPtr,
                               void* inputCstBufPtr,
                               const GrainParams& params,
                               const float* curveData) {
    @autoreleasepool {
        if (!srcCpuPtr || !dstCpuPtr) {
            NSLog(@"LookSupplySpectra: metalDispatchGrainFromCpu: null src/dst ptr");
            return false;
        }

        void* queuePtr = metalGetSelfHostedQueue();
        if (!queuePtr) {
            NSLog(@"LookSupplySpectra: metalDispatchGrainFromCpu: no self-hosted queue");
            return false;
        }
        bool initOk = metalInit(queuePtr);
        if (!initOk) {
            NSLog(@"LookSupplySpectra: metalDispatchGrainFromCpu: metalInit failed");
            return false;
        }

        id<MTLCommandQueue> queue  = (__bridge id<MTLCommandQueue>)queuePtr;
        id<MTLDevice>       device = queue.device;

        // Use srcHeight (not height) for the source buffer: adjustment/timeline layers can
        // supply a source image taller than the destination render rect.  Using params.height
        // would truncate the upload, causing filmResolveSample to read off the end of the
        // MTLBuffer (returning zeros / garbage) for pixels near the top of the frame.
        size_t srcBytes = (size_t)params.srcHeight * (size_t)params.srcRowBytes;
        size_t dstBytes = (size_t)params.height * (size_t)params.dstRowBytes;

        // Shared buffers: GPU-accessible, no separate copy-back needed on Apple Silicon.
        id<MTLBuffer> srcBuf = [device newBufferWithBytes:srcCpuPtr
                                                   length:srcBytes
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> dstBuf = [device newBufferWithLength:dstBytes
                                                   options:MTLResourceStorageModeShared];
        if (!srcBuf || !dstBuf) {
            NSLog(@"LookSupplySpectra: metalDispatchGrainFromCpu: src/dst buffer alloc failed srcBuf=%p dstBuf=%p",
                  (void*)srcBuf, (void*)dstBuf);
            return false;
        }

        metalDispatch(s_cwPipeline, queuePtr,
                      (__bridge void*)srcBuf, (__bridge void*)dstBuf,
                      lutBufPtr, outputCstBufPtr, inputCstBufPtr,
                      params, curveData, /*waitForCompletion=*/true);

        // Copy active pixels only; row padding belongs to the host.
        for (int y = 0; y < params.height; ++y)
            memcpy((char*)dstCpuPtr + size_t(y)*params.dstRowBytes,
                   (char*)[dstBuf contents] + size_t(y)*params.dstRowBytes,
                   size_t(params.width)*4*sizeof(float));
        return true;
    }
}

#endif // __APPLE__
