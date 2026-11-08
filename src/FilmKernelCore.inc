// ============================================================
// Look Supply Film — shared GPU kernel source (single source of truth).
//
// THIS FILE IS THE SHARED MATH FOR ALL FILM SHADER PASSES.
// It is written against a small abstraction-macro layer so the same
// text compiles on both Metal (as a runtime-compiled string) and
// CUDA (as a .cu file #include).
//
// HOW IT IS CONSUMED
//   Metal : CMake embeds this file's text into FilmKernelCoreText.h
//           as a C++ string literal. MetalKernelSource.h prepends a
//           Metal-specific PREFIX (defining the abstraction macros
//           for MSL) and appends a SUFFIX with the `kernel void`
//           entry shims that forward to the DEVFN body functions
//           defined here. The final string is handed to
//           MTLDevice newLibraryWithSource.
//
//   CUDA  : A future CudaKernels.cu file does
//              #include "CudaKernelSupport.h"   // helper_math, macros
//              #include "FilmKernelCore.inc"
//              #include "CudaKernelEntries.h"   // __global__ wrappers
//           and compiles via nvcc normally.
//
// REQUIRED MACROS (must be defined by the platform wrapper before
// this file is consumed):
//   DEVFN              qualifier for device-scope inline functions
//   GLOBAL_RO(T)       read-only global memory pointer type
//   GLOBAL_RW(T)       read/write global memory pointer type
//   CONSTANT_REF(T)    small constant-memory parameter reference
//   READ_TEX           sampleable 2D float4 texture type
//   WRITE_TEX          writeable 2D float4 texture type
//   TEX_SAMPLE(t,u,v)  sample a READ_TEX at normalised UV, returns float4
//   TEX_WRITE(t,x,y,v) write a float4 to a WRITE_TEX at integer pixel
//   MAKE_FLOAT2/3/4(...) and MAKE_INT2/3/4(...) constructor macros
//   EXP2 / LOG2        highest-precision exp2/log2 available
//
// DO NOT add Metal-specific (`precise::`, `.sample(`, `access::write`,
// `kernel void`, `[[buffer(N)]]`, `device`, `constant`) or CUDA-specific
// (`__device__`, `tex2D`, `surf2Dwrite`) tokens directly to this file.
// Use the abstraction macros above. If you need a new abstraction,
// add it to BOTH platform prefixes and document it in the list above.
// ============================================================

// FAST_* fall back to standard math on platforms that don't define them
// (e.g. CUDA). Metal defines fast::exp/sqrt/pow which are ~2-4× faster
// at ≤0.1% relative error — acceptable for grain and blur math.
#ifndef FAST_EXP
#  define FAST_EXP  exp
#endif
#ifndef FAST_SQRT
#  define FAST_SQRT sqrt
#endif
#ifndef FAST_POW
#  define FAST_POW  pow
#endif
#ifndef FAST_RSQRT
#  define FAST_RSQRT(x) (1.0f / FAST_SQRT(x))
#endif
// ============================================================
// Shared param struct — must match GrainParams in C++ exactly
// ============================================================
struct GrainParams {
    float grain_size;
    float grain_strength;
    float grain_blur;
    float grain_opacity;
    float grain_sat;
    float image_blur;
    float hi_rolloff;
    int   seed_offset;
    int   frame_index;
    int   width;
    int   height;
    int   srcWidth;
    int   srcHeight;
    int   srcX1;
    int   srcY1;
    int   dstX1;
    int   dstY1;
    int   srcRowBytes;
    int   dstRowBytes;
    int   lutEnabled;
    int   lutSize;
    float lutIntensity;
    float lutContrastAmt;   // look contrast/luma blend 0..1
    float lutPushPull;
    int   halEnabled;
    float halSize;          // exponential blur r0 (0.0-5.0)
    float halColor;         // green pass-through (0.0-0.5)
    float halMethod;        // 0.0 = Proosa, 1.0 = Minuth
    float _halReserved0;
    float _halReserved1;
    float _halReserved2;
    float _halReserved3;
    float _halReserved4;
    float _halReserved5;
    float halOpacity;
    float diffusionHaloWarmth; // family_base + user warmth offset (0..2)
    int   diffusionPsfFamily;  // 0=glimmerglass 1=bpm 2=pro_mist 3=cinebloom
    int   diffusionEnabled;
    float diffusionAmount;
    float diffusionSize;
    float diffusionSoftness;
    float diffusionBodyStrength;
    float diffusionFalloffStrength;
    float diffusionHighlightGate;
    float diffusionVeil;
    float diffusionContrastReduction;
    float diffusionHighlightBoost;    // extra glow gain for blowout on practicals 0..1
    float diffusionTintStrength;
    float diffusionTintR;
    float diffusionTintG;
    float diffusionTintB;
    int   diffusionScatterMode;
    int   diffusionScatterWidth;
    int   diffusionScatterHeight;
    int   printEnabled;
    float printGainR;
    float printGainG;
    float printGainB;
    int   wpEnabled;       // 0 = identity, 1 = apply Bradford CAT below
    float wpMat[9];        // Bradford CAT DWG-linear→DWG-linear, row-major 3×3
    int   outTransform;    // 0=none, 1..N from discovered CST output LUTs
    int   cstLutSize;      // output CST LUT dimension, 0 = no CST
    int   inTransform;     // 0=none (DWG/DI), 1..N from discovered CST input LUTs
    int   inputCstLutSize; // input CST LUT dimension, 0 = no input CST
    // ---- Split Toning ----
    int   splitEnabled;
    float splitStrength;
    float splitShadowHue;
    float splitHighlightHue;
    float splitLoVsHi;
    float splitPivot;
    float splitSubLo;
    float splitSubHi;
    float splitNeutralWidth;
    float splitFilmDensity; // Beer-Lambert optical density multiplier (0.2..3.0)
    float splitKMStrength;  // Kubelka-Munk hue drift 0..1
    float splitHiProtect;   // highlight protection 0..1
    float splitDyeSat;      // dye saturation / target-chroma scale 0.1..2.0
    float splitCrossBleed;  // cross-zone hue bleed 0..0.5
    int   curveEnabled;
    float renderScale;        // OFX render scale (1.0 = full res)
    float shape_irregularity; // 0..1 crystal size/density variation
    float grain_blur_link;    // 0..1 amount base-plate resolution loss tracks grain size
    float grain_contrast;     // 0..1 grain histogram crunch (0 = neutral)
    int   debugMode;
    int   filmBlurPrepassEnabled; // 1 = separable prepass ran; sample filmBlurInput tex
    int   halBlurPrepassEnabled;  // 1 = halation blur prepass ran; read halBlurInput tex
};

// ============================================================
// Pixel read helper — clamp-to-edge, RGBA float buffer
// ============================================================
DEVFN float readPx(GLOBAL_RO(float) buf, int x, int y,
                    int w, int h, int rowBytes, int comp) {
    x = clamp(x, 0, w - 1);
    y = clamp(y, 0, h - 1);
    int floatsPerRow = rowBytes / 4;
    return buf[y * floatsPerRow + x * 4 + comp];
}

// Bilinear read — sub-pixel accurate, clamp-to-edge
DEVFN float readPxBilinear(GLOBAL_RO(float) buf, float fx, float fy,
                             int w, int h, int rowBytes, int comp) {
    int x0 = int(floor(fx));  int x1 = x0 + 1;
    int y0 = int(floor(fy));  int y1 = y0 + 1;
    float tx = fx - float(x0);
    float ty = fy - float(y0);
    float v00 = readPx(buf, x0, y0, w, h, rowBytes, comp);
    float v10 = readPx(buf, x1, y0, w, h, rowBytes, comp);
    float v01 = readPx(buf, x0, y1, w, h, rowBytes, comp);
    float v11 = readPx(buf, x1, y1, w, h, rowBytes, comp);
    return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

// Vectorised 3-channel bilinear read — computes corner addresses once for R/G/B.
// ~3× fewer address computations than calling readPxBilinear() per channel.
DEVFN float3 readPxBilinear3(GLOBAL_RO(float) buf, float fx, float fy,
                               int w, int h, int rowBytes) {
    int x0 = clamp(int(floor(fx)),     0, w - 1);
    int x1 = clamp(int(floor(fx)) + 1, 0, w - 1);
    int y0 = clamp(int(floor(fy)),     0, h - 1);
    int y1 = clamp(int(floor(fy)) + 1, 0, h - 1);
    float tx = fx - floor(fx);
    float ty = fy - floor(fy);
    int row = rowBytes / 4;
    int i00 = y0 * row + x0 * 4;
    int i10 = y0 * row + x1 * 4;
    int i01 = y1 * row + x0 * 4;
    int i11 = y1 * row + x1 * 4;
    float3 v00 = MAKE_FLOAT3(buf[i00], buf[i00 + 1], buf[i00 + 2]);
    float3 v10 = MAKE_FLOAT3(buf[i10], buf[i10 + 1], buf[i10 + 2]);
    float3 v01 = MAKE_FLOAT3(buf[i01], buf[i01 + 1], buf[i01 + 2]);
    float3 v11 = MAKE_FLOAT3(buf[i11], buf[i11 + 1], buf[i11 + 2]);
    return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

// ============================================================
// PRNG — FNV-1a combine + xxhash finaliser
// ============================================================
DEVFN uint cw_hash(uint seed) {
    seed ^= seed >> 15u;
    seed *= 0x2C1B3C6Du;
    seed ^= seed >> 12u;
    seed *= 0x297A2D39u;
    seed ^= seed >> 15u;
    return seed;
}

DEVFN uint cw_cell_seed(int cx, int cy, uint offset) {
    uint h = offset ^ 0x811C9DC5u;
    h = (h ^ (uint(cx) & 0xFFFFu)) * 0x01000193u;
    h = (h ^ (uint(cy) & 0xFFFFu)) * 0x01000193u;
    return cw_hash(h);
}

// ============================================================
// Value noise — quintic hermite (C2 continuity)
// Returns [-0.5, 0.5]
// ============================================================
DEVFN float cw_vnoise(float x, float y, uint seed) {
    int ix = int(floor(x));
    int iy = int(floor(y));
    float fx = x - float(ix);
    float fy = y - float(iy);

    float ux = fx*fx*fx * (fx * (fx * 6.0f - 15.0f) + 10.0f);
    float uy = fy*fy*fy * (fy * (fy * 6.0f - 15.0f) + 10.0f);

    float h00 = float(cw_cell_seed(ix,   iy,   seed)) * 2.3283064365e-10f;
    float h10 = float(cw_cell_seed(ix+1, iy,   seed)) * 2.3283064365e-10f;
    float h01 = float(cw_cell_seed(ix,   iy+1, seed)) * 2.3283064365e-10f;
    float h11 = float(cw_cell_seed(ix+1, iy+1, seed)) * 2.3283064365e-10f;

    float v0 = h00 + ux * (h10 - h00);
    float v1 = h01 + ux * (h11 - h01);
    return (v0 + uy * (v1 - v0)) - 0.5f;
}

// ============================================================
// Film-resolution transfer (base-plate detail loss)
// ============================================================
// Single-channel film blur (kept for any future per-channel use).
DEVFN float filmResolveSample(GLOBAL_RO(float) buf, int px, int py,
                              int w, int h, int rowBytes, int comp, float amount) {
    if (amount < 0.01f) return readPx(buf, px, py, w, h, rowBytes, comp);

    float center = readPx(buf, px, py, w, h, rowBytes, comp);
    float sigma = clamp(amount * (8.0f + amount * 2.0f), 0.10f, 80.0f);
    float step = max(sigma * 0.85f, 1.0f);
    float invTwoSigma2 = 0.5f / max(sigma * sigma, 1e-4f);

    float acc = 0.0f;
    float weightSum = 0.0f;
    for (int oy = -3; oy <= 3; ++oy) {
        for (int ox = -3; ox <= 3; ++ox) {
            float sx = float(px) + float(ox) * step;
            float sy = float(py) + float(oy) * step;
            float d2 = float(ox * ox + oy * oy) * step * step;
            float wt = FAST_EXP(-d2 * invTwoSigma2);
            acc += readPxBilinear(buf, sx, sy, w, h, rowBytes, comp) * wt;
            weightSum += wt;
        }
    }

    float lowPass = acc / max(weightSum, 1e-6f);
    float transfer = smoothstep(0.01f, 0.20f, amount);
    return mix(center, lowPass, clamp(transfer, 0.0f, 1.0f));
}

// Vectorised 3-channel film blur — 7×7 Gaussian pass.
// Step = sigma * (2/3) so the ±3 ring lands at ±2σ, giving good Gaussian
// coverage without visible grid artifacts from sparse isolated sample points.
// 49 taps keeps the bilinear footprints overlapping for smooth output.
DEVFN float3 filmResolveSample3(GLOBAL_RO(float) buf, int px, int py,
                                int w, int h, int rowBytes, float amount) {
    float3 center = MAKE_FLOAT3(readPx(buf, px, py, w, h, rowBytes, 0),
                                readPx(buf, px, py, w, h, rowBytes, 1),
                                readPx(buf, px, py, w, h, rowBytes, 2));
    if (amount < 0.01f) return center;

    float sigma = clamp(amount * (8.0f + amount * 2.0f), 0.10f, 80.0f);
    // Step places the ±3 ring at ±2σ; taps are dense enough to avoid aliasing.
    float step = max(sigma * (2.0f / 3.0f), 1.0f);
    float invTwoSigma2 = 0.5f / max(sigma * sigma, 1e-4f);

    float3 acc = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;
    for (int oy = -3; oy <= 3; ++oy) {
        for (int ox = -3; ox <= 3; ++ox) {
            float sx = float(px) + float(ox) * step;
            float sy = float(py) + float(oy) * step;
            float d2 = float(ox * ox + oy * oy) * step * step;
            float wt = FAST_EXP(-d2 * invTwoSigma2);
            float3 s = readPxBilinear3(buf, sx, sy, w, h, rowBytes);
            acc.x += s.x * wt;
            acc.y += s.y * wt;
            acc.z += s.z * wt;
            weightSum += wt;
        }
    }

    float invWS = 1.0f / max(weightSum, 1e-6f);
    float3 lowPass = acc * invWS;
    float t = clamp(smoothstep(0.01f, 0.20f, amount), 0.0f, 1.0f);
    return center + (lowPass - center) * t;
}

// ============================================================
// Film blur separable prepass kernels
//
// Two-pass separable Gaussian — H then V — each with 7 taps.
// Running these as dedicated passes lets the main kernel read a single
// bilinear tap from the finished blur texture instead of running 49 taps
// per pixel inside the hot main kernel.
//
// H pass: reads from the raw input float buffer (DI-encoded, dst-coord-offset
//         corrected), writes DI values to an rgba16f output texture.
// V pass: reads the H-pass texture (output space), writes fully-blurred DI.
// Main:   samples V-pass texture once; blends with center via smoothstep(t).
// ============================================================

DEVFN void cwFilmBlurHKernel_body(
    GLOBAL_RO(float) input, WRITE_TEX hOut,
    int x, int y, CONSTANT_REF(GrainParams) p)
{
    if (x >= p.width || y >= p.height) return;

    int srcIX = x + p.dstX1 - p.srcX1;
    int srcIY = y + p.dstY1 - p.srcY1;

    float sigma = clamp(p.image_blur * (8.0f + p.image_blur * 2.0f), 0.10f, 80.0f);
    float step  = max(sigma * (2.0f / 3.0f), 1.0f);
    float invTwoSigma2 = 0.5f / max(sigma * sigma, 1e-4f);

    float3 acc = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;
    for (int ox = -3; ox <= 3; ++ox) {
        float sx = float(srcIX) + float(ox) * step;
        float sy = float(srcIY);
        float d2 = float(ox * ox) * step * step;
        float wt = FAST_EXP(-d2 * invTwoSigma2);
        float3 s = readPxBilinear3(input, sx, sy, p.srcWidth, p.srcHeight, p.srcRowBytes);
        acc.x += s.x * wt;
        acc.y += s.y * wt;
        acc.z += s.z * wt;
        weightSum += wt;
    }
    float invWS = 1.0f / max(weightSum, 1e-6f);
    TEX_WRITE(hOut, x, y, MAKE_FLOAT4(acc.x * invWS, acc.y * invWS, acc.z * invWS, 1.0f));
}

DEVFN void cwFilmBlurVKernel_body(
    READ_TEX hTex, WRITE_TEX vOut,
    int x, int y, CONSTANT_REF(GrainParams) p)
{
    if (x >= p.width || y >= p.height) return;

    float sigma = clamp(p.image_blur * (8.0f + p.image_blur * 2.0f), 0.10f, 80.0f);
    float step  = max(sigma * (2.0f / 3.0f), 1.0f);
    float invTwoSigma2 = 0.5f / max(sigma * sigma, 1e-4f);

    float3 acc = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;
    for (int oy = -3; oy <= 3; ++oy) {
        // H-pass texture is in output space; sample at (x, y + oy*step).
        float u = (float(x) + 0.5f) / float(p.width);
        float v = (float(y) + float(oy) * step + 0.5f) / float(p.height);
        float d2 = float(oy * oy) * step * step;
        float wt = FAST_EXP(-d2 * invTwoSigma2);
        float4 s = TEX_SAMPLE(hTex, u, v);
        acc.x += s.x * wt;
        acc.y += s.y * wt;
        acc.z += s.z * wt;
        weightSum += wt;
    }
    float invWS = 1.0f / max(weightSum, 1e-6f);
    TEX_WRITE(vOut, x, y, MAKE_FLOAT4(acc.x * invWS, acc.y * invWS, acc.z * invWS, 1.0f));
}


//
// The grain signal is built from independent jittered impulse fields rather than
// smooth value noise or area coverage.  Each layer has stable crystal positions
// and frame-varying developed density, then the combined zero-mean signal is
// shaped by channel exposure so shadows/highlights stay quieter than midtones.
// ============================================================

// 35mm reference effective grain size at 4K authoring height.
#define GRAIN_REF_SIZE  2.3f

DEVFN float grain_rand01(int ix, int iy, uint seed, uint salt) {
    return float(cw_cell_seed(ix, iy, seed ^ salt)) * 2.3283064365e-10f;
}

DEVFN float crystalLayer(float sx, float sy, float cellSize, float spread,
                         float irr, uint static_seed, uint frame_seed) {
    float invCell = 1.0f / max(cellSize, 0.1f);
    float gx = sx * invCell;
    float gy = sy * invCell;
    int baseX = int(floor(gx));
    int baseY = int(floor(gy));

    float acc = 0.0f;
    float norm = 0.0f;
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            int cx = baseX + ox;
            int cy = baseY + oy;

            float jx = grain_rand01(cx, cy, static_seed, 0x8DA6B343u) - 0.5f;
            float jy = grain_rand01(cx, cy, static_seed, 0xD8163841u) - 0.5f;
            float centerX = (float(cx) + 0.5f + jx * 0.9f) * cellSize;
            float centerY = (float(cy) + 0.5f + jy * 0.9f) * cellSize;

            float localSpread = cellSize * spread *
                                (0.72f + 0.56f * grain_rand01(cx, cy, static_seed, 0xA4093822u));
            if (irr > 0.01f) {
                float anatomy = cw_vnoise(float(cx) * 0.43f, float(cy) * 0.43f,
                                          static_seed ^ 0x7FEB352Du);
                localSpread *= 1.0f + anatomy * irr * 0.45f;
            }
            localSpread = max(localSpread, 0.12f);

            float dx = sx - centerX;
            float dy = sy - centerY;
            float d2 = dx * dx + dy * dy;
            // Polynomial (Epanechnikov-squared) footprint replaces exp().
            // Bounded support eliminates all exp() from the inner loop;
            // the inv2 scale is widened by 2× relative to the Gaussian so
            // the effective radius (zero-crossing) matches the old ~2.5σ support.
            float inv2 = 0.25f / max(localSpread * localSpread, 1e-4f);
            float t = max(0.0f, 1.0f - d2 * inv2);
            float footprint = t * t;

            float density = grain_rand01(cx, cy, frame_seed, 0xB5297A4Du) * 2.0f - 1.0f;
            float crunch = grain_rand01(cx, cy, frame_seed, 0x68E31DA4u) * 2.0f - 1.0f;
            density = density * 0.72f + crunch * abs(crunch) * 0.28f;

            acc += density * footprint;
            norm += footprint * footprint;
        }
    }

    return acc / max(FAST_SQRT(norm), 0.55f);
}

DEVFN float filmGrainChannel(int px, int py,
                             float cellSize, float softness,
                             float contrast, float lum, float irr,
                             uint static_seed, uint frame_seed,
                             float highlight_suppress) {
    float sx = float(px);
    float sy = float(py);
    // Two layers only (weights renormalised from 0.56/0.32 after dropping the
    // 0.12-weight coarse layer; overall 0.62 output scale is preserved).
    float n = 0.636f * crystalLayer(sx + 0.17f, sy - 0.11f,
                                    cellSize * 0.72f, 0.28f + softness * 0.12f,
                                    irr, static_seed, frame_seed);
    n += 0.364f * crystalLayer(sx - 0.23f, sy + 0.19f,
                               cellSize * 1.18f, 0.36f + softness * 0.15f,
                               irr, static_seed + 1013u, frame_seed + 2027u);
    n *= 0.62f;

    if (contrast > 1e-4f) {
        float signN = (n < 0.0f) ? -1.0f : 1.0f;
        float mag = min(abs(n), 1.0f);
        float shaped = FAST_POW(mag, 1.0f / (1.0f + 1.7f * contrast));
        n = signN * mix(mag, shaped, contrast * 0.85f);
    }

    lum = clamp(lum, 0.0f, 1.0f);
    float exposureAmp = 2.0f * FAST_SQRT(max(lum * (1.0f - lum), 0.0f));
    n *= exposureAmp * FAST_POW(max(1.0f - lum, 0.0f), highlight_suppress);
    return n;
}

// ============================================================
// Vectorised 3-channel grain — optimised RNG.
//
// Per cell, we compute ONE cw_cell_seed for spatial structure (jitter,
// spread, anatomy) and ONE cw_cell_seed per channel for density.
// Two values are extracted from each 32-bit hash via bit-splitting
// (low/high 16 bits), replacing 10+ separate grain_rand01 calls and
// the cw_vnoise call.  The result is ~5 hash ops/cell vs ~24 before.
// ============================================================

DEVFN float3 crystalLayer3(float sx, float sy, float cellSize, float spread,
                            float irr, uint static_seed,
                            uint frame_r, uint frame_g, uint frame_b,
                            uint env_seed) {
    float invCell = 1.0f / max(cellSize, 0.1f);
    int baseX = int(floor(sx * invCell));
    int baseY = int(floor(sy * invCell));

    // Slowly-varying development-density envelope — low spatial frequency
    // (period ~11 cells), updates every 8 frames.  Keeps the large-scale
    // grain energy map stable across frames so the eye sees natural boiling
    // rather than global flicker, while individual crystal densities remain
    // fully independent each frame.
    float env     = cw_vnoise(float(baseX) * 0.09f, float(baseY) * 0.09f, env_seed);
    float env_mod = 1.0f + (env * 2.0f - 1.0f) * 0.25f;   // [0.75 .. 1.25]

    float3 acc  = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
    float  norm = 0.0f;

    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            // Diagonal cells (|ox|=1 AND |oy|=1) are √2× farther than cardinal
            // neighbors.  For our spread range (0.28–0.51 × cellSize) their reach
            // barely overlaps a pixel at a cell corner, and they are never the
            // sole contributor.  Skipping them eliminates 4/9 spatial-hash pairs
            // unconditionally with no perceptible quality loss.
            if (ox != 0 && oy != 0) continue;

            int cx = baseX + ox;
            int cy = baseY + oy;
            // Chain a second hash for uncorrelated secondary values.
            uint hs  = cw_cell_seed(cx, cy, static_seed);
            uint hs2 = cw_hash(hs ^ 0x5C3A47B1u);

            float jx = float(hs  & 0xFFFFu) * (1.0f / 65535.0f) - 0.5f;
            float jy = float(hs  >> 16u)    * (1.0f / 65535.0f) - 0.5f;
            float centerX = (float(cx) + 0.5f + jx * 0.9f) * cellSize;
            float centerY = (float(cy) + 0.5f + jy * 0.9f) * cellSize;

            float spread_rng = float(hs2 & 0xFFFFu) * (1.0f / 65535.0f);
            float localSpread = cellSize * spread * (0.72f + 0.56f * spread_rng);
            if (irr > 0.01f) {
                // Anatomy variation: use high bits of second hash instead of vnoise.
                float anatomy = float(hs2 >> 16u) * (1.0f / 65535.0f) - 0.5f;
                localSpread *= 1.0f + anatomy * irr * 0.45f;
            }
            localSpread = max(localSpread, 0.12f);

            float dx = sx - centerX;
            float dy = sy - centerY;
            float d2 = dx * dx + dy * dy;
            float inv2 = 0.25f / max(localSpread * localSpread, 1e-4f);
            float t = max(0.0f, 1.0f - d2 * inv2);

            // Skip the three per-channel hashes when this cell has no contribution.
            if (t <= 0.0f) continue;

            float footprint = t * t;

            // Per-crystal densities — fully independent each frame.
            // env_mod anchors the slowly-varying regional density so large-scale
            // grain energy is stable across frames; fine texture still boils freely.
            uint hr = cw_cell_seed(cx, cy, frame_r);
            uint hg = cw_cell_seed(cx, cy, frame_g);
            uint hb = cw_cell_seed(cx, cy, frame_b);

            float dr  = (float(hr & 0xFFFFu) * (2.0f / 65535.0f) - 1.0f) * env_mod;
            float cr  = (float(hr >> 16u)    * (2.0f / 65535.0f) - 1.0f) * env_mod;
            float dg  = (float(hg & 0xFFFFu) * (2.0f / 65535.0f) - 1.0f) * env_mod;
            float cg2 = (float(hg >> 16u)    * (2.0f / 65535.0f) - 1.0f) * env_mod;
            float db  = (float(hb & 0xFFFFu) * (2.0f / 65535.0f) - 1.0f) * env_mod;
            float cb  = (float(hb >> 16u)    * (2.0f / 65535.0f) - 1.0f) * env_mod;

            acc.x += (dr * 0.72f + cr  * abs(cr)  * 0.28f) * footprint;
            acc.y += (dg * 0.72f + cg2 * abs(cg2) * 0.28f) * footprint;
            acc.z += (db * 0.72f + cb  * abs(cb)  * 0.28f) * footprint;
            norm  += footprint * footprint;
        }
    }

    return acc * FAST_RSQRT(max(norm, 0.3025f));
}

DEVFN float3 filmGrain3(int px, int py,
                         float cellSize, float softness,
                         float contrast, float3 lum,
                         float irr, uint static_seed,
                         uint frame_r, uint frame_g, uint frame_b,
                         uint env_seed,
                         float highlight_suppress) {
    float sx = float(px);
    float sy = float(py);

    float3 n = crystalLayer3(sx + 0.17f, sy - 0.11f,
                              cellSize * 0.72f, 0.28f + softness * 0.12f,
                              irr, static_seed,
                              frame_r, frame_g, frame_b,
                              env_seed) * 0.636f;
    n = n + crystalLayer3(sx - 0.23f, sy + 0.19f,
                           cellSize * 1.18f, 0.36f + softness * 0.15f,
                           irr, static_seed + 1013u,
                           frame_r + 2027u, frame_g + 2027u, frame_b + 2027u,
                           env_seed + 3571u) * 0.364f;
    n = n * 0.62f;

    if (contrast > 1e-4f) {
        float3 absN  = abs(n);
        float3 signN = MAKE_FLOAT3(n.x < 0.0f ? -1.0f : 1.0f,
                                   n.y < 0.0f ? -1.0f : 1.0f,
                                   n.z < 0.0f ? -1.0f : 1.0f);
        float3 mag   = min(absN, MAKE_FLOAT3(1.0f, 1.0f, 1.0f));
        float  cpow  = 1.0f / (1.0f + 1.7f * contrast);
        float3 shaped = MAKE_FLOAT3(FAST_POW(mag.x, cpow),
                                    FAST_POW(mag.y, cpow),
                                    FAST_POW(mag.z, cpow));
        n = signN * (mag + (shaped - mag) * (contrast * 0.85f));
    }

    float3 clampLum = clamp(lum, 0.0f, 1.0f);
    float3 invLum   = MAKE_FLOAT3(1.0f - clampLum.x,
                                  1.0f - clampLum.y,
                                  1.0f - clampLum.z);
    float3 expAmp = MAKE_FLOAT3(
        2.0f * FAST_SQRT(max(clampLum.x * invLum.x, 0.0f)),
        2.0f * FAST_SQRT(max(clampLum.y * invLum.y, 0.0f)),
        2.0f * FAST_SQRT(max(clampLum.z * invLum.z, 0.0f)));
    float3 hlSup = MAKE_FLOAT3(
        FAST_POW(max(invLum.x, 0.0f), highlight_suppress),
        FAST_POW(max(invLum.y, 0.0f), highlight_suppress),
        FAST_POW(max(invLum.z, 0.0f), highlight_suppress));

    return n * expAmp * hlSup;
}

// ============================================================
// HSL colour-space helpers
// ============================================================
DEVFN float3 rgb_to_hsl(float3 c) {
    float maxC  = max(max(c.x, c.y), c.z);
    float minC  = min(min(c.x, c.y), c.z);
    float delta = maxC - minC;
    float l = (maxC + minC) * 0.5f;
    float s = (delta < 1e-6f) ? 0.0f :
              (l < 0.5f ? delta / (maxC + minC) : delta / (2.0f - maxC - minC));
    float h = 0.0f;
    if (delta > 1e-6f) {
        if (maxC == c.x)       h = (c.y - c.z) / delta + (c.y < c.z ? 6.0f : 0.0f);
        else if (maxC == c.y)  h = (c.z - c.x) / delta + 2.0f;
        else                   h = (c.x - c.y) / delta + 4.0f;
        h /= 6.0f;
    }
    return MAKE_FLOAT3(h, s, l);
}

DEVFN float hsl_hue_to_rgb(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f/6.0f)  return p + (q - p) * 6.0f * t;
    if (t < 0.5f)        return q;
    if (t < 2.0f/3.0f)  return p + (q - p) * (2.0f/3.0f - t) * 6.0f;
    return p;
}

DEVFN float3 hsl_to_rgb(float3 hsl) {
    float h = hsl.x, s = hsl.y, l = hsl.z;
    if (s < 1e-6f) return MAKE_FLOAT3(l, l, l);
    float q = (l < 0.5f) ? l * (1.0f + s) : l + s - l * s;
    float p = 2.0f * l - q;
    return MAKE_FLOAT3(hsl_hue_to_rgb(p, q, h + 1.0f/3.0f),
                  hsl_hue_to_rgb(p, q, h),
                  hsl_hue_to_rgb(p, q, h - 1.0f/3.0f));
}

// ============================================================
// 3D LUT tetrahedral interpolation
// lutData: flat array of size^3 * 3 floats (RGB), B-major ordering
// Tetrahedral interpolation is more accurate than trilinear for 3D LUTs
// because it avoids the "cube diagonal" artifact.
// ============================================================
DEVFN float3 applyLut3D(GLOBAL_RO(float) lutData, int lutSize,
                         float3 color) {
    float3 scaled = clamp(color, 0.0f, 1.0f) * float(lutSize - 1);
    int3 lo = MAKE_INT3(int(floor((scaled).x)), int(floor((scaled).y)), int(floor((scaled).z)));
    int3 hi = min(lo + 1, lutSize - 1);
    float3 f = scaled - MAKE_FLOAT3(float((lo).x), float((lo).y), float((lo).z));  // fractional position within cube cell

    int s = lutSize;
    int s2 = s * s;
    #define LUT_IDX(ri, gi, bi) ((bi) * s2 + (gi) * s + (ri)) * 3

    // Fetch the 4 vertices needed for the tetrahedron
    // (we always need c000 and c111, plus 2 others depending on which
    // tetrahedron the fractional position falls into)
    int idx000 = LUT_IDX(lo.x, lo.y, lo.z);
    int idx111 = LUT_IDX(hi.x, hi.y, hi.z);
    float3 c000 = MAKE_FLOAT3(lutData[idx000], lutData[idx000+1], lutData[idx000+2]);
    float3 c111 = MAKE_FLOAT3(lutData[idx111], lutData[idx111+1], lutData[idx111+2]);

    float3 lutResult;

    if (f.x > f.y) {
        if (f.y > f.z) {
            // T1: r > g > b
            int idx100 = LUT_IDX(hi.x, lo.y, lo.z);
            int idx110 = LUT_IDX(hi.x, hi.y, lo.z);
            float3 c100 = MAKE_FLOAT3(lutData[idx100], lutData[idx100+1], lutData[idx100+2]);
            float3 c110 = MAKE_FLOAT3(lutData[idx110], lutData[idx110+1], lutData[idx110+2]);
            lutResult = (1.0f - f.x) * c000 + (f.x - f.y) * c100 +
                        (f.y - f.z) * c110 + f.z * c111;
        } else if (f.x > f.z) {
            // T2: r > b > g
            int idx100 = LUT_IDX(hi.x, lo.y, lo.z);
            int idx101 = LUT_IDX(hi.x, lo.y, hi.z);
            float3 c100 = MAKE_FLOAT3(lutData[idx100], lutData[idx100+1], lutData[idx100+2]);
            float3 c101 = MAKE_FLOAT3(lutData[idx101], lutData[idx101+1], lutData[idx101+2]);
            lutResult = (1.0f - f.x) * c000 + (f.x - f.z) * c100 +
                        (f.z - f.y) * c101 + f.y * c111;
        } else {
            // T3: b > r > g
            int idx001 = LUT_IDX(lo.x, lo.y, hi.z);
            int idx101 = LUT_IDX(hi.x, lo.y, hi.z);
            float3 c001 = MAKE_FLOAT3(lutData[idx001], lutData[idx001+1], lutData[idx001+2]);
            float3 c101 = MAKE_FLOAT3(lutData[idx101], lutData[idx101+1], lutData[idx101+2]);
            lutResult = (1.0f - f.z) * c000 + (f.z - f.x) * c001 +
                        (f.x - f.y) * c101 + f.y * c111;
        }
    } else {
        if (f.z > f.y) {
            // T4: b > g > r
            int idx001 = LUT_IDX(lo.x, lo.y, hi.z);
            int idx011 = LUT_IDX(lo.x, hi.y, hi.z);
            float3 c001 = MAKE_FLOAT3(lutData[idx001], lutData[idx001+1], lutData[idx001+2]);
            float3 c011 = MAKE_FLOAT3(lutData[idx011], lutData[idx011+1], lutData[idx011+2]);
            lutResult = (1.0f - f.z) * c000 + (f.z - f.y) * c001 +
                        (f.y - f.x) * c011 + f.x * c111;
        } else if (f.z > f.x) {
            // T5: g > b > r
            int idx010 = LUT_IDX(lo.x, hi.y, lo.z);
            int idx011 = LUT_IDX(lo.x, hi.y, hi.z);
            float3 c010 = MAKE_FLOAT3(lutData[idx010], lutData[idx010+1], lutData[idx010+2]);
            float3 c011 = MAKE_FLOAT3(lutData[idx011], lutData[idx011+1], lutData[idx011+2]);
            lutResult = (1.0f - f.y) * c000 + (f.y - f.z) * c010 +
                        (f.z - f.x) * c011 + f.x * c111;
        } else {
            // T6: g > r > b
            int idx010 = LUT_IDX(lo.x, hi.y, lo.z);
            int idx110 = LUT_IDX(hi.x, hi.y, lo.z);
            float3 c010 = MAKE_FLOAT3(lutData[idx010], lutData[idx010+1], lutData[idx010+2]);
            float3 c110 = MAKE_FLOAT3(lutData[idx110], lutData[idx110+1], lutData[idx110+2]);
            lutResult = (1.0f - f.y) * c000 + (f.y - f.x) * c010 +
                        (f.x - f.z) * c110 + f.z * c111;
        }
    }

    #undef LUT_IDX

    return lutResult;
}

// ============================================================
// Output colour-space transfer functions
// ============================================================

// DaVinci Intermediate → scene linear (inverse OETF)
// Matches colour-science oetf_inverse_DaVinciIntermediate exactly.
DEVFN float di_to_linear(float V) {
    const float A       = 0.0075f;
    const float B       = 7.0f;
    const float C       = 0.07329248f;
    const float M       = 10.44426855f;
    const float LOG_CUT = 0.02740668f;
    if (V > LOG_CUT)
        return EXP2(V / C - B) - A;
    else
        return V / M;
}

// Scene linear → DaVinci Intermediate (forward OETF)
// Matches colour-science oetf_DaVinciIntermediate exactly.
DEVFN float linear_to_di(float L) {
    const float A       = 0.0075f;
    const float B       = 7.0f;
    const float C       = 0.07329248f;
    const float M       = 10.44426855f;
    const float LIN_CUT = 0.00262409f;
    if (L > LIN_CUT)
        return C * (LOG2(L + A) + B);
    else
        return L * M;
}

// Printer points act as per-channel exposure changes ahead of the print stock.
// In the DWG/DI working path that means: decode DI -> apply exact stop gain ->
// encode back to DI. This preserves the lab-style 1/12-stop printer-point model
// and leaves the downstream output transform to shape the tone scale.
//
// To match the *subtractive* feel of real photochemical printer-point timing,
// the trim is weighted by a soft toe-rolloff so it has zero effect at deep
// black (the print's toe / Dmax holds shadows regardless of printer light) and
// reaches full strength by mid-gray. Without this, a per-channel linear gain
// uniformly lifts shadows into the trim colour, washing the whole frame
// instead of timing the midtones and highlights the way a print does.
//
// Toe band is in DI: ~0.10 (near max black) to ~0.34 (mid-gray, DI of 0.18).
DEVFN float apply_printer_lights(float diValue, float exposureGain) {
    float gained = linear_to_di(di_to_linear(diValue) * exposureGain);
    float t = saturate((diValue - 0.10f) / (0.34f - 0.10f));
    float w = t * t * (3.0f - 2.0f * t);  // smoothstep
    return diValue + (gained - diValue) * w;
}

// ============================================================
// Film Halation — exponential falloff model
//
//   Based on https://github.com/hotgluebanjo/halation-dctl
//
//   Two blend methods:
//     Proosa — additive energy-preserving method:
//       color = (1.0, COLOR, 0.0)
//       halated = rgb + blurred * color
//       result  = halated / (color + 1.0)
//     Minuth — frequency-separation method:
//       color = (0.7, 1.0 - COLOR/3.0, 1.0)
//       halated = (rgb - blurred) * color
//       result  = halated + blurred
//
//   Both methods operate on scene-linear values.
//   Input must be scene-linear (use prepass texture).
// ============================================================

// Sample the prepass (scene-linear) texture at integer offsets.
// Off-frame samples return 0 (no light from outside the gate).
DEVFN float3 hal_sample_prepass(READ_TEX linTex,
                                 float fx, float fy, int w, int h) {
    if (fx < 0.0f || fx > float(w - 1) || fy < 0.0f || fy > float(h - 1))
        return MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
    float2 uv = MAKE_FLOAT2((fx + 0.5f) / float(w), (fy + 0.5f) / float(h));
    return TEX_SAMPLE(linTex, (uv).x, (uv).y).xyz;
}

// Fallback 2D exponential blur — used only when the separable prepass
// is unavailable (e.g. CPU path). O(radius²) per pixel.
DEVFN float3 hal_exp_blur(READ_TEX linTex, int px, int py, int w, int h, float r0) {
    if (r0 < 1e-10f) return hal_sample_prepass(linTex, float(px), float(py), w, h);
    int radius = int(r0 * 4.5f + 0.9999f);
    float3 sum = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            float r = FAST_SQRT(float(i * i + j * j));
            float weight = FAST_EXP(-r / r0);
            float3 s = hal_sample_prepass(linTex, float(px + i), float(py + j), w, h);
            sum.x += weight * s.x;
            sum.y += weight * s.y;
            sum.z += weight * s.z;
            weight_sum += weight;
        }
    }
    float inv = weight_sum > 0.0f ? 1.0f / weight_sum : 0.0f;
    return MAKE_FLOAT3(sum.x * inv, sum.y * inv, sum.z * inv);
}

// ============================================================
// Separable halation blur — H pass then V pass.
//
// The 2D kernel exp(-sqrt(dx²+dy²)/r0) is not separable, but
// exp(-|dx|/r0) * exp(-|dy|/r0) = exp(-(|dx|+|dy|)/r0) is.
// This is the L1-norm (diamond) Laplacian exponential kernel —
// visually indistinguishable from the radially-symmetric version
// for a diffuse halation glow. Each pass is O(radius) per pixel
// instead of O(radius²), giving ~7–15× fewer texture reads at
// typical halSize values.
// ============================================================
DEVFN void cwHalationBlurHKernel_body(
    int x, int y,
    READ_TEX linIn,
    WRITE_TEX halBlurHOut,
    CONSTANT_REF(GrainParams) p)
{
    if (x >= p.width || y >= p.height) return;
    float r0 = p.halSize;
    if (r0 < 1e-10f) {
        float3 s = hal_sample_prepass(linIn, float(x), float(y), p.width, p.height);
        TEX_WRITE(halBlurHOut, x, y, MAKE_FLOAT4(s.x, s.y, s.z, 1.0f));
        return;
    }
    int radius = int(r0 * 4.5f + 0.9999f);
    float3 sum = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    for (int dx = -radius; dx <= radius; dx++) {
        float weight = FAST_EXP(-float(dx < 0 ? -dx : dx) / r0);
        float3 s = hal_sample_prepass(linIn, float(x + dx), float(y), p.width, p.height);
        sum.x += weight * s.x;
        sum.y += weight * s.y;
        sum.z += weight * s.z;
        weight_sum += weight;
    }
    float inv = weight_sum > 0.0f ? 1.0f / weight_sum : 0.0f;
    TEX_WRITE(halBlurHOut, x, y, MAKE_FLOAT4(sum.x * inv, sum.y * inv, sum.z * inv, 1.0f));
}

DEVFN void cwHalationBlurVKernel_body(
    int x, int y,
    READ_TEX halBlurHIn,
    WRITE_TEX halBlurOut,
    CONSTANT_REF(GrainParams) p)
{
    if (x >= p.width || y >= p.height) return;
    float r0 = p.halSize;
    if (r0 < 1e-10f) {
        float u = (float(x) + 0.5f) / float(p.width);
        float v = (float(y) + 0.5f) / float(p.height);
        float4 s = TEX_SAMPLE(halBlurHIn, u, v);
        TEX_WRITE(halBlurOut, x, y, MAKE_FLOAT4(s.x, s.y, s.z, 1.0f));
        return;
    }
    int radius = int(r0 * 4.5f + 0.9999f);
    float w = float(p.width);
    float h = float(p.height);
    float u = (float(x) + 0.5f) / w;
    float3 sum = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    for (int dy = -radius; dy <= radius; dy++) {
        float weight = FAST_EXP(-float(dy < 0 ? -dy : dy) / r0);
        float v = (float(y + dy) + 0.5f) / h;
        float4 s = TEX_SAMPLE(halBlurHIn, u, v);
        sum.x += weight * s.x;
        sum.y += weight * s.y;
        sum.z += weight * s.z;
        weight_sum += weight;
    }
    float inv = weight_sum > 0.0f ? 1.0f / weight_sum : 0.0f;
    TEX_WRITE(halBlurOut, x, y, MAKE_FLOAT4(sum.x * inv, sum.y * inv, sum.z * inv, 1.0f));
}

// Nearest-integer rounding helper (used by diffusion below)
DEVFN int hal_round(float x) {
    return int(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

// ============================================================
// Diffusion scatter texture smoothing — separable Gaussian blur
//
// The scatter prepass runs at 1/2–1/4 resolution. Each scatter
// pixel independently integrates 192 Monte Carlo PSF samples,
// producing noisy per-block values near sharp edges. This
// separable Gaussian pass (H then V, sigma=2 scatter pixels,
// radius=6 taps each side) smooths those block boundaries before
// the scatter texture is upsampled into the main kernel.
// ============================================================

#define DIFFUSION_BLUR_SIGMA   2.0f
#define DIFFUSION_BLUR_RADIUS  6

DEVFN void cwDiffusionBlurHKernel_body(
    int x, int y,
    READ_TEX  scatterIn,
    WRITE_TEX scatterOut,
    CONSTANT_REF(GrainParams) p)
{
    int sw = p.diffusionScatterWidth;
    int sh = p.diffusionScatterHeight;
    if (x >= sw || y >= sh) return;
    float invSigmaSq = 1.0f / (DIFFUSION_BLUR_SIGMA * DIFFUSION_BLUR_SIGMA);
    float4 sum = MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    for (int dx = -DIFFUSION_BLUR_RADIUS; dx <= DIFFUSION_BLUR_RADIUS; dx++) {
        float weight = FAST_EXP(-0.5f * float(dx * dx) * invSigmaSq);
        int sx = x + dx;
        if (sx < 0)   sx = 0;
        if (sx >= sw) sx = sw - 1;
        float u = (float(sx) + 0.5f) / float(sw);
        float v = (float(y)  + 0.5f) / float(sh);
        float4 s = TEX_SAMPLE(scatterIn, u, v);
        sum.x += weight * s.x;
        sum.y += weight * s.y;
        sum.z += weight * s.z;
        sum.w += weight * s.w;
        weight_sum += weight;
    }
    float inv = weight_sum > 0.0f ? 1.0f / weight_sum : 0.0f;
    TEX_WRITE(scatterOut, x, y,
              MAKE_FLOAT4(sum.x * inv, sum.y * inv, sum.z * inv, sum.w * inv));
}

DEVFN void cwDiffusionBlurVKernel_body(
    int x, int y,
    READ_TEX  scatterIn,
    WRITE_TEX scatterOut,
    CONSTANT_REF(GrainParams) p)
{
    int sw = p.diffusionScatterWidth;
    int sh = p.diffusionScatterHeight;
    if (x >= sw || y >= sh) return;
    float invSigmaSq = 1.0f / (DIFFUSION_BLUR_SIGMA * DIFFUSION_BLUR_SIGMA);
    float4 sum = MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    for (int dy = -DIFFUSION_BLUR_RADIUS; dy <= DIFFUSION_BLUR_RADIUS; dy++) {
        float weight = FAST_EXP(-0.5f * float(dy * dy) * invSigmaSq);
        int sy = y + dy;
        if (sy < 0)   sy = 0;
        if (sy >= sh) sy = sh - 1;
        float u = (float(x)  + 0.5f) / float(sw);
        float v = (float(sy) + 0.5f) / float(sh);
        float4 s = TEX_SAMPLE(scatterIn, u, v);
        sum.x += weight * s.x;
        sum.y += weight * s.y;
        sum.z += weight * s.z;
        sum.w += weight * s.w;
        weight_sum += weight;
    }
    float inv = weight_sum > 0.0f ? 1.0f / weight_sum : 0.0f;
    TEX_WRITE(scatterOut, x, y,
              MAKE_FLOAT4(sum.x * inv, sum.y * inv, sum.z * inv, sum.w * inv));
}



// ============================================================
// Optical Diffusion — profiled image scatter
//
//   Glass diffusion filters work by forward-scattering a fraction of incoming
//   light off small particles in the glass. The visible result is not just
//   highlight bloom: a sharp image remains, but a soft low-contrast
//   image layer leaks back over it, with extra energy around bright
//   sources.
//
//   Correct model:  out = mix(image, PSF * image, strength)
//                         + highlight_scatter
//
//   PSF is a three-lobe diffusion profile:
//     - Core: small-angle softening near edges and highlight rims
//     - Body: visible pearlescent glow around bright/midtone structure
//     - Falloff: broad low-level veiling halo
//
//   Highlight extraction is a soft-knee scalar bright pass in scene-linear
//   light. The scalar gate preserves source colour while letting intense
//   practicals add more scatter than ordinary midtones.
//
//   veil term: BPM-character shadow lift, proportional to local
//   scene brightness (approximates light scattered everywhere
//   off carbon-loaded particles).
//
//   References:
//     Spencer et al. 1995, "Physically-based glare effects"
//     Ritschel et al. 2009, "Temporal Glare"
// ============================================================
DEVFN float3 diffusion_highlight_linear(float3 lin, float thresholdLin) {
    float3 c = max(lin, MAKE_FLOAT3(0.0f, 0.0f, 0.0f));
    float brightness = max(max(c.x, c.y), c.z);
    if (thresholdLin <= 1e-6f) return c;

    float knee = max(thresholdLin * 0.5f, 1e-6f);
    float softness = clamp(brightness - thresholdLin + knee, 0.0f, 2.0f * knee);
    softness = (softness * softness) / (4.0f * knee + 1e-6f);
    float contribution = max(brightness - thresholdLin, softness);
    float multiplier = contribution / max(brightness, 1e-6f);
    return c * clamp(multiplier, 0.0f, 1.0f);
}

struct DiffusionScatterResult {
    float3 blur;
    float3 glow;
    float veil;
};

DEVFN DiffusionScatterResult diffusion_scatter_center_table(READ_TEX linTex,
                                     float centerX, float centerY,
                                     uint seedX, uint seedY,
                                     int w, int h, GLOBAL_RO(float) sampleTable) {
    DiffusionScatterResult out;
    out.blur = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
    out.glow = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
    out.veil = 0.0f;

    // Header: [0]=count [1]=cW_r [2]=cW_g [3]=cW_b [4]=thresholdLin [5]=pad
    int sampleCount  = int(sampleTable[0]);
    float cW_r       = sampleTable[1];
    float cW_g       = sampleTable[2];
    float cW_b       = sampleTable[3];
    float thresholdLin = sampleTable[4];

    float3 center = hal_sample_prepass(linTex, centerX, centerY, w, h);
    out.blur += MAKE_FLOAT3(center.x * cW_r, center.y * cW_g, center.z * cW_b);
    float cW_avg = (cW_r + cW_g + cW_b) * 0.333333f;
    out.glow += diffusion_highlight_linear(center, thresholdLin) * cW_avg;
    out.veil += max(max(center.x, center.y), center.z) * cW_avg;

    const int maxSamples = 192;
    const float twoPi = 6.28318530718f;
    const float cosGolden = -0.7373688781f;
    const float sinGolden =  0.6754902943f;
    uint hseed = cw_hash(seedX ^ (seedY * 0x9E3779B9u));
    // Smooth spatial gradient rotation instead of per-pixel random:
    // adjacent scatter pixels differ by ~8-9° so bilinear upsampling
    // stays consistent — eliminates granularity at edges with zero cost.
    float rotate = fmod(float(int(seedX)) * 0.15f + float(int(seedY)) * 0.23f, twoPi);
    (void)hseed;  // kept for potential future use
    float dirX = cos(rotate);
    float dirY = sin(rotate);

    // Samples: [6 + s*4 + 0]=r  [+1]=w_r  [+2]=w_g  [+3]=w_b
    for (int sample = 0; sample < maxSamples; ++sample) {
        if (sample >= sampleCount) break;
        int ti = 6 + sample * 4;
        float sampleRadius = sampleTable[ti];
        float w_r = sampleTable[ti + 1];
        float w_g = sampleTable[ti + 2];
        float w_b = sampleTable[ti + 3];
        float ox = dirX * sampleRadius;
        float oy = dirY * sampleRadius;
        float3 sc = hal_sample_prepass(linTex, centerX + ox, centerY + oy, w, h);
        out.blur += MAKE_FLOAT3(sc.x * w_r, sc.y * w_g, sc.z * w_b);
        float w_avg = (w_r + w_g + w_b) * 0.333333f;
        out.glow += diffusion_highlight_linear(sc, thresholdLin) * w_avg;
        out.veil += max(max(sc.x, sc.y), sc.z) * w_avg;

        float nextDirX = dirX * cosGolden - dirY * sinGolden;
        dirY = dirX * sinGolden + dirY * cosGolden;
        dirX = nextDirX;
    }

    return out;
}

DEVFN DiffusionScatterResult diffusion_scatter_precomputed(READ_TEX blurTex,
                                     READ_TEX glowTex, float fx, float fy,
                                     int w, int h) {
    DiffusionScatterResult out;
    float2 uv = MAKE_FLOAT2((fx + 0.5f) / float(w), (fy + 0.5f) / float(h));
    float4 blurVeil = TEX_SAMPLE(blurTex, uv.x, uv.y);
    float4 glow     = TEX_SAMPLE(glowTex, uv.x, uv.y);
    out.blur = max(MAKE_FLOAT3(blurVeil.x, blurVeil.y, blurVeil.z), MAKE_FLOAT3(0.0f, 0.0f, 0.0f));
    out.veil = max(blurVeil.w, 0.0f);
    out.glow = max(MAKE_FLOAT3(glow.x, glow.y, glow.z), MAKE_FLOAT3(0.0f, 0.0f, 0.0f));
    return out;
}

DEVFN void cwDiffusionScatterKernel_body(
    int x, int y, READ_TEX linInput, WRITE_TEX blurOut, WRITE_TEX glowOut,
    CONSTANT_REF(GrainParams) p, GLOBAL_RO(float) sampleTable)
{
    if (x >= p.diffusionScatterWidth || y >= p.diffusionScatterHeight) return;

    float scatterW = max(float(p.diffusionScatterWidth), 1.0f);
    float scatterH = max(float(p.diffusionScatterHeight), 1.0f);
    float centerX = (float(x) + 0.5f) * (float(p.width) / scatterW) - 0.5f;
    float centerY = (float(y) + 0.5f) * (float(p.height) / scatterH) - 0.5f;

    DiffusionScatterResult scatter = diffusion_scatter_center_table(
        linInput, centerX, centerY, uint(x), uint(y), p.width, p.height, sampleTable);

    TEX_WRITE(blurOut, x, y, MAKE_FLOAT4(scatter.blur.x, scatter.blur.y, scatter.blur.z, scatter.veil));
    TEX_WRITE(glowOut, x, y, MAKE_FLOAT4(scatter.glow.x, scatter.glow.y, scatter.glow.z, 0.0f));
}

DEVFN float3 diffusion_picker_tint(float r, float g, float b) {
    float3 c = max(MAKE_FLOAT3(r, g, b), MAKE_FLOAT3(0.0f, 0.0f, 0.0f));
    float peak = max(max(c.x, c.y), c.z);
    if (peak <= 1e-6f) return MAKE_FLOAT3(1.0f, 1.0f, 1.0f);
    float luma = max(0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z, 1e-6f);
    c = c / luma;
    return MAKE_FLOAT3(min(c.x, 4.0f), min(c.y, 4.0f), min(c.z, 4.0f));
}

// ============================================================
//  Split Toning — DI-luma masked opponent chroma bias
//
//  The old implementation added large RGB offsets directly to DI log code
//  values, then tried to repair luminance with a ratio. This version follows
//  modern color-balance practice: select shadows/highlights from scene-linear
//  DWG luminance encoded back to DI, then push bounded Oklab opponent chroma
//  while keeping perceptual lightness stable.
// ============================================================

DEVFN float split_dwg_y_linear(float3 lin) {
    return 0.2741335367f * lin.x + 0.8735913831f * lin.y - 0.1477249199f * lin.z;
}

DEVFN float split_luma(float3 di) {
    float3 lin = MAKE_FLOAT3(di_to_linear(di.x), di_to_linear(di.y), di_to_linear(di.z));
    return linear_to_di(max(split_dwg_y_linear(lin), 0.0f));
}

DEVFN float splitCurve(float t) {
    float s = clamp(t, 0.0f, 1.0f);
    return s * s * s * (s * (s * 6.0f - 15.0f) + 10.0f);
}

DEVFN float split_signed_cuberoot(float v) {
    return (v < 0.0f) ? -pow(-v, 0.3333333333f) : pow(v, 0.3333333333f);
}

DEVFN float3 split_dwg_to_xyz(float3 c) {
    return MAKE_FLOAT3(
        0.7006607968f * c.x + 0.1487679160f * c.y + 0.1010412872f * c.z,
        0.2741335367f * c.x + 0.8735913831f * c.y - 0.1477249199f * c.z,
       -0.0989683375f * c.x - 0.1378889305f * c.y + 1.3256872680f * c.z);
}

DEVFN float3 split_xyz_to_dwg(float3 c) {
    return MAKE_FLOAT3(
        1.5165889100f * c.x - 0.2814626195f * c.y - 0.1469555778f * c.z,
       -0.4649386618f * c.x + 1.2514818105f * c.y + 0.1748927191f * c.z,
        0.0648602355f * c.x + 0.1091581735f * c.y + 0.7615459883f * c.z);
}

DEVFN float3 split_dwg_linear_to_oklab(float3 c) {
    float3 xyz = split_dwg_to_xyz(c);
    float l = 0.8189330101f * xyz.x + 0.3618667424f * xyz.y - 0.1288597137f * xyz.z;
    float m = 0.0329845436f * xyz.x + 0.9293118715f * xyz.y + 0.0361456387f * xyz.z;
    float s = 0.0482003018f * xyz.x + 0.2643662691f * xyz.y + 0.6338517070f * xyz.z;

    float lp = split_signed_cuberoot(l);
    float mp = split_signed_cuberoot(m);
    float sp = split_signed_cuberoot(s);

    return MAKE_FLOAT3(
        0.2104542553f * lp + 0.7936177850f * mp - 0.0040720468f * sp,
        1.9779984951f * lp - 2.4285922050f * mp + 0.4505937099f * sp,
        0.0259040371f * lp + 0.7827717662f * mp - 0.8086757660f * sp);
}

DEVFN float3 split_oklab_to_dwg_linear(float3 lab) {
    float lp = lab.x + 0.3963377774f * lab.y + 0.2158037573f * lab.z;
    float mp = lab.x - 0.1055613458f * lab.y - 0.0638541728f * lab.z;
    float sp = lab.x - 0.0894841775f * lab.y - 1.2914855480f * lab.z;

    float l = lp * lp * lp;
    float m = mp * mp * mp;
    float s = sp * sp * sp;

    float3 xyz = MAKE_FLOAT3(
         1.2270138511f * l - 0.5577999807f * m + 0.2812561490f * s,
        -0.0405801784f * l + 1.1122568696f * m - 0.0716766787f * s,
        -0.0763812845f * l - 0.4214819784f * m + 1.5861632204f * s);
    return split_xyz_to_dwg(xyz);
}

DEVFN float2 hueToOpponent(float hueDeg) {
    float h = fmod(hueDeg, 360.0f);
    if (h < 0.0f) h += 360.0f;
    float a = h * 0.01745329252f;
    return MAKE_FLOAT2(cos(a), sin(a));
}

DEVFN float3 split_preserve_dwg_y(float3 original, float3 graded) {
    float targetY = split_dwg_y_linear(original);
    float gradedY = split_dwg_y_linear(graded);
    if (targetY > 1e-6f && gradedY > 1e-6f) {
        graded *= clamp(targetY / gradedY, 0.25f, 4.0f);
    }
    float neutralTrim = targetY - split_dwg_y_linear(graded);
    return graded + MAKE_FLOAT3(neutralTrim, neutralTrim, neutralTrim);
}

// Apply split toning to a pixel. Input/output are DWG/DI code values.
//
// Algorithm: Megajuicer-inspired, fully original implementation.
//   • Per-zone Beer-Lambert dye attenuation in OKLab space:
//       opacity = 1 - exp(-filmDensity × subLo/Hi × strength × zoneMask)
//     splitSubLo/splitSubHi act as per-zone density multipliers so shadow
//     and highlight zones can be weighted independently of the balance control.
//   • Cross-zone bleed: shadow and highlight dye vectors are blended slightly,
//     mimicking the way real dye layers partially absorb each other's bandwidth.
//   • Kubelka-Munk hue drift: net density difference rotates the OKLab ab
//     vector, producing the density-dependent hue separation found in film prints.
//   • Highlight protection: a smooth rolloff above DI 0.70 prevents the toning
//     from pulling specular highlights and skin in bright areas.
//   • Dye saturation: user-adjustable target chroma replaces the old hardcoded
//     caps (0.105 / 0.150).
DEVFN float3 applySplitTone(float3 c, CONSTANT_REF(GrainParams) p) {
    float3 lin = MAKE_FLOAT3(di_to_linear(c.x), di_to_linear(c.y), di_to_linear(c.z));
    float lumDi = linear_to_di(max(split_dwg_y_linear(lin), 0.0f));

    float pivot = clamp(p.splitPivot, 0.001f, 0.999f);
    float halfNeutral = clamp(p.splitNeutralWidth, 0.0f, 0.98f) * 0.5f;
    float shadowEdge = max(pivot - halfNeutral, 0.0f);
    float highEdge = min(pivot + halfNeutral, 1.0f);

    float shadowRaw = (shadowEdge - lumDi) / max(shadowEdge, 1e-5f);
    float highRaw = (lumDi - highEdge) / max(1.0f - highEdge, 1e-5f);
    // Toe guard: fade shadow toning to zero in the very deepest blacks
    float toeGuard = splitCurve((lumDi - 0.035f) / (0.12f - 0.035f));
    float shadowMask = splitCurve(shadowRaw) * toeGuard;
    float highMask = splitCurve(highRaw);

    float loAmt = clamp((1.0f - p.splitLoVsHi) * 2.0f, 0.0f, 1.0f);
    float hiAmt = clamp(p.splitLoVsHi * 2.0f, 0.0f, 1.0f);
    float lowWeight = shadowMask * loAmt;
    float highWeight = highMask * hiAmt;
    if (lowWeight + highWeight <= 1e-6f) return c;

    float3 lab = split_dwg_linear_to_oklab(lin);
    float chroma = sqrt(lab.y * lab.y + lab.z * lab.z);
    // Material protection: reduce effect on already-saturated areas
    float materialProtect = 1.0f - 0.45f * splitCurve((chroma - 0.06f) / 0.28f);

    // Per-zone Beer-Lambert concentrations
    float filmDensity = clamp(p.splitFilmDensity, 0.1f, 3.0f);
    float str = clamp(p.splitStrength, 0.0f, 4.0f);
    float subLo = clamp(p.splitSubLo, 0.0f, 1.0f);
    float subHi = clamp(p.splitSubHi, 0.0f, 1.0f);
    float shadowConc = filmDensity * subLo * str * lowWeight;
    float highConc = filmDensity * subHi * str * highWeight;

    float shadowOp = (1.0f - exp(-shadowConc)) * materialProtect;
    // Highlight protection: suppress toning in near-white specular highlights
    float hiProtect = splitCurve((lumDi - 0.70f) / 0.30f);
    float hiProtectMask = 1.0f - clamp(p.splitHiProtect, 0.0f, 1.0f) * hiProtect;
    float highOp = (1.0f - exp(-highConc)) * materialProtect * hiProtectMask;

    // Dye vectors: hue angles → OKLab opponent (a*, b*) unit vectors
    float2 sh = hueToOpponent(p.splitShadowHue);
    float2 hi = hueToOpponent(p.splitHighlightHue);

    // Cross-zone bleed: mix a fraction of each zone's hue vector into the other,
    // mimicking how real dye layers partially overlap in their spectral sensitivity
    float bleed = clamp(p.splitCrossBleed, 0.0f, 0.5f);
    float shBlendA = sh.x + bleed * hi.x;
    float shBlendB = sh.y + bleed * hi.y;
    float hiBlendA = hi.x + bleed * sh.x;
    float hiBlendB = hi.y + bleed * sh.y;
    // Re-normalise so blended vectors stay on the unit circle
    float shLen = max(sqrt(shBlendA * shBlendA + shBlendB * shBlendB), 1e-6f);
    float hiLen = max(sqrt(hiBlendA * hiBlendA + hiBlendB * hiBlendB), 1e-6f);
    shBlendA /= shLen;
    shBlendB /= shLen;
    hiBlendA /= hiLen;
    hiBlendB /= hiLen;

    // Dye saturation: user-adjustable target chroma (replaces hardcoded 0.105/0.150)
    float dyeSat = clamp(p.splitDyeSat, 0.1f, 2.0f);
    float shadowTargetC = min(0.105f * dyeSat, max(chroma * 0.45f, 0.018f + lab.x * 0.16f));
    float highTargetC = min(0.150f * dyeSat, max(chroma * 0.50f, 0.028f + lab.x * 0.20f));

    lab.y = mix(lab.y, shBlendA * shadowTargetC, shadowOp);
    lab.z = mix(lab.z, shBlendB * shadowTargetC, shadowOp);
    lab.y = mix(lab.y, hiBlendA * highTargetC, highOp);
    lab.z = mix(lab.z, hiBlendB * highTargetC, highOp);

    // Kubelka-Munk hue drift: density-dependent ab-vector rotation.
    // In thick dye layers the dominant wavelengths shift; we model this as a
    // small angular rotation proportional to the net shadow/highlight density.
    float kmStr = clamp(p.splitKMStrength, 0.0f, 1.0f);
    if (kmStr > 1e-4f) {
        float curChroma = sqrt(lab.y * lab.y + lab.z * lab.z);
        if (curChroma > 1e-4f) {
            float kmDrift = kmStr * 0.20f * (shadowConc - highConc);
            float angle = atan2(lab.z, lab.y) + kmDrift;
            lab.y = cos(angle) * curChroma;
            lab.z = sin(angle) * curChroma;
        }
    }

    float3 toned = split_oklab_to_dwg_linear(lab);
    toned = split_preserve_dwg_y(lin, toned);

    // Do not let the chroma push create new deep negative channels. Existing
    // negative scene-linear samples are preserved, but not made worse.
    float origMin = min(lin.x, min(lin.y, lin.z));
    float tonedMin = min(toned.x, min(toned.y, toned.z));
    float floorMin = min(origMin, -0.02f);
    if (tonedMin < floorMin) {
        float denom = tonedMin - origMin;
        if (denom < -1e-6f) {
            float fit = clamp((floorMin - origMin) / denom, 0.0f, 1.0f);
            toned = lin + (toned - lin) * fit;
        }
    }

    return MAKE_FLOAT3(linear_to_di(toned.x),
                       linear_to_di(toned.y),
                       linear_to_di(toned.z));
}

// ============================================================
//  Linearize prepass — writes (CST → di_to_linear)(input) to a 2D texture
//  (rgba16f). Multi-tap consumers (diffusion + halation) sample with hardware bilinear,
//  which is dramatically faster than 4 manual buffer reads + 6 lerps per tap
//  thanks to the texture cache and dedicated filter unit.
// ============================================================


// ============================================================
//  Coverage film grain — main kernel (RGB per-channel)
// ============================================================


DEVFN void cwLinearizeKernel_body(
    int x,
    int y,
    GLOBAL_RO(float) input,
    WRITE_TEX linOut,
    CONSTANT_REF(GrainParams) p,
    GLOBAL_RO(float) inputCstLutData)
{
    if (x >= p.width || y >= p.height) return;
    int px = x;
    int py = y;
    int srcIX = px + p.dstX1 - p.srcX1;
    int srcIY = py + p.dstY1 - p.srcY1;
    float3 c = MAKE_FLOAT3(readPx(input, srcIX, srcIY, p.srcWidth, p.srcHeight, p.srcRowBytes, 0),
                      readPx(input, srcIX, srcIY, p.srcWidth, p.srcHeight, p.srcRowBytes, 1),
                      readPx(input, srcIX, srcIY, p.srcWidth, p.srcHeight, p.srcRowBytes, 2));
    if (p.inputCstLutSize > 0) {
        c = applyLut3D(inputCstLutData, p.inputCstLutSize, c);
    }
    float4 lin = MAKE_FLOAT4(di_to_linear(c.x),
                        di_to_linear(c.y),
                        di_to_linear(c.z),
                        0.0f);
    TEX_WRITE(linOut, x, y, lin);
}

// ============================================================
//  Halation prepass — writes halated scene-linear image so the
//  diffusion scatter prepass can compute PSF(halated_image) rather
//  than PSF(original). Opacity is already blended in; the output
//  is: mix(original_lin, halResult_lin, halOpacity).
// ============================================================
DEVFN void cwHalationPrepassKernel_body(
    int x,
    int y,
    READ_TEX linIn,
    READ_TEX halBlurIn,
    WRITE_TEX halLinOut,
    CONSTANT_REF(GrainParams) p)
{
    if (x >= p.width || y >= p.height) return;
    int w = p.width;
    int h = p.height;

    float3 rgb     = hal_sample_prepass(linIn, float(x), float(y), w, h);
    // Read blurred result from dedicated prepass (one tap instead of O(radius²) loop)
    float3 blurred;
    if (p.halBlurPrepassEnabled) {
        float u = (float(x) + 0.5f) / float(w);
        float v = (float(y) + 0.5f) / float(h);
        float4 b = TEX_SAMPLE(halBlurIn, u, v);
        blurred = MAKE_FLOAT3(b.x, b.y, b.z);
    } else {
        blurred = hal_exp_blur(linIn, x, y, w, h, p.halSize);
    }

    float cg     = p.halColor;
    float method = p.halMethod;
    float3 halResult_lin;
    if (method < 0.5f) {
        float3 halated = MAKE_FLOAT3(
            rgb.x + blurred.x * 1.0f,
            rgb.y + blurred.y * cg,
            rgb.z + blurred.z * 0.0f);
        halResult_lin = MAKE_FLOAT3(
            halated.x / 2.0f,
            halated.y / (cg + 1.0f),
            rgb.z);
    } else {
        float cr = 0.7f, cgm = 1.0f - cg / 3.0f;
        float3 halated = MAKE_FLOAT3(
            (rgb.x - blurred.x) * cr,
            (rgb.y - blurred.y) * cgm,
            (rgb.z - blurred.z) * 1.0f);
        halResult_lin = MAKE_FLOAT3(
            halated.x + blurred.x,
            halated.y + blurred.y,
            halated.z + blurred.z);
    }

    float3 out = MAKE_FLOAT3(
        rgb.x + (max(halResult_lin.x, 0.0f) - rgb.x) * p.halOpacity,
        rgb.y + (max(halResult_lin.y, 0.0f) - rgb.y) * p.halOpacity,
        rgb.z + (max(halResult_lin.z, 0.0f) - rgb.z) * p.halOpacity);
    TEX_WRITE(halLinOut, x, y, MAKE_FLOAT4(out.x, out.y, out.z, 1.0f));
}

DEVFN void cwGrainKernel_body(
    int x,
    int y,
    GLOBAL_RO(float) input,
    GLOBAL_RW(float) output,
    CONSTANT_REF(GrainParams) p,
    GLOBAL_RO(float) lutData,
    GLOBAL_RO(float) cstLutData,
    GLOBAL_RO(float) inputCstLutData,
    GLOBAL_RO(float) curveData,
    READ_TEX linInput,
    READ_TEX diffusionBlurInput,
    READ_TEX diffusionGlowInput,
    READ_TEX filmBlurInput,
    READ_TEX halBlurInput)
{
    if (x >= p.width || y >= p.height) return;

    int px = x;
    int py = y;
    int w  = p.width;
    int h  = p.height;

    if (p.debugMode == 1) {
        int dstFloatsPerRow = p.dstRowBytes / 4;
        int dstIdx = py * dstFloatsPerRow + px * 4;
        float band = ((px / 32 + py / 32) & 1) ? 1.0f : 0.0f;
        output[dstIdx + 0] = band;
        output[dstIdx + 1] = 1.0f - band;
        output[dstIdx + 2] = 1.0f;
        output[dstIdx + 3] = 1.0f;
        return;
    }

    int srcW = p.srcWidth;
    int srcH = p.srcHeight;
    int srcIX = px + p.dstX1 - p.srcX1;
    int srcIY = py + p.dstY1 - p.srcY1;

    // Preserve source alpha while the RGB path is rebuilt in DWG/DI.
    float alpha = readPx(input, srcIX, srcIY, srcW, srcH, p.srcRowBytes, 3);

    // Base-plate: film resolution transfer (emulsion resolving power).
    // The film sees a soft/blurry version of the scene; halation and diffusion
    // then add their effects on top of that soft base.
    // Pipeline: film softening → (input CST) → diffusion → halation → grain.
    float3 base;
    if (p.image_blur >= 0.01f && p.filmBlurPrepassEnabled) {
        // Prepass ran — sample the finished blur in one tap. No further blend
        // needed; the prepass already has the correct pixel sigma applied.
        float u = (float(px) + 0.5f) / float(p.width);
        float v = (float(py) + 0.5f) / float(p.height);
        float4 blurred = TEX_SAMPLE(filmBlurInput, u, v);
        base = MAKE_FLOAT3(blurred.x, blurred.y, blurred.z);
    } else if (p.image_blur >= 0.01f) {
        // Fallback: inline Gaussian (CPU path / prepass unavailable).
        base = filmResolveSample3(input, srcIX, srcIY, srcW, srcH, p.srcRowBytes, p.image_blur);
    } else {
        base = MAKE_FLOAT3(
            readPx(input, srcIX, srcIY, srcW, srcH, p.srcRowBytes, 0),
            readPx(input, srcIX, srcIY, srcW, srcH, p.srcRowBytes, 1),
            readPx(input, srcIX, srcIY, srcW, srcH, p.srcRowBytes, 2));
    }

    float3 result = base;

    // ---- Input colour-space transform (camera native → DWG/DI) ----
    if (p.inputCstLutSize > 0) {
        base = applyLut3D(inputCstLutData, p.inputCstLutSize, base);
        result = base;
    }

    // Optical diffusion: energy-conserving scatter fraction.
    // out = mix(input, PSF*input, p_s) where p_s = diffusionAmount.
    // Diffusion is a lens-filter effect (pre-film), so it runs before halation.
    // recv_lin is the pre-halation scene-linear value; the scatter prepass was
    // computed from the original linearized image.
    if (p.diffusionEnabled && (p.diffusionAmount > 1e-6f || p.diffusionVeil > 1e-6f)) {
        DiffusionScatterResult scatter = diffusion_scatter_precomputed(
            diffusionBlurInput, diffusionGlowInput, float(px), float(py), w, h);

        float3 recv_lin = MAKE_FLOAT3(di_to_linear(base.x),
                                 di_to_linear(base.y),
                                 di_to_linear(base.z));

        float p_s = clamp(p.diffusionAmount, 0.0f, 1.0f);
        // highlightBoost 0..1 scales glowGain from 1× up to 9× — drives blown-out practicals
        float boostMult = 1.0f + clamp(p.diffusionHighlightBoost, 0.0f, 1.0f) * 8.0f;
        float glowGain = p_s * 0.72f * boostMult;
        float veilGain = clamp(p.diffusionVeil, 0.0f, 0.35f) * p_s * 1.4f;

        // Optional color tint overlay (neutral by default; controlled by tint color picker)
        float tintStrength = clamp(p.diffusionTintStrength, 0.0f, 1.0f);
        float3 diffusionTint = mix(MAKE_FLOAT3(1.0f, 1.0f, 1.0f),
                                  diffusion_picker_tint(p.diffusionTintR, p.diffusionTintG, p.diffusionTintB),
                                  tintStrength);

        float3 soft_lin = max(scatter.blur * diffusionTint, MAKE_FLOAT3(0.0f, 0.0f, 0.0f));
        float3 total_lin = mix(recv_lin, soft_lin, p_s);
        total_lin += max(scatter.glow * diffusionTint, MAKE_FLOAT3(0.0f, 0.0f, 0.0f)) * glowGain;
        if (veilGain > 1e-6f) {
            float veil_lin = max(scatter.veil, 0.0f) * veilGain;
            total_lin += MAKE_FLOAT3(veil_lin, veil_lin, veil_lin) * diffusionTint;
        }

        total_lin = max(total_lin, MAKE_FLOAT3(0.0f, 0.0f, 0.0f));
        result = MAKE_FLOAT3(linear_to_di(total_lin.x),
                             linear_to_di(total_lin.y),
                             linear_to_di(total_lin.z));
        base = result;
    }

    // Halation — exponential falloff model. Runs after diffusion and film softening
    // (halation is a film emulsion artifact; diffusion is a lens-filter effect).
    // Operates in scene-linear; result is re-encoded to DI.
    if (p.halEnabled && p.halOpacity > 1e-6f) {
        float r0     = p.halSize;
        float cg     = p.halColor;
        float method = p.halMethod;

        // Scene-linear value of the current pixel.
        // When film blur is active, `result` is the film-blurred DI base; linearise
        // it so both `rgb` and `origDI` are consistent with the soft base plate.
        // This eliminates the sharp-edge orange fringing that occurs when halation
        // computes a crisp glow from the sharp `linInput` and adds it to a blurry base.
        float3 rgb;
        float3 origDI;
        if (p.filmBlurPrepassEnabled && p.image_blur >= 0.01f) {
            rgb = MAKE_FLOAT3(di_to_linear(result.x),
                              di_to_linear(result.y),
                              di_to_linear(result.z));
            origDI = result;
        } else {
            rgb = hal_sample_prepass(linInput, float(px), float(py), w, h);
            origDI = MAKE_FLOAT3(linear_to_di(max(rgb.x, 0.0f)),
                                 linear_to_di(max(rgb.y, 0.0f)),
                                 linear_to_di(max(rgb.z, 0.0f)));
        }

        // Exponential-blurred version: read from halation blur prepass (one tap)
        // or recompute inline as fallback (CPU path / prepass unavailable).
        float3 blurred;
        if (p.halBlurPrepassEnabled) {
            float u = (float(px) + 0.5f) / float(w);
            float v = (float(py) + 0.5f) / float(h);
            float4 b = TEX_SAMPLE(halBlurInput, u, v);
            blurred = MAKE_FLOAT3(b.x, b.y, b.z);
        } else {
            blurred = hal_exp_blur(linInput, px, py, w, h, r0);
        }

        float3 halResult_lin;
        if (method < 0.5f) {
            // Proosa: energy-preserving additive method
            // color = (1.0, cg, 0.0)
            float3 halated = MAKE_FLOAT3(
                rgb.x + blurred.x * 1.0f,
                rgb.y + blurred.y * cg,
                rgb.z + blurred.z * 0.0f);
            halResult_lin = MAKE_FLOAT3(
                halated.x / 2.0f,
                halated.y / (cg + 1.0f),
                rgb.z);
        } else {
            // Minuth: frequency-separation method
            // color = (0.7, 1.0 - cg/3.0, 1.0)
            float cr = 0.7f, cgm = 1.0f - cg / 3.0f;
            float3 halated = MAKE_FLOAT3(
                (rgb.x - blurred.x) * cr,
                (rgb.y - blurred.y) * cgm,
                (rgb.z - blurred.z) * 1.0f);
            halResult_lin = MAKE_FLOAT3(
                halated.x + blurred.x,
                halated.y + blurred.y,
                halated.z + blurred.z);
        }

        // Re-encode to DI. Halation is an additive effect: compute the delta
        // relative to the original pixel so the blend preserves any diffusion
        // (or other effects) already in result rather than overwriting them.
        float3 halResult = MAKE_FLOAT3(
            linear_to_di(max(halResult_lin.x, 0.0f)),
            linear_to_di(max(halResult_lin.y, 0.0f)),
            linear_to_di(max(halResult_lin.z, 0.0f)));

        result = result + (halResult - origDI) * p.halOpacity;
        base = result;
    }

    // Contrast reduction — two steps:
    //  1. One-sided compression: only values above DI 0.336 are squeezed down,
    //     so offset/blacks are not lifted.
    //  2. Global offset pulldown: the whole image shifts down proportionally to k,
    //     so overall brightness decreases, not just the highlights.
    //     Scale factor 0.08 → at k=1.0 the image shifts down ~0.08 DI units.
    if (p.diffusionEnabled && p.diffusionContrastReduction > 1e-6f) {
        const float cPivot    = 0.336f;
        const float kPulldown = 0.04f;
        float k = clamp(p.diffusionContrastReduction, 0.0f, 1.0f);
        if (result.x > cPivot) result.x = cPivot + (result.x - cPivot) * (1.0f - k);
        if (result.y > cPivot) result.y = cPivot + (result.y - cPivot) * (1.0f - k);
        if (result.z > cPivot) result.z = cPivot + (result.z - cPivot) * (1.0f - k);
        float pulldown = k * kPulldown;
        result.x = max(result.x - pulldown, 0.0f);
        result.y = max(result.y - pulldown, 0.0f);
        result.z = max(result.z - pulldown, 0.0f);
        base = result;
    }

    // Grain — develops on the combined exposure map (base + halation + diffusion).
    // The layered impulse model keeps crystal positions stable while changing
    // density per frame, then modulates amplitude by channel exposure.
    if (p.grain_opacity >= 1e-6f) {
        // Cell size controls visible crystal spacing; grain blur softens the
        // impulse footprint (more crystal overlap) without widening the lattice.
        float cellSize = max(0.45f, p.grain_size);
        float softness = clamp(p.grain_blur / max(p.grain_size, 0.25f), 0.0f, 2.0f);

        // Size -> amplitude law (resolution-invariant): coarse grain is punchier.
        float eff_size  = max(p.grain_size, 0.1f);
        float ref       = GRAIN_REF_SIZE * float(p.height) / 2160.0f;
        float size_gain = clamp(sqrt(eff_size / max(ref, 1e-3f)), 0.6f, 3.0f);

        float contrast = clamp(p.grain_contrast, 0.0f, 1.0f);

        // One seed: static = emulsion identity (frame-invariant spatial structure),
        //           frame  = per-frame white-noise realisation (fully independent).
        //           env    = development-density envelope, quantized to 8-frame blocks
        //                    so large-scale grain energy drifts slowly (~3×/sec at 24fps).
        uint static_seed = cw_hash(uint(p.seed_offset) ^ 0xDEADBEEFu);
        uint frame_base  = cw_hash(uint(p.seed_offset) + 0xA5A5A5A5u);
        uint frame_seed  = cw_hash(cw_hash(uint(p.frame_index) ^ frame_base));
        uint env_seed    = cw_hash(uint(p.frame_index & ~7u) ^ frame_base ^ 0xE6A31C7Fu);

        float s = p.grain_strength * size_gain;
        float highlight_suppress = 1.35f * p.hi_rolloff;
        float irr = clamp(p.shape_irregularity, 0.0f, 1.0f);

        // RGB grain — one vectorised call evaluates all three channels in a
        // single 9-cell neighbourhood loop instead of six separate passes.
        float3 grain = filmGrain3(px, py, cellSize, softness,
                                   contrast, clamp(result, 0.0f, 1.0f),
                                   irr, static_seed,
                                   frame_seed,
                                   frame_seed + 111111u,
                                   frame_seed + 222222u,
                                   env_seed,
                                   highlight_suppress);

        // Luma-preserving saturation on grain signal
        float g_lum = 0.2126f * grain.x + 0.7152f * grain.y + 0.0722f * grain.z;
        grain.x = g_lum + (grain.x - g_lum) * p.grain_sat;
        grain.y = g_lum + (grain.y - g_lum) * p.grain_sat;
        grain.z = g_lum + (grain.z - g_lum) * p.grain_sat;

        float3 grained = result + grain * s;
        result = result + (grained - result) * p.grain_opacity;
    }

    // Apply split toning (before Look LUT, after diffusion + halation + grain)
    if (p.splitEnabled && p.splitStrength > 1e-6f) {
        result = applySplitTone(result, p);
    }

    // Apply printer points as true exposure trims in DWG/DI (1 point = 1/12 stop).
    // Placed BEFORE the Look LUT so the trim models printer-light exposure of
    // the print stock through the negative, matching the photochemical chain.
    if (p.printEnabled) {
        result.x = apply_printer_lights(result.x, p.printGainR);
        result.y = apply_printer_lights(result.y, p.printGainG);
        result.z = apply_printer_lights(result.z, p.printGainB);
    }

    // Apply 3D Look, then blend it back, then apply push/pull.
    // Push/pull simulates altered film development time as a per-channel
    // gamma curve about a linear mid-grey pivot (see block below).
    if (p.lutEnabled && p.lutSize > 0) {
        float3 looked = applyLut3D(lutData, p.lutSize, result);
        // Bradford CAT: matrix is precomputed in C++ from the user white point
        // to the LUT's native D65 white, in DWG linear light.
        // Passed row-major; Metal float3x3 constructor takes columns, so we transpose.
        if (p.wpEnabled) {
            float3x3 cat = float3x3(
                MAKE_FLOAT3(p.wpMat[0], p.wpMat[3], p.wpMat[6]),  // col 0
                MAKE_FLOAT3(p.wpMat[1], p.wpMat[4], p.wpMat[7]),  // col 1
                MAKE_FLOAT3(p.wpMat[2], p.wpMat[5], p.wpMat[8])); // col 2
            float3 lin = MAKE_FLOAT3(
                max(di_to_linear(looked.x), 0.0f),
                max(di_to_linear(looked.y), 0.0f),
                max(di_to_linear(looked.z), 0.0f));
            lin = cat * lin;
            lin = max(lin, MAKE_FLOAT3(0.0f, 0.0f, 0.0f));
            looked = MAKE_FLOAT3(linear_to_di(lin.x),
                                 linear_to_di(lin.y),
                                 linear_to_di(lin.z));
        }
        // Blend the LUT result back. lutIntensity controls saturation/chroma contribution;
        // lutContrastAmt controls contrast curve/luma contribution (Film Emulation only).
        // When equal (Commercial Looks path always sets them equal), the fast mix path is used.
        if (p.lutContrastAmt == p.lutIntensity) {
            result = mix(result, looked, p.lutIntensity);
        } else {
            // Decompose original and LUT output in Oklab; blend L* and a*/b* independently.
            float3 orig_lin   = MAKE_FLOAT3(di_to_linear(result.x),
                                            di_to_linear(result.y),
                                            di_to_linear(result.z));
            float3 looked_lin = MAKE_FLOAT3(di_to_linear(looked.x),
                                            di_to_linear(looked.y),
                                            di_to_linear(looked.z));
            float3 orig_lab   = split_dwg_linear_to_oklab(orig_lin);
            float3 looked_lab = split_dwg_linear_to_oklab(looked_lin);

            float3 blended_lab = MAKE_FLOAT3(
                mix(orig_lab.x, looked_lab.x, p.lutContrastAmt),  // L* — tone/contrast
                mix(orig_lab.y, looked_lab.y, p.lutIntensity),    // a* — color
                mix(orig_lab.z, looked_lab.z, p.lutIntensity)     // b* — color
            );

            float3 out_lin = split_oklab_to_dwg_linear(blended_lab);
            result = MAKE_FLOAT3(linear_to_di(out_lin.x),
                                 linear_to_di(out_lin.y),
                                 linear_to_di(out_lin.z));
        }

        // ---- Push / Pull processing ----------------------------------------
        // Models the net visual signature of altered development time on a
        // film negative.  The Look LUT *is* the film response at normal
        // development, so push/pull is applied as a post-LUT dev-stage curve
        // rather than literal pre-exposure (which would just darken the
        // image).  Mid-grey is preserved; shadows crush / highlights extend
        // on push and the inverse on pull, mirroring the characteristic
        // curve change documented for Kodak Vision3 / Ilford HP5+ etc.
        //
        //   out_c = pivot * (in_c / pivot)^(1 + k*N)         per channel
        //   pivot = 0.18 (linear mid-grey)
        //   k     ≈ 0.18 stops⁻¹  (≈ 13–20% gamma change per stop)
        //
        // The per-channel power curve naturally produces the "saturated and
        // distorted colours" signature of pushed colour negative (each dye
        // layer's curve diverges in the extremes).  A small global HSL
        // saturation nudge (±4 %/stop) finishes the look.
        // --------------------------------------------------------------------
        if (p.lutPushPull != 0.0f) {
            const float pivot     = 0.18f;
            const float kGamma    = 0.18f;   // contrast slope per stop
            const float kSat      = 0.04f;   // saturation slope per stop
            float gamma = 1.0f + kGamma * p.lutPushPull;

            // Per-channel power curve about the linear mid-grey pivot.
            // Negative values are reflected so the curve stays well-defined
            // for sub-zero scene-linear samples (DWG can carry them).
            float3 r = result / pivot;
            float3 sgn = MAKE_FLOAT3(r.x < 0.0f ? -1.0f : 1.0f,
                                r.y < 0.0f ? -1.0f : 1.0f,
                                r.z < 0.0f ? -1.0f : 1.0f);
            float3 mag = pow(max(abs(r), MAKE_FLOAT3(1e-8f, 1e-8f, 1e-8f)), MAKE_FLOAT3(gamma, gamma, gamma));
            result = pivot * sgn * mag;

            // Global saturation trim: push +sat (distorted colour),
            // pull –sat (flatter palette).
            float3 hsl = rgb_to_hsl(result);
            hsl.y = clamp(hsl.y * (1.0f + kSat * p.lutPushPull), 0.0f, 1.0f);
            result = hsl_to_rgb(hsl);
        }
    }

    // ---- Output colour-space transform (3D LUT baked from Resolve CST) ----
    if (p.cstLutSize > 0) {
        result = applyLut3D(cstLutData, p.cstLutSize, result);
    }

    // ---- Contrast curve overlay (full-image) ----
    // curveData layout: [256 R][256 G][256 B]
    // X axis: input value (left=0/dark, right=1/bright).
    // Y axis: output value (bottom=0/dark, top=1/bright).
    // Curve is inset 5% from top and bottom so clipped values stay clearly visible.
    if (p.curveEnabled) {
        float fx = float(px) / float(max(w - 1, 1));        // 0..1 horizontal (input)
        float fy = float(py) / float(max(h - 1, 1));        // 0..1 vertical (py=0 is bottom in OFX)

        // Sample R/G/B curves at fx (linear interp between adjacent samples)
        float fSample = clamp(fx, 0.0f, 1.0f) * 255.0f;
        int   i0 = int(floor(fSample));
        int   i1 = min(i0 + 1, 255);
        float ft = fSample - float(i0);
        float curveValR = mix(curveData[i0],         curveData[i1],         ft);
        float curveValG = mix(curveData[256 + i0],   curveData[256 + i1],   ft);
        float curveValB = mix(curveData[512 + i0],   curveData[512 + i1],   ft);

        // Inset: map curve value [0,1] → screen Y [0.05, 0.95] so clipped
        // values (0 or 1) render visibly inside the frame, not on the edge.
        const float kInset = 0.05f;
        const float kRange = 1.0f - 2.0f * kInset;
        float yR = kInset + clamp(curveValR, 0.0f, 1.0f) * kRange;
        float yG = kInset + clamp(curveValG, 0.0f, 1.0f) * kRange;
        float yB = kInset + clamp(curveValB, 0.0f, 1.0f) * kRange;

        // Line thickness scales with image height for consistent on-screen size
        float lineHalfPx = max(1.5f, float(h) * 0.0015f);
        float dR = abs(fy - yR) * float(h);
        float dG = abs(fy - yG) * float(h);
        float dB = abs(fy - yB) * float(h);
        if (dR < lineHalfPx) {
            float a = (1.0f - dR / lineHalfPx) * 0.95f;
            result = mix(result, MAKE_FLOAT3(1.0f, 0.25f, 0.25f), a);
        }
        if (dG < lineHalfPx) {
            float a = (1.0f - dG / lineHalfPx) * 0.95f;
            result = mix(result, MAKE_FLOAT3(0.25f, 1.0f, 0.30f), a);
        }
        if (dB < lineHalfPx) {
            float a = (1.0f - dB / lineHalfPx) * 0.95f;
            result = mix(result, MAKE_FLOAT3(0.40f, 0.55f, 1.0f), a);
        }
    }

    // Write output
    int dstFloatsPerRow = p.dstRowBytes / 4;
    int dstIdx = py * dstFloatsPerRow + px * 4;
    output[dstIdx + 0] = result.x;
    output[dstIdx + 1] = result.y;
    output[dstIdx + 2] = result.z;
    output[dstIdx + 3] = alpha;
}
