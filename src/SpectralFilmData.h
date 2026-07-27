// ============================================================
// SpectralFilm OFX — Film stock data structures
// Matches the binary layout written by gen_filmdata.py exactly.
// ============================================================
#pragma once
#include <cstdint>

// DaVinci Resolve Metal GPU rendering extension property strings
#ifndef kOfxImageEffectPropMetalRenderSupported
#define kOfxImageEffectPropMetalRenderSupported  "OfxImageEffectPropMetalRenderSupported"
#endif
#ifndef kOfxImageEffectPropMetalEnabled
#define kOfxImageEffectPropMetalEnabled          "OfxImageEffectPropMetalEnabled"
#endif
#ifndef kOfxImageEffectPropMetalCommandQueue
#define kOfxImageEffectPropMetalCommandQueue     "OfxImageEffectPropMetalCommandQueue"
#endif

static constexpr int SF_LUT_RES     = 33;   // 2D input LUT resolution
static constexpr int SF_CURVE_N     = 512;  // Samples per characteristic curve
static constexpr int SF_N_WAVEL     = 81;   // 380–780 nm at 5 nm steps
static constexpr int SF_N_BANDS     = 27;   // 3 wavelengths per band
static constexpr int SF_LUT3D_SIZE  = 33;   // 3D film LUT resolution

static constexpr int SF_MAX_STOCKS  = 64;   // Maximum stocks in binary file
static constexpr int SF_NAME_LEN    = 64;

// ============================================================
// GPU-side per-pixel parameter block (uploaded once per render)
//
// Used as a flat float buffer on the GPU — int fields are bit-cast.
// Layout must match SpectralKernelCore.inc offset table exactly.
//
// Removed from UI (hardcoded in buildParams):
//   wbEnabled / wbMat  → always identity (WB handled upstream)
//   expComp / pushPull → always 0 (baked into LUT at gen time)
//   printerR/G/B       → always 0
//   outputDWG          → always 1 (pipeline always outputs DWG/DI)
// ============================================================
struct SpectralParams {
    // Image layout
    int   width;
    int   height;
    int   srcRowBytes;
    int   dstRowBytes;

    // [4]      wbEnabled  (hardcoded 0)
    // [5..13]  wbMat[9]   (hardcoded identity)
    // [14]     expComp    (hardcoded 0)
    // [15]     pushPull   (hardcoded 0)
    // [16..18] printerR/G/B (hardcoded 0)
    int   _wbEnabled;
    float _wbMat[9];
    float _expComp;
    float _pushPull;
    float _printerR, _printerG, _printerB;

    int   lutSize;
    int   outputDWG;    // hardcoded 1
};

// ============================================================
// Host-side per-stock record (loaded from binary file)
// ============================================================
struct NegativeStockRecord {
    char  name[SF_NAME_LEN];

    // 2D input LUT: linear exposure per unit S at D65 6500K
    float inputLut[SF_LUT_RES][SF_LUT_RES][3];

    // Characteristic curve domain
    float logHMin;
    float logHMax;

    // Characteristic curves [3][CURVE_N]
    float curve[3][SF_CURVE_N];

    // Photometric references
    float dRef[3];
    float dMin[3];

    // Reduced spectral data [27][3]
    float specDens27[SF_N_BANDS][3];
    float dMinSd27[SF_N_BANDS];

    // Full spectral data [81] or [81][3]
    float dRefSdFull[SF_N_WAVEL];
    float dMinSdFull[SF_N_WAVEL];
    float specDensFull[SF_N_WAVEL][3];
    float sensitivityFull[SF_N_WAVEL][3];

    float colorMasking;
};

struct PrintStockRecord {
    char  name[SF_NAME_LEN];

    float logHMin;
    float logHMax;
    float curve[3][SF_CURVE_N];
    float dRef[3];
    float dMin[3];

    // Reduced spectral data [27][3]
    float specDens27[SF_N_BANDS][3];
    float projOutputMat27[SF_N_BANDS][3]; // XYZ output matrix per band

    float hRef[3];

    // Full spectral data
    float dMinSdFull[SF_N_WAVEL];
    float sensitivityFull[SF_N_WAVEL][3];
    float specDensFull[SF_N_WAVEL][3];
};
