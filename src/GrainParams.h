#ifndef GRAIN_PARAMS_H
#define GRAIN_PARAMS_H

// ============================================================
// Shared parameter struct — identical layout in C++ and Metal/CUDA.
// All fields are 4-byte (int/float); no padding surprises.
// Expected size is enforced by the static_assert below.
// ============================================================

struct GrainParams {
    float grain_size;       // grain crystal cell px
    float grain_strength;   // additive amplitude multiplier
    float grain_blur;       // grain crystal edge softness px
    float grain_opacity;    // cross-fade clean-soft ↔ grained
    float grain_sat;        // luma-preserving saturation on RGB grain
    float image_blur;       // base-plate film-resolution transfer amount
    float hi_rolloff;       // highlight grain attenuation 0..1
    int   seed_offset;      // user seed variant
    int   frame_index;      // timeline frame number
    int   width;            // image width  (pixels)
    int   height;           // image height (pixels)
    int   srcWidth;         // source image width (pixels)
    int   srcHeight;        // source image height (pixels)
    int   srcX1;            // source image bounds origin X
    int   srcY1;            // source image bounds origin Y
    int   dstX1;            // output image bounds origin X
    int   dstY1;            // output image bounds origin Y
    int   srcRowBytes;      // source row stride in bytes
    int   dstRowBytes;      // output row stride in bytes
    int   lutEnabled;       // look: 0 = off, 1 = on
    int   lutSize;          // look dimension (e.g. 65)
    float lutIntensity;     // look saturation/chroma blend 0..1
    float lutContrastAmt;   // look contrast curve/luma blend 0..1
    float lutPushPull;      // push/pull stops (post-LUT dev-stage gamma curve)
    int   halEnabled;       // halation: 0 = off, 1 = on
    float halSize;          // exponential blur radius r0 (0.0–5.0, default 1.0)
    float halColor;         // green channel pass-through (0.0–0.5, default 0.15)
    float halMethod;        // 0.0 = Proosa, 1.0 = Minuth (stored as float for layout)
    float _halReserved0;    // reserved
    float _halReserved1;    // reserved
    float _halReserved2;    // reserved
    float _halReserved3;    // reserved
    float _halReserved4;    // reserved
    float _halReserved5;    // reserved
    float halOpacity;           // final blend opacity
    float diffusionHaloWarmth;  // halo warmth (family_base + tint_strength offset, 0..2)
    int   diffusionPsfFamily;   // PSF family: 0=glimmerglass 1=bpm 2=pro_mist 3=cinebloom
    int   diffusionEnabled;      // diffusion: 0 = off, 1 = on
    float diffusionAmount;       // overall diffusion strength 0..2
    float diffusionSize;       // overall diffusion size/radius in output pixels
    float diffusionSoftness;   // profile softness 1.0 = legacy shape
    float diffusionBodyStrength;      // body-lobe strength 0..1
    float diffusionFalloffStrength;  // broad falloff-lobe strength 0..1
    float diffusionHighlightGate;    // DI-domain highlight gate for extra glow
    float diffusionVeil;         // neutral veiling lift 0..0.35
    float diffusionContrastReduction; // contrast reduction 0..1 pivoted at DI 0.336
    float diffusionHighlightBoost;    // extra glow gain for blowout on practicals 0..1
    float diffusionTintStrength; // diffusion tint amount 0..1
    float diffusionTintR;        // diffusion tint color red
    float diffusionTintG;        // diffusion tint color green
    float diffusionTintB;        // diffusion tint color blue
    int   diffusionScatterMode;  // 0 = inline full-res, 1 = precomputed texture
    int   diffusionScatterWidth; // precomputed scatter texture width
    int   diffusionScatterHeight;// precomputed scatter texture height
    int   printEnabled;     // printer points: 0 = off, 1 = on
    float printGainR;       // printer points: red exposure gain (1 point = 1/12 stop)
    float printGainG;       // printer points: green exposure gain (1 point = 1/12 stop)
    float printGainB;       // printer points: blue exposure gain (1 point = 1/12 stop)
    int   wpEnabled;        // 0 = identity, 1 = apply Bradford CAT below
    float wpMat[9];         // Bradford CAT DWG-linear→DWG-linear, row-major 3×3
    int   outTransform;    // output colour space: 0=none, 1..N from discovered CST output LUTs
    int   cstLutSize;      // output CST LUT dimension (e.g. 33 or 65), 0 = no CST
    int   inTransform;     // input colour space: 0=none (DWG/DI), 1..N from discovered CST input LUTs
    int   inputCstLutSize; // input CST LUT dimension, 0 = no input CST
    // ---- Split Toning ----
    int   splitEnabled;     // 0 = off, 1 = on
    float splitStrength;    // split-tone amount / saturation 0..4
    float splitShadowHue;   // shadow hue angle in degrees (0..360)
    float splitHighlightHue;// highlight hue angle in degrees (0..360)
    float splitLoVsHi;      // balance: 0 = shadow only, 1 = highlight only
    float splitPivot;       // luminance pivot (DWG/DI mid-grey ≈ 0.336)
    float splitSubLo;       // subtractive blend for shadows 0..1
    float splitSubHi;       // subtractive blend for highlights 0..1
    float splitNeutralWidth;// neutral deadzone width around pivot
    float splitFilmDensity; // Beer-Lambert optical density multiplier (0.2..3.0, default 1.0)
    float splitKMStrength;  // Kubelka-Munk hue drift strength 0..1
    float splitHiProtect;   // suppress toning in specular highlights 0..1
    float splitDyeSat;      // dye saturation / target-chroma scale 0.1..2.0 (default 1.0)
    float splitCrossBleed;  // cross-zone hue bleed 0..0.5 (shadow↔highlight dye mix)
    int   curveEnabled;     // contrast curve overlay: 0 = off, 1 = on
    // ---- Resolution / proxy awareness ----
    float renderScale;      // OFX render scale (1.0 = full res); used by host-side scaling
    // ---- Grain shape ----
    float shape_irregularity; // 0..1 crystal size/density variation (emulsion anatomy)
    float grain_blur_link;    // 0..1 amount base-plate resolution loss tracks grain size
    float grain_contrast;     // 0..1 grain histogram crunch (0 = neutral)
    int   debugMode;          // gated diagnostic render mode; 0 = normal
    int   filmBlurPrepassEnabled; // 1 = separable prepass ran; main kernel samples filmBlurInput tex
    int   halBlurPrepassEnabled;  // 1 = halation blur prepass ran; hal/main kernels read halBlurInput
};

static_assert(sizeof(GrainParams) == 376, "GrainParams size mismatch — Metal/C++ layout diverged");

// ---- Film format IDs ----
#define FMT_65MM   0
#define FMT_35MM   1
#define FMT_16MM   2
#define FMT_8MM    3
#define FMT_CUSTOM 4

// ---- Diffusion profile preset IDs ----
// Numeric index order matches the dropdown in the OFX UI;
// DIFFUSION_PROFILE_CUSTOM must remain LAST.
#define DIFFUSION_PROFILE_BPM_18            0   // Black Pro-Mist 1/8
#define DIFFUSION_PROFILE_BPM_14            1   // Black Pro-Mist 1/4
#define DIFFUSION_PROFILE_BPM_12            2   // Black Pro-Mist 1/2
#define DIFFUSION_PROFILE_CUSTOM            3

// ---- PSF family IDs (stored in GrainParams.diffusionPsfFamily) ----
#define PSF_FAMILY_GLIMMERGLASS  0
#define PSF_FAMILY_BPM           1
#define PSF_FAMILY_PRO_MIST      2
#define PSF_FAMILY_CINEBLOOM     3

// ---- Apparent grain radii at 4K (pixels) ----
#define RADIUS_65MM  0.5f
#define RADIUS_35MM  1.5f
#define RADIUS_16MM  3.5f
#define RADIUS_8MM   7.0f

// ============================================================
// OFX GPU render property strings.
// Defined here as fallbacks in case the OFX headers are older
// than the GPU render extension (OFX ≥ 1.4).
// ============================================================
#ifndef kOfxImageEffectPropCudaRenderSupported
#define kOfxImageEffectPropCudaRenderSupported   "OfxImageEffectPropCudaRenderSupported"
#endif
#ifndef kOfxImageEffectPropCudaEnabled
#define kOfxImageEffectPropCudaEnabled           "OfxImageEffectPropCudaEnabled"
#endif
#ifndef kOfxImageEffectPropCudaStream
#define kOfxImageEffectPropCudaStream            "OfxImageEffectPropCudaStream"
#endif
#ifndef kOfxImageEffectPropMetalRenderSupported
#define kOfxImageEffectPropMetalRenderSupported   "OfxImageEffectPropMetalRenderSupported"
#endif
#ifndef kOfxImageEffectPropMetalEnabled
#define kOfxImageEffectPropMetalEnabled           "OfxImageEffectPropMetalEnabled"
#endif
#ifndef kOfxImageEffectPropMetalCommandQueue
#define kOfxImageEffectPropMetalCommandQueue      "OfxImageEffectPropMetalCommandQueue"
#endif
#ifndef kOfxImageEffectPropOpenCLRenderSupported
#define kOfxImageEffectPropOpenCLRenderSupported  "OfxImageEffectPropOpenCLRenderSupported"
#endif

// ============================================================
// OFX Parameter name constants
// ============================================================
#define PARAM_GRAIN_ENABLE    "grain_enable"
#define PARAM_FILM_FORMAT     "film_format"
#define PARAM_GRAIN_SIZE      "grain_size"
#define PARAM_GRAIN_STRENGTH  "grain_strength"
#define PARAM_GRAIN_BLUR      "grain_blur"
#define PARAM_GRAIN_OPACITY   "grain_opacity"
#define PARAM_GRAIN_SAT       "grain_sat"
#define PARAM_SHAPE_IRREG     "grain_shape_irregularity"
#define PARAM_GRAIN_BLUR_LINK "grain_blur_link"
#define PARAM_GRAIN_CONTRAST  "grain_contrast"
#define PARAM_IMAGE_BLUR_ENABLE "image_blur_enable"
#define PARAM_IMAGE_BLUR      "image_blur"
#define PARAM_HI_ROLLOFF      "hi_rolloff"
#define PARAM_SEED_OFFSET     "seed_offset"
#define PARAM_LUT_ENABLE      "lut_enable"
#define PARAM_LUT_SELECT      "lut_select"
#define PARAM_LUT_INTENSITY   "lut_intensity"
#define PARAM_LUT_PUSH_PULL   "lut_push_pull"
#define PARAM_WP_ADAPT        "wp_adapt"
#define PARAM_WP_KELVIN_ENABLE "wp_kelvin_enable"
#define PARAM_WP_KELVIN       "wp_kelvin"

#define PARAM_HAL_ENABLE      "hal_enable"
#define PARAM_HAL_SIZE        "hal_size"
#define PARAM_HAL_COLOR       "hal_color"
#define PARAM_HAL_METHOD      "hal_method"
#define PARAM_HAL_OPACITY     "hal_opacity"

// Serialized keys remain mist_* so existing Resolve projects and presets load.
#define PARAM_DIFFUSION_ENABLE      "mist_enable"
#define PARAM_DIFFUSION_PROFILE     "mist_filter"
#define PARAM_DIFFUSION_AMOUNT      "mist_amount"
#define PARAM_DIFFUSION_SIZE        "mist_spread"
#define PARAM_DIFFUSION_SOFTNESS    "mist_softness"
#define PARAM_DIFFUSION_BODY_STRENGTH    "mist_tail_mix"
#define PARAM_DIFFUSION_FALLOFF_STRENGTH "mist_tail_falloff"
#define PARAM_DIFFUSION_HIGHLIGHT_GATE   "mist_threshold"
#define PARAM_DIFFUSION_VEIL        "mist_veil"
#define PARAM_DIFFUSION_CONTRAST    "mist_contrast"
#define PARAM_DIFFUSION_HIGHLIGHT_BOOST "mist_bloom"
#define PARAM_DIFFUSION_TINT_STRENGTH "mist_tint_strength"
#define PARAM_DIFFUSION_TINT_COLOR  "mist_tint_color"
#define PRESET_DIFFUSION_TINT_R     "mist_tint_r"
#define PRESET_DIFFUSION_TINT_G     "mist_tint_g"
#define PRESET_DIFFUSION_TINT_B     "mist_tint_b"

#define PARAM_PRINT_ENABLE    "print_enable"
#define PARAM_PRINT_GANG      "print_gang"
#define PARAM_PRINT_R         "print_r"
#define PARAM_PRINT_G         "print_g"
#define PARAM_PRINT_B         "print_b"

#define PARAM_WEAVE_ENABLE    "weave_enable"
#define PARAM_WEAVE_AMOUNT    "weave_amount"
#define PARAM_WEAVE_SPEED     "weave_speed"
#define PARAM_WEAVE_SEED      "weave_seed"

#define PARAM_SPLIT_ENABLE      "split_enable"
#define PARAM_SPLIT_STRENGTH    "split_strength"
#define PARAM_SPLIT_SHADOW_HUE  "split_shadow_hue"
#define PARAM_SPLIT_HI_HUE      "split_highlight_hue"
#define PARAM_SPLIT_LO_VS_HI    "split_lo_vs_hi"
#define PARAM_SPLIT_PIVOT        "split_pivot"
#define PARAM_SPLIT_SUB_LO       "split_sub_lo"
#define PARAM_SPLIT_SUB_HI       "split_sub_hi"
#define PARAM_SPLIT_NEUTRAL_W    "split_neutral_width"
#define PARAM_SPLIT_FILM_DENSITY "split_film_density"
#define PARAM_SPLIT_KM_STRENGTH  "split_km_strength"
#define PARAM_SPLIT_HI_PROTECT   "split_hi_protect"
#define PARAM_SPLIT_DYE_SAT      "split_dye_sat"
#define PARAM_SPLIT_CROSS_BLEED  "split_cross_bleed"

#define PARAM_OUTPUT_TRANSFORM "output_transform"
#define PARAM_INPUT_TRANSFORM  "input_transform"

#define PARAM_CURVE_ENABLE    "curve_enable"

#define PARAM_PRESET_NAME     "preset_name"
#define PARAM_SAVE_PRESET     "save_preset"
#define PARAM_REFRESH_PRESETS "refresh_presets"

#define PARAM_MASTER_PRESET   "master_preset"

// Sentinel: 0 = fresh instance (apply defaults), 1 = copy/restore (skip defaults).
// Hidden from the UI; saved/restored by the host so copies and project reloads
// carry the correct value and do not get the Kodak 5219 override applied again.
#define PARAM_INITIALIZED     "_initialized"

// ---- Film Emulation mode ----
// Top-level mode switch: 0 = Commercial Looks, 1 = Film Emulation
#define PARAM_LOOK_TYPE       "look_type"
// Spectral Film negative stock selection (choice; choices populated at instance creation)
#define PARAM_SF_NEG_STOCK    "sf_neg_stock"
// Spectral Film print stock selection (choice; choices populated at instance creation)
#define PARAM_SF_PRT_STOCK    "sf_prt_stock"
// Film Emulation: White Point Kelvin slider (no dropdown; D65=6500K default)
#define PARAM_SF_WP_KELVIN    "sf_wp_kelvin"
// Film Emulation: Enable Look toggle (enables the spectral film LUT)
#define PARAM_SF_LOOK_ENABLE  "sf_look_enable"
// Film Emulation: Look Saturation blend 0..1 (chroma/hue contribution from the LUT)
#define PARAM_SF_LOOK_SATURATION "sf_look_saturation"
// Film Emulation: Contrast Curve blend 0..1 (luma/tone contribution from the LUT)
#define PARAM_SF_CONTRAST_CURVE  "sf_contrast_curve"
// Legacy: kept for backward-compat loading but hidden in UI
#define PARAM_SF_LOOK_INTENSITY "sf_look_intensity"

#define PLUGIN_ID             "com.lofifxfilm.grain"
#define PLUGIN_LABEL          "LoFi FX Film"
#define PLUGIN_GROUP          "LoFi FX"
#define PLUGIN_VERSION_MAJOR  1
#define PLUGIN_VERSION_MINOR  2

#endif // GRAIN_PARAMS_H
