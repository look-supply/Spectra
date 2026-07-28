// ============================================================
// Lofi FX Film — OFX Plugin for DaVinci Resolve
//
// Film texture and finishing pipeline with optional post-grain 3D Look.
//
// GPU rendering: Metal (macOS), CUDA (Linux/Windows).
// ============================================================

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <fstream>
#include <algorithm>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxParam.h"
#include "ofxProperty.h"

#include "GrainParams.h"
#include "CubeParser.h"
#include "LutPack.h"
#include "PresetParser.h"
#include "SpectralFilmData.h"

#ifdef __APPLE__
#include "MetalKernels.h"
#endif
#include "CudaKernels.h"

// ============================================================
// Suite pointers (fetched once at load time)
// ============================================================
static const OfxPropertySuiteV1*     gPropHost     = nullptr;
static const OfxImageEffectSuiteV1*  gEffectHost   = nullptr;
static const OfxParameterSuiteV1*    gParamHost    = nullptr;
static OfxHost*                      gHost         = nullptr;

// ============================================================
// Per-instance data
// ============================================================
struct InstanceData {
    std::vector<CubeLut> inputCsts;  // Discovered input CST LUTs
    std::vector<CubeLut> outputCsts; // Discovered output CST LUTs
    int     currentInputCstIdx;      // Currently uploaded input CST index (or -1)
    int     currentOutputCstIdx;     // Currently uploaded output CST index (or -1)
    void*   metalInputCstBuffer;     // Retained id<MTLBuffer> for input CST
    void*   cudaInputCstBuffer;      // CUDA device ptr for input CST
    void*   metalOutputCstBuffer;    // Retained id<MTLBuffer> for output CST
    void*   cudaOutputCstBuffer;     // CUDA device ptr for output CST

    std::vector<MasterPreset> presets;  // Discovered master presets (bundled then user)
    std::vector<int> choiceToPreset;   // Maps dropdown choice index → presets[] index (-1 = Custom/separator)
    bool applyingPreset;               // Guard against re-entrant detect

    // Embedded LUT loaded from a preset (bypasses built-in look system)
    CubeLut embeddedLut;
    bool    hasEmbeddedLut;
    bool    embeddedLutUploaded;
    void*   metalEmbeddedLutBuffer;
    void*   cudaEmbeddedLutBuffer;

    // ---- Film Emulation (Spectral Film) state ----
    int sfLutCacheNeg;              // neg choice index when sfLutBuffer was last built (-99 = invalid)
    int sfLutCachePrt;              // prt choice index when sfLutBuffer was last built
    std::vector<float> sfConvertedLut; // CPU-side DWG/DI-output converted LUT (33^3 * 3 floats)
    void* metalSfLutBuffer;         // GPU buffer for sfConvertedLut (id<MTLBuffer>)

    InstanceData()
        : currentInputCstIdx(-1)
        , currentOutputCstIdx(-1)
        , metalInputCstBuffer(nullptr)
        , cudaInputCstBuffer(nullptr)
        , metalOutputCstBuffer(nullptr)
        , cudaOutputCstBuffer(nullptr)
        , presets()
        , applyingPreset(false)
        , hasEmbeddedLut(false)
        , embeddedLutUploaded(false)
        , metalEmbeddedLutBuffer(nullptr)
        , cudaEmbeddedLutBuffer(nullptr)
        , sfLutCacheNeg(-99)
        , sfLutCachePrt(-99)
        , metalSfLutBuffer(nullptr)
    {}
};

// Global plugin bundle path (set once during kOfxActionLoad)
static std::string gBundlePath;

// ============================================================
// Spectral Film stock registry (loaded on demand from .bin)
// ============================================================
static std::vector<NegativeStockRecord> gSfNegStocks;
static std::vector<PrintStockRecord>    gSfPrtStocks;
static std::mutex                       gSfStockMutex;
using SfLutKey = std::pair<int,int>;
static std::map<SfLutKey, std::vector<float>> gSfFilmLuts;

static const char* kSfNegStockNames[] = {
    "Agfa Vista 100 (2001)",                    //  0
    "Fuji C200 (2017)",                         //  1
    "Fuji Eterna 500 (2007)",                   //  2
    "Fuji Eterna 500 Vivid (2007)",             //  3
    "Fuji Natura 1600 (2003)",                  //  4
    "Fuji Pro 160C (2004)",                     //  5
    "Fuji Pro 160S (2004)",                     //  6
    "Fuji Pro 400H (2004)",                     //  7
    "Fuji Superia Reala (1998)",                //  8
    "Fuji Superia X-Tra 400 (1998)",            //  9
    "Kodak Vision3 50D 5203 (2011)",            // 10
    "Kodak Verita 200D 5206 (2026)",            // 11
    "Kodak Vision3 250D 5207 (2009)",           // 12
    "Kodak Vision3 200T 5213 (2010)",           // 13
    "Kodak Vision3 500T 5219 (2007)",           // 14
    "Eastman Color Neg II 100T 5247 (1974)",    // 15
    "Eastman Color Neg II 100T 5247 (1976)",    // 16
    "Eastman Color Neg II 100T 5247 Alt (1976)",// 17
    "Eastman Color Neg 5248 (1952)",            // 18
    "Eastman EXR 100T 5248 (1989)",             // 19
    "Eastman Color Neg 50T 5250 (1959)",        // 20
    "Kodak Vision 320T 5277 (1996)",            // 21
    "Eastman EXR 200T 5293 (1992)",             // 22
    "Kodak Aerocolor IV 2460 (2009)",           // 23
    "Kodak Aerocolor IV 2460 Low (2009)",       // 24
    "Kodak Aerocolor IV 2460 High (2009)",      // 25
    "Kodak Ektar 100 (2008)",                   // 26
    "Kodak Gold 200 (2007)",                    // 27
    "Kodak Portra 160 (2011)",                  // 28
    "Kodak Portra 400 (2010)",                  // 29
    "Kodak Portra 800 (1998)",                  // 30
    "Kodak Portra 800 @1600 (1998)",            // 31
    "Kodak Portra 800 @3200 (1998)",            // 32
    "Kodak Ultramax 400 (2007)",                // 33
    "Kodak Vericolor III (1983)",               // 34
};

static const char* kSfPrintStockNames[] = {
    "Kodak Vision Color Print 2383 (1998)",         //  0
    "Fuji Eterna-CP Type 3513DI (2002)",            //  1
    "Fuji Eterna-CP 3523XD (2010)",                 //  2
    "Fuji Crystal Archive Pro PDII (2003)",         //  3
    "Fuji Crystal Archive DPII (2006)",             //  4
    "Fuji Crystal Archive Maxima (2006)",           //  5
    "Fuji Crystal Archive Super Type C (2007)",     //  6
    "Fujiflex Crystal Archive New Version",         //  7
    "Fujiflex Crystal Archive Old Version",         //  8
    "Kodak Vision Premier Color Print 2393 (1998)", //  9
    "Eastman Color Print 5381 (1972)",              // 10
    "Eastman Color SP Print 5383 (1974)",           // 11
    "Eastman Color Print LPP 5384 (1982)",          // 12
    "Kodak Duraflex Plus (2004)",                   // 13
    "Kodak Endura Premier Paper (2003)",            // 14
    "Eastman EXR Color Print 5386 (1993)",          // 15
    "Kodak Portra Endura Paper (1999)",             // 16
    "Kodak Supra Endura Paper (1999)",              // 17
};

struct SfChoiceEntry { int idx; const char* label; };

static const SfChoiceEntry kSfNegChoices[] = {
    { -1, "(none)" },
    { -1, "── Motion / Fujifilm ──" },
    {  2, nullptr },    // Fuji Eterna 500 (2007)
    {  3, nullptr },    // Fuji Eterna 500 Vivid (2007)
    { -1, "── Motion / Kodak ──" },
    { 11, nullptr },    // Kodak Verita 200D 5206 (2026)
    { 10, nullptr },    // Kodak Vision3 50D 5203 (2011)
    { 13, nullptr },    // Kodak Vision3 200T 5213 (2010)
    { 23, nullptr },    // Kodak Aerocolor IV 2460 (2009)
    { 25, nullptr },    // Kodak Aerocolor IV 2460 High (2009)
    { 24, nullptr },    // Kodak Aerocolor IV 2460 Low (2009)
    { 12, nullptr },    // Kodak Vision3 250D 5207 (2009)
    { 14, nullptr },    // Kodak Vision3 500T 5219 (2007)
    { 21, nullptr },    // Kodak Vision 320T 5277 (1996)
    { 22, nullptr },    // Eastman EXR 200T 5293 (1992)
    { 19, nullptr },    // Eastman EXR 100T 5248 (1989)
    { 17, nullptr },    // Eastman Color Neg II 100T 5247 Alt (1976)
    { 16, nullptr },    // Eastman Color Neg II 100T 5247 (1976)
    { 15, nullptr },    // Eastman Color Neg II 100T 5247 (1974)
    { 20, nullptr },    // Eastman Color Neg 50T 5250 (1959)
    { 18, nullptr },    // Eastman Color Neg 5248 (1952)
    { -1, "── Stills / Agfa ──" },
    {  0, nullptr },    // Agfa Vista 100 (2001)
    { -1, "── Stills / Fujifilm ──" },
    {  1, nullptr },    // Fuji C200 (2017)
    {  5, nullptr },    // Fuji Pro 160C (2004)
    {  6, nullptr },    // Fuji Pro 160S (2004)
    {  7, nullptr },    // Fuji Pro 400H (2004)
    {  4, nullptr },    // Fuji Natura 1600 (2003)
    {  8, nullptr },    // Fuji Superia Reala (1998)
    {  9, nullptr },    // Fuji Superia X-Tra 400 (1998)
    { -1, "── Stills / Kodak ──" },
    { 28, nullptr },    // Kodak Portra 160 (2011)
    { 29, nullptr },    // Kodak Portra 400 (2010)
    { 26, nullptr },    // Kodak Ektar 100 (2008)
    { 27, nullptr },    // Kodak Gold 200 (2007)
    { 33, nullptr },    // Kodak Ultramax 400 (2007)
    { 30, nullptr },    // Kodak Portra 800 (1998)
    { 31, nullptr },    // Kodak Portra 800 @1600 (1998)
    { 32, nullptr },    // Kodak Portra 800 @3200 (1998)
    { 34, nullptr },    // Kodak Vericolor III (1983)
};
static const int kSfNegChoicesN = (int)(sizeof(kSfNegChoices) / sizeof(kSfNegChoices[0]));

static const SfChoiceEntry kSfPrtChoices[] = {
    { -1, "(none)" },
    { -1, "── Motion / Fujifilm ──" },
    {  2, nullptr },    // Fuji Eterna-CP 3523XD (2010)
    {  1, nullptr },    // Fuji Eterna-CP Type 3513DI (2002)
    { -1, "── Motion / Kodak ──" },
    {  0, nullptr },    // Kodak Vision Color Print 2383 (1998)
    {  9, nullptr },    // Kodak Vision Premier Color Print 2393 (1998)
    { 15, nullptr },    // Eastman EXR Color Print 5386 (1993)
    { 12, nullptr },    // Eastman Color Print LPP 5384 (1982)
    { 11, nullptr },    // Eastman Color SP Print 5383 (1974)
    { 10, nullptr },    // Eastman Color Print 5381 (1972)
    { -1, "── Stills / Fujifilm ──" },
    {  6, nullptr },    // Fuji Crystal Archive Super Type C (2007)
    {  4, nullptr },    // Fuji Crystal Archive DPII (2006)
    {  5, nullptr },    // Fuji Crystal Archive Maxima (2006)
    {  3, nullptr },    // Fuji Crystal Archive Pro PDII (2003)
    {  7, nullptr },    // Fujiflex Crystal Archive New Version
    {  8, nullptr },    // Fujiflex Crystal Archive Old Version
    { -1, "── Stills / Kodak ──" },
    { 13, nullptr },    // Kodak Duraflex Plus (2004)
    { 14, nullptr },    // Kodak Endura Premier Paper (2003)
    { 16, nullptr },    // Kodak Portra Endura Paper (1999)
    { 17, nullptr },    // Kodak Supra Endura Paper (1999)
};
static const int kSfPrtChoicesN = (int)(sizeof(kSfPrtChoices) / sizeof(kSfPrtChoices[0]));

// ============================================================
// Load spectral film stock data from binary file
// ============================================================
static bool loadSfStockData(const std::string& bundlePath) {
    std::lock_guard<std::mutex> lock(gSfStockMutex);
    if (!gSfNegStocks.empty()) return true;
    gSfFilmLuts.clear();

    std::string path = bundlePath + "/Contents/Resources/spectral_film_stocks.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "LoFiFxFilm: cannot open %s\n", path.c_str());
        return false;
    }

    char magic[8];
    f.read(magic, 8);
    if (memcmp(magic, "SPFILM\0\0", 8) != 0) {
        fprintf(stderr, "LoFiFxFilm: bad magic in spectral_film_stocks.bin\n");
        return false;
    }

    uint32_t version = 0, nNeg = 0, nPrt = 0;
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&nNeg), 4);
    f.read(reinterpret_cast<char*>(&nPrt), 4);
    if (!f || version != 4 || nNeg > SF_MAX_STOCKS || nPrt > SF_MAX_STOCKS) {
        fprintf(stderr, "LoFiFxFilm: invalid spectral stock header\n");
        return false;
    }

    gSfNegStocks.resize(nNeg);
    for (uint32_t i = 0; i < nNeg; ++i) {
        uint32_t recSize = 0;
        f.read(reinterpret_cast<char*>(&recSize), 4);
        if (!f || recSize < sizeof(NegativeStockRecord)) { gSfNegStocks.clear(); gSfPrtStocks.clear(); return false; }
        f.read(reinterpret_cast<char*>(&gSfNegStocks[i]), (std::streamsize)sizeof(NegativeStockRecord));
        if (!f) { gSfNegStocks.clear(); gSfPrtStocks.clear(); return false; }
        gSfNegStocks[i].name[SF_NAME_LEN - 1] = '\0';
        long diff = (long)recSize - (long)sizeof(NegativeStockRecord);
        if (diff > 0) f.seekg(diff, std::ios::cur);
    }
    gSfPrtStocks.resize(nPrt);
    for (uint32_t i = 0; i < nPrt; ++i) {
        uint32_t recSize = 0;
        f.read(reinterpret_cast<char*>(&recSize), 4);
        if (!f || recSize < sizeof(PrintStockRecord)) { gSfNegStocks.clear(); gSfPrtStocks.clear(); return false; }
        f.read(reinterpret_cast<char*>(&gSfPrtStocks[i]), (std::streamsize)sizeof(PrintStockRecord));
        if (!f) { gSfNegStocks.clear(); gSfPrtStocks.clear(); return false; }
        gSfPrtStocks[i].name[SF_NAME_LEN - 1] = '\0';
        long diff = (long)recSize - (long)sizeof(PrintStockRecord);
        if (diff > 0) f.seekg(diff, std::ios::cur);
    }

    uint32_t nLuts = 0;
    if (f.read(reinterpret_cast<char*>(&nLuts), 4) && f) {
        const size_t kLutFloats = (size_t)SF_LUT3D_SIZE * SF_LUT3D_SIZE * SF_LUT3D_SIZE * 3;
        for (uint32_t i = 0; i < nLuts; ++i) {
            uint32_t negI = 0, prtI = 0;
            f.read(reinterpret_cast<char*>(&negI), 4);
            f.read(reinterpret_cast<char*>(&prtI), 4);
            if (!f) break;
            std::vector<float> lutData(kLutFloats);
            f.read(reinterpret_cast<char*>(lutData.data()), kLutFloats * sizeof(float));
            if (!f) break;
            int ni = (int)negI;
            int pi = (prtI == 0xFFFFFFFFu) ? -1 : (int)prtI;
            gSfFilmLuts[{ni, pi}] = std::move(lutData);
        }
    }

    fprintf(stderr, "LoFiFxFilm: loaded %u neg + %u print stocks + %zu spectral LUTs\n",
            nNeg, nPrt, gSfFilmLuts.size());
    return true;
}

// ============================================================
// Convert spectral film LUT (Rec.709 Gamma 2.4 output) to DWG/DI log output.
// The raw LUT data from the .bin maps DI log [0..1]^3 → Rec.709 Gamma 2.4.
// We convert each output triple to DWG/DI log so the result can be used
// directly as the "Look LUT" in the existing LoFi Film GPU kernel.
// Conversion mirrors step 6 in SpectralKernelCore.inc exactly.
// ============================================================
static void convertSfLutToDwgDi(const std::vector<float>& srcLut,
                                  std::vector<float>& outLut)
{
    outLut.resize(srcLut.size());

    // DI log constants
    constexpr float kDI_A       = 0.0075f;
    constexpr float kDI_B       = 7.0f;
    constexpr float kDI_C       = 0.07329248f;
    constexpr float kDI_M       = 10.44426855f;
    constexpr float kDI_LIN_CUT = 0.00262409f;

    // Inverse DaVinci tone-map constants (matches SpectralKernelCore.inc)
    constexpr float kTM_INPUT_WHITE  = 100.0f;
    constexpr float kTM_OUTPUT_WHITE = 1.0f;
    constexpr float kTM_ADAPTATION   = 9.0f;
    constexpr float kTM_SCENE_GAIN   = 1.0f;

    auto linearToDI = [&](float v) -> float {
        return (v > kDI_LIN_CUT) ? kDI_C * (std::log2(v + kDI_A) + kDI_B) : v * kDI_M;
    };

    auto rec709ToLinear = [](float v) -> float {
        return (v < 0.081f) ? (v / 4.5f)
                             : std::pow((v + 0.099f) / 1.099f, 1.0f / 0.45f);
    };

    auto invDaVinciToneMap = [&](float v) -> float {
        float b = (kTM_INPUT_WHITE - (kTM_ADAPTATION / 100.0f) * (kTM_INPUT_WHITE / kTM_OUTPUT_WHITE))
                / ((kTM_INPUT_WHITE / kTM_OUTPUT_WHITE) - 1.0f);
        float a = kTM_OUTPUT_WHITE / (kTM_INPUT_WHITE / (kTM_INPUT_WHITE + b));
        float y = std::min(std::max(v, 0.0f), kTM_OUTPUT_WHITE * 0.9999f);
        return (b * y / std::max(a - y, 1.0e-6f)) * kTM_SCENE_GAIN;
    };

    for (size_t i = 0; i + 2 < srcLut.size(); i += 3) {
        // Source values are Rec.709 Gamma 2.4
        float r = srcLut[i], g = srcLut[i + 1], b = srcLut[i + 2];

        // Gamma decode: Rec.709 Gamma 2.4 → Rec.709 linear
        r = rec709ToLinear(std::min(std::max(r, 0.0f), 1.0f));
        g = rec709ToLinear(std::min(std::max(g, 0.0f), 1.0f));
        b = rec709ToLinear(std::min(std::max(b, 0.0f), 1.0f));

        // Restore scene-referred headroom (inverse DaVinci tone map)
        r = invDaVinciToneMap(r);
        g = invDaVinciToneMap(g);
        b = invDaVinciToneMap(b);

        // Rec.709 linear → CIE XYZ D65
        float X = 0.4124564f * r + 0.3575761f * g + 0.1804375f * b;
        float Y = 0.2126729f * r + 0.7151522f * g + 0.0721750f * b;
        float Z = 0.0193339f * r + 0.1191920f * g + 0.9503041f * b;

        // CIE XYZ → DWG linear
        float dr = 1.5165889263f * X + (-0.2814626098f) * Y + (-0.1469555795f) * Z;
        float dg = (-0.4649386704f) * X + 1.2514817715f * Y + 0.1748927236f * Z;
        float db = 0.0648602322f * X + 0.1091581732f * Y + 0.7615460157f * Z;

        // DWG linear → DI log
        outLut[i]     = linearToDI(std::max(dr, 0.0f));
        outLut[i + 1] = linearToDI(std::max(dg, 0.0f));
        outLut[i + 2] = linearToDI(std::max(db, 0.0f));
    }
}

// ============================================================
// Utility: get the bundle path from the plugin's property set
// ============================================================
static std::string getBundlePath(OfxImageEffectHandle effect) {
    OfxPropertySetHandle effectProps;
    gEffectHost->getPropertySet(effect, &effectProps);
    char* path = nullptr;
    gPropHost->propGetString(effectProps, kOfxPluginPropFilePath, 0, &path);
    return path ? std::string(path) : std::string();
}

// ============================================================
// Utility: set parameter visibility
// ============================================================
static void setParamSecret(OfxImageEffectHandle effect,
                           const char* paramName, bool secret)
{
    OfxParamSetHandle paramSet;
    if (gEffectHost->getParamSet(effect, &paramSet) != kOfxStatOK || !paramSet) return;
    OfxParamHandle param = nullptr;
    if (gParamHost->paramGetHandle(paramSet, paramName, &param, nullptr) != kOfxStatOK || !param) return;
    OfxPropertySetHandle props = nullptr;
    if (gParamHost->paramGetPropertySet(param, &props) != kOfxStatOK || !props) return;
    gPropHost->propSetInt(props, kOfxParamPropSecret, 0, secret ? 1 : 0);
}

// ============================================================
// Utility: set parameter enabled/disabled
// ============================================================
static void setParamEnabled(OfxImageEffectHandle effect,
                            const char* paramName, bool enabled)
{
    OfxParamSetHandle paramSet;
    if (gEffectHost->getParamSet(effect, &paramSet) != kOfxStatOK || !paramSet) return;
    OfxParamHandle param = nullptr;
    if (gParamHost->paramGetHandle(paramSet, paramName, &param, nullptr) != kOfxStatOK || !param) return;
    OfxPropertySetHandle props = nullptr;
    if (gParamHost->paramGetPropertySet(param, &props) != kOfxStatOK || !props) return;
    gPropHost->propSetInt(props, kOfxParamPropEnabled, 0, enabled ? 1 : 0);
}

static bool envEnabled(const char* name)
{
    const char* value = std::getenv(name);
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool timelineDiagEnabled()
{
    return envEnabled("LOFI_FX_TIMELINE_DIAG") || envEnabled("LOFI_FX_PROOF_WRITE");
}

static bool proofWriteEnabled()
{
    return envEnabled("LOFI_FX_PROOF_WRITE");
}

// ============================================================
// Utility: read param values at given time
// ============================================================
static double getDouble(OfxParamSetHandle ps, const char* name, double time) {
    OfxParamHandle p = nullptr; double v = 0;
    if (gParamHost->paramGetHandle(ps, name, &p, nullptr) != kOfxStatOK || !p) return v;
    gParamHost->paramGetValueAtTime(p, time, &v);
    return v;
}
static int getInt(OfxParamSetHandle ps, const char* name) {
    OfxParamHandle p = nullptr; int v = 0;
    if (gParamHost->paramGetHandle(ps, name, &p, nullptr) != kOfxStatOK || !p) return v;
    gParamHost->paramGetValue(p, &v);
    return v;
}
static bool getBool(OfxParamSetHandle ps, const char* name) {
    return getInt(ps, name) != 0;
}
static void getRGB(OfxParamSetHandle ps, const char* name, double time,
                   double& r, double& g, double& b) {
    r = 1.0; g = 1.0; b = 1.0;
    OfxParamHandle p = nullptr;
    if (gParamHost->paramGetHandle(ps, name, &p, nullptr) != kOfxStatOK || !p) return;
    gParamHost->paramGetValueAtTime(p, time, &r, &g, &b);
}
static void setDoubleParam(OfxParamSetHandle ps, const char* name, double value) {
    OfxParamHandle p = nullptr;
    if (gParamHost->paramGetHandle(ps, name, &p, nullptr) != kOfxStatOK || !p) return;
    gParamHost->paramSetValue(p, value);
}
static void setRGBParam(OfxParamSetHandle ps, const char* name, double r, double g, double b) {
    OfxParamHandle p = nullptr;
    if (gParamHost->paramGetHandle(ps, name, &p, nullptr) != kOfxStatOK || !p) return;
    gParamHost->paramSetValue(p, r, g, b);
}
static void setIntParam(OfxParamSetHandle ps, const char* name, int value) {
    OfxParamHandle p = nullptr;
    if (gParamHost->paramGetHandle(ps, name, &p, nullptr) != kOfxStatOK || !p) return;
    gParamHost->paramSetValue(p, value);
}
static std::string getString(OfxParamSetHandle ps, const char* name) {
    OfxParamHandle p = nullptr;
    if (gParamHost->paramGetHandle(ps, name, &p, nullptr) != kOfxStatOK || !p) return {};
    char* v = nullptr;
    gParamHost->paramGetValue(p, &v);
    return v ? std::string(v) : std::string();
}

// ============================================================
// Film size presets — grain_size and grain_blur per format
// ============================================================
struct FilmSizePreset {
    double grain_size;
    double grain_blur;
};

static const FilmSizePreset kPresets[] = {
    { 0.55, 0.25 }, // FMT_65MM  — Fine 65mm
    { 1.0,  0.65 }, // FMT_35MM  — Medium 35mm
    { 2.8,  1.10 }, // FMT_16MM  — Coarse 16mm
    { 5.4,  1.85 }, // FMT_8MM   — Very Coarse 8mm
};
static const int kNumPresets = 4;

static float linkedFilmBlurSigma(float grainSize, float link) {
    // Film blur tracks grain size only — grain_blur softens crystal footprints
    // but does not reduce the underlying image's resolving power.
    float linkedSigma = 0.34f * grainSize + 0.090f * grainSize * grainSize;
    return link * std::min(linkedSigma, 12.0f);
}

// ============================================================
// Diffusion profile presets.
// One entry per DIFFUSION_PROFILE_* index (see GrainParams.h).
// DIFFUSION_PROFILE_CUSTOM is last and intentionally has no values.
// ============================================================
struct DiffusionProfilePreset {
    double overallStrength;
    double overallSize;
    double softness;
    double bodyStrength;
    double falloffStrength;
    double highlightGate;
    double veilLift;
    double contrastReduction;
    double highlightBoost;
    double tintStrength;
    double tintR;
    double tintG;
    double tintB;
};

static const DiffusionProfilePreset kDiffusionProfilePresets[] = {
    // overallStrength, overallSize, softness, bodyStrength, falloffStrength, highlightGate, veilLift, contrastReduction, highlightBoost, tintStrength, tintRGB
    { 0.075, 64.0, 1.0, 1.0, 1.0,  0.38, 0.000, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 }, // BPM 1/8
    { 0.150, 64.0, 1.0, 1.0, 1.0,  0.38, 0.000, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 }, // BPM 1/4
    { 0.263, 64.0, 1.0, 1.0, 1.0,  0.38, 0.000, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 }, // BPM 1/2
};
static const int kNumDiffusionProfilePresets =
    (int)(sizeof(kDiffusionProfilePresets) / sizeof(kDiffusionProfilePresets[0]));

// ============================================================
// Instance data helpers
// ============================================================
static InstanceData* getInstanceData(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;
    gEffectHost->getPropertySet(effect, &props);
    void* ptr = nullptr;
    gPropHost->propGetPointer(props, kOfxPropInstanceData, 0, &ptr);
    return static_cast<InstanceData*>(ptr);
}

static void setInstanceData(OfxImageEffectHandle effect, InstanceData* data) {
    OfxPropertySetHandle props;
    gEffectHost->getPropertySet(effect, &props);
    gPropHost->propSetPointer(props, kOfxPropInstanceData, 0, data);
}

// ============================================================
// ACTION: Load — fetch suites
// ============================================================
static OfxStatus actionLoad() {
    gPropHost   = (const OfxPropertySuiteV1*)   gHost->fetchSuite(gHost->host,
                    kOfxPropertySuite, 1);
    gEffectHost = (const OfxImageEffectSuiteV1*) gHost->fetchSuite(gHost->host,
                    kOfxImageEffectSuite, 1);
    gParamHost  = (const OfxParameterSuiteV1*)   gHost->fetchSuite(gHost->host,
                    kOfxParameterSuite, 1);

    if (!gPropHost || !gEffectHost || !gParamHost)
        return kOfxStatErrMissingHostFeature;

    return kOfxStatOK;
}

// ============================================================
// ACTION: Describe — plugin-level properties
// ============================================================
static OfxStatus actionDescribe(OfxImageEffectHandle effect) {
    OfxPropertySetHandle effectProps;
    gEffectHost->getPropertySet(effect, &effectProps);

    gPropHost->propSetString(effectProps, kOfxPropLabel, 0, PLUGIN_LABEL);
    gPropHost->propSetString(effectProps, kOfxImageEffectPluginPropGrouping, 0,
                             PLUGIN_GROUP);

    // Contexts: filter only (single-input effect)
    gPropHost->propSetString(effectProps, kOfxImageEffectPropSupportedContexts,
                             0, kOfxImageEffectContextFilter);

    // Bit depths: float only
    gPropHost->propSetString(effectProps,
                             kOfxImageEffectPropSupportedPixelDepths,
                             0, kOfxBitDepthFloat);

    // GPU render support — "true" means use Metal when available; Resolve can
    // still call render with metalEnabled=0 (e.g. timeline adjustment nodes),
    // in which case the self-hosted Metal fallback handles rendering.
#ifdef __APPLE__
    gPropHost->propSetString(effectProps,
                             kOfxImageEffectPropMetalRenderSupported, 0, "true");
#else
    gPropHost->propSetString(effectProps,
                             kOfxImageEffectPropCudaRenderSupported, 0, "true");
#endif

    // Frame varying (grain changes every frame)
    gPropHost->propSetInt(effectProps, kOfxImageEffectFrameVarying, 0, 1);

    // Single render call (not tiled)
    gPropHost->propSetInt(effectProps,
                          kOfxImageEffectPropSupportsTiles, 0, 0);

    return kOfxStatOK;
}

// ============================================================
// ACTION: DescribeInContext — clips + parameters
// ============================================================
static OfxStatus actionDescribeInContext(OfxImageEffectHandle effect,
                                        OfxPropertySetHandle inArgs)
{
    // ---- Clips ----
    OfxPropertySetHandle clipProps;
    gEffectHost->clipDefine(effect, kOfxImageEffectSimpleSourceClipName,
                            &clipProps);
    gPropHost->propSetString(clipProps, kOfxImageEffectPropSupportedComponents,
                             0, kOfxImageComponentRGBA);

    gEffectHost->clipDefine(effect, kOfxImageEffectOutputClipName, &clipProps);
    gPropHost->propSetString(clipProps, kOfxImageEffectPropSupportedComponents,
                             0, kOfxImageComponentRGBA);

    // ---- Parameters ----
    OfxParamSetHandle paramSet;
    gEffectHost->getParamSet(effect, &paramSet);
    OfxPropertySetHandle pp;

    // ---------- Initialisation sentinel (hidden) ----------
    // 0 = brand-new instance (apply factory defaults in createInstance).
    // 1 = copy / project-reload (host restores saved values; skip defaults).
    gParamHost->paramDefine(paramSet, kOfxParamTypeInteger, PARAM_INITIALIZED, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "_initialized");
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 0);
    gPropHost->propSetInt(pp, kOfxParamPropSecret, 0, 1);
    gPropHost->propSetInt(pp, kOfxParamPropAnimates, 0, 0);

    // ---------- Master Preset (top-level, before all groups) ----------
    gParamHost->paramDefine(paramSet, kOfxParamTypeChoice, PARAM_MASTER_PRESET, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Preset");
    gPropHost->propSetString(pp, kOfxParamPropChoiceOption, 0, "Custom");
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 0);

    // ---------- Input / Output Color Space (top-level) ----------
    gParamHost->paramDefine(paramSet, kOfxParamTypeChoice, PARAM_INPUT_TRANSFORM, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Input Color Space");
    gPropHost->propSetString(pp, kOfxParamPropChoiceOption, 0, "DaVinci WG / Intermediate");
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 0);

    gParamHost->paramDefine(paramSet, kOfxParamTypeChoice, PARAM_OUTPUT_TRANSFORM, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Output Color Space");
    gPropHost->propSetString(pp, kOfxParamPropChoiceOption, 0, "DaVinci WG / Intermediate");
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 0);

    // ---------- Show Contrast Curve (hidden for now) ----------
    gParamHost->paramDefine(paramSet, kOfxParamTypeBoolean, PARAM_CURVE_ENABLE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Show Contrast Curve");
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 0);
    gPropHost->propSetInt(pp, kOfxParamPropSecret, 0, 1);

    // ---------- GROUP: Film Look (Spectral Film stocks) ----------
    gParamHost->paramDefine(paramSet, kOfxParamTypeGroup, "grp_film_emul", &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Film Emulation");
    gPropHost->propSetInt(pp, kOfxParamPropGroupOpen, 0, 1);

    gParamHost->paramDefine(paramSet, kOfxParamTypeBoolean, PARAM_SF_LOOK_ENABLE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Enable Look");
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 1);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_film_emul");

    gParamHost->paramDefine(paramSet, kOfxParamTypeChoice, PARAM_SF_NEG_STOCK, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Film Negative");
    // Populate choices from kSfNegChoices
    for (int ci = 0; ci < kSfNegChoicesN; ++ci) {
        const char* label = kSfNegChoices[ci].label
            ? kSfNegChoices[ci].label
            : (kSfNegChoices[ci].idx >= 0 ? kSfNegStockNames[kSfNegChoices[ci].idx] : "(none)");
        gPropHost->propSetString(pp, kOfxParamPropChoiceOption, ci, label);
    }
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_film_emul");

    gParamHost->paramDefine(paramSet, kOfxParamTypeChoice, PARAM_SF_PRT_STOCK, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Film Print");
    // Populate choices from kSfPrtChoices
    for (int ci = 0; ci < kSfPrtChoicesN; ++ci) {
        const char* label = kSfPrtChoices[ci].label
            ? kSfPrtChoices[ci].label
            : (kSfPrtChoices[ci].idx >= 0 ? kSfPrintStockNames[kSfPrtChoices[ci].idx] : "(none)");
        gPropHost->propSetString(pp, kOfxParamPropChoiceOption, ci, label);
    }
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_film_emul");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SF_WP_KELVIN, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "White Point (K)");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 6500.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 2000.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 20000.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 2000.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 12000.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_film_emul");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SF_LOOK_INTENSITY, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Look Intensity");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_film_emul");
    gPropHost->propSetInt(pp, kOfxParamPropSecret, 0, 1); // hidden — superseded by saturation + contrast curve

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SF_LOOK_SATURATION, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Look Saturation");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_film_emul");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SF_CONTRAST_CURVE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Contrast Curve");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_film_emul");

    // ---------- GROUP: Split Toning ----------
    gParamHost->paramDefine(paramSet, kOfxParamTypeGroup, "grp_split", &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Split Toning");
    gPropHost->propSetInt(pp, kOfxParamPropGroupOpen, 0, 0);

    gParamHost->paramDefine(paramSet, kOfxParamTypeBoolean, PARAM_SPLIT_ENABLE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Enable Split Toning");
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_split");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SPLIT_STRENGTH, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Amount (Saturation)");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 4.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 4.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_split");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SPLIT_SHADOW_HUE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Shadow Hue");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 172.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 360.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 360.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_split");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SPLIT_HI_HUE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Highlight Hue");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 340.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 360.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 360.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_split");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SPLIT_LO_VS_HI, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Lo vs Hi Strength");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.5);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_split");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SPLIT_PIVOT, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Pivot");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.336);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_split");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SPLIT_NEUTRAL_W, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Neutral Segment Width");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.08);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 0.5);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 0.5);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_split");

    // ---------- Split Toning: Advanced sub-group ----------
    gParamHost->paramDefine(paramSet, kOfxParamTypeGroup, "grp_split_adv", &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Advanced");
    gPropHost->propSetInt(pp, kOfxParamPropGroupOpen, 0, 0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_split");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SPLIT_SUB_LO, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Shadow Density");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_split_adv");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SPLIT_SUB_HI, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Highlight Density");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_split_adv");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SPLIT_FILM_DENSITY, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Film Density");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.2);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 3.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.2);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 3.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_split_adv");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SPLIT_KM_STRENGTH, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "K-M Hue Drift");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_split_adv");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SPLIT_HI_PROTECT, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Highlight Protect");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_split_adv");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SPLIT_DYE_SAT, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Dye Saturation");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.1);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 2.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.1);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 2.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_split_adv");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SPLIT_CROSS_BLEED, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Cross-Zone Bleed");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 0.5);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 0.5);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_split_adv");

    // ---------- GROUP: Printer Points ----------
    // 1 printer point = 1/12 stop (~0.025 log exposure). Neutral = 25.
    // Range -11..61 gives the conventional ±3 stops of timing latitude.
    gParamHost->paramDefine(paramSet, kOfxParamTypeGroup, "grp_print", &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Printer Points");
    gPropHost->propSetInt(pp, kOfxParamPropGroupOpen, 0, 0);

    gParamHost->paramDefine(paramSet, kOfxParamTypeBoolean, PARAM_PRINT_ENABLE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Enable Printer Points");
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_print");

    gParamHost->paramDefine(paramSet, kOfxParamTypeBoolean, PARAM_PRINT_GANG, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Gang Printer Points");
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_print");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_PRINT_R, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Red Printer Points");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 25.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, -11.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 61.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, -11.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 61.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_print");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_PRINT_G, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Green Printer Points");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 25.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, -11.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 61.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, -11.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 61.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_print");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_PRINT_B, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Blue Printer Points");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 25.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, -11.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 61.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, -11.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 61.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_print");

    // ---------- GROUP: Grain ----------
    gParamHost->paramDefine(paramSet, kOfxParamTypeGroup, "grp_grain", &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Grain");
    gPropHost->propSetInt(pp, kOfxParamPropGroupOpen, 0, 0);

    gParamHost->paramDefine(paramSet, kOfxParamTypeBoolean, PARAM_GRAIN_ENABLE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Enable Grain");
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 1);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_grain");

    gParamHost->paramDefine(paramSet, kOfxParamTypeChoice, PARAM_FILM_FORMAT, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Film Size");
    gPropHost->propSetString(pp, kOfxParamPropChoiceOption, 0, "Fine 65mm");
    gPropHost->propSetString(pp, kOfxParamPropChoiceOption, 1, "Medium 35mm");
    gPropHost->propSetString(pp, kOfxParamPropChoiceOption, 2, "Coarse 16mm");
    gPropHost->propSetString(pp, kOfxParamPropChoiceOption, 3, "Very Coarse 8mm");
    gPropHost->propSetString(pp, kOfxParamPropChoiceOption, 4, "Custom");
    gPropHost->propSetInt(pp,    kOfxParamPropDefault, 0, FMT_35MM);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_grain");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_GRAIN_OPACITY, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Grain Opacity");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.300);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_grain");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_GRAIN_SAT, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Grain Saturation");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.5);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 2.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 2.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_grain");

    // Shape Irregularity — varies crystal size and local density. The anatomy
    // field uses a frame-invariant seed so the emulsion structure is stable
    // across frames while the developed grains themselves re-roll.
    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_SHAPE_IRREG, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Shape Irregularity");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.5);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_grain");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_GRAIN_CONTRAST, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Grain Contrast");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_grain");

    gParamHost->paramDefine(paramSet, kOfxParamTypeBoolean, PARAM_IMAGE_BLUR_ENABLE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Enable Film Blur");
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_grain");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_IMAGE_BLUR, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Film Blur Base");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 10.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 10.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_grain");
    gPropHost->propSetInt(pp, kOfxParamPropSecret, 0, 1);  // superseded by Film Blur (grain-linked)

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_GRAIN_BLUR_LINK, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Film Blur");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.63);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_grain");

    // ---------- Grain: Advanced sub-group ----------
    gParamHost->paramDefine(paramSet, kOfxParamTypeGroup, "grp_grain_adv", &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Advanced");
    gPropHost->propSetInt(pp, kOfxParamPropGroupOpen, 0, 0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_grain");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_GRAIN_SIZE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Grain Size");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.5);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 10.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.5);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 10.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_grain_adv");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_GRAIN_STRENGTH, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Grain Strength");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.3);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_grain_adv");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_GRAIN_BLUR, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Grain Blur");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.65);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 5.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 5.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_grain_adv");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_HI_ROLLOFF, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Highlight Rolloff");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.650);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_grain_adv");

    gParamHost->paramDefine(paramSet, kOfxParamTypeInteger, PARAM_SEED_OFFSET, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Seed Variant");
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 42);
    gPropHost->propSetInt(pp, kOfxParamPropMin, 0, 0);
    gPropHost->propSetInt(pp, kOfxParamPropMax, 0, 999999);
    gPropHost->propSetInt(pp, kOfxParamPropDisplayMin, 0, 0);
    gPropHost->propSetInt(pp, kOfxParamPropDisplayMax, 0, 999999);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_grain_adv");

    // ---------- GROUP: Halation ----------
    gParamHost->paramDefine(paramSet, kOfxParamTypeGroup, "grp_halation", &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Halation");
    gPropHost->propSetInt(pp, kOfxParamPropGroupOpen, 0, 0);

    gParamHost->paramDefine(paramSet, kOfxParamTypeBoolean, PARAM_HAL_ENABLE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Enable Halation");
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_halation");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_HAL_SIZE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Size");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 5.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 5.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_halation");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_HAL_COLOR, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Color");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.15);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 0.5);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 0.5);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_halation");

    {
        static const char* methodChoices[] = { "Proosa", "Minuth", nullptr };
        gParamHost->paramDefine(paramSet, kOfxParamTypeChoice, PARAM_HAL_METHOD, &pp);
        gPropHost->propSetString(pp, kOfxPropLabel, 0, "Method");
        gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 0);
        for (int ci = 0; methodChoices[ci]; ++ci)
            gPropHost->propSetString(pp, kOfxParamPropChoiceOption, ci, methodChoices[ci]);
        gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_halation");
    }

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_HAL_OPACITY, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Opacity");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_halation");

    // ---------- GROUP: Diffusion ----------
    gParamHost->paramDefine(paramSet, kOfxParamTypeGroup, "grp_diffusion", &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Diffusion");
    gPropHost->propSetInt(pp, kOfxParamPropGroupOpen, 0, 0);

    gParamHost->paramDefine(paramSet, kOfxParamTypeBoolean, PARAM_DIFFUSION_ENABLE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Enable Diffusion");
    gPropHost->propSetInt(pp, kOfxParamPropDefault, 0, 0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_diffusion");

    gParamHost->paramDefine(paramSet, kOfxParamTypeChoice, PARAM_DIFFUSION_PROFILE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Diffusion Profile");
    gPropHost->propSetString(pp, kOfxParamPropChoiceOption, 0, "Mist 1/8");
    gPropHost->propSetString(pp, kOfxParamPropChoiceOption, 1, "Mist 1/4");
    gPropHost->propSetString(pp, kOfxParamPropChoiceOption, 2, "Mist 1/2");
    gPropHost->propSetString(pp, kOfxParamPropChoiceOption, 3, "Custom");
    gPropHost->propSetInt(pp,    kOfxParamPropDefault, 0, DIFFUSION_PROFILE_BPM_18);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_diffusion");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_DIFFUSION_AMOUNT, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Overall Strength");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.075);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 2.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_diffusion");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_DIFFUSION_SIZE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Overall Size px");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 64.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.25);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 192.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.25);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 160.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_diffusion");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_DIFFUSION_SOFTNESS, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Softness");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.25);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 2.5);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.5);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 2.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_diffusion");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_DIFFUSION_BODY_STRENGTH, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Body Strength");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.05);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_diffusion");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_DIFFUSION_FALLOFF_STRENGTH, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Falloff Strength");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.0643);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_diffusion");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_DIFFUSION_HIGHLIGHT_GATE, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Highlight Gate");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.15);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 1.0);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_diffusion");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_DIFFUSION_VEIL, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Veil Lift");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 0.35);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 0.15);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_diffusion");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_DIFFUSION_CONTRAST, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Contrast Reduction");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 0.5);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_diffusion");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_DIFFUSION_HIGHLIGHT_BOOST, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Highlight Bloom");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.3315);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 0.5);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_diffusion");

    gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, PARAM_DIFFUSION_TINT_STRENGTH, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Tint Amount");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 0.3511);
    gPropHost->propSetDouble(pp, kOfxParamPropMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropMax, 0, 1.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMin, 0, 0.0);
    gPropHost->propSetDouble(pp, kOfxParamPropDisplayMax, 0, 0.5);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_diffusion");

    gParamHost->paramDefine(paramSet, kOfxParamTypeRGB, PARAM_DIFFUSION_TINT_COLOR, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Tint Color");
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 0, 1.3979);
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 1, 0.6993);
    gPropHost->propSetDouble(pp, kOfxParamPropDefault, 2, 0.4314);
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_diffusion");

    // ---------- GROUP: Save Preset ----------
    gParamHost->paramDefine(paramSet, kOfxParamTypeGroup, "grp_save_preset", &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Save Preset");
    gPropHost->propSetInt(pp, kOfxParamPropGroupOpen, 0, 0);

    gParamHost->paramDefine(paramSet, kOfxParamTypeString, PARAM_PRESET_NAME, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Preset Name");
    gPropHost->propSetString(pp, kOfxParamPropDefault, 0, "My Preset");
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_save_preset");

    gParamHost->paramDefine(paramSet, kOfxParamTypePushButton, PARAM_SAVE_PRESET, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Save Preset");
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_save_preset");

    gParamHost->paramDefine(paramSet, kOfxParamTypePushButton, PARAM_REFRESH_PRESETS, &pp);
    gPropHost->propSetString(pp, kOfxPropLabel, 0, "Refresh Presets");
    gPropHost->propSetString(pp, kOfxParamPropParent, 0, "grp_save_preset");

    return kOfxStatOK;
}

// ============================================================
// Utility: set a group param's open/collapsed state at runtime
// ============================================================
static void setGroupOpen(OfxImageEffectHandle effect,
                         const char* groupName, bool open) {
    OfxParamSetHandle ps;
    gEffectHost->getParamSet(effect, &ps);
    OfxParamHandle grp;
    if (gParamHost->paramGetHandle(ps, groupName, &grp, nullptr) == kOfxStatOK) {
        OfxPropertySetHandle props;
        gParamHost->paramGetPropertySet(grp, &props);
        gPropHost->propSetInt(props, kOfxParamPropGroupOpen, 0, open ? 1 : 0);
    }
}

// ============================================================
// Utility: update grain controls enabled state
// ============================================================
static void updateGrainEnabled(OfxImageEffectHandle effect) {
    OfxParamSetHandle ps;
    gEffectHost->getParamSet(effect, &ps);
    bool en = getBool(ps, PARAM_GRAIN_ENABLE);
    setParamEnabled(effect, PARAM_FILM_FORMAT,        en);
    setParamEnabled(effect, PARAM_GRAIN_SIZE,         en);
    setParamEnabled(effect, PARAM_GRAIN_STRENGTH,     en);
    setParamEnabled(effect, PARAM_GRAIN_BLUR,         en);
    setParamEnabled(effect, PARAM_GRAIN_OPACITY,      en);
    setParamEnabled(effect, PARAM_GRAIN_SAT,          en);
    setParamEnabled(effect, PARAM_SHAPE_IRREG,        en);
    setParamEnabled(effect, PARAM_GRAIN_CONTRAST,     en);
    setParamEnabled(effect, PARAM_HI_ROLLOFF,         en);
    setParamEnabled(effect, PARAM_SEED_OFFSET,        en);
    setParamEnabled(effect, PARAM_IMAGE_BLUR_ENABLE,  en);
    bool filmBlurEn = en && getBool(ps, PARAM_IMAGE_BLUR_ENABLE);
    setParamEnabled(effect, PARAM_GRAIN_BLUR_LINK, filmBlurEn);
    if (en) setGroupOpen(effect, "grp_grain", true);
}

// ============================================================
// Bradford chromatic adaptation — full cone-space CAT.
// Computes a 3×3 matrix (row-major, DaVinci Wide Gamut linear →
// DaVinci Wide Gamut linear) adapting from illuminant srcWP to dstWP.
// Indices: 0=D50, 1=D55, 2=D60, 3=D65, 4=D70, 5=D75.
// Returns false (identity) when srcWP == dstWP.
// ============================================================
static int normalizeWpChoice(int value)
{
    if (value >= 0 && value <= 5) return value;

    // Compatibility for projects saved while wp_adapt briefly used Kelvin.
    if (value >= 4500) {
        if (value < 5250) return 0;
        if (value < 5750) return 1;
        if (value < 6250) return 2;
        if (value < 6750) return 3;
        if (value < 7250) return 4;
        return 5;
    }

    return value < 0 ? 0 : 5;
}

static const double kWpChoiceKelvin[6] = { 5000.0, 5500.0, 6000.0, 6500.0, 7000.0, 7500.0 };

static bool paramIsType(OfxParamSetHandle ps, const char* name, const char* typeName)
{
    OfxParamHandle p = nullptr;
    if (gParamHost->paramGetHandle(ps, name, &p, nullptr) != kOfxStatOK || !p) return false;
    OfxPropertySetHandle props = nullptr;
    if (gParamHost->paramGetPropertySet(p, &props) != kOfxStatOK || !props) return false;
    char* type = nullptr;
    if (gPropHost->propGetString(props, kOfxParamPropType, 0, &type) != kOfxStatOK || !type) return false;
    return strcmp(type, typeName) == 0;
}

static int getWpChoiceParam(OfxParamSetHandle ps)
{
    OfxParamHandle p = nullptr;
    if (gParamHost->paramGetHandle(ps, PARAM_WP_ADAPT, &p, nullptr) != kOfxStatOK || !p) return 3;

    if (paramIsType(ps, PARAM_WP_ADAPT, kOfxParamTypeDouble)) {
        double v = 6500.0;
        if (gParamHost->paramGetValue(p, &v) != kOfxStatOK) return 3;
        return normalizeWpChoice((int)std::round(v));
    }

    int v = 3;
    if (gParamHost->paramGetValue(p, &v) != kOfxStatOK) return 3;
    return normalizeWpChoice(v);
}

static void setWpChoiceParam(OfxParamSetHandle ps, int value)
{
    value = normalizeWpChoice(value);
    OfxParamHandle p = nullptr;
    if (gParamHost->paramGetHandle(ps, PARAM_WP_ADAPT, &p, nullptr) != kOfxStatOK || !p) return;

    if (paramIsType(ps, PARAM_WP_ADAPT, kOfxParamTypeDouble)) {
        gParamHost->paramSetValue(p, kWpChoiceKelvin[value]);
    } else {
        gParamHost->paramSetValue(p, value);
    }
}

static double normalizeWpKelvin(double value)
{
    if (!std::isfinite(value)) return 6500.0;

    if (value <= 10.0) {
        int choice = normalizeWpChoice((int)std::round(value));
        return kWpChoiceKelvin[choice];
    }

    if (value < 2000.0) value = 2000.0;
    if (value > 20000.0) value = 20000.0;
    return value;
}

static double getWpKelvinParam(OfxParamSetHandle ps)
{
    OfxParamHandle p = nullptr;
    if (gParamHost->paramGetHandle(ps, PARAM_WP_KELVIN, &p, nullptr) != kOfxStatOK || !p)
        return kWpChoiceKelvin[getWpChoiceParam(ps)];

    double value = 6500.0;
    if (gParamHost->paramGetValue(p, &value) != kOfxStatOK)
        return kWpChoiceKelvin[getWpChoiceParam(ps)];
    return normalizeWpKelvin(value);
}

static void setWpKelvinParam(OfxParamSetHandle ps, double value)
{
    value = normalizeWpKelvin(value);
    OfxParamHandle p = nullptr;
    if (gParamHost->paramGetHandle(ps, PARAM_WP_KELVIN, &p, nullptr) != kOfxStatOK || !p) return;
    gParamHost->paramSetValue(p, value);
}

static bool bradfordCat(int srcWP, int dstWP, float out[9])
{
    srcWP = normalizeWpChoice(srcWP);
    dstWP = normalizeWpChoice(dstWP);

    if (srcWP == dstWP) { return false; }

    // Bradford "sharpened" cone matrix (XYZ → cone)
    static const double MB[3][3] = {
        { 0.8951,  0.2664, -0.1614},
        {-0.7502,  1.7135,  0.0367},
        { 0.0389, -0.0685,  1.0296}
    };
    // Inverse (cone → XYZ)
    static const double MB_INV[3][3] = {
        { 0.9869929, -0.1470543,  0.1599627},
        { 0.4323053,  0.5183603,  0.0492912},
        {-0.0085287,  0.0400428,  0.9684867}
    };
    // DaVinci Wide Gamut linear → XYZ (D65 white point)
    static const double M_TO_XYZ[3][3] = {
        { 0.7006607968,  0.1487679160,  0.1010412872},
        { 0.2741335367,  0.8735913831, -0.1477249199},
        {-0.0989683375, -0.1378889305,  1.3256872680}
    };
    // XYZ → DaVinci Wide Gamut linear (D65 white point)
    static const double M_TO_RGB[3][3] = {
        { 1.5165889100, -0.2814626195, -0.1469555778},
        {-0.4649386618,  1.2514818105,  0.1748927191},
        { 0.0648602355,  0.1091581735,  0.7615459883}
    };
    // Illuminant XYZ (Y=1 normalised, CIE 1931 2° observer)
    static const double WP[6][3] = {
        {0.96422, 1.00000, 0.82521},  // D50
        {0.95682, 1.00000, 0.92149},  // D55
        {0.95265, 1.00000, 1.00883},  // D60 (ACES)
        {0.95047, 1.00000, 1.08883},  // D65
        {0.94936, 1.00000, 1.16006},  // D70
        {0.94972, 1.00000, 1.22638}   // D75
    };

    // Cone responses for source and destination illuminants
    double rhoS[3], rhoD[3];
    for (int i = 0; i < 3; i++) {
        rhoS[i] = rhoD[i] = 0.0;
        for (int j = 0; j < 3; j++) {
            rhoS[i] += MB[i][j] * WP[srcWP][j];
            rhoD[i] += MB[i][j] * WP[dstWP][j];
        }
    }

    // Diagonal scale in cone space
    double scale[3] = { rhoD[0]/rhoS[0], rhoD[1]/rhoS[1], rhoD[2]/rhoS[2] };

    // M_cat (XYZ→XYZ) = MB_INV × diag(scale) × MB
    double Stmp[3][3], M_cat[3][3], tmp[3][3], M_final[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            Stmp[i][j] = scale[i] * MB[i][j];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            M_cat[i][j] = 0.0;
            for (int k = 0; k < 3; k++) M_cat[i][j] += MB_INV[i][k] * Stmp[k][j];
        }

    // Bring into sRGB space: M_final = M_TO_RGB × M_cat × M_TO_XYZ
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            tmp[i][j] = 0.0;
            for (int k = 0; k < 3; k++) tmp[i][j] += M_cat[i][k] * M_TO_XYZ[k][j];
        }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            M_final[i][j] = 0.0;
            for (int k = 0; k < 3; k++) M_final[i][j] += M_TO_RGB[i][k] * tmp[k][j];
        }

    // Pack row-major into output
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            out[i*3 + j] = (float)M_final[i][j];

    return true;
}

static void kelvinToDIlluminantXYZ(double kelvin, double xyz[3])
{
    double T = normalizeWpKelvin(kelvin);
    double xD;
    if (T <= 7000.0)
        xD = -4.607e9/(T*T*T) + 2.9678e6/(T*T) + 0.09911e3/T + 0.244063;
    else
        xD = -2.0064e9/(T*T*T) + 1.9018e6/(T*T) + 0.24748e3/T + 0.237040;

    double yD = -3.000*xD*xD + 2.870*xD - 0.275;
    if (yD < 1e-6) yD = 1e-6;

    xyz[0] = xD / yD;
    xyz[1] = 1.0;
    xyz[2] = (1.0 - xD - yD) / yD;
}

static bool bradfordCatKelvin(int srcWP, double targetKelvin, float out[9])
{
    srcWP = normalizeWpChoice(srcWP);
    targetKelvin = normalizeWpKelvin(targetKelvin);
    if (std::fabs(targetKelvin - kWpChoiceKelvin[srcWP]) < 5.0) return false;

    static const double MB[3][3] = {
        { 0.8951,  0.2664, -0.1614},
        {-0.7502,  1.7135,  0.0367},
        { 0.0389, -0.0685,  1.0296}
    };
    static const double MB_INV[3][3] = {
        { 0.9869929, -0.1470543,  0.1599627},
        { 0.4323053,  0.5183603,  0.0492912},
        {-0.0085287,  0.0400428,  0.9684867}
    };
    static const double M_TO_XYZ[3][3] = {
        { 0.7006607968,  0.1487679160,  0.1010412872},
        { 0.2741335367,  0.8735913831, -0.1477249199},
        {-0.0989683375, -0.1378889305,  1.3256872680}
    };
    static const double M_TO_RGB[3][3] = {
        { 1.5165889100, -0.2814626195, -0.1469555778},
        {-0.4649386618,  1.2514818105,  0.1748927191},
        { 0.0648602355,  0.1091581735,  0.7615459883}
    };
    static const double WP_SRC[6][3] = {
        {0.96422, 1.00000, 0.82521},
        {0.95682, 1.00000, 0.92149},
        {0.95265, 1.00000, 1.00883},
        {0.95047, 1.00000, 1.08883},
        {0.94936, 1.00000, 1.16006},
        {0.94972, 1.00000, 1.22638}
    };

    double dstXYZ[3];
    kelvinToDIlluminantXYZ(targetKelvin, dstXYZ);

    double rhoS[3], rhoD[3];
    for (int i = 0; i < 3; i++) {
        rhoS[i] = rhoD[i] = 0.0;
        for (int j = 0; j < 3; j++) {
            rhoS[i] += MB[i][j] * WP_SRC[srcWP][j];
            rhoD[i] += MB[i][j] * dstXYZ[j];
        }
    }

    double scale[3] = { rhoD[0]/rhoS[0], rhoD[1]/rhoS[1], rhoD[2]/rhoS[2] };

    double Stmp[3][3], M_cat[3][3], tmp[3][3], M_final[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            Stmp[i][j] = scale[i] * MB[i][j];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            M_cat[i][j] = 0.0;
            for (int k = 0; k < 3; k++) M_cat[i][j] += MB_INV[i][k] * Stmp[k][j];
        }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            tmp[i][j] = 0.0;
            for (int k = 0; k < 3; k++) tmp[i][j] += M_cat[i][k] * M_TO_XYZ[k][j];
        }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            M_final[i][j] = 0.0;
            for (int k = 0; k < 3; k++) M_final[i][j] += M_TO_RGB[i][k] * tmp[k][j];
        }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            out[i*3 + j] = (float)M_final[i][j];

    return true;
}

static void updateImageBlurEnabled(OfxImageEffectHandle effect) {
    OfxParamSetHandle ps;
    gEffectHost->getParamSet(effect, &ps);
    bool grainEn    = getBool(ps, PARAM_GRAIN_ENABLE);
    bool filmBlurEn = getBool(ps, PARAM_IMAGE_BLUR_ENABLE);
    setParamEnabled(effect, PARAM_GRAIN_BLUR_LINK, grainEn && filmBlurEn);
}

static void updateSfLookEnabled(OfxImageEffectHandle effect) {
    OfxParamSetHandle ps;
    gEffectHost->getParamSet(effect, &ps);
    bool en = getBool(ps, PARAM_SF_LOOK_ENABLE);
    setParamEnabled(effect, PARAM_SF_LOOK_SATURATION, en);
    setParamEnabled(effect, PARAM_SF_CONTRAST_CURVE,  en);
    setParamEnabled(effect, PARAM_SF_WP_KELVIN,     en);
}

static void updateHalationEnabled(OfxImageEffectHandle effect) {
    OfxParamSetHandle ps;
    gEffectHost->getParamSet(effect, &ps);
    bool en = getBool(ps, PARAM_HAL_ENABLE);
    setParamEnabled(effect, PARAM_HAL_SIZE,    en);
    setParamEnabled(effect, PARAM_HAL_COLOR,   en);
    setParamEnabled(effect, PARAM_HAL_METHOD,  en);
    setParamEnabled(effect, PARAM_HAL_OPACITY, en);
    if (en) setGroupOpen(effect, "grp_halation", true);
}

static void updateDiffusionEnabled(OfxImageEffectHandle effect) {
    OfxParamSetHandle ps;
    gEffectHost->getParamSet(effect, &ps);
    bool en = getBool(ps, PARAM_DIFFUSION_ENABLE);
    setParamEnabled(effect, PARAM_DIFFUSION_PROFILE,       en);
    setParamEnabled(effect, PARAM_DIFFUSION_AMOUNT,       en);
    setParamEnabled(effect, PARAM_DIFFUSION_SIZE,       en);
    setParamEnabled(effect, PARAM_DIFFUSION_SOFTNESS,   en);
    setParamEnabled(effect, PARAM_DIFFUSION_BODY_STRENGTH,     en);
    setParamEnabled(effect, PARAM_DIFFUSION_FALLOFF_STRENGTH, en);
    setParamEnabled(effect, PARAM_DIFFUSION_HIGHLIGHT_GATE,    en);
    setParamEnabled(effect, PARAM_DIFFUSION_VEIL,         en);
    setParamEnabled(effect, PARAM_DIFFUSION_CONTRAST,      en);
    setParamEnabled(effect, PARAM_DIFFUSION_HIGHLIGHT_BOOST, en);
    setParamEnabled(effect, PARAM_DIFFUSION_TINT_STRENGTH,en);
    setParamEnabled(effect, PARAM_DIFFUSION_TINT_COLOR,   en);
    if (en) setGroupOpen(effect, "grp_diffusion", true);
}

static void updateSplitEnabled(OfxImageEffectHandle effect) {
    OfxParamSetHandle ps;
    gEffectHost->getParamSet(effect, &ps);
    bool en = getBool(ps, PARAM_SPLIT_ENABLE);
    setParamEnabled(effect, PARAM_SPLIT_STRENGTH,    en);
    setParamEnabled(effect, PARAM_SPLIT_SHADOW_HUE,  en);
    setParamEnabled(effect, PARAM_SPLIT_HI_HUE,      en);
    setParamEnabled(effect, PARAM_SPLIT_LO_VS_HI,    en);
    setParamEnabled(effect, PARAM_SPLIT_PIVOT,        en);
    setParamEnabled(effect, PARAM_SPLIT_SUB_LO,       en);
    setParamEnabled(effect, PARAM_SPLIT_SUB_HI,       en);
    setParamEnabled(effect, PARAM_SPLIT_NEUTRAL_W,    en);
    if (en) setGroupOpen(effect, "grp_split", true);
}

static void updatePrintEnabled(OfxImageEffectHandle effect) {
    OfxParamSetHandle ps;
    gEffectHost->getParamSet(effect, &ps);
    bool en = getBool(ps, PARAM_PRINT_ENABLE);
    setParamEnabled(effect, PARAM_PRINT_GANG, en);
    setParamEnabled(effect, PARAM_PRINT_R,    en);
    setParamEnabled(effect, PARAM_PRINT_G,    en);
    setParamEnabled(effect, PARAM_PRINT_B,    en);
    if (en) setGroupOpen(effect, "grp_print", true);
}

// ============================================================
// Master Preset: list of all preset-managed param names + types
// ============================================================
struct PresetParamDef {
    const char* name;
    bool isInt;  // true = int/bool, false = double
};

static const PresetParamDef kPresetParams[] = {
    { PARAM_GRAIN_ENABLE,     true },
    { PARAM_FILM_FORMAT,      true },
    { PARAM_GRAIN_SIZE,       false },
    { PARAM_GRAIN_STRENGTH,   false },
    { PARAM_GRAIN_BLUR,       false },
    { PARAM_GRAIN_OPACITY,    false },
    { PARAM_GRAIN_SAT,        false },
    { PARAM_SHAPE_IRREG,      false },
    { PARAM_GRAIN_CONTRAST,   false },
    { PARAM_HI_ROLLOFF,       false },
    { PARAM_SEED_OFFSET,      true },
    { PARAM_IMAGE_BLUR_ENABLE,true },
    { PARAM_IMAGE_BLUR,       false },
    { PARAM_GRAIN_BLUR_LINK,  false },
    { PARAM_SF_LOOK_ENABLE,   true  },
    { PARAM_SF_LOOK_SATURATION, false },
    { PARAM_SF_CONTRAST_CURVE,  false },
    { PARAM_SF_WP_KELVIN,     false },
    { PARAM_HAL_ENABLE,       true },
    { PARAM_HAL_SIZE,         false },
    { PARAM_HAL_COLOR,        false },
    { PARAM_HAL_METHOD,       true },
    { PARAM_HAL_OPACITY,      false },
    { PARAM_DIFFUSION_ENABLE,      true },
    { PARAM_DIFFUSION_PROFILE,      true },
    { PARAM_DIFFUSION_AMOUNT,      false },
    { PARAM_DIFFUSION_SIZE,      false },
    { PARAM_DIFFUSION_SOFTNESS,  false },
    { PARAM_DIFFUSION_BODY_STRENGTH,    false },
    { PARAM_DIFFUSION_FALLOFF_STRENGTH,false },
    { PARAM_DIFFUSION_HIGHLIGHT_GATE,   false },
    { PARAM_DIFFUSION_VEIL,        false },
    { PARAM_DIFFUSION_CONTRAST,     false },
    { PARAM_DIFFUSION_HIGHLIGHT_BOOST, false },
    { PARAM_DIFFUSION_TINT_STRENGTH,false },
    { PARAM_SPLIT_ENABLE,       true },
    { PARAM_SPLIT_STRENGTH,     false },
    { PARAM_SPLIT_SHADOW_HUE,   false },
    { PARAM_SPLIT_HI_HUE,       false },
    { PARAM_SPLIT_LO_VS_HI,     false },
    { PARAM_SPLIT_PIVOT,         false },
    { PARAM_SPLIT_SUB_LO,        false },
    { PARAM_SPLIT_SUB_HI,        false },
    { PARAM_SPLIT_NEUTRAL_W,     false },
    { PARAM_PRINT_ENABLE,     true },
    { PARAM_PRINT_GANG,       true },
    { PARAM_PRINT_R,          false },
    { PARAM_PRINT_G,          false },
    { PARAM_PRINT_B,          false },
    { PARAM_OUTPUT_TRANSFORM, true },
};
static const int kNumPresetParams =
    (int)(sizeof(kPresetParams) / sizeof(kPresetParams[0]));

// Apply a master preset: set all param values from the preset map.
static void applyMasterPreset(OfxImageEffectHandle effect,
                              const MasterPreset& preset,
                              InstanceData* data, double time) {
    OfxParamSetHandle ps;
    gEffectHost->getParamSet(effect, &ps);

    for (int i = 0; i < kNumPresetParams; ++i) {
        auto it = preset.values.find(kPresetParams[i].name);
        if (it == preset.values.end()) continue;

        if (kPresetParams[i].isInt) {
            int value = (int)it->second;
            if (strcmp(kPresetParams[i].name, PARAM_WP_ADAPT) == 0) {
                setWpChoiceParam(ps, value);
            } else {
                setIntParam(ps, kPresetParams[i].name, value);
            }
        } else if (strcmp(kPresetParams[i].name, PARAM_WP_KELVIN) == 0) {
            setWpKelvinParam(ps, it->second);
        } else {
            setDoubleParam(ps, kPresetParams[i].name, it->second);
        }
    }

    if (preset.values.find(PARAM_DIFFUSION_SOFTNESS) == preset.values.end()) {
        setDoubleParam(ps, PARAM_DIFFUSION_SOFTNESS, 1.0);
    }

    auto tr = preset.values.find(PRESET_DIFFUSION_TINT_R);
    auto tg = preset.values.find(PRESET_DIFFUSION_TINT_G);
    auto tb = preset.values.find(PRESET_DIFFUSION_TINT_B);
    if (tr != preset.values.end() || tg != preset.values.end() || tb != preset.values.end()) {
        double r, g, b;
        getRGB(ps, PARAM_DIFFUSION_TINT_COLOR, time, r, g, b);
        if (tr != preset.values.end()) r = tr->second;
        if (tg != preset.values.end()) g = tg->second;
        if (tb != preset.values.end()) b = tb->second;
        setRGBParam(ps, PARAM_DIFFUSION_TINT_COLOR, r, g, b);
    }

    // ---- Embedded LUT handling ----
    if (data && preset.hasEmbeddedLut && !preset.embeddedLutPath.empty()) {
        std::string cubePath = preset.sourceDir + "/" + preset.embeddedLutPath;
        CubeLut cube;
        if (parseCubeFile(cubePath.c_str(), cube)) {
            if (!preset.embeddedLutTitle.empty())
                cube.title = preset.embeddedLutTitle;
            cube.nativeWP = preset.embeddedLutWp;
            cube.filename = preset.embeddedLutPath;
            data->embeddedLut = std::move(cube);
            data->hasEmbeddedLut = true;
            data->embeddedLutUploaded = false;
        }
    } else if (data) {
        // Non-embedded preset or Custom — clear any embedded LUT
        data->hasEmbeddedLut = false;
        data->embeddedLutUploaded = false;
    }
}

// Detect which preset (if any) matches the current param values.
// Returns preset choice index (0 = Custom, 1+ = preset index).
static int detectMasterPreset(OfxImageEffectHandle effect,
                              const std::vector<MasterPreset>& presets,
                              const std::vector<int>& choiceToPreset,
                              double time) {
    if (presets.empty()) return 0;

    OfxParamSetHandle ps;
    gEffectHost->getParamSet(effect, &ps);

    for (size_t pi = 0; pi < presets.size(); ++pi) {
        if (presets[pi].hasEmbeddedLut) continue;

        bool match = true;
        for (int i = 0; i < kNumPresetParams && match; ++i) {
            auto it = presets[pi].values.find(kPresetParams[i].name);
            if (it == presets[pi].values.end()) {
                if (strcmp(kPresetParams[i].name, PARAM_DIFFUSION_SOFTNESS) == 0) {
                    double cur = getDouble(ps, PARAM_DIFFUSION_SOFTNESS, time);
                    if (fabs(cur - 1.0) > 0.001) match = false;
                }
                continue;
            }

            if (kPresetParams[i].isInt) {
                int cur = getInt(ps, kPresetParams[i].name);
                int expected = (int)it->second;
                if (strcmp(kPresetParams[i].name, PARAM_WP_ADAPT) == 0) {
                    cur = getWpChoiceParam(ps);
                    expected = normalizeWpChoice(expected);
                }
                if (cur != expected) match = false;
            } else {
                double cur = getDouble(ps, kPresetParams[i].name, time);
                double expected = it->second;
                if (strcmp(kPresetParams[i].name, PARAM_WP_KELVIN) == 0) {
                    cur = getWpKelvinParam(ps);
                    expected = normalizeWpKelvin(expected);
                }
                if (fabs(cur - expected) > 0.001) match = false;
            }
        }
        if (match) {
            double r, g, b;
            getRGB(ps, PARAM_DIFFUSION_TINT_COLOR, time, r, g, b);
            auto tr = presets[pi].values.find(PRESET_DIFFUSION_TINT_R);
            auto tg = presets[pi].values.find(PRESET_DIFFUSION_TINT_G);
            auto tb = presets[pi].values.find(PRESET_DIFFUSION_TINT_B);
            if (tr != presets[pi].values.end() && fabs(r - tr->second) > 0.001) match = false;
            if (tg != presets[pi].values.end() && fabs(g - tg->second) > 0.001) match = false;
            if (tb != presets[pi].values.end() && fabs(b - tb->second) > 0.001) match = false;
        }
        if (match) {
            // Map preset index pi back to the dropdown choice index via choiceToPreset
            for (size_t ci = 0; ci < choiceToPreset.size(); ++ci) {
                if (choiceToPreset[ci] == (int)pi)
                    return (int)ci;
            }
            return (int)(pi + 1);  // fallback: direct mapping (no separator)
        }
    }
    return 0;  // Custom
}

// ============================================================
// Utility: (re-)discover presets from bundle + user dir and repopulate
// the Master Preset choice dropdown. Keeps the current selection if it
// still exists, otherwise resets to Custom (0).
// ============================================================
static void refreshPresets(OfxImageEffectHandle effect, InstanceData* data) {
    if (!data) return;

    std::string bundlePath = getBundlePath(effect);

    // Re-scan both locations.
    data->presets.clear();
    size_t bundleCount = 0;
    if (!bundlePath.empty()) {
        std::string presetDir = bundlePath + "/Contents/Resources/Presets";
        auto bundlePresets = discoverPresets(presetDir.c_str());
        bundleCount = bundlePresets.size();
        for (auto& p : bundlePresets)
            data->presets.push_back(std::move(p));
    }
    size_t userCount = 0;
    {
        const char* home = getenv("HOME");
        if (home) {
            std::string userPresetDir = std::string(home) + "/Documents/Lofi FX Film Presets";
            auto userPresets = discoverPresets(userPresetDir.c_str());
            userCount = userPresets.size();
            for (auto& p : userPresets)
                data->presets.push_back(std::move(p));
        }
    }

    // Repopulate the dropdown with "── User Presets ──" separator.
    OfxParamSetHandle ps;
    gEffectHost->getParamSet(effect, &ps);
    OfxParamHandle presetParam;
    gParamHost->paramGetHandle(ps, PARAM_MASTER_PRESET, &presetParam, nullptr);
    OfxPropertySetHandle presetProps;
    gParamHost->paramGetPropertySet(presetParam, &presetProps);

    data->choiceToPreset.clear();
    int choiceIdx = 0;

    gPropHost->propSetString(presetProps, kOfxParamPropChoiceOption, choiceIdx, "Custom");
    data->choiceToPreset.push_back(-1);
    choiceIdx++;

    for (size_t i = 0; i < bundleCount; ++i) {
        gPropHost->propSetString(presetProps, kOfxParamPropChoiceOption,
                                 choiceIdx, data->presets[i].name.c_str());
        data->choiceToPreset.push_back((int)i);
        choiceIdx++;
    }

    if (userCount > 0) {
        gPropHost->propSetString(presetProps, kOfxParamPropChoiceOption,
                                 choiceIdx, "── User Presets ──");
        data->choiceToPreset.push_back(-1);
        choiceIdx++;
        for (size_t i = bundleCount; i < data->presets.size(); ++i) {
            gPropHost->propSetString(presetProps, kOfxParamPropChoiceOption,
                                     choiceIdx, data->presets[i].name.c_str());
            data->choiceToPreset.push_back((int)i);
            choiceIdx++;
        }
    }
}

// ============================================================
// ACTION: CreateInstance — create per-instance data, discover resources
// ============================================================
static OfxStatus actionCreateInstance(OfxImageEffectHandle effect) {
    InstanceData* data = new InstanceData();

    // Discover bundle path and cache it globally (used for spectral stock loading)
    std::string bundlePath = getBundlePath(effect);
    if (!bundlePath.empty()) gBundlePath = bundlePath;

    // Must set instance data before refreshPresets (which reads it back).
    setInstanceData(effect, data);

    // Determine whether this is a brand-new instance or a copy / project-reload.
    OfxParamSetHandle earlyParamSet;
    gEffectHost->getParamSet(effect, &earlyParamSet);
    const bool isFreshInstance = (getInt(earlyParamSet, PARAM_INITIALIZED) == 0);

    // Apply factory defaults on brand-new instances (host may not initialize to
    // descriptor defaults reliably).  Mirrors the old basic.preset values.
    if (isFreshInstance) {
        MasterPreset defaultPreset;
        defaultPreset.values = {
            // grain
            { PARAM_GRAIN_ENABLE,      1.0 },
            { PARAM_FILM_FORMAT,       (double)FMT_35MM },
            { PARAM_GRAIN_SIZE,        1.5 },
            { PARAM_GRAIN_STRENGTH,    0.3 },
            { PARAM_GRAIN_BLUR,        0.65 },
            { PARAM_GRAIN_OPACITY,     0.25 },
            { PARAM_GRAIN_SAT,         0.5 },
            { PARAM_SHAPE_IRREG,       0.5 },
            { PARAM_GRAIN_CONTRAST,    0.5 },
            { PARAM_HI_ROLLOFF,        0.650 },
            { PARAM_SEED_OFFSET,       42.0 },
            { PARAM_IMAGE_BLUR_ENABLE, 0.0 },
            { PARAM_IMAGE_BLUR,        0.7 },
            { PARAM_GRAIN_BLUR_LINK,   0.5 },
            // look (film emulation defaults)
            { PARAM_SF_LOOK_ENABLE,    1.0 },
            { PARAM_SF_LOOK_SATURATION, 1.0 },
            { PARAM_SF_CONTRAST_CURVE, 1.0 },
            { PARAM_SF_WP_KELVIN,     6500.0 },
            // halation
            { PARAM_HAL_ENABLE,        0.0 },
            { PARAM_HAL_SIZE,          1.0 },
            { PARAM_HAL_COLOR,         0.15 },
            { PARAM_HAL_METHOD,        0.0 },
            { PARAM_HAL_OPACITY,       1.0 },
            // diffusion
            { PARAM_DIFFUSION_ENABLE,           0.0 },
            { PARAM_DIFFUSION_PROFILE,          (double)DIFFUSION_PROFILE_BPM_18 },
            { PARAM_DIFFUSION_AMOUNT,          0.075 },
            { PARAM_DIFFUSION_SIZE,            64.0 },
            { PARAM_DIFFUSION_SOFTNESS,         1.0 },
            { PARAM_DIFFUSION_BODY_STRENGTH,    1.0 },
            { PARAM_DIFFUSION_FALLOFF_STRENGTH, 1.0 },
            { PARAM_DIFFUSION_HIGHLIGHT_GATE,   0.38 },
            { PARAM_DIFFUSION_VEIL,             0.0 },
            { PARAM_DIFFUSION_CONTRAST,         0.0 },
            { PARAM_DIFFUSION_HIGHLIGHT_BOOST,  0.0 },
            { PARAM_DIFFUSION_TINT_STRENGTH,    0.0 },
            { PRESET_DIFFUSION_TINT_R,          1.0 },
            { PRESET_DIFFUSION_TINT_G,          1.0 },
            { PRESET_DIFFUSION_TINT_B,          1.0 },
            // split
            { PARAM_SPLIT_ENABLE,      0.0 },
            { PARAM_SPLIT_STRENGTH,    0.0 },
            { PARAM_SPLIT_SHADOW_HUE,  0.0 },
            { PARAM_SPLIT_HI_HUE,      0.0 },
            { PARAM_SPLIT_LO_VS_HI,    0.5 },
            { PARAM_SPLIT_PIVOT,        0.336 },
            { PARAM_SPLIT_SUB_LO,       1.0 },
            { PARAM_SPLIT_SUB_HI,       1.0 },
            { PARAM_SPLIT_NEUTRAL_W,    0.15 },
            // print
            { PARAM_PRINT_ENABLE,      0.0 },
            { PARAM_PRINT_GANG,        0.0 },
            { PARAM_PRINT_R,          25.0 },
            { PARAM_PRINT_G,          25.0 },
            { PARAM_PRINT_B,          25.0 },
            // transform
            { PARAM_OUTPUT_TRANSFORM,  0.0 },
        };
        applyMasterPreset(effect, defaultPreset, data, 0);
    }

    // ---- Discover master presets ----
    refreshPresets(effect, data);

    // ---- Load CST LUTs from packed blob ----
    if (!bundlePath.empty()) {
        std::string pakPath = bundlePath + "/Contents/Resources/resources.pak";
        loadLutPackSplit(pakPath.c_str(),
                         data->inputCsts, data->outputCsts);
    }

    // Populate Output Transform choice parameter dynamically
    if (!data->outputCsts.empty()) {
        OfxParamSetHandle paramSet;
        gEffectHost->getParamSet(effect, &paramSet);
        OfxParamHandle outParam;
        gParamHost->paramGetHandle(paramSet, PARAM_OUTPUT_TRANSFORM, &outParam, nullptr);
        OfxPropertySetHandle outProps;
        gParamHost->paramGetPropertySet(outParam, &outProps);

        gPropHost->propSetString(outProps, kOfxParamPropChoiceOption, 0, "DaVinci WG / Intermediate");
        for (size_t i = 0; i < data->outputCsts.size(); ++i) {
            gPropHost->propSetString(outProps, kOfxParamPropChoiceOption,
                                     (int)(i + 1),
                                     data->outputCsts[i].title.c_str());
        }
    }

    // Populate Input Transform choice parameter dynamically
    if (!data->inputCsts.empty()) {
        OfxParamSetHandle paramSet;
        gEffectHost->getParamSet(effect, &paramSet);
        OfxParamHandle inParam;
        gParamHost->paramGetHandle(paramSet, PARAM_INPUT_TRANSFORM, &inParam, nullptr);
        OfxPropertySetHandle inProps;
        gParamHost->paramGetPropertySet(inParam, &inProps);

        gPropHost->propSetString(inProps, kOfxParamPropChoiceOption, 0, "DaVinci WG / Intermediate");
        for (size_t i = 0; i < data->inputCsts.size(); ++i) {
            gPropHost->propSetString(inProps, kOfxParamPropChoiceOption,
                                     (int)(i + 1),
                                     data->inputCsts[i].title.c_str());
        }
    }

    // Mark this instance as fully initialised so that future copies and
    // project-reloads skip the factory-default overrides above.
    {
        OfxParamSetHandle ps;
        gEffectHost->getParamSet(effect, &ps);
        setIntParam(ps, PARAM_INITIALIZED, 1);
    }

    // Set initial enabled state for param groups
    updateGrainEnabled(effect);
    updateImageBlurEnabled(effect);
    updateSfLookEnabled(effect);
    updateHalationEnabled(effect);
    updateDiffusionEnabled(effect);
    updateSplitEnabled(effect);
    updatePrintEnabled(effect);

    // Group open states are set by each update*Enabled call above.

    return kOfxStatOK;
}

// ============================================================
// ACTION: DestroyInstance
// ============================================================
static OfxStatus actionDestroyInstance(OfxImageEffectHandle effect) {
    InstanceData* data = getInstanceData(effect);
    if (data) {
#ifdef __APPLE__
        if (data->metalOutputCstBuffer)
            metalFreeLut(data->metalOutputCstBuffer);
        if (data->metalInputCstBuffer)
            metalFreeLut(data->metalInputCstBuffer);
        if (data->metalSfLutBuffer)
            metalFreeLut(data->metalSfLutBuffer);
        if (data->metalEmbeddedLutBuffer)
            metalFreeLut(data->metalEmbeddedLutBuffer);
#endif
        if (data->cudaOutputCstBuffer)
            cudaFreeLut(data->cudaOutputCstBuffer);
        if (data->cudaInputCstBuffer)
            cudaFreeLut(data->cudaInputCstBuffer);
        if (data->cudaEmbeddedLutBuffer)
            cudaFreeLut(data->cudaEmbeddedLutBuffer);
        delete data;
    }
    setInstanceData(effect, nullptr);
    return kOfxStatOK;
}

// ============================================================
// Save current param state as a .preset file to ~/Documents/Lofi FX Film Presets/
// ============================================================
static void saveCurrentPreset(OfxImageEffectHandle effect, double time) {
    OfxParamSetHandle ps;
    gEffectHost->getParamSet(effect, &ps);

    std::string presetName = getString(ps, PARAM_PRESET_NAME);
    if (presetName.empty()) presetName = "My Preset";

    // Build safe filename from preset name
    std::string safeFilename;
    for (char c : presetName) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == ' ')
            safeFilename += c;
        else
            safeFilename += '_';
    }
    if (safeFilename.empty()) safeFilename = "preset";

    // Determine output directory: ~/Documents/Lofi FX Film Presets/
    const char* home = getenv("HOME");
    if (!home) home = "/tmp";
    std::string dir = std::string(home) + "/Documents/Lofi FX Film Presets";

#ifdef _WIN32
    _mkdir(dir.c_str());
#else
    mkdir(dir.c_str(), 0755);
#endif

    std::string path = dir + "/" + safeFilename + ".preset";

    std::ofstream f(path);
    if (!f.is_open()) return;

    f << "# ============================================================\n";
    f << "# Lofi FX Film — User Preset: " << presetName << "\n";
    f << "# ============================================================\n\n";
    f << "name = " << presetName << "\n\n";

    for (int i = 0; i < kNumPresetParams; ++i) {
        if (kPresetParams[i].isInt) {
            int v = getInt(ps, kPresetParams[i].name);
            f << kPresetParams[i].name << " = " << v << "\n";
        } else {
            double v = getDouble(ps, kPresetParams[i].name, time);
            f << kPresetParams[i].name << " = " << v << "\n";
        }
    }

    double diffusionTintR, diffusionTintG, diffusionTintB;
    getRGB(ps, PARAM_DIFFUSION_TINT_COLOR, time, diffusionTintR, diffusionTintG, diffusionTintB);
    f << PRESET_DIFFUSION_TINT_R << " = " << diffusionTintR << "\n";
    f << PRESET_DIFFUSION_TINT_G << " = " << diffusionTintG << "\n";
    f << PRESET_DIFFUSION_TINT_B << " = " << diffusionTintB << "\n";

    f.close();
}

// ============================================================
// ACTION: InstanceChanged — Master presets, Film Size, enable toggles
// ============================================================
static OfxStatus actionInstanceChanged(OfxImageEffectHandle effect,
                                       OfxPropertySetHandle inArgs)
{
    char* name = nullptr;
    gPropHost->propGetString(inArgs, kOfxPropName, 0, &name);
    if (!name) return kOfxStatReplyDefault;

    double time = 0;
    gPropHost->propGetDouble(inArgs, kOfxPropTime, 0, &time);

    OfxParamSetHandle ps;
    gEffectHost->getParamSet(effect, &ps);

    InstanceData* instData = getInstanceData(effect);

    // ---- Master Preset changed ----
    if (strcmp(name, PARAM_MASTER_PRESET) == 0) {
        int idx = getInt(ps, PARAM_MASTER_PRESET);
        if (instData && idx >= 0 && idx < (int)instData->choiceToPreset.size()) {
            int pi = instData->choiceToPreset[idx];
            if (pi >= 0 && pi < (int)instData->presets.size()) {
                instData->applyingPreset = true;
                applyMasterPreset(effect, instData->presets[pi], instData, time);
                instData->applyingPreset = false;
                updateGrainEnabled(effect);
                updateImageBlurEnabled(effect);
                updateSfLookEnabled(effect);
                updateHalationEnabled(effect);
                updateDiffusionEnabled(effect);
                updateSplitEnabled(effect);
                updatePrintEnabled(effect);
            } else {
                instData->hasEmbeddedLut = false;
                instData->embeddedLutUploaded = false;
            }
        }
        return kOfxStatOK;
    }

    // ---- Guard: skip preset detect while applying ----
    bool skipDetect = instData && instData->applyingPreset;

    // Enable Grain toggled
    if (strcmp(name, PARAM_GRAIN_ENABLE) == 0) {
        updateGrainEnabled(effect);
        if (!skipDetect && instData) {
            int det = detectMasterPreset(effect, instData->presets, instData->choiceToPreset, time);
            int cur = getInt(ps, PARAM_MASTER_PRESET);
            if (cur != det) setIntParam(ps, PARAM_MASTER_PRESET, det);
        }
        return kOfxStatOK;
    }

    // Enable Image Blur toggled
    if (strcmp(name, PARAM_IMAGE_BLUR_ENABLE) == 0) {
        updateImageBlurEnabled(effect);
        if (!skipDetect && instData) {
            int det = detectMasterPreset(effect, instData->presets, instData->choiceToPreset, time);
            int cur = getInt(ps, PARAM_MASTER_PRESET);
            if (cur != det) setIntParam(ps, PARAM_MASTER_PRESET, det);
        }
        return kOfxStatOK;
    }

    // Save Preset button clicked
    if (strcmp(name, PARAM_SAVE_PRESET) == 0) {
        saveCurrentPreset(effect, time);
        if (instData) refreshPresets(effect, instData);
        return kOfxStatOK;
    }

    // Refresh Presets button clicked
    if (strcmp(name, PARAM_REFRESH_PRESETS) == 0) {
        if (instData) refreshPresets(effect, instData);
        return kOfxStatOK;
    }

    // Film Emulation stock selections changed: invalidate cached GPU LUT
    if (strcmp(name, PARAM_SF_NEG_STOCK) == 0 ||
        strcmp(name, PARAM_SF_PRT_STOCK) == 0)
    {
        if (instData) instData->sfLutCacheNeg = -99;  // force LUT rebuild on next render
        return kOfxStatOK;
    }

    // Film Emulation Enable Look toggled
    if (strcmp(name, PARAM_SF_LOOK_ENABLE) == 0) {
        updateSfLookEnabled(effect);
        return kOfxStatOK;
    }

    // Enable Halation toggled
    if (strcmp(name, PARAM_HAL_ENABLE) == 0) {
        updateHalationEnabled(effect);
        if (!skipDetect && instData) {
            int det = detectMasterPreset(effect, instData->presets, instData->choiceToPreset, time);
            int cur = getInt(ps, PARAM_MASTER_PRESET);
            if (cur != det) setIntParam(ps, PARAM_MASTER_PRESET, det);
        }
        return kOfxStatOK;
    }

    // Enable Diffusion toggled
    if (strcmp(name, PARAM_DIFFUSION_ENABLE) == 0) {
        updateDiffusionEnabled(effect);
        if (!skipDetect && instData) {
            int det = detectMasterPreset(effect, instData->presets, instData->choiceToPreset, time);
            int cur = getInt(ps, PARAM_MASTER_PRESET);
            if (cur != det) setIntParam(ps, PARAM_MASTER_PRESET, det);
        }
        return kOfxStatOK;
    }

    // Enable Split Toning toggled
    if (strcmp(name, PARAM_SPLIT_ENABLE) == 0) {
        updateSplitEnabled(effect);
        if (!skipDetect && instData) {
            int det = detectMasterPreset(effect, instData->presets, instData->choiceToPreset, time);
            int cur = getInt(ps, PARAM_MASTER_PRESET);
            if (cur != det) setIntParam(ps, PARAM_MASTER_PRESET, det);
        }
        return kOfxStatOK;
    }

    // Enable Print toggled
    if (strcmp(name, PARAM_PRINT_ENABLE) == 0) {
        updatePrintEnabled(effect);
        if (!skipDetect && instData) {
            int det = detectMasterPreset(effect, instData->presets, instData->choiceToPreset, time);
            int cur = getInt(ps, PARAM_MASTER_PRESET);
            if (cur != det) setIntParam(ps, PARAM_MASTER_PRESET, det);
        }
        return kOfxStatOK;
    }

    // Gang Printer Points: when a channel changes, set all three
    // Static guard prevents infinite recursion (setDoubleParam re-enters instanceChanged)
    static bool sGangingPrint = false;
    if (!sGangingPrint && !skipDetect &&
        (strcmp(name, PARAM_PRINT_R) == 0 ||
         strcmp(name, PARAM_PRINT_G) == 0 ||
         strcmp(name, PARAM_PRINT_B) == 0))
    {
        bool gang = getBool(ps, PARAM_PRINT_GANG);
        if (gang) {
            sGangingPrint = true;
            double val = getDouble(ps, name, time);
            if (strcmp(name, PARAM_PRINT_R) != 0)
                setDoubleParam(ps, PARAM_PRINT_R, val);
            if (strcmp(name, PARAM_PRINT_G) != 0)
                setDoubleParam(ps, PARAM_PRINT_G, val);
            if (strcmp(name, PARAM_PRINT_B) != 0)
                setDoubleParam(ps, PARAM_PRINT_B, val);
            sGangingPrint = false;
        }
        if (instData) {
            int det = detectMasterPreset(effect, instData->presets, instData->choiceToPreset, time);
            int cur = getInt(ps, PARAM_MASTER_PRESET);
            if (cur != det) setIntParam(ps, PARAM_MASTER_PRESET, det);
        }
        return kOfxStatOK;
    }

    // Film Size changed → apply film size preset values
    if (strcmp(name, PARAM_FILM_FORMAT) == 0) {
        int fmt = getInt(ps, PARAM_FILM_FORMAT);
        if (fmt >= 0 && fmt < kNumPresets) {
            setDoubleParam(ps, PARAM_GRAIN_SIZE, kPresets[fmt].grain_size);
            setDoubleParam(ps, PARAM_GRAIN_BLUR, kPresets[fmt].grain_blur);
        }
        if (!skipDetect && instData) {
            int det = detectMasterPreset(effect, instData->presets, instData->choiceToPreset, time);
            int cur = getInt(ps, PARAM_MASTER_PRESET);
            if (cur != det) setIntParam(ps, PARAM_MASTER_PRESET, det);
        }
        return kOfxStatOK;
    }

    // Diffusion Profile changed: apply profile recipe values.
    if (strcmp(name, PARAM_DIFFUSION_PROFILE) == 0) {
        int profile = getInt(ps, PARAM_DIFFUSION_PROFILE);
        if (profile >= 0 && profile < kNumDiffusionProfilePresets) {
            const DiffusionProfilePreset& preset = kDiffusionProfilePresets[profile];
            setDoubleParam(ps, PARAM_DIFFUSION_AMOUNT,       preset.overallStrength);
            setDoubleParam(ps, PARAM_DIFFUSION_SIZE,         preset.overallSize);
            setDoubleParam(ps, PARAM_DIFFUSION_SOFTNESS,     preset.softness);
            setDoubleParam(ps, PARAM_DIFFUSION_BODY_STRENGTH,     preset.bodyStrength);
            setDoubleParam(ps, PARAM_DIFFUSION_FALLOFF_STRENGTH, preset.falloffStrength);
            setDoubleParam(ps, PARAM_DIFFUSION_HIGHLIGHT_GATE,    preset.highlightGate);
            setDoubleParam(ps, PARAM_DIFFUSION_VEIL,         preset.veilLift);
            setDoubleParam(ps, PARAM_DIFFUSION_CONTRAST,      preset.contrastReduction);
            setDoubleParam(ps, PARAM_DIFFUSION_HIGHLIGHT_BOOST, preset.highlightBoost);
            setDoubleParam(ps, PARAM_DIFFUSION_TINT_STRENGTH,preset.tintStrength);
            setRGBParam(ps, PARAM_DIFFUSION_TINT_COLOR, preset.tintR, preset.tintG, preset.tintB);
        }
        if (!skipDetect && instData) {
            int det = detectMasterPreset(effect, instData->presets, instData->choiceToPreset, time);
            int cur = getInt(ps, PARAM_MASTER_PRESET);
            if (cur != det) setIntParam(ps, PARAM_MASTER_PRESET, det);
        }
        return kOfxStatOK;
    }

    // Any individual diffusion value changed: detect matching profile or set Custom.
    if (strcmp(name, PARAM_DIFFUSION_AMOUNT) == 0       ||
        strcmp(name, PARAM_DIFFUSION_SIZE) == 0       ||
        strcmp(name, PARAM_DIFFUSION_SOFTNESS) == 0   ||
        strcmp(name, PARAM_DIFFUSION_BODY_STRENGTH) == 0     ||
        strcmp(name, PARAM_DIFFUSION_FALLOFF_STRENGTH) == 0 ||
        strcmp(name, PARAM_DIFFUSION_HIGHLIGHT_GATE) == 0    ||
        strcmp(name, PARAM_DIFFUSION_VEIL) == 0         ||
        strcmp(name, PARAM_DIFFUSION_CONTRAST) == 0      ||
        strcmp(name, PARAM_DIFFUSION_HIGHLIGHT_BOOST) == 0 ||
        strcmp(name, PARAM_DIFFUSION_TINT_STRENGTH) == 0||
        strcmp(name, PARAM_DIFFUSION_TINT_COLOR) == 0)
    {
        double currentStrength = getDouble(ps, PARAM_DIFFUSION_AMOUNT, time);
        double currentSize     = getDouble(ps, PARAM_DIFFUSION_SIZE, time);
        double currentSoftness = getDouble(ps, PARAM_DIFFUSION_SOFTNESS, time);
        double currentBody     = getDouble(ps, PARAM_DIFFUSION_BODY_STRENGTH, time);
        double currentFalloff  = getDouble(ps, PARAM_DIFFUSION_FALLOFF_STRENGTH, time);
        double currentGate     = getDouble(ps, PARAM_DIFFUSION_HIGHLIGHT_GATE, time);
        double currentVeil     = getDouble(ps, PARAM_DIFFUSION_VEIL, time);
        double currentContrast = getDouble(ps, PARAM_DIFFUSION_CONTRAST, time);
        double currentBoost    = getDouble(ps, PARAM_DIFFUSION_HIGHLIGHT_BOOST, time);
        double currentTint     = getDouble(ps, PARAM_DIFFUSION_TINT_STRENGTH, time);
        double currentTintR, currentTintG, currentTintB;
        getRGB(ps, PARAM_DIFFUSION_TINT_COLOR, time, currentTintR, currentTintG, currentTintB);

        int matched = DIFFUSION_PROFILE_CUSTOM;
        for (int i = 0; i < kNumDiffusionProfilePresets; ++i) {
            const DiffusionProfilePreset& preset = kDiffusionProfilePresets[i];
            if (fabs(currentStrength - preset.overallStrength)  < 0.001 &&
                fabs(currentSize     - preset.overallSize)      < 0.001 &&
                fabs(currentSoftness - preset.softness)         < 0.001 &&
                fabs(currentBody     - preset.bodyStrength)     < 0.001 &&
                fabs(currentFalloff  - preset.falloffStrength)  < 0.001 &&
                fabs(currentGate     - preset.highlightGate)    < 0.001 &&
                fabs(currentVeil     - preset.veilLift)         < 0.001 &&
                fabs(currentContrast - preset.contrastReduction) < 0.001 &&
                fabs(currentBoost    - preset.highlightBoost)    < 0.001 &&
                fabs(currentTint     - preset.tintStrength)     < 0.001 &&
                fabs(currentTintR    - preset.tintR)            < 0.001 &&
                fabs(currentTintG    - preset.tintG)            < 0.001 &&
                fabs(currentTintB    - preset.tintB)            < 0.001)
            {
                matched = i;
                break;
            }
        }

        int current = getInt(ps, PARAM_DIFFUSION_PROFILE);
        if (current != matched)
            setIntParam(ps, PARAM_DIFFUSION_PROFILE, matched);

        if (!skipDetect && instData) {
            int det = detectMasterPreset(effect, instData->presets, instData->choiceToPreset, time);
            int cur = getInt(ps, PARAM_MASTER_PRESET);
            if (cur != det) setIntParam(ps, PARAM_MASTER_PRESET, det);
        }
        return kOfxStatOK;
    }

    // Grain Size or Grain Blur changed → detect film size preset
    if (strcmp(name, PARAM_GRAIN_SIZE) == 0 ||
        strcmp(name, PARAM_GRAIN_BLUR) == 0)
    {
        double gs = getDouble(ps, PARAM_GRAIN_SIZE, time);
        double gb = getDouble(ps, PARAM_GRAIN_BLUR, time);

        int matched = FMT_CUSTOM;
        for (int i = 0; i < kNumPresets; ++i) {
            if (fabs(gs - kPresets[i].grain_size) < 0.001 &&
                fabs(gb - kPresets[i].grain_blur) < 0.001) {
                matched = i;
                break;
            }
        }

        int current = getInt(ps, PARAM_FILM_FORMAT);
        if (current != matched)
            setIntParam(ps, PARAM_FILM_FORMAT, matched);

        if (!skipDetect && instData) {
            int det = detectMasterPreset(effect, instData->presets, instData->choiceToPreset, time);
            int cur = getInt(ps, PARAM_MASTER_PRESET);
            if (cur != det) setIntParam(ps, PARAM_MASTER_PRESET, det);
        }
        return kOfxStatOK;
    }

    // ---- Any other param changed → detect master preset ----
    if (!skipDetect && instData && strcmp(name, PARAM_MASTER_PRESET) != 0) {
        int det = detectMasterPreset(effect, instData->presets, instData->choiceToPreset, time);
        int cur = getInt(ps, PARAM_MASTER_PRESET);
        if (cur != det) setIntParam(ps, PARAM_MASTER_PRESET, det);
    }

    return kOfxStatReplyDefault;
}

// ============================================================
// ACTION: IsIdentity — skip render when effect is inactive
// ============================================================
static OfxStatus actionIsIdentity(OfxImageEffectHandle effect,
                                  OfxPropertySetHandle inArgs,
                                  OfxPropertySetHandle outArgs)
{
    OfxParamSetHandle ps;
    gEffectHost->getParamSet(effect, &ps);

    double time = 0;
    gPropHost->propGetDouble(inArgs, kOfxPropTime, 0, &time);

    bool   grainOn   = getBool(ps, PARAM_GRAIN_ENABLE);
    bool   filmBlurOn = grainOn && getBool(ps, PARAM_IMAGE_BLUR_ENABLE);
    double opacity   = grainOn ? getDouble(ps, PARAM_GRAIN_OPACITY, time) : 0;
    double filmResolveAmount = 0.0;  // base image_blur is always 0; driven only by grain_blur_link
    if (filmBlurOn) {
        float linkedBlur = linkedFilmBlurSigma(
            (float)getDouble(ps, PARAM_GRAIN_SIZE, time),
            (float)getDouble(ps, PARAM_GRAIN_BLUR_LINK, time));
        filmResolveAmount = linkedBlur;
    }
    bool   halOn     = getBool(ps, PARAM_HAL_ENABLE);
    bool   diffusionOn = getBool(ps, PARAM_DIFFUSION_ENABLE);
    bool   splitOn   = getBool(ps, PARAM_SPLIT_ENABLE);
    bool   printOn   = getBool(ps, PARAM_PRINT_ENABLE);
    int    outXform  = getInt(ps, PARAM_OUTPUT_TRANSFORM);
    int    inXform   = getInt(ps, PARAM_INPUT_TRANSFORM);
    // Film Emulation look + embedded LUT (loaded via preset) are the only look paths
    InstanceData* id = getInstanceData(effect);
    bool   filmEmulActive = getBool(ps, PARAM_SF_LOOK_ENABLE)
                            && (getInt(ps, PARAM_SF_NEG_STOCK) > 0);
    bool   anyLutActive   = filmEmulActive
                            || (id && id->hasEmbeddedLut);

    if (timelineDiagEnabled()) {
        std::fprintf(stderr,
                     "[LoFiFxSpectra] IsIdentity t=%.3f grain=%d blurEnable=%d opacity=%.6f blur=%.6f lut=%d hal=%d diff=%d split=%d print=%d inXform=%d outXform=%d\n",
                     time, grainOn ? 1 : 0, filmBlurOn ? 1 : 0, opacity,
                     filmResolveAmount, anyLutActive ? 1 : 0, halOn ? 1 : 0,
                     diffusionOn ? 1 : 0, splitOn ? 1 : 0, printOn ? 1 : 0,
                     inXform, outXform);
        return kOfxStatReplyDefault;
    }

    if (opacity < 1e-6 && filmResolveAmount < 0.05 && !anyLutActive && !halOn && !diffusionOn && !splitOn && !printOn && outXform == 0 && inXform == 0) {
        gPropHost->propSetString(outArgs, kOfxPropName, 0,
                                 kOfxImageEffectSimpleSourceClipName);
        return kOfxStatOK;
    }

    return kOfxStatReplyDefault;
}

// ============================================================
// ACTION: Render
// ============================================================

// Forward declarations — defined after actionRender (CPU helpers used to
// compute the contrast curve preview samples passed to the GPU).
static void cpuLut3D(const float* data, int sz,
                     float inR, float inG, float inB,
                     float& outR, float& outG, float& outB);
static void cpuSplitTone(float& r, float& g, float& b,
                         float strength, float shHue, float hiHue, float loHi,
                         float pivot, float subLo, float subHi, float neutW,
                         float filmDensity, float kmStrength, float hiProtect,
                         float dyeSat, float crossBleed);
static void cpuRgbToHsl(float r, float g, float b,
                        float& h, float& s, float& l);
static void cpuHslToRgb(float h, float s, float l,
                        float& r, float& g, float& b);

static OfxStatus actionRender(OfxImageEffectHandle effect,
                              OfxPropertySetHandle inArgs)
{
    // ---- Time ----
    double time = 0;
    gPropHost->propGetDouble(inArgs, kOfxPropTime, 0, &time);

    // ---- Render scale (proxy / scrub awareness) ----
    // OFX hosts call render at fractional scale for proxy timelines, viewer
    // scrubbing and thumbnails. Every parameter expressed in pixels must be
    // multiplied by this factor so the visual result is invariant to the
    // host's chosen render resolution.
    double rsX = 1.0, rsY = 1.0;
    gPropHost->propGetDouble(inArgs, kOfxImageEffectPropRenderScale, 0, &rsX);
    gPropHost->propGetDouble(inArgs, kOfxImageEffectPropRenderScale, 1, &rsY);
    float renderScale = (float)(0.5 * (rsX + rsY));
    if (renderScale < 1e-3f) renderScale = 1.0f; // hosts that don't set it

    // ---- Parameter values ----
    OfxParamSetHandle ps;
    gEffectHost->getParamSet(effect, &ps);

    GrainParams gp;
    memset(&gp, 0, sizeof(gp));
    gp.renderScale = renderScale;
    gp.debugMode = proofWriteEnabled() ? 1 : 0;

    bool grainOn = getBool(ps, PARAM_GRAIN_ENABLE);
    bool filmBlurOn = grainOn && getBool(ps, PARAM_IMAGE_BLUR_ENABLE);

    gp.grain_size      = (float)getDouble(ps, PARAM_GRAIN_SIZE, time);
    gp.grain_strength  = grainOn ? (float)getDouble(ps, PARAM_GRAIN_STRENGTH, time) : 0.0f;
    gp.grain_blur      = (float)getDouble(ps, PARAM_GRAIN_BLUR, time);
    gp.grain_opacity   = grainOn ? (float)getDouble(ps, PARAM_GRAIN_OPACITY, time) : 0.0f;
    gp.grain_sat       = (float)getDouble(ps, PARAM_GRAIN_SAT, time);
    gp.image_blur      = 0.0f;  // base is always 0; grain_blur_link (Film Blur) drives amount
    gp.hi_rolloff      = (float)getDouble(ps, PARAM_HI_ROLLOFF, time);
    gp.shape_irregularity = (float)getDouble(ps, PARAM_SHAPE_IRREG, time);
    gp.grain_blur_link = filmBlurOn ? (float)getDouble(ps, PARAM_GRAIN_BLUR_LINK, time) : 0.0f;
    gp.grain_contrast  = (float)getDouble(ps, PARAM_GRAIN_CONTRAST, time);
    gp.seed_offset     = getInt(ps, PARAM_SEED_OFFSET);
    gp.frame_index     = (int)time;

    // Halation params
    bool halOn = getBool(ps, PARAM_HAL_ENABLE);
    gp.halEnabled  = halOn ? 1 : 0;
    gp.halSize     = (float)getDouble(ps, PARAM_HAL_SIZE, time);
    gp.halColor    = (float)getDouble(ps, PARAM_HAL_COLOR, time);
    gp.halMethod   = (float)getInt(ps, PARAM_HAL_METHOD);
    gp._halReserved0 = 0.0f;
    gp._halReserved1 = 0.0f;
    gp._halReserved2 = 0.0f;
    gp._halReserved3 = 0.0f;
    gp._halReserved4 = 0.0f;
    gp._halReserved5 = 0.0f;
    gp.halOpacity  = halOn ? (float)getDouble(ps, PARAM_HAL_OPACITY, time) : 0.0f;

    // Diffusion params
    bool diffusionOn = getBool(ps, PARAM_DIFFUSION_ENABLE);
    gp.diffusionEnabled     = diffusionOn ? 1 : 0;
    gp.diffusionAmount      = diffusionOn ? (float)getDouble(ps, PARAM_DIFFUSION_AMOUNT, time) : 0.0f;
    gp.diffusionSize      = (float)getDouble(ps, PARAM_DIFFUSION_SIZE, time);
    gp.diffusionSoftness  = (float)getDouble(ps, PARAM_DIFFUSION_SOFTNESS, time);
    gp.diffusionBodyStrength     = (float)getDouble(ps, PARAM_DIFFUSION_BODY_STRENGTH, time);
    gp.diffusionFalloffStrength = (float)getDouble(ps, PARAM_DIFFUSION_FALLOFF_STRENGTH, time);
    gp.diffusionHighlightGate   = (float)getDouble(ps, PARAM_DIFFUSION_HIGHLIGHT_GATE, time);
    gp.diffusionVeil               = diffusionOn ? (float)getDouble(ps, PARAM_DIFFUSION_VEIL, time) : 0.0f;
    gp.diffusionContrastReduction  = diffusionOn ? (float)getDouble(ps, PARAM_DIFFUSION_CONTRAST, time) : 0.0f;
    gp.diffusionHighlightBoost     = diffusionOn ? (float)getDouble(ps, PARAM_DIFFUSION_HIGHLIGHT_BOOST, time) : 0.0f;
    gp.diffusionTintStrength = diffusionOn ? (float)getDouble(ps, PARAM_DIFFUSION_TINT_STRENGTH, time) : 0.0f;
    double diffusionTintR, diffusionTintG, diffusionTintB;
    getRGB(ps, PARAM_DIFFUSION_TINT_COLOR, time, diffusionTintR, diffusionTintG, diffusionTintB);
    gp.diffusionTintR = (float)diffusionTintR;
    gp.diffusionTintG = (float)diffusionTintG;
    gp.diffusionTintB = (float)diffusionTintB;
    gp.diffusionScatterMode = 0;
    gp.diffusionScatterWidth = 0;
    gp.diffusionScatterHeight = 0;

    // PSF family: derived from diffusion profile selection.
    // Mapping: BPM 1/8,1/4,1/2 → BPM(1); Glimmerglass→0; BPM 1x→1; Pro-Mist→2; CineBloom→3; Custom→0
    {
        static const int   kProfileToFamily[4]  = { 1, 1, 1, 0 };
        static const float kFamilyWarmthBase[4] = { 0.0f, 0.65f, 0.40f, 0.85f };
        int diffProfile = getInt(ps, PARAM_DIFFUSION_PROFILE);
        diffProfile = (diffProfile < 0) ? 0 : (diffProfile > 3 ? 3 : diffProfile);
        gp.diffusionPsfFamily  = kProfileToFamily[diffProfile];
        float warmthOffset = diffusionOn ? gp.diffusionTintStrength : 0.0f;
        gp.diffusionHaloWarmth = kFamilyWarmthBase[gp.diffusionPsfFamily] + warmthOffset;
    }

    // Split toning params
    bool splitOn = getBool(ps, PARAM_SPLIT_ENABLE);
    gp.splitEnabled      = splitOn ? 1 : 0;
    gp.splitStrength     = splitOn ? (float)getDouble(ps, PARAM_SPLIT_STRENGTH, time) : 0.0f;
    gp.splitShadowHue    = (float)getDouble(ps, PARAM_SPLIT_SHADOW_HUE, time);
    gp.splitHighlightHue = (float)getDouble(ps, PARAM_SPLIT_HI_HUE, time);
    gp.splitLoVsHi       = (float)getDouble(ps, PARAM_SPLIT_LO_VS_HI, time);
    gp.splitPivot        = (float)getDouble(ps, PARAM_SPLIT_PIVOT, time);
    gp.splitSubLo        = (float)getDouble(ps, PARAM_SPLIT_SUB_LO, time);
    gp.splitSubHi        = (float)getDouble(ps, PARAM_SPLIT_SUB_HI, time);
    gp.splitNeutralWidth = (float)getDouble(ps, PARAM_SPLIT_NEUTRAL_W, time);
    gp.splitFilmDensity  = (float)getDouble(ps, PARAM_SPLIT_FILM_DENSITY, time);
    gp.splitKMStrength   = (float)getDouble(ps, PARAM_SPLIT_KM_STRENGTH, time);
    gp.splitHiProtect    = (float)getDouble(ps, PARAM_SPLIT_HI_PROTECT, time);
    gp.splitDyeSat       = (float)getDouble(ps, PARAM_SPLIT_DYE_SAT, time);
    gp.splitCrossBleed   = (float)getDouble(ps, PARAM_SPLIT_CROSS_BLEED, time);

    // Printer points: classic printer-point exposure trim.
    // One printer point is modeled as 1/12 stop, matching the common lab /
    // color-timer convention. Higher values darken that channel, mirroring the
    // subtractive print result of increasing that printer light.
    bool printOn = getBool(ps, PARAM_PRINT_ENABLE);
    gp.printEnabled = printOn ? 1 : 0;
    if (printOn) {
        double pr = getDouble(ps, PARAM_PRINT_R, time);
        double pg = getDouble(ps, PARAM_PRINT_G, time);
        double pb = getDouble(ps, PARAM_PRINT_B, time);
        gp.printGainR = (float)std::exp2(-(pr - 25.0) / 12.0);
        gp.printGainG = (float)std::exp2(-(pg - 25.0) / 12.0);
        gp.printGainB = (float)std::exp2(-(pb - 25.0) / 12.0);
    } else {
        gp.printGainR = 1.0f;
        gp.printGainG = 1.0f;
        gp.printGainB = 1.0f;
    }

    // Output colour space transform (LUT-based)
    int outXform = getInt(ps, PARAM_OUTPUT_TRANSFORM);
    gp.outTransform = outXform;
    gp.cstLutSize   = 0;  // default: no CST

    // Film Emulation (spectral film) params
    bool   sfLookEnabled   = getBool(ps, PARAM_SF_LOOK_ENABLE);
    float  sfLookSaturation = (float)getDouble(ps, PARAM_SF_LOOK_SATURATION, time);
    float  sfContrastCurve  = (float)getDouble(ps, PARAM_SF_CONTRAST_CURVE, time);
    double sfWpKelvin      = normalizeWpKelvin(getDouble(ps, PARAM_SF_WP_KELVIN, time));

    // ---- Fetch source and output images ----
    OfxImageClipHandle srcClip, dstClip;
    gEffectHost->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName,
                               &srcClip, nullptr);
    gEffectHost->clipGetHandle(effect, kOfxImageEffectOutputClipName,
                               &dstClip, nullptr);

    OfxPropertySetHandle srcImg = nullptr, dstImg = nullptr;
    gEffectHost->clipGetImage(srcClip, time, nullptr, &srcImg);
    gEffectHost->clipGetImage(dstClip, time, nullptr, &dstImg);

    if (!srcImg || !dstImg) {
        if (srcImg) gEffectHost->clipReleaseImage(srcImg);
        if (dstImg) gEffectHost->clipReleaseImage(dstImg);
        return kOfxStatFailed;
    }

    // ---- Image dimensions / pointers ----
    void* srcPtr = nullptr;  void* dstPtr = nullptr;
    int srcRowBytes = 0;     int dstRowBytes = 0;
    OfxRectI srcBounds, dstBounds;

    gPropHost->propGetPointer(srcImg, kOfxImagePropData, 0, &srcPtr);
    gPropHost->propGetPointer(dstImg, kOfxImagePropData, 0, &dstPtr);
    gPropHost->propGetInt(srcImg, kOfxImagePropRowBytes, 0, &srcRowBytes);
    gPropHost->propGetInt(dstImg, kOfxImagePropRowBytes, 0, &dstRowBytes);
    gPropHost->propGetIntN(srcImg, kOfxImagePropBounds, 4, &srcBounds.x1);
    gPropHost->propGetIntN(dstImg, kOfxImagePropBounds, 4, &dstBounds.x1);

    gp.width       = dstBounds.x2 - dstBounds.x1;
    gp.height      = dstBounds.y2 - dstBounds.y1;
    gp.srcWidth    = std::max(1, srcBounds.x2 - srcBounds.x1);
    gp.srcHeight   = std::max(1, srcBounds.y2 - srcBounds.y1);
    gp.srcX1       = srcBounds.x1;
    gp.srcY1       = srcBounds.y1;
    gp.dstX1       = dstBounds.x1;
    gp.dstY1       = dstBounds.y1;
    gp.srcRowBytes = srcRowBytes;
    gp.dstRowBytes = dstRowBytes;

    if (timelineDiagEnabled()) {
        std::fprintf(stderr,
                     "[LoFiFxSpectra] Render t=%.3f scale=%.4f debug=%d srcImg=%p dstImg=%p srcData=%p dstData=%p srcRB=%d dstRB=%d srcBounds=(%d,%d)-(%d,%d) dstBounds=(%d,%d)-(%d,%d) grain=%d strength=%.6f opacity=%.6f blurEnable=%d blur=%.6f link=%.6f\n",
                     time, gp.renderScale, gp.debugMode, (void*)srcImg, (void*)dstImg,
                     srcPtr, dstPtr, srcRowBytes, dstRowBytes,
                     srcBounds.x1, srcBounds.y1, srcBounds.x2, srcBounds.y2,
                     dstBounds.x1, dstBounds.y1, dstBounds.x2, dstBounds.y2,
                     grainOn ? 1 : 0, gp.grain_strength, gp.grain_opacity,
                     filmBlurOn ? 1 : 0, gp.image_blur, gp.grain_blur_link);
    }

    // ---- Resolution / proxy normalisation -----------------------------------
    // Two corrections are applied to every pixel-unit parameter:
    //   1. renderScale  — host may render at proxy/scrub resolution. All px
    //                     values shrink proportionally so the look is invariant.
    //   2. resRefScale  — Film-stock parameters (grain_size / grain_blur) are
    //                     authored against a 4K (2160p) reference frame so the
    //                     "Medium 35mm" preset reads the same on a 1080p or
    //                     8K timeline. Other px params (image_blur,
    //                     diffusionSize, weave) are left in absolute output
    //                     pixels — those sliders are user-facing px controls.
    // gp.height here is the proxy height; full-res height is gp.height/rs.
    // -------------------------------------------------------------------------
    {
        float rs        = gp.renderScale;
        float fullHeight = (float)gp.height / std::max(rs, 1e-3f);
        float resRef    = std::max(fullHeight / 2160.0f, 0.01f);

        // Dynamic film blur coupled to grain size (a film-stock property):
        // larger grain implies lower resolving power, so the base plate loses
        // detail nonlinearly as the film gauge gets smaller. Computed from the
        // authored pre-scale grain cell and normalised like other film-stock px
        // values, so the link is resolution-invariant.
        float coupledBlur = 0.0f;
        if (filmBlurOn) {
            coupledBlur = linkedFilmBlurSigma(gp.grain_size, gp.grain_blur_link);
        }

        // Film-stock: scale by 4K reference AND render scale.
        gp.grain_size *= resRef * rs;
        gp.grain_blur *= resRef * rs;

        // Film blur is a film-stock property (resolution loss), so it scales
        // with both 4K reference AND render scale — same as grain_size.
        gp.image_blur  *= resRef * rs;
        gp.image_blur  += coupledBlur * resRef * rs;
        gp.diffusionSize  *= rs;
    }

    // ---- Look setup ----
    InstanceData* instData = getInstanceData(effect);
    void* lutBufPtr = nullptr;

    // Embedded LUT from preset takes priority over Film Emulation
    bool usedEmbedded = false;
    if (instData && instData->hasEmbeddedLut && instData->embeddedLut.size > 0)
    {
        const CubeLut& lut = instData->embeddedLut;
        gp.lutEnabled     = 1;
        gp.lutSize        = lut.size;
        gp.lutIntensity   = 1.0f;
        gp.lutContrastAmt = 1.0f;
        gp.lutPushPull    = 0.0f;
        // Adapt from the LUT's native white point to D65
        float wpMatrix[9];
        gp.wpEnabled = bradfordCat(lut.nativeWP, 3, wpMatrix) ? 1 : 0;
        if (gp.wpEnabled)
            for (int i = 0; i < 9; i++) gp.wpMat[i] = wpMatrix[i];

        if (!instData->embeddedLutUploaded) {
            instData->embeddedLutUploaded = true;
#ifdef __APPLE__
            {
                void* queuePtr = nullptr;
                int metalEn = 0;
                gPropHost->propGetInt(inArgs, kOfxImageEffectPropMetalEnabled,
                                     0, &metalEn);
                if (metalEn)
                    gPropHost->propGetPointer(inArgs,
                                              kOfxImageEffectPropMetalCommandQueue,
                                              0, &queuePtr);
                if (!queuePtr) queuePtr = metalGetSelfHostedQueue();
                if (queuePtr) {
                    if (instData->metalEmbeddedLutBuffer)
                        metalFreeLut(instData->metalEmbeddedLutBuffer);
                    instData->metalEmbeddedLutBuffer = metalUploadLut(
                        queuePtr, lut.data.data(), lut.size);
                }
            }
#endif
        }

        lutBufPtr = instData->metalEmbeddedLutBuffer;
        if (!lutBufPtr) lutBufPtr = instData->cudaEmbeddedLutBuffer;
        usedEmbedded = true;
    }

    // ---- Film Emulation (Spectral Film) pipeline ----
    if (!usedEmbedded) {
        // Load stock data on demand (once globally)
        if (gSfNegStocks.empty()) {
            std::string bp = getBundlePath(effect);
            loadSfStockData(bp.empty() ? gBundlePath : bp);
        }

        int negChoice = getInt(ps, PARAM_SF_NEG_STOCK);
        int prtChoice = getInt(ps, PARAM_SF_PRT_STOCK);

        int negIdx = (negChoice >= 0 && negChoice < kSfNegChoicesN) ? kSfNegChoices[negChoice].idx : -1;
        int prtIdx = (prtChoice >= 0 && prtChoice < kSfPrtChoicesN) ? kSfPrtChoices[prtChoice].idx : -1;

        if (negIdx >= 0 && instData) {
            bool lutDirty = (instData->sfLutCacheNeg != negChoice ||
                             instData->sfLutCachePrt != prtChoice);

            if (lutDirty) {
                std::lock_guard<std::mutex> lock(gSfStockMutex);
                SfLutKey key{negIdx, prtIdx};
                auto it = gSfFilmLuts.find(key);
                if (it != gSfFilmLuts.end()) {
                    convertSfLutToDwgDi(it->second, instData->sfConvertedLut);
                    instData->sfLutCacheNeg = negChoice;
                    instData->sfLutCachePrt = prtChoice;
                    if (instData->metalSfLutBuffer) {
                        metalFreeLut(instData->metalSfLutBuffer);
                        instData->metalSfLutBuffer = nullptr;
                    }
                } else {
                    instData->sfConvertedLut.clear();
                    instData->sfLutCacheNeg = -99;
                }
            }

            if (!instData->sfConvertedLut.empty()) {
                gp.lutEnabled     = sfLookEnabled ? 1 : 0;
                gp.lutSize        = SF_LUT3D_SIZE;
                gp.lutIntensity   = sfLookSaturation;
                gp.lutContrastAmt = sfContrastCurve;
                gp.lutPushPull    = 0.0f;

                if (sfLookEnabled) {
                    float wpMatrix[9];
                    gp.wpEnabled = bradfordCatKelvin(3, sfWpKelvin, wpMatrix) ? 1 : 0;
                    if (gp.wpEnabled)
                        for (int i = 0; i < 9; i++) gp.wpMat[i] = wpMatrix[i];
                } else {
                    gp.wpEnabled = 0;
                }

#ifdef __APPLE__
                if (!instData->metalSfLutBuffer) {
                    void* queuePtr = nullptr;
                    int metalEnabledSf = 0;
                    gPropHost->propGetInt(inArgs, kOfxImageEffectPropMetalEnabled,
                                          0, &metalEnabledSf);
                    if (metalEnabledSf)
                        gPropHost->propGetPointer(inArgs,
                                                   kOfxImageEffectPropMetalCommandQueue,
                                                   0, &queuePtr);
                    if (!queuePtr) queuePtr = metalGetSelfHostedQueue();
                    if (queuePtr) {
                        instData->metalSfLutBuffer = metalUploadLut(
                            queuePtr,
                            instData->sfConvertedLut.data(),
                            SF_LUT3D_SIZE);
                    }
                }
#endif
                lutBufPtr = instData->metalSfLutBuffer;
            } else {
                gp.lutEnabled     = 0;
                gp.lutSize        = 0;
                gp.lutPushPull    = 0.0f;
                gp.lutContrastAmt = 1.0f;
                gp.wpEnabled      = 0;
            }
        } else {
            gp.lutEnabled     = 0;
            gp.lutSize        = 0;
            gp.lutPushPull    = 0.0f;
            gp.lutContrastAmt = 1.0f;
            gp.wpEnabled      = 0;
        }
    }

    // ---- Output CST LUT setup ----
    void* outputCstBufPtr = nullptr;
    if (instData && outXform > 0 && (outXform - 1) < (int)instData->outputCsts.size()) {
        int cstIdx = outXform - 1;
        const CubeLut& cst = instData->outputCsts[cstIdx];
        gp.cstLutSize = cst.size;

        if (instData->currentOutputCstIdx != cstIdx) {
            instData->currentOutputCstIdx = cstIdx;
#ifdef __APPLE__
            {
                void* queuePtr = nullptr;
                int metalEnabledCst = 0;
                gPropHost->propGetInt(inArgs, kOfxImageEffectPropMetalEnabled,
                                     0, &metalEnabledCst);
                if (metalEnabledCst)
                    gPropHost->propGetPointer(inArgs,
                                              kOfxImageEffectPropMetalCommandQueue,
                                              0, &queuePtr);
                if (!queuePtr) queuePtr = metalGetSelfHostedQueue();
                if (queuePtr) {
                    if (instData->metalOutputCstBuffer)
                        metalFreeLut(instData->metalOutputCstBuffer);
                    instData->metalOutputCstBuffer = metalUploadLut(
                        queuePtr, cst.data.data(), cst.size);
                }
            }
#endif
        }

        outputCstBufPtr = instData->metalOutputCstBuffer;
        if (!outputCstBufPtr) outputCstBufPtr = instData->cudaOutputCstBuffer;
    }

    // ---- Input CST LUT setup ----
    int inXform = getInt(ps, PARAM_INPUT_TRANSFORM);
    gp.inTransform     = inXform;
    gp.inputCstLutSize = 0;  // default: no input CST
    void* inputCstBufPtr = nullptr;
    if (instData && inXform > 0 && (inXform - 1) < (int)instData->inputCsts.size()) {
        int cstIdx = inXform - 1;
        const CubeLut& cst = instData->inputCsts[cstIdx];
        gp.inputCstLutSize = cst.size;

        if (instData->currentInputCstIdx != cstIdx) {
            instData->currentInputCstIdx = cstIdx;
#ifdef __APPLE__
            {
                void* queuePtr = nullptr;
                int metalEnabledCst = 0;
                gPropHost->propGetInt(inArgs, kOfxImageEffectPropMetalEnabled,
                                     0, &metalEnabledCst);
                if (metalEnabledCst)
                    gPropHost->propGetPointer(inArgs,
                                              kOfxImageEffectPropMetalCommandQueue,
                                              0, &queuePtr);
                if (!queuePtr) queuePtr = metalGetSelfHostedQueue();
                if (queuePtr) {
                    if (instData->metalInputCstBuffer)
                        metalFreeLut(instData->metalInputCstBuffer);
                    instData->metalInputCstBuffer = metalUploadLut(
                        queuePtr, cst.data.data(), cst.size);
                }
            }
#endif
        }

        inputCstBufPtr = instData->metalInputCstBuffer;
        if (!inputCstBufPtr) inputCstBufPtr = instData->cudaInputCstBuffer;
    }

    // ---- Compute contrast curve samples (256 R + 256 G + 256 B) ----
    // GPU kernel reads this buffer if gp.curveEnabled is set.
    bool curveOn = false; // getBool(ps, PARAM_CURVE_ENABLE); // hidden/disabled for now
    gp.curveEnabled = 0;
    float curveBuf[768];
    if (curveOn) {
        // Read all curve-affecting params
        bool splitOn  = getBool(ps, PARAM_SPLIT_ENABLE);
        float spStr   = splitOn ? (float)getDouble(ps, PARAM_SPLIT_STRENGTH, time) : 0.0f;
        float spShHue = (float)getDouble(ps, PARAM_SPLIT_SHADOW_HUE, time);
        float spHiHue = (float)getDouble(ps, PARAM_SPLIT_HI_HUE, time);
        float spLoHi  = (float)getDouble(ps, PARAM_SPLIT_LO_VS_HI, time);
        float spPivot = (float)getDouble(ps, PARAM_SPLIT_PIVOT, time);
        float spSubLo = (float)getDouble(ps, PARAM_SPLIT_SUB_LO, time);
        float spSubHi = (float)getDouble(ps, PARAM_SPLIT_SUB_HI, time);
        float spNeutW = (float)getDouble(ps, PARAM_SPLIT_NEUTRAL_W, time);
        float spFilmDensity = (float)getDouble(ps, PARAM_SPLIT_FILM_DENSITY, time);
        float spKMStr       = (float)getDouble(ps, PARAM_SPLIT_KM_STRENGTH, time);
        float spHiProt      = (float)getDouble(ps, PARAM_SPLIT_HI_PROTECT, time);
        float spDyeSat      = (float)getDouble(ps, PARAM_SPLIT_DYE_SAT, time);
        float spCrossBleed  = (float)getDouble(ps, PARAM_SPLIT_CROSS_BLEED, time);

        bool  cLutOn  = (instData && instData->hasEmbeddedLut);
        float cLutInt = 1.0f;
        float cLutPP  = 0.0f;

        auto cpuDiToLinear = [](float value) -> float {
            constexpr float kA = 0.0075f;
            constexpr float kB = 7.0f;
            constexpr float kC = 0.07329248f;
            constexpr float kM = 10.44426855f;
            constexpr float kLogCut = 0.02740668f;
            return (value > kLogCut) ? std::exp2(value / kC - kB) - kA : value / kM;
        };
        auto cpuLinearToDi = [](float value) -> float {
            constexpr float kA = 0.0075f;
            constexpr float kB = 7.0f;
            constexpr float kC = 0.07329248f;
            constexpr float kM = 10.44426855f;
            constexpr float kLinCut = 0.00262409f;
            return (value > kLinCut) ? kC * (std::log2(value + kA) + kB) : value * kM;
        };
        auto cpuApplyPrinterLights = [&](float value, float exposureGain) -> float {
            // Match Metal kernel: toe-protected per-channel exposure trim.
            // Zero effect at deep DI black, full effect by mid-gray (DI ~0.34).
            float gained = cpuLinearToDi(cpuDiToLinear(value) * exposureGain);
            float t = (value - 0.10f) / (0.34f - 0.10f);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            float w = t * t * (3.0f - 2.0f * t);
            return value + (gained - value) * w;
        };

        bool  printOn = getBool(ps, PARAM_PRINT_ENABLE);
        float pgR = 1.0f, pgG = 1.0f, pgB = 1.0f;
        if (printOn) {
            double pr = getDouble(ps, PARAM_PRINT_R, time);
            double pg = getDouble(ps, PARAM_PRINT_G, time);
            double pb = getDouble(ps, PARAM_PRINT_B, time);
            pgR = (float)std::exp2(-(pr - 25.0) / 12.0);
            pgG = (float)std::exp2(-(pg - 25.0) / 12.0);
            pgB = (float)std::exp2(-(pb - 25.0) / 12.0);
        }

        const float* cLutData = nullptr;
        int cLutSize = 0;
        int cLutNativeWP = 3;
        if (cLutOn && instData->embeddedLut.size > 0) {
            cLutData = instData->embeddedLut.data.data();
            cLutSize = instData->embeddedLut.size;
            cLutNativeWP = instData->embeddedLut.nativeWP;
        }
        bool cWpOn = false;
        float cWpMat[9];
        if (cLutOn && cLutData)
            cWpOn = bradfordCat(cLutNativeWP, 3, cWpMat);

        for (int i = 0; i < 256; i++) {
            float v = (float)i / 255.0f;
            float r = v, g = v, b = v;

            if (splitOn && spStr > 1e-6f)
                cpuSplitTone(r, g, b, spStr, spShHue, spHiHue, spLoHi,
                             spPivot, spSubLo, spSubHi, spNeutW,
                             spFilmDensity, spKMStr, spHiProt, spDyeSat, spCrossBleed);

            // Printer points BEFORE the Look LUT (photochemical chain order).
            if (printOn) {
                r = cpuApplyPrinterLights(r, pgR);
                g = cpuApplyPrinterLights(g, pgG);
                b = cpuApplyPrinterLights(b, pgB);
            }

            if (cLutOn && cLutData && cLutSize > 0) {
                float lr, lg, lb;
                cpuLut3D(cLutData, cLutSize, r, g, b, lr, lg, lb);
                if (cWpOn) {
                    float tr = cWpMat[0]*lr + cWpMat[1]*lg + cWpMat[2]*lb;
                    float tg = cWpMat[3]*lr + cWpMat[4]*lg + cWpMat[5]*lb;
                    float tb = cWpMat[6]*lr + cWpMat[7]*lg + cWpMat[8]*lb;
                    lr = tr; lg = tg; lb = tb;
                }
                r += (lr - r) * cLutInt;
                g += (lg - g) * cLutInt;
                b += (lb - b) * cLutInt;
                if (fabsf(cLutPP) > 1e-6f) {
                    float gain = powf(2.0f, cLutPP);
                    r *= gain; g *= gain; b *= gain;
                    float h, s, l;
                    cpuRgbToHsl(r, g, b, h, s, l);
                    s = fminf(fmaxf(s + cLutPP * 0.1f, 0.0f), 1.0f);
                    cpuHslToRgb(h, s, l, r, g, b);
                }
            }

            curveBuf[      i] = fminf(fmaxf(r, 0.0f), 1.0f);
            curveBuf[256 + i] = fminf(fmaxf(g, 0.0f), 1.0f);
            curveBuf[512 + i] = fminf(fmaxf(b, 0.0f), 1.0f);
        }
    } else {
        memset(curveBuf, 0, sizeof(curveBuf));
    }

    // ---- GPU dispatch ----
    OfxStatus status = kOfxStatFailed;

#ifdef __APPLE__
    {
        int metalEnabled = 0;
        gPropHost->propGetInt(inArgs, kOfxImageEffectPropMetalEnabled,
                             0, &metalEnabled);
        if (metalEnabled) {
            void* queuePtr = nullptr;
            gPropHost->propGetPointer(inArgs,
                                      kOfxImageEffectPropMetalCommandQueue,
                                      0, &queuePtr);
            if (queuePtr && metalInit(queuePtr)) {
                metalDispatchGrain(queuePtr, srcPtr, dstPtr,
                                   lutBufPtr, outputCstBufPtr, inputCstBufPtr,
                                   gp, curveBuf, true);
                status = kOfxStatOK;
            }
        }
    }
#endif

#ifndef __APPLE__
    if (status != kOfxStatOK) {
        int cudaEnabled = 0;
        gPropHost->propGetInt(inArgs, kOfxImageEffectPropCudaEnabled,
                             0, &cudaEnabled);
        if (cudaEnabled) {
            void* streamPtr = nullptr;
            gPropHost->propGetPointer(inArgs, kOfxImageEffectPropCudaStream,
                                      0, &streamPtr);
            cudaDispatchGrain(streamPtr, srcPtr, dstPtr,
                              lutBufPtr, outputCstBufPtr, inputCstBufPtr, gp);
            status = kOfxStatOK;
        }
    }
#endif

    // GPU-only plugin — always attempt Metal.
    // If the host did not enable Metal for this render, run via a self-hosted
    // queue: CPU buffers are wrapped in shared MTLBuffers, the full grain
    // kernel executes on GPU, and the result is copied back to the CPU
    // destination before returning.
#ifdef __APPLE__
    if (status != kOfxStatOK && srcPtr && dstPtr) {
        if (metalDispatchGrainFromCpu(srcPtr, dstPtr,
                                      lutBufPtr, outputCstBufPtr, inputCstBufPtr,
                                      gp, curveBuf)) {
            status = kOfxStatOK;
        }
    }
#endif

    // ---- Release images ----
    gEffectHost->clipReleaseImage(srcImg);
    gEffectHost->clipReleaseImage(dstImg);

    return status;
}

// ============================================================
// Overlay: CPU-side pipeline functions for contrast curve
// ============================================================

// CPU tetrahedral 3D LUT interpolation (mirrors Metal applyLut3D)
static void cpuLut3D(const float* data, int sz,
                     float inR, float inG, float inB,
                     float& outR, float& outG, float& outB)
{
    float sr = fminf(fmaxf(inR, 0.0f), 1.0f) * (float)(sz - 1);
    float sg = fminf(fmaxf(inG, 0.0f), 1.0f) * (float)(sz - 1);
    float sb = fminf(fmaxf(inB, 0.0f), 1.0f) * (float)(sz - 1);
    int ri = std::min((int)sr, sz - 2);
    int gi = std::min((int)sg, sz - 2);
    int bi = std::min((int)sb, sz - 2);
    float fr = sr - (float)ri, fg = sg - (float)gi, fb = sb - (float)bi;

    int s2 = sz * sz;
    auto idx = [&](int r, int g, int b) { return (b*s2 + g*sz + r) * 3; };
    int i000 = idx(ri, gi, bi);
    int i111 = idx(ri+1, gi+1, bi+1);

    int iA, iB;
    float w0, wA, wB, w1;
    if (fr > fg) {
        if (fg > fb) {                          // T1: r > g > b
            iA = idx(ri+1,gi,bi);   iB = idx(ri+1,gi+1,bi);
            w0=1-fr; wA=fr-fg; wB=fg-fb; w1=fb;
        } else if (fr > fb) {                   // T2: r > b > g
            iA = idx(ri+1,gi,bi);   iB = idx(ri+1,gi,bi+1);
            w0=1-fr; wA=fr-fb; wB=fb-fg; w1=fg;
        } else {                                // T3: b > r > g
            iA = idx(ri,gi,bi+1);   iB = idx(ri+1,gi,bi+1);
            w0=1-fb; wA=fb-fr; wB=fr-fg; w1=fg;
        }
    } else {
        if (fb > fg) {                          // T4: b > g > r
            iA = idx(ri,gi,bi+1);   iB = idx(ri,gi+1,bi+1);
            w0=1-fb; wA=fb-fg; wB=fg-fr; w1=fr;
        } else if (fb > fr) {                   // T5: g > b > r
            iA = idx(ri,gi+1,bi);   iB = idx(ri,gi+1,bi+1);
            w0=1-fg; wA=fg-fb; wB=fb-fr; w1=fr;
        } else {                                // T6: g > r > b
            iA = idx(ri,gi+1,bi);   iB = idx(ri+1,gi+1,bi);
            w0=1-fg; wA=fg-fr; wB=fr-fb; w1=fb;
        }
    }
    outR = w0*data[i000]   + wA*data[iA]   + wB*data[iB]   + w1*data[i111];
    outG = w0*data[i000+1] + wA*data[iA+1] + wB*data[iB+1] + w1*data[i111+1];
    outB = w0*data[i000+2] + wA*data[iA+2] + wB*data[iB+2] + w1*data[i111+2];
}

static float cpuSplitCurve(float t) {
    float s = fminf(fmaxf(t, 0.0f), 1.0f);
    return s * s * s * (s * (s * 6.0f - 15.0f) + 10.0f);
}

static float cpuDiToLinear(float value) {
    constexpr float kA = 0.0075f;
    constexpr float kB = 7.0f;
    constexpr float kC = 0.07329248f;
    constexpr float kM = 10.44426855f;
    constexpr float kLogCut = 0.02740668f;
    return (value > kLogCut) ? std::exp2(value / kC - kB) - kA : value / kM;
}

static float cpuLinearToDi(float value) {
    constexpr float kA = 0.0075f;
    constexpr float kB = 7.0f;
    constexpr float kC = 0.07329248f;
    constexpr float kM = 10.44426855f;
    constexpr float kLinCut = 0.00262409f;
    return (value > kLinCut) ? kC * (std::log2(value + kA) + kB) : value * kM;
}

static float cpuSplitDwgYLinear(float r, float g, float b) {
    return 0.2741335367f * r + 0.8735913831f * g - 0.1477249199f * b;
}

static float cpuSignedCuberoot(float value) {
    return (value < 0.0f) ? -powf(-value, 0.3333333333f) : powf(value, 0.3333333333f);
}

static void cpuDwgToXyz(float r, float g, float b,
                        float& x, float& y, float& z) {
    x = 0.7006607968f * r + 0.1487679160f * g + 0.1010412872f * b;
    y = 0.2741335367f * r + 0.8735913831f * g - 0.1477249199f * b;
    z = -0.0989683375f * r - 0.1378889305f * g + 1.3256872680f * b;
}

static void cpuXyzToDwg(float x, float y, float z,
                        float& r, float& g, float& b) {
    r = 1.5165889100f * x - 0.2814626195f * y - 0.1469555778f * z;
    g = -0.4649386618f * x + 1.2514818105f * y + 0.1748927191f * z;
    b = 0.0648602355f * x + 0.1091581735f * y + 0.7615459883f * z;
}

static void cpuDwgLinearToOklab(float r, float g, float b,
                                float& L, float& A, float& B) {
    float x, y, z;
    cpuDwgToXyz(r, g, b, x, y, z);
    float l = 0.8189330101f * x + 0.3618667424f * y - 0.1288597137f * z;
    float m = 0.0329845436f * x + 0.9293118715f * y + 0.0361456387f * z;
    float s = 0.0482003018f * x + 0.2643662691f * y + 0.6338517070f * z;

    float lp = cpuSignedCuberoot(l);
    float mp = cpuSignedCuberoot(m);
    float sp = cpuSignedCuberoot(s);

    L = 0.2104542553f * lp + 0.7936177850f * mp - 0.0040720468f * sp;
    A = 1.9779984951f * lp - 2.4285922050f * mp + 0.4505937099f * sp;
    B = 0.0259040371f * lp + 0.7827717662f * mp - 0.8086757660f * sp;
}

static void cpuOklabToDwgLinear(float L, float A, float B,
                                float& r, float& g, float& b) {
    float lp = L + 0.3963377774f * A + 0.2158037573f * B;
    float mp = L - 0.1055613458f * A - 0.0638541728f * B;
    float sp = L - 0.0894841775f * A - 1.2914855480f * B;

    float l = lp * lp * lp;
    float m = mp * mp * mp;
    float s = sp * sp * sp;

    float x = 1.2270138511f * l - 0.5577999807f * m + 0.2812561490f * s;
    float y = -0.0405801784f * l + 1.1122568696f * m - 0.0716766787f * s;
    float z = -0.0763812845f * l - 0.4214819784f * m + 1.5861632204f * s;
    cpuXyzToDwg(x, y, z, r, g, b);
}

static void cpuHueToOpponent(float hueDeg, float& A, float& B) {
    float h = fmodf(hueDeg, 360.0f);
    if (h < 0.0f) h += 360.0f;
    float angle = h * 0.01745329252f;
    A = cosf(angle);
    B = sinf(angle);
}

static void cpuPreserveDwgY(float originalR, float originalG, float originalB,
                            float& gradedR, float& gradedG, float& gradedB) {
    float targetY = cpuSplitDwgYLinear(originalR, originalG, originalB);
    float gradedY = cpuSplitDwgYLinear(gradedR, gradedG, gradedB);
    if (targetY > 1e-6f && gradedY > 1e-6f) {
        float gain = fminf(fmaxf(targetY / gradedY, 0.25f), 4.0f);
        gradedR *= gain;
        gradedG *= gain;
        gradedB *= gain;
    }
    float neutralTrim = targetY - cpuSplitDwgYLinear(gradedR, gradedG, gradedB);
    gradedR += neutralTrim;
    gradedG += neutralTrim;
    gradedB += neutralTrim;
}

static void cpuSplitTone(float& r, float& g, float& b,
                          float strength, float shadowHue, float highlightHue,
                          float loVsHi, float pivot, float subLo, float subHi,
                          float neutralWidth, float filmDensity, float kmStrength,
                          float hiProtect, float dyeSat, float crossBleed)
{
    float linR = cpuDiToLinear(r);
    float linG = cpuDiToLinear(g);
    float linB = cpuDiToLinear(b);
    float lumDi = cpuLinearToDi(fmaxf(cpuSplitDwgYLinear(linR, linG, linB), 0.0f));

    float pvt = fminf(fmaxf(pivot, 0.001f), 0.999f);
    float halfN = fminf(fmaxf(neutralWidth, 0.0f), 0.98f) * 0.5f;
    float shadowEdge = fmaxf(pvt - halfN, 0.0f);
    float highEdge = fminf(pvt + halfN, 1.0f);

    float shadowRaw = (shadowEdge - lumDi) / fmaxf(shadowEdge, 1e-5f);
    float highRaw = (lumDi - highEdge) / fmaxf(1.0f - highEdge, 1e-5f);
    float toeGuard = cpuSplitCurve((lumDi - 0.035f) / (0.12f - 0.035f));
    float shadowMask = cpuSplitCurve(shadowRaw) * toeGuard;
    float highMask = cpuSplitCurve(highRaw);

    float loAmt = fminf(fmaxf((1.0f - loVsHi) * 2.0f, 0.0f), 1.0f);
    float hiAmt = fminf(fmaxf(loVsHi * 2.0f, 0.0f), 1.0f);
    float lowWeight = shadowMask * loAmt;
    float highWeight = highMask * hiAmt;
    if (lowWeight + highWeight <= 1e-6f) return;

    float labL, labA, labB;
    cpuDwgLinearToOklab(linR, linG, linB, labL, labA, labB);
    float chroma = sqrtf(labA * labA + labB * labB);
    float materialProtect = 1.0f - 0.45f * cpuSplitCurve((chroma - 0.06f) / 0.28f);

    // Per-zone Beer-Lambert concentrations
    float fd = fminf(fmaxf(filmDensity, 0.1f), 3.0f);
    float str = fminf(fmaxf(strength, 0.0f), 4.0f);
    float sLo = fminf(fmaxf(subLo, 0.0f), 1.0f);
    float sHi = fminf(fmaxf(subHi, 0.0f), 1.0f);
    float shadowConc = fd * sLo * str * lowWeight;
    float highConc = fd * sHi * str * highWeight;

    float shadowOp = (1.0f - expf(-shadowConc)) * materialProtect;
    float hiProtectMask = 1.0f - fminf(fmaxf(hiProtect, 0.0f), 1.0f) *
                          cpuSplitCurve((lumDi - 0.70f) / 0.30f);
    float highOp = (1.0f - expf(-highConc)) * materialProtect * hiProtectMask;

    // Dye vectors with cross-zone bleed
    float shA, shB, hiA, hiB;
    cpuHueToOpponent(shadowHue, shA, shB);
    cpuHueToOpponent(highlightHue, hiA, hiB);
    float bl = fminf(fmaxf(crossBleed, 0.0f), 0.5f);
    float shBlendA = shA + bl * hiA;
    float shBlendB = shB + bl * hiB;
    float hiBlendA = hiA + bl * shA;
    float hiBlendB = hiB + bl * shB;
    float shLen = fmaxf(sqrtf(shBlendA * shBlendA + shBlendB * shBlendB), 1e-6f);
    float hiLen = fmaxf(sqrtf(hiBlendA * hiBlendA + hiBlendB * hiBlendB), 1e-6f);
    shBlendA /= shLen; shBlendB /= shLen;
    hiBlendA /= hiLen; hiBlendB /= hiLen;

    // Dye saturation-controlled target chroma
    float ds = fminf(fmaxf(dyeSat, 0.1f), 2.0f);
    float shadowTargetC = fminf(0.105f * ds, fmaxf(chroma * 0.45f, 0.018f + labL * 0.16f));
    float highTargetC = fminf(0.150f * ds, fmaxf(chroma * 0.50f, 0.028f + labL * 0.20f));

    labA = labA + (shBlendA * shadowTargetC - labA) * shadowOp;
    labB = labB + (shBlendB * shadowTargetC - labB) * shadowOp;
    labA = labA + (hiBlendA * highTargetC - labA) * highOp;
    labB = labB + (hiBlendB * highTargetC - labB) * highOp;

    // Kubelka-Munk hue drift
    float kmStr = fminf(fmaxf(kmStrength, 0.0f), 1.0f);
    if (kmStr > 1e-4f) {
        float curChroma = sqrtf(labA * labA + labB * labB);
        if (curChroma > 1e-4f) {
            float kmDrift = kmStr * 0.20f * (shadowConc - highConc);
            float angle = atan2f(labB, labA) + kmDrift;
            labA = cosf(angle) * curChroma;
            labB = sinf(angle) * curChroma;
        }
    }

    float tonedR, tonedG, tonedB;
    cpuOklabToDwgLinear(labL, labA, labB, tonedR, tonedG, tonedB);
    cpuPreserveDwgY(linR, linG, linB, tonedR, tonedG, tonedB);

    float origMin = fminf(linR, fminf(linG, linB));
    float tonedMin = fminf(tonedR, fminf(tonedG, tonedB));
    float floorMin = fminf(origMin, -0.02f);
    if (tonedMin < floorMin) {
        float denom = tonedMin - origMin;
        if (denom < -1e-6f) {
            float fit = fminf(fmaxf((floorMin - origMin) / denom, 0.0f), 1.0f);
            tonedR = linR + (tonedR - linR) * fit;
            tonedG = linG + (tonedG - linG) * fit;
            tonedB = linB + (tonedB - linB) * fit;
        }
    }

    r = cpuLinearToDi(tonedR);
    g = cpuLinearToDi(tonedG);
    b = cpuLinearToDi(tonedB);
}

// CPU RGB→HSL (matches Metal rgb_to_hsl)
static void cpuRgbToHsl(float r, float g, float b,
                         float& h, float& s, float& l) {
    float maxC = fmaxf(fmaxf(r, g), b);
    float minC = fminf(fminf(r, g), b);
    float delta = maxC - minC;
    l = (maxC + minC) * 0.5f;
    s = (delta < 1e-6f) ? 0.0f :
        (l < 0.5f ? delta / (maxC + minC) : delta / (2.0f - maxC - minC));
    h = 0.0f;
    if (delta > 1e-6f) {
        if (maxC == r)      h = (g - b) / delta + (g < b ? 6.0f : 0.0f);
        else if (maxC == g) h = (b - r) / delta + 2.0f;
        else                h = (r - g) / delta + 4.0f;
        h /= 6.0f;
    }
}

static float cpuHslHueToRgb(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f/6.0f)  return p + (q - p) * 6.0f * t;
    if (t < 0.5f)        return q;
    if (t < 2.0f/3.0f)  return p + (q - p) * (2.0f/3.0f - t) * 6.0f;
    return p;
}

static void cpuHslToRgb(float h, float s, float l,
                         float& r, float& g, float& b) {
    if (s < 1e-6f) { r = g = b = l; return; }
    float q = (l < 0.5f) ? l * (1.0f + s) : l + s - l * s;
    float p = 2.0f * l - q;
    r = cpuHslHueToRgb(p, q, h + 1.0f/3.0f);
    g = cpuHslHueToRgb(p, q, h);
    b = cpuHslHueToRgb(p, q, h - 1.0f/3.0f);
}

// ============================================================
// Main entry point dispatcher
// ============================================================
static OfxStatus pluginMain(const char* action, const void* handle,
                            OfxPropertySetHandle inArgs,
                            OfxPropertySetHandle outArgs)
{
    OfxImageEffectHandle effect = (OfxImageEffectHandle)handle;

    if (strcmp(action, kOfxActionLoad) == 0)
        return actionLoad();
    if (strcmp(action, kOfxActionUnload) == 0) {
#ifdef __APPLE__
        metalRelease();
#endif
        cudaRelease();
        return kOfxStatOK;
    }
    if (strcmp(action, kOfxActionDescribe) == 0)
        return actionDescribe(effect);
    if (strcmp(action, kOfxImageEffectActionDescribeInContext) == 0)
        return actionDescribeInContext(effect, inArgs);
    if (strcmp(action, kOfxActionCreateInstance) == 0)
        return actionCreateInstance(effect);
    if (strcmp(action, kOfxActionDestroyInstance) == 0)
        return actionDestroyInstance(effect);
    if (strcmp(action, kOfxActionInstanceChanged) == 0)
        return actionInstanceChanged(effect, inArgs);
    if (strcmp(action, kOfxImageEffectActionIsIdentity) == 0)
        return actionIsIdentity(effect, inArgs, outArgs);
    if (strcmp(action, kOfxImageEffectActionRender) == 0)
        return actionRender(effect, inArgs);

    return kOfxStatReplyDefault;
}

// ============================================================
// setHost callback
// ============================================================
static void setHost(OfxHost* host) {
    gHost = host;
}

// ============================================================
// Plugin struct
// ============================================================
static OfxPlugin pluginStruct = {
    /* pluginApi          */ kOfxImageEffectPluginApi,
    /* apiVersion         */ 1,
    /* pluginIdentifier   */ PLUGIN_ID,
    /* pluginVersionMajor */ PLUGIN_VERSION_MAJOR,
    /* pluginVersionMinor */ PLUGIN_VERSION_MINOR,
    /* setHost            */ setHost,
    /* mainEntry          */ pluginMain
};

// ============================================================
// Exported bootstrap functions
// ============================================================
extern "C" {

int OfxGetNumberOfPlugins(void) {
    return 1;
}

OfxPlugin* OfxGetPlugin(int nth) {
    if (nth == 0) return &pluginStruct;
    return nullptr;
}

} // extern "C"
