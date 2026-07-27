#ifndef LUT_PACK_H
#define LUT_PACK_H

// ============================================================
// LutPack.h — Unpacks obfuscated LUT binary blob (resources.pak)
//
// Format: see pack_luts.py for the binary layout.
// Data is XOR-encrypted with a hardcoded key.
// ============================================================

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

// Matches CubeLut from CubeParser.h
struct PackedLut {
    int         category;   // 0=Look, 1=CST_Input, 2=CST_Output
    std::string title;
    std::string folder;     // subfolder (Looks only)
    int         nativeWP;   // 0=D50..5=D75
    int         size;       // cube dimension (e.g. 33, 65)
    std::vector<float> data;
};

// XOR key — must match pack_luts.py
static const uint8_t kLutPackKey[] = {
    0xA7, 0x3B, 0xF1, 0x5D, 0x82, 0xC6, 0x49, 0xE0,
    0x1F, 0x94, 0x6A, 0xD3, 0x58, 0xBB, 0x07, 0x7E,
    0xC1, 0x36, 0xAD, 0xE9, 0x4B, 0x00, 0x73, 0xF8,
    0x25, 0x9C, 0x61, 0xDA, 0x8F, 0x44, 0xB2, 0x17,
};
static const size_t kLutPackKeyLen = sizeof(kLutPackKey);

static inline void xorDecrypt(uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i)
        data[i] ^= kLutPackKey[i % kLutPackKeyLen];
}

// Read helpers for little-endian packed data
static inline uint16_t readU16(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static inline uint32_t readU32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline float readF32(const uint8_t* p) {
    float f;
    memcpy(&f, p, 4);
    return f;
}

// Load all LUTs from a packed .pak file
inline std::vector<PackedLut> loadLutPack(const char* pakPath) {
    std::vector<PackedLut> result;

    FILE* f = fopen(pakPath, "rb");
    if (!f) return result;

    // Read entire file
    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fileSize < 12) { fclose(f); return result; }

    std::vector<uint8_t> buf((size_t)fileSize);
    if (fread(buf.data(), 1, (size_t)fileSize, f) != (size_t)fileSize) {
        fclose(f);
        return result;
    }
    fclose(f);

    // Verify magic
    if (memcmp(buf.data(), "LFPK", 4) != 0) return result;

    uint32_t version    = readU32(buf.data() + 4);
    uint32_t numEntries = readU32(buf.data() + 8);
    (void)version;

    // Decrypt payload (everything after the 12-byte header)
    size_t payloadLen = (size_t)fileSize - 12;
    xorDecrypt(buf.data() + 12, payloadLen);

    // Parse entries
    const uint8_t* p   = buf.data() + 12;
    const uint8_t* end = buf.data() + (size_t)fileSize;

    result.reserve(numEntries);
    for (uint32_t i = 0; i < numEntries; ++i) {
        if (p + 8 > end) break;  // safety

        PackedLut lut;
        lut.category = (int)p[0];
        lut.nativeWP = (int)p[1];
        lut.size     = (int)readU32(p + 2);
        p += 6;

        // Title
        if (p + 2 > end) break;
        uint16_t titleLen = readU16(p); p += 2;
        if (p + titleLen > end) break;
        lut.title.assign((const char*)p, titleLen); p += titleLen;

        // Folder
        if (p + 2 > end) break;
        uint16_t folderLen = readU16(p); p += 2;
        if (p + folderLen > end) break;
        lut.folder.assign((const char*)p, folderLen); p += folderLen;

        // Float data
        if (p + 4 > end) break;
        uint32_t numFloats = readU32(p); p += 4;
        size_t dataBytes = (size_t)numFloats * 4;
        if (p + dataBytes > end) break;

        lut.data.resize(numFloats);
        memcpy(lut.data.data(), p, dataBytes);
        p += dataBytes;

        result.push_back(std::move(lut));
    }

    return result;
}

// ============================================================
// Convert PackedLut → CubeLut (from CubeParser.h)
// ============================================================
#include "CubeParser.h"

inline CubeLut packedToCubeLut(PackedLut&& packed) {
    CubeLut lut;
    lut.title    = std::move(packed.title);
    lut.folder   = std::move(packed.folder);
    lut.size     = packed.size;
    lut.nativeWP = packed.nativeWP;
    lut.data     = std::move(packed.data);
    return lut;
}

// Load and split by category — returns three vectors filling
// looks, inputCsts, outputCsts (in CubeLut form).
inline void loadLutPackSplit(const char* pakPath,
                             std::vector<CubeLut>& inputCsts,
                             std::vector<CubeLut>& outputCsts) {
    auto all = loadLutPack(pakPath);
    for (auto& p : all) {
        int cat = p.category;
        CubeLut c = packedToCubeLut(std::move(p));
        switch (cat) {
            case 1:  inputCsts.push_back(std::move(c));  break;
            case 2:  outputCsts.push_back(std::move(c)); break;
            default: break;  // skip commercial look LUTs (category 0)
        }
    }
}

#endif // LUT_PACK_H
