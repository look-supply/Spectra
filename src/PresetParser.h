#ifndef PRESET_PARSER_H
#define PRESET_PARSER_H

#include <map>
#include <string>
#include <vector>

// A single master preset: display name + key→value map.
struct MasterPreset {
    std::string name;        // Display name (from "name" key, or filename)
    std::string filename;    // Original filename for sorting
    std::string sourceDir;   // Directory the preset was loaded from
    std::map<std::string, double> values;

    // Optional embedded LUT data (for presets that carry their own .cube)
    bool        hasEmbeddedLut   = false;
    std::string embeddedLutPath;  // Filename of companion .cube (relative to sourceDir)
    std::string embeddedLutTitle; // Display title override
    int         embeddedLutWp    = 3;  // Native white point: 0=D50..5=D75
};

// Discover all .preset files in a directory, sorted alphabetically
// by filename. Returns empty vector if dir does not exist.
std::vector<MasterPreset> discoverPresets(const char* dirPath);

#endif // PRESET_PARSER_H
