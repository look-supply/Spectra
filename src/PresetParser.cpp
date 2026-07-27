#include "PresetParser.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

// ---- trim whitespace ----
static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// ---- derive a display name from filename (strip extension, capitalise) ----
static std::string nameFromFilename(const std::string& fn) {
    std::string base = fn;
    size_t dot = base.rfind('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    if (!base.empty()) base[0] = (char)toupper((unsigned char)base[0]);
    return base;
}

// ---- parse a single .preset file ----
static MasterPreset parsePresetFile(const std::string& path,
                                    const std::string& filename,
                                    const std::string& sourceDir) {
    MasterPreset p;
    p.filename = filename;
    p.sourceDir = sourceDir;
    p.name = nameFromFilename(filename);

    std::ifstream f(path);
    if (!f.is_open()) return p;

    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(line.substr(0, eq));
        std::string valStr = trim(line.substr(eq + 1));

        // Strip inline comments (# after value)
        size_t hash = valStr.find('#');
        if (hash != std::string::npos)
            valStr = trim(valStr.substr(0, hash));

        if (key.empty() || valStr.empty()) continue;

        // Special key: "name" sets the display name
        if (key == "name") {
            p.name = valStr;
            continue;
        }

        // Embedded LUT string keys
        if (key == "lut_embedded") {
            p.hasEmbeddedLut = (valStr == "1");
            p.values[key] = (valStr == "1") ? 1.0 : 0.0;
            continue;
        }
        if (key == "lut_embedded_path") {
            p.embeddedLutPath = valStr;
            continue;
        }
        if (key == "lut_embedded_title") {
            p.embeddedLutTitle = valStr;
            continue;
        }
        if (key == "lut_embedded_wp") {
            p.embeddedLutWp = atoi(valStr.c_str());
            continue;
        }

        // All other keys store as double
        char* end = nullptr;
        double v = strtod(valStr.c_str(), &end);
        if (end != valStr.c_str())
            p.values[key] = v;
    }

    return p;
}

// ---- discover all .preset files in directory ----
std::vector<MasterPreset> discoverPresets(const char* dirPath) {
    std::vector<MasterPreset> result;
    if (!dirPath || !dirPath[0]) return result;

    std::vector<std::string> files;

#ifdef _WIN32
    std::string pattern = std::string(dirPath) + "\\*.preset";
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            files.push_back(fd.cFileName);
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }
#else
    DIR* dir = opendir(dirPath);
    if (!dir) return result;
    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        std::string fn = ent->d_name;
        if (fn.size() > 7 &&
            fn.substr(fn.size() - 7) == ".preset") {
            files.push_back(fn);
        }
    }
    closedir(dir);
#endif

    // Sort alphabetically by filename
    std::sort(files.begin(), files.end());

    for (const auto& fn : files) {
        std::string fullPath = std::string(dirPath) + "/" + fn;
        MasterPreset p = parsePresetFile(fullPath, fn, dirPath);
        if (!p.values.empty() || p.hasEmbeddedLut)
            result.push_back(std::move(p));
    }

    return result;
}
