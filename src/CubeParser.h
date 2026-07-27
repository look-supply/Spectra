#ifndef CUBE_PARSER_H
#define CUBE_PARSER_H

// ============================================================
// Iridas / Adobe .cube 3D Look parser  (header-only)
//
// Supports:
//   - LUT_3D_SIZE
//   - DOMAIN_MIN / DOMAIN_MAX
//   - TITLE
//   - # comment lines
//   - Blank lines
//
// Stores data as a flat float array: R,G,B triples in
// B-major, G-middle, R-minor order (standard .cube ordering):
//   index = (b * size * size + g * size + r) * 3
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dirent.h>
#include <algorithm>

struct CubeLut {
    std::string title;
    std::string filename;
    std::string folder;        // subfolder name (empty for root)
    int   size;            // dimension (e.g. 65)
    float domainMin[3];
    float domainMax[3];
    std::vector<float> data;  // size^3 * 3 floats (RGB triples)
    int   nativeWP;       // native white point: 0=D50, 1=D55, 2=D60, 3=D65, 4=D70, 5=D75

    CubeLut() : size(0), nativeWP(3) {
        domainMin[0] = domainMin[1] = domainMin[2] = 0.0f;
        domainMax[0] = domainMax[1] = domainMax[2] = 1.0f;
    }
};

// Parse a single .cube file. Returns true on success.
inline bool parseCubeFile(const char* path, CubeLut& lut) {
    FILE* f = fopen(path, "r");
    if (!f) return false;

    lut = CubeLut();
    lut.size = 0;

    char line[512];
    int expectedEntries = 0;
    int readEntries = 0;

    while (fgets(line, sizeof(line), f)) {
        // Strip trailing whitespace / newline
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r' ||
                           line[len-1] == ' '  || line[len-1] == '\t'))
            line[--len] = '\0';

        // Skip empty lines and comments
        if (len == 0 || line[0] == '#')
            continue;

        // TITLE
        if (strncmp(line, "TITLE", 5) == 0) {
            const char* p = line + 5;
            while (*p == ' ' || *p == '\t') ++p;
            if (*p == '"') ++p;
            const char* end = p + strlen(p);
            if (end > p && *(end-1) == '"') --end;
            lut.title.assign(p, end);
            continue;
        }

        // LUT_3D_SIZE
        if (strncmp(line, "LUT_3D_SIZE", 11) == 0) {
            lut.size = atoi(line + 11);
            if (lut.size < 2 || lut.size > 256) {
                fclose(f);
                return false;
            }
            expectedEntries = lut.size * lut.size * lut.size;
            lut.data.resize((size_t)expectedEntries * 3);
            continue;
        }

        // DOMAIN_MIN
        if (strncmp(line, "DOMAIN_MIN", 10) == 0) {
            sscanf(line + 10, " %f %f %f",
                   &lut.domainMin[0], &lut.domainMin[1], &lut.domainMin[2]);
            continue;
        }

        // DOMAIN_MAX
        if (strncmp(line, "DOMAIN_MAX", 10) == 0) {
            sscanf(line + 10, " %f %f %f",
                   &lut.domainMax[0], &lut.domainMax[1], &lut.domainMax[2]);
            continue;
        }

        // LUT_1D_SIZE — skip 1D LUTs
        if (strncmp(line, "LUT_1D_SIZE", 11) == 0) {
            fclose(f);
            return false;
        }

        // Data line: three floats
        if (lut.size > 0 && readEntries < expectedEntries) {
            float r, g, b;
            if (sscanf(line, "%f %f %f", &r, &g, &b) == 3) {
                size_t idx = (size_t)readEntries * 3;
                lut.data[idx + 0] = r;
                lut.data[idx + 1] = g;
                lut.data[idx + 2] = b;
                ++readEntries;
            }
        }
    }

    fclose(f);

    if (lut.size == 0 || readEntries != expectedEntries)
        return false;

    // Always derive title from filename (strip path and .cube extension).
    // Embedded TITLE fields are often generic and not useful for display.
    {
        std::string p(path);
        size_t slash = p.find_last_of('/');
        if (slash == std::string::npos) slash = p.find_last_of('\\');
        std::string base = (slash != std::string::npos) ? p.substr(slash + 1) : p;
        size_t dot = base.find_last_of('.');
        if (dot != std::string::npos) base = base.substr(0, dot);
        lut.title = base;
    }

    return true;
}

// Discover all .cube files in a directory and its immediate subdirectories.
// Files in subdirectories are grouped by folder name (stored in CubeLut::folder).
// Returns vector of parsed LUTs, sorted by folder then filename.
inline std::vector<CubeLut> discoverLuts(const char* dirPath) {
    std::vector<CubeLut> luts;

    // Collect (folder, filename) pairs from root and subdirectories
    struct LutEntry {
        std::string folder;   // "" for root
        std::string path;     // full path to .cube file
        std::string sortKey;  // "folder/filename" for sorting
    };
    std::vector<LutEntry> entries;

    DIR* dir = opendir(dirPath);
    if (!dir) return luts;

    struct dirent* entry;
    std::vector<std::string> subdirs;

    while ((entry = readdir(dir)) != nullptr) {
        std::string name(entry->d_name);
        if (name == "." || name == "..") continue;

        std::string fullPath = std::string(dirPath) + "/" + name;

        // Check if it's a directory
        DIR* subdir = opendir(fullPath.c_str());
        if (subdir) {
            subdirs.push_back(name);
            closedir(subdir);
            continue;
        }

        // Root-level .cube file
        size_t len = name.size();
        if (len > 5 && name.substr(len - 5) == ".cube") {
            LutEntry e;
            e.folder  = "";
            e.path    = fullPath;
            e.sortKey = name;
            entries.push_back(std::move(e));
        }
    }
    closedir(dir);

    // Scan subdirectories
    std::sort(subdirs.begin(), subdirs.end());
    for (const auto& sub : subdirs) {
        std::string subPath = std::string(dirPath) + "/" + sub;
        DIR* sd = opendir(subPath.c_str());
        if (!sd) continue;

        while ((entry = readdir(sd)) != nullptr) {
            std::string name(entry->d_name);
            size_t len = name.size();
            if (len > 5 && name.substr(len - 5) == ".cube") {
                LutEntry e;
                e.folder  = sub;
                e.path    = subPath + "/" + name;
                e.sortKey = sub + "/" + name;
                entries.push_back(std::move(e));
            }
        }
        closedir(sd);
    }

    // Sort by folder then filename
    std::sort(entries.begin(), entries.end(),
              [](const LutEntry& a, const LutEntry& b) {
                  return a.sortKey < b.sortKey;
              });

    // Parse each file
    for (const auto& e : entries) {
        CubeLut lut;
        if (parseCubeFile(e.path.c_str(), lut)) {
            lut.filename = e.path.substr(e.path.find_last_of('/') + 1);
            lut.folder   = e.folder;
            // Load companion .meta file (same stem, .meta extension)
            // Format:  name: Human Readable Name
            //          wp: D65         (optional, default D65)
            {
                std::string metaPath = e.path.substr(0, e.path.size() - 5) + ".meta";
                lut.nativeWP = 3; // default D65
                FILE* mf = fopen(metaPath.c_str(), "r");
                if (mf) {
                    char buf[256] = {};
                    while (fgets(buf, sizeof(buf), mf)) {
                        std::string line(buf);
                        while (!line.empty() && (line.back() == '\n' || line.back() == '\r' ||
                                                  line.back() == ' '  || line.back() == '\t'))
                            line.pop_back();
                        if (line.substr(0, 6) == "name: ") {
                            lut.title = line.substr(6);
                        } else if (line.substr(0, 4) == "wp: ") {
                            std::string s = line.substr(4);
                            if      (s == "D50") lut.nativeWP = 0;
                            else if (s == "D55") lut.nativeWP = 1;
                            else if (s == "D60") lut.nativeWP = 2;
                            else if (s == "D65") lut.nativeWP = 3;
                            else if (s == "D70") lut.nativeWP = 4;
                            else if (s == "D75") lut.nativeWP = 5;
                        }
                    }
                    fclose(mf);
                }
            }
            luts.push_back(std::move(lut));
        }
    }

    return luts;
}

#endif // CUBE_PARSER_H
