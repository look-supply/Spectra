#!/usr/bin/env python3
"""
Build-time script: packs all LUT (.cube + .meta) files into a single
obfuscated binary blob (resources.pak).

Format:
  Magic:      4 bytes  "LFPK"
  Version:    4 bytes  uint32-LE (1)
  NumEntries: 4 bytes  uint32-LE
  --- XOR-encrypted payload follows ---
  For each entry:
    category:   1 byte  (0=Look, 1=CST_Input, 2=CST_Output)
    nativeWP:   1 byte  (0-5)
    lutSize:    4 bytes uint32-LE  (e.g. 33 or 65)
    titleLen:   2 bytes uint16-LE
    title:      titleLen bytes UTF-8
    folderLen:  2 bytes uint16-LE
    folder:     folderLen bytes UTF-8
    numFloats:  4 bytes uint32-LE
    floatData:  numFloats * 4 bytes (float32-LE)
"""

import os
import struct
import sys
import glob

# --- Hardcoded XOR key (must match C++ unpacker) ---
XOR_KEY = bytes([
    0xA7, 0x3B, 0xF1, 0x5D, 0x82, 0xC6, 0x49, 0xE0,
    0x1F, 0x94, 0x6A, 0xD3, 0x58, 0xBB, 0x07, 0x7E,
    0xC1, 0x36, 0xAD, 0xE9, 0x4B, 0x00, 0x73, 0xF8,
    0x25, 0x9C, 0x61, 0xDA, 0x8F, 0x44, 0xB2, 0x17,
])


def xor_encrypt(data: bytes) -> bytes:
    key = XOR_KEY
    klen = len(key)
    return bytes(b ^ key[i % klen] for i, b in enumerate(data))


def parse_cube_file(path):
    """Parse a .cube file, return (size, floats_list) or None."""
    size = 0
    floats = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.upper().startswith('LUT_3D_SIZE'):
                size = int(line.split()[-1])
                continue
            if line.upper().startswith(('TITLE', 'DOMAIN_MIN', 'DOMAIN_MAX',
                                        'LUT_1D_SIZE', 'LUT_1D_INPUT_RANGE',
                                        'LUT_3D_INPUT_RANGE')):
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    floats.extend(float(x) for x in parts)
                except ValueError:
                    continue
    if size == 0:
        return None
    expected = size * size * size * 3
    if len(floats) != expected:
        print(f"  WARNING: {path}: expected {expected} floats, got {len(floats)}", file=sys.stderr)
        return None
    return size, floats


def parse_meta_file(meta_path):
    """Parse companion .meta file. Returns (title, nativeWP)."""
    title = ""
    native_wp = 3  # default D65
    wp_map = {"D50": 0, "D55": 1, "D60": 2, "D65": 3, "D70": 4, "D75": 5}
    if os.path.isfile(meta_path):
        with open(meta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("name: "):
                    title = line[6:]
                elif line.startswith("wp: "):
                    wp_str = line[4:].strip()
                    native_wp = wp_map.get(wp_str, 3)
    return title, native_wp


def discover_luts(directory, category):
    """Discover .cube files in a directory (with optional subdirs for Looks).
    Returns list of (category, folder, title, nativeWP, size, floats)."""
    entries = []
    if not os.path.isdir(directory):
        return entries

    # Root-level cubes
    for cube_path in sorted(glob.glob(os.path.join(directory, '*.cube'))):
        result = parse_cube_file(cube_path)
        if result is None:
            continue
        size, floats = result
        meta_path = cube_path[:-5] + '.meta'
        title, native_wp = parse_meta_file(meta_path)
        if not title:
            title = os.path.splitext(os.path.basename(cube_path))[0]
        entries.append((category, "", title, native_wp, size, floats))

    # Subdirectory cubes (for Looks)
    subdirs = sorted([
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d)) and not d.startswith('.')
    ])
    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        for cube_path in sorted(glob.glob(os.path.join(subdir_path, '*.cube'))):
            result = parse_cube_file(cube_path)
            if result is None:
                continue
            size, floats = result
            meta_path = cube_path[:-5] + '.meta'
            title, native_wp = parse_meta_file(meta_path)
            if not title:
                title = os.path.splitext(os.path.basename(cube_path))[0]
            entries.append((category, subdir, title, native_wp, size, floats))

    return entries


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <LUTs_source_dir> <output.pak>", file=sys.stderr)
        sys.exit(1)

    luts_dir = sys.argv[1]
    output_path = sys.argv[2]

    all_entries = []

    # Discover CST categories only (commercial looks removed)
    cst_input_dir = os.path.join(luts_dir, 'CST', 'Input')
    cst_output_dir = os.path.join(luts_dir, 'CST', 'Output')

    all_entries.extend(discover_luts(cst_input_dir, 1))    # 1 = CST Input
    all_entries.extend(discover_luts(cst_output_dir, 2))   # 2 = CST Output

    print(f"Packing {len(all_entries)} LUTs into {output_path}")
    for cat, folder, title, wp, size, _ in all_entries:
        cat_name = ["Look", "CST/In", "CST/Out"][cat]
        print(f"  [{cat_name}] {title} (WP={wp}, {size}x{size}x{size})"
              + (f" [{folder}]" if folder else ""))

    # Build the payload (to be XOR'd)
    payload = bytearray()
    for category, folder, title, native_wp, size, floats in all_entries:
        title_bytes = title.encode('utf-8')
        folder_bytes = folder.encode('utf-8')

        payload.append(category & 0xFF)
        payload.append(native_wp & 0xFF)
        payload.extend(struct.pack('<I', size))
        payload.extend(struct.pack('<H', len(title_bytes)))
        payload.extend(title_bytes)
        payload.extend(struct.pack('<H', len(folder_bytes)))
        payload.extend(folder_bytes)

        num_floats = len(floats)
        payload.extend(struct.pack('<I', num_floats))
        for f in floats:
            payload.extend(struct.pack('<f', f))

    encrypted = xor_encrypt(bytes(payload))

    # Write file
    with open(output_path, 'wb') as f:
        f.write(b'LFPK')                                  # magic
        f.write(struct.pack('<I', 1))                      # version
        f.write(struct.pack('<I', len(all_entries)))        # num entries
        f.write(encrypted)                                 # encrypted payload

    file_size = os.path.getsize(output_path)
    print(f"Wrote {file_size} bytes ({file_size / 1024:.1f} KB)")


if __name__ == '__main__':
    main()
