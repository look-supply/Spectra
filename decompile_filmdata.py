#!/usr/bin/env python3
"""Decompile spectral_film_stocks.bin into transparent individual files.

Output:
  Resources/spectral_film_data/stocks/  — JSON files per stock (readable)
  Resources/spectral_film_data/luts/    — Raw .float32 files per LUT

The original .bin can be recreated with:
  python3 pack_filmdata.py Resources/spectral_film_data/stocks Resources/spectral_film_data/luts Resources/spectral_film_stocks.bin
"""

import json
import os
import struct
import sys

SF_N_WAVEL = 81
SF_N_BANDS = 27
SF_LUT3D_SIZE = 33
SF_CURVE_N = 512
SF_NAME_LEN = 64
SF_LUT_FLOATS = SF_LUT3D_SIZE ** 3 * 3


def read_cstr(data, offset, max_len):
    end = data.find(b'\0', offset, offset + max_len)
    if end < 0:
        end = offset + max_len
    return data[offset:end].decode('utf-8', errors='replace')


def read_float(data, offset):
    return struct.unpack_from('<f', data, offset)[0]


def read_floats(data, offset, count):
    return list(struct.unpack_from(f'<{count}f', data, offset))


def make_safe_name(name):
    s = name.replace('/', '_').replace(' ', '_').replace('.', '_')
    return ''.join(c for c in s if c.isalnum() or c == '_')


def decompile(bin_path, stocks_dir, luts_dir):
    os.makedirs(stocks_dir, exist_ok=True)
    os.makedirs(luts_dir, exist_ok=True)

    with open(bin_path, 'rb') as f:
        data = f.read()

    off = 0
    magic = data[off:off+8]; off += 8
    assert magic == b'SPFILM\0\0', f"Bad magic: {magic}"
    _version = struct.unpack_from('<I', data, off)[0]; off += 4
    n_neg = struct.unpack_from('<I', data, off)[0]; off += 4
    n_prt = struct.unpack_from('<I', data, off)[0]; off += 4
    print(f"Negatives: {n_neg}, Prints: {n_prt}")

    neg_names = []
    for i in range(n_neg):
        rec_size = struct.unpack_from('<I', data, off)[0]; off += 4
        rec_start = off
        s = {}
        name = read_cstr(data, off, SF_NAME_LEN)
        s['name'] = name
        neg_names.append(name)
        off += SF_NAME_LEN

        s['inputLut'] = read_floats(data, off, SF_LUT3D_SIZE * SF_LUT3D_SIZE * 3)
        off += SF_LUT3D_SIZE * SF_LUT3D_SIZE * 3 * 4

        s['logHMin'] = read_float(data, off); off += 4
        s['logHMax'] = read_float(data, off); off += 4

        s['curve'] = read_floats(data, off, 3 * SF_CURVE_N)
        off += 3 * SF_CURVE_N * 4

        s['dRef'] = read_floats(data, off, 3); off += 12
        s['dMin'] = read_floats(data, off, 3); off += 12

        s['specDens27'] = read_floats(data, off, SF_N_BANDS * 3)
        off += SF_N_BANDS * 3 * 4
        s['dMinSd27'] = read_floats(data, off, SF_N_BANDS)
        off += SF_N_BANDS * 4

        s['dRefSdFull'] = read_floats(data, off, SF_N_WAVEL); off += SF_N_WAVEL * 4
        s['dMinSdFull'] = read_floats(data, off, SF_N_WAVEL); off += SF_N_WAVEL * 4
        s['specDensFull'] = read_floats(data, off, SF_N_WAVEL * 3)
        off += SF_N_WAVEL * 3 * 4
        s['sensitivityFull'] = read_floats(data, off, SF_N_WAVEL * 3)
        off += SF_N_WAVEL * 3 * 4
        s['colorMasking'] = read_float(data, off); off += 4

        bytes_read = off - rec_start
        if bytes_read < rec_size: off += (rec_size - bytes_read)

        safe = make_safe_name(name)
        fp = os.path.join(stocks_dir, f"neg_{i:02d}_{safe}.json")
        with open(fp, 'w') as f:
            json.dump(s, f, indent=2)
        print(f"  [neg {i:2d}] {name}")

    for i in range(n_prt):
        rec_size = struct.unpack_from('<I', data, off)[0]; off += 4
        rec_start = off
        s = {}
        name = read_cstr(data, off, SF_NAME_LEN)
        s['name'] = name
        off += SF_NAME_LEN

        s['logHMin'] = read_float(data, off); off += 4
        s['logHMax'] = read_float(data, off); off += 4
        s['curve'] = read_floats(data, off, 3 * SF_CURVE_N)
        off += 3 * SF_CURVE_N * 4
        s['dRef'] = read_floats(data, off, 3); off += 12
        s['dMin'] = read_floats(data, off, 3); off += 12
        s['specDens27'] = read_floats(data, off, SF_N_BANDS * 3)
        off += SF_N_BANDS * 3 * 4
        s['projOutputMat27'] = read_floats(data, off, SF_N_BANDS * 3)
        off += SF_N_BANDS * 3 * 4
        s['hRef'] = read_floats(data, off, 3); off += 12
        s['dMinSdFull'] = read_floats(data, off, SF_N_WAVEL); off += SF_N_WAVEL * 4
        s['sensitivityFull'] = read_floats(data, off, SF_N_WAVEL * 3)
        off += SF_N_WAVEL * 3 * 4
        s['specDensFull'] = read_floats(data, off, SF_N_WAVEL * 3)
        off += SF_N_WAVEL * 3 * 4

        bytes_read = off - rec_start
        if bytes_read < rec_size: off += (rec_size - bytes_read)

        safe = make_safe_name(name)
        fp = os.path.join(stocks_dir, f"prt_{i:02d}_{safe}.json")
        with open(fp, 'w') as f:
            json.dump(s, f, indent=2)
        print(f"  [prt {i:2d}] {name}")

    # LUTs
    n_luts = struct.unpack_from('<I', data, off)[0]; off += 4

    for i in range(n_luts):
        neg_idx = struct.unpack_from('<i', data, off)[0]; off += 4
        prt_idx_32 = struct.unpack_from('<i', data, off)[0]; off += 4
        lut_data = read_floats(data, off, SF_LUT_FLOATS)
        off += SF_LUT_FLOATS * 4

        if prt_idx_32 >= 0:
            fname = f"lut_{neg_idx:02d}_{prt_idx_32:02d}.float32"
        else:
            fname = f"lut_{neg_idx:02d}_-1.float32"

        fp = os.path.join(luts_dir, fname)
        with open(fp, 'wb') as f:
            f.write(struct.pack(f'<{SF_LUT_FLOATS}f', *lut_data))

    print(f"\n  {n_luts} LUTs")

    # Verify round-trip
    print(f"\n  Total consumed: {off} / {len(data)} bytes")
    if off != len(data):
        print(f"  WARNING: consumed {off} but file has {len(data)} bytes")
    else:
        print("  No bytes left over — round-trip check passed.")

    return n_luts


if __name__ == '__main__':
    bin_path = sys.argv[1] if len(sys.argv) > 1 else 'Resources/spectral_film_stocks.bin'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'Resources/spectral_film_data'
    stocks_dir = os.path.join(out_dir, 'stocks')
    luts_dir = os.path.join(out_dir, 'luts')
    n_luts = decompile(bin_path, stocks_dir, luts_dir)
    print(f"\nDecompiled {n_luts} LUTs")
    print(f"  Stocks: {stocks_dir}/ ({len(os.listdir(stocks_dir))} files)")
    print(f"  LUTs:   {luts_dir}/ ({len(os.listdir(luts_dir))} files)")
