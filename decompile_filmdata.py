#!/usr/bin/env python3
"""Decompile spectral_film_stocks.bin into transparent individual files.

Output:
  Resources/spectral_film_data/stocks/  — JSON per stock
  Resources/spectral_film_data/luts/    — .float32 per LUT

Recreate .bin with:
  python3 pack_filmdata.py
"""

import json, os, struct, sys

SF_N_WAVEL = 81; SF_N_BANDS = 27; SF_LUT3D_SIZE = 33
SF_CURVE_N = 512; SF_NAME_LEN = 64
SF_LUT_FLOATS = SF_LUT3D_SIZE ** 3 * 3

def rs(data, off, fmt):
    return struct.unpack_from(fmt, data, off)[0], off + struct.calcsize(fmt)

def rf(data, off): return rs(data, off, '<f')
def rI(data, off): return rs(data, off, '<I')
def rFloats(data, off, n):
    return list(struct.unpack_from(f'<{n}f', data, off)), off + n*4

def make_safe(s):
    return ''.join(c for c in s.replace('/', '_').replace(' ', '_').replace('.', '_') if c.isalnum() or c == '_')

def decompile(path, stocks_dir, luts_dir):
    os.makedirs(stocks_dir, exist_ok=True); os.makedirs(luts_dir, exist_ok=True)
    with open(path, 'rb') as f: data = f.read()
    off = 8; version, off = rI(data, off)
    n_neg, off = rI(data, off); n_prt, off = rI(data, off)
    print(f"v{version}: {n_neg} neg, {n_prt} prt")

    neg_names = []
    for i in range(n_neg):
        recs, off = rI(data, off); start = off
        s = {}; name, off = data[off:off+SF_NAME_LEN].split(b'\0',1)[0].decode(), off+SF_NAME_LEN
        s['name'] = name; neg_names.append(name)
        s['inputLut'], off = rFloats(data, off, SF_LUT3D_SIZE*SF_LUT3D_SIZE*3)
        s['logHMin'], off = rf(data, off); s['logHMax'], off = rf(data, off)
        s['curve'], off = rFloats(data, off, 3*SF_CURVE_N)
        s['dRef'], off = rFloats(data, off, 3); s['dMin'], off = rFloats(data, off, 3)
        s['specDens27'], off = rFloats(data, off, SF_N_BANDS*3)
        s['dMinSd27'], off = rFloats(data, off, SF_N_BANDS)
        s['dRefSdFull'], off = rFloats(data, off, SF_N_WAVEL)
        s['dMinSdFull'], off = rFloats(data, off, SF_N_WAVEL)
        s['specDensFull'], off = rFloats(data, off, SF_N_WAVEL*3)
        s['sensitivityFull'], off = rFloats(data, off, SF_N_WAVEL*3)
        s['colorMasking'], off = rf(data, off)
        if off - start < recs: off += recs - (off - start)
        safe = make_safe(name)
        with open(os.path.join(stocks_dir, f"neg_{i:02d}_{safe}.json"), 'w') as f: json.dump(s, f, indent=2)
        print(f"  [neg {i}] {name}")

    prt_names = []
    for i in range(n_prt):
        recs, off = rI(data, off); start = off
        s = {}; name, off = data[off:off+SF_NAME_LEN].split(b'\0',1)[0].decode(), off+SF_NAME_LEN
        s['name'] = name; prt_names.append(name)
        s['logHMin'], off = rf(data, off); s['logHMax'], off = rf(data, off)
        s['curve'], off = rFloats(data, off, 3*SF_CURVE_N)
        s['dRef'], off = rFloats(data, off, 3); s['dMin'], off = rFloats(data, off, 3)
        s['specDens27'], off = rFloats(data, off, SF_N_BANDS*3)
        s['projOutputMat27'], off = rFloats(data, off, SF_N_BANDS*3)
        s['hRef'], off = rFloats(data, off, 3)
        s['dMinSdFull'], off = rFloats(data, off, SF_N_WAVEL)
        s['sensitivityFull'], off = rFloats(data, off, SF_N_WAVEL*3)
        s['specDensFull'], off = rFloats(data, off, SF_N_WAVEL*3)
        if off - start < recs: off += recs - (off - start)
        safe = make_safe(name)
        with open(os.path.join(stocks_dir, f"prt_{i:02d}_{safe}.json"), 'w') as f: json.dump(s, f, indent=2)
        print(f"  [prt {i}] {name}")

    n_luts, off = rI(data, off)
    for i in range(n_luts):
        neg_idx, off = rI(data, off); prt_idx_32, off = rs(data, off, '<i')
        lut_data, off = rFloats(data, off, SF_LUT_FLOATS)
        fname = f"lut_{neg_idx:02d}_{prt_idx_32:02d}.float32" if prt_idx_32 >= 0 else f"lut_{neg_idx:02d}_-1.float32"
        with open(os.path.join(luts_dir, fname), 'wb') as f:
            f.write(struct.pack(f'<{SF_LUT_FLOATS}f', *lut_data))

    print(f"\nDecompiled {n_luts} LUTs")
    print(f"Stocks: {stocks_dir}/  LUTs: {luts_dir}/")

if __name__ == '__main__':
    bin_path = sys.argv[1] if len(sys.argv) > 1 else 'Resources/spectral_film_stocks.bin'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'Resources/spectral_film_data'
    decompile(bin_path, os.path.join(out_dir, 'stocks'), os.path.join(out_dir, 'luts'))
