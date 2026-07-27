#!/usr/bin/env python3
"""Pack spectral film data into the binary format used by the plugin.

Reads from Resources/spectral_film_data/ (JSON stocks + .float32 LUTs)
and writes spectral_film_stocks.bin in the exact format expected by
the C++ loader.

Usage:
  python3 pack_filmdata.py [stocks_dir] [luts_dir] [output_path]
"""

import json
import os
import struct
import sys

SF_N_WAVEL = 41
SF_N_BANDS = 13
SF_N_WAVEL_USED = SF_N_BANDS * 3  # 39 — drop last 2 of 41
SF_LUT3D_SIZE = 33
SF_CURVE_N = 512
SF_NAME_LEN = 64
SF_LUT_FLOATS = SF_LUT3D_SIZE ** 3 * 3
SF_MAX_STOCKS = 64


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_cstr(buf, s, max_len):
    encoded = s.encode('utf-8')[:max_len - 1]
    buf.extend(encoded)
    buf.extend(b'\0' * (max_len - len(encoded)))


def write_float(buf, v):
    buf.extend(struct.pack('<f', v))


def write_floats(buf, vals):
    for v in vals:
        buf.extend(struct.pack('<f', v))


def pack(stocks_dir, luts_dir, output_path):
    # Load all stock JSONs
    neg_stocks = []
    prt_stocks = []

    for fname in sorted(os.listdir(stocks_dir)):
        if fname.startswith('neg_') and fname.endswith('.json'):
            s = read_json(os.path.join(stocks_dir, fname))
            neg_stocks.append(s)
        elif fname.startswith('prt_') and fname.endswith('.json'):
            s = read_json(os.path.join(stocks_dir, fname))
            prt_stocks.append(s)

    n_neg = len(neg_stocks)
    n_prt = len(prt_stocks)

    print(f"Packing {n_neg} negative + {n_prt} print stocks")

    buf = bytearray()

    # Magic + header
    buf.extend(b'SPFILM\0\0')
    write_float(buf, 3.0)  # version (stored as uint32 but float is easier)
    buf[-4:] = struct.pack('<I', 3)  # ensure proper uint32
    buf[-8:-4] = struct.pack('<I', n_neg)
    buf[-12:-8] = struct.pack('<I', n_prt)

    # Rewrite header properly
    buf = bytearray()
    buf.extend(b'SPFILM\0\0')
    buf.extend(struct.pack('<I', 4))  # version
    buf.extend(struct.pack('<I', n_neg))
    buf.extend(struct.pack('<I', n_prt))

    # Negative stocks
    for idx, s in enumerate(neg_stocks):
        record = bytearray()

        write_cstr(record, s['name'], SF_NAME_LEN)

        # inputLut[33][33][3]
        for v in s['inputLut']:
            write_float(record, v)

        write_float(record, s['logHMin'])
        write_float(record, s['logHMax'])

        for v in s['curve']:
            write_float(record, v)

        write_floats(record, s['dRef'])
        write_floats(record, s['dMin'])

        for v in s['specDens27']:
            write_float(record, v)
        for v in s['dMinSd27']:
            write_float(record, v)
        for v in s['dRefSdFull']:
            write_float(record, v)
        for v in s['dMinSdFull']:
            write_float(record, v)
        for v in s['specDensFull']:
            write_float(record, v)
        for v in s['sensitivityFull']:
            write_float(record, v)

        write_float(record, s['colorMasking'])

        # Write record size prefix + record
        buf.extend(struct.pack('<I', len(record)))
        buf.extend(record)

    # Print stocks
    for idx, s in enumerate(prt_stocks):
        record = bytearray()

        write_cstr(record, s['name'], SF_NAME_LEN)
        write_float(record, s['logHMin'])
        write_float(record, s['logHMax'])

        for v in s['curve']:
            write_float(record, v)

        write_floats(record, s['dRef'])
        write_floats(record, s['dMin'])

        for v in s['specDens27']:
            write_float(record, v)
        for v in s['projOutputMat27']:
            write_float(record, v)
        write_floats(record, s['hRef'])

        for v in s['dMinSdFull']:
            write_float(record, v)
        for v in s['sensitivityFull']:
            write_float(record, v)
        for v in s['specDensFull']:
            write_float(record, v)

        buf.extend(struct.pack('<I', len(record)))
        buf.extend(record)

    # LUTs
    lut_files = sorted(os.listdir(luts_dir))
    n_luts = len(lut_files)
    buf.extend(struct.pack('<I', n_luts))

    # Build lookup: filename -> (negIdx, prtIdx)
    for fname in lut_files:
        if not fname.endswith('.float32'):
            continue
        # Parse filename: lut_<neg>_<prt>.float32 or lut_<neg>_none.float32
        parts = fname.replace('.float32', '').split('_')
        # parts = ['lut', 'NN', 'MM'] or ['lut', 'NN', 'm1']
        neg_idx = int(parts[1])
        prt_idx = -1 if parts[2] == '-1' else int(parts[2])

        filepath = os.path.join(luts_dir, fname)
        with open(filepath, 'rb') as f:
            lut_data = f.read()

        expected_len = SF_LUT_FLOATS * 4
        if len(lut_data) != expected_len:
            print(f"  WARNING: {fname} has {len(lut_data)} bytes, expected {expected_len}")

        buf.extend(struct.pack('<I', neg_idx))
        buf.extend(struct.pack('<i', prt_idx))
        buf.extend(lut_data)

    with open(output_path, 'wb') as f:
        f.write(buf)

    print(f"Wrote {len(buf)} bytes to {output_path}")


if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__) or '.', 'Resources', 'spectral_film_data')
    stocks_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, 'stocks')
    luts_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base_dir, 'luts')
    output = sys.argv[3] if len(sys.argv) > 3 else 'Resources/spectral_film_stocks.bin'
    pack(stocks_dir, luts_dir, output)
