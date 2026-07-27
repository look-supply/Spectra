#!/usr/bin/env python3
"""Generate spectral film stock data from the vendored spectral_film_lut source.

Replicates the exact process from the original gen_filmdata.py in
~/Documents/Archive/LoFi-FX-plugins/spectral-film-ofx/

Usage:
  python3 gen_filmdata.py
"""

import sys, os, struct, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Use vendored spectral_film_lut (5nm steps, 81 wavelengths — archive version)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "vendor"))

import colour
from spectral_film_lut.film_spectral import FilmSpectral
from spectral_film_lut.config import SPECTRAL_SHAPE
from spectral_film_lut.densiometry import PRINTER_LIGHTS
from spectral_film_lut import NEGATIVE_FILM, PRINT_FILM
from spectral_film_lut.utils import film_conversion

# ---------------------------------------------------------------------------
# DaVinci Wide Gamut <-> CIE XYZ (D65)
# ---------------------------------------------------------------------------
DWG_to_XYZ = np.array([
    [0.7006607968,  0.1487679160,  0.1010412872],
    [0.2741335367,  0.8735913831, -0.1477249199],
    [-0.0989683375, -0.1378889305,  1.3256872680],
], dtype=np.float32)
XYZ_to_DWG = np.array([
    [1.5165889100, -0.2814626195, -0.1469555778],
    [-0.4649386618,  1.2514818105,  0.1748927191],
    [0.0648602355,  0.1091581735,  0.7615459883],
], dtype=np.float32)

N_WAVEL = len(SPECTRAL_SHAPE.wavelengths)
N_BANDS = N_WAVEL // 3
PRINTER_LIGHTS_ARRAY = np.asarray(PRINTER_LIGHTS, dtype=np.float32)
PRINTER_LIGHTS_27 = PRINTER_LIGHTS_ARRAY.reshape(N_BANDS, 3, 3).sum(axis=1)

# ---------------------------------------------------------------------------
# Stock selection — color only (no BW)
# ---------------------------------------------------------------------------
COLOR_NEGATIVES = []; seen = set()
for s in NEGATIVE_FILM:
    if s.density_measure != "bw" and s.name not in seen:
        COLOR_NEGATIVES.append(s); seen.add(s.name)

COLOR_PRINTS = []; seen = set()
for s in PRINT_FILM:
    if s.density_measure == "status_a" and s.name not in seen:
        COLOR_PRINTS.append(s); seen.add(s.name)

# Sort by the exact order the C++ UI expects
NEG_ORDER = [
    "Agfa Vista 100", "Fuji C200", "Fuji Eterna 500", "Fuji Eterna 500 Vivid",
    "Fuji Natura 1600", "Fuji Pro 160C", "Fuji Pro 160S", "Fuji Pro 400H",
    "Fuji Superia Reala", "Fuji Superia X-Tra 400",
    "Kodak Vision3 50D 5203", "Kodak Verita 200D 5206",
    "Kodak Vision3 250D 5207", "Kodak Vision3 200T 5213",
    "Kodak Vision3 500T 5219", "Kodak 5247", "Kodak 5247 II",
    "Kodak 5247 II Alt", "Kodak 5248", "Kodak EXR 100T 5248",
    "Kodak 5250", "Kodak Vision 320T 5277", "Kodak EXR 200T 5293",
    "Kodak Aerocolor IV 2460", "Kodak Aerocolor IV 2460 Low",
    "Kodak Aerocolor IV 2460 High", "Kodak Ektar 100", "Kodak Gold 200",
    "Kodak Portra 160", "Kodak Portra 400", "Kodak Portra 800",
    "Kodak Portra 800 @1600", "Kodak Portra 800 @3200",
    "Kodak Ultramax 400", "Kodak Vericolor III",
]
PRT_ORDER = [
    "Kodak Vision 2383", "Fuji Eterna-CP Type 3513DI", "Fuji Eterna-CP 3523XD",
    "Fuji Crystal Archive Pro PDII", "Fuji Crystal Archive DPII",
    "Fuji Crystal Archive Maxima", "Fuji Crystal Archive Super Type C",
    "Fujiflex Crystal Archive New Version", "Fujiflex Crystal Archive Old Version",
    "Kodak Vision Premier 2393", "Kodak 5381", "Kodak 5383", "Kodak 5384",
    "Kodak Duraflex Plus", "Kodak Endura Premier Paper", "Kodak EXR 5386",
    "Kodak Portra Endura Paper", "Kodak Supra Endura Paper",
]
name_to_fd = {s.name: s for s in COLOR_NEGATIVES}
COLOR_NEGATIVES = [name_to_fd[n] for n in NEG_ORDER if n in name_to_fd]
name_to_fd = {s.name: s for s in COLOR_PRINTS}
COLOR_PRINTS = [name_to_fd[n] for n in PRT_ORDER if n in name_to_fd]

print(f"Negatives: {len(COLOR_NEGATIVES)}, Prints: {len(COLOR_PRINTS)}")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAGIC = b"SPFILM\0\0"
VERSION = 3
MAX_NAME = 64
CURVE_SAMPLES = 512

def pack_name(name):
    encoded = name.encode("utf-8")[:MAX_NAME-1]
    return encoded + b"\0" * (MAX_NAME - len(encoded))

# ---------------------------------------------------------------------------
# Build negative stock record
# ---------------------------------------------------------------------------
def build_negative_data(film_data):
    print(f"  Neg: {film_data.name}")
    film = FilmSpectral(film_data)

    lut_linear = film.get_input_lut(6500, tint=0.0, exp_comp=-0.6).astype(np.float32)
    log_H = film.log_exposure
    log_H_min = np.float32(log_H.min())
    log_H_max = np.float32(log_H.max())
    x_sample = np.linspace(log_H.min(), log_H.max(), CURVE_SAMPLES)
    curves = np.zeros((3, CURVE_SAMPLES), dtype=np.float32)
    for ch in range(3):
        curves[ch] = np.interp(x_sample, log_H, film.density_curve[ch]).astype(np.float32)

    d_ref = film.d_ref.astype(np.float32)
    d_min = film.d_min.astype(np.float32)

    spec_dens_full = film.get_spectral_density(film.color_masking).astype(np.float32)
    spec_dens_27 = spec_dens_full.reshape(N_BANDS, 3, 3).mean(axis=1).astype(np.float32)
    d_min_sd_full = film.d_min_sd.astype(np.float32)
    d_min_sd_27 = d_min_sd_full.reshape(N_BANDS, 3).mean(axis=1).astype(np.float32)
    d_ref_sd_full = film.d_ref_sd.astype(np.float32) if film.d_ref_sd is not None else np.zeros(N_WAVEL, np.float32)
    sensitivity_full = film.sensitivity.astype(np.float32)
    color_masking = np.float32(film.color_masking if film.color_masking is not None else 0.0)

    buf = bytearray()
    buf += pack_name(film_data.name)
    buf += lut_linear.flatten().tobytes()
    buf += struct.pack("<ff", log_H_min, log_H_max)
    buf += curves.flatten().tobytes()
    buf += d_ref.tobytes()
    buf += d_min.tobytes()
    buf += spec_dens_27.flatten().tobytes()
    buf += d_min_sd_27.tobytes()
    buf += d_ref_sd_full.tobytes()
    buf += d_min_sd_full.tobytes()
    buf += spec_dens_full.flatten().tobytes()
    buf += sensitivity_full.flatten().tobytes()
    buf += struct.pack("<f", color_masking)
    return bytes(buf)

# ---------------------------------------------------------------------------
# Build print stock record
# ---------------------------------------------------------------------------
def build_print_data(film_data):
    print(f"  Prt: {film_data.name}")
    film = FilmSpectral(film_data)

    log_H = film.log_exposure
    x_sample = np.linspace(log_H.min(), log_H.max(), CURVE_SAMPLES)
    curves = np.zeros((3, CURVE_SAMPLES), dtype=np.float32)
    for ch in range(3):
        curves[ch] = np.interp(x_sample, log_H, film.density_curve[ch]).astype(np.float32)

    d_ref = film.d_ref.astype(np.float32)
    d_min = film.d_min.astype(np.float32)
    H_ref = film.H_ref.astype(np.float32) if film.H_ref is not None else np.ones(3, np.float32)

    spec_dens_full = film.get_spectral_density().astype(np.float32)
    spec_dens_27 = spec_dens_full.reshape(N_BANDS, 3, 3).mean(axis=1).astype(np.float32)

    proj_light, xyz_cmfs = film.compute_projection_light(projector_kelvin=6500, white_comp=False)
    d_min_sd_full = film.d_min_sd.astype(np.float32)
    output_mat_full = (xyz_cmfs.T * proj_light * 10**-d_min_sd_full).T.astype(np.float32)
    proj_output_mat_27 = output_mat_full.reshape(N_BANDS, 3, 3).sum(axis=1).astype(np.float32)
    sensitivity_full = film.sensitivity.astype(np.float32)

    buf = bytearray()
    buf += pack_name(film_data.name)
    buf += struct.pack("<ff", np.float32(log_H.min()), np.float32(log_H.max()))
    buf += curves.flatten().tobytes()
    buf += d_ref.tobytes()
    buf += d_min.tobytes()
    buf += spec_dens_27.flatten().tobytes()
    buf += proj_output_mat_27.flatten().tobytes()
    buf += H_ref.tobytes()
    buf += d_min_sd_full.tobytes()
    buf += sensitivity_full.flatten().tobytes()
    buf += spec_dens_full.flatten().tobytes()
    return bytes(buf)

# ---------------------------------------------------------------------------
# Build 3D LUT
# ---------------------------------------------------------------------------
def build_film_lut_3d(neg_fd, prt_fd):
    neg_film = FilmSpectral(neg_fd) if not hasattr(neg_fd, "input_transform") else neg_fd
    prt_film = None
    if prt_fd is not None:
        prt_film = FilmSpectral(prt_fd) if not hasattr(prt_fd, "input_transform") else prt_fd

    lut = colour.LUT3D(size=33)
    result = film_conversion(
        lut.table, neg_film, print_film=prt_film,
        input_colorspace="DaVinci Wide Gamut",
        white_clip=False,
        output_gamut="Rec. 709",
        gamma_func="Gamma 2.4",
        exp_comp=-0.6,
        sat_adjust=1.35,
    )
    # Transpose for GPU B-major storage: (R,G,B,ch) -> (B,G,R,ch)
    return np.clip(result, 0.0, 1.0).transpose(2, 1, 0, 3).astype(np.float32).flatten()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    output_dir = os.path.join(SCRIPT_DIR, "Resources")
    bin_path = os.path.join(output_dir, "spectral_film_stocks.bin")

    print("\nBuilding negative stock records...")
    neg_records = [(fd.name, build_negative_data(fd)) for fd in COLOR_NEGATIVES]

    print("\nBuilding print stock records...")
    prt_records = [(fd.name, build_print_data(fd)) for fd in COLOR_PRINTS]

    n_luts = len(COLOR_NEGATIVES) * (1 + len(COLOR_PRINTS))

    with open(bin_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(neg_records)))
        f.write(struct.pack("<I", len(prt_records)))
        for name, data in neg_records:
            f.write(struct.pack("<I", len(data))); f.write(data)
        for name, data in prt_records:
            f.write(struct.pack("<I", len(data))); f.write(data)
        f.write(struct.pack("<I", n_luts))

        for ni, neg_fd in enumerate(COLOR_NEGATIVES):
            print(f"  LUT: {neg_fd.name} + none")
            lut = build_film_lut_3d(neg_fd, None)
            f.write(struct.pack("<I", ni))
            f.write(struct.pack("<i", -1))
            f.write(lut.tobytes())
            for pi, prt_fd in enumerate(COLOR_PRINTS):
                print(f"  LUT: {neg_fd.name} + {prt_fd.name}")
                lut = build_film_lut_3d(neg_fd, prt_fd)
                f.write(struct.pack("<I", ni))
                f.write(struct.pack("<I", pi))
                f.write(lut.tobytes())

    size_kb = os.path.getsize(bin_path) / 1024
    print(f"\nWrote {bin_path}  ({size_kb:.1f} KB)")
    print(f"  {len(neg_records)} neg, {len(prt_records)} prt, {n_luts} LUTs")

if __name__ == "__main__":
    main()
