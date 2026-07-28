#!/usr/bin/env python3
"""Generate spectral film stock data from the vendored spectral_film_lut source.

Writes individual JSON stock files and float32 3D LUTs to
  Resources/spectral_film_data/stocks/
  Resources/spectral_film_data/luts/

The data is the transparent source of truth — commit these files so repo
viewers can inspect exactly what film data is included.  At build time
pack_filmdata.py assembles them into the binary format the C++ loader
expects.

Usage:
  python3 gen_filmdata.py
"""

import sys, os, json, struct, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "vendor", "spectral_film_lut", "src"))

import colour
from spectral_film_lut.film_spectral import FilmSpectral
from spectral_film_lut.config import SPECTRAL_SHAPE
from spectral_film_lut.densiometry import PRINTER_LIGHTS
from spectral_film_lut import NEGATIVE_FILM, PRINT_FILM
from spectral_film_lut.utils import film_conversion

N_WAVEL = len(SPECTRAL_SHAPE.wavelengths)
N_BANDS = (N_WAVEL // 3) * 3 // 3
N_WAVEL_USED = N_BANDS * 3
PRINTER_LIGHTS_ARRAY = np.asarray(PRINTER_LIGHTS, dtype=np.float32)
PRINTER_LIGHTS_27 = PRINTER_LIGHTS_ARRAY[:N_WAVEL_USED].reshape(N_BANDS, 3, 3).sum(axis=1)

COLOR_NEGATIVES = []; seen = set()
for s in NEGATIVE_FILM:
    if s.density_measure != "bw" and s.name not in seen:
        COLOR_NEGATIVES.append(s); seen.add(s.name)

COLOR_PRINTS = []; seen = set()
for s in PRINT_FILM:
    if s.density_measure == "status_a" and s.name not in seen:
        COLOR_PRINTS.append(s); seen.add(s.name)

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

CURVE_SAMPLES = 512
SF_LUT3D_SIZE = 33

def make_safe(name):
    return ''.join(c for c in name.replace('/', '_').replace(' ', '_').replace('.', '_') if c.isalnum() or c == '_')

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
    spec_dens_27 = spec_dens_full[:N_WAVEL_USED].reshape(N_BANDS, 3, 3).mean(axis=1).astype(np.float32)
    d_min_sd_full = film.d_min_sd.astype(np.float32)
    d_min_sd_27 = d_min_sd_full[:N_WAVEL_USED].reshape(N_BANDS, 3).mean(axis=1).astype(np.float32)
    d_ref_sd_full = film.d_ref_sd.astype(np.float32) if film.d_ref_sd is not None else np.zeros(N_WAVEL, np.float32)
    sensitivity_full = film.sensitivity.astype(np.float32)
    color_masking = np.float32(film.color_masking if film.color_masking is not None else 0.0)

    return {
        "name": film_data.name,
        "inputLut": lut_linear.flatten().tolist(),
        "logHMin": float(log_H_min),
        "logHMax": float(log_H_max),
        "curve": curves.flatten().tolist(),
        "dRef": d_ref.tolist(),
        "dMin": d_min.tolist(),
        "specDens27": spec_dens_27.flatten().tolist(),
        "dMinSd27": d_min_sd_27.tolist(),
        "dRefSdFull": d_ref_sd_full.tolist(),
        "dMinSdFull": d_min_sd_full.tolist(),
        "specDensFull": spec_dens_full.flatten().tolist(),
        "sensitivityFull": sensitivity_full.flatten().tolist(),
        "colorMasking": float(color_masking),
    }

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
    spec_dens_27 = spec_dens_full[:N_WAVEL_USED].reshape(N_BANDS, 3, 3).mean(axis=1).astype(np.float32)

    proj_light, xyz_cmfs = film.compute_projection_light(projector_kelvin=6500, white_comp=False)
    d_min_sd_full = film.d_min_sd.astype(np.float32)
    output_mat_full = (xyz_cmfs.T * proj_light * 10**-d_min_sd_full).T.astype(np.float32)
    proj_output_mat_27 = output_mat_full[:N_WAVEL_USED].reshape(N_BANDS, 3, 3).sum(axis=1).astype(np.float32)
    sensitivity_full = film.sensitivity.astype(np.float32)

    return {
        "name": film_data.name,
        "logHMin": float(np.float32(log_H.min())),
        "logHMax": float(np.float32(log_H.max())),
        "curve": curves.flatten().tolist(),
        "dRef": d_ref.tolist(),
        "dMin": d_min.tolist(),
        "specDens27": spec_dens_27.flatten().tolist(),
        "projOutputMat27": proj_output_mat_27.flatten().tolist(),
        "hRef": H_ref.tolist(),
        "dMinSdFull": d_min_sd_full.tolist(),
        "sensitivityFull": sensitivity_full.flatten().tolist(),
        "specDensFull": spec_dens_full.flatten().tolist(),
    }

def build_film_lut_3d(neg_fd, prt_fd):
    neg_film = FilmSpectral(neg_fd) if not hasattr(neg_fd, "input_transform") else neg_fd
    prt_film = None
    if prt_fd is not None:
        prt_film = FilmSpectral(prt_fd) if not hasattr(prt_fd, "input_transform") else prt_fd

    lut = colour.LUT3D(size=SF_LUT3D_SIZE)
    result = film_conversion(
        lut.table, neg_film, print_film=prt_film,
        input_colorspace="DaVinci Wide Gamut",
        white_clip=False,
        output_gamut="Rec. 709",
        gamma_func="Gamma 2.4",
        exp_comp=-0.6,
        sat_adjust=1.35,
    )
    return np.clip(result, 0.0, 1.0).transpose(2, 1, 0, 3).astype(np.float32).flatten()

class _JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        return super().default(obj)

def main():
    data_dir = os.path.join(SCRIPT_DIR, "Resources", "spectral_film_data")
    stocks_dir = os.path.join(data_dir, "stocks")
    luts_dir = os.path.join(data_dir, "luts")
    os.makedirs(stocks_dir, exist_ok=True)
    os.makedirs(luts_dir, exist_ok=True)

    print("\nWriting negative stock JSON files...")
    for i, fd in enumerate(COLOR_NEGATIVES):
        rec = build_negative_data(fd)
        safe = make_safe(fd.name)
        with open(os.path.join(stocks_dir, f"neg_{i:02d}_{safe}.json"), 'w') as f:
            json.dump(rec, f, indent=2, cls=_JsonEncoder)

    print("\nWriting print stock JSON files...")
    for i, fd in enumerate(COLOR_PRINTS):
        rec = build_print_data(fd)
        safe = make_safe(fd.name)
        with open(os.path.join(stocks_dir, f"prt_{i:02d}_{safe}.json"), 'w') as f:
            json.dump(rec, f, indent=2, cls=_JsonEncoder)

    print("\nWriting 3D LUT float32 files...")
    n_luts = 0
    for ni, neg_fd in enumerate(COLOR_NEGATIVES):
        print(f"  LUT: {neg_fd.name} + none")
        lut = build_film_lut_3d(neg_fd, None)
        lut.tofile(os.path.join(luts_dir, f"lut_{ni:02d}_-1.float32"))
        n_luts += 1
        for pi, prt_fd in enumerate(COLOR_PRINTS):
            print(f"  LUT: {neg_fd.name} + {prt_fd.name}")
            lut = build_film_lut_3d(neg_fd, prt_fd)
            lut.tofile(os.path.join(luts_dir, f"lut_{ni:02d}_{pi:02d}.float32"))
            n_luts += 1

    print(f"\nDone — {len(COLOR_NEGATIVES)} neg, {len(COLOR_PRINTS)} prt, {n_luts} LUTs")
    print(f"  Stocks: {stocks_dir}/")
    print(f"  LUTs:   {luts_dir}/")
    print("Run pack_filmdata.py to assemble into spectral_film_stocks.bin for the build.")

if __name__ == "__main__":
    main()
