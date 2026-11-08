#!/usr/bin/env python3
"""Generate self-contained .preset files for every .cube in LUTs/Looks/.

Each preset embeds a reference to a local copy of the .cube file so that
recalling the preset loads the LUT from the companion file rather than
from the plugin's built-in resources.pak.

Usage:  python3 generate_look_presets.py
"""

import os
import shutil
import sys

# ── paths ──────────────────────────────────────────────────────────
HOME       = os.path.expanduser("~")
# Source .cube files are copies already placed in the user preset directory
CUBE_DIR   = os.path.join(HOME, "Documents", "Lofi FX Film Presets")
OUT_DIR    = CUBE_DIR

# WP string → numeric index (matches CubeParser.h / GrainParams.h)
WP_MAP = {
    "D50": 0,
    "D55": 1,
    "D60": 2,
    "D65": 3,
    "D70": 4,
    "D75": 5,
}
DEFAULT_WP = 3  # D65


def read_meta(meta_path):
    """Return (title, wp_index) from a .meta file, or (None, DEFAULT_WP)."""
    title = None
    wp    = DEFAULT_WP
    if not os.path.isfile(meta_path):
        return title, wp
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("name:"):
                title = line[5:].strip()
            elif line.startswith("wp:"):
                s = line[3:].strip()
                wp = WP_MAP.get(s, DEFAULT_WP)
    return title, wp


def safe_filename(name):
    """Sanitize a name for use as a filename (same logic as saveCurrentPreset)."""
    safe = []
    for c in name:
        if c.isalnum() or c in "-_ .":
            safe.append(c)
        else:
            safe.append("_")
    result = "".join(safe)
    return result if result else "preset"


def generate_preset(cube_rel, meta_title, wp_index):
    """Build a .preset file content string.

    cube_rel is the relative path used inside the preset to locate the
    companion .cube file (e.g.  "Kodak Vision 3 5219 500T.cube").
    """
    title   = meta_title or os.path.splitext(os.path.basename(cube_rel))[0]
    esc     = title.replace('"', "'")

    return f"""# ============================================================
# Look Supply Film — Look Preset: {title}
#
# This preset carries an embedded copy of the .cube file so it
# loads the LUT from the companion file instead of resources.pak.
# ============================================================

name = {title}

# ---- Grain (neutral / off) ----
grain_enable     = 0
film_format      = 4        # 4 = Custom
grain_size       = 0.0
grain_strength   = 0.0
grain_blur       = 0.0
grain_opacity    = 0.0
grain_sat        = 0.0
grain_shape_irregularity = 0.5
grain_contrast   = 0.5
hi_rolloff       = 0.0
seed_offset      = 0
image_blur_enable = 0
image_blur       = 0.0
grain_blur_link  = 0.5

# ---- Film Emulation (off) ----
sf_look_enable    = 0
sf_look_saturation = 1.0
sf_contrast_curve  = 1.0
sf_wp_kelvin      = 6500

# ---- Halation (off) ----
hal_enable       = 0
hal_size         = 1.0
hal_color        = 0.15
hal_method       = 0
hal_opacity      = 1.0

# ---- Diffusion (off) ----
mist_enable      = 0
mist_filter      = 7        # 7 = Custom
mist_amount      = 0.0
mist_spread      = 0.25
mist_softness    = 1.0
mist_tail_mix    = 0.0
mist_tail_falloff = 1.0
mist_threshold   = 1.0
mist_veil        = 0.0
mist_contrast    = 0.0
mist_bloom       = 0.0
mist_tint_strength = 0.0
mist_tint_r      = 1.0
mist_tint_g      = 1.0
mist_tint_b      = 1.0

# ---- Printer Points (off) ----
print_enable     = 0
print_gang       = 0
print_r          = 25.0
print_g          = 25.0
print_b          = 25.0

# ---- Split Toning (off) ----
split_enable     = 0
split_strength   = 0.0
split_shadow_hue = 0.0
split_highlight_hue = 0.0
split_lo_vs_hi   = 0.5
split_pivot      = 0.336
split_sub_lo     = 1.0
split_sub_hi     = 1.0
split_neutral_width = 0.15

# ---- Output Transform ----
output_transform = 0

# ---- Input Transform ----
input_transform  = 0

# ---- Embedded LUT (local .cube copy) ----
lut_embedded     = 1
lut_embedded_path = {cube_rel}
lut_embedded_title = {esc}
lut_embedded_wp  = {wp_index}
"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cubes = []

    # Walk CUBE_DIR for .cube files (already placed alongside presets)
    for fn in os.listdir(CUBE_DIR):
        if fn.endswith(".cube"):
            full = os.path.join(CUBE_DIR, fn)
            cubes.append((full, fn, CUBE_DIR))

    if not cubes:
        print("No .cube files found in", CUBE_DIR)
        sys.exit(1)

    cubes.sort(key=lambda t: t[1])

    for full_path, basename, root_dir in cubes:
        meta_path = full_path[:-5] + ".meta"
        meta_title, wp_index = read_meta(meta_path)
        title = meta_title or os.path.splitext(basename)[0]

        # Generate preset content (no copy needed — .cube is already in OUT_DIR)
        preset_content = generate_preset(
            basename,
            meta_title,
            wp_index,
        )

        # Write .preset file
        safe_name = safe_filename(title)
        preset_path = os.path.join(OUT_DIR, safe_name + ".preset")
        with open(preset_path, "w") as f:
            f.write(preset_content)
        print(f"  Preset: {safe_name}.preset  ({title})")

    print(f"\nDone — {len(cubes)} presets written to {OUT_DIR}")


if __name__ == "__main__":
    main()
