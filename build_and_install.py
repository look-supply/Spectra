#!/usr/bin/env python3
"""Build LofiFxFilm OFX plugin and install to DaVinci Resolve plugins directory."""

import os
import subprocess
import sys
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")
BUNDLE_NAME = "LofiFxFilm.ofx.bundle"
BUNDLE_SRC = os.path.join(BUILD_DIR, BUNDLE_NAME)
INSTALL_DIR = "/Library/OFX/Plugins"
BUNDLE_DST = os.path.join(INSTALL_DIR, BUNDLE_NAME)


def run(cmd, **kwargs):
    print(f">>> {' '.join(cmd)}")
    subprocess.check_call(cmd, **kwargs)


def adhoc_sign(bundle_path: str):
    """Ad-hoc sign the bundle on macOS (free, no Developer account required).

    This strips any quarantine issues that would prevent DaVinci Resolve from
    loading the plugin after the user downloads and copies the bundle manually.
    Silently skipped on non-macOS platforms.
    """
    if sys.platform != "darwin":
        return
    if not shutil.which("codesign"):
        print("codesign not found — skipping ad-hoc signing")
        return
    print("\n=== Ad-hoc signing bundle ===")
    run(["codesign", "--sign", "-", "--force", "--deep", bundle_path])


def strip_quarantine(bundle_path: str):
    """Remove the com.apple.quarantine extended attribute from the installed bundle.

    macOS sets this on files downloaded from the internet, which prevents
    DaVinci Resolve from loading the plugin. Silently skipped on non-macOS.
    """
    if sys.platform != "darwin":
        return
    if not shutil.which("xattr"):
        print("xattr not found — skipping quarantine removal")
        return
    print("\n=== Removing quarantine attribute ===")
    run(["sudo", "xattr", "-r", "-d", "com.apple.quarantine", bundle_path])


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)

    # Configure
    print("\n=== Configuring ===")
    run(["cmake", ".."], cwd=BUILD_DIR)

    # Build
    print("\n=== Building ===")
    run(["cmake", "--build", "."], cwd=BUILD_DIR)

    # Always repack LUTs and refresh Resources/, since CMake's POST_BUILD
    # copies only fire when the C++ target relinks. Switching branches or
    # re-baking data files without touching C++ would otherwise silently
    # leave the bundle pointing at the previous branch's data.
    bundle_res_dir = os.path.join(BUILD_DIR, BUNDLE_NAME, "Contents", "Resources")
    luts_dir = os.path.join(SCRIPT_DIR, "LUTs")
    if os.path.isdir(luts_dir):
        pak_path = os.path.join(bundle_res_dir, "resources.pak")
        print("\n=== Packing LUTs ===")
        run([sys.executable, os.path.join(SCRIPT_DIR, "pack_luts.py"), luts_dir, pak_path])

    stocks_dir = os.path.join(SCRIPT_DIR, "Resources", "spectral_film_data", "stocks")
    luts_dir = os.path.join(SCRIPT_DIR, "Resources", "spectral_film_data", "luts")
    if os.path.isdir(stocks_dir) and os.path.isdir(luts_dir):
        os.makedirs(bundle_res_dir, exist_ok=True)
        stocks_dst = os.path.join(bundle_res_dir, "spectral_film_stocks.bin")
        pack_script = os.path.join(SCRIPT_DIR, "pack_filmdata.py")
        print("\n=== Packing spectral film stock data into bundle ===")
        run([sys.executable, pack_script, stocks_dir, luts_dir, stocks_dst])
    else:
        stocks_dst = os.path.join(bundle_res_dir, "spectral_film_stocks.bin")
        if os.path.exists(stocks_dst):
            print("\n=== Removing stale spectral_film_stocks.bin from bundle ===")
            os.remove(stocks_dst)

    # Ad-hoc sign (macOS only)
    adhoc_sign(BUNDLE_SRC)

    # Install
    print(f"\n=== Installing to {INSTALL_DIR} ===")
    if os.path.exists(BUNDLE_DST):
        run(["sudo", "rm", "-rf", BUNDLE_DST])
    run(["sudo", "cp", "-R", BUNDLE_SRC, BUNDLE_DST])

    # Strip quarantine so macOS doesn't block the plugin on other machines
    strip_quarantine(BUNDLE_DST)

    print("\nDone. Restart DaVinci Resolve to load the updated plugin.")


if __name__ == "__main__":
    main()
