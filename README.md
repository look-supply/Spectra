# LoFi FX Spectra

Spectra is an OFX plugin by LoFi FX for DaVinci Resolve that emulates motion-picture film stocks using manufacturer-published spectral sensitivity and density data from film stock technical white papers.

The plugin runs in scene-linear space on your GPU, so the emulation scales correctly with exposure and reacts to color temperature changes like actual negative would. Pick a stock and it behaves like the film it's based on. More importantly, Spectra is built for speed and reliability in a working grade session. The controls are straightforward, the response is predictable, and the results hold up across shots, lighting conditions, and formats. It takes the guesswork out of film emulation so you can build consistent, bulletproof grades without fighting the tool.

## Features

- **Dual grain engines** — CW Stochastic (fast, clean) and SFG Organic (domain-warped, filmic) with format-specific grain profiles (8 mm through 65 mm)
- **Spectral film data** — digitized datasheet measurements from a wide library of negative, print, and slide materials spanning still photography, motion picture, colour, black-and-white, Kodak and Fuji stocks
- **Three-lobe optical diffusion** — a tight core, pearlescent body glow, and broad veiling halo driven by a scene-linear prepass with adaptive-resolution sampling
- **Halation** — exponential-falloff red-layer scatter around high-contrast edges, computed in the linear domain
- **LUT-based looks** — bundled .cube transforms for creative grading

## Build

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
```

Requires: CMake 3.20+, macOS 11+ (Metal GPU), Xcode Command Line Tools. Internet connection on first build (fetches OpenFX SDK headers).

Output: `build/LofiFxFilm.ofx.bundle`

## Install

```
sudo cmake --build build --target install_local
```

Or copy `build/LofiFxFilm.ofx.bundle` to `/Library/OFX/Plugins/`. Restart Resolve — the plugin appears under **LoFi FX Film** in the OFX effects list.

## Download

[Download v0.9 beta (macOS)](https://github.com/lofi-fx/Spectra/releases/tag/v0.9-beta)

## References

Spectra is built on concepts and data from the following open-source projects:

- **[Spectral Film LUT](https://github.com/JanLohse/spectral_film_lut)** (Jan Lohse, MIT) — digitized film stock datasheet measurements and the spectral film pipeline that inspired this plugin's colour science.
- **[Halation DCTL](https://github.com/hotgluebanjo/halation-dctl)** (hotgluebanjo) — informed the exponential-falloff halation approach implemented in Spectra's Metal compute kernels.

Both projects were studied, rebuilt, and extended as GPU-accelerated OFX-native code rather than LUT-generation tools or DCTL scripts.
