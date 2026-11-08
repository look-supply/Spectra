cmake_minimum_required(VERSION 3.20)
project(LookSupplySpectra LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Fetch OpenFX SDK headers
# ============================================================
include(FetchContent)
FetchContent_Declare(
    openfx
    GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openfx.git
    GIT_TAG        OFX_Release_1.5.1
)
# Only populate (download) — do NOT add_subdirectory (avoids OpenGL deps)
FetchContent_GetProperties(openfx)
if(NOT openfx_POPULATED)
    FetchContent_Populate(openfx)
endif()

set(OFX_INCLUDE_DIR "${openfx_SOURCE_DIR}/include")

# ============================================================
# Platform detection
# ============================================================
if(APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
endif()

# ============================================================
# Embed the shared Film kernel source (FilmKernelCore.inc) into a
# C++ string-literal header so the Metal runtime compiler can
# consume the same text that the CUDA preprocessor will #include
# directly. Re-runs automatically whenever the .inc changes.
# ============================================================
set(FILM_KERNEL_CORE_INC "${CMAKE_CURRENT_SOURCE_DIR}/src/FilmKernelCore.inc")
set(FILM_KERNEL_CORE_TEXT_GEN
    "${CMAKE_CURRENT_BINARY_DIR}/generated/FilmKernelCoreText.h")

file(READ "${FILM_KERNEL_CORE_INC}" FILM_KERNEL_CORE_TEXT)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/FilmKernelCoreText.h.in"
    "${FILM_KERNEL_CORE_TEXT_GEN}"
    @ONLY
)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${FILM_KERNEL_CORE_INC}")

# ============================================================
# Sources
# ============================================================
set(PLUGIN_SOURCES
    src/LookSupplySpectraPlugin.cpp
    src/CudaKernels.cpp
    src/PresetParser.cpp
)

if(APPLE)
    list(APPEND PLUGIN_SOURCES src/MetalKernels.mm)
endif()

# ============================================================
# Build the .ofx shared library
# ============================================================
add_library(LookSupplySpectra MODULE ${PLUGIN_SOURCES})

target_include_directories(LookSupplySpectra PRIVATE
    ${OFX_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

set_target_properties(LookSupplySpectra PROPERTIES
    PREFIX ""
    SUFFIX ".ofx"
    # Ensure exported symbols are visible
    CXX_VISIBILITY_PRESET default
    C_VISIBILITY_PRESET   default
)

if(APPLE)
    # Link Metal and Foundation frameworks
    target_link_libraries(LookSupplySpectra
        "-framework Metal"
        "-framework Foundation"
    )

    # Universal binary (Apple Silicon + Intel)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "" FORCE)

    # Minimum deployment target
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "" FORCE)
endif()

# ============================================================
# Post-build: create OFX bundle structure
# ============================================================
set(BUNDLE_DIR "${CMAKE_BINARY_DIR}/LookSupplySpectra.ofx.bundle")

if(APPLE)
    set(BUNDLE_BIN_DIR "${BUNDLE_DIR}/Contents/MacOS")
else()
    set(BUNDLE_BIN_DIR "${BUNDLE_DIR}/Contents/Linux-x86-64")
endif()
set(BUNDLE_RES_DIR "${BUNDLE_DIR}/Contents/Resources")
set(BUNDLE_PRESET_DIR "${BUNDLE_DIR}/Contents/Resources/Presets")

add_custom_command(TARGET LookSupplySpectra POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_BIN_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_RES_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_PRESET_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE:LookSupplySpectra>"
        "${BUNDLE_BIN_DIR}/LookSupplySpectra.ofx"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist"
        "${BUNDLE_DIR}/Contents/Info.plist"
    COMMENT "Assembling LookSupplySpectra.ofx.bundle"
)

# Pack all LUTs into an obfuscated binary blob
# Prefer the system Python that has numpy/scipy/colour-science installed
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(APPLE AND Python3_EXECUTABLE MATCHES "python3\.1[3-9]")
    # Homebrew Python 3.13+ lacks build deps — use system Python instead
    set(Python3_EXECUTABLE "/usr/bin/python3" CACHE STRING "" FORCE)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/LUTs")
    add_custom_command(TARGET LookSupplySpectra POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${BUNDLE_RES_DIR}/LUTs"
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/pack_luts.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/LUTs"
            "${BUNDLE_RES_DIR}/resources.pak"
        COMMENT "Packing LUTs into obfuscated resources.pak"
    )
endif()

# Copy preset files into bundle — clean first so stale files are removed
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/presets")
    add_custom_command(TARGET LookSupplySpectra POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${BUNDLE_PRESET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/presets"
            "${BUNDLE_PRESET_DIR}"
        COMMENT "Syncing preset files into bundle (clean copy)"
    )
endif()

# Pack spectral film stock data from individual files into the .bin
# that the C++ loader expects.  The individual JSON/float32 files are
# the committed source of truth.
set(SPECTRAL_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Resources/spectral_film_data")
if(EXISTS "${SPECTRAL_DATA_DIR}")
    add_custom_command(TARGET LookSupplySpectra POST_BUILD
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/pack_filmdata.py"
            "${SPECTRAL_DATA_DIR}/stocks"
            "${SPECTRAL_DATA_DIR}/luts"
            "${BUNDLE_RES_DIR}/spectral_film_stocks.bin"
        COMMENT "Packing spectral film data into bundle"
    )
endif()

# ============================================================
# Install target: copy bundle to system OFX plugin directory
# ============================================================
if(APPLE)
    set(OFX_PLUGIN_INSTALL_DIR "/Library/OFX/Plugins")
else()
    set(OFX_PLUGIN_INSTALL_DIR "/usr/OFX/Plugins")
endif()

install(
    DIRECTORY "${BUNDLE_DIR}"
    DESTINATION "${OFX_PLUGIN_INSTALL_DIR}"
    USE_SOURCE_PERMISSIONS
)

# ============================================================
# Convenience target: install to user's OFX plugin dir
# ============================================================
if(APPLE)
    add_custom_target(install_local
        COMMAND ${CMAKE_COMMAND} -E rm -rf "/Library/OFX/Plugins/LookSupplySpectra.ofx.bundle"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${BUNDLE_DIR}"
            "/Library/OFX/Plugins/LookSupplySpectra.ofx.bundle"

        DEPENDS LookSupplySpectra
        COMMENT "Installing to /Library/OFX/Plugins/ (clean copy)"
    )
endif()
