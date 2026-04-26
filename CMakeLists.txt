# Copyright (C) 2026, Lux Partners Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# MPCVM (M-Chain) GPU substrate.
# v0.60: FROST + CGGMP21 + Ringtail threshold ceremonies, GPU-native
#        across CUDA + Metal + WGSL with byte-for-byte cross-backend
#        determinism against a CPU reference oracle.

cmake_minimum_required(VERSION 3.20)

project(lux-mpcvm
    VERSION 0.60.0
    DESCRIPTION "GPU-native M-Chain (MPC) ceremony substrate"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LUX_MPCVM_BUILD_TESTS "Build MPCVM tests" ON)
option(LUX_MPCVM_ENABLE_METAL "Build Metal backend (Apple)" ${APPLE})
option(LUX_MPCVM_ENABLE_CUDA  "Build CUDA backend" OFF)
option(LUX_MPCVM_ENABLE_WGPU  "Build WebGPU/Dawn backend (deferred to v0.61)" OFF)

# ---- core library (host code: layout + CPU reference) --------------------

add_library(lux_mpcvm_core STATIC
    src/mpcvm_cpu_reference.cpp
)

target_include_directories(lux_mpcvm_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(lux_mpcvm_core PUBLIC cxx_std_20)

set_target_properties(lux_mpcvm_core PROPERTIES
    OUTPUT_NAME "lux_mpcvm_core"
    POSITION_INDEPENDENT_CODE ON
)

# ---- Metal driver (Apple) -------------------------------------------------

if(LUX_MPCVM_ENABLE_METAL AND APPLE)
    add_library(lux_mpcvm_metal STATIC
        src/mpcvm_gpu_engine.mm
    )
    target_include_directories(lux_mpcvm_metal PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_compile_features(lux_mpcvm_metal PUBLIC cxx_std_20)
    target_link_libraries(lux_mpcvm_metal PUBLIC lux_mpcvm_core
        "-framework Metal" "-framework Foundation")
    set_target_properties(lux_mpcvm_metal PROPERTIES
        OUTPUT_NAME "lux_mpcvm_metal"
        POSITION_INDEPENDENT_CODE ON
    )

    foreach(src
        mpcvm_kernels_common.h.metal
        mpcvm_ceremony.metal
        mpcvm_frost.metal
        mpcvm_cggmp21.metal
        mpcvm_ringtail.metal
        mpcvm_transition.metal)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/${src}
            ${CMAKE_CURRENT_BINARY_DIR}/src/${src}
            COPYONLY)
    endforeach()

    set(MPCVM_METAL_COMBINED ${CMAKE_CURRENT_BINARY_DIR}/src/mpcvm_combined.metal)
    add_custom_command(
        OUTPUT ${MPCVM_METAL_COMBINED}
        COMMAND bash -c "cat ${CMAKE_CURRENT_SOURCE_DIR}/src/mpcvm_kernels_common.h.metal > ${MPCVM_METAL_COMBINED} && for f in mpcvm_ceremony.metal mpcvm_transition.metal mpcvm_frost.metal mpcvm_cggmp21.metal mpcvm_ringtail.metal; do grep -v 'mpcvm_kernels_common.h.metal' ${CMAKE_CURRENT_SOURCE_DIR}/src/$f >> ${MPCVM_METAL_COMBINED}; done"
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/src/mpcvm_kernels_common.h.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/mpcvm_ceremony.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/mpcvm_transition.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/mpcvm_frost.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/mpcvm_cggmp21.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/mpcvm_ringtail.metal
        VERBATIM)

    set(MPCVM_METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/src/mpcvm_combined.air)
    set(MPCVM_METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/src/mpcvm.metallib)

    add_custom_command(
        OUTPUT ${MPCVM_METAL_AIR}
        COMMAND xcrun -sdk macosx metal -std=metal3.0
            -Wno-pragma-once-outside-header
            -Wno-unused-const-variable
            -c ${MPCVM_METAL_COMBINED} -o ${MPCVM_METAL_AIR}
        DEPENDS ${MPCVM_METAL_COMBINED}
        VERBATIM)

    add_custom_command(
        OUTPUT ${MPCVM_METAL_LIB}
        COMMAND xcrun -sdk macosx metallib ${MPCVM_METAL_AIR} -o ${MPCVM_METAL_LIB}
        DEPENDS ${MPCVM_METAL_AIR}
        VERBATIM)

    add_custom_target(mpcvm_metallib ALL DEPENDS ${MPCVM_METAL_LIB})
    add_dependencies(lux_mpcvm_metal mpcvm_metallib)
endif()

# ---- CUDA driver ---------------------------------------------------------

if(LUX_MPCVM_ENABLE_CUDA)
    enable_language(CUDA)
    add_library(lux_mpcvm_cuda STATIC
        src/mpcvm_gpu_engine_cuda.cpp
        src/mpcvm_ceremony.cu
        src/mpcvm_frost.cu
        src/mpcvm_cggmp21.cu
        src/mpcvm_ringtail.cu
        src/mpcvm_transition.cu
    )
    target_include_directories(lux_mpcvm_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    )
    target_compile_definitions(lux_mpcvm_cuda PUBLIC LUX_MPCVM_HAVE_CUDA=1)
    set_target_properties(lux_mpcvm_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME "lux_mpcvm_cuda"
    )
    target_link_libraries(lux_mpcvm_cuda PUBLIC lux_mpcvm_core)
endif()

# ---- tests ----------------------------------------------------------------

if(LUX_MPCVM_BUILD_TESTS)
    enable_testing()

    # Layout test — host-only.
    add_executable(mpcvm-layout-test test/mpcvm_layout_test.cpp)
    target_link_libraries(mpcvm-layout-test PRIVATE lux_mpcvm_core)
    target_compile_features(mpcvm-layout-test PRIVATE cxx_std_20)
    add_test(NAME mpcvm-layout-test COMMAND mpcvm-layout-test)

    # GPU engine test — Apple-only (.mm).
    if(LUX_MPCVM_ENABLE_METAL AND APPLE)
        add_executable(mpcvm-gpu-engine-test test/mpcvm_gpu_engine_test.mm)
        target_link_libraries(mpcvm-gpu-engine-test PRIVATE lux_mpcvm_metal)
        target_compile_features(mpcvm-gpu-engine-test PRIVATE cxx_std_20)
        add_test(NAME mpcvm-gpu-engine-test COMMAND mpcvm-gpu-engine-test)
    endif()

    # Cross-backend determinism — links to whichever GPU backend is enabled.
    add_executable(mpcvm-determinism-test test/mpcvm_determinism_test.cpp)
    target_compile_features(mpcvm-determinism-test PRIVATE cxx_std_20)
    if(LUX_MPCVM_ENABLE_METAL AND APPLE)
        target_link_libraries(mpcvm-determinism-test PRIVATE lux_mpcvm_metal)
    elseif(LUX_MPCVM_ENABLE_CUDA)
        target_link_libraries(mpcvm-determinism-test PRIVATE lux_mpcvm_cuda)
    else()
        target_sources(mpcvm-determinism-test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/mpcvm_engine_stub.cpp)
        target_link_libraries(mpcvm-determinism-test PRIVATE lux_mpcvm_core)
    endif()
    add_test(NAME mpcvm-determinism-test COMMAND mpcvm-determinism-test)
endif()

message(STATUS "")
message(STATUS "=== Lux MPCVM Configuration ===")
message(STATUS "  Version:      ${PROJECT_VERSION}")
message(STATUS "  Build Tests:  ${LUX_MPCVM_BUILD_TESTS}")
message(STATUS "  Metal:        ${LUX_MPCVM_ENABLE_METAL}")
message(STATUS "  CUDA:         ${LUX_MPCVM_ENABLE_CUDA}")
message(STATUS "  WebGPU:       ${LUX_MPCVM_ENABLE_WGPU}")
message(STATUS "")
