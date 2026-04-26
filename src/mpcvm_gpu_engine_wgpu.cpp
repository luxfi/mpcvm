// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_gpu_engine_wgpu.cpp — WebGPU/Dawn driver entry.
//
// In v0.60 the WGSL kernels are structurally complete (layouts +
// ceremony begin handling) but the Dawn engine driver is deferred to
// v0.61 along with the in-shader keccak port. This file ships the
// MPCVMGPUEngine::create() weak fallback for the wgpu build path so
// linking succeeds in CPU-only configurations; the Metal and CUDA
// drivers provide the canonical create() in their respective builds.

#include "lux/mpcvm/mpcvm_gpu_engine.hpp"

namespace mpcvm::gpu {

#if !defined(__APPLE__) && !defined(LUX_MPCVM_HAVE_CUDA)
__attribute__((weak)) std::unique_ptr<MPCVMGPUEngine> MPCVMGPUEngine::create() {
    return nullptr;
}
#endif

}  // namespace mpcvm::gpu
