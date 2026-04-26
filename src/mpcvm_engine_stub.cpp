// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_engine_stub.cpp — CPU-only fallback for MPCVMGPUEngine::create().
// Linked when neither Metal nor CUDA is enabled, so tests still build.
// The stub returns nullptr, exercising the CPU-only test paths.

#include "lux/mpcvm/mpcvm_gpu_engine.hpp"

namespace mpcvm::gpu {

#if !defined(__APPLE__)
std::unique_ptr<MPCVMGPUEngine> MPCVMGPUEngine::create() {
    return nullptr;
}
#else
__attribute__((weak)) std::unique_ptr<MPCVMGPUEngine> MPCVMGPUEngine::create() {
    return nullptr;
}
#endif

}  // namespace mpcvm::gpu
