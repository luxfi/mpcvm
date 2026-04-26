// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file mpcvm_gpu_engine.hpp
/// MPCVMGPUEngine — GPU-native M-Chain ceremony substrate.
///
/// Lifecycle (mirrors PVMGPUEngine):
///   begin_round(MPCVMRoundDescriptor)
///   push_ceremony_ops / push_contribution_ops
///   run_epoch / run_until_done
///   poll_round_result -> MPCVMTransitionResult (with mpcvm_state_root)
///   end_round
///
/// One MPCVMTransitionResult.mpcvm_state_root equals the Quasar round
/// descriptor's mpcvm_state_root for the same epoch — this is the
/// linkage that makes the M-Chain GPU-native under LP-137.

#pragma once

#include "mpcvm_gpu_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace mpcvm::gpu {

struct MPCVMRoundHandle {
    uint64_t opaque = 0;
    bool valid() const { return opaque != 0; }
};

class MPCVMGPUEngine {
public:
    virtual ~MPCVMGPUEngine() = default;

    static std::unique_ptr<MPCVMGPUEngine> create();

    virtual MPCVMRoundHandle begin_round(const MPCVMRoundDescriptor& desc) = 0;

    virtual void push_ceremony_ops(MPCVMRoundHandle h,
                                   std::span<const CeremonyOp> ops) = 0;
    virtual void push_contribution_ops(MPCVMRoundHandle h,
                                       std::span<const ContributionOp> ops) = 0;

    virtual MPCVMTransitionResult run_epoch(MPCVMRoundHandle h) = 0;
    virtual MPCVMTransitionResult run_until_done(MPCVMRoundHandle h,
                                                 std::size_t max_epochs = 64) = 0;
    virtual MPCVMTransitionResult poll_round_result(MPCVMRoundHandle h) const = 0;

    virtual void end_round(MPCVMRoundHandle h) = 0;

    virtual bool round_active() const = 0;
    virtual const char* device_name() const = 0;

protected:
    MPCVMGPUEngine() = default;
    MPCVMGPUEngine(const MPCVMGPUEngine&) = delete;
    MPCVMGPUEngine& operator=(const MPCVMGPUEngine&) = delete;
};

}  // namespace mpcvm::gpu
