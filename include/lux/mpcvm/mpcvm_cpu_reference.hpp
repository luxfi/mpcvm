// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file mpcvm_cpu_reference.hpp
/// CPU reference implementation of the MPCVM transition kernels — the
/// differential-fuzz oracle for cross-backend determinism (CPU vs Metal
/// vs CUDA vs WGSL must produce byte-identical roots on the same input).
///
/// The reference processes input ops in canonical order:
///   1. Ceremony begin / cancel ops (in supplied order)
///   2. Contribution ops (in supplied order; dedup per (cid, round, holder))
///   3. CeremonyStateMachine sweep (advance round / timeout absent)
///   4. MpcvmTransition (closes round and emits roots)
///
/// State carries forward across run_reference() calls only via the
/// arenas the caller threads through — the reference is pure on
/// (state, ops) and does not retain hidden state.

#pragma once

#include "mpcvm_gpu_layout.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace mpcvm::gpu::ref {

struct MPCVMReferenceState {
    std::vector<Ceremony>     ceremonies;       ///< sized to kDefaultCeremonySlots
    std::vector<KeyShare>     key_shares;       ///< sized to kDefaultKeyShareSlots
    std::vector<Contribution> contributions;    ///< sized to kDefaultContributionSlots
    MPCVMState                state{};
    uint64_t                  next_share_id = 1;
    uint64_t                  next_contribution_id = 1;

    static MPCVMReferenceState empty();
};

MPCVMTransitionResult run_reference(MPCVMReferenceState& state,
                                    const MPCVMRoundDescriptor& desc,
                                    std::span<const CeremonyOp>     ceremony_ops,
                                    std::span<const ContributionOp> contribution_ops);

}  // namespace mpcvm::gpu::ref
