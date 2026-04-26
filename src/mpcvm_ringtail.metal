// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_ringtail.metal — Ringtail kernel entry.
//
// Ringtail DKG:  2 rounds.
// Ringtail sign: 2 rounds.
//
// The lattice arithmetic (NTT, sampling, Module-LWE multiplication) lives
// in luxcpp/lattice/src/metal/ — this kernel only handles the ceremony
// state machine + the deterministic share envelope. Full LWE response
// verification is performed by the host via the existing Ringtail
// verifier (luxcpp/cevm/lib/consensus/quasar/gpu/quasar_ringtail_verifier.cpp).

#include "mpcvm_kernels_common.h.metal"

inline bool ringtail_kind(uint k) {
    return k == kKindRingtailDkg || k == kKindRingtailSign;
}

kernel void mpcvm_ringtail_step(
    device const MPCVMRoundDescriptor* desc                    [[buffer(0)]],
    device const CeremonyOp*           ceremony_ops            [[buffer(1)]],
    device const ContributionOp*       contribution_ops        [[buffer(2)]],
    device Ceremony*                   ceremonies              [[buffer(3)]],
    device atomic_uint*                applied_out             [[buffer(4)]],
    constant uint&                     ceremony_count          [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;
    uint applied = 0;
    uint cer_op_count = desc->ceremony_op_count;
    for (uint i = 0; i < cer_op_count; ++i) {
        device const CeremonyOp& op = ceremony_ops[i];
        if (!ringtail_kind(op.ceremony_kind)) continue;
        if (op.kind == kCeremonyOpBegin) {
            uint idx = ceremony_locate(ceremonies, ceremony_count, op.ceremony_id, true);
            if (idx == 0xFFFFFFFFu) continue;
            device Ceremony& c = ceremonies[idx];
            c.kind = op.ceremony_kind;
            c.threshold = op.threshold;
            c.total_participants = op.total_participants;
            c.deadline_ns = op.deadline_ns;
            c.round = 0;
            c.contribution_count = 0;
            c.participants_bitmap = 0;
            c.status = kCeremonyStatusInProgress;
            for (uint k = 0; k < 32u; ++k) c.subject[k] = op.subject[k];
            for (uint k = 0; k < 32u; ++k) c.ceremony_seed[k] = op.ceremony_seed[k];
            ++applied;
        }
    }
    atomic_store_explicit(applied_out, applied, memory_order_relaxed);
}
