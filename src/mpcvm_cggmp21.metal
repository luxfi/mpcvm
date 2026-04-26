// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_cggmp21.metal — CGGMP21 kernel entry.
//
// CGGMP21 keygen: 3 rounds (commitments / broadcast / Feldman VSS).
// CGGMP21 sign:   5 rounds (4 offline MtA + 1 online).
//
// Single-thread canonical traversal — only processes ceremony ops whose
// kind is Cggmp21Keygen or Cggmp21Sign.

#include "mpcvm_kernels_common.h.metal"

inline bool cggmp21_kind(uint k) {
    return k == kKindCggmp21Keygen || k == kKindCggmp21Sign;
}

kernel void mpcvm_cggmp21_step(
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
        if (!cggmp21_kind(op.ceremony_kind)) continue;
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
