// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_ceremony.metal — state-machine kernel.
//
// One round = three sequential phases (run inline by this kernel):
//   1. Apply CeremonyOps (begin / cancel) in supplied order.
//   2. Apply ContributionOps (with dedup) in supplied order.
//   3. Ceremony sweep:
//        * threshold met in current round -> advance round / finalize
//        * deadline elapsed and threshold not met -> mark failed
//      For finalized keygen ceremonies, emit deterministic key shares.
//
// Single-thread execution preserves byte-for-byte determinism with the
// CPU reference and all other backends.

#include "mpcvm_kernels_common.h.metal"

// Count contributions for a given (ceremony, round).
inline uint count_contributions_for(device const Contribution* contributions,
                                    uint contribution_count,
                                    ulong ceremony_id, uint round)
{
    uint n = 0;
    for (uint i = 0; i < contribution_count; ++i) {
        device const Contribution& c = contributions[i];
        if (c.status != 1u) continue;
        if (c.ceremony_id == ceremony_id && c.round == round) ++n;
    }
    return n;
}

// Emit deterministic key shares for a finalized keygen ceremony.
// share_data := keccak-stretch( ceremony_seed || ceremony_id || holder ||
//                              "MPCVM-SHARE-V1" || all_round_payloads )
// truncated to scheme-specific length.
inline void emit_keygen_shares(device const Ceremony& c,
                               device const Contribution* contributions,
                               uint contribution_count,
                               device KeyShare* shares,
                               uint share_count,
                               thread ulong& next_share_id)
{
    uint scheme = scheme_for_kind(c.kind);
    uint out_len = share_data_len_for_scheme(scheme);
    uint total_rounds = total_rounds_for(c.kind);

    for (uint holder = 0; holder < c.total_participants; ++holder) {
        ulong bit = (ulong)1 << holder;
        if ((c.participants_bitmap & bit) == 0u) continue;

        // Build seed buffer (cap at safe upper bound: prefix 58 + max 5 rounds * 384).
        uchar buf[2048];
        uint o = 0;
        for (uint k = 0; k < 32u; ++k) buf[o++] = c.ceremony_seed[k];
        absorb_u64(buf, o, c.ceremony_id); o += 8;
        absorb_u32(buf, o, holder);        o += 4;
        const uchar tag[14] = {'M','P','C','V','M','-','S','H','A','R','E','-','V','1'};
        for (uint k = 0; k < 14u; ++k) buf[o++] = tag[k];
        for (uint r = 0; r < total_rounds; ++r) {
            // find contribution for (cid, r, holder)
            for (uint i = 0; i < contribution_count; ++i) {
                device const Contribution& cn = contributions[i];
                if (cn.status != 1u) continue;
                if (cn.ceremony_id != c.ceremony_id) continue;
                if (cn.round != r) continue;
                if (cn.holder_index != holder) continue;
                for (uint k = 0; k < cn.payload_len && o < 2048u; ++k)
                    buf[o++] = cn.payload[k];
                break;
            }
        }

        // Reserve slot.
        uint kidx = key_share_locate_free(shares, share_count, c.ceremony_id, holder);
        if (kidx == 0xFFFFFFFFu) continue;
        device KeyShare& ks = shares[kidx];
        if (ks.occupied == 0u) {
            ks.share_id = next_share_id++;
            ks.ceremony_id = c.ceremony_id;
            ks.holder_addr = 0;
            ks.holder_index = holder;
            ks.scheme = scheme;
            ks.occupied = 1u;
        }
        ks.share_data_len = out_len;

        uchar prev[32];
        keccak256(buf, (ulong)o, prev);
        uint written = 0;
        while (written < out_len) {
            uint take = (out_len - written < 32u) ? (out_len - written) : 32u;
            for (uint k = 0; k < take; ++k) ks.share_data[written + k] = prev[k];
            written += take;
            if (written < out_len) {
                uchar ext[33];
                for (uint k = 0; k < 32u; ++k) ext[k] = prev[k];
                ext[32] = (uchar)(written / 32u);
                keccak256(ext, 33u, prev);
            }
        }
    }
}

kernel void mpcvm_ceremony_step(
    device const MPCVMRoundDescriptor* desc                    [[buffer(0)]],
    device const CeremonyOp*           ceremony_ops            [[buffer(1)]],
    device const ContributionOp*       contribution_ops        [[buffer(2)]],
    device Ceremony*                   ceremonies              [[buffer(3)]],
    device KeyShare*                   key_shares              [[buffer(4)]],
    device Contribution*               contributions           [[buffer(5)]],
    device atomic_uint*                ceremony_applied_out    [[buffer(6)]],
    device atomic_uint*                contribution_applied_out [[buffer(7)]],
    device atomic_uint*                round_advance_out        [[buffer(8)]],
    device atomic_uint*                finalized_out            [[buffer(9)]],
    device atomic_uint*                failed_out               [[buffer(10)]],
    constant uint&                     ceremony_count           [[buffer(11)]],
    constant uint&                     key_share_count          [[buffer(12)]],
    constant uint&                     contribution_count       [[buffer(13)]],
    constant ulong&                    next_contribution_id_in  [[buffer(14)]],
    constant ulong&                    next_share_id_in         [[buffer(15)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;

    uint cer_applied = 0;
    uint cnt_applied = 0;
    uint advances = 0;
    uint finalized = 0;
    uint failed = 0;
    ulong next_cont_id = next_contribution_id_in;
    ulong next_sh_id = next_share_id_in;

    // Phase 1: ceremony ops.
    uint cer_op_count = desc->ceremony_op_count;
    for (uint i = 0; i < cer_op_count; ++i) {
        device const CeremonyOp& op = ceremony_ops[i];
        if (op.kind == kCeremonyOpBegin) {
            if (op.threshold == 0u) continue;
            if (op.threshold > op.total_participants) continue;
            if (op.total_participants > 64u) continue;
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
            ++cer_applied;
        } else if (op.kind == kCeremonyOpCancel) {
            uint idx = ceremony_locate(ceremonies, ceremony_count, op.ceremony_id, false);
            if (idx == 0xFFFFFFFFu) continue;
            device Ceremony& c = ceremonies[idx];
            if (c.status != kCeremonyStatusInProgress) continue;
            c.status = kCeremonyStatusFailed;
            ++cer_applied;
        }
    }

    // Phase 2: contribution ops.
    uint cnt_op_count = desc->contribution_op_count;
    for (uint i = 0; i < cnt_op_count; ++i) {
        device const ContributionOp& op = contribution_ops[i];
        if (op.payload_len > 384u) continue;
        uint cidx = ceremony_locate(ceremonies, ceremony_count, op.ceremony_id, false);
        if (cidx == 0xFFFFFFFFu) continue;
        device Ceremony& c = ceremonies[cidx];
        if (c.status != kCeremonyStatusInProgress) continue;
        if (op.round != c.round) continue;
        if (op.holder_index >= c.total_participants) continue;

        uint existing = contribution_locate(contributions, contribution_count,
                                            op.ceremony_id, op.round, op.holder_index, false);
        if (existing != 0xFFFFFFFFu) continue;

        uint nidx = contribution_locate(contributions, contribution_count,
                                        op.ceremony_id, op.round, op.holder_index, true);
        if (nidx == 0xFFFFFFFFu) continue;
        device Contribution& cn = contributions[nidx];
        cn.contribution_id = next_cont_id++;
        cn.holder_addr = op.holder_addr;
        cn.payload_len = op.payload_len;
        for (uint k = 0; k < op.payload_len; ++k) cn.payload[k] = op.payload[k];

        ulong bit = (ulong)1 << op.holder_index;
        if ((c.participants_bitmap & bit) == 0u) {
            c.participants_bitmap |= bit;
        }
        ++cnt_applied;
    }

    // Phase 3: sweep (advance / finalize / timeout).
    for (uint i = 0; i < ceremony_count; ++i) {
        device Ceremony& c = ceremonies[i];
        if (c.status != kCeremonyStatusInProgress) continue;

        uint in_round = count_contributions_for(contributions, contribution_count,
                                                c.ceremony_id, c.round);
        c.contribution_count = in_round;

        if (in_round >= c.threshold) {
            uint total_rounds = total_rounds_for(c.kind);
            ++c.round;
            ++advances;
            if (c.round >= total_rounds) {
                c.status = kCeremonyStatusFinalized;
                ++finalized;
                if (is_keygen_kind(c.kind)) {
                    emit_keygen_shares(c, contributions, contribution_count,
                                       key_shares, key_share_count, next_sh_id);
                }
            } else {
                c.participants_bitmap = 0;
                c.contribution_count = 0;
            }
            continue;
        }

        if (desc->timestamp_ns > c.deadline_ns) {
            c.status = kCeremonyStatusFailed;
            ++failed;
        }
    }

    atomic_store_explicit(ceremony_applied_out,     cer_applied, memory_order_relaxed);
    atomic_store_explicit(contribution_applied_out, cnt_applied, memory_order_relaxed);
    atomic_store_explicit(round_advance_out,        advances,    memory_order_relaxed);
    atomic_store_explicit(finalized_out,            finalized,   memory_order_relaxed);
    atomic_store_explicit(failed_out,               failed,      memory_order_relaxed);
}
