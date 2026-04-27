// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_ceremony.metal — v0.62 per-slot fan-out kernels.
//
// One round = two kernel dispatches:
//   1. mpcvm_ceremony_apply (1x1x1) — Phase 1 (begin/cancel) + Phase 2
//      (contributions), processed in canonical input-stream order.
//      Counter increments (next_contribution_id) and slot placement match
//      the CPU reference byte-for-byte.
//   2. mpcvm_ceremony_sweep (gridSize = ceremony_count, workgroup_size =
//      ceremony_count) — Phase 3 sweep with intra-threadgroup prefix sum
//      for deterministic share_id assignment.
//
// Determinism: slot-order = canonical-order (open-addressing hash is
// deterministic), so per-slot fan-out preserves byte-equality with CPU.
// share_ids are allocated by a prefix sum over per-slot emit-counts so the
// same (ceremony_slot, holder) gets the same share_id as the CPU reference.

#include "mpcvm_kernels_common.h.metal"

// kCeremonySlots — must match kDefaultCeremonySlots in mpcvm_gpu_layout.hpp.
constant uint kSweepWorkgroupSize = 256u;

inline void emit_keygen_shares_for(device const Ceremony& c,
                                   device Contribution* contributions,
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

        uchar buf[2048];
        uint o = 0;
        for (uint k = 0; k < 32u; ++k) buf[o++] = c.ceremony_seed[k];
        absorb_u64(buf, o, c.ceremony_id); o += 8;
        absorb_u32(buf, o, holder);        o += 4;
        const uchar tag[14] = {'M','P','C','V','M','-','S','H','A','R','E','-','V','1'};
        for (uint k = 0; k < 14u; ++k) buf[o++] = tag[k];
        // Hash-lookup the contribution for (cid, r, holder) instead of
        // scanning the whole table — O(1) vs O(N).
        for (uint r = 0; r < total_rounds; ++r) {
            uint ci = contribution_locate(contributions, contribution_count,
                                          c.ceremony_id, r, holder, false);
            if (ci == 0xFFFFFFFFu) continue;
            device const Contribution& cn = contributions[ci];
            for (uint k = 0; k < cn.payload_len && o < 2048u; ++k)
                buf[o++] = cn.payload[k];
        }

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

// =============================================================================
// Phase 1+2: serial ops apply (canonical input-stream order).
// =============================================================================

kernel void mpcvm_ceremony_apply(
    device const MPCVMRoundDescriptor* desc                    [[buffer(0)]],
    device const CeremonyOp*           ceremony_ops            [[buffer(1)]],
    device const ContributionOp*       contribution_ops        [[buffer(2)]],
    device Ceremony*                   ceremonies              [[buffer(3)]],
    device Contribution*               contributions           [[buffer(4)]],
    device atomic_uint*                ceremony_applied_out    [[buffer(5)]],
    device atomic_uint*                contribution_applied_out [[buffer(6)]],
    constant uint&                     ceremony_count           [[buffer(7)]],
    constant uint&                     contribution_count       [[buffer(8)]],
    constant ulong&                    next_contribution_id_in  [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;

    uint cer_applied = 0;
    uint cnt_applied = 0;
    ulong next_cont_id = next_contribution_id_in;

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

    atomic_store_explicit(ceremony_applied_out,     cer_applied, memory_order_relaxed);
    atomic_store_explicit(contribution_applied_out, cnt_applied, memory_order_relaxed);
}

// =============================================================================
// Phase 3: per-slot fan-out sweep with intra-threadgroup prefix sum for
// deterministic share_id assignment.
//
// Layout: dispatchThreads = (kCeremonySlots, 1, 1),
//         threadsPerThreadgroup = (kCeremonySlots, 1, 1).
// All threads live in one threadgroup so threadgroup_barrier syncs the lot.
// =============================================================================

kernel void mpcvm_ceremony_sweep(
    device const MPCVMRoundDescriptor* desc                    [[buffer(0)]],
    device Ceremony*                   ceremonies              [[buffer(1)]],
    device KeyShare*                   key_shares              [[buffer(2)]],
    device Contribution*               contributions           [[buffer(3)]],
    device atomic_uint*                round_advance_out       [[buffer(4)]],
    device atomic_uint*                finalized_out           [[buffer(5)]],
    device atomic_uint*                failed_out              [[buffer(6)]],
    constant uint&                     ceremony_count          [[buffer(7)]],
    constant uint&                     key_share_count         [[buffer(8)]],
    constant uint&                     contribution_count      [[buffer(9)]],
    constant ulong&                    next_share_id_in        [[buffer(10)]],
    threadgroup uint*                  emit_counts             [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]])
{
    // Each tid owns ceremonies[tid]. Threads beyond ceremony_count idle.
    bool in_range = tid < ceremony_count;

    // Phase A: per-slot decision (advance/finalize/timeout) and emit_count.
    uint local_emit_count = 0u;
    bool will_finalize_keygen = false;
    bool will_advance         = false;
    bool will_timeout         = false;
    uint in_round_total       = 0u;
    if (in_range) {
        device Ceremony& c = ceremonies[tid];
        if (c.status == kCeremonyStatusInProgress) {
            uint in_round = 0;
            for (uint i = 0; i < contribution_count; ++i) {
                device const Contribution& cn = contributions[i];
                if (cn.status != 1u) continue;
                if (cn.ceremony_id == c.ceremony_id && cn.round == c.round) ++in_round;
            }
            in_round_total = in_round;

            if (in_round >= c.threshold) {
                will_advance = true;
                uint total_rounds = total_rounds_for(c.kind);
                if (c.round + 1u >= total_rounds && is_keygen_kind(c.kind)) {
                    will_finalize_keygen = true;
                    for (uint h = 0; h < c.total_participants; ++h) {
                        ulong bit = (ulong)1 << h;
                        if ((c.participants_bitmap & bit) != 0u) ++local_emit_count;
                    }
                }
            } else if (desc->timestamp_ns > c.deadline_ns) {
                will_timeout = true;
            }
        }
    }
    emit_counts[tid] = local_emit_count;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase B: prefix sum over emit_counts on tid 0 (small N=256).
    // We compute exclusive prefix sum so emit_counts[tid] becomes the
    // base_share_id offset for ceremony slot tid.
    if (tid == 0u) {
        uint acc = 0u;
        for (uint i = 0; i < kSweepWorkgroupSize; ++i) {
            uint v = emit_counts[i];
            emit_counts[i] = acc;
            acc += v;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase C: actually advance/finalize/timeout, emitting shares with the
    // correct base_share_id.
    if (in_range) {
        device Ceremony& c = ceremonies[tid];
        if (c.status == kCeremonyStatusInProgress) {
            c.contribution_count = in_round_total;
            if (will_advance) {
                ++c.round;
                atomic_fetch_add_explicit(round_advance_out, 1u, memory_order_relaxed);
                uint total_rounds = total_rounds_for(c.kind);
                if (c.round >= total_rounds) {
                    c.status = kCeremonyStatusFinalized;
                    atomic_fetch_add_explicit(finalized_out, 1u, memory_order_relaxed);
                    if (will_finalize_keygen) {
                        ulong base = next_share_id_in + (ulong)emit_counts[tid];
                        emit_keygen_shares_for(c, contributions, contribution_count,
                                               key_shares, key_share_count, base);
                    }
                } else {
                    c.participants_bitmap = 0;
                    c.contribution_count = 0;
                }
            } else if (will_timeout) {
                c.status = kCeremonyStatusFailed;
                atomic_fetch_add_explicit(failed_out, 1u, memory_order_relaxed);
            }
        }
    }
}
