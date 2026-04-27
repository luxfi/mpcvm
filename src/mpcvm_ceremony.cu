// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_ceremony.cu — v0.62 per-slot fan-out kernels (CUDA peer of
// mpcvm_ceremony.metal). Two kernels:
//   1. mpcvm_ceremony_apply  (1 block × 1 thread) — Phase 1+2 ops apply.
//   2. mpcvm_ceremony_sweep  (1 block × kCeremonySlots threads) — Phase 3
//      sweep with intra-block prefix sum for share_id allocation.
//
// Determinism: slot ordering = canonical ordering. share_ids assigned by
// exclusive prefix sum over emit counts.

#include "mpcvm_kernels_common.cuh"

namespace mpcvm::cuda {

constexpr uint32_t kSweepWorkgroupSize = 256u;

__device__ inline void emit_keygen_shares_for(const Ceremony& c,
                                              Contribution* contributions,
                                              uint32_t contribution_count,
                                              KeyShare* shares,
                                              uint32_t share_count,
                                              uint64_t& next_share_id) {
    uint32_t scheme = scheme_for_kind(c.kind);
    uint32_t out_len = share_data_len_for_scheme(scheme);
    uint32_t total_rounds = total_rounds_for(c.kind);

    for (uint32_t holder = 0; holder < c.total_participants; ++holder) {
        uint64_t bit = uint64_t(1) << holder;
        if ((c.participants_bitmap & bit) == 0u) continue;

        uint8_t buf[2048];
        uint32_t o = 0;
        for (uint32_t k = 0; k < 32u; ++k) buf[o++] = c.ceremony_seed[k];
        absorb_u64(buf, o, c.ceremony_id); o += 8;
        absorb_u32(buf, o, holder);        o += 4;
        const uint8_t tag[14] = {'M','P','C','V','M','-','S','H','A','R','E','-','V','1'};
        for (uint32_t k = 0; k < 14u; ++k) buf[o++] = tag[k];
        // Hash-lookup the contribution for (cid, r, holder) — O(1) vs O(N).
        for (uint32_t r = 0; r < total_rounds; ++r) {
            uint32_t ci = contribution_locate(contributions, contribution_count,
                                              c.ceremony_id, r, holder, false);
            if (ci == 0xFFFFFFFFu) continue;
            const Contribution& cn = contributions[ci];
            for (uint32_t k = 0; k < cn.payload_len && o < 2048u; ++k)
                buf[o++] = cn.payload[k];
        }

        uint32_t kidx = key_share_locate_free(shares, share_count, c.ceremony_id, holder);
        if (kidx == 0xFFFFFFFFu) continue;
        KeyShare& ks = shares[kidx];
        if (ks.occupied == 0u) {
            ks.share_id = next_share_id++;
            ks.ceremony_id = c.ceremony_id;
            ks.holder_addr = 0;
            ks.holder_index = holder;
            ks.scheme = scheme;
            ks.occupied = 1u;
        }
        ks.share_data_len = out_len;

        uint8_t prev[32];
        keccak256(buf, uint64_t(o), prev);
        uint32_t written = 0;
        while (written < out_len) {
            uint32_t take = (out_len - written < 32u) ? (out_len - written) : 32u;
            for (uint32_t k = 0; k < take; ++k) ks.share_data[written + k] = prev[k];
            written += take;
            if (written < out_len) {
                uint8_t ext[33];
                for (uint32_t k = 0; k < 32u; ++k) ext[k] = prev[k];
                ext[32] = uint8_t(written / 32u);
                keccak256(ext, 33u, prev);
            }
        }
    }
}

extern "C" __global__ void mpcvm_ceremony_apply(
    const MPCVMRoundDescriptor* desc,
    const CeremonyOp*           ceremony_ops,
    const ContributionOp*       contribution_ops,
    Ceremony*                   ceremonies,
    Contribution*               contributions,
    uint32_t*                   ceremony_applied_out,
    uint32_t*                   contribution_applied_out,
    uint32_t                    ceremony_count,
    uint32_t                    contribution_count,
    uint64_t                    next_contribution_id_in)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t cer_applied = 0;
    uint32_t cnt_applied = 0;
    uint64_t next_cont_id = next_contribution_id_in;

    uint32_t cer_op_count = desc->ceremony_op_count;
    for (uint32_t i = 0; i < cer_op_count; ++i) {
        const CeremonyOp& op = ceremony_ops[i];
        if (op.kind == kCeremonyOpBegin) {
            if (op.threshold == 0u) continue;
            if (op.threshold > op.total_participants) continue;
            if (op.total_participants > 64u) continue;
            uint32_t idx = ceremony_locate(ceremonies, ceremony_count, op.ceremony_id, true);
            if (idx == 0xFFFFFFFFu) continue;
            Ceremony& c = ceremonies[idx];
            c.kind = op.ceremony_kind;
            c.threshold = op.threshold;
            c.total_participants = op.total_participants;
            c.deadline_ns = op.deadline_ns;
            c.round = 0;
            c.contribution_count = 0;
            c.participants_bitmap = 0;
            c.status = kCeremonyStatusInProgress;
            for (uint32_t k = 0; k < 32u; ++k) c.subject[k] = op.subject[k];
            for (uint32_t k = 0; k < 32u; ++k) c.ceremony_seed[k] = op.ceremony_seed[k];
            ++cer_applied;
        } else if (op.kind == kCeremonyOpCancel) {
            uint32_t idx = ceremony_locate(ceremonies, ceremony_count, op.ceremony_id, false);
            if (idx == 0xFFFFFFFFu) continue;
            Ceremony& c = ceremonies[idx];
            if (c.status != kCeremonyStatusInProgress) continue;
            c.status = kCeremonyStatusFailed;
            ++cer_applied;
        }
    }

    uint32_t cnt_op_count = desc->contribution_op_count;
    for (uint32_t i = 0; i < cnt_op_count; ++i) {
        const ContributionOp& op = contribution_ops[i];
        if (op.payload_len > 384u) continue;
        uint32_t cidx = ceremony_locate(ceremonies, ceremony_count, op.ceremony_id, false);
        if (cidx == 0xFFFFFFFFu) continue;
        Ceremony& c = ceremonies[cidx];
        if (c.status != kCeremonyStatusInProgress) continue;
        if (op.round != c.round) continue;
        if (op.holder_index >= c.total_participants) continue;

        uint32_t existing = contribution_locate(contributions, contribution_count,
                                                op.ceremony_id, op.round, op.holder_index, false);
        if (existing != 0xFFFFFFFFu) continue;

        uint32_t nidx = contribution_locate(contributions, contribution_count,
                                            op.ceremony_id, op.round, op.holder_index, true);
        if (nidx == 0xFFFFFFFFu) continue;
        Contribution& cn = contributions[nidx];
        cn.contribution_id = next_cont_id++;
        cn.holder_addr = op.holder_addr;
        cn.payload_len = op.payload_len;
        for (uint32_t k = 0; k < op.payload_len; ++k) cn.payload[k] = op.payload[k];

        uint64_t bit = uint64_t(1) << op.holder_index;
        if ((c.participants_bitmap & bit) == 0u) {
            c.participants_bitmap |= bit;
        }
        ++cnt_applied;
    }

    *ceremony_applied_out     = cer_applied;
    *contribution_applied_out = cnt_applied;
}

// Per-slot fan-out sweep. Launch with <<<1, 256, sizeof(uint32_t)*256>>>.
extern "C" __global__ void mpcvm_ceremony_sweep(
    const MPCVMRoundDescriptor* desc,
    Ceremony*                   ceremonies,
    KeyShare*                   key_shares,
    Contribution*               contributions,
    uint32_t*                   round_advance_out,
    uint32_t*                   finalized_out,
    uint32_t*                   failed_out,
    uint32_t                    ceremony_count,
    uint32_t                    key_share_count,
    uint32_t                    contribution_count,
    uint64_t                    next_share_id_in)
{
    extern __shared__ uint32_t emit_counts[];
    uint32_t tid = threadIdx.x;
    bool in_range = tid < ceremony_count;

    uint32_t local_emit_count = 0u;
    bool will_finalize_keygen = false;
    bool will_advance         = false;
    bool will_timeout         = false;
    uint32_t in_round_total   = 0u;
    if (in_range) {
        Ceremony& c = ceremonies[tid];
        if (c.status == kCeremonyStatusInProgress) {
            uint32_t in_round = 0;
            for (uint32_t i = 0; i < contribution_count; ++i) {
                const Contribution& cn = contributions[i];
                if (cn.status != 1u) continue;
                if (cn.ceremony_id == c.ceremony_id && cn.round == c.round) ++in_round;
            }
            in_round_total = in_round;

            if (in_round >= c.threshold) {
                will_advance = true;
                uint32_t total_rounds = total_rounds_for(c.kind);
                if (c.round + 1u >= total_rounds && is_keygen_kind(c.kind)) {
                    will_finalize_keygen = true;
                    for (uint32_t h = 0; h < c.total_participants; ++h) {
                        uint64_t bit = uint64_t(1) << h;
                        if ((c.participants_bitmap & bit) != 0u) ++local_emit_count;
                    }
                }
            } else if (desc->timestamp_ns > c.deadline_ns) {
                will_timeout = true;
            }
        }
    }
    emit_counts[tid] = local_emit_count;

    __syncthreads();

    // Prefix sum on tid 0 (small N=256).
    if (tid == 0u) {
        uint32_t acc = 0u;
        for (uint32_t i = 0; i < kSweepWorkgroupSize; ++i) {
            uint32_t v = emit_counts[i];
            emit_counts[i] = acc;
            acc += v;
        }
    }

    __syncthreads();

    if (in_range) {
        Ceremony& c = ceremonies[tid];
        if (c.status == kCeremonyStatusInProgress) {
            c.contribution_count = in_round_total;
            if (will_advance) {
                ++c.round;
                atomicAdd(round_advance_out, 1u);
                uint32_t total_rounds = total_rounds_for(c.kind);
                if (c.round >= total_rounds) {
                    c.status = kCeremonyStatusFinalized;
                    atomicAdd(finalized_out, 1u);
                    if (will_finalize_keygen) {
                        uint64_t base = next_share_id_in + uint64_t(emit_counts[tid]);
                        emit_keygen_shares_for(c, contributions, contribution_count,
                                               key_shares, key_share_count, base);
                    }
                } else {
                    c.participants_bitmap = 0;
                    c.contribution_count = 0;
                }
            } else if (will_timeout) {
                c.status = kCeremonyStatusFailed;
                atomicAdd(failed_out, 1u);
            }
        }
    }
}

}  // namespace mpcvm::cuda
