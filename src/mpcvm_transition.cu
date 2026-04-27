// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_transition.cu — v0.62 parallel leaf hashing + serial fold (CUDA).
//
// Two kernels:
//   1. mpcvm_compute_leaves (gridSize = max(slot counts), blockSize = 64) —
//      parallel keccak per leaf into precomputed leaf-hash arenas.
//   2. mpcvm_compose_root  (1 block × 1 thread) — serial fold over the
//      precomputed leaves + state-root composition + epoch advance.

#include "mpcvm_kernels_common.cuh"

namespace mpcvm::cuda {

extern "C" __global__ void mpcvm_compute_leaves(
    const Ceremony*    ceremonies,
    const KeyShare*    shares,
    const Contribution* contributions,
    uint8_t*           ceremony_leaf_hashes,
    uint8_t*           share_leaf_hashes,
    uint8_t*           contribution_leaf_hashes,
    uint32_t*          active_count_out,
    uint32_t*          finalized_count_out,
    uint32_t*          failed_count_out,
    uint32_t*          share_count_out,
    uint8_t*           ceremony_used_mask,
    uint8_t*           share_used_mask,
    uint8_t*           contribution_used_mask,
    uint32_t           ceremony_count,
    uint32_t           share_count,
    uint32_t           contribution_count)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < ceremony_count) {
        const Ceremony& c = ceremonies[tid];
        if (c.status == kCeremonyStatusFree) {
            ceremony_used_mask[tid] = 0u;
        } else {
            ceremony_used_mask[tid] = 1u;
            if (c.status == kCeremonyStatusInProgress) atomicAdd(active_count_out, 1u);
            else if (c.status == kCeremonyStatusFinalized) atomicAdd(finalized_count_out, 1u);
            else if (c.status == kCeremonyStatusFailed) atomicAdd(failed_count_out, 1u);

            uint8_t leaf[8 + 8 + 8 + 8 + 4 + 4 + 4 + 4 + 4 + 4 + 32 + 32 + 4];
            uint32_t o = 0;
            absorb_u64(leaf, o, c.ceremony_id);            o += 8;
            absorb_u64(leaf, o, c.started_at_ns);           o += 8;
            absorb_u64(leaf, o, c.deadline_ns);             o += 8;
            absorb_u64(leaf, o, c.participants_bitmap);     o += 8;
            absorb_u32(leaf, o, c.kind);                    o += 4;
            absorb_u32(leaf, o, c.round);                   o += 4;
            absorb_u32(leaf, o, c.threshold);               o += 4;
            absorb_u32(leaf, o, c.total_participants);      o += 4;
            absorb_u32(leaf, o, c.status);                  o += 4;
            absorb_u32(leaf, o, c.contribution_count);      o += 4;
            for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = c.subject[k];        o += 32;
            for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = c.ceremony_seed[k];  o += 32;
            absorb_u32(leaf, o, tid);                       o += 4;

            uint8_t h[32];
            keccak256(leaf, uint64_t(o), h);
            for (uint32_t k = 0; k < 32u; ++k) ceremony_leaf_hashes[tid * 32u + k] = h[k];
        }
    }

    if (tid < share_count) {
        const KeyShare& s = shares[tid];
        if (s.occupied == 0u) {
            share_used_mask[tid] = 0u;
        } else {
            share_used_mask[tid] = 1u;
            atomicAdd(share_count_out, 1u);

            uint8_t leaf[8 + 8 + 8 + 4 + 4 + 4 + 320 + 4];
            uint32_t o = 0;
            absorb_u64(leaf, o, s.share_id);     o += 8;
            absorb_u64(leaf, o, s.ceremony_id);  o += 8;
            absorb_u64(leaf, o, s.holder_addr);  o += 8;
            absorb_u32(leaf, o, s.scheme);       o += 4;
            absorb_u32(leaf, o, s.holder_index); o += 4;
            absorb_u32(leaf, o, s.share_data_len); o += 4;
            for (uint32_t k = 0; k < s.share_data_len && k < 320u; ++k)
                leaf[o + k] = s.share_data[k];
            o += s.share_data_len;
            absorb_u32(leaf, o, tid);            o += 4;

            uint8_t h[32];
            keccak256(leaf, uint64_t(o), h);
            for (uint32_t k = 0; k < 32u; ++k) share_leaf_hashes[tid * 32u + k] = h[k];
        }
    }

    if (tid < contribution_count) {
        const Contribution& c = contributions[tid];
        if (c.status != 1u) {
            contribution_used_mask[tid] = 0u;
        } else {
            contribution_used_mask[tid] = 1u;

            uint8_t leaf[8 + 8 + 8 + 4 + 4 + 4 + 384 + 4];
            uint32_t o = 0;
            absorb_u64(leaf, o, c.contribution_id); o += 8;
            absorb_u64(leaf, o, c.ceremony_id);     o += 8;
            absorb_u64(leaf, o, c.holder_addr);     o += 8;
            absorb_u32(leaf, o, c.round);           o += 4;
            absorb_u32(leaf, o, c.holder_index);    o += 4;
            absorb_u32(leaf, o, c.payload_len);     o += 4;
            for (uint32_t k = 0; k < c.payload_len && k < 384u; ++k)
                leaf[o + k] = c.payload[k];
            o += c.payload_len;
            absorb_u32(leaf, o, tid);               o += 4;

            uint8_t h[32];
            keccak256(leaf, uint64_t(o), h);
            for (uint32_t k = 0; k < 32u; ++k) contribution_leaf_hashes[tid * 32u + k] = h[k];
        }
    }
}

extern "C" __global__ void mpcvm_compose_root(
    const MPCVMRoundDescriptor* desc,
    const uint8_t*              ceremony_leaf_hashes,
    const uint8_t*              share_leaf_hashes,
    const uint8_t*              contribution_leaf_hashes,
    const uint8_t*              ceremony_used_mask,
    const uint8_t*              share_used_mask,
    const uint8_t*              contribution_used_mask,
    const uint32_t*             active_count_in,
    const uint32_t*             finalized_count_in,
    const uint32_t*             failed_count_in,
    const uint32_t*             share_count_in,
    MPCVMState*                 state,
    MPCVMTransitionResult*      result,
    uint32_t                    ceremony_count,
    uint32_t                    share_count,
    uint32_t                    contribution_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint8_t acc[32]; for (uint32_t k = 0; k < 32u; ++k) acc[k] = 0;
    for (uint32_t i = 0; i < ceremony_count; ++i) {
        if (ceremony_used_mask[i] == 0u) continue;
        uint8_t buf[64];
        for (uint32_t k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint32_t k = 0; k < 32u; ++k) buf[32 + k] = ceremony_leaf_hashes[i * 32u + k];
        keccak256(buf, 64u, acc);
    }
    for (uint32_t k = 0; k < 32u; ++k) state->ceremony_root[k] = acc[k];
    state->active_ceremony_count    = *active_count_in;
    state->finalized_ceremony_count = *finalized_count_in;
    state->failed_ceremony_count    = *failed_count_in;

    for (uint32_t k = 0; k < 32u; ++k) acc[k] = 0;
    for (uint32_t i = 0; i < share_count; ++i) {
        if (share_used_mask[i] == 0u) continue;
        uint8_t buf[64];
        for (uint32_t k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint32_t k = 0; k < 32u; ++k) buf[32 + k] = share_leaf_hashes[i * 32u + k];
        keccak256(buf, 64u, acc);
    }
    for (uint32_t k = 0; k < 32u; ++k) state->key_share_root[k] = acc[k];
    state->key_share_count = *share_count_in;

    for (uint32_t k = 0; k < 32u; ++k) acc[k] = 0;
    for (uint32_t i = 0; i < contribution_count; ++i) {
        if (contribution_used_mask[i] == 0u) continue;
        uint8_t buf[64];
        for (uint32_t k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint32_t k = 0; k < 32u; ++k) buf[32 + k] = contribution_leaf_hashes[i * 32u + k];
        keccak256(buf, 64u, acc);
    }
    for (uint32_t k = 0; k < 32u; ++k) state->contribution_root[k] = acc[k];

    state->now_ns = desc->timestamp_ns;
    if (desc->closing_flag != 0u) {
        state->current_epoch = desc->epoch + 1u;
    }

    uint8_t composed[32 + 32 + 32 + 32 + 8 + 8 + 4 + 4 + 4 + 4];
    uint32_t o = 0;
    for (uint32_t k = 0; k < 32u; ++k) composed[o + k] = desc->parent_state_root[k];   o += 32;
    for (uint32_t k = 0; k < 32u; ++k) composed[o + k] = state->ceremony_root[k];      o += 32;
    for (uint32_t k = 0; k < 32u; ++k) composed[o + k] = state->key_share_root[k];     o += 32;
    for (uint32_t k = 0; k < 32u; ++k) composed[o + k] = state->contribution_root[k];  o += 32;
    absorb_u64(composed, o, state->current_epoch);                  o += 8;
    absorb_u64(composed, o, state->now_ns);                          o += 8;
    absorb_u32(composed, o, state->active_ceremony_count);           o += 4;
    absorb_u32(composed, o, state->finalized_ceremony_count);        o += 4;
    absorb_u32(composed, o, state->failed_ceremony_count);           o += 4;
    absorb_u32(composed, o, state->key_share_count);                 o += 4;

    uint8_t local[32];
    keccak256(composed, uint64_t(o), local);
    for (uint32_t k = 0; k < 32u; ++k) state->mpcvm_state_root[k] = local[k];

    for (uint32_t k = 0; k < 32u; ++k) result->ceremony_root[k]      = state->ceremony_root[k];
    for (uint32_t k = 0; k < 32u; ++k) result->key_share_root[k]     = state->key_share_root[k];
    for (uint32_t k = 0; k < 32u; ++k) result->contribution_root[k]  = state->contribution_root[k];
    for (uint32_t k = 0; k < 32u; ++k) result->mpcvm_state_root[k]   = state->mpcvm_state_root[k];
    result->active_ceremony_count = *active_count_in;
    result->key_share_count       = *share_count_in;
    result->epoch                 = state->current_epoch;
    result->now_ns                = state->now_ns;
    result->status                = 1u;
}

}  // namespace mpcvm::cuda
