// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_transition.cu — CUDA peer of mpcvm_transition.metal.

#include "mpcvm_kernels_common.cuh"

namespace mpcvm::cuda {

extern "C" __global__ void mpcvm_transition(
    const MPCVMRoundDescriptor* desc,
    Ceremony*                   ceremonies,
    KeyShare*                   shares,
    Contribution*               contributions,
    MPCVMState*                 state,
    MPCVMTransitionResult*      result,
    uint32_t                    ceremony_count,
    uint32_t                    share_count,
    uint32_t                    contribution_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint8_t acc[32]; for (uint32_t k = 0; k < 32u; ++k) acc[k] = 0;
    uint32_t active = 0, finalized = 0, failed = 0;
    for (uint32_t i = 0; i < ceremony_count; ++i) {
        Ceremony& c = ceremonies[i];
        if (c.status == kCeremonyStatusFree) continue;
        if (c.status == kCeremonyStatusInProgress) ++active;
        if (c.status == kCeremonyStatusFinalized)  ++finalized;
        if (c.status == kCeremonyStatusFailed)     ++failed;

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
        absorb_u32(leaf, o, i);                         o += 4;

        uint8_t leaf_hash[32];
        keccak256(leaf, uint64_t(o), leaf_hash);
        uint8_t buf[64];
        for (uint32_t k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint32_t k = 0; k < 32u; ++k) buf[32 + k] = leaf_hash[k];
        keccak256(buf, 64u, acc);
    }
    for (uint32_t k = 0; k < 32u; ++k) state->ceremony_root[k] = acc[k];
    state->active_ceremony_count    = active;
    state->finalized_ceremony_count = finalized;
    state->failed_ceremony_count    = failed;

    for (uint32_t k = 0; k < 32u; ++k) acc[k] = 0;
    uint32_t shares_n = 0;
    for (uint32_t i = 0; i < share_count; ++i) {
        KeyShare& s = shares[i];
        if (s.occupied == 0u) continue;
        ++shares_n;

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
        absorb_u32(leaf, o, i);              o += 4;

        uint8_t leaf_hash[32];
        keccak256(leaf, uint64_t(o), leaf_hash);
        uint8_t buf[64];
        for (uint32_t k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint32_t k = 0; k < 32u; ++k) buf[32 + k] = leaf_hash[k];
        keccak256(buf, 64u, acc);
    }
    for (uint32_t k = 0; k < 32u; ++k) state->key_share_root[k] = acc[k];
    state->key_share_count = shares_n;

    for (uint32_t k = 0; k < 32u; ++k) acc[k] = 0;
    for (uint32_t i = 0; i < contribution_count; ++i) {
        Contribution& c = contributions[i];
        if (c.status != 1u) continue;
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
        absorb_u32(leaf, o, i);                 o += 4;

        uint8_t leaf_hash[32];
        keccak256(leaf, uint64_t(o), leaf_hash);
        uint8_t buf[64];
        for (uint32_t k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint32_t k = 0; k < 32u; ++k) buf[32 + k] = leaf_hash[k];
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
    result->active_ceremony_count = active;
    result->key_share_count       = shares_n;
    result->epoch                 = state->current_epoch;
    result->now_ns                = state->now_ns;
    result->status                = 1u;
}

}  // namespace mpcvm::cuda
