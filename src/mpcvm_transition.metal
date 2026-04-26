// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_transition.metal — root computation kernel.
//
// Composes:
//   ceremony_root      = fold-keccak over occupied Ceremony leaves
//   key_share_root     = fold-keccak over occupied KeyShare leaves
//   contribution_root  = fold-keccak over accepted Contribution leaves
//   mpcvm_state_root   = keccak(parent || cer || share || contr ||
//                              epoch || now || active || finalized ||
//                              failed || share_count)
//
// The state-root is what the Quasar round descriptor binds as
// mpcvm_state_root for cross-chain attestation.

#include "mpcvm_kernels_common.h.metal"

kernel void mpcvm_transition(
    device const MPCVMRoundDescriptor* desc           [[buffer(0)]],
    device Ceremony*                   ceremonies     [[buffer(1)]],
    device KeyShare*                   shares         [[buffer(2)]],
    device Contribution*               contributions  [[buffer(3)]],
    device MPCVMState*                 state          [[buffer(4)]],
    device MPCVMTransitionResult*      result         [[buffer(5)]],
    constant uint&                     ceremony_count    [[buffer(6)]],
    constant uint&                     share_count       [[buffer(7)]],
    constant uint&                     contribution_count[[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;

    // -- ceremony root + counts --
    uchar acc[32]; for (uint k = 0; k < 32u; ++k) acc[k] = 0;
    uint active = 0, finalized = 0, failed = 0;
    for (uint i = 0; i < ceremony_count; ++i) {
        device Ceremony& c = ceremonies[i];
        if (c.status == kCeremonyStatusFree) continue;
        if (c.status == kCeremonyStatusInProgress) ++active;
        if (c.status == kCeremonyStatusFinalized)  ++finalized;
        if (c.status == kCeremonyStatusFailed)     ++failed;

        uchar leaf[8 + 8 + 8 + 8 + 4 + 4 + 4 + 4 + 4 + 4 + 32 + 32 + 4];
        uint o = 0;
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
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = c.subject[k];        o += 32;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = c.ceremony_seed[k];  o += 32;
        absorb_u32(leaf, o, i);                         o += 4;

        uchar leaf_hash[32];
        keccak256(leaf, (ulong)o, leaf_hash);
        uchar buf[64];
        for (uint k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint k = 0; k < 32u; ++k) buf[32 + k] = leaf_hash[k];
        keccak256(buf, 64u, acc);
    }
    for (uint k = 0; k < 32u; ++k) state->ceremony_root[k] = acc[k];
    state->active_ceremony_count    = active;
    state->finalized_ceremony_count = finalized;
    state->failed_ceremony_count    = failed;

    // -- key_share root + count --
    for (uint k = 0; k < 32u; ++k) acc[k] = 0;
    uint shares_n = 0;
    for (uint i = 0; i < share_count; ++i) {
        device KeyShare& s = shares[i];
        if (s.occupied == 0u) continue;
        ++shares_n;

        // leaf has variable share_data_len; build fixed buffer up to 320 + prefix.
        uchar leaf[8 + 8 + 8 + 4 + 4 + 4 + 320 + 4];
        uint o = 0;
        absorb_u64(leaf, o, s.share_id);     o += 8;
        absorb_u64(leaf, o, s.ceremony_id);  o += 8;
        absorb_u64(leaf, o, s.holder_addr);  o += 8;
        absorb_u32(leaf, o, s.scheme);       o += 4;
        absorb_u32(leaf, o, s.holder_index); o += 4;
        absorb_u32(leaf, o, s.share_data_len); o += 4;
        for (uint k = 0; k < s.share_data_len && k < 320u; ++k)
            leaf[o + k] = s.share_data[k];
        o += s.share_data_len;
        absorb_u32(leaf, o, i);              o += 4;

        uchar leaf_hash[32];
        keccak256(leaf, (ulong)o, leaf_hash);
        uchar buf[64];
        for (uint k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint k = 0; k < 32u; ++k) buf[32 + k] = leaf_hash[k];
        keccak256(buf, 64u, acc);
    }
    for (uint k = 0; k < 32u; ++k) state->key_share_root[k] = acc[k];
    state->key_share_count = shares_n;

    // -- contribution root --
    for (uint k = 0; k < 32u; ++k) acc[k] = 0;
    for (uint i = 0; i < contribution_count; ++i) {
        device Contribution& c = contributions[i];
        if (c.status != 1u) continue;
        uchar leaf[8 + 8 + 8 + 4 + 4 + 4 + 384 + 4];
        uint o = 0;
        absorb_u64(leaf, o, c.contribution_id); o += 8;
        absorb_u64(leaf, o, c.ceremony_id);     o += 8;
        absorb_u64(leaf, o, c.holder_addr);     o += 8;
        absorb_u32(leaf, o, c.round);           o += 4;
        absorb_u32(leaf, o, c.holder_index);    o += 4;
        absorb_u32(leaf, o, c.payload_len);     o += 4;
        for (uint k = 0; k < c.payload_len && k < 384u; ++k)
            leaf[o + k] = c.payload[k];
        o += c.payload_len;
        absorb_u32(leaf, o, i);                 o += 4;

        uchar leaf_hash[32];
        keccak256(leaf, (ulong)o, leaf_hash);
        uchar buf[64];
        for (uint k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint k = 0; k < 32u; ++k) buf[32 + k] = leaf_hash[k];
        keccak256(buf, 64u, acc);
    }
    for (uint k = 0; k < 32u; ++k) state->contribution_root[k] = acc[k];

    // -- epoch advance --
    state->now_ns = desc->timestamp_ns;
    if (desc->closing_flag != 0u) {
        state->current_epoch = desc->epoch + 1u;
    }

    // -- composed state root --
    uchar composed[32 + 32 + 32 + 32 + 8 + 8 + 4 + 4 + 4 + 4];
    uint o = 0;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = desc->parent_state_root[k];   o += 32;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = state->ceremony_root[k];      o += 32;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = state->key_share_root[k];     o += 32;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = state->contribution_root[k];  o += 32;
    absorb_u64(composed, o, state->current_epoch);                  o += 8;
    absorb_u64(composed, o, state->now_ns);                          o += 8;
    absorb_u32(composed, o, state->active_ceremony_count);           o += 4;
    absorb_u32(composed, o, state->finalized_ceremony_count);        o += 4;
    absorb_u32(composed, o, state->failed_ceremony_count);           o += 4;
    absorb_u32(composed, o, state->key_share_count);                 o += 4;

    uchar local[32];
    keccak256(composed, (ulong)o, local);
    for (uint k = 0; k < 32u; ++k) state->mpcvm_state_root[k] = local[k];

    // -- write result --
    for (uint k = 0; k < 32u; ++k) result->ceremony_root[k]      = state->ceremony_root[k];
    for (uint k = 0; k < 32u; ++k) result->key_share_root[k]     = state->key_share_root[k];
    for (uint k = 0; k < 32u; ++k) result->contribution_root[k]  = state->contribution_root[k];
    for (uint k = 0; k < 32u; ++k) result->mpcvm_state_root[k]   = state->mpcvm_state_root[k];
    result->active_ceremony_count = active;
    result->key_share_count       = shares_n;
    result->epoch                 = state->current_epoch;
    result->now_ns                = state->now_ns;
    result->status                = 1u;
}
