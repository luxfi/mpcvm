// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_transition.metal — v0.62 parallel leaf hashing + serial fold.
//
// Two kernel dispatches:
//   1. mpcvm_compute_leaves (gridSize = max(ceremony, share, contrib) slots)
//      — parallel keccak per leaf. Each tid hashes leaves[tid] into the
//      precomputed leaf_hashes buffer. Slot ordering = canonical ordering.
//   2. mpcvm_compose_root (1x1x1) — serial fold over precomputed leaf
//      hashes (root accumulator is non-associative, fold stays sequential)
//      plus epoch advance and mpcvm_state_root composition.
//
// Determinism: leaf hashes computed independently per slot; fold consumes
// them in canonical slot order. Byte-equal to the v0.61.1 single-thread
// implementation and to the CPU reference.

#include "mpcvm_kernels_common.h.metal"

// =============================================================================
// Pass 1: parallel leaf hash computation.
//
// We dispatch a single grid sized to max(ceremony_count, share_count,
// contribution_count). Each thread checks its tid against each table size
// and emits a leaf hash for the corresponding occupied slot, or zeros for
// free slots. The fold pass below reads `leaf_hashes` in slot order and
// skips zero hashes (slot was free).
//
// Output buffers:
//   ceremony_leaf_hashes    [ceremony_count   * 32]
//   share_leaf_hashes       [share_count      * 32]
//   contribution_leaf_hashes[contribution_count * 32]
//   ceremony_status_counts  [3]   atomic — active / finalized / failed
//   share_count_out         [1]   atomic — occupied share count
//   ceremony_used_mask      [ceremony_count]      — 1 if status != Free
//   share_used_mask         [share_count]         — 1 if occupied
//   contribution_used_mask  [contribution_count]  — 1 if status == 1
//
// (We only need the masks because the fold pass needs to know which slots
// to fold; storing them avoids re-reading the entire ceremony/share/
// contribution buffers from the fold pass.)
// =============================================================================

kernel void mpcvm_compute_leaves(
    device const Ceremony*    ceremonies              [[buffer(0)]],
    device const KeyShare*    shares                  [[buffer(1)]],
    device const Contribution* contributions          [[buffer(2)]],
    device uchar*             ceremony_leaf_hashes    [[buffer(3)]],
    device uchar*             share_leaf_hashes       [[buffer(4)]],
    device uchar*             contribution_leaf_hashes[[buffer(5)]],
    device atomic_uint*       active_count_out        [[buffer(6)]],
    device atomic_uint*       finalized_count_out     [[buffer(7)]],
    device atomic_uint*       failed_count_out        [[buffer(8)]],
    device atomic_uint*       share_count_out         [[buffer(9)]],
    device uchar*             ceremony_used_mask      [[buffer(10)]],
    device uchar*             share_used_mask         [[buffer(11)]],
    device uchar*             contribution_used_mask  [[buffer(12)]],
    constant uint&            ceremony_count          [[buffer(13)]],
    constant uint&            share_count             [[buffer(14)]],
    constant uint&            contribution_count      [[buffer(15)]],
    uint tid [[thread_position_in_grid]])
{
    // -- ceremony leaf --
    if (tid < ceremony_count) {
        device const Ceremony& c = ceremonies[tid];
        if (c.status == kCeremonyStatusFree) {
            ceremony_used_mask[tid] = 0u;
        } else {
            ceremony_used_mask[tid] = 1u;
            if (c.status == kCeremonyStatusInProgress)
                atomic_fetch_add_explicit(active_count_out, 1u, memory_order_relaxed);
            else if (c.status == kCeremonyStatusFinalized)
                atomic_fetch_add_explicit(finalized_count_out, 1u, memory_order_relaxed);
            else if (c.status == kCeremonyStatusFailed)
                atomic_fetch_add_explicit(failed_count_out, 1u, memory_order_relaxed);

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
            absorb_u32(leaf, o, tid);                       o += 4;

            uchar h[32];
            keccak256(leaf, (ulong)o, h);
            for (uint k = 0; k < 32u; ++k) ceremony_leaf_hashes[tid * 32u + k] = h[k];
        }
    }

    // -- key share leaf --
    if (tid < share_count) {
        device const KeyShare& s = shares[tid];
        if (s.occupied == 0u) {
            share_used_mask[tid] = 0u;
        } else {
            share_used_mask[tid] = 1u;
            atomic_fetch_add_explicit(share_count_out, 1u, memory_order_relaxed);

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
            absorb_u32(leaf, o, tid);            o += 4;

            uchar h[32];
            keccak256(leaf, (ulong)o, h);
            for (uint k = 0; k < 32u; ++k) share_leaf_hashes[tid * 32u + k] = h[k];
        }
    }

    // -- contribution leaf --
    if (tid < contribution_count) {
        device const Contribution& c = contributions[tid];
        if (c.status != 1u) {
            contribution_used_mask[tid] = 0u;
        } else {
            contribution_used_mask[tid] = 1u;

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
            absorb_u32(leaf, o, tid);               o += 4;

            uchar h[32];
            keccak256(leaf, (ulong)o, h);
            for (uint k = 0; k < 32u; ++k) contribution_leaf_hashes[tid * 32u + k] = h[k];
        }
    }
}

// =============================================================================
// Pass 2: serial fold over precomputed leaves + state root composition.
//
// Fold is keccak(acc || leaf_hash) — non-associative — so it must run in
// slot order on a single thread. The work per fold is one keccak on 64
// bytes; total work ≈ (ceremony_count + share_count + contribution_count)
// keccak-of-64 invocations (~8400 max), versus the v0.61.1 path which also
// computed each leaf serially.
// =============================================================================

kernel void mpcvm_compose_root(
    device const MPCVMRoundDescriptor* desc           [[buffer(0)]],
    device const uchar*                ceremony_leaf_hashes    [[buffer(1)]],
    device const uchar*                share_leaf_hashes       [[buffer(2)]],
    device const uchar*                contribution_leaf_hashes[[buffer(3)]],
    device const uchar*                ceremony_used_mask      [[buffer(4)]],
    device const uchar*                share_used_mask         [[buffer(5)]],
    device const uchar*                contribution_used_mask  [[buffer(6)]],
    device const uint*                 active_count_in         [[buffer(7)]],
    device const uint*                 finalized_count_in      [[buffer(8)]],
    device const uint*                 failed_count_in         [[buffer(9)]],
    device const uint*                 share_count_in          [[buffer(10)]],
    device MPCVMState*                 state          [[buffer(11)]],
    device MPCVMTransitionResult*      result         [[buffer(12)]],
    constant uint&                     ceremony_count    [[buffer(13)]],
    constant uint&                     share_count       [[buffer(14)]],
    constant uint&                     contribution_count[[buffer(15)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;

    uchar acc[32]; for (uint k = 0; k < 32u; ++k) acc[k] = 0;
    for (uint i = 0; i < ceremony_count; ++i) {
        if (ceremony_used_mask[i] == 0u) continue;
        uchar buf[64];
        for (uint k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint k = 0; k < 32u; ++k) buf[32 + k] = ceremony_leaf_hashes[i * 32u + k];
        keccak256(buf, 64u, acc);
    }
    for (uint k = 0; k < 32u; ++k) state->ceremony_root[k] = acc[k];
    state->active_ceremony_count    = *active_count_in;
    state->finalized_ceremony_count = *finalized_count_in;
    state->failed_ceremony_count    = *failed_count_in;

    for (uint k = 0; k < 32u; ++k) acc[k] = 0;
    for (uint i = 0; i < share_count; ++i) {
        if (share_used_mask[i] == 0u) continue;
        uchar buf[64];
        for (uint k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint k = 0; k < 32u; ++k) buf[32 + k] = share_leaf_hashes[i * 32u + k];
        keccak256(buf, 64u, acc);
    }
    for (uint k = 0; k < 32u; ++k) state->key_share_root[k] = acc[k];
    state->key_share_count = *share_count_in;

    for (uint k = 0; k < 32u; ++k) acc[k] = 0;
    for (uint i = 0; i < contribution_count; ++i) {
        if (contribution_used_mask[i] == 0u) continue;
        uchar buf[64];
        for (uint k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint k = 0; k < 32u; ++k) buf[32 + k] = contribution_leaf_hashes[i * 32u + k];
        keccak256(buf, 64u, acc);
    }
    for (uint k = 0; k < 32u; ++k) state->contribution_root[k] = acc[k];

    // -- epoch advance --
    state->now_ns = desc->timestamp_ns;
    if (desc->closing_flag != 0u) {
        state->current_epoch = desc->epoch + 1u;
    }

    // -- composed mpcvm_state_root --
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

    for (uint k = 0; k < 32u; ++k) result->ceremony_root[k]      = state->ceremony_root[k];
    for (uint k = 0; k < 32u; ++k) result->key_share_root[k]     = state->key_share_root[k];
    for (uint k = 0; k < 32u; ++k) result->contribution_root[k]  = state->contribution_root[k];
    for (uint k = 0; k < 32u; ++k) result->mpcvm_state_root[k]   = state->mpcvm_state_root[k];
    result->active_ceremony_count = *active_count_in;
    result->key_share_count       = *share_count_in;
    result->epoch                 = state->current_epoch;
    result->now_ns                = state->now_ns;
    result->status                = 1u;
}
