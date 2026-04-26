// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_kernels_common.wgsl — shared WGSL declarations for MPCVM.
//
// Layout structs MUST match mpcvm_gpu_layout.hpp byte-for-byte.
// WGSL has no native u64; vec2<u32> is used as the 64-bit carrier with
// (lo, hi) ordering matching little-endian wire format. Keccak-f[1600]
// runs on a 25-element array<vec2<u32>>; the reduce contract matches
// CPU / Metal / CUDA byte-for-byte.

struct Ceremony {
    ceremony_id_lo: u32,
    ceremony_id_hi: u32,
    started_at_ns_lo: u32,
    started_at_ns_hi: u32,
    deadline_ns_lo: u32,
    deadline_ns_hi: u32,
    participants_bitmap_lo: u32,
    participants_bitmap_hi: u32,
    kind: u32,
    round: u32,
    threshold: u32,
    total_participants: u32,
    status: u32,
    contribution_count: u32,
    subject: array<u32, 8>,        // 32 bytes
    ceremony_seed: array<u32, 8>,  // 32 bytes
};

struct KeyShare {
    share_id_lo: u32,
    share_id_hi: u32,
    ceremony_id_lo: u32,
    ceremony_id_hi: u32,
    holder_addr_lo: u32,
    holder_addr_hi: u32,
    scheme: u32,
    holder_index: u32,
    share_data_len: u32,
    occupied: u32,
    share_data: array<u32, 80>,  // 320 bytes
    pad0_lo: u32,
    pad0_hi: u32,
};

struct Contribution {
    contribution_id_lo: u32,
    contribution_id_hi: u32,
    ceremony_id_lo: u32,
    ceremony_id_hi: u32,
    holder_addr_lo: u32,
    holder_addr_hi: u32,
    round: u32,
    holder_index: u32,
    payload_len: u32,
    status: u32,
    payload: array<u32, 96>,  // 384 bytes
    pad0_lo: u32,
    pad0_hi: u32,
};

struct MPCVMState {
    current_epoch_lo: u32,
    current_epoch_hi: u32,
    now_ns_lo: u32,
    now_ns_hi: u32,
    active_ceremony_count: u32,
    finalized_ceremony_count: u32,
    failed_ceremony_count: u32,
    key_share_count: u32,
    ceremony_root: array<u32, 8>,
    key_share_root: array<u32, 8>,
    contribution_root: array<u32, 8>,
    mpcvm_state_root: array<u32, 8>,
};

struct MPCVMRoundDescriptor {
    chain_id_lo: u32,
    chain_id_hi: u32,
    round_lo: u32,
    round_hi: u32,
    timestamp_ns_lo: u32,
    timestamp_ns_hi: u32,
    epoch_lo: u32,
    epoch_hi: u32,
    mode: u32,
    ceremony_op_count: u32,
    contribution_op_count: u32,
    closing_flag: u32,
    pad0: u32,
    pad1: u32,
    pad2_lo: u32,
    pad2_hi: u32,
    parent_state_root: array<u32, 8>,
};

struct CeremonyOp {
    ceremony_id_lo: u32,
    ceremony_id_hi: u32,
    deadline_ns_lo: u32,
    deadline_ns_hi: u32,
    kind: u32,
    ceremony_kind: u32,
    threshold: u32,
    total_participants: u32,
    subject: array<u32, 8>,
    ceremony_seed: array<u32, 8>,
};

struct ContributionOp {
    ceremony_id_lo: u32,
    ceremony_id_hi: u32,
    holder_addr_lo: u32,
    holder_addr_hi: u32,
    round: u32,
    holder_index: u32,
    payload_len: u32,
    pad0: u32,
    payload: array<u32, 96>,
};

struct MPCVMTransitionResult {
    status: u32,
    ceremony_apply_count: u32,
    contribution_apply_count: u32,
    finalized_this_round: u32,
    failed_this_round: u32,
    active_ceremony_count: u32,
    key_share_count: u32,
    round_advance_count: u32,
    epoch_lo: u32,
    epoch_hi: u32,
    now_ns_lo: u32,
    now_ns_hi: u32,
    ceremony_root: array<u32, 8>,
    key_share_root: array<u32, 8>,
    contribution_root: array<u32, 8>,
    mpcvm_state_root: array<u32, 8>,
};

// Status / kind constants
const kCeremonyStatusFree: u32       = 0u;
const kCeremonyStatusInProgress: u32 = 1u;
const kCeremonyStatusFinalized: u32  = 2u;
const kCeremonyStatusFailed: u32     = 3u;

const kCeremonyOpBegin: u32  = 0u;
const kCeremonyOpCancel: u32 = 1u;

const kKindFrostKeygen: u32   = 0u;
const kKindFrostSign: u32     = 1u;
const kKindCggmp21Keygen: u32 = 2u;
const kKindCggmp21Sign: u32   = 3u;
const kKindRingtailDkg: u32   = 4u;
const kKindRingtailSign: u32  = 5u;

// Comparison helpers for emulated u64 (vec2<u32> as (lo, hi)).
fn u64_eq(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    return a_lo == b_lo && a_hi == b_hi;
}
