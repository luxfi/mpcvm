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

// Layout MUST be 128 bytes to match C++ `alignas(16) Ceremony` host struct.
// WGSL packs to 120 bytes naturally (no 64-bit native types); we add an
// explicit 8-byte trailing pad so the storage-buffer stride matches host.
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
    _pad_lo: u32,                  // align to 128
    _pad_hi: u32,
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

fn u64_gt(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    if (a_hi != b_hi) { return a_hi > b_hi; }
    return a_lo > b_lo;
}

// =============================================================================
// keccak-f[1600] over 25 vec2<u32> lanes (each = u64 as (lo, hi)).
// Byte-for-byte equivalent to CPU keccak in mpcvm_cpu_reference.cpp.
// rate = 136 bytes (0x01 ... 0x80 padding), output = 32 bytes (keccak256).
// =============================================================================

// 24 round constants split into (lo, hi) pairs.
const kKeccakRC_lo: array<u32, 24> = array<u32, 24>(
    0x00000001u, 0x00008082u, 0x0000808Au, 0x80008000u,
    0x0000808Bu, 0x80000001u, 0x80008081u, 0x00008009u,
    0x0000008Au, 0x00000088u, 0x80008009u, 0x8000000Au,
    0x8000808Bu, 0x0000008Bu, 0x00008089u, 0x00008003u,
    0x00008002u, 0x00000080u, 0x0000800Au, 0x8000000Au,
    0x80008081u, 0x00008080u, 0x80000001u, 0x80008008u,
);
const kKeccakRC_hi: array<u32, 24> = array<u32, 24>(
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x80000000u, 0x80000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
);

const kKeccakRot: array<u32, 25> = array<u32, 25>(
     0u,  1u, 62u, 28u, 27u,
    36u, 44u,  6u, 55u, 20u,
     3u, 10u, 43u, 25u, 39u,
    41u, 45u, 15u, 21u,  8u,
    18u,  2u, 61u, 56u, 14u,
);

// 64-bit XOR over (lo, hi).
fn u64_xor(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    return vec2<u32>(a_lo ^ b_lo, a_hi ^ b_hi);
}

// AND-NOT: (~a) & b
fn u64_andnot(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    return vec2<u32>((~a_lo) & b_lo, (~a_hi) & b_hi);
}

// 64-bit left-rotation by n bits, n in [0..63]. Implemented as two 32-bit
// rotations cross-laned. Matches CPU rotl64 byte-for-byte.
fn rotl64_u32x2(lo: u32, hi: u32, n_in: u32) -> vec2<u32> {
    let n: u32 = n_in & 63u;
    if (n == 0u) {
        return vec2<u32>(lo, hi);
    }
    if (n < 32u) {
        let new_lo: u32 = (lo << n) | (hi >> (32u - n));
        let new_hi: u32 = (hi << n) | (lo >> (32u - n));
        return vec2<u32>(new_lo, new_hi);
    }
    let m: u32 = n - 32u;
    if (m == 0u) {
        // n == 32: simple swap
        return vec2<u32>(hi, lo);
    }
    let new_lo: u32 = (hi << m) | (lo >> (32u - m));
    let new_hi: u32 = (lo << m) | (hi >> (32u - m));
    return vec2<u32>(new_lo, new_hi);
}

// In-place keccak-f[1600] permutation. State is 25 lanes of (lo, hi).
fn keccak_f1600(state: ptr<function, array<vec2<u32>, 25>>) {
    var c: array<vec2<u32>, 5>;
    var d: array<vec2<u32>, 5>;
    var b: array<vec2<u32>, 25>;
    for (var round: u32 = 0u; round < 24u; round = round + 1u) {
        // theta
        for (var x: u32 = 0u; x < 5u; x = x + 1u) {
            let s0 = (*state)[x];
            let s1 = (*state)[x + 5u];
            let s2 = (*state)[x + 10u];
            let s3 = (*state)[x + 15u];
            let s4 = (*state)[x + 20u];
            c[x] = vec2<u32>(s0.x ^ s1.x ^ s2.x ^ s3.x ^ s4.x,
                             s0.y ^ s1.y ^ s2.y ^ s3.y ^ s4.y);
        }
        for (var x: u32 = 0u; x < 5u; x = x + 1u) {
            let cm = c[(x + 4u) % 5u];
            let cn = c[(x + 1u) % 5u];
            let r = rotl64_u32x2(cn.x, cn.y, 1u);
            d[x] = vec2<u32>(cm.x ^ r.x, cm.y ^ r.y);
        }
        for (var y: u32 = 0u; y < 25u; y = y + 5u) {
            for (var x: u32 = 0u; x < 5u; x = x + 1u) {
                let s = (*state)[y + x];
                (*state)[y + x] = vec2<u32>(s.x ^ d[x].x, s.y ^ d[x].y);
            }
        }
        // rho + pi
        for (var y: u32 = 0u; y < 5u; y = y + 1u) {
            for (var x: u32 = 0u; x < 5u; x = x + 1u) {
                let i: u32 = x + 5u * y;
                let j: u32 = y + 5u * ((2u * x + 3u * y) % 5u);
                let s = (*state)[i];
                b[j] = rotl64_u32x2(s.x, s.y, kKeccakRot[i]);
            }
        }
        // chi
        for (var y: u32 = 0u; y < 25u; y = y + 5u) {
            let t0 = b[y + 0u];
            let t1 = b[y + 1u];
            let t2 = b[y + 2u];
            let t3 = b[y + 3u];
            let t4 = b[y + 4u];
            let an1 = u64_andnot(t1.x, t1.y, t2.x, t2.y);
            let an2 = u64_andnot(t2.x, t2.y, t3.x, t3.y);
            let an3 = u64_andnot(t3.x, t3.y, t4.x, t4.y);
            let an4 = u64_andnot(t4.x, t4.y, t0.x, t0.y);
            let an5 = u64_andnot(t0.x, t0.y, t1.x, t1.y);
            (*state)[y + 0u] = vec2<u32>(t0.x ^ an1.x, t0.y ^ an1.y);
            (*state)[y + 1u] = vec2<u32>(t1.x ^ an2.x, t1.y ^ an2.y);
            (*state)[y + 2u] = vec2<u32>(t2.x ^ an3.x, t2.y ^ an3.y);
            (*state)[y + 3u] = vec2<u32>(t3.x ^ an4.x, t3.y ^ an4.y);
            (*state)[y + 4u] = vec2<u32>(t4.x ^ an5.x, t4.y ^ an5.y);
        }
        // iota
        let s0 = (*state)[0];
        (*state)[0] = vec2<u32>(s0.x ^ kKeccakRC_lo[round],
                                s0.y ^ kKeccakRC_hi[round]);
    }
}

// XOR a single byte into the keccak state lane at byte index i.
fn keccak_absorb_byte(state: ptr<function, array<vec2<u32>, 25>>,
                      i: u32, byte_val: u32) {
    let lane: u32  = i / 8u;
    let in_lane: u32 = i % 8u;
    let cur = (*state)[lane];
    if (in_lane < 4u) {
        let sh: u32 = in_lane * 8u;
        (*state)[lane] = vec2<u32>(cur.x ^ ((byte_val & 0xFFu) << sh), cur.y);
    } else {
        let sh: u32 = (in_lane - 4u) * 8u;
        (*state)[lane] = vec2<u32>(cur.x, cur.y ^ ((byte_val & 0xFFu) << sh));
    }
}

// Read byte i from the keccak state.
fn keccak_read_byte(state: ptr<function, array<vec2<u32>, 25>>, i: u32) -> u32 {
    let lane: u32  = i / 8u;
    let in_lane: u32 = i % 8u;
    let cur = (*state)[lane];
    if (in_lane < 4u) {
        let sh: u32 = in_lane * 8u;
        return (cur.x >> sh) & 0xFFu;
    }
    let sh: u32 = (in_lane - 4u) * 8u;
    return (cur.y >> sh) & 0xFFu;
}

// Buffer-backed keccak256 — input lives in a function-local byte array.
// CAP is the static upper bound on input bytes; len <= CAP must hold.
// MPCVM uses CAP up to 2048 (share emission) and a few hundred (leaves).
//
// keccak-256: rate = 136 bytes, capacity = 64 bytes, output = 32 bytes.
// pad: append 0x01 at offset (len % rate), set 0x80 at byte (rate-1) of
// the last block. Identical recipe to CPU reference.
const kKeccakRate: u32 = 136u;

fn keccak256_buf2048(data: ptr<function, array<u32, 2048>>, len: u32,
                     out: ptr<function, array<u32, 32>>) {
    var s: array<vec2<u32>, 25>;
    for (var k: u32 = 0u; k < 25u; k = k + 1u) {
        s[k] = vec2<u32>(0u, 0u);
    }
    var off: u32 = 0u;
    while (len - off >= kKeccakRate) {
        for (var i: u32 = 0u; i < kKeccakRate; i = i + 1u) {
            keccak_absorb_byte(&s, i, (*data)[off + i]);
        }
        keccak_f1600(&s);
        off = off + kKeccakRate;
    }
    // Final block — copy remaining bytes into a stack-resident block, pad,
    // then absorb. We absorb by rebuilding a fresh state-XOR; since the
    // outer loop already consumed full blocks, it's safe to merge the
    // padded tail directly into s.
    let rem: u32 = len - off;
    for (var i: u32 = 0u; i < rem; i = i + 1u) {
        keccak_absorb_byte(&s, i, (*data)[off + i]);
    }
    // 0x01 at byte rem of the *current* block (i.e. position rem within rate).
    keccak_absorb_byte(&s, rem, 0x01u);
    // 0x80 at byte (rate - 1).
    keccak_absorb_byte(&s, kKeccakRate - 1u, 0x80u);
    keccak_f1600(&s);
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        (*out)[i] = keccak_read_byte(&s, i);
    }
}

// 64-byte-input variant for the fold step (acc || leaf_hash).
fn keccak256_buf64(data: ptr<function, array<u32, 64>>,
                   out: ptr<function, array<u32, 32>>) {
    var s: array<vec2<u32>, 25>;
    for (var k: u32 = 0u; k < 25u; k = k + 1u) {
        s[k] = vec2<u32>(0u, 0u);
    }
    // 64 < rate (136), so just one absorb + pad.
    for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        keccak_absorb_byte(&s, i, (*data)[i]);
    }
    keccak_absorb_byte(&s, 64u, 0x01u);
    keccak_absorb_byte(&s, kKeccakRate - 1u, 0x80u);
    keccak_f1600(&s);
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        (*out)[i] = keccak_read_byte(&s, i);
    }
}

// 33-byte stretch step for share derivation.
fn keccak256_buf33(data: ptr<function, array<u32, 33>>,
                   out: ptr<function, array<u32, 32>>) {
    var s: array<vec2<u32>, 25>;
    for (var k: u32 = 0u; k < 25u; k = k + 1u) {
        s[k] = vec2<u32>(0u, 0u);
    }
    for (var i: u32 = 0u; i < 33u; i = i + 1u) {
        keccak_absorb_byte(&s, i, (*data)[i]);
    }
    keccak_absorb_byte(&s, 33u, 0x01u);
    keccak_absorb_byte(&s, kKeccakRate - 1u, 0x80u);
    keccak_f1600(&s);
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        (*out)[i] = keccak_read_byte(&s, i);
    }
}

// Helper: write u32 little-endian into a byte buffer (4 bytes).
fn write_u32_le(buf: ptr<function, array<u32, 2048>>, off: u32, v: u32) {
    (*buf)[off + 0u] = v & 0xFFu;
    (*buf)[off + 1u] = (v >> 8u) & 0xFFu;
    (*buf)[off + 2u] = (v >> 16u) & 0xFFu;
    (*buf)[off + 3u] = (v >> 24u) & 0xFFu;
}

// Helper: write u64 (lo, hi) little-endian into a byte buffer (8 bytes).
fn write_u64_le(buf: ptr<function, array<u32, 2048>>, off: u32, lo: u32, hi: u32) {
    (*buf)[off + 0u] = lo & 0xFFu;
    (*buf)[off + 1u] = (lo >> 8u) & 0xFFu;
    (*buf)[off + 2u] = (lo >> 16u) & 0xFFu;
    (*buf)[off + 3u] = (lo >> 24u) & 0xFFu;
    (*buf)[off + 4u] = hi & 0xFFu;
    (*buf)[off + 5u] = (hi >> 8u) & 0xFFu;
    (*buf)[off + 6u] = (hi >> 16u) & 0xFFu;
    (*buf)[off + 7u] = (hi >> 24u) & 0xFFu;
}
