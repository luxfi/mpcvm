// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_kernels_common.h.metal — shared device code for MPCVM kernels.
//
// Layout structs MUST match mpcvm_gpu_layout.hpp byte-for-byte. keccak256
// is the same Keccak-f[1600] / 0x01 / 0x80 padding used by the CPU
// reference (mpcvm_cpu_reference.cpp) and pvm/quasar. Determinism across
// CPU / Metal / CUDA / WGSL hinges on this exact byte-for-byte recipe.

#pragma once

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Layout structs — must match mpcvm_gpu_layout.hpp byte-for-byte.
// =============================================================================

struct alignas(16) Ceremony {
    ulong  ceremony_id;            // 0
    ulong  started_at_ns;          // 8
    ulong  deadline_ns;            // 16
    ulong  participants_bitmap;    // 24
    uint   kind;                   // 32
    uint   round;                  // 36
    uint   threshold;              // 40
    uint   total_participants;     // 44
    uint   status;                 // 48
    uint   contribution_count;     // 52
    uchar  subject[32];            // 56
    uchar  ceremony_seed[32];      // 88
    // 120 -> aligns to 128
};

struct alignas(16) KeyShare {
    ulong  share_id;               // 0
    ulong  ceremony_id;            // 8
    ulong  holder_addr;            // 16
    uint   scheme;                 // 24
    uint   holder_index;           // 28
    uint   share_data_len;         // 32
    uint   occupied;               // 36
    uchar  share_data[320];        // 40
    ulong  _pad0;                  // 360 -> 368
};

struct alignas(16) Contribution {
    ulong  contribution_id;        // 0
    ulong  ceremony_id;            // 8
    ulong  holder_addr;            // 16
    uint   round;                  // 24
    uint   holder_index;           // 28
    uint   payload_len;            // 32
    uint   status;                 // 36
    uchar  payload[384];           // 40
    ulong  _pad0;                  // 424 -> 432
};

struct alignas(16) MPCVMState {
    ulong  current_epoch;
    ulong  now_ns;
    uint   active_ceremony_count;
    uint   finalized_ceremony_count;
    uint   failed_ceremony_count;
    uint   key_share_count;
    uchar  ceremony_root[32];
    uchar  key_share_root[32];
    uchar  contribution_root[32];
    uchar  mpcvm_state_root[32];
};

struct alignas(16) MPCVMRoundDescriptor {
    ulong  chain_id;
    ulong  round;
    ulong  timestamp_ns;
    ulong  epoch;
    uint   mode;
    uint   ceremony_op_count;
    uint   contribution_op_count;
    uint   closing_flag;
    uint   _pad0;
    uint   _pad1;
    ulong  _pad2;
    uchar  parent_state_root[32];
};

struct alignas(16) CeremonyOp {
    ulong  ceremony_id;
    ulong  deadline_ns;
    uint   kind;
    uint   ceremony_kind;
    uint   threshold;
    uint   total_participants;
    uchar  subject[32];
    uchar  ceremony_seed[32];
};

struct alignas(16) ContributionOp {
    ulong  ceremony_id;
    ulong  holder_addr;
    uint   round;
    uint   holder_index;
    uint   payload_len;
    uint   _pad0;
    uchar  payload[384];
};

struct alignas(16) MPCVMTransitionResult {
    uint   status;
    uint   ceremony_apply_count;
    uint   contribution_apply_count;
    uint   finalized_this_round;
    uint   failed_this_round;
    uint   active_ceremony_count;
    uint   key_share_count;
    uint   round_advance_count;
    ulong  epoch;
    ulong  now_ns;
    uchar  ceremony_root[32];
    uchar  key_share_root[32];
    uchar  contribution_root[32];
    uchar  mpcvm_state_root[32];
};

// =============================================================================
// Status / kind constants
// =============================================================================

constant uint kCeremonyStatusFree       = 0u;
constant uint kCeremonyStatusInProgress = 1u;
constant uint kCeremonyStatusFinalized  = 2u;
constant uint kCeremonyStatusFailed     = 3u;

constant uint kCeremonyOpBegin  = 0u;
constant uint kCeremonyOpCancel = 1u;

constant uint kKindFrostKeygen   = 0u;
constant uint kKindFrostSign     = 1u;
constant uint kKindCggmp21Keygen = 2u;
constant uint kKindCggmp21Sign   = 3u;
constant uint kKindRingtailDkg   = 4u;
constant uint kKindRingtailSign  = 5u;

constant uint kFrostKeygenRounds   = 3u;
constant uint kFrostSignRounds     = 2u;
constant uint kCggmp21KeygenRounds = 3u;
constant uint kCggmp21SignRounds   = 5u;
constant uint kRingtailDkgRounds   = 2u;
constant uint kRingtailSignRounds  = 2u;

constant uint kSchemeFrost    = 0u;
constant uint kSchemeCggmp21  = 1u;
constant uint kSchemeRingtail = 2u;

constant uint kModeFrostKeygen   = 0u;
constant uint kModeFrostSign     = 1u;
constant uint kModeCggmp21Keygen = 2u;
constant uint kModeCggmp21Sign   = 3u;
constant uint kModeRingtailDkg   = 4u;
constant uint kModeRingtailSign  = 5u;
constant uint kModeCeremonyStep  = 6u;
constant uint kModeFullRound     = 7u;

// =============================================================================
// keccak256 — bit-identical to mpcvm_cpu_reference.cpp
// =============================================================================

constant ulong kKeccakRC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL,
};

constant uint kKeccakRot[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

inline ulong rotl64(ulong x, uint n) {
    return (x << n) | (x >> (64u - n));
}

inline void keccak_f1600(thread ulong* s) {
    for (uint round = 0; round < 24u; ++round) {
        ulong c[5];
        for (uint x = 0; x < 5u; ++x)
            c[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        ulong d[5];
        for (uint x = 0; x < 5u; ++x)
            d[x] = c[(x + 4u) % 5u] ^ rotl64(c[(x + 1u) % 5u], 1u);
        for (uint y = 0; y < 25u; y += 5u)
            for (uint x = 0; x < 5u; ++x)
                s[y + x] ^= d[x];
        ulong b[25];
        for (uint y = 0; y < 5u; ++y)
            for (uint x = 0; x < 5u; ++x) {
                uint i = x + 5u * y;
                uint j = y + 5u * ((2u * x + 3u * y) % 5u);
                b[j] = rotl64(s[i], kKeccakRot[i]);
            }
        for (uint y = 0; y < 25u; y += 5u) {
            ulong t0 = b[y+0], t1 = b[y+1], t2 = b[y+2], t3 = b[y+3], t4 = b[y+4];
            s[y+0] = t0 ^ ((~t1) & t2);
            s[y+1] = t1 ^ ((~t2) & t3);
            s[y+2] = t2 ^ ((~t3) & t4);
            s[y+3] = t3 ^ ((~t4) & t0);
            s[y+4] = t4 ^ ((~t0) & t1);
        }
        s[0] ^= kKeccakRC[round];
    }
}

inline void keccak256(thread const uchar* data, ulong len, thread uchar* out) {
    ulong s[25] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    const uint rate = 136u;
    ulong off = 0;
    while (len - off >= rate) {
        for (uint i = 0; i < rate; ++i) {
            uint lane = i / 8u, sh = (i % 8u) * 8u;
            s[lane] ^= ((ulong)data[off + i]) << sh;
        }
        keccak_f1600(s);
        off += rate;
    }
    uchar block[136] = {};
    ulong rem = len - off;
    for (ulong i = 0; i < rem; ++i) block[i] = data[off + i];
    block[rem]      ^= 0x01;
    block[rate - 1] ^= 0x80;
    for (uint i = 0; i < rate; ++i) {
        uint lane = i / 8u, sh = (i % 8u) * 8u;
        s[lane] ^= ((ulong)block[i]) << sh;
    }
    keccak_f1600(s);
    for (uint i = 0; i < 32u; ++i) {
        uint lane = i / 8u, sh = (i % 8u) * 8u;
        out[i] = (uchar)((s[lane] >> sh) & 0xFFu);
    }
}

inline void absorb_u32(thread uchar* dst, uint off, uint v) {
    for (uint k = 0; k < 4u; ++k) dst[off + k] = (uchar)((v >> (k*8u)) & 0xFFu);
}

inline void absorb_u64(thread uchar* dst, uint off, ulong v) {
    for (uint k = 0; k < 8u; ++k) dst[off + k] = (uchar)((v >> (k*8u)) & 0xFFu);
}

// =============================================================================
// Open-addressing locators (same hash as CPU ref)
// =============================================================================

inline uint ceremony_index_hash(ulong cid, uint mask) {
    ulong h = 0xcbf29ce484222325UL;
    h = (h ^ cid) * 0x100000001b3UL;
    return (uint)h & mask;
}

inline uint ceremony_locate(device Ceremony* tab, uint count,
                            ulong cid, bool insert_if_missing)
{
    uint mask = count - 1u;
    uint idx  = ceremony_index_hash(cid, mask);
    for (uint probe = 0; probe < count; ++probe) {
        device Ceremony& s = tab[idx];
        if (s.status == kCeremonyStatusFree) {
            if (insert_if_missing) {
                s.ceremony_id = cid;
                s.started_at_ns = 0;
                s.deadline_ns = 0;
                s.participants_bitmap = 0;
                s.kind = 0;
                s.round = 0;
                s.threshold = 0;
                s.total_participants = 0;
                s.contribution_count = 0;
                s.status = kCeremonyStatusInProgress;
                for (uint k = 0; k < 32u; ++k) s.subject[k] = 0;
                for (uint k = 0; k < 32u; ++k) s.ceremony_seed[k] = 0;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (s.ceremony_id == cid) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

inline uint contribution_locate(device Contribution* tab, uint count,
                                ulong cid, uint round, uint holder,
                                bool insert_if_missing)
{
    uint mask = count - 1u;
    ulong composite = cid ^ (((ulong)round << 32) | (ulong)holder);
    composite ^= 0x9E3779B97F4A7C15UL + (cid << 6) + (cid >> 2);
    uint idx = (uint)composite & mask;
    for (uint probe = 0; probe < count; ++probe) {
        device Contribution& s = tab[idx];
        if (s.status == 0u) {
            if (insert_if_missing) {
                s.contribution_id = 0;
                s.ceremony_id = cid;
                s.holder_addr = 0;
                s.round = round;
                s.holder_index = holder;
                s.payload_len = 0;
                s.status = 1u;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (s.ceremony_id == cid && s.round == round && s.holder_index == holder)
            return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

inline uint key_share_locate_free(device KeyShare* tab, uint count,
                                  ulong cid, uint holder)
{
    uint mask = count - 1u;
    ulong composite = cid ^ (((ulong)holder + 0x9E3779B97F4A7C15UL)
                              + (cid << 6) + (cid >> 2));
    uint idx = (uint)composite & mask;
    for (uint probe = 0; probe < count; ++probe) {
        device KeyShare& s = tab[idx];
        if (s.occupied == 0u) {
            return idx;
        }
        if (s.ceremony_id == cid && s.holder_index == holder) {
            return idx;
        }
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

inline uint total_rounds_for(uint kind) {
    if (kind == kKindFrostKeygen)    return kFrostKeygenRounds;
    if (kind == kKindFrostSign)      return kFrostSignRounds;
    if (kind == kKindCggmp21Keygen)  return kCggmp21KeygenRounds;
    if (kind == kKindCggmp21Sign)    return kCggmp21SignRounds;
    if (kind == kKindRingtailDkg)    return kRingtailDkgRounds;
    if (kind == kKindRingtailSign)   return kRingtailSignRounds;
    return 1u;
}

inline bool is_keygen_kind(uint kind) {
    return kind == kKindFrostKeygen
        || kind == kKindCggmp21Keygen
        || kind == kKindRingtailDkg;
}

inline uint scheme_for_kind(uint kind) {
    if (kind == kKindFrostKeygen || kind == kKindFrostSign)         return kSchemeFrost;
    if (kind == kKindCggmp21Keygen || kind == kKindCggmp21Sign)     return kSchemeCggmp21;
    return kSchemeRingtail;
}

inline uint share_data_len_for_scheme(uint scheme) {
    if (scheme == kSchemeFrost)    return 65u;
    if (scheme == kSchemeCggmp21)  return 65u;
    return 256u;
}
