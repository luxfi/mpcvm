// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_kernels_common.cuh — shared device code for the CUDA MPCVM kernels.
// Layout MUST match mpcvm_gpu_layout.hpp byte-for-byte.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace mpcvm::cuda {

struct alignas(16) Ceremony {
    uint64_t ceremony_id;
    uint64_t started_at_ns;
    uint64_t deadline_ns;
    uint64_t participants_bitmap;
    uint32_t kind;
    uint32_t round;
    uint32_t threshold;
    uint32_t total_participants;
    uint32_t status;
    uint32_t contribution_count;
    uint8_t  subject[32];
    uint8_t  ceremony_seed[32];
};

struct alignas(16) KeyShare {
    uint64_t share_id;
    uint64_t ceremony_id;
    uint64_t holder_addr;
    uint32_t scheme;
    uint32_t holder_index;
    uint32_t share_data_len;
    uint32_t occupied;
    uint8_t  share_data[320];
    uint64_t _pad0;
};

struct alignas(16) Contribution {
    uint64_t contribution_id;
    uint64_t ceremony_id;
    uint64_t holder_addr;
    uint32_t round;
    uint32_t holder_index;
    uint32_t payload_len;
    uint32_t status;
    uint8_t  payload[384];
    uint64_t _pad0;
};

struct alignas(16) MPCVMState {
    uint64_t current_epoch;
    uint64_t now_ns;
    uint32_t active_ceremony_count;
    uint32_t finalized_ceremony_count;
    uint32_t failed_ceremony_count;
    uint32_t key_share_count;
    uint8_t  ceremony_root[32];
    uint8_t  key_share_root[32];
    uint8_t  contribution_root[32];
    uint8_t  mpcvm_state_root[32];
};

struct alignas(16) MPCVMRoundDescriptor {
    uint64_t chain_id;
    uint64_t round;
    uint64_t timestamp_ns;
    uint64_t epoch;
    uint32_t mode;
    uint32_t ceremony_op_count;
    uint32_t contribution_op_count;
    uint32_t closing_flag;
    uint32_t _pad0;
    uint32_t _pad1;
    uint64_t _pad2;
    uint8_t  parent_state_root[32];
};

struct alignas(16) CeremonyOp {
    uint64_t ceremony_id;
    uint64_t deadline_ns;
    uint32_t kind;
    uint32_t ceremony_kind;
    uint32_t threshold;
    uint32_t total_participants;
    uint8_t  subject[32];
    uint8_t  ceremony_seed[32];
};

struct alignas(16) ContributionOp {
    uint64_t ceremony_id;
    uint64_t holder_addr;
    uint32_t round;
    uint32_t holder_index;
    uint32_t payload_len;
    uint32_t _pad0;
    uint8_t  payload[384];
};

struct alignas(16) MPCVMTransitionResult {
    uint32_t status;
    uint32_t ceremony_apply_count;
    uint32_t contribution_apply_count;
    uint32_t finalized_this_round;
    uint32_t failed_this_round;
    uint32_t active_ceremony_count;
    uint32_t key_share_count;
    uint32_t round_advance_count;
    uint64_t epoch;
    uint64_t now_ns;
    uint8_t  ceremony_root[32];
    uint8_t  key_share_root[32];
    uint8_t  contribution_root[32];
    uint8_t  mpcvm_state_root[32];
};

__device__ __constant__ uint64_t kKeccakRC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

__device__ __constant__ uint32_t kKeccakRot[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

constexpr uint32_t kCeremonyStatusFree       = 0u;
constexpr uint32_t kCeremonyStatusInProgress = 1u;
constexpr uint32_t kCeremonyStatusFinalized  = 2u;
constexpr uint32_t kCeremonyStatusFailed     = 3u;

constexpr uint32_t kCeremonyOpBegin  = 0u;
constexpr uint32_t kCeremonyOpCancel = 1u;

constexpr uint32_t kKindFrostKeygen   = 0u;
constexpr uint32_t kKindFrostSign     = 1u;
constexpr uint32_t kKindCggmp21Keygen = 2u;
constexpr uint32_t kKindCggmp21Sign   = 3u;
constexpr uint32_t kKindRingtailDkg   = 4u;
constexpr uint32_t kKindRingtailSign  = 5u;

constexpr uint32_t kFrostKeygenRounds   = 3u;
constexpr uint32_t kFrostSignRounds     = 2u;
constexpr uint32_t kCggmp21KeygenRounds = 3u;
constexpr uint32_t kCggmp21SignRounds   = 5u;
constexpr uint32_t kRingtailDkgRounds   = 2u;
constexpr uint32_t kRingtailSignRounds  = 2u;

constexpr uint32_t kSchemeFrost    = 0u;
constexpr uint32_t kSchemeCggmp21  = 1u;
constexpr uint32_t kSchemeRingtail = 2u;

__device__ inline uint64_t rotl64(uint64_t x, uint32_t n) {
    return (x << n) | (x >> (64u - n));
}

__device__ inline void keccak_f1600(uint64_t* s) {
    for (uint32_t round = 0; round < 24u; ++round) {
        uint64_t c[5];
        for (uint32_t x = 0; x < 5u; ++x)
            c[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        uint64_t d[5];
        for (uint32_t x = 0; x < 5u; ++x)
            d[x] = c[(x + 4u) % 5u] ^ rotl64(c[(x + 1u) % 5u], 1u);
        for (uint32_t y = 0; y < 25u; y += 5u)
            for (uint32_t x = 0; x < 5u; ++x)
                s[y + x] ^= d[x];
        uint64_t b[25];
        for (uint32_t y = 0; y < 5u; ++y)
            for (uint32_t x = 0; x < 5u; ++x) {
                uint32_t i = x + 5u * y;
                uint32_t j = y + 5u * ((2u * x + 3u * y) % 5u);
                b[j] = rotl64(s[i], kKeccakRot[i]);
            }
        for (uint32_t y = 0; y < 25u; y += 5u) {
            uint64_t t0 = b[y+0], t1 = b[y+1], t2 = b[y+2], t3 = b[y+3], t4 = b[y+4];
            s[y+0] = t0 ^ ((~t1) & t2);
            s[y+1] = t1 ^ ((~t2) & t3);
            s[y+2] = t2 ^ ((~t3) & t4);
            s[y+3] = t3 ^ ((~t4) & t0);
            s[y+4] = t4 ^ ((~t0) & t1);
        }
        s[0] ^= kKeccakRC[round];
    }
}

__device__ inline void keccak256(const uint8_t* data, uint64_t len, uint8_t* out) {
    uint64_t s[25] = {};
    constexpr uint32_t rate = 136;
    uint64_t off = 0;
    while (len - off >= rate) {
        for (uint32_t i = 0; i < rate; ++i) {
            uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
            s[lane] ^= uint64_t(data[off + i]) << sh;
        }
        keccak_f1600(s);
        off += rate;
    }
    uint8_t block[rate] = {};
    uint64_t rem = len - off;
    for (uint64_t i = 0; i < rem; ++i) block[i] = data[off + i];
    block[rem]      ^= 0x01;
    block[rate - 1] ^= 0x80;
    for (uint32_t i = 0; i < rate; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        s[lane] ^= uint64_t(block[i]) << sh;
    }
    keccak_f1600(s);
    for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        out[i] = uint8_t((s[lane] >> sh) & 0xFFu);
    }
}

__device__ inline void absorb_u32(uint8_t* dst, uint32_t off, uint32_t v) {
    for (uint32_t k = 0; k < 4u; ++k) dst[off + k] = uint8_t((v >> (k*8u)) & 0xFFu);
}
__device__ inline void absorb_u64(uint8_t* dst, uint32_t off, uint64_t v) {
    for (uint32_t k = 0; k < 8u; ++k) dst[off + k] = uint8_t((v >> (k*8u)) & 0xFFu);
}

__device__ inline uint32_t ceremony_index_hash(uint64_t cid, uint32_t mask) {
    uint64_t h = 0xcbf29ce484222325ULL;
    h = (h ^ cid) * 0x100000001b3ULL;
    return uint32_t(h) & mask;
}

__device__ inline uint32_t ceremony_locate(Ceremony* tab, uint32_t count,
                                           uint64_t cid, bool insert_if_missing) {
    uint32_t mask = count - 1u;
    uint32_t idx  = ceremony_index_hash(cid, mask);
    for (uint32_t probe = 0; probe < count; ++probe) {
        Ceremony& s = tab[idx];
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
                for (uint32_t k = 0; k < 32u; ++k) s.subject[k] = 0;
                for (uint32_t k = 0; k < 32u; ++k) s.ceremony_seed[k] = 0;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (s.ceremony_id == cid) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

__device__ inline uint32_t contribution_locate(Contribution* tab, uint32_t count,
                                               uint64_t cid, uint32_t round, uint32_t holder,
                                               bool insert_if_missing) {
    uint32_t mask = count - 1u;
    uint64_t composite = cid ^ ((uint64_t(round) << 32) | uint64_t(holder));
    composite ^= 0x9E3779B97F4A7C15ULL + (cid << 6) + (cid >> 2);
    uint32_t idx = uint32_t(composite) & mask;
    for (uint32_t probe = 0; probe < count; ++probe) {
        Contribution& s = tab[idx];
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

__device__ inline uint32_t key_share_locate_free(KeyShare* tab, uint32_t count,
                                                 uint64_t cid, uint32_t holder) {
    uint32_t mask = count - 1u;
    uint64_t composite = cid ^ ((uint64_t(holder) + 0x9E3779B97F4A7C15ULL)
                                + (cid << 6) + (cid >> 2));
    uint32_t idx = uint32_t(composite) & mask;
    for (uint32_t probe = 0; probe < count; ++probe) {
        KeyShare& s = tab[idx];
        if (s.occupied == 0u) return idx;
        if (s.ceremony_id == cid && s.holder_index == holder) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

__device__ inline uint32_t total_rounds_for(uint32_t kind) {
    if (kind == kKindFrostKeygen)    return kFrostKeygenRounds;
    if (kind == kKindFrostSign)      return kFrostSignRounds;
    if (kind == kKindCggmp21Keygen)  return kCggmp21KeygenRounds;
    if (kind == kKindCggmp21Sign)    return kCggmp21SignRounds;
    if (kind == kKindRingtailDkg)    return kRingtailDkgRounds;
    if (kind == kKindRingtailSign)   return kRingtailSignRounds;
    return 1u;
}

__device__ inline bool is_keygen_kind(uint32_t kind) {
    return kind == kKindFrostKeygen
        || kind == kKindCggmp21Keygen
        || kind == kKindRingtailDkg;
}

__device__ inline uint32_t scheme_for_kind(uint32_t kind) {
    if (kind == kKindFrostKeygen || kind == kKindFrostSign)         return kSchemeFrost;
    if (kind == kKindCggmp21Keygen || kind == kKindCggmp21Sign)     return kSchemeCggmp21;
    return kSchemeRingtail;
}

__device__ inline uint32_t share_data_len_for_scheme(uint32_t scheme) {
    if (scheme == kSchemeFrost)    return 65u;
    if (scheme == kSchemeCggmp21)  return 65u;
    return 256u;
}

}  // namespace mpcvm::cuda
