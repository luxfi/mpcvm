// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file mpcvm_gpu_layout.hpp
/// Shared host/GPU memory layouts for the MPCVM (M-Chain) GPU substrate.
///
/// **Scope** — this module is the **GPU-native ceremony substrate for the
/// M-Chain**. State (ceremonies, key shares, contributions) lives on the
/// GPU and the canonical state-machine transitions also run on the GPU.
/// This is what closes the LP-137 gap from "GPU-resident" to "GPU-native"
/// for threshold MPC: a small dedicated set of ceremony kernels (FROST,
/// CGGMP21, Corona) plus a shared state machine, all matching a
/// deterministic CPU reference byte-for-byte across CPU/Metal/CUDA/WGSL.
///
///   mpcvm/                              the substrate this header describes
///   mpcvm/src/mpcvm_cpu_reference.cpp   deterministic CPU oracle
///   mpcvm/src/mpcvm_*.cu                CUDA kernels
///   mpcvm/src/mpcvm_*.metal             Metal kernels
///   mpcvm/src/mpcvm_*.wgsl              WGSL kernels
///
/// Round model — one MPCVMTransitionRound covers:
///   * FrostKeygen / FrostSign       : threshold-Schnorr ceremonies
///   * Cggmp21Keygen / Cggmp21Sign   : threshold-ECDSA ceremonies
///   * CoronaDkg / CoronaSign    : threshold-lattice ceremonies (link
///                                     to existing lattice GPU)
///   * CeremonyStateMachine          : round advance, timeout, dedup
///   * MpcvmTransition               : emit ceremony_root, key_share_root,
///                                     mpcvm_state_root for Quasar binding
///
/// All offsets here MUST match mpcvm_*.metal / mpcvm_*.cu / mpcvm_*.wgsl
/// byte-for-byte. The CPU reference in mpcvm_cpu_reference.cpp is the
/// equivalence oracle for the four-way determinism contract.

#pragma once

#include <cstdint>

namespace mpcvm::gpu {

// =============================================================================
// Residency class tags
// =============================================================================
//
// MPCVM splits its working set into:
//   * DeviceHot — active ceremonies, recent contributions, key-share table.
//                 GPU memory; transitions read and write here.
//   * HostCold  — completed/expired ceremonies, archived shares. Off-GPU.
//
// A transition kernel only reads/writes DeviceHot arenas. v0.60 ships the
// DeviceHot path only.

enum class ResidencyClass : uint32_t {
    DeviceHot = 0,
    HostCold  = 1,
};

// =============================================================================
// Ceremony arena (DeviceHot)
// =============================================================================
//
// Open-addressing table keyed by ceremony_id. Each slot tracks one
// in-flight (or recently-completed) ceremony of one kind. Round-state
// transitions are gated on participants_bitmap meeting the threshold.
//
// kind:
//   0 frost-keygen     1 frost-sign
//   2 cggmp21-keygen   3 cggmp21-sign
//   4 corona-dkg     5 corona-sign
//
// status:
//   0 free        1 in_progress
//   2 finalized   3 failed (timeout / threshold not met)

enum class CeremonyKind : uint32_t {
    FrostKeygen     = 0,
    FrostSign       = 1,
    Cggmp21Keygen   = 2,
    Cggmp21Sign     = 3,
    CoronaDkg     = 4,
    CoronaSign    = 5,
};

constexpr uint32_t kCeremonyStatusFree       = 0u;
constexpr uint32_t kCeremonyStatusInProgress = 1u;
constexpr uint32_t kCeremonyStatusFinalized  = 2u;
constexpr uint32_t kCeremonyStatusFailed     = 3u;

struct alignas(16) Ceremony {
    uint64_t ceremony_id;            ///< 0 means empty (gated by status)
    uint64_t started_at_ns;
    uint64_t deadline_ns;             ///< absolute deadline; >= now or fail
    uint64_t participants_bitmap;     ///< up to 64 participants per ceremony
    uint32_t kind;                    ///< CeremonyKind
    uint32_t round;                   ///< 0..max_rounds-1
    uint32_t threshold;               ///< t (need t-of-n)
    uint32_t total_participants;      ///< n
    uint32_t status;                  ///< kCeremonyStatus*
    uint32_t contribution_count;      ///< number of contributions in this round
    uint8_t  subject[32];             ///< message hash being signed (or 0 for keygen)
    uint8_t  ceremony_seed[32];       ///< binding seed (chain_id || epoch || nonce)
};
static_assert(sizeof(Ceremony) == 128, "Ceremony layout drift");
static_assert(alignof(Ceremony) == 16, "Ceremony alignment drift");

inline constexpr uint32_t kDefaultCeremonySlots = 256u;

// =============================================================================
// KeyShare arena (DeviceHot)
// =============================================================================
//
// Open-addressing table keyed by share_id. Holds the share data emitted
// by a successful keygen. share_data holds the scheme-specific payload:
//   FROST    : 32-byte secret-share scalar + 33-byte public-key share
//   CGGMP21  : 32-byte secret-share scalar + 33-byte public-key share
//   Corona : up to 256-byte module-LWE secret + 32-byte commitment

enum class ShareScheme : uint32_t {
    Frost    = 0,
    Cggmp21  = 1,
    Corona = 2,
};

inline constexpr uint32_t kKeyShareDataMax = 320u;  ///< max per-share payload

struct alignas(16) KeyShare {
    uint64_t share_id;                ///< 0 means empty (gated by occupied)
    uint64_t ceremony_id;             ///< ceremony that produced the share
    uint64_t holder_addr;             ///< participant address (low 64 bits)
    uint32_t scheme;                  ///< ShareScheme
    uint32_t holder_index;            ///< participant index 0..n-1
    uint32_t share_data_len;
    uint32_t occupied;                ///< 0=free, 1=occupied
    uint8_t  share_data[kKeyShareDataMax];
    uint64_t _pad0;                   ///< pad to 16 alignment
};
static_assert(sizeof(KeyShare) == 368, "KeyShare layout drift");
static_assert(alignof(KeyShare) == 16, "KeyShare alignment drift");

inline constexpr uint32_t kDefaultKeyShareSlots = 4096u;

// =============================================================================
// Contribution arena (DeviceHot)
// =============================================================================
//
// Append-only ring of per-round contributions. Contribution dedup: the
// state machine keeps one contribution per (ceremony_id, round, holder).
// Replay attempts (same triple submitted again) are silently dropped.

inline constexpr uint32_t kContributionPayloadMax = 384u;

struct alignas(16) Contribution {
    uint64_t contribution_id;         ///< monotonic per-arena
    uint64_t ceremony_id;
    uint64_t holder_addr;
    uint32_t round;
    uint32_t holder_index;
    uint32_t payload_len;
    uint32_t status;                  ///< 0=free, 1=accepted, 2=rejected
    uint8_t  payload[kContributionPayloadMax];
    uint64_t _pad0;                   ///< pad to 16 alignment
};
static_assert(sizeof(Contribution) == 432, "Contribution layout drift");
static_assert(alignof(Contribution) == 16, "Contribution alignment drift");

inline constexpr uint32_t kDefaultContributionSlots = 4096u;

// =============================================================================
// MPCVM state (DeviceHot)
// =============================================================================
//
// One slot summarising the current epoch's MPCVM state. Roots feed the
// Quasar round descriptor's mpcvm_state_root binding.

struct alignas(16) MPCVMState {
    uint64_t current_epoch;
    uint64_t now_ns;                  ///< notional "now" used for timeouts
    uint32_t active_ceremony_count;
    uint32_t finalized_ceremony_count;
    uint32_t failed_ceremony_count;
    uint32_t key_share_count;
    uint8_t  ceremony_root[32];       ///< keccak over occupied Ceremony leaves
    uint8_t  key_share_root[32];      ///< keccak over occupied KeyShare leaves
    uint8_t  contribution_root[32];   ///< keccak over accepted contribution leaves
    uint8_t  mpcvm_state_root[32];    ///< composed root (== mpcvm_state_root for Quasar)
};
static_assert(sizeof(MPCVMState) == 160, "MPCVMState layout drift");
static_assert(alignof(MPCVMState) == 16, "MPCVMState alignment drift");

// =============================================================================
// Round descriptor (host -> GPU, written once per round)
// =============================================================================

enum class MPCVMTransitionMode : uint32_t {
    FrostKeygen      = 0,
    FrostSign        = 1,
    Cggmp21Keygen    = 2,
    Cggmp21Sign      = 3,
    CoronaDkg      = 4,
    CoronaSign     = 5,
    CeremonyStep     = 6,   ///< advance-round / timeout sweep / dedup
    FullRound        = 7,   ///< chain ceremony-step + transition
};

struct alignas(16) MPCVMRoundDescriptor {
    uint64_t chain_id;
    uint64_t round;                   ///< monotonic
    uint64_t timestamp_ns;            ///< notional now for timeout sweep
    uint64_t epoch;
    uint32_t mode;                    ///< MPCVMTransitionMode
    uint32_t ceremony_op_count;       ///< new ceremony begins this round
    uint32_t contribution_op_count;
    uint32_t closing_flag;            ///< 1 = run MpcvmTransition at end
    uint32_t _pad0;
    uint32_t _pad1;
    uint64_t _pad2;
    uint8_t  parent_state_root[32];
};
static_assert(sizeof(MPCVMRoundDescriptor) == 96, "MPCVMRoundDescriptor layout drift");
static_assert(alignof(MPCVMRoundDescriptor) == 16,
              "MPCVMRoundDescriptor alignment drift");

// =============================================================================
// Ceremony op (host-supplied input — start a new ceremony)
// =============================================================================

enum class CeremonyOpKind : uint32_t {
    Begin      = 0,       ///< begin a new ceremony (first round, no contributions)
    Cancel     = 1,       ///< explicit cancellation (sets status=failed)
};

struct alignas(16) CeremonyOp {
    uint64_t ceremony_id;
    uint64_t deadline_ns;
    uint32_t kind;                    ///< CeremonyOpKind
    uint32_t ceremony_kind;           ///< CeremonyKind
    uint32_t threshold;
    uint32_t total_participants;
    uint8_t  subject[32];
    uint8_t  ceremony_seed[32];
};
static_assert(sizeof(CeremonyOp) == 96, "CeremonyOp layout drift");
static_assert(alignof(CeremonyOp) == 16, "CeremonyOp alignment drift");

// =============================================================================
// Contribution op (host-supplied input — submit a contribution)
// =============================================================================

struct alignas(16) ContributionOp {
    uint64_t ceremony_id;
    uint64_t holder_addr;
    uint32_t round;
    uint32_t holder_index;
    uint32_t payload_len;
    uint32_t _pad0;
    uint8_t  payload[kContributionPayloadMax];
};
static_assert(sizeof(ContributionOp) == 416, "ContributionOp layout drift");
static_assert(alignof(ContributionOp) == 16, "ContributionOp alignment drift");

// =============================================================================
// Round result (GPU -> host)
// =============================================================================

struct alignas(16) MPCVMTransitionResult {
    uint32_t status;                  ///< 0=in-progress, 1=finalized, 2=needs_state, 3=failed
    uint32_t ceremony_apply_count;    ///< ceremonies started or cancelled
    uint32_t contribution_apply_count;///< contributions accepted (post-dedup)
    uint32_t finalized_this_round;
    uint32_t failed_this_round;
    uint32_t active_ceremony_count;
    uint32_t key_share_count;
    uint32_t round_advance_count;     ///< ceremonies that advanced a round
    uint64_t epoch;
    uint64_t now_ns;
    uint8_t  ceremony_root[32];
    uint8_t  key_share_root[32];
    uint8_t  contribution_root[32];
    uint8_t  mpcvm_state_root[32];    ///< == mpcvm_state_root for Quasar binding
};
static_assert(sizeof(MPCVMTransitionResult) == 176, "MPCVMTransitionResult layout drift");
static_assert(alignof(MPCVMTransitionResult) == 16,
              "MPCVMTransitionResult alignment drift");

// =============================================================================
// Per-scheme constants (round counts, payload sizes)
// =============================================================================

inline constexpr uint32_t kFrostKeygenRounds   = 3u;   ///< commitment, broadcast, share
inline constexpr uint32_t kFrostSignRounds     = 2u;   ///< nonce, partial-sign
inline constexpr uint32_t kCggmp21KeygenRounds = 3u;
inline constexpr uint32_t kCggmp21SignRounds   = 5u;   ///< 4 offline + 1 online
inline constexpr uint32_t kCoronaDkgRounds   = 2u;
inline constexpr uint32_t kCoronaSignRounds  = 2u;

}  // namespace mpcvm::gpu
