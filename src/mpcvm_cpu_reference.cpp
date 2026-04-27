// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file mpcvm_cpu_reference.cpp
/// MPCVM CPU reference — deterministic oracle for cross-backend determinism.
///
/// Mirrors what mpcvm_*.metal / mpcvm_*.cu / mpcvm_*.wgsl must produce
/// byte-for-byte:
///   * Ceremony begin / cancel (in supplied order)
///   * Contribution accept / dedup (one per (ceremony, round, holder))
///   * CeremonyStateMachine sweep:
///       - threshold met in current round -> advance round (or finalize)
///       - deadline elapsed and threshold not met -> mark failed
///   * MpcvmTransition: emit ceremony_root, key_share_root,
///     contribution_root, mpcvm_state_root (composed)
///
/// Determinism contract: identical input -> identical
///   (ceremonies[], key_shares[], contributions[], mpcvm_state) -> identical roots.
///
/// Cryptographic contract — the share emitted by a finalized keygen
/// ceremony is the keccak-derived deterministic function:
///   share_data := keccak(ceremony_seed || ceremony_id || holder_index ||
///                        round_payloads_concat) truncated to scheme-specific
///                        length. This is byte-stable across backends and
///                        bound to the ceremony's seed (preventing replay).
/// The full Ed25519 / secp256k1 / lattice arithmetic for actual signing is
/// performed by host-side verifiers that consume these shares (see
/// quasar_ringtail_verifier.{hpp,cpp} for the analogous lattice path).

#include "lux/mpcvm/mpcvm_cpu_reference.hpp"

#include <algorithm>
#include <array>
#include <cstring>

namespace mpcvm::gpu::ref {

namespace {

// =============================================================================
// keccak256 — bit-identical to pvm_cpu_reference.cpp / quasar
// =============================================================================

constexpr std::array<uint64_t, 24> kKeccakRC = {
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

constexpr std::array<uint32_t, 25> kKeccakRot = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

// Masked rotation — defined for n=0..63. Avoids UB at n=0 from naked
// `x >> (64 - n)` and avoids the branch (which Apple Clang -O3 has been
// observed to miscompile in some loop-unrolled forms).
inline uint64_t rotl64(uint64_t x, uint32_t n) {
    n &= 63u;
    return (x << n) | (x >> ((64u - n) & 63u));
}

// Apple Clang -O3 has been observed to miscompile keccak_f[1600] when the
// 136-byte rate path is exercised through repeated absorbs (XVM caught this).
// MPCVM hits that path on every leaf encoding, so we pin this function to
// no-optimization to lock in the canonical bit pattern across compilers.
#if defined(__clang__) && defined(__APPLE__)
__attribute__((optnone))
#endif
void keccak_f1600(uint64_t* s) {
    for (uint32_t round = 0; round < 24u; ++round) {
        uint64_t c[5];
        for (uint32_t x = 0; x < 5u; ++x)
            c[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        uint64_t d[5];
        for (uint32_t x = 0; x < 5u; ++x)
            d[x] = c[(x + 4u) % 5u] ^ rotl64(c[(x + 1u) % 5u], 1);
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

void keccak256(const uint8_t* data, uint64_t len, uint8_t* out) {
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

void absorb_u32(uint8_t* dst, uint32_t off, uint32_t v) {
    for (uint32_t k = 0; k < 4u; ++k) dst[off + k] = uint8_t((v >> (k*8)) & 0xFFu);
}
void absorb_u64(uint8_t* dst, uint32_t off, uint64_t v) {
    for (uint32_t k = 0; k < 8u; ++k) dst[off + k] = uint8_t((v >> (k*8)) & 0xFFu);
}

// =============================================================================
// Open-addressing locators
// =============================================================================

inline uint32_t ceremony_index(uint64_t ceremony_id, uint32_t mask) {
    uint64_t h = 0xcbf29ce484222325ULL;
    h = (h ^ ceremony_id) * 0x100000001b3ULL;
    return uint32_t(h) & mask;
}

uint32_t ceremony_locate(std::vector<Ceremony>& tab, uint64_t ceremony_id,
                         bool insert_if_missing)
{
    uint32_t mask = uint32_t(tab.size()) - 1u;
    uint32_t idx  = ceremony_index(ceremony_id, mask);
    for (uint32_t probe = 0; probe < tab.size(); ++probe) {
        auto& s = tab[idx];
        if (s.status == kCeremonyStatusFree) {
            if (insert_if_missing) {
                std::memset(&s, 0, sizeof(s));
                s.ceremony_id = ceremony_id;
                s.status = kCeremonyStatusInProgress;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (s.ceremony_id == ceremony_id) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

uint32_t key_share_locate_free(std::vector<KeyShare>& tab,
                               uint64_t ceremony_id, uint32_t holder_index)
{
    // Lookup by (ceremony_id, holder_index); insert into first free slot.
    // Hash composite for direct hit; linear probe for free slot otherwise.
    uint32_t mask = uint32_t(tab.size()) - 1u;
    uint64_t composite = ceremony_id ^ ((uint64_t(holder_index) + 0x9E3779B97F4A7C15ULL)
                                        + (ceremony_id << 6) + (ceremony_id >> 2));
    uint32_t idx = uint32_t(composite) & mask;
    for (uint32_t probe = 0; probe < tab.size(); ++probe) {
        auto& s = tab[idx];
        if (s.occupied == 0u) {
            std::memset(&s, 0, sizeof(s));
            return idx;
        }
        if (s.ceremony_id == ceremony_id && s.holder_index == holder_index) {
            // existing slot for the same (ceremony, holder)
            return idx;
        }
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

uint32_t contribution_locate(std::vector<Contribution>& tab,
                             uint64_t ceremony_id, uint32_t round,
                             uint32_t holder_index, bool insert_if_missing)
{
    uint32_t mask = uint32_t(tab.size()) - 1u;
    uint64_t composite = ceremony_id ^ ((uint64_t(round) << 32) | uint64_t(holder_index));
    composite ^= 0x9E3779B97F4A7C15ULL + (ceremony_id << 6) + (ceremony_id >> 2);
    uint32_t idx = uint32_t(composite) & mask;
    for (uint32_t probe = 0; probe < tab.size(); ++probe) {
        auto& s = tab[idx];
        if (s.status == 0u) {
            if (insert_if_missing) {
                std::memset(&s, 0, sizeof(s));
                s.ceremony_id = ceremony_id;
                s.round = round;
                s.holder_index = holder_index;
                s.status = 1u;  // accepted
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (s.ceremony_id == ceremony_id && s.round == round
            && s.holder_index == holder_index) {
            return idx;
        }
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

uint32_t total_rounds_for(uint32_t kind)
{
    switch (static_cast<CeremonyKind>(kind)) {
        case CeremonyKind::FrostKeygen:    return kFrostKeygenRounds;
        case CeremonyKind::FrostSign:      return kFrostSignRounds;
        case CeremonyKind::Cggmp21Keygen:  return kCggmp21KeygenRounds;
        case CeremonyKind::Cggmp21Sign:    return kCggmp21SignRounds;
        case CeremonyKind::RingtailDkg:    return kRingtailDkgRounds;
        case CeremonyKind::RingtailSign:   return kRingtailSignRounds;
        default: return 1u;
    }
}

bool is_keygen_kind(uint32_t kind)
{
    auto k = static_cast<CeremonyKind>(kind);
    return k == CeremonyKind::FrostKeygen
        || k == CeremonyKind::Cggmp21Keygen
        || k == CeremonyKind::RingtailDkg;
}

uint32_t scheme_for_kind(uint32_t kind)
{
    auto k = static_cast<CeremonyKind>(kind);
    if (k == CeremonyKind::FrostKeygen || k == CeremonyKind::FrostSign)
        return uint32_t(ShareScheme::Frost);
    if (k == CeremonyKind::Cggmp21Keygen || k == CeremonyKind::Cggmp21Sign)
        return uint32_t(ShareScheme::Cggmp21);
    return uint32_t(ShareScheme::Ringtail);
}

uint32_t share_data_len_for_scheme(uint32_t scheme)
{
    if (scheme == uint32_t(ShareScheme::Frost))    return 65u;  // 32 secret + 33 pub
    if (scheme == uint32_t(ShareScheme::Cggmp21))  return 65u;  // 32 secret + 33 pub
    return 256u;  // Ringtail Module-LWE secret + 32 pub
}

// Count contributions in the current round for a ceremony.
uint32_t count_contributions(const std::vector<Contribution>& tab,
                             uint64_t ceremony_id, uint32_t round)
{
    uint32_t n = 0;
    for (const auto& c : tab) {
        if (c.status == 0u) continue;
        if (c.ceremony_id == ceremony_id && c.round == round && c.status == 1u)
            ++n;
    }
    return n;
}

// =============================================================================
// Ceremony begin / cancel
// =============================================================================

uint32_t apply_ceremony_ops(MPCVMReferenceState& state,
                            std::span<const CeremonyOp> ops)
{
    uint32_t applied = 0;
    for (const auto& op : ops) {
        switch (static_cast<CeremonyOpKind>(op.kind)) {
            case CeremonyOpKind::Begin: {
                if (op.threshold == 0u || op.threshold > op.total_participants) break;
                if (op.total_participants > 64u) break;  // bitmap is 64-bit
                uint32_t idx = ceremony_locate(state.ceremonies, op.ceremony_id, true);
                if (idx == 0xFFFFFFFFu) break;
                auto& c = state.ceremonies[idx];
                // If slot was newly inserted, status was set to InProgress.
                // Re-init full record.
                c.kind = op.ceremony_kind;
                c.threshold = op.threshold;
                c.total_participants = op.total_participants;
                c.deadline_ns = op.deadline_ns;
                c.round = 0;
                c.contribution_count = 0;
                c.participants_bitmap = 0;
                c.status = kCeremonyStatusInProgress;
                std::memcpy(c.subject, op.subject, 32);
                std::memcpy(c.ceremony_seed, op.ceremony_seed, 32);
                ++applied;
                break;
            }
            case CeremonyOpKind::Cancel: {
                uint32_t idx = ceremony_locate(state.ceremonies, op.ceremony_id, false);
                if (idx == 0xFFFFFFFFu) break;
                auto& c = state.ceremonies[idx];
                if (c.status != kCeremonyStatusInProgress) break;
                c.status = kCeremonyStatusFailed;
                ++applied;
                break;
            }
        }
    }
    return applied;
}

// =============================================================================
// Contribution accept / dedup
// =============================================================================

uint32_t apply_contribution_ops(MPCVMReferenceState& state,
                                std::span<const ContributionOp> ops)
{
    uint32_t applied = 0;
    for (const auto& op : ops) {
        if (op.payload_len > kContributionPayloadMax) continue;
        uint32_t cidx = ceremony_locate(state.ceremonies, op.ceremony_id, false);
        if (cidx == 0xFFFFFFFFu) continue;
        auto& c = state.ceremonies[cidx];
        if (c.status != kCeremonyStatusInProgress) continue;
        if (op.round != c.round) continue;            // wrong round = reject
        if (op.holder_index >= c.total_participants) continue;

        // Replay check via locator: insert_if_missing = false to detect.
        uint32_t existing = contribution_locate(state.contributions,
                                                op.ceremony_id, op.round,
                                                op.holder_index, false);
        if (existing != 0xFFFFFFFFu) continue;        // already submitted = drop

        // Reserve a fresh slot.
        uint32_t cont_idx = contribution_locate(state.contributions,
                                                op.ceremony_id, op.round,
                                                op.holder_index, true);
        if (cont_idx == 0xFFFFFFFFu) continue;
        auto& cont = state.contributions[cont_idx];
        cont.contribution_id = state.next_contribution_id++;
        cont.holder_addr = op.holder_addr;
        cont.payload_len = op.payload_len;
        std::memcpy(cont.payload, op.payload, op.payload_len);

        // Update ceremony bitmap + counter.
        uint64_t bit = uint64_t(1) << op.holder_index;
        if ((c.participants_bitmap & bit) == 0u) {
            c.participants_bitmap |= bit;
            c.contribution_count = count_contributions(state.contributions,
                                                       op.ceremony_id, op.round);
        }
        ++applied;
    }
    return applied;
}

// =============================================================================
// Share emission for a finalized keygen ceremony
// =============================================================================
//
// share_data := keccak(ceremony_seed || ceremony_id_le8 || holder_index_le4 ||
//                      "MPCVM-SHARE-V1" || all_round_payloads_concat),
// truncated to share_data_len_for_scheme(scheme). This is byte-stable
// across backends, bound to the ceremony seed (no replay across ceremonies),
// and threshold-correct (each holder gets a distinct share).

void emit_keygen_shares(MPCVMReferenceState& state, Ceremony& c)
{
    uint32_t scheme = scheme_for_kind(c.kind);
    uint32_t out_len = share_data_len_for_scheme(scheme);
    uint32_t total_rounds = total_rounds_for(c.kind);

    // Each participant that contributed at least once gets a share.
    for (uint32_t holder = 0; holder < c.total_participants; ++holder) {
        uint64_t bit = uint64_t(1) << holder;
        if ((c.participants_bitmap & bit) == 0u) continue;

        // Build seed buffer.
        std::vector<uint8_t> buf;
        buf.reserve(32 + 8 + 4 + 16 + total_rounds * 64);
        buf.insert(buf.end(), c.ceremony_seed, c.ceremony_seed + 32);
        for (uint32_t k = 0; k < 8u; ++k)
            buf.push_back(uint8_t((c.ceremony_id >> (k*8)) & 0xFFu));
        for (uint32_t k = 0; k < 4u; ++k)
            buf.push_back(uint8_t((holder >> (k*8)) & 0xFFu));
        const char* tag = "MPCVM-SHARE-V1";
        for (uint32_t k = 0; k < 14u; ++k) buf.push_back(uint8_t(tag[k]));
        // include the holder's contributions across all rounds, in round order
        for (uint32_t r = 0; r < total_rounds; ++r) {
            uint32_t cidx = contribution_locate(state.contributions,
                                                c.ceremony_id, r, holder, false);
            if (cidx == 0xFFFFFFFFu) continue;
            const auto& cont = state.contributions[cidx];
            for (uint32_t k = 0; k < cont.payload_len; ++k)
                buf.push_back(cont.payload[k]);
        }

        // Stretch keccak to out_len bytes.
        uint32_t kidx = key_share_locate_free(state.key_shares, c.ceremony_id, holder);
        if (kidx == 0xFFFFFFFFu) continue;
        auto& ks = state.key_shares[kidx];
        if (ks.occupied == 0u) {
            ks.share_id = state.next_share_id++;
            ks.ceremony_id = c.ceremony_id;
            ks.holder_index = holder;
            ks.scheme = scheme;
            ks.occupied = 1u;
        }
        ks.share_data_len = out_len;

        uint8_t prev[32] = {};
        keccak256(buf.data(), buf.size(), prev);
        uint32_t written = 0;
        while (written < out_len) {
            uint32_t take = std::min<uint32_t>(32u, out_len - written);
            std::memcpy(ks.share_data + written, prev, take);
            written += take;
            if (written < out_len) {
                // Re-hash with counter byte to extend.
                uint8_t ext[33];
                std::memcpy(ext, prev, 32);
                ext[32] = uint8_t(written / 32u);
                keccak256(ext, 33, prev);
            }
        }
    }
}

// =============================================================================
// CeremonyStateMachine sweep
// =============================================================================

void run_ceremony_step(MPCVMReferenceState& state,
                       const MPCVMRoundDescriptor& desc,
                       MPCVMTransitionResult& r)
{
    for (auto& c : state.ceremonies) {
        if (c.status != kCeremonyStatusInProgress) continue;

        uint32_t in_round = count_contributions(state.contributions,
                                                c.ceremony_id, c.round);
        c.contribution_count = in_round;

        if (in_round >= c.threshold) {
            // Threshold met — advance round (or finalize).
            uint32_t total_rounds = total_rounds_for(c.kind);
            ++c.round;
            ++r.round_advance_count;
            if (c.round >= total_rounds) {
                c.status = kCeremonyStatusFinalized;
                ++r.finalized_this_round;
                if (is_keygen_kind(c.kind)) {
                    emit_keygen_shares(state, c);
                }
            } else {
                // Reset bitmap for next round's contributions.
                c.participants_bitmap = 0;
                c.contribution_count = 0;
            }
            continue;
        }

        // Threshold not met — check deadline.
        if (desc.timestamp_ns > c.deadline_ns) {
            c.status = kCeremonyStatusFailed;
            ++r.failed_this_round;
        }
    }
}

// =============================================================================
// Root computation (canonical leaf encoding)
// =============================================================================

void compute_ceremony_root(const std::vector<Ceremony>& ceremonies,
                           uint8_t out[32],
                           uint32_t& active_count,
                           uint32_t& finalized_count,
                           uint32_t& failed_count)
{
    std::array<uint8_t, 32> acc{};
    active_count = finalized_count = failed_count = 0;
    for (uint32_t i = 0; i < ceremonies.size(); ++i) {
        const auto& c = ceremonies[i];
        if (c.status == kCeremonyStatusFree) continue;
        if (c.status == kCeremonyStatusInProgress) ++active_count;
        if (c.status == kCeremonyStatusFinalized)  ++finalized_count;
        if (c.status == kCeremonyStatusFailed)     ++failed_count;

        // leaf = keccak(ceremony_id || started || deadline || bitmap ||
        //              kind || round || threshold || total ||
        //              status || contribution_count || subject || seed || index)
        uint8_t leaf[8 + 8 + 8 + 8 + 4 + 4 + 4 + 4 + 4 + 4 + 32 + 32 + 4] = {};
        uint32_t o = 0;
        absorb_u64(leaf, o, c.ceremony_id);          o += 8;
        absorb_u64(leaf, o, c.started_at_ns);         o += 8;
        absorb_u64(leaf, o, c.deadline_ns);           o += 8;
        absorb_u64(leaf, o, c.participants_bitmap);   o += 8;
        absorb_u32(leaf, o, c.kind);                  o += 4;
        absorb_u32(leaf, o, c.round);                 o += 4;
        absorb_u32(leaf, o, c.threshold);             o += 4;
        absorb_u32(leaf, o, c.total_participants);    o += 4;
        absorb_u32(leaf, o, c.status);                o += 4;
        absorb_u32(leaf, o, c.contribution_count);    o += 4;
        std::memcpy(leaf + o, c.subject, 32);         o += 32;
        std::memcpy(leaf + o, c.ceremony_seed, 32);   o += 32;
        absorb_u32(leaf, o, i);                       o += 4;

        uint8_t leaf_hash[32];
        keccak256(leaf, o, leaf_hash);
        uint8_t buf[64];
        std::memcpy(buf, acc.data(), 32);
        std::memcpy(buf + 32, leaf_hash, 32);
        keccak256(buf, 64, acc.data());
    }
    std::memcpy(out, acc.data(), 32);
}

void compute_key_share_root(const std::vector<KeyShare>& shares,
                            uint8_t out[32],
                            uint32_t& share_count)
{
    std::array<uint8_t, 32> acc{};
    share_count = 0;
    for (uint32_t i = 0; i < shares.size(); ++i) {
        const auto& s = shares[i];
        if (s.occupied == 0u) continue;
        ++share_count;

        // leaf = keccak(share_id || ceremony_id || holder_addr || scheme ||
        //              holder_index || share_data_len || share_data[..len] || index)
        std::vector<uint8_t> leaf;
        leaf.reserve(8 + 8 + 8 + 4 + 4 + 4 + s.share_data_len + 4);
        for (uint32_t k = 0; k < 8u; ++k) leaf.push_back(uint8_t((s.share_id >> (k*8)) & 0xFFu));
        for (uint32_t k = 0; k < 8u; ++k) leaf.push_back(uint8_t((s.ceremony_id >> (k*8)) & 0xFFu));
        for (uint32_t k = 0; k < 8u; ++k) leaf.push_back(uint8_t((s.holder_addr >> (k*8)) & 0xFFu));
        for (uint32_t k = 0; k < 4u; ++k) leaf.push_back(uint8_t((s.scheme >> (k*8)) & 0xFFu));
        for (uint32_t k = 0; k < 4u; ++k) leaf.push_back(uint8_t((s.holder_index >> (k*8)) & 0xFFu));
        for (uint32_t k = 0; k < 4u; ++k) leaf.push_back(uint8_t((s.share_data_len >> (k*8)) & 0xFFu));
        for (uint32_t k = 0; k < s.share_data_len; ++k) leaf.push_back(s.share_data[k]);
        for (uint32_t k = 0; k < 4u; ++k) leaf.push_back(uint8_t((i >> (k*8)) & 0xFFu));

        uint8_t leaf_hash[32];
        keccak256(leaf.data(), leaf.size(), leaf_hash);
        uint8_t buf[64];
        std::memcpy(buf, acc.data(), 32);
        std::memcpy(buf + 32, leaf_hash, 32);
        keccak256(buf, 64, acc.data());
    }
    std::memcpy(out, acc.data(), 32);
}

void compute_contribution_root(const std::vector<Contribution>& tab,
                               uint8_t out[32])
{
    std::array<uint8_t, 32> acc{};
    for (uint32_t i = 0; i < tab.size(); ++i) {
        const auto& c = tab[i];
        if (c.status != 1u) continue;  // accepted only

        std::vector<uint8_t> leaf;
        leaf.reserve(8 + 8 + 8 + 4 + 4 + 4 + c.payload_len + 4);
        for (uint32_t k = 0; k < 8u; ++k) leaf.push_back(uint8_t((c.contribution_id >> (k*8)) & 0xFFu));
        for (uint32_t k = 0; k < 8u; ++k) leaf.push_back(uint8_t((c.ceremony_id >> (k*8)) & 0xFFu));
        for (uint32_t k = 0; k < 8u; ++k) leaf.push_back(uint8_t((c.holder_addr >> (k*8)) & 0xFFu));
        for (uint32_t k = 0; k < 4u; ++k) leaf.push_back(uint8_t((c.round >> (k*8)) & 0xFFu));
        for (uint32_t k = 0; k < 4u; ++k) leaf.push_back(uint8_t((c.holder_index >> (k*8)) & 0xFFu));
        for (uint32_t k = 0; k < 4u; ++k) leaf.push_back(uint8_t((c.payload_len >> (k*8)) & 0xFFu));
        for (uint32_t k = 0; k < c.payload_len; ++k) leaf.push_back(c.payload[k]);
        for (uint32_t k = 0; k < 4u; ++k) leaf.push_back(uint8_t((i >> (k*8)) & 0xFFu));

        uint8_t leaf_hash[32];
        keccak256(leaf.data(), leaf.size(), leaf_hash);
        uint8_t buf[64];
        std::memcpy(buf, acc.data(), 32);
        std::memcpy(buf + 32, leaf_hash, 32);
        keccak256(buf, 64, acc.data());
    }
    std::memcpy(out, acc.data(), 32);
}

void close_round(MPCVMReferenceState& state,
                 const MPCVMRoundDescriptor& desc,
                 MPCVMTransitionResult& r)
{
    uint32_t active = 0, finalized = 0, failed = 0;
    compute_ceremony_root(state.ceremonies, state.state.ceremony_root,
                          active, finalized, failed);
    uint32_t share_count = 0;
    compute_key_share_root(state.key_shares, state.state.key_share_root,
                           share_count);
    compute_contribution_root(state.contributions, state.state.contribution_root);

    state.state.active_ceremony_count = active;
    state.state.finalized_ceremony_count = finalized;
    state.state.failed_ceremony_count = failed;
    state.state.key_share_count = share_count;
    state.state.now_ns = desc.timestamp_ns;
    if (desc.closing_flag != 0u) {
        state.state.current_epoch = desc.epoch + 1u;
    }

    // mpcvm_state_root = keccak(parent || ceremony_root || key_share_root ||
    //                          contribution_root || epoch_u64 || now_u64 ||
    //                          active_u32 || finalized_u32 || failed_u32 || shares_u32)
    uint8_t composed[32 + 32 + 32 + 32 + 8 + 8 + 4 + 4 + 4 + 4] = {};
    uint32_t o = 0;
    std::memcpy(composed + o, desc.parent_state_root, 32);            o += 32;
    std::memcpy(composed + o, state.state.ceremony_root, 32);         o += 32;
    std::memcpy(composed + o, state.state.key_share_root, 32);        o += 32;
    std::memcpy(composed + o, state.state.contribution_root, 32);     o += 32;
    absorb_u64(composed, o, state.state.current_epoch);               o += 8;
    absorb_u64(composed, o, state.state.now_ns);                      o += 8;
    absorb_u32(composed, o, state.state.active_ceremony_count);       o += 4;
    absorb_u32(composed, o, state.state.finalized_ceremony_count);    o += 4;
    absorb_u32(composed, o, state.state.failed_ceremony_count);       o += 4;
    absorb_u32(composed, o, state.state.key_share_count);             o += 4;
    keccak256(composed, o, state.state.mpcvm_state_root);

    std::memcpy(r.ceremony_root,        state.state.ceremony_root,        32);
    std::memcpy(r.key_share_root,       state.state.key_share_root,       32);
    std::memcpy(r.contribution_root,    state.state.contribution_root,    32);
    std::memcpy(r.mpcvm_state_root,     state.state.mpcvm_state_root,     32);
    r.active_ceremony_count = active;
    r.key_share_count = share_count;
    r.epoch = state.state.current_epoch;
    r.now_ns = state.state.now_ns;
}

}  // anonymous namespace

MPCVMReferenceState MPCVMReferenceState::empty()
{
    MPCVMReferenceState s;
    s.ceremonies.assign(kDefaultCeremonySlots, Ceremony{});
    s.key_shares.assign(kDefaultKeyShareSlots, KeyShare{});
    s.contributions.assign(kDefaultContributionSlots, Contribution{});
    s.state = MPCVMState{};
    s.next_share_id = 1;
    s.next_contribution_id = 1;
    return s;
}

MPCVMTransitionResult run_reference(MPCVMReferenceState& state,
                                    const MPCVMRoundDescriptor& desc,
                                    std::span<const CeremonyOp>     ceremony_ops,
                                    std::span<const ContributionOp> contribution_ops)
{
    if (state.ceremonies.empty())
        state.ceremonies.assign(kDefaultCeremonySlots, Ceremony{});
    if (state.key_shares.empty())
        state.key_shares.assign(kDefaultKeyShareSlots, KeyShare{});
    if (state.contributions.empty())
        state.contributions.assign(kDefaultContributionSlots, Contribution{});

    MPCVMTransitionResult r{};
    auto mode = static_cast<MPCVMTransitionMode>(desc.mode);

    if (mode == MPCVMTransitionMode::FullRound
        || mode == MPCVMTransitionMode::CeremonyStep
        || mode == MPCVMTransitionMode::FrostKeygen
        || mode == MPCVMTransitionMode::FrostSign
        || mode == MPCVMTransitionMode::Cggmp21Keygen
        || mode == MPCVMTransitionMode::Cggmp21Sign
        || mode == MPCVMTransitionMode::RingtailDkg
        || mode == MPCVMTransitionMode::RingtailSign) {
        r.ceremony_apply_count    = apply_ceremony_ops(state, ceremony_ops);
        r.contribution_apply_count = apply_contribution_ops(state, contribution_ops);
        run_ceremony_step(state, desc, r);
    }

    // Always close round so caller gets fresh roots.
    close_round(state, desc, r);
    r.status = 1u;
    return r;
}

}  // namespace mpcvm::gpu::ref
