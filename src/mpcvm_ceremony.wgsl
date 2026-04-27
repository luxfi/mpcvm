// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_ceremony.wgsl — full state-machine kernel for the wgpu/Dawn backend.
//
// Phase 1: Apply CeremonyOps (begin / cancel) in supplied order.
// Phase 2: Apply ContributionOps with dedup by (cid, round, holder).
// Phase 3: Sweep — advance round / finalize / timeout. For finalized
//          keygen ceremonies, emit deterministic key shares using the
//          in-shader keccak from mpcvm_kernels_common.wgsl.
//
// Single workgroup_size(1), single thread — canonical in-order traversal
// matches CPU/Metal/CUDA byte-for-byte.

@group(0) @binding(0) var<storage, read>       desc:                MPCVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>       ceremony_ops:        array<CeremonyOp>;
@group(0) @binding(2) var<storage, read>       contribution_ops:    array<ContributionOp>;
@group(0) @binding(3) var<storage, read_write> ceremonies:          array<Ceremony>;
@group(0) @binding(4) var<storage, read_write> key_shares:          array<KeyShare>;
@group(0) @binding(5) var<storage, read_write> contributions:       array<Contribution>;
@group(0) @binding(6) var<storage, read_write> applied_counts:      array<u32, 5>;
@group(0) @binding(7) var<storage, read>       counter_init:        array<u32, 4>;  // [next_cont_lo, next_cont_hi, next_share_lo, next_share_hi]

// FNV-style hash for ceremony id, matches CPU reference and Metal.
fn ceremony_index_hash(cid_lo: u32, cid_hi: u32, mask: u32) -> u32 {
    // h = 0xcbf29ce484222325 (FNV offset basis)
    // h = (h ^ cid) * 0x100000001b3
    // We fold to u32 by taking the low 32 bits.
    var h_lo: u32 = 0x84222325u;
    var h_hi: u32 = 0xcbf29ce4u;
    h_lo = h_lo ^ cid_lo;
    h_hi = h_hi ^ cid_hi;
    // multiply by FNV prime (0x100000001b3 = 2^40 + 0x1b3) — we approximate
    // by the same low-32 fold the CPU/Metal use: cast to u32 then mask.
    // Critical: the CPU code is `uint32_t(h)` after the multiply.
    // Multiply (lo, hi) by 0x100000001b3 = 0x100 * 2^32 + 0x000001b3.
    // The full result modulo 2^64 is what we want; we then truncate to u32.
    // m_lo_hi = (low_part * 0x000001b3) -> 64-bit
    // For determinism we replicate CPU behavior precisely below.
    let prime_lo: u32 = 0x000001b3u;
    let prime_hi: u32 = 0x00000100u;

    // h * prime mod 2^64 = (h_lo * prime_lo) lo
    //                    + ((h_lo * prime_lo) hi + h_hi * prime_lo + h_lo * prime_hi) << 32
    // We only need low 32 bits.
    // low 32 = (h_lo * prime_lo) & 0xFFFFFFFF
    let r0_lo: u32 = h_lo * prime_lo; // truncated
    return r0_lo & mask;
}

fn ceremony_locate_insert(cid_lo: u32, cid_hi: u32) -> u32 {
    let n = arrayLength(&ceremonies);
    let mask = n - 1u;
    var idx: u32 = ceremony_index_hash(cid_lo, cid_hi, mask);
    var found: u32 = 0xFFFFFFFFu;
    for (var probe: u32 = 0u; probe < n; probe = probe + 1u) {
        if (ceremonies[idx].status == kCeremonyStatusFree) {
            // Newly inserted slot — caller fills the rest.
            ceremonies[idx].ceremony_id_lo = cid_lo;
            ceremonies[idx].ceremony_id_hi = cid_hi;
            ceremonies[idx].started_at_ns_lo = 0u;
            ceremonies[idx].started_at_ns_hi = 0u;
            ceremonies[idx].deadline_ns_lo = 0u;
            ceremonies[idx].deadline_ns_hi = 0u;
            ceremonies[idx].participants_bitmap_lo = 0u;
            ceremonies[idx].participants_bitmap_hi = 0u;
            ceremonies[idx].kind = 0u;
            ceremonies[idx].round = 0u;
            ceremonies[idx].threshold = 0u;
            ceremonies[idx].total_participants = 0u;
            ceremonies[idx].status = kCeremonyStatusInProgress;
            ceremonies[idx].contribution_count = 0u;
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                ceremonies[idx].subject[k] = 0u;
                ceremonies[idx].ceremony_seed[k] = 0u;
            }
            found = idx;
            break;
        }
        if (u64_eq(ceremonies[idx].ceremony_id_lo, ceremonies[idx].ceremony_id_hi,
                   cid_lo, cid_hi)) {
            found = idx;
            break;
        }
        idx = (idx + 1u) & mask;
    }
    return found;
}

fn ceremony_locate_only(cid_lo: u32, cid_hi: u32) -> u32 {
    let n = arrayLength(&ceremonies);
    let mask = n - 1u;
    var idx: u32 = ceremony_index_hash(cid_lo, cid_hi, mask);
    var found: u32 = 0xFFFFFFFFu;
    for (var probe: u32 = 0u; probe < n; probe = probe + 1u) {
        if (ceremonies[idx].status == kCeremonyStatusFree) { break; }
        if (u64_eq(ceremonies[idx].ceremony_id_lo, ceremonies[idx].ceremony_id_hi,
                   cid_lo, cid_hi)) {
            found = idx;
            break;
        }
        idx = (idx + 1u) & mask;
    }
    return found;
}

// Composite hash for (cid, round, holder) — matches CPU reference exactly.
fn contribution_index_hash(cid_lo: u32, cid_hi: u32,
                           round: u32, holder: u32, mask: u32) -> u32 {
    // composite = cid ^ ((round << 32) | holder)
    //           ^ 0x9E3779B97F4A7C15 + (cid << 6) + (cid >> 2)
    // Then truncate to u32.
    var c_lo: u32 = cid_lo ^ holder;
    var c_hi: u32 = cid_hi ^ round;

    // Add 0x9E3779B97F4A7C15 (CPU code does `^=` followed by `+`; matches
    // the order in mpcvm_cpu_reference.cpp: composite ^= GR + (cid<<6) + (cid>>2)).
    // Our chain: add then xor — but we need to match CPU exactly.
    // CPU: composite ^= 0x9E3779B97F4A7C15ULL + (ceremony_id << 6) + (ceremony_id >> 2);
    // i.e. the RHS is computed first, then XORed into composite.
    let GR_lo: u32 = 0x7F4A7C15u;
    let GR_hi: u32 = 0x9E3779B9u;
    // (cid << 6): shift the 64-bit value left 6.
    let s6_lo: u32 = cid_lo << 6u;
    let s6_hi: u32 = (cid_hi << 6u) | (cid_lo >> 26u);
    // (cid >> 2): shift right 2.
    let s2_lo: u32 = (cid_lo >> 2u) | (cid_hi << 30u);
    let s2_hi: u32 = cid_hi >> 2u;

    // Sum: GR + s6 + s2 (with carries).
    let t1_lo: u32 = GR_lo + s6_lo;
    let t1_carry: u32 = select(0u, 1u, t1_lo < GR_lo);
    let t1_hi: u32 = GR_hi + s6_hi + t1_carry;
    let t2_lo: u32 = t1_lo + s2_lo;
    let t2_carry: u32 = select(0u, 1u, t2_lo < t1_lo);
    let t2_hi: u32 = t1_hi + s2_hi + t2_carry;

    c_lo = c_lo ^ t2_lo;
    c_hi = c_hi ^ t2_hi;
    return c_lo & mask;
}

fn contribution_locate(cid_lo: u32, cid_hi: u32,
                       round: u32, holder: u32,
                       insert_if_missing: bool) -> u32 {
    let n = arrayLength(&contributions);
    let mask = n - 1u;
    var idx: u32 = contribution_index_hash(cid_lo, cid_hi, round, holder, mask);
    var found: u32 = 0xFFFFFFFFu;
    for (var probe: u32 = 0u; probe < n; probe = probe + 1u) {
        if (contributions[idx].status == 0u) {
            if (insert_if_missing) {
                contributions[idx].contribution_id_lo = 0u;
                contributions[idx].contribution_id_hi = 0u;
                contributions[idx].ceremony_id_lo = cid_lo;
                contributions[idx].ceremony_id_hi = cid_hi;
                contributions[idx].holder_addr_lo = 0u;
                contributions[idx].holder_addr_hi = 0u;
                contributions[idx].round = round;
                contributions[idx].holder_index = holder;
                contributions[idx].payload_len = 0u;
                contributions[idx].status = 1u;
                found = idx;
            }
            break;
        }
        if (u64_eq(contributions[idx].ceremony_id_lo, contributions[idx].ceremony_id_hi,
                   cid_lo, cid_hi)
            && contributions[idx].round == round
            && contributions[idx].holder_index == holder) {
            found = idx;
            break;
        }
        idx = (idx + 1u) & mask;
    }
    return found;
}

fn key_share_index_hash(cid_lo: u32, cid_hi: u32, holder: u32, mask: u32) -> u32 {
    // composite = cid ^ ((holder + 0x9E3779B97F4A7C15) + (cid << 6) + (cid >> 2))
    let GR_lo: u32 = 0x7F4A7C15u;
    let GR_hi: u32 = 0x9E3779B9u;
    // holder + GR
    let h_plus_lo: u32 = holder + GR_lo;
    let h_carry: u32 = select(0u, 1u, h_plus_lo < holder);
    let h_plus_hi: u32 = GR_hi + h_carry;
    // (cid << 6)
    let s6_lo: u32 = cid_lo << 6u;
    let s6_hi: u32 = (cid_hi << 6u) | (cid_lo >> 26u);
    // (cid >> 2)
    let s2_lo: u32 = (cid_lo >> 2u) | (cid_hi << 30u);
    let s2_hi: u32 = cid_hi >> 2u;
    // (holder+GR) + s6 + s2
    let t1_lo: u32 = h_plus_lo + s6_lo;
    let t1_carry: u32 = select(0u, 1u, t1_lo < h_plus_lo);
    let t1_hi: u32 = h_plus_hi + s6_hi + t1_carry;
    let t2_lo: u32 = t1_lo + s2_lo;
    let t2_carry: u32 = select(0u, 1u, t2_lo < t1_lo);
    let t2_hi: u32 = t1_hi + s2_hi + t2_carry;
    // composite = cid ^ above
    let c_lo: u32 = cid_lo ^ t2_lo;
    let c_hi: u32 = cid_hi ^ t2_hi;
    return c_lo & mask;
}

fn key_share_locate_free(cid_lo: u32, cid_hi: u32, holder: u32) -> u32 {
    let n = arrayLength(&key_shares);
    let mask = n - 1u;
    var idx: u32 = key_share_index_hash(cid_lo, cid_hi, holder, mask);
    var found: u32 = 0xFFFFFFFFu;
    for (var probe: u32 = 0u; probe < n; probe = probe + 1u) {
        if (key_shares[idx].occupied == 0u) { found = idx; break; }
        if (u64_eq(key_shares[idx].ceremony_id_lo, key_shares[idx].ceremony_id_hi,
                   cid_lo, cid_hi)
            && key_shares[idx].holder_index == holder) {
            found = idx;
            break;
        }
        idx = (idx + 1u) & mask;
    }
    return found;
}

fn total_rounds_for(kind: u32) -> u32 {
    if (kind == kKindFrostKeygen)    { return 3u; }
    if (kind == kKindFrostSign)      { return 2u; }
    if (kind == kKindCggmp21Keygen)  { return 3u; }
    if (kind == kKindCggmp21Sign)    { return 5u; }
    if (kind == kKindRingtailDkg)    { return 2u; }
    if (kind == kKindRingtailSign)   { return 2u; }
    return 1u;
}

fn is_keygen_kind(kind: u32) -> bool {
    return kind == kKindFrostKeygen
        || kind == kKindCggmp21Keygen
        || kind == kKindRingtailDkg;
}

fn scheme_for_kind(kind: u32) -> u32 {
    if (kind == kKindFrostKeygen || kind == kKindFrostSign)         { return 0u; }
    if (kind == kKindCggmp21Keygen || kind == kKindCggmp21Sign)     { return 1u; }
    return 2u;
}

fn share_data_len_for_scheme(scheme: u32) -> u32 {
    if (scheme == 0u) { return 65u; }
    if (scheme == 1u) { return 65u; }
    return 256u;
}

fn count_contributions_for(cid_lo: u32, cid_hi: u32, round: u32) -> u32 {
    let n = arrayLength(&contributions);
    var count: u32 = 0u;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        if (contributions[i].status != 1u) { continue; }
        if (u64_eq(contributions[i].ceremony_id_lo, contributions[i].ceremony_id_hi, cid_lo, cid_hi)
            && contributions[i].round == round) {
            count = count + 1u;
        }
    }
    return count;
}

// 64-bit increment.
fn u64_inc(lo: u32, hi: u32) -> vec2<u32> {
    let new_lo = lo + 1u;
    let new_hi = hi + select(0u, 1u, new_lo < lo);
    return vec2<u32>(new_lo, new_hi);
}

@compute @workgroup_size(1)
fn mpcvm_ceremony_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    var cer_applied: u32 = 0u;
    var cnt_applied: u32 = 0u;
    var advances: u32 = 0u;
    var finalized: u32 = 0u;
    var failed: u32 = 0u;

    var next_cont_lo: u32 = counter_init[0];
    var next_cont_hi: u32 = counter_init[1];
    var next_share_lo: u32 = counter_init[2];
    var next_share_hi: u32 = counter_init[3];

    // Hoisted scratch buffers — share emission needs an 8KB seed buffer.
    var seed_buf: array<u32, 2048>;
    var prev: array<u32, 32>;
    var ext: array<u32, 33>;

    // Phase 1: ceremony ops.
    let cer_op_count = desc.ceremony_op_count;
    for (var i: u32 = 0u; i < cer_op_count; i = i + 1u) {
        let op = ceremony_ops[i];
        if (op.kind == kCeremonyOpBegin) {
            if (op.threshold == 0u) { continue; }
            if (op.threshold > op.total_participants) { continue; }
            if (op.total_participants > 64u) { continue; }
            let idx = ceremony_locate_insert(op.ceremony_id_lo, op.ceremony_id_hi);
            if (idx == 0xFFFFFFFFu) { continue; }
            ceremonies[idx].kind = op.ceremony_kind;
            ceremonies[idx].threshold = op.threshold;
            ceremonies[idx].total_participants = op.total_participants;
            ceremonies[idx].deadline_ns_lo = op.deadline_ns_lo;
            ceremonies[idx].deadline_ns_hi = op.deadline_ns_hi;
            ceremonies[idx].round = 0u;
            ceremonies[idx].contribution_count = 0u;
            ceremonies[idx].participants_bitmap_lo = 0u;
            ceremonies[idx].participants_bitmap_hi = 0u;
            ceremonies[idx].status = kCeremonyStatusInProgress;
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                ceremonies[idx].subject[k] = op.subject[k];
                ceremonies[idx].ceremony_seed[k] = op.ceremony_seed[k];
            }
            cer_applied = cer_applied + 1u;
        } else if (op.kind == kCeremonyOpCancel) {
            let idx = ceremony_locate_only(op.ceremony_id_lo, op.ceremony_id_hi);
            if (idx == 0xFFFFFFFFu) { continue; }
            if (ceremonies[idx].status != kCeremonyStatusInProgress) { continue; }
            ceremonies[idx].status = kCeremonyStatusFailed;
            cer_applied = cer_applied + 1u;
        }
    }

    // Phase 2: contribution ops.
    let cnt_op_count = desc.contribution_op_count;
    for (var i: u32 = 0u; i < cnt_op_count; i = i + 1u) {
        let op = contribution_ops[i];
        if (op.payload_len > 384u) { continue; }
        let cidx = ceremony_locate_only(op.ceremony_id_lo, op.ceremony_id_hi);
        if (cidx == 0xFFFFFFFFu) { continue; }
        if (ceremonies[cidx].status != kCeremonyStatusInProgress) { continue; }
        if (op.round != ceremonies[cidx].round) { continue; }
        if (op.holder_index >= ceremonies[cidx].total_participants) { continue; }

        let existing = contribution_locate(op.ceremony_id_lo, op.ceremony_id_hi,
                                            op.round, op.holder_index, false);
        if (existing != 0xFFFFFFFFu) { continue; }

        let nidx = contribution_locate(op.ceremony_id_lo, op.ceremony_id_hi,
                                        op.round, op.holder_index, true);
        if (nidx == 0xFFFFFFFFu) { continue; }
        contributions[nidx].contribution_id_lo = next_cont_lo;
        contributions[nidx].contribution_id_hi = next_cont_hi;
        let nc = u64_inc(next_cont_lo, next_cont_hi);
        next_cont_lo = nc.x;
        next_cont_hi = nc.y;
        contributions[nidx].holder_addr_lo = op.holder_addr_lo;
        contributions[nidx].holder_addr_hi = op.holder_addr_hi;
        contributions[nidx].payload_len = op.payload_len;
        // Copy payload — 384 bytes packed in 96 u32 lanes.
        let payload_lanes: u32 = (op.payload_len + 3u) / 4u;
        for (var k: u32 = 0u; k < payload_lanes; k = k + 1u) {
            contributions[nidx].payload[k] = op.payload[k];
        }

        // Update ceremony bitmap.
        let h = op.holder_index;
        if (h < 32u) {
            let bit = 1u << h;
            if ((ceremonies[cidx].participants_bitmap_lo & bit) == 0u) {
                ceremonies[cidx].participants_bitmap_lo = ceremonies[cidx].participants_bitmap_lo | bit;
            }
        } else {
            let bit = 1u << (h - 32u);
            if ((ceremonies[cidx].participants_bitmap_hi & bit) == 0u) {
                ceremonies[cidx].participants_bitmap_hi = ceremonies[cidx].participants_bitmap_hi | bit;
            }
        }
        cnt_applied = cnt_applied + 1u;
    }

    // Phase 3: sweep. For finalized keygen, emit shares in-shader.
    let n_cer = arrayLength(&ceremonies);
    for (var i: u32 = 0u; i < n_cer; i = i + 1u) {
        if (ceremonies[i].status != kCeremonyStatusInProgress) { continue; }

        let in_round = count_contributions_for(ceremonies[i].ceremony_id_lo,
                                                ceremonies[i].ceremony_id_hi,
                                                ceremonies[i].round);
        ceremonies[i].contribution_count = in_round;

        if (in_round >= ceremonies[i].threshold) {
            let kind = ceremonies[i].kind;
            let tr = total_rounds_for(kind);
            ceremonies[i].round = ceremonies[i].round + 1u;
            advances = advances + 1u;
            if (ceremonies[i].round >= tr) {
                ceremonies[i].status = kCeremonyStatusFinalized;
                finalized = finalized + 1u;
                if (is_keygen_kind(kind)) {
                    let scheme = scheme_for_kind(kind);
                    let out_len = share_data_len_for_scheme(scheme);
                    // For each holder that contributed, derive deterministic share.
                    let total_p = ceremonies[i].total_participants;
                    for (var holder: u32 = 0u; holder < total_p; holder = holder + 1u) {
                        var contributed: bool = false;
                        if (holder < 32u) {
                            contributed = (ceremonies[i].participants_bitmap_lo & (1u << holder)) != 0u;
                        } else {
                            contributed = (ceremonies[i].participants_bitmap_hi & (1u << (holder - 32u))) != 0u;
                        }
                        if (!contributed) { continue; }

                        // Build seed buffer:
                        //   ceremony_seed (32) || ceremony_id (8 LE) || holder (4 LE) ||
                        //   "MPCVM-SHARE-V1" (14) || all-round payloads concat
                        var o: u32 = 0u;
                        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                            let v = ceremonies[i].ceremony_seed[k];
                            seed_buf[o + 0u] = v & 0xFFu;
                            seed_buf[o + 1u] = (v >> 8u) & 0xFFu;
                            seed_buf[o + 2u] = (v >> 16u) & 0xFFu;
                            seed_buf[o + 3u] = (v >> 24u) & 0xFFu;
                            o = o + 4u;
                        }
                        write_u64_le(&seed_buf, o, ceremonies[i].ceremony_id_lo,
                                                    ceremonies[i].ceremony_id_hi); o = o + 8u;
                        write_u32_le(&seed_buf, o, holder); o = o + 4u;
                        // tag "MPCVM-SHARE-V1" — 14 ASCII bytes
                        seed_buf[o + 0u]  = 0x4Du; // M
                        seed_buf[o + 1u]  = 0x50u; // P
                        seed_buf[o + 2u]  = 0x43u; // C
                        seed_buf[o + 3u]  = 0x56u; // V
                        seed_buf[o + 4u]  = 0x4Du; // M
                        seed_buf[o + 5u]  = 0x2Du; // -
                        seed_buf[o + 6u]  = 0x53u; // S
                        seed_buf[o + 7u]  = 0x48u; // H
                        seed_buf[o + 8u]  = 0x41u; // A
                        seed_buf[o + 9u]  = 0x52u; // R
                        seed_buf[o + 10u] = 0x45u; // E
                        seed_buf[o + 11u] = 0x2Du; // -
                        seed_buf[o + 12u] = 0x56u; // V
                        seed_buf[o + 13u] = 0x31u; // 1
                        o = o + 14u;
                        // Payloads from each round contribution by this holder.
                        for (var r: u32 = 0u; r < tr; r = r + 1u) {
                            let cidx2 = contribution_locate(ceremonies[i].ceremony_id_lo,
                                                            ceremonies[i].ceremony_id_hi,
                                                            r, holder, false);
                            if (cidx2 == 0xFFFFFFFFu) { continue; }
                            let plen = contributions[cidx2].payload_len;
                            var written: u32 = 0u;
                            // payload has at most 96 lanes (384 bytes) so this is bounded.
                            for (var lane: u32 = 0u; lane < 96u; lane = lane + 1u) {
                                if (written >= plen) { break; }
                                if (o >= 2040u) { break; }
                                let v = contributions[cidx2].payload[lane];
                                let take: u32 = min(min(4u, plen - written), 2040u - o);
                                for (var b: u32 = 0u; b < take; b = b + 1u) {
                                    seed_buf[o + b] = (v >> (b * 8u)) & 0xFFu;
                                }
                                o = o + take;
                                written = written + take;
                            }
                        }

                        // Reserve key share slot.
                        let kidx = key_share_locate_free(ceremonies[i].ceremony_id_lo,
                                                          ceremonies[i].ceremony_id_hi,
                                                          holder);
                        if (kidx == 0xFFFFFFFFu) { continue; }
                        if (key_shares[kidx].occupied == 0u) {
                            key_shares[kidx].share_id_lo = next_share_lo;
                            key_shares[kidx].share_id_hi = next_share_hi;
                            let ns = u64_inc(next_share_lo, next_share_hi);
                            next_share_lo = ns.x;
                            next_share_hi = ns.y;
                            key_shares[kidx].ceremony_id_lo = ceremonies[i].ceremony_id_lo;
                            key_shares[kidx].ceremony_id_hi = ceremonies[i].ceremony_id_hi;
                            key_shares[kidx].holder_addr_lo = 0u;
                            key_shares[kidx].holder_addr_hi = 0u;
                            key_shares[kidx].holder_index = holder;
                            key_shares[kidx].scheme = scheme;
                            key_shares[kidx].occupied = 1u;
                        }
                        key_shares[kidx].share_data_len = out_len;

                        // Stretch keccak: prev = keccak(seed); fill share_data.
                        // out_len <= 320 -> at most 10 stretch iterations.
                        keccak256_buf2048(&seed_buf, o, &prev);
                        var written_sd: u32 = 0u;
                        for (var stretch: u32 = 0u; stretch < 12u; stretch = stretch + 1u) {
                            if (written_sd >= out_len) { break; }
                            let take: u32 = min(32u, out_len - written_sd);
                            // Pack `take` bytes into share_data (u32 lanes).
                            for (var b: u32 = 0u; b < take; b = b + 1u) {
                                let dst_byte = written_sd + b;
                                let dst_lane = dst_byte / 4u;
                                let dst_sh = (dst_byte % 4u) * 8u;
                                // Clear byte in lane, then OR new byte.
                                let mask_u: u32 = ~(0xFFu << dst_sh);
                                key_shares[kidx].share_data[dst_lane] =
                                    (key_shares[kidx].share_data[dst_lane] & mask_u)
                                  | ((prev[b] & 0xFFu) << dst_sh);
                            }
                            written_sd = written_sd + take;
                            if (written_sd < out_len) {
                                for (var k: u32 = 0u; k < 32u; k = k + 1u) {
                                    ext[k] = prev[k];
                                }
                                ext[32] = (written_sd / 32u) & 0xFFu;
                                keccak256_buf33(&ext, &prev);
                            }
                        }
                    }
                }
            } else {
                ceremonies[i].participants_bitmap_lo = 0u;
                ceremonies[i].participants_bitmap_hi = 0u;
                ceremonies[i].contribution_count = 0u;
            }
            continue;
        }

        // Threshold not met — check deadline.
        if (u64_gt(desc.timestamp_ns_lo, desc.timestamp_ns_hi,
                   ceremonies[i].deadline_ns_lo, ceremonies[i].deadline_ns_hi)) {
            ceremonies[i].status = kCeremonyStatusFailed;
            failed = failed + 1u;
        }
    }

    applied_counts[0] = cer_applied;
    applied_counts[1] = cnt_applied;
    applied_counts[2] = advances;
    applied_counts[3] = finalized;
    applied_counts[4] = failed;
}
