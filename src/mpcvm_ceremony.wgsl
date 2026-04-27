// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_ceremony.wgsl — v0.62 per-slot fan-out kernels for wgpu/Dawn.
//
// Split into two compute entry points (mirror of the Metal split):
//   1. mpcvm_ceremony_apply  — workgroup_size(1), serial Phase 1+2
//   2. mpcvm_ceremony_sweep  — workgroup_size(256), per-slot Phase 3 with
//                              intra-workgroup prefix sum for share_id
//                              allocation (kCeremonySlots = 256).
//
// Determinism: slot ordering = canonical ordering. share_ids assigned
// from a base computed by exclusive prefix sum over emit counts.

// ---- shared bindings (apply) ---------------------------------------------
@group(0) @binding(0) var<storage, read>       desc:                MPCVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>       ceremony_ops:        array<CeremonyOp>;
@group(0) @binding(2) var<storage, read>       contribution_ops:    array<ContributionOp>;
@group(0) @binding(3) var<storage, read_write> ceremonies:          array<Ceremony>;
@group(0) @binding(4) var<storage, read_write> key_shares:          array<KeyShare>;
@group(0) @binding(5) var<storage, read_write> contributions:       array<Contribution>;
@group(0) @binding(6) var<storage, read_write> applied_counts:      array<atomic<u32>, 5>;
@group(0) @binding(7) var<storage, read>       counter_init:        array<u32, 4>;  // [next_cont_lo, next_cont_hi, next_share_lo, next_share_hi]

// FNV hash matching CPU/Metal.
fn ceremony_index_hash(cid_lo: u32, cid_hi: u32, mask: u32) -> u32 {
    var h_lo: u32 = 0x84222325u;
    var h_hi: u32 = 0xcbf29ce4u;
    h_lo = h_lo ^ cid_lo;
    h_hi = h_hi ^ cid_hi;
    let prime_lo: u32 = 0x000001b3u;
    let r0_lo: u32 = h_lo * prime_lo;
    return r0_lo & mask;
}

fn ceremony_locate_insert(cid_lo: u32, cid_hi: u32) -> u32 {
    let n = arrayLength(&ceremonies);
    let mask = n - 1u;
    var idx: u32 = ceremony_index_hash(cid_lo, cid_hi, mask);
    var found: u32 = 0xFFFFFFFFu;
    for (var probe: u32 = 0u; probe < n; probe = probe + 1u) {
        if (ceremonies[idx].status == kCeremonyStatusFree) {
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

fn contribution_index_hash(cid_lo: u32, cid_hi: u32,
                           round: u32, holder: u32, mask: u32) -> u32 {
    var c_lo: u32 = cid_lo ^ holder;
    var c_hi: u32 = cid_hi ^ round;

    let GR_lo: u32 = 0x7F4A7C15u;
    let GR_hi: u32 = 0x9E3779B9u;
    let s6_lo: u32 = cid_lo << 6u;
    let s6_hi: u32 = (cid_hi << 6u) | (cid_lo >> 26u);
    let s2_lo: u32 = (cid_lo >> 2u) | (cid_hi << 30u);
    let s2_hi: u32 = cid_hi >> 2u;

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
    let GR_lo: u32 = 0x7F4A7C15u;
    let GR_hi: u32 = 0x9E3779B9u;
    let h_plus_lo: u32 = holder + GR_lo;
    let h_carry: u32 = select(0u, 1u, h_plus_lo < holder);
    let h_plus_hi: u32 = GR_hi + h_carry;
    let s6_lo: u32 = cid_lo << 6u;
    let s6_hi: u32 = (cid_hi << 6u) | (cid_lo >> 26u);
    let s2_lo: u32 = (cid_lo >> 2u) | (cid_hi << 30u);
    let s2_hi: u32 = cid_hi >> 2u;
    let t1_lo: u32 = h_plus_lo + s6_lo;
    let t1_carry: u32 = select(0u, 1u, t1_lo < h_plus_lo);
    let t1_hi: u32 = h_plus_hi + s6_hi + t1_carry;
    let t2_lo: u32 = t1_lo + s2_lo;
    let t2_carry: u32 = select(0u, 1u, t2_lo < t1_lo);
    let t2_hi: u32 = t1_hi + s2_hi + t2_carry;
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

fn u64_inc(lo: u32, hi: u32) -> vec2<u32> {
    let new_lo = lo + 1u;
    let new_hi = hi + select(0u, 1u, new_lo < lo);
    return vec2<u32>(new_lo, new_hi);
}

fn u64_add(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let new_lo = a_lo + b_lo;
    let carry = select(0u, 1u, new_lo < a_lo);
    let new_hi = a_hi + b_hi + carry;
    return vec2<u32>(new_lo, new_hi);
}

// Emit deterministic key shares for one finalized keygen ceremony given
// its slot index. base_share_id is the share_id assigned to the first
// emitted share; subsequent emitted shares get base+1, base+2, ...
// Uses contribution_locate (hash) for O(1) per (round, holder) lookup.
fn emit_keygen_shares_for(slot: u32, base_lo: u32, base_hi: u32) {
    let kind = ceremonies[slot].kind;
    let scheme = scheme_for_kind(kind);
    let out_len = share_data_len_for_scheme(scheme);
    let total_rounds = total_rounds_for(kind);
    let total_p = ceremonies[slot].total_participants;

    var seed_buf: array<u32, 2048>;
    var prev: array<u32, 32>;
    var ext: array<u32, 33>;

    var next_lo: u32 = base_lo;
    var next_hi: u32 = base_hi;

    for (var holder: u32 = 0u; holder < total_p; holder = holder + 1u) {
        var contributed: bool = false;
        if (holder < 32u) {
            contributed = (ceremonies[slot].participants_bitmap_lo & (1u << holder)) != 0u;
        } else {
            contributed = (ceremonies[slot].participants_bitmap_hi & (1u << (holder - 32u))) != 0u;
        }
        if (!contributed) { continue; }

        // Build seed buffer:
        //   ceremony_seed (32) || ceremony_id (8 LE) || holder (4 LE) ||
        //   "MPCVM-SHARE-V1" (14) || all-round payloads concat
        var o: u32 = 0u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let v = ceremonies[slot].ceremony_seed[k];
            seed_buf[o + 0u] = v & 0xFFu;
            seed_buf[o + 1u] = (v >> 8u) & 0xFFu;
            seed_buf[o + 2u] = (v >> 16u) & 0xFFu;
            seed_buf[o + 3u] = (v >> 24u) & 0xFFu;
            o = o + 4u;
        }
        write_u64_le(&seed_buf, o, ceremonies[slot].ceremony_id_lo,
                                    ceremonies[slot].ceremony_id_hi); o = o + 8u;
        write_u32_le(&seed_buf, o, holder); o = o + 4u;
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
        for (var r: u32 = 0u; r < total_rounds; r = r + 1u) {
            let cidx2 = contribution_locate(ceremonies[slot].ceremony_id_lo,
                                            ceremonies[slot].ceremony_id_hi,
                                            r, holder, false);
            if (cidx2 == 0xFFFFFFFFu) { continue; }
            let plen = contributions[cidx2].payload_len;
            var written: u32 = 0u;
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

        let kidx = key_share_locate_free(ceremonies[slot].ceremony_id_lo,
                                          ceremonies[slot].ceremony_id_hi,
                                          holder);
        if (kidx == 0xFFFFFFFFu) { continue; }
        if (key_shares[kidx].occupied == 0u) {
            key_shares[kidx].share_id_lo = next_lo;
            key_shares[kidx].share_id_hi = next_hi;
            let nxt = u64_inc(next_lo, next_hi);
            next_lo = nxt.x;
            next_hi = nxt.y;
            key_shares[kidx].ceremony_id_lo = ceremonies[slot].ceremony_id_lo;
            key_shares[kidx].ceremony_id_hi = ceremonies[slot].ceremony_id_hi;
            key_shares[kidx].holder_addr_lo = 0u;
            key_shares[kidx].holder_addr_hi = 0u;
            key_shares[kidx].holder_index = holder;
            key_shares[kidx].scheme = scheme;
            key_shares[kidx].occupied = 1u;
        }
        key_shares[kidx].share_data_len = out_len;

        keccak256_buf2048(&seed_buf, o, &prev);
        var written_sd: u32 = 0u;
        for (var stretch: u32 = 0u; stretch < 12u; stretch = stretch + 1u) {
            if (written_sd >= out_len) { break; }
            let take: u32 = min(32u, out_len - written_sd);
            for (var b: u32 = 0u; b < take; b = b + 1u) {
                let dst_byte = written_sd + b;
                let dst_lane = dst_byte / 4u;
                let dst_sh = (dst_byte % 4u) * 8u;
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

// =============================================================================
// Phase 1+2: serial ops apply.
// =============================================================================

@compute @workgroup_size(1)
fn mpcvm_ceremony_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    var cer_applied: u32 = 0u;
    var cnt_applied: u32 = 0u;

    var next_cont_lo: u32 = counter_init[0];
    var next_cont_hi: u32 = counter_init[1];

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
        let payload_lanes: u32 = (op.payload_len + 3u) / 4u;
        for (var k: u32 = 0u; k < payload_lanes; k = k + 1u) {
            contributions[nidx].payload[k] = op.payload[k];
        }

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

    atomicStore(&applied_counts[0], cer_applied);
    atomicStore(&applied_counts[1], cnt_applied);
    // applied_counts[2..4] are zero-initialized by host; sweep kernel atomically
    // increments them.
}

// =============================================================================
// Phase 3: per-slot fan-out sweep with intra-workgroup prefix sum.
//
// workgroup_size(256) — must equal kCeremonySlots in mpcvm_gpu_layout.hpp.
// =============================================================================

const kSweepWorkgroupSize: u32 = 256u;

var<workgroup> emit_counts_tg: array<u32, 256>;
// Per-slot decision bookkeeping shared across the workgroup. We use storage
// in workgroup memory only for emit_counts (small N=256). Per-thread state
// stays in registers.

@compute @workgroup_size(256)
fn mpcvm_ceremony_sweep(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n_cer = arrayLength(&ceremonies);
    let in_range = tid < n_cer;

    // Phase A: per-slot decision + emit_count.
    var local_emit_count: u32 = 0u;
    var will_finalize_keygen: bool = false;
    var will_advance: bool = false;
    var will_timeout: bool = false;
    var in_round_total: u32 = 0u;
    var slot_kind: u32 = 0u;
    if (in_range) {
        if (ceremonies[tid].status == kCeremonyStatusInProgress) {
            in_round_total = count_contributions_for(ceremonies[tid].ceremony_id_lo,
                                                     ceremonies[tid].ceremony_id_hi,
                                                     ceremonies[tid].round);
            if (in_round_total >= ceremonies[tid].threshold) {
                will_advance = true;
                slot_kind = ceremonies[tid].kind;
                let tr = total_rounds_for(slot_kind);
                if (ceremonies[tid].round + 1u >= tr && is_keygen_kind(slot_kind)) {
                    will_finalize_keygen = true;
                    let total_p = ceremonies[tid].total_participants;
                    for (var h: u32 = 0u; h < total_p; h = h + 1u) {
                        var on: bool = false;
                        if (h < 32u) {
                            on = (ceremonies[tid].participants_bitmap_lo & (1u << h)) != 0u;
                        } else {
                            on = (ceremonies[tid].participants_bitmap_hi & (1u << (h - 32u))) != 0u;
                        }
                        if (on) { local_emit_count = local_emit_count + 1u; }
                    }
                }
            } else if (u64_gt(desc.timestamp_ns_lo, desc.timestamp_ns_hi,
                              ceremonies[tid].deadline_ns_lo, ceremonies[tid].deadline_ns_hi)) {
                will_timeout = true;
            }
        }
    }
    emit_counts_tg[tid] = local_emit_count;

    workgroupBarrier();

    // Phase B: prefix sum on tid 0 (small N, single thread serial scan).
    if (tid == 0u) {
        var acc: u32 = 0u;
        for (var i: u32 = 0u; i < kSweepWorkgroupSize; i = i + 1u) {
            let v = emit_counts_tg[i];
            emit_counts_tg[i] = acc;
            acc = acc + v;
        }
    }

    workgroupBarrier();

    // Phase C: actually advance/finalize/timeout.
    if (in_range) {
        if (ceremonies[tid].status == kCeremonyStatusInProgress) {
            ceremonies[tid].contribution_count = in_round_total;
            if (will_advance) {
                ceremonies[tid].round = ceremonies[tid].round + 1u;
                atomicAdd(&applied_counts[2], 1u);
                let tr = total_rounds_for(slot_kind);
                if (ceremonies[tid].round >= tr) {
                    ceremonies[tid].status = kCeremonyStatusFinalized;
                    atomicAdd(&applied_counts[3], 1u);
                    if (will_finalize_keygen) {
                        let base = u64_add(counter_init[2], counter_init[3],
                                           emit_counts_tg[tid], 0u);
                        emit_keygen_shares_for(tid, base.x, base.y);
                    }
                } else {
                    ceremonies[tid].participants_bitmap_lo = 0u;
                    ceremonies[tid].participants_bitmap_hi = 0u;
                    ceremonies[tid].contribution_count = 0u;
                }
            } else if (will_timeout) {
                ceremonies[tid].status = kCeremonyStatusFailed;
                atomicAdd(&applied_counts[4], 1u);
            }
        }
    }
}

