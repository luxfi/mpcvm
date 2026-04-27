// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_transition.wgsl — v0.62 parallel leaf hashing + serial fold (WGSL).
//
// Two compute entry points (mirror of Metal split):
//   1. mpcvm_compute_leaves  — workgroup_size(64), per-slot keccak.
//      Each thread hashes one occupied leaf into the leaf_hashes array.
//   2. mpcvm_compose_root    — workgroup_size(1), serial fold + state-root
//      composition.
//
// Determinism: leaves computed independently; fold consumed in canonical
// slot order. Byte-equal to v0.61.1 single-thread implementation.

@group(0) @binding(0) var<storage, read>       desc:           MPCVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>       ceremonies:     array<Ceremony>;
@group(0) @binding(2) var<storage, read>       shares:         array<KeyShare>;
@group(0) @binding(3) var<storage, read>       contributions:  array<Contribution>;
@group(0) @binding(4) var<storage, read_write> ceremony_leaves:    array<u32>;
@group(0) @binding(5) var<storage, read_write> share_leaves:       array<u32>;
@group(0) @binding(6) var<storage, read_write> contribution_leaves: array<u32>;
@group(0) @binding(7) var<storage, read_write> ceremony_used_mask: array<u32>;
@group(0) @binding(8) var<storage, read_write> share_used_mask:    array<u32>;
@group(0) @binding(9) var<storage, read_write> contribution_used_mask: array<u32>;
@group(0) @binding(10) var<storage, read_write> count_outs: array<atomic<u32>, 4>;  // active, finalized, failed, share_count
@group(0) @binding(11) var<storage, read_write> state:          MPCVMState;
@group(0) @binding(12) var<storage, read_write> result:         MPCVMTransitionResult;

// Pack 32 bytes (held in 32 u32 lane bytes) into 8 u32 lanes.
fn pack_hash(src: ptr<function, array<u32, 32>>, lane_idx: u32) -> u32 {
    return ((*src)[lane_idx * 4u + 0u]      )
         | ((*src)[lane_idx * 4u + 1u] <<  8u)
         | ((*src)[lane_idx * 4u + 2u] << 16u)
         | ((*src)[lane_idx * 4u + 3u] << 24u);
}

// =============================================================================
// Pass 1: parallel leaf hash computation.
// =============================================================================

@compute @workgroup_size(64)
fn mpcvm_compute_leaves(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let n_cer = arrayLength(&ceremonies);
    let n_share = arrayLength(&shares);
    let n_cont = arrayLength(&contributions);

    var leaf_buf: array<u32, 2048>;
    var leaf_hash: array<u32, 32>;

    // -- ceremony leaf --
    if (tid < n_cer) {
        let st = ceremonies[tid].status;
        if (st == kCeremonyStatusFree) {
            ceremony_used_mask[tid] = 0u;
        } else {
            ceremony_used_mask[tid] = 1u;
            if (st == kCeremonyStatusInProgress) { atomicAdd(&count_outs[0], 1u); }
            if (st == kCeremonyStatusFinalized)  { atomicAdd(&count_outs[1], 1u); }
            if (st == kCeremonyStatusFailed)     { atomicAdd(&count_outs[2], 1u); }

            var o: u32 = 0u;
            write_u64_le(&leaf_buf, o, ceremonies[tid].ceremony_id_lo, ceremonies[tid].ceremony_id_hi); o = o + 8u;
            write_u64_le(&leaf_buf, o, ceremonies[tid].started_at_ns_lo, ceremonies[tid].started_at_ns_hi); o = o + 8u;
            write_u64_le(&leaf_buf, o, ceremonies[tid].deadline_ns_lo, ceremonies[tid].deadline_ns_hi); o = o + 8u;
            write_u64_le(&leaf_buf, o, ceremonies[tid].participants_bitmap_lo, ceremonies[tid].participants_bitmap_hi); o = o + 8u;
            write_u32_le(&leaf_buf, o, ceremonies[tid].kind);                  o = o + 4u;
            write_u32_le(&leaf_buf, o, ceremonies[tid].round);                 o = o + 4u;
            write_u32_le(&leaf_buf, o, ceremonies[tid].threshold);             o = o + 4u;
            write_u32_le(&leaf_buf, o, ceremonies[tid].total_participants);    o = o + 4u;
            write_u32_le(&leaf_buf, o, ceremonies[tid].status);                o = o + 4u;
            write_u32_le(&leaf_buf, o, ceremonies[tid].contribution_count);    o = o + 4u;
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                let v = ceremonies[tid].subject[k];
                leaf_buf[o + 0u] = v & 0xFFu;
                leaf_buf[o + 1u] = (v >> 8u) & 0xFFu;
                leaf_buf[o + 2u] = (v >> 16u) & 0xFFu;
                leaf_buf[o + 3u] = (v >> 24u) & 0xFFu;
                o = o + 4u;
            }
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                let v = ceremonies[tid].ceremony_seed[k];
                leaf_buf[o + 0u] = v & 0xFFu;
                leaf_buf[o + 1u] = (v >> 8u) & 0xFFu;
                leaf_buf[o + 2u] = (v >> 16u) & 0xFFu;
                leaf_buf[o + 3u] = (v >> 24u) & 0xFFu;
                o = o + 4u;
            }
            write_u32_le(&leaf_buf, o, tid); o = o + 4u;

            keccak256_buf2048(&leaf_buf, o, &leaf_hash);
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                ceremony_leaves[tid * 8u + k] = pack_hash(&leaf_hash, k);
            }
        }
    }

    // -- key share leaf --
    if (tid < n_share) {
        if (shares[tid].occupied == 0u) {
            share_used_mask[tid] = 0u;
        } else {
            share_used_mask[tid] = 1u;
            atomicAdd(&count_outs[3], 1u);

            var o: u32 = 0u;
            write_u64_le(&leaf_buf, o, shares[tid].share_id_lo,    shares[tid].share_id_hi);    o = o + 8u;
            write_u64_le(&leaf_buf, o, shares[tid].ceremony_id_lo, shares[tid].ceremony_id_hi); o = o + 8u;
            write_u64_le(&leaf_buf, o, shares[tid].holder_addr_lo, shares[tid].holder_addr_hi); o = o + 8u;
            write_u32_le(&leaf_buf, o, shares[tid].scheme);          o = o + 4u;
            write_u32_le(&leaf_buf, o, shares[tid].holder_index);    o = o + 4u;
            let sd_len = shares[tid].share_data_len;
            write_u32_le(&leaf_buf, o, sd_len);                    o = o + 4u;
            var written: u32 = 0u;
            for (var lane: u32 = 0u; lane < 80u; lane = lane + 1u) {
                if (written >= sd_len) { break; }
                let v = shares[tid].share_data[lane];
                let take: u32 = min(4u, sd_len - written);
                for (var b: u32 = 0u; b < take; b = b + 1u) {
                    leaf_buf[o + b] = (v >> (b * 8u)) & 0xFFu;
                }
                o = o + take;
                written = written + take;
            }
            write_u32_le(&leaf_buf, o, tid); o = o + 4u;

            keccak256_buf2048(&leaf_buf, o, &leaf_hash);
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                share_leaves[tid * 8u + k] = pack_hash(&leaf_hash, k);
            }
        }
    }

    // -- contribution leaf --
    if (tid < n_cont) {
        if (contributions[tid].status != 1u) {
            contribution_used_mask[tid] = 0u;
        } else {
            contribution_used_mask[tid] = 1u;

            var o: u32 = 0u;
            write_u64_le(&leaf_buf, o, contributions[tid].contribution_id_lo, contributions[tid].contribution_id_hi); o = o + 8u;
            write_u64_le(&leaf_buf, o, contributions[tid].ceremony_id_lo,     contributions[tid].ceremony_id_hi);     o = o + 8u;
            write_u64_le(&leaf_buf, o, contributions[tid].holder_addr_lo,     contributions[tid].holder_addr_hi);     o = o + 8u;
            write_u32_le(&leaf_buf, o, contributions[tid].round);          o = o + 4u;
            write_u32_le(&leaf_buf, o, contributions[tid].holder_index);   o = o + 4u;
            let plen = contributions[tid].payload_len;
            write_u32_le(&leaf_buf, o, plen);                            o = o + 4u;
            var written: u32 = 0u;
            for (var lane: u32 = 0u; lane < 96u; lane = lane + 1u) {
                if (written >= plen) { break; }
                let v = contributions[tid].payload[lane];
                let take: u32 = min(4u, plen - written);
                for (var b: u32 = 0u; b < take; b = b + 1u) {
                    leaf_buf[o + b] = (v >> (b * 8u)) & 0xFFu;
                }
                o = o + take;
                written = written + take;
            }
            write_u32_le(&leaf_buf, o, tid); o = o + 4u;

            keccak256_buf2048(&leaf_buf, o, &leaf_hash);
            for (var k: u32 = 0u; k < 8u; k = k + 1u) {
                contribution_leaves[tid * 8u + k] = pack_hash(&leaf_hash, k);
            }
        }
    }
}

// =============================================================================
// Pass 2: serial fold + state-root composition (workgroup_size(1)).
// =============================================================================

@compute @workgroup_size(1)
fn mpcvm_compose_root(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    var fold_buf: array<u32, 64>;
    var leaf_buf: array<u32, 2048>;
    var leaf_hash: array<u32, 32>;
    var acc: array<u32, 32>;

    let n_cer = arrayLength(&ceremonies);
    let n_share = arrayLength(&shares);
    let n_cont = arrayLength(&contributions);

    // -- ceremony root fold --
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc[k] = 0u; }
    for (var i: u32 = 0u; i < n_cer; i = i + 1u) {
        if (ceremony_used_mask[i] == 0u) { continue; }
        for (var k: u32 = 0u; k < 32u; k = k + 1u) { fold_buf[k] = acc[k]; }
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let v = ceremony_leaves[i * 8u + k];
            fold_buf[32u + k * 4u + 0u] = v & 0xFFu;
            fold_buf[32u + k * 4u + 1u] = (v >> 8u) & 0xFFu;
            fold_buf[32u + k * 4u + 2u] = (v >> 16u) & 0xFFu;
            fold_buf[32u + k * 4u + 3u] = (v >> 24u) & 0xFFu;
        }
        keccak256_buf64(&fold_buf, &acc);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        state.ceremony_root[k] =
            (acc[k * 4u + 0u]      ) |
            (acc[k * 4u + 1u] <<  8u) |
            (acc[k * 4u + 2u] << 16u) |
            (acc[k * 4u + 3u] << 24u);
    }

    // -- key share root fold --
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc[k] = 0u; }
    for (var i: u32 = 0u; i < n_share; i = i + 1u) {
        if (share_used_mask[i] == 0u) { continue; }
        for (var k: u32 = 0u; k < 32u; k = k + 1u) { fold_buf[k] = acc[k]; }
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let v = share_leaves[i * 8u + k];
            fold_buf[32u + k * 4u + 0u] = v & 0xFFu;
            fold_buf[32u + k * 4u + 1u] = (v >> 8u) & 0xFFu;
            fold_buf[32u + k * 4u + 2u] = (v >> 16u) & 0xFFu;
            fold_buf[32u + k * 4u + 3u] = (v >> 24u) & 0xFFu;
        }
        keccak256_buf64(&fold_buf, &acc);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        state.key_share_root[k] =
            (acc[k * 4u + 0u]      ) |
            (acc[k * 4u + 1u] <<  8u) |
            (acc[k * 4u + 2u] << 16u) |
            (acc[k * 4u + 3u] << 24u);
    }

    // -- contribution root fold --
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc[k] = 0u; }
    for (var i: u32 = 0u; i < n_cont; i = i + 1u) {
        if (contribution_used_mask[i] == 0u) { continue; }
        for (var k: u32 = 0u; k < 32u; k = k + 1u) { fold_buf[k] = acc[k]; }
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let v = contribution_leaves[i * 8u + k];
            fold_buf[32u + k * 4u + 0u] = v & 0xFFu;
            fold_buf[32u + k * 4u + 1u] = (v >> 8u) & 0xFFu;
            fold_buf[32u + k * 4u + 2u] = (v >> 16u) & 0xFFu;
            fold_buf[32u + k * 4u + 3u] = (v >> 24u) & 0xFFu;
        }
        keccak256_buf64(&fold_buf, &acc);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        state.contribution_root[k] =
            (acc[k * 4u + 0u]      ) |
            (acc[k * 4u + 1u] <<  8u) |
            (acc[k * 4u + 2u] << 16u) |
            (acc[k * 4u + 3u] << 24u);
    }

    // -- counts and now/epoch --
    let n_active  = atomicLoad(&count_outs[0]);
    let n_final   = atomicLoad(&count_outs[1]);
    let n_failed  = atomicLoad(&count_outs[2]);
    let shares_n  = atomicLoad(&count_outs[3]);
    state.active_ceremony_count    = n_active;
    state.finalized_ceremony_count = n_final;
    state.failed_ceremony_count    = n_failed;
    state.key_share_count          = shares_n;
    state.now_ns_lo                = desc.timestamp_ns_lo;
    state.now_ns_hi                = desc.timestamp_ns_hi;
    if (desc.closing_flag != 0u) {
        let new_lo = desc.epoch_lo + 1u;
        var new_hi = desc.epoch_hi;
        if (new_lo < desc.epoch_lo) { new_hi = new_hi + 1u; }
        state.current_epoch_lo = new_lo;
        state.current_epoch_hi = new_hi;
    }

    // -- composed mpcvm_state_root --
    var o: u32 = 0u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let v = desc.parent_state_root[k];
        leaf_buf[o + 0u] = v & 0xFFu;
        leaf_buf[o + 1u] = (v >> 8u) & 0xFFu;
        leaf_buf[o + 2u] = (v >> 16u) & 0xFFu;
        leaf_buf[o + 3u] = (v >> 24u) & 0xFFu;
        o = o + 4u;
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let v = state.ceremony_root[k];
        leaf_buf[o + 0u] = v & 0xFFu;
        leaf_buf[o + 1u] = (v >> 8u) & 0xFFu;
        leaf_buf[o + 2u] = (v >> 16u) & 0xFFu;
        leaf_buf[o + 3u] = (v >> 24u) & 0xFFu;
        o = o + 4u;
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let v = state.key_share_root[k];
        leaf_buf[o + 0u] = v & 0xFFu;
        leaf_buf[o + 1u] = (v >> 8u) & 0xFFu;
        leaf_buf[o + 2u] = (v >> 16u) & 0xFFu;
        leaf_buf[o + 3u] = (v >> 24u) & 0xFFu;
        o = o + 4u;
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let v = state.contribution_root[k];
        leaf_buf[o + 0u] = v & 0xFFu;
        leaf_buf[o + 1u] = (v >> 8u) & 0xFFu;
        leaf_buf[o + 2u] = (v >> 16u) & 0xFFu;
        leaf_buf[o + 3u] = (v >> 24u) & 0xFFu;
        o = o + 4u;
    }
    write_u64_le(&leaf_buf, o, state.current_epoch_lo, state.current_epoch_hi); o = o + 8u;
    write_u64_le(&leaf_buf, o, state.now_ns_lo, state.now_ns_hi);               o = o + 8u;
    write_u32_le(&leaf_buf, o, state.active_ceremony_count);                    o = o + 4u;
    write_u32_le(&leaf_buf, o, state.finalized_ceremony_count);                 o = o + 4u;
    write_u32_le(&leaf_buf, o, state.failed_ceremony_count);                    o = o + 4u;
    write_u32_le(&leaf_buf, o, state.key_share_count);                          o = o + 4u;
    keccak256_buf2048(&leaf_buf, o, &leaf_hash);
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        state.mpcvm_state_root[k] =
            (leaf_hash[k * 4u + 0u]      ) |
            (leaf_hash[k * 4u + 1u] <<  8u) |
            (leaf_hash[k * 4u + 2u] << 16u) |
            (leaf_hash[k * 4u + 3u] << 24u);
    }

    // -- write result --
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        result.ceremony_root[k]     = state.ceremony_root[k];
        result.key_share_root[k]    = state.key_share_root[k];
        result.contribution_root[k] = state.contribution_root[k];
        result.mpcvm_state_root[k]  = state.mpcvm_state_root[k];
    }
    result.active_ceremony_count = n_active;
    result.key_share_count       = shares_n;
    result.epoch_lo              = state.current_epoch_lo;
    result.epoch_hi              = state.current_epoch_hi;
    result.now_ns_lo             = state.now_ns_lo;
    result.now_ns_hi             = state.now_ns_hi;
    result.status                = 1u;
}
