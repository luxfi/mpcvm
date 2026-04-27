// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_transition.wgsl — root composition kernel for the wgpu/Dawn backend.
//
// Composes ceremony_root, key_share_root, contribution_root, and
// mpcvm_state_root from the on-device arenas using the in-shader keccak
// from mpcvm_kernels_common.wgsl. Byte-for-byte equivalent to the Metal
// and CUDA peers.

@group(0) @binding(0) var<storage, read>       desc:           MPCVMRoundDescriptor;
@group(0) @binding(1) var<storage, read_write> ceremonies:     array<Ceremony>;
@group(0) @binding(2) var<storage, read_write> shares:         array<KeyShare>;
@group(0) @binding(3) var<storage, read_write> contributions:  array<Contribution>;
@group(0) @binding(4) var<storage, read_write> state:          MPCVMState;
@group(0) @binding(5) var<storage, read_write> result:         MPCVMTransitionResult;

// Read byte k from a packed array<u32, 8> field (32 bytes).
// In the WGSL Ceremony struct, subject[8]/ceremony_seed[8] are u32 lanes
// holding 4 packed bytes each in little-endian.
fn read_packed_byte(field: ptr<function, array<u32, 8>>, k: u32) -> u32 {
    let lane: u32 = k / 4u;
    let sh: u32   = (k % 4u) * 8u;
    return ((*field)[lane] >> sh) & 0xFFu;
}

fn write_byte_to_buf(buf: ptr<function, array<u32, 2048>>, off: u32, b: u32) {
    (*buf)[off] = b & 0xFFu;
}

@compute @workgroup_size(1)
fn mpcvm_transition(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    var leaf_buf: array<u32, 2048>;
    var fold_buf: array<u32, 64>;
    var leaf_hash: array<u32, 32>;
    var acc: array<u32, 32>;

    // Zero acc.
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc[k] = 0u; }

    // -- ceremony root + counts --
    var n_active: u32 = 0u;
    var finalized: u32 = 0u;
    var failed: u32 = 0u;
    let n_cer = arrayLength(&ceremonies);

    for (var i: u32 = 0u; i < n_cer; i = i + 1u) {
        let st = ceremonies[i].status;
        if (st == kCeremonyStatusFree) { continue; }
        if (st == kCeremonyStatusInProgress) { n_active = n_active + 1u; }
        if (st == kCeremonyStatusFinalized)  { finalized = finalized + 1u; }
        if (st == kCeremonyStatusFailed)     { failed = failed + 1u; }

        // leaf = ceremony_id (8) || started (8) || deadline (8) || bitmap (8) ||
        //        kind (4) || round (4) || threshold (4) || total (4) ||
        //        status (4) || cont_count (4) || subject (32) || seed (32) || index (4)
        var o: u32 = 0u;
        write_u64_le(&leaf_buf, o, ceremonies[i].ceremony_id_lo, ceremonies[i].ceremony_id_hi); o = o + 8u;
        write_u64_le(&leaf_buf, o, ceremonies[i].started_at_ns_lo, ceremonies[i].started_at_ns_hi); o = o + 8u;
        write_u64_le(&leaf_buf, o, ceremonies[i].deadline_ns_lo, ceremonies[i].deadline_ns_hi); o = o + 8u;
        write_u64_le(&leaf_buf, o, ceremonies[i].participants_bitmap_lo, ceremonies[i].participants_bitmap_hi); o = o + 8u;
        write_u32_le(&leaf_buf, o, ceremonies[i].kind);                  o = o + 4u;
        write_u32_le(&leaf_buf, o, ceremonies[i].round);                 o = o + 4u;
        write_u32_le(&leaf_buf, o, ceremonies[i].threshold);             o = o + 4u;
        write_u32_le(&leaf_buf, o, ceremonies[i].total_participants);    o = o + 4u;
        write_u32_le(&leaf_buf, o, ceremonies[i].status);                o = o + 4u;
        write_u32_le(&leaf_buf, o, ceremonies[i].contribution_count);    o = o + 4u;
        // subject (32 bytes packed in 8 u32 lanes)
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let v = ceremonies[i].subject[k];
            leaf_buf[o + 0u] = v & 0xFFu;
            leaf_buf[o + 1u] = (v >> 8u) & 0xFFu;
            leaf_buf[o + 2u] = (v >> 16u) & 0xFFu;
            leaf_buf[o + 3u] = (v >> 24u) & 0xFFu;
            o = o + 4u;
        }
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let v = ceremonies[i].ceremony_seed[k];
            leaf_buf[o + 0u] = v & 0xFFu;
            leaf_buf[o + 1u] = (v >> 8u) & 0xFFu;
            leaf_buf[o + 2u] = (v >> 16u) & 0xFFu;
            leaf_buf[o + 3u] = (v >> 24u) & 0xFFu;
            o = o + 4u;
        }
        write_u32_le(&leaf_buf, o, i); o = o + 4u;

        keccak256_buf2048(&leaf_buf, o, &leaf_hash);
        // fold: acc = keccak(acc || leaf_hash)
        for (var k: u32 = 0u; k < 32u; k = k + 1u) { fold_buf[k]      = acc[k]; }
        for (var k: u32 = 0u; k < 32u; k = k + 1u) { fold_buf[32u + k] = leaf_hash[k]; }
        keccak256_buf64(&fold_buf, &acc);
    }
    // copy ceremony_root acc into state.
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        state.ceremony_root[k] =
            (acc[k * 4u + 0u]      ) |
            (acc[k * 4u + 1u] <<  8u) |
            (acc[k * 4u + 2u] << 16u) |
            (acc[k * 4u + 3u] << 24u);
    }

    // -- key_share root + count --
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc[k] = 0u; }
    var shares_n: u32 = 0u;
    let n_share = arrayLength(&shares);
    for (var i: u32 = 0u; i < n_share; i = i + 1u) {
        if (shares[i].occupied == 0u) { continue; }
        shares_n = shares_n + 1u;

        var o: u32 = 0u;
        write_u64_le(&leaf_buf, o, shares[i].share_id_lo,    shares[i].share_id_hi);    o = o + 8u;
        write_u64_le(&leaf_buf, o, shares[i].ceremony_id_lo, shares[i].ceremony_id_hi); o = o + 8u;
        write_u64_le(&leaf_buf, o, shares[i].holder_addr_lo, shares[i].holder_addr_hi); o = o + 8u;
        write_u32_le(&leaf_buf, o, shares[i].scheme);          o = o + 4u;
        write_u32_le(&leaf_buf, o, shares[i].holder_index);    o = o + 4u;
        let sd_len = shares[i].share_data_len;
        write_u32_le(&leaf_buf, o, sd_len);                    o = o + 4u;
        // share_data: stored as u32 lanes (4 bytes each, 80 lanes = 320 bytes).
        // sd_len <= 320 always (Frost/CGGMP21 = 65, Ringtail = 256).
        var written: u32 = 0u;
        for (var lane: u32 = 0u; lane < 80u; lane = lane + 1u) {
            if (written >= sd_len) { break; }
            let v = shares[i].share_data[lane];
            let take: u32 = min(4u, sd_len - written);
            for (var b: u32 = 0u; b < take; b = b + 1u) {
                leaf_buf[o + b] = (v >> (b * 8u)) & 0xFFu;
            }
            o = o + take;
            written = written + take;
        }
        write_u32_le(&leaf_buf, o, i); o = o + 4u;

        keccak256_buf2048(&leaf_buf, o, &leaf_hash);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) { fold_buf[k]      = acc[k]; }
        for (var k: u32 = 0u; k < 32u; k = k + 1u) { fold_buf[32u + k] = leaf_hash[k]; }
        keccak256_buf64(&fold_buf, &acc);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        state.key_share_root[k] =
            (acc[k * 4u + 0u]      ) |
            (acc[k * 4u + 1u] <<  8u) |
            (acc[k * 4u + 2u] << 16u) |
            (acc[k * 4u + 3u] << 24u);
    }

    // -- contribution root --
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc[k] = 0u; }
    let n_cont = arrayLength(&contributions);
    for (var i: u32 = 0u; i < n_cont; i = i + 1u) {
        if (contributions[i].status != 1u) { continue; }
        var o: u32 = 0u;
        write_u64_le(&leaf_buf, o, contributions[i].contribution_id_lo, contributions[i].contribution_id_hi); o = o + 8u;
        write_u64_le(&leaf_buf, o, contributions[i].ceremony_id_lo,     contributions[i].ceremony_id_hi);     o = o + 8u;
        write_u64_le(&leaf_buf, o, contributions[i].holder_addr_lo,     contributions[i].holder_addr_hi);     o = o + 8u;
        write_u32_le(&leaf_buf, o, contributions[i].round);          o = o + 4u;
        write_u32_le(&leaf_buf, o, contributions[i].holder_index);   o = o + 4u;
        let plen = contributions[i].payload_len;
        write_u32_le(&leaf_buf, o, plen);                            o = o + 4u;
        var written: u32 = 0u;
        for (var lane: u32 = 0u; lane < 96u; lane = lane + 1u) {
            if (written >= plen) { break; }
            let v = contributions[i].payload[lane];
            let take: u32 = min(4u, plen - written);
            for (var b: u32 = 0u; b < take; b = b + 1u) {
                leaf_buf[o + b] = (v >> (b * 8u)) & 0xFFu;
            }
            o = o + take;
            written = written + take;
        }
        write_u32_le(&leaf_buf, o, i); o = o + 4u;

        keccak256_buf2048(&leaf_buf, o, &leaf_hash);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) { fold_buf[k]      = acc[k]; }
        for (var k: u32 = 0u; k < 32u; k = k + 1u) { fold_buf[32u + k] = leaf_hash[k]; }
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
    state.active_ceremony_count    = n_active;
    state.finalized_ceremony_count = finalized;
    state.failed_ceremony_count    = failed;
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
    // composed = parent_state_root (32) || ceremony_root (32) || key_share_root (32) ||
    //            contribution_root (32) || epoch (8) || now (8) || active (4) ||
    //            finalized (4) || failed (4) || share_count (4)
    var o: u32 = 0u;
    // parent_state_root: 32 bytes packed in 8 u32 lanes.
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
