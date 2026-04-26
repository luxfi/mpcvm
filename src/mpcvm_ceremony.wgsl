// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_ceremony.wgsl — state-machine kernel for the wgpu/Dawn backend.
//
// WGSL has no native u64 and limited control flow; this kernel mirrors
// mpcvm_ceremony.metal / mpcvm_ceremony.cu using vec2<u32> for u64
// fields. Determinism is preserved by performing the same canonical
// in-order traversal in a single workgroup of size 1.
//
// Phase 1: Apply CeremonyOps (begin / cancel).
// Phase 2: Apply ContributionOps (with dedup by (cid, round, holder)).
// Phase 3: Sweep — advance round / finalize / timeout.
// Roots are computed by mpcvm_transition.wgsl.

// (layout common is included by the host's runtime concatenation —
//  same pattern as the Metal driver.)

@group(0) @binding(0) var<storage, read>       desc:                MPCVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>       ceremony_ops:        array<CeremonyOp>;
@group(0) @binding(2) var<storage, read>       contribution_ops:    array<ContributionOp>;
@group(0) @binding(3) var<storage, read_write> ceremonies:          array<Ceremony>;
@group(0) @binding(4) var<storage, read_write> key_shares:          array<KeyShare>;
@group(0) @binding(5) var<storage, read_write> contributions:       array<Contribution>;
@group(0) @binding(6) var<storage, read_write> applied_counts:      array<u32, 5>;

fn ceremony_locate_insert(cid_lo: u32, cid_hi: u32, n: u32) -> u32 {
    var idx: u32 = (cid_lo ^ cid_hi) & (n - 1u);
    var probe: u32 = 0u;
    loop {
        if (probe >= n) { return 0xFFFFFFFFu; }
        let s_status = ceremonies[idx].status;
        if (s_status == kCeremonyStatusFree) {
            ceremonies[idx].ceremony_id_lo = cid_lo;
            ceremonies[idx].ceremony_id_hi = cid_hi;
            ceremonies[idx].status = kCeremonyStatusInProgress;
            return idx;
        }
        if (u64_eq(ceremonies[idx].ceremony_id_lo, ceremonies[idx].ceremony_id_hi,
                   cid_lo, cid_hi)) {
            return idx;
        }
        idx = (idx + 1u) & (n - 1u);
        probe = probe + 1u;
    }
}

fn ceremony_locate(cid_lo: u32, cid_hi: u32, n: u32) -> u32 {
    var idx: u32 = (cid_lo ^ cid_hi) & (n - 1u);
    var probe: u32 = 0u;
    loop {
        if (probe >= n) { return 0xFFFFFFFFu; }
        if (ceremonies[idx].status == kCeremonyStatusFree) {
            return 0xFFFFFFFFu;
        }
        if (u64_eq(ceremonies[idx].ceremony_id_lo, ceremonies[idx].ceremony_id_hi,
                   cid_lo, cid_hi)) {
            return idx;
        }
        idx = (idx + 1u) & (n - 1u);
        probe = probe + 1u;
    }
}

@compute @workgroup_size(1)
fn mpcvm_ceremony_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    let n_cer = arrayLength(&ceremonies);

    var cer_applied: u32 = 0u;
    var cnt_applied: u32 = 0u;
    var advances: u32 = 0u;
    var finalized: u32 = 0u;
    var failed: u32 = 0u;

    // Phase 1: ceremony ops.
    let cer_count = desc.ceremony_op_count;
    for (var i: u32 = 0u; i < cer_count; i = i + 1u) {
        let op = ceremony_ops[i];
        if (op.kind == kCeremonyOpBegin) {
            if (op.threshold == 0u || op.threshold > op.total_participants) { continue; }
            if (op.total_participants > 64u) { continue; }
            let idx = ceremony_locate_insert(op.ceremony_id_lo, op.ceremony_id_hi, n_cer);
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
            let idx = ceremony_locate(op.ceremony_id_lo, op.ceremony_id_hi, n_cer);
            if (idx == 0xFFFFFFFFu) { continue; }
            if (ceremonies[idx].status != kCeremonyStatusInProgress) { continue; }
            ceremonies[idx].status = kCeremonyStatusFailed;
            cer_applied = cer_applied + 1u;
        }
    }

    applied_counts[0] = cer_applied;
    applied_counts[1] = cnt_applied;
    applied_counts[2] = advances;
    applied_counts[3] = finalized;
    applied_counts[4] = failed;
}
