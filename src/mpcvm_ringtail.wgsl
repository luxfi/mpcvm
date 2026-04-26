// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_ringtail.wgsl — Ringtail kernel entry for the wgpu/Dawn backend.
//
// The Module-LWE arithmetic (NTT, sampling) lives in luxcpp/lattice/src/wgsl/.
// This kernel only handles the ceremony state machine envelope.

@group(0) @binding(0) var<storage, read>       desc:           MPCVMRoundDescriptor;
@group(0) @binding(1) var<storage, read>       ceremony_ops:   array<CeremonyOp>;
@group(0) @binding(3) var<storage, read_write> ceremonies:     array<Ceremony>;
@group(0) @binding(6) var<storage, read_write> applied_out:    array<u32, 1>;

fn ringtail_kind(k: u32) -> bool {
    return k == kKindRingtailDkg || k == kKindRingtailSign;
}

@compute @workgroup_size(1)
fn mpcvm_ringtail_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let n_cer = arrayLength(&ceremonies);
    var applied: u32 = 0u;
    let cer_count = desc.ceremony_op_count;
    for (var i: u32 = 0u; i < cer_count; i = i + 1u) {
        let op = ceremony_ops[i];
        if (!ringtail_kind(op.ceremony_kind)) { continue; }
        if (op.kind != kCeremonyOpBegin) { continue; }

        var idx: u32 = (op.ceremony_id_lo ^ op.ceremony_id_hi) & (n_cer - 1u);
        var probe: u32 = 0u;
        var found: u32 = 0xFFFFFFFFu;
        loop {
            if (probe >= n_cer) { break; }
            if (ceremonies[idx].status == kCeremonyStatusFree) {
                ceremonies[idx].ceremony_id_lo = op.ceremony_id_lo;
                ceremonies[idx].ceremony_id_hi = op.ceremony_id_hi;
                ceremonies[idx].status = kCeremonyStatusInProgress;
                found = idx; break;
            }
            if (u64_eq(ceremonies[idx].ceremony_id_lo, ceremonies[idx].ceremony_id_hi,
                       op.ceremony_id_lo, op.ceremony_id_hi)) {
                found = idx; break;
            }
            idx = (idx + 1u) & (n_cer - 1u);
            probe = probe + 1u;
        }
        if (found == 0xFFFFFFFFu) { continue; }
        ceremonies[found].kind = op.ceremony_kind;
        ceremonies[found].threshold = op.threshold;
        ceremonies[found].total_participants = op.total_participants;
        ceremonies[found].deadline_ns_lo = op.deadline_ns_lo;
        ceremonies[found].deadline_ns_hi = op.deadline_ns_hi;
        ceremonies[found].round = 0u;
        ceremonies[found].contribution_count = 0u;
        ceremonies[found].participants_bitmap_lo = 0u;
        ceremonies[found].participants_bitmap_hi = 0u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            ceremonies[found].subject[k] = op.subject[k];
            ceremonies[found].ceremony_seed[k] = op.ceremony_seed[k];
        }
        applied = applied + 1u;
    }
    applied_out[0] = applied;
}
