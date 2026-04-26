// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_transition.wgsl — root composition kernel for the wgpu/Dawn backend.
//
// Composes ceremony_root, key_share_root, contribution_root, and
// mpcvm_state_root from the on-device arenas. The keccak-256 routine is
// supplied by the host runtime concatenation (it's the same Keccak-f[1600]
// used by the CPU / Metal / CUDA reference).
//
// In v0.60 the WGSL backend is structurally complete (kernel entry +
// bindings); the in-shader keccak port lands in v0.61 alongside the wgpu
// engine driver. Until then this kernel emits status only and the host
// composes roots from the device arenas via the transparent CPU
// fallback. The Metal and CUDA paths produce the canonical roots.

@group(0) @binding(0) var<storage, read>       desc:           MPCVMRoundDescriptor;
@group(0) @binding(1) var<storage, read_write> ceremonies:     array<Ceremony>;
@group(0) @binding(2) var<storage, read_write> shares:         array<KeyShare>;
@group(0) @binding(3) var<storage, read_write> contributions:  array<Contribution>;
@group(0) @binding(4) var<storage, read_write> state:          MPCVMState;
@group(0) @binding(5) var<storage, read_write> result:         MPCVMTransitionResult;

@compute @workgroup_size(1)
fn mpcvm_transition(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    var active: u32 = 0u;
    var finalized: u32 = 0u;
    var failed: u32 = 0u;
    let n_cer = arrayLength(&ceremonies);
    for (var i: u32 = 0u; i < n_cer; i = i + 1u) {
        let s = ceremonies[i].status;
        if (s == kCeremonyStatusInProgress) { active = active + 1u; }
        if (s == kCeremonyStatusFinalized)  { finalized = finalized + 1u; }
        if (s == kCeremonyStatusFailed)     { failed = failed + 1u; }
    }

    var shares_n: u32 = 0u;
    let n_share = arrayLength(&shares);
    for (var i: u32 = 0u; i < n_share; i = i + 1u) {
        if (shares[i].occupied != 0u) { shares_n = shares_n + 1u; }
    }

    state.active_ceremony_count    = active;
    state.finalized_ceremony_count = finalized;
    state.failed_ceremony_count    = failed;
    state.key_share_count          = shares_n;
    state.now_ns_lo                = desc.timestamp_ns_lo;
    state.now_ns_hi                = desc.timestamp_ns_hi;
    if (desc.closing_flag != 0u) {
        // current_epoch = desc.epoch + 1 (carry into hi if lo overflows)
        let new_lo = desc.epoch_lo + 1u;
        var new_hi = desc.epoch_hi;
        if (new_lo < desc.epoch_lo) { new_hi = new_hi + 1u; }
        state.current_epoch_lo = new_lo;
        state.current_epoch_hi = new_hi;
    }

    result.active_ceremony_count = active;
    result.key_share_count       = shares_n;
    result.epoch_lo              = state.current_epoch_lo;
    result.epoch_hi              = state.current_epoch_hi;
    result.now_ns_lo             = state.now_ns_lo;
    result.now_ns_hi             = state.now_ns_hi;
    result.status                = 1u;
}
