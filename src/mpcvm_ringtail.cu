// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_ringtail.cu — CUDA peer of mpcvm_ringtail.metal.

#include "mpcvm_kernels_common.cuh"

namespace mpcvm::cuda {

__device__ inline bool ringtail_kind(uint32_t k) {
    return k == kKindRingtailDkg || k == kKindRingtailSign;
}

extern "C" __global__ void mpcvm_ringtail_step(
    const MPCVMRoundDescriptor* desc,
    const CeremonyOp*           ceremony_ops,
    const ContributionOp*       /*contribution_ops*/,
    Ceremony*                   ceremonies,
    uint32_t*                   applied_out,
    uint32_t                    ceremony_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t applied = 0;
    uint32_t cer_op_count = desc->ceremony_op_count;
    for (uint32_t i = 0; i < cer_op_count; ++i) {
        const CeremonyOp& op = ceremony_ops[i];
        if (!ringtail_kind(op.ceremony_kind)) continue;
        if (op.kind == kCeremonyOpBegin) {
            uint32_t idx = ceremony_locate(ceremonies, ceremony_count, op.ceremony_id, true);
            if (idx == 0xFFFFFFFFu) continue;
            Ceremony& c = ceremonies[idx];
            c.kind = op.ceremony_kind;
            c.threshold = op.threshold;
            c.total_participants = op.total_participants;
            c.deadline_ns = op.deadline_ns;
            c.round = 0;
            c.contribution_count = 0;
            c.participants_bitmap = 0;
            c.status = kCeremonyStatusInProgress;
            for (uint32_t k = 0; k < 32u; ++k) c.subject[k] = op.subject[k];
            for (uint32_t k = 0; k < 32u; ++k) c.ceremony_seed[k] = op.ceremony_seed[k];
            ++applied;
        }
    }
    *applied_out = applied;
}

}  // namespace mpcvm::cuda
