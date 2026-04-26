// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file mpcvm_layout_test.cpp
/// MPCVM v0.60 — layout invariants for cross-backend determinism.
///
/// Validates struct sizes, alignment, member offsets, and basic CPU-reference
/// determinism on a small canonical workload (FROST 7-of-10 keygen, replay
/// attempt, timeout). Anything that drifts in the host header without a
/// parallel update in the GPU kernels would break CPU/Metal/CUDA/WGSL
/// equivalence — these checks are the first line of defense.

#include "lux/mpcvm/mpcvm_gpu_layout.hpp"
#include "lux/mpcvm/mpcvm_cpu_reference.hpp"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace mpcvm::gpu;

namespace {

int g_passed = 0;
int g_failed = 0;

#define EXPECT(name, cond)                                                  \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("  FAIL[%s]: %s\n", (name), #cond);                 \
            std::fflush(stdout);                                            \
            ++g_failed;                                                     \
            return;                                                         \
        }                                                                   \
    } while (0)

#define PASS(name)                                                          \
    do {                                                                    \
        std::printf("  ok  : %s\n", (name));                                \
        std::fflush(stdout);                                                \
        ++g_passed;                                                         \
    } while (0)

void test_ceremony_layout()
{
    EXPECT("Ceremony.size",  sizeof(Ceremony) == 128);
    EXPECT("Ceremony.align", alignof(Ceremony) == 16);
    EXPECT("Ceremony.id.off",       offsetof(Ceremony, ceremony_id)        == 0);
    EXPECT("Ceremony.started.off",  offsetof(Ceremony, started_at_ns)      == 8);
    EXPECT("Ceremony.deadline.off", offsetof(Ceremony, deadline_ns)        == 16);
    EXPECT("Ceremony.bitmap.off",   offsetof(Ceremony, participants_bitmap) == 24);
    EXPECT("Ceremony.kind.off",     offsetof(Ceremony, kind)               == 32);
    EXPECT("Ceremony.round.off",    offsetof(Ceremony, round)              == 36);
    EXPECT("Ceremony.thresh.off",   offsetof(Ceremony, threshold)          == 40);
    EXPECT("Ceremony.total.off",    offsetof(Ceremony, total_participants) == 44);
    EXPECT("Ceremony.status.off",   offsetof(Ceremony, status)             == 48);
    EXPECT("Ceremony.contcnt.off",  offsetof(Ceremony, contribution_count) == 52);
    EXPECT("Ceremony.subj.off",     offsetof(Ceremony, subject)            == 56);
    EXPECT("Ceremony.seed.off",     offsetof(Ceremony, ceremony_seed)      == 88);
    PASS("Ceremony layout");
}

void test_key_share_layout()
{
    EXPECT("KeyShare.size",  sizeof(KeyShare) == 368);
    EXPECT("KeyShare.align", alignof(KeyShare) == 16);
    EXPECT("KeyShare.id.off",        offsetof(KeyShare, share_id)        == 0);
    EXPECT("KeyShare.cer.off",       offsetof(KeyShare, ceremony_id)     == 8);
    EXPECT("KeyShare.holder.off",    offsetof(KeyShare, holder_addr)     == 16);
    EXPECT("KeyShare.scheme.off",    offsetof(KeyShare, scheme)          == 24);
    EXPECT("KeyShare.hidx.off",      offsetof(KeyShare, holder_index)    == 28);
    EXPECT("KeyShare.dlen.off",      offsetof(KeyShare, share_data_len)  == 32);
    EXPECT("KeyShare.occ.off",       offsetof(KeyShare, occupied)        == 36);
    EXPECT("KeyShare.data.off",      offsetof(KeyShare, share_data)      == 40);
    PASS("KeyShare layout");
}

void test_contribution_layout()
{
    EXPECT("Contribution.size",  sizeof(Contribution) == 432);
    EXPECT("Contribution.align", alignof(Contribution) == 16);
    EXPECT("Contribution.id.off",       offsetof(Contribution, contribution_id) == 0);
    EXPECT("Contribution.cer.off",      offsetof(Contribution, ceremony_id)     == 8);
    EXPECT("Contribution.holder.off",   offsetof(Contribution, holder_addr)     == 16);
    EXPECT("Contribution.round.off",    offsetof(Contribution, round)           == 24);
    EXPECT("Contribution.hidx.off",     offsetof(Contribution, holder_index)    == 28);
    EXPECT("Contribution.plen.off",     offsetof(Contribution, payload_len)     == 32);
    EXPECT("Contribution.status.off",   offsetof(Contribution, status)          == 36);
    EXPECT("Contribution.payload.off",  offsetof(Contribution, payload)         == 40);
    PASS("Contribution layout");
}

void test_state_layout()
{
    EXPECT("MPCVMState.size",  sizeof(MPCVMState) == 160);
    EXPECT("MPCVMState.align", alignof(MPCVMState) == 16);
    EXPECT("MPCVMState.epoch.off",  offsetof(MPCVMState, current_epoch)              == 0);
    EXPECT("MPCVMState.now.off",    offsetof(MPCVMState, now_ns)                     == 8);
    EXPECT("MPCVMState.active.off", offsetof(MPCVMState, active_ceremony_count)      == 16);
    EXPECT("MPCVMState.fin.off",    offsetof(MPCVMState, finalized_ceremony_count)   == 20);
    EXPECT("MPCVMState.fail.off",   offsetof(MPCVMState, failed_ceremony_count)      == 24);
    EXPECT("MPCVMState.shares.off", offsetof(MPCVMState, key_share_count)            == 28);
    EXPECT("MPCVMState.croot.off",  offsetof(MPCVMState, ceremony_root)              == 32);
    EXPECT("MPCVMState.kroot.off",  offsetof(MPCVMState, key_share_root)             == 64);
    EXPECT("MPCVMState.contr.off",  offsetof(MPCVMState, contribution_root)          == 96);
    EXPECT("MPCVMState.sroot.off",  offsetof(MPCVMState, mpcvm_state_root)           == 128);
    PASS("MPCVMState layout");
}

void test_descriptor_layout()
{
    EXPECT("MPCVMRoundDescriptor.size",  sizeof(MPCVMRoundDescriptor) == 96);
    EXPECT("MPCVMRoundDescriptor.align", alignof(MPCVMRoundDescriptor) == 16);
    EXPECT("Desc.mode.off",   offsetof(MPCVMRoundDescriptor, mode)               == 32);
    EXPECT("Desc.parent.off", offsetof(MPCVMRoundDescriptor, parent_state_root)  == 64);
    PASS("MPCVMRoundDescriptor layout");
}

void test_ceremony_op_layout()
{
    EXPECT("CeremonyOp.size",  sizeof(CeremonyOp) == 96);
    EXPECT("CeremonyOp.align", alignof(CeremonyOp) == 16);
    EXPECT("CerOp.kind.off",      offsetof(CeremonyOp, kind)              == 16);
    EXPECT("CerOp.cerkind.off",   offsetof(CeremonyOp, ceremony_kind)     == 20);
    EXPECT("CerOp.subj.off",      offsetof(CeremonyOp, subject)           == 32);
    EXPECT("CerOp.seed.off",      offsetof(CeremonyOp, ceremony_seed)     == 64);
    PASS("CeremonyOp layout");
}

void test_contribution_op_layout()
{
    EXPECT("ContributionOp.size",  sizeof(ContributionOp) == 416);
    EXPECT("ContributionOp.align", alignof(ContributionOp) == 16);
    EXPECT("CntOp.cer.off",     offsetof(ContributionOp, ceremony_id)  == 0);
    EXPECT("CntOp.holder.off",  offsetof(ContributionOp, holder_addr)  == 8);
    EXPECT("CntOp.round.off",   offsetof(ContributionOp, round)        == 16);
    EXPECT("CntOp.hidx.off",    offsetof(ContributionOp, holder_index) == 20);
    EXPECT("CntOp.plen.off",    offsetof(ContributionOp, payload_len)  == 24);
    EXPECT("CntOp.payload.off", offsetof(ContributionOp, payload)      == 32);
    PASS("ContributionOp layout");
}

void test_result_layout()
{
    EXPECT("MPCVMTransitionResult.size",  sizeof(MPCVMTransitionResult) == 176);
    EXPECT("MPCVMTransitionResult.align", alignof(MPCVMTransitionResult) == 16);
    EXPECT("Result.epoch.off",        offsetof(MPCVMTransitionResult, epoch)               == 32);
    EXPECT("Result.now.off",          offsetof(MPCVMTransitionResult, now_ns)              == 40);
    EXPECT("Result.croot.off",        offsetof(MPCVMTransitionResult, ceremony_root)       == 48);
    EXPECT("Result.kroot.off",        offsetof(MPCVMTransitionResult, key_share_root)      == 80);
    EXPECT("Result.contr.off",        offsetof(MPCVMTransitionResult, contribution_root)   == 112);
    EXPECT("Result.sroot.off",        offsetof(MPCVMTransitionResult, mpcvm_state_root)    == 144);
    PASS("MPCVMTransitionResult layout");
}

// =============================================================================
// CPU reference tests
// =============================================================================

MPCVMRoundDescriptor make_desc(uint64_t round, uint64_t now_ns = 1700000000000000000ULL,
                               uint32_t mode = 7 /*FullRound*/)
{
    MPCVMRoundDescriptor d{};
    d.chain_id = 1u;
    d.round = round;
    d.timestamp_ns = now_ns;
    d.epoch = 0u;
    d.mode = mode;
    d.closing_flag = 1u;
    return d;
}

CeremonyOp make_begin(uint64_t cid, uint32_t threshold, uint32_t total,
                      uint32_t kind, uint64_t deadline = 2000000000000000000ULL,
                      uint8_t seed_fill = 0xAB)
{
    CeremonyOp op{};
    op.ceremony_id = cid;
    op.deadline_ns = deadline;
    op.kind = static_cast<uint32_t>(CeremonyOpKind::Begin);
    op.ceremony_kind = kind;
    op.threshold = threshold;
    op.total_participants = total;
    for (auto& b : op.subject)        b = uint8_t(seed_fill ^ 1u);
    for (auto& b : op.ceremony_seed)  b = seed_fill;
    return op;
}

ContributionOp make_contribution(uint64_t cid, uint32_t round, uint32_t holder,
                                 uint8_t fill = 0xCC, uint32_t plen = 64u)
{
    ContributionOp op{};
    op.ceremony_id = cid;
    op.holder_addr = 0xA00000000ULL + holder;
    op.round = round;
    op.holder_index = holder;
    op.payload_len = plen;
    for (uint32_t k = 0; k < plen; ++k) op.payload[k] = uint8_t(fill ^ k);
    return op;
}

void test_frost_keygen_7_of_10()
{
    auto state = ref::MPCVMReferenceState::empty();

    // Round 0: begin ceremony.
    std::vector<CeremonyOp> begins{
        make_begin(/*cid=*/42u, /*t=*/7u, /*n=*/10u,
                   /*kind=*/static_cast<uint32_t>(CeremonyKind::FrostKeygen)),
    };

    // Round 0 contributions: 7 participants submit (threshold met).
    std::vector<ContributionOp> r0;
    for (uint32_t i = 0; i < 7u; ++i) r0.push_back(make_contribution(42u, 0u, i, 0xC0u));

    auto desc0 = make_desc(1u);
    auto r = ref::run_reference(state, desc0, begins, r0);
    EXPECT("frost.kg.r0.applied", r.ceremony_apply_count == 1u);
    EXPECT("frost.kg.r0.contrib", r.contribution_apply_count == 7u);
    EXPECT("frost.kg.r0.advance", r.round_advance_count == 1u);

    // Round 1 contributions.
    std::vector<ContributionOp> r1;
    for (uint32_t i = 0; i < 7u; ++i) r1.push_back(make_contribution(42u, 1u, i, 0xD0u));
    auto desc1 = make_desc(2u);
    r = ref::run_reference(state, desc1, {}, r1);
    EXPECT("frost.kg.r1.contrib", r.contribution_apply_count == 7u);
    EXPECT("frost.kg.r1.advance", r.round_advance_count == 1u);

    // Round 2 contributions — finalizes.
    std::vector<ContributionOp> r2;
    for (uint32_t i = 0; i < 7u; ++i) r2.push_back(make_contribution(42u, 2u, i, 0xE0u));
    auto desc2 = make_desc(3u);
    r = ref::run_reference(state, desc2, {}, r2);
    EXPECT("frost.kg.r2.contrib", r.contribution_apply_count == 7u);
    EXPECT("frost.kg.r2.advance", r.round_advance_count == 1u);
    EXPECT("frost.kg.r2.final",   r.finalized_this_round == 1u);
    EXPECT("frost.kg.r2.shares",  r.key_share_count == 7u);

    // mpcvm_state_root non-zero.
    bool nz = false;
    for (auto b : r.mpcvm_state_root) if (b != 0) { nz = true; break; }
    EXPECT("frost.kg.root.nz", nz);

    PASS("FROST 7-of-10 keygen completes and emits 7 shares");
}

void test_frost_sign_7_of_10()
{
    auto state = ref::MPCVMReferenceState::empty();
    std::vector<CeremonyOp> begins{
        make_begin(/*cid=*/100u, 7u, 10u, static_cast<uint32_t>(CeremonyKind::FrostSign)),
    };
    std::vector<ContributionOp> r0;
    for (uint32_t i = 0; i < 7u; ++i) r0.push_back(make_contribution(100u, 0u, i));
    auto desc0 = make_desc(1u);
    auto r = ref::run_reference(state, desc0, begins, r0);
    EXPECT("frost.sg.r0.advance", r.round_advance_count == 1u);

    std::vector<ContributionOp> r1;
    for (uint32_t i = 0; i < 7u; ++i) r1.push_back(make_contribution(100u, 1u, i, 0xC1u));
    auto desc1 = make_desc(2u);
    r = ref::run_reference(state, desc1, {}, r1);
    EXPECT("frost.sg.r1.final", r.finalized_this_round == 1u);
    // Sign is not keygen — no shares emitted.
    EXPECT("frost.sg.r1.no-shares", r.key_share_count == 0u);

    PASS("FROST 7-of-10 sign completes (no shares)");
}

void test_cggmp21_5_of_9_keygen()
{
    auto state = ref::MPCVMReferenceState::empty();
    std::vector<CeremonyOp> begins{
        make_begin(7u, 5u, 9u, static_cast<uint32_t>(CeremonyKind::Cggmp21Keygen)),
    };
    auto desc = make_desc(1u);
    auto r = ref::run_reference(state, desc, begins, {});
    EXPECT("cgg.kg.begin.applied", r.ceremony_apply_count == 1u);

    // Need 3 rounds with 5 contributions each.
    for (uint32_t round = 0; round < 3u; ++round) {
        std::vector<ContributionOp> ops;
        for (uint32_t i = 0; i < 5u; ++i)
            ops.push_back(make_contribution(7u, round, i, uint8_t(0xA0u ^ round)));
        auto d = make_desc(2u + round);
        r = ref::run_reference(state, d, {}, ops);
    }
    EXPECT("cgg.kg.final",  r.finalized_this_round == 1u);
    EXPECT("cgg.kg.shares", r.key_share_count == 5u);
    PASS("CGGMP21 5-of-9 keygen finalizes with 5 shares");
}

void test_replay_rejected()
{
    auto state = ref::MPCVMReferenceState::empty();
    std::vector<CeremonyOp> begins{
        make_begin(11u, 3u, 5u, static_cast<uint32_t>(CeremonyKind::FrostSign)),
    };
    auto desc0 = make_desc(1u);

    // Submit holder 0 in round 0 once.
    std::vector<ContributionOp> first{ make_contribution(11u, 0u, 0u) };
    auto r = ref::run_reference(state, desc0, begins, first);
    EXPECT("replay.first.applied", r.contribution_apply_count == 1u);

    // Resubmit the same holder/round in next call — must be silently dropped.
    std::vector<ContributionOp> dup{ make_contribution(11u, 0u, 0u, 0xFFu) };
    auto desc1 = make_desc(2u);
    r = ref::run_reference(state, desc1, {}, dup);
    EXPECT("replay.dup.dropped", r.contribution_apply_count == 0u);

    PASS("Replay attempt dropped");
}

void test_timeout_marks_failed()
{
    auto state = ref::MPCVMReferenceState::empty();
    // Deadline in the past — should fail at first sweep.
    auto desc0 = make_desc(1u, /*now=*/2000u);
    std::vector<CeremonyOp> begins{
        make_begin(13u, 7u, 10u, static_cast<uint32_t>(CeremonyKind::FrostKeygen),
                   /*deadline=*/1000u),
    };
    // Only 5 contributions — under threshold.
    std::vector<ContributionOp> r0;
    for (uint32_t i = 0; i < 5u; ++i) r0.push_back(make_contribution(13u, 0u, i));
    auto r = ref::run_reference(state, desc0, begins, r0);
    EXPECT("timeout.failed", r.failed_this_round == 1u);
    PASS("Timeout marks ceremony failed");
}

void test_concurrent_ceremonies()
{
    auto state = ref::MPCVMReferenceState::empty();
    std::vector<CeremonyOp> begins{
        make_begin(1u, 7u, 10u, static_cast<uint32_t>(CeremonyKind::FrostKeygen),
                   /*deadline=*/2000000000000000000ULL, /*fill=*/0xA0u),
        make_begin(2u, 5u, 9u,  static_cast<uint32_t>(CeremonyKind::Cggmp21Keygen),
                   /*deadline=*/2000000000000000000ULL, /*fill=*/0xB0u),
        make_begin(3u, 3u, 5u,  static_cast<uint32_t>(CeremonyKind::FrostSign),
                   /*deadline=*/2000000000000000000ULL, /*fill=*/0xC0u),
    };
    std::vector<ContributionOp> r0;
    for (uint32_t i = 0; i < 7u; ++i) r0.push_back(make_contribution(1u, 0u, i, 0xA0u));
    for (uint32_t i = 0; i < 5u; ++i) r0.push_back(make_contribution(2u, 0u, i, 0xB0u));
    for (uint32_t i = 0; i < 3u; ++i) r0.push_back(make_contribution(3u, 0u, i, 0xC0u));

    auto desc0 = make_desc(1u);
    auto state2 = state;  // copy for determinism check
    auto r1 = ref::run_reference(state, desc0, begins, r0);
    auto r2 = ref::run_reference(state2, desc0, begins, r0);
    EXPECT("conc.applied",     r1.ceremony_apply_count == 3u);
    EXPECT("conc.contrib",     r1.contribution_apply_count == 15u);
    EXPECT("conc.advance",     r1.round_advance_count == 3u);
    EXPECT("conc.det.croot",   std::memcmp(r1.ceremony_root,    r2.ceremony_root,    32) == 0);
    EXPECT("conc.det.kroot",   std::memcmp(r1.key_share_root,   r2.key_share_root,   32) == 0);
    EXPECT("conc.det.contr",   std::memcmp(r1.contribution_root,r2.contribution_root,32) == 0);
    EXPECT("conc.det.sroot",   std::memcmp(r1.mpcvm_state_root, r2.mpcvm_state_root, 32) == 0);
    PASS("3 concurrent ceremonies, deterministic across runs");
}

void test_empty_round_deterministic()
{
    auto desc = make_desc(1u);
    auto state1 = ref::MPCVMReferenceState::empty();
    auto state2 = ref::MPCVMReferenceState::empty();
    auto r1 = ref::run_reference(state1, desc, {}, {});
    auto r2 = ref::run_reference(state2, desc, {}, {});
    EXPECT("empty.det.sroot", std::memcmp(r1.mpcvm_state_root, r2.mpcvm_state_root, 32) == 0);
    bool nz = false;
    for (auto b : r1.mpcvm_state_root) if (b != 0) { nz = true; break; }
    EXPECT("empty.sroot.nz", nz);
    PASS("Empty round deterministic non-zero");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("[mpcvm_layout_test] starting\n");

    test_ceremony_layout();
    test_key_share_layout();
    test_contribution_layout();
    test_state_layout();
    test_descriptor_layout();
    test_ceremony_op_layout();
    test_contribution_op_layout();
    test_result_layout();

    test_frost_keygen_7_of_10();
    test_frost_sign_7_of_10();
    test_cggmp21_5_of_9_keygen();
    test_replay_rejected();
    test_timeout_marks_failed();
    test_concurrent_ceremonies();
    test_empty_round_deterministic();

    std::printf("[mpcvm_layout_test] passed=%d failed=%d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
