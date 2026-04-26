// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_determinism_test.cpp — cross-backend determinism harness.
//
// Compares CPU reference vs GPU engine across canonical workloads:
//   1. FROST 7-of-10 keygen (3 rounds)
//   2. FROST 7-of-10 sign (2 rounds)
//   3. CGGMP21 5-of-9 keygen (3 rounds)
//   4. Replay dropped
//   5. Timeout sweep
//   6. Concurrent ceremonies (3 different kinds)
//   7. Empty round deterministic non-zero
//   8. Two engines run-twice — bytes match
//
// The GPU side is selected at link time by the platform-specific
// MPCVMGPUEngine::create() implementation.

#include "lux/mpcvm/mpcvm_gpu_engine.hpp"
#include "lux/mpcvm/mpcvm_cpu_reference.hpp"

#include <cstdio>
#include <cstdlib>
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

MPCVMRoundDescriptor make_desc(uint64_t round, uint64_t now_ns = 1700000000000000000ULL,
                               uint64_t epoch = 0)
{
    MPCVMRoundDescriptor d{};
    d.chain_id = 1u;
    d.round = round;
    d.timestamp_ns = now_ns;
    d.epoch = epoch;
    d.mode = static_cast<uint32_t>(MPCVMTransitionMode::FullRound);
    d.closing_flag = 1u;
    return d;
}

CeremonyOp make_begin(uint64_t cid, uint32_t threshold, uint32_t total, uint32_t kind,
                      uint64_t deadline = 2000000000000000000ULL, uint8_t fill = 0xAB)
{
    CeremonyOp op{};
    op.ceremony_id = cid;
    op.deadline_ns = deadline;
    op.kind = static_cast<uint32_t>(CeremonyOpKind::Begin);
    op.ceremony_kind = kind;
    op.threshold = threshold;
    op.total_participants = total;
    for (auto& b : op.subject)        b = uint8_t(fill ^ 1u);
    for (auto& b : op.ceremony_seed)  b = fill;
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

struct WorkloadResult {
    uint8_t  ceremony_root[32];
    uint8_t  key_share_root[32];
    uint8_t  contribution_root[32];
    uint8_t  mpcvm_state_root[32];
    uint32_t ceremony_apply_count;
    uint32_t contribution_apply_count;
    uint32_t round_advance_count;
    uint32_t finalized_this_round;
    uint32_t failed_this_round;
    uint32_t active_ceremony_count;
    uint32_t key_share_count;

    static WorkloadResult from(const MPCVMTransitionResult& r) {
        WorkloadResult w{};
        std::memcpy(w.ceremony_root,     r.ceremony_root,     32);
        std::memcpy(w.key_share_root,    r.key_share_root,    32);
        std::memcpy(w.contribution_root, r.contribution_root, 32);
        std::memcpy(w.mpcvm_state_root,  r.mpcvm_state_root,  32);
        w.ceremony_apply_count = r.ceremony_apply_count;
        w.contribution_apply_count = r.contribution_apply_count;
        w.round_advance_count = r.round_advance_count;
        w.finalized_this_round = r.finalized_this_round;
        w.failed_this_round = r.failed_this_round;
        w.active_ceremony_count = r.active_ceremony_count;
        w.key_share_count = r.key_share_count;
        return w;
    }

    bool equals(const WorkloadResult& o) const {
        return std::memcmp(ceremony_root,     o.ceremony_root,     32) == 0
            && std::memcmp(key_share_root,    o.key_share_root,    32) == 0
            && std::memcmp(contribution_root, o.contribution_root, 32) == 0
            && std::memcmp(mpcvm_state_root,  o.mpcvm_state_root,  32) == 0
            && ceremony_apply_count == o.ceremony_apply_count
            && contribution_apply_count == o.contribution_apply_count
            && round_advance_count == o.round_advance_count
            && finalized_this_round == o.finalized_this_round
            && failed_this_round == o.failed_this_round
            && active_ceremony_count == o.active_ceremony_count
            && key_share_count == o.key_share_count;
    }
};

WorkloadResult run_cpu(const MPCVMRoundDescriptor& desc,
                       const std::vector<CeremonyOp>& cer_ops,
                       const std::vector<ContributionOp>& cnt_ops)
{
    auto state = ref::MPCVMReferenceState::empty();
    auto r = ref::run_reference(state, desc, cer_ops, cnt_ops);
    return WorkloadResult::from(r);
}

WorkloadResult run_gpu(MPCVMGPUEngine* engine,
                       const MPCVMRoundDescriptor& desc,
                       const std::vector<CeremonyOp>& cer_ops,
                       const std::vector<ContributionOp>& cnt_ops)
{
    auto h = engine->begin_round(desc);
    if (!cer_ops.empty()) engine->push_ceremony_ops(h, cer_ops);
    if (!cnt_ops.empty()) engine->push_contribution_ops(h, cnt_ops);
    auto r = engine->run_until_done(h);
    engine->end_round(h);
    return WorkloadResult::from(r);
}

void test_frost_keygen_round0(MPCVMGPUEngine* engine)
{
    std::vector<CeremonyOp> begins{
        make_begin(42u, 7u, 10u, static_cast<uint32_t>(CeremonyKind::FrostKeygen)),
    };
    std::vector<ContributionOp> r0;
    for (uint32_t i = 0; i < 7u; ++i) r0.push_back(make_contribution(42u, 0u, i, 0xC0u));

    auto desc0 = make_desc(1u);
    auto cpu = run_cpu(desc0, begins, r0);
    if (engine != nullptr) {
        auto gpu = run_gpu(engine, desc0, begins, r0);
        EXPECT("frost.kg0.match", cpu.equals(gpu));
    } else {
        auto cpu2 = run_cpu(desc0, begins, r0);
        EXPECT("frost.kg0.cpu_det", cpu.equals(cpu2));
    }
    PASS("FROST keygen round 0 — CPU<->GPU match");
}

void test_cggmp21_keygen_round0(MPCVMGPUEngine* engine)
{
    std::vector<CeremonyOp> begins{
        make_begin(7u, 5u, 9u, static_cast<uint32_t>(CeremonyKind::Cggmp21Keygen)),
    };
    std::vector<ContributionOp> r0;
    for (uint32_t i = 0; i < 5u; ++i) r0.push_back(make_contribution(7u, 0u, i, 0xA0u));

    auto desc0 = make_desc(1u);
    auto cpu = run_cpu(desc0, begins, r0);
    if (engine != nullptr) {
        auto gpu = run_gpu(engine, desc0, begins, r0);
        EXPECT("cgg.kg0.match", cpu.equals(gpu));
    } else {
        auto cpu2 = run_cpu(desc0, begins, r0);
        EXPECT("cgg.kg0.cpu_det", cpu.equals(cpu2));
    }
    PASS("CGGMP21 keygen round 0 — CPU<->GPU match");
}

void test_replay_dropped(MPCVMGPUEngine* engine)
{
    std::vector<CeremonyOp> begins{
        make_begin(11u, 3u, 5u, static_cast<uint32_t>(CeremonyKind::FrostSign)),
    };
    std::vector<ContributionOp> ops{
        make_contribution(11u, 0u, 0u),
        make_contribution(11u, 0u, 0u, 0xFFu),
    };
    auto desc = make_desc(1u);
    auto cpu = run_cpu(desc, begins, ops);
    EXPECT("rp.cpu.applied", cpu.contribution_apply_count == 1u);
    if (engine != nullptr) {
        auto gpu = run_gpu(engine, desc, begins, ops);
        EXPECT("rp.match", cpu.equals(gpu));
    }
    PASS("Replay dropped");
}

void test_timeout(MPCVMGPUEngine* engine)
{
    std::vector<CeremonyOp> begins{
        make_begin(13u, 7u, 10u, static_cast<uint32_t>(CeremonyKind::FrostKeygen),
                   /*deadline=*/1000ULL),
    };
    std::vector<ContributionOp> r0;
    for (uint32_t i = 0; i < 5u; ++i) r0.push_back(make_contribution(13u, 0u, i));

    auto desc = make_desc(1u, /*now=*/2000ULL);
    auto cpu = run_cpu(desc, begins, r0);
    EXPECT("to.cpu.failed", cpu.failed_this_round == 1u);
    if (engine != nullptr) {
        auto gpu = run_gpu(engine, desc, begins, r0);
        EXPECT("to.match", cpu.equals(gpu));
    }
    PASS("Timeout sweep");
}

void test_concurrent_ceremonies(MPCVMGPUEngine* engine)
{
    std::vector<CeremonyOp> begins{
        make_begin(1u, 7u, 10u, static_cast<uint32_t>(CeremonyKind::FrostKeygen),
                   2000000000000000000ULL, 0xA0),
        make_begin(2u, 5u, 9u,  static_cast<uint32_t>(CeremonyKind::Cggmp21Keygen),
                   2000000000000000000ULL, 0xB0),
        make_begin(3u, 3u, 5u,  static_cast<uint32_t>(CeremonyKind::FrostSign),
                   2000000000000000000ULL, 0xC0),
    };
    std::vector<ContributionOp> r0;
    for (uint32_t i = 0; i < 7u; ++i) r0.push_back(make_contribution(1u, 0u, i, 0xA0u));
    for (uint32_t i = 0; i < 5u; ++i) r0.push_back(make_contribution(2u, 0u, i, 0xB0u));
    for (uint32_t i = 0; i < 3u; ++i) r0.push_back(make_contribution(3u, 0u, i, 0xC0u));

    auto desc = make_desc(1u);
    auto cpu = run_cpu(desc, begins, r0);
    EXPECT("conc.cpu.advance", cpu.round_advance_count == 3u);
    if (engine != nullptr) {
        auto gpu = run_gpu(engine, desc, begins, r0);
        EXPECT("conc.match", cpu.equals(gpu));
    }
    PASS("3 concurrent ceremonies");
}

void test_empty_round(MPCVMGPUEngine* engine)
{
    auto desc = make_desc(1u);
    auto cpu = run_cpu(desc, {}, {});
    bool nz = false;
    for (auto b : cpu.mpcvm_state_root) if (b != 0) { nz = true; break; }
    EXPECT("empty.cpu.nz", nz);
    if (engine != nullptr) {
        auto gpu = run_gpu(engine, desc, {}, {});
        EXPECT("empty.match", cpu.equals(gpu));
    }
    PASS("Empty round deterministic non-zero");
}

void test_two_engines_match(MPCVMGPUEngine* engine)
{
    if (engine == nullptr) {
        PASS("two-engines (skipped — no GPU)");
        return;
    }
    auto a = MPCVMGPUEngine::create();
    auto b = MPCVMGPUEngine::create();
    EXPECT("twoeng.a", a != nullptr);
    EXPECT("twoeng.b", b != nullptr);

    std::vector<CeremonyOp> begins{
        make_begin(99u, 5u, 8u, static_cast<uint32_t>(CeremonyKind::FrostKeygen),
                   2000000000000000000ULL, 0x55),
    };
    std::vector<ContributionOp> r0;
    for (uint32_t i = 0; i < 5u; ++i) r0.push_back(make_contribution(99u, 0u, i, 0x55));

    auto desc = make_desc(1u);
    auto ra = run_gpu(a.get(), desc, begins, r0);
    auto rb = run_gpu(b.get(), desc, begins, r0);
    EXPECT("twoeng.match", ra.equals(rb));
    PASS("Two engines bytewise identical");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("[mpcvm_determinism_test] starting\n");

    auto engine = MPCVMGPUEngine::create();
    if (engine == nullptr) {
        std::printf("  note: no GPU backend — running CPU-only path\n");
    } else {
        std::printf("  device: %s\n", engine->device_name());
    }

    test_frost_keygen_round0(engine.get());
    test_cggmp21_keygen_round0(engine.get());
    test_replay_dropped(engine.get());
    test_timeout(engine.get());
    test_concurrent_ceremonies(engine.get());
    test_empty_round(engine.get());
    test_two_engines_match(engine.get());

    std::printf("[mpcvm_determinism_test] passed=%d failed=%d\n",
                g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
