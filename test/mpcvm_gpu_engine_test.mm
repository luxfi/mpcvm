// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_gpu_engine_test.mm — Metal-side correctness for MPCVMGPUEngine.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "lux/mpcvm/mpcvm_gpu_engine.hpp"
#include "lux/mpcvm/mpcvm_cpu_reference.hpp"

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

MPCVMRoundDescriptor make_desc(uint64_t round, uint64_t now_ns = 1700000000000000000ULL,
                               uint32_t mode = 7)
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
                      uint8_t fill = 0xAB)
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

void test_engine_creates()
{
    auto engine = MPCVMGPUEngine::create();
    EXPECT("engine.create", engine != nullptr);
    std::printf("  engine.device: %s\n", engine->device_name());
    PASS("engine creates");
}

void test_engine_full_round_matches_cpu()
{
    auto engine = MPCVMGPUEngine::create();
    EXPECT("matches.engine", engine != nullptr);

    std::vector<CeremonyOp> begins{
        make_begin(42u, 7u, 10u,
                   static_cast<uint32_t>(CeremonyKind::FrostKeygen)),
    };
    std::vector<ContributionOp> r0;
    for (uint32_t i = 0; i < 7u; ++i) r0.push_back(make_contribution(42u, 0u, i, 0xC0u));

    auto desc0 = make_desc(1u);
    auto h = engine->begin_round(desc0);
    EXPECT("matches.handle", h.valid());
    engine->push_ceremony_ops(h, begins);
    engine->push_contribution_ops(h, r0);
    auto gpu_r = engine->run_until_done(h);
    engine->end_round(h);

    auto state = ref::MPCVMReferenceState::empty();
    auto cpu_r = ref::run_reference(state, desc0, begins, r0);

    EXPECT("matches.cer_app",   gpu_r.ceremony_apply_count == cpu_r.ceremony_apply_count);
    EXPECT("matches.cnt_app",   gpu_r.contribution_apply_count == cpu_r.contribution_apply_count);
    EXPECT("matches.advance",   gpu_r.round_advance_count == cpu_r.round_advance_count);
    EXPECT("matches.croot",     std::memcmp(gpu_r.ceremony_root,    cpu_r.ceremony_root,    32) == 0);
    EXPECT("matches.kroot",     std::memcmp(gpu_r.key_share_root,   cpu_r.key_share_root,   32) == 0);
    EXPECT("matches.contr",     std::memcmp(gpu_r.contribution_root,cpu_r.contribution_root,32) == 0);
    EXPECT("matches.sroot",     std::memcmp(gpu_r.mpcvm_state_root, cpu_r.mpcvm_state_root, 32) == 0);

    std::printf("  gpu cer=%u cnt=%u advance=%u\n",
                gpu_r.ceremony_apply_count, gpu_r.contribution_apply_count,
                gpu_r.round_advance_count);
    PASS("FROST keygen round 0 matches CPU reference");
}

void test_engine_concurrent_ceremonies()
{
    auto engine = MPCVMGPUEngine::create();
    EXPECT("conc.engine", engine != nullptr);

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

    auto desc0 = make_desc(1u);
    auto h = engine->begin_round(desc0);
    engine->push_ceremony_ops(h, begins);
    engine->push_contribution_ops(h, r0);
    auto gpu_r = engine->run_until_done(h);
    engine->end_round(h);

    auto state = ref::MPCVMReferenceState::empty();
    auto cpu_r = ref::run_reference(state, desc0, begins, r0);
    EXPECT("conc.match.cer", gpu_r.ceremony_apply_count == cpu_r.ceremony_apply_count);
    EXPECT("conc.match.cnt", gpu_r.contribution_apply_count == cpu_r.contribution_apply_count);
    EXPECT("conc.match.adv", gpu_r.round_advance_count == cpu_r.round_advance_count);
    EXPECT("conc.match.sroot", std::memcmp(gpu_r.mpcvm_state_root, cpu_r.mpcvm_state_root, 32) == 0);
    PASS("3 concurrent ceremonies match CPU reference");
}

void test_engine_replay_dropped()
{
    auto engine = MPCVMGPUEngine::create();
    EXPECT("rp.engine", engine != nullptr);

    std::vector<CeremonyOp> begins{
        make_begin(11u, 3u, 5u, static_cast<uint32_t>(CeremonyKind::FrostSign)),
    };
    std::vector<ContributionOp> ops{
        make_contribution(11u, 0u, 0u),
        make_contribution(11u, 0u, 0u, 0xFF),  // duplicate
    };
    auto desc = make_desc(1u);
    auto h = engine->begin_round(desc);
    engine->push_ceremony_ops(h, begins);
    engine->push_contribution_ops(h, ops);
    auto gpu_r = engine->run_until_done(h);
    engine->end_round(h);

    EXPECT("rp.cnt_app", gpu_r.contribution_apply_count == 1u);
    PASS("Replay dropped on GPU");
}

void test_engine_empty_round_deterministic()
{
    auto engine = MPCVMGPUEngine::create();
    EXPECT("empty.engine", engine != nullptr);
    auto desc = make_desc(1u);
    auto h1 = engine->begin_round(desc);
    auto r1 = engine->run_until_done(h1);
    engine->end_round(h1);
    auto h2 = engine->begin_round(desc);
    auto r2 = engine->run_until_done(h2);
    engine->end_round(h2);
    EXPECT("empty.match", std::memcmp(r1.mpcvm_state_root, r2.mpcvm_state_root, 32) == 0);
    bool nz = false;
    for (auto b : r1.mpcvm_state_root) if (b != 0) { nz = true; break; }
    EXPECT("empty.nz", nz);

    auto state = ref::MPCVMReferenceState::empty();
    auto cpu_r = ref::run_reference(state, desc, {}, {});
    EXPECT("empty.match.cpu", std::memcmp(r1.mpcvm_state_root, cpu_r.mpcvm_state_root, 32) == 0);
    PASS("Empty round deterministic and matches CPU");
}

void test_engine_timeout()
{
    auto engine = MPCVMGPUEngine::create();
    EXPECT("to.engine", engine != nullptr);
    std::vector<CeremonyOp> begins{
        make_begin(13u, 7u, 10u, static_cast<uint32_t>(CeremonyKind::FrostKeygen),
                   /*deadline=*/1000ULL),
    };
    std::vector<ContributionOp> r0;
    for (uint32_t i = 0; i < 5u; ++i) r0.push_back(make_contribution(13u, 0u, i));

    auto desc = make_desc(1u, /*now=*/2000ULL);
    auto h = engine->begin_round(desc);
    engine->push_ceremony_ops(h, begins);
    engine->push_contribution_ops(h, r0);
    auto gpu_r = engine->run_until_done(h);
    engine->end_round(h);

    auto state = ref::MPCVMReferenceState::empty();
    auto cpu_r = ref::run_reference(state, desc, begins, r0);
    EXPECT("to.match.failed", gpu_r.failed_this_round == cpu_r.failed_this_round);
    EXPECT("to.match.sroot",  std::memcmp(gpu_r.mpcvm_state_root, cpu_r.mpcvm_state_root, 32) == 0);
    PASS("Timeout sweep matches CPU");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    @autoreleasepool {
        std::printf("[mpcvm_gpu_engine_test] starting\n");

        test_engine_creates();
        test_engine_full_round_matches_cpu();
        test_engine_concurrent_ceremonies();
        test_engine_replay_dropped();
        test_engine_empty_round_deterministic();
        test_engine_timeout();

        std::printf("[mpcvm_gpu_engine_test] passed=%d failed=%d\n", g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    }
}
