// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_benchmark.cpp — GPU acceleration benchmark harness.
//
// Per-scheme microbenchmarks and mixed-workload macrobenchmarks across
// every available backend (CPU reference, Metal, WGPU/wgpu-native, CUDA
// when built). Each scenario runs 3 warm-up + 10 measured iterations.
// Output is a stable, machine-parseable line stream consumed by
// BENCHMARKS.md generation.

#include "lux/mpcvm/mpcvm_gpu_engine.hpp"
#include "lux/mpcvm/mpcvm_cpu_reference.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace mpcvm::gpu;
using clock_t_ = std::chrono::steady_clock;

namespace {

// ---------- workload synthesis ---------------------------------------------

MPCVMRoundDescriptor make_desc(uint64_t round)
{
    MPCVMRoundDescriptor d{};
    d.chain_id = 1u;
    d.round = round;
    d.timestamp_ns = 1700000000000000000ULL;
    d.epoch = 0;
    d.mode = static_cast<uint32_t>(MPCVMTransitionMode::FullRound);
    d.closing_flag = 1u;
    return d;
}

CeremonyOp make_begin(uint64_t cid, uint32_t threshold, uint32_t total,
                      uint32_t kind, uint8_t fill)
{
    CeremonyOp op{};
    op.ceremony_id = cid;
    op.deadline_ns = 2000000000000000000ULL;
    op.kind = static_cast<uint32_t>(CeremonyOpKind::Begin);
    op.ceremony_kind = kind;
    op.threshold = threshold;
    op.total_participants = total;
    for (auto& b : op.subject)       b = uint8_t(fill ^ 0x5A);
    for (auto& b : op.ceremony_seed) b = fill;
    return op;
}

ContributionOp make_contribution(uint64_t cid, uint32_t round, uint32_t holder,
                                  uint8_t fill, uint32_t plen = 64u)
{
    ContributionOp op{};
    op.ceremony_id = cid;
    op.holder_addr = 0xA00000000ULL + (cid << 8) + holder;
    op.round = round;
    op.holder_index = holder;
    op.payload_len = plen;
    for (uint32_t k = 0; k < plen; ++k) op.payload[k] = uint8_t(fill ^ k);
    return op;
}

struct Workload {
    std::string name;
    uint64_t    ceremony_count = 0;
    uint64_t    contribution_count = 0;
    std::vector<CeremonyOp>     begins;
    std::vector<ContributionOp> contribs;
};

// Mixed workloads cap per-round operations at the arena bounds:
//   kDefaultCeremonySlots     = 256
//   kDefaultContributionSlots = 4096
// Larger ceremony counts are spread across multiple rounds.
struct WorkloadShape {
    uint64_t frost_keygen   = 0;   ///< 7-of-10
    uint64_t frost_sign     = 0;   ///< 5-of-7
    uint64_t cggmp21_keygen = 0;   ///< 5-of-9
    uint64_t cggmp21_sign   = 0;   ///< 4-of-7
    uint64_t ringtail_dkg   = 0;   ///< 4-of-7
};

// Per-round chunks that fit inside the arena.
struct ChunkLimits {
    static constexpr uint64_t kCeremonyCap = 200u;       ///< leave headroom
    static constexpr uint64_t kContributionCap = 3500u;  ///< leave headroom
};

void append_frost_keygen(std::vector<CeremonyOp>& begins,
                         std::vector<ContributionOp>& contribs,
                         uint64_t& cid, uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i, ++cid) {
        begins.push_back(make_begin(cid, 7u, 10u,
            static_cast<uint32_t>(CeremonyKind::FrostKeygen),
            uint8_t(0xC0 + (cid & 0x0F))));
        for (uint32_t h = 0; h < 7u; ++h) {
            contribs.push_back(make_contribution(cid, 0u, h,
                uint8_t(0xC0 + (cid & 0x0F))));
        }
    }
}

void append_frost_sign(std::vector<CeremonyOp>& begins,
                       std::vector<ContributionOp>& contribs,
                       uint64_t& cid, uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i, ++cid) {
        begins.push_back(make_begin(cid, 5u, 7u,
            static_cast<uint32_t>(CeremonyKind::FrostSign),
            uint8_t(0xD0 + (cid & 0x0F))));
        for (uint32_t h = 0; h < 5u; ++h) {
            contribs.push_back(make_contribution(cid, 0u, h,
                uint8_t(0xD0 + (cid & 0x0F))));
        }
    }
}

void append_cggmp21_keygen(std::vector<CeremonyOp>& begins,
                           std::vector<ContributionOp>& contribs,
                           uint64_t& cid, uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i, ++cid) {
        begins.push_back(make_begin(cid, 5u, 9u,
            static_cast<uint32_t>(CeremonyKind::Cggmp21Keygen),
            uint8_t(0xA0 + (cid & 0x0F))));
        for (uint32_t h = 0; h < 5u; ++h) {
            contribs.push_back(make_contribution(cid, 0u, h,
                uint8_t(0xA0 + (cid & 0x0F))));
        }
    }
}

void append_cggmp21_sign(std::vector<CeremonyOp>& begins,
                         std::vector<ContributionOp>& contribs,
                         uint64_t& cid, uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i, ++cid) {
        begins.push_back(make_begin(cid, 4u, 7u,
            static_cast<uint32_t>(CeremonyKind::Cggmp21Sign),
            uint8_t(0xB0 + (cid & 0x0F))));
        for (uint32_t h = 0; h < 4u; ++h) {
            contribs.push_back(make_contribution(cid, 0u, h,
                uint8_t(0xB0 + (cid & 0x0F))));
        }
    }
}

void append_ringtail_dkg(std::vector<CeremonyOp>& begins,
                         std::vector<ContributionOp>& contribs,
                         uint64_t& cid, uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i, ++cid) {
        begins.push_back(make_begin(cid, 4u, 7u,
            static_cast<uint32_t>(CeremonyKind::RingtailDkg),
            uint8_t(0xE0 + (cid & 0x0F))));
        for (uint32_t h = 0; h < 4u; ++h) {
            contribs.push_back(make_contribution(cid, 0u, h,
                uint8_t(0xE0 + (cid & 0x0F)), /*plen=*/256u));
        }
    }
}

// Split a shape into a sequence of per-round Workloads, each fitting
// arena bounds.
std::vector<Workload> shape_to_chunks(const std::string& name,
                                      const WorkloadShape& s)
{
    std::vector<Workload> chunks;
    uint64_t cid = 1ULL;
    uint64_t total = s.frost_keygen + s.frost_sign + s.cggmp21_keygen
                   + s.cggmp21_sign + s.ringtail_dkg;
    uint64_t produced = 0;
    uint64_t fk = s.frost_keygen, fs = s.frost_sign;
    uint64_t ck = s.cggmp21_keygen, cs = s.cggmp21_sign, rd = s.ringtail_dkg;

    auto take = [&](uint64_t& remaining, uint64_t want, auto append_fn,
                    std::vector<CeremonyOp>& begins,
                    std::vector<ContributionOp>& contribs,
                    uint64_t per_ceremony_contribs) -> uint64_t {
        if (remaining == 0) return 0;
        uint64_t cap_by_cer = (ChunkLimits::kCeremonyCap > begins.size())
            ? (ChunkLimits::kCeremonyCap - begins.size()) : 0;
        uint64_t cap_by_cnt = (ChunkLimits::kContributionCap > contribs.size())
            ? ((ChunkLimits::kContributionCap - contribs.size()) / per_ceremony_contribs) : 0;
        uint64_t take_n = std::min({remaining, want, cap_by_cer, cap_by_cnt});
        if (take_n == 0) return 0;
        append_fn(begins, contribs, cid, take_n);
        remaining -= take_n;
        return take_n;
    };

    while (produced < total) {
        Workload w{};
        w.name = name;
        // Interleave schemes per chunk for balanced GPU occupancy.
        bool advanced = false;
        uint64_t step;
        step = take(fk, 64, append_frost_keygen,    w.begins, w.contribs, 7);
        if (step) { advanced = true; produced += step; }
        step = take(fs, 64, append_frost_sign,      w.begins, w.contribs, 5);
        if (step) { advanced = true; produced += step; }
        step = take(ck, 64, append_cggmp21_keygen,  w.begins, w.contribs, 5);
        if (step) { advanced = true; produced += step; }
        step = take(cs, 64, append_cggmp21_sign,    w.begins, w.contribs, 4);
        if (step) { advanced = true; produced += step; }
        step = take(rd, 32, append_ringtail_dkg,    w.begins, w.contribs, 4);
        if (step) { advanced = true; produced += step; }
        if (!advanced) {
            // No scheme could fit any more; flush bigger chunks via single-scheme
            // greedy fill.
            step = take(fk, fk, append_frost_keygen,   w.begins, w.contribs, 7);
            produced += step;
            step = take(fs, fs, append_frost_sign,     w.begins, w.contribs, 5);
            produced += step;
            step = take(ck, ck, append_cggmp21_keygen, w.begins, w.contribs, 5);
            produced += step;
            step = take(cs, cs, append_cggmp21_sign,   w.begins, w.contribs, 4);
            produced += step;
            step = take(rd, rd, append_ringtail_dkg,   w.begins, w.contribs, 4);
            produced += step;
            if (w.begins.empty()) break;  // safety
        }
        w.ceremony_count = w.begins.size();
        w.contribution_count = w.contribs.size();
        chunks.push_back(std::move(w));
    }
    return chunks;
}

// ---------- runners ---------------------------------------------------------

// Run one chunk through CPU reference. Returns elapsed nanoseconds.
uint64_t run_cpu_chunk(const Workload& w)
{
    auto state = ref::MPCVMReferenceState::empty();
    auto desc = make_desc(1u);
    auto t0 = clock_t_::now();
    (void)ref::run_reference(state, desc, w.begins, w.contribs);
    auto t1 = clock_t_::now();
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

uint64_t run_gpu_chunk(MPCVMGPUEngine* engine, const Workload& w)
{
    auto desc = make_desc(1u);
    auto t0 = clock_t_::now();
    auto h = engine->begin_round(desc);
    if (!w.begins.empty())  engine->push_ceremony_ops(h, w.begins);
    if (!w.contribs.empty()) engine->push_contribution_ops(h, w.contribs);
    (void)engine->run_until_done(h);
    engine->end_round(h);
    auto t1 = clock_t_::now();
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

// Sum elapsed across chunks.
uint64_t run_workload_cpu(const std::vector<Workload>& chunks)
{
    uint64_t total = 0;
    for (const auto& w : chunks) total += run_cpu_chunk(w);
    return total;
}

uint64_t run_workload_gpu(MPCVMGPUEngine* engine,
                          const std::vector<Workload>& chunks)
{
    uint64_t total = 0;
    for (const auto& w : chunks) total += run_gpu_chunk(engine, w);
    return total;
}

// ---------- statistics ------------------------------------------------------

struct Stats {
    double mean_ns = 0;
    double p50_ns = 0;
    double p95_ns = 0;
    double p99_ns = 0;
    double min_ns = 0;
    double max_ns = 0;
};

Stats compute_stats(std::vector<uint64_t> samples)
{
    Stats s{};
    if (samples.empty()) return s;
    std::sort(samples.begin(), samples.end());
    uint64_t sum = 0;
    for (auto v : samples) sum += v;
    s.mean_ns = double(sum) / double(samples.size());
    auto pick = [&](double q) -> double {
        if (samples.size() == 1) return double(samples.front());
        double idx = q * double(samples.size() - 1);
        size_t lo = size_t(idx);
        size_t hi = std::min(lo + 1, samples.size() - 1);
        double frac = idx - double(lo);
        return double(samples[lo]) * (1.0 - frac) + double(samples[hi]) * frac;
    };
    s.p50_ns = pick(0.50);
    s.p95_ns = pick(0.95);
    s.p99_ns = pick(0.99);
    s.min_ns = double(samples.front());
    s.max_ns = double(samples.back());
    return s;
}

// ---------- backend dispatch -----------------------------------------------

enum class Backend : int {
    CPU = 0,
    Metal = 1,
    WGPU = 2,
    CUDA = 3,
};

const char* backend_name(Backend b)
{
    switch (b) {
        case Backend::CPU:   return "cpu";
        case Backend::Metal: return "metal";
        case Backend::WGPU:  return "wgpu";
        case Backend::CUDA:  return "cuda";
    }
    return "?";
}

constexpr int kWarmup = 2;
constexpr int kIters  = 5;

// Runs (warmup + iters) on a workload for one backend. CPU has no engine.
std::vector<uint64_t> run_iters_cpu(const std::vector<Workload>& chunks)
{
    for (int i = 0; i < kWarmup; ++i) (void)run_workload_cpu(chunks);
    std::vector<uint64_t> samples; samples.reserve(kIters);
    for (int i = 0; i < kIters; ++i) samples.push_back(run_workload_cpu(chunks));
    return samples;
}

std::vector<uint64_t> run_iters_gpu(MPCVMGPUEngine* engine,
                                    const std::vector<Workload>& chunks)
{
    for (int i = 0; i < kWarmup; ++i) (void)run_workload_gpu(engine, chunks);
    std::vector<uint64_t> samples; samples.reserve(kIters);
    for (int i = 0; i < kIters; ++i) samples.push_back(run_workload_gpu(engine, chunks));
    return samples;
}

// ---------- emission --------------------------------------------------------

uint64_t total_ceremonies(const std::vector<Workload>& chunks)
{
    uint64_t n = 0;
    for (const auto& w : chunks) n += w.ceremony_count;
    return n;
}

uint64_t total_contributions(const std::vector<Workload>& chunks)
{
    uint64_t n = 0;
    for (const auto& w : chunks) n += w.contribution_count;
    return n;
}

void emit_row(const char* scenario,
              const char* backend,
              uint64_t ceremonies,
              uint64_t contribs,
              const Stats& s,
              double cpu_mean_ns)
{
    double mean_ms = s.mean_ns / 1e6;
    double p50_ms  = s.p50_ns  / 1e6;
    double p95_ms  = s.p95_ns  / 1e6;
    double p99_ms  = s.p99_ns  / 1e6;
    double cer_per_sec = (s.mean_ns > 0) ? (double(ceremonies) * 1e9 / s.mean_ns) : 0.0;
    double cnt_per_sec = (s.mean_ns > 0) ? (double(contribs)   * 1e9 / s.mean_ns) : 0.0;
    double speedup = (cpu_mean_ns > 0 && s.mean_ns > 0)
        ? (cpu_mean_ns / s.mean_ns) : 0.0;
    std::printf("ROW\t%s\t%s\t%llu\t%llu\t%.3f\t%.3f\t%.3f\t%.3f\t%.1f\t%.1f\t%.2fx\n",
                scenario, backend,
                (unsigned long long)ceremonies,
                (unsigned long long)contribs,
                mean_ms, p50_ms, p95_ms, p99_ms,
                cer_per_sec, cnt_per_sec, speedup);
    std::fflush(stdout);
}

// ---------- top-level scenarios --------------------------------------------

struct Scenario {
    std::string name;
    WorkloadShape shape;
};

void run_scenario(const Scenario& sc,
                  MPCVMGPUEngine* metal,
                  MPCVMGPUEngine* wgpu)
{
    auto chunks = shape_to_chunks(sc.name, sc.shape);
    uint64_t cer = total_ceremonies(chunks);
    uint64_t cnt = total_contributions(chunks);
    std::printf("SCENARIO\t%s\tceremonies=%llu\tcontribs=%llu\tchunks=%zu\n",
                sc.name.c_str(),
                (unsigned long long)cer,
                (unsigned long long)cnt,
                chunks.size());
    std::fflush(stdout);

    auto cpu_samples = run_iters_cpu(chunks);
    auto cpu_stats = compute_stats(cpu_samples);
    emit_row(sc.name.c_str(), backend_name(Backend::CPU), cer, cnt, cpu_stats,
             cpu_stats.mean_ns);

    if (metal != nullptr) {
        auto s = compute_stats(run_iters_gpu(metal, chunks));
        emit_row(sc.name.c_str(), backend_name(Backend::Metal), cer, cnt, s,
                 cpu_stats.mean_ns);
    }
    if (wgpu != nullptr) {
        auto s = compute_stats(run_iters_gpu(wgpu, chunks));
        emit_row(sc.name.c_str(), backend_name(Backend::WGPU), cer, cnt, s,
                 cpu_stats.mean_ns);
    }
}

}  // namespace

int main()
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("MPCVM-BENCHMARK\tv0.61.1\n");

    auto metal = MPCVMGPUEngine::create();
    std::unique_ptr<MPCVMGPUEngine> wgpu;
#if defined(LUX_MPCVM_ENABLE_WGPU) && LUX_MPCVM_ENABLE_WGPU
    wgpu = create_mpcvm_wgpu_engine();
#endif

    std::printf("BACKEND\tcpu\tCPU reference (host)\n");
    if (metal) std::printf("BACKEND\tmetal\t%s\n", metal->device_name());
    else       std::printf("BACKEND\tmetal\t(unavailable)\n");
    if (wgpu)  std::printf("BACKEND\twgpu\t%s\n", wgpu->device_name());
    else       std::printf("BACKEND\twgpu\t(unavailable)\n");

    std::printf("HEADER\tscenario\tbackend\tceremonies\tcontribs\tmean_ms\tp50_ms\tp95_ms\tp99_ms\tcer_per_s\tcnt_per_s\tspeedup\n");

    // ---------- per-scheme microbenchmarks --------------------------------
    //
    // Each microbench runs many ceremonies of a single scheme so the
    // per-scheme cost is exposed without cross-scheme dilution.

    std::vector<Scenario> micro = {
        { "micro.frost_keygen",   { 100, 0, 0, 0, 0 } },
        { "micro.frost_sign",     {   0, 100, 0, 0, 0 } },
        { "micro.cggmp21_keygen", {   0, 0, 100, 0, 0 } },
        { "micro.cggmp21_sign",   {   0, 0, 0, 100, 0 } },
        { "micro.ringtail_dkg",   {   0, 0, 0, 0, 100 } },
    };

    // ---------- mixed-workload macrobenchmarks ---------------------------

    std::vector<Scenario> macro = {
        // small: 10 × FROST keygen (single chunk, single round)
        { "small",
          { /*frost_keygen=*/10,    0, 0, 0, 0 } },
        // medium: 100 × FROST keygen + 100 × CGGMP21 keygen
        { "medium",
          { /*frost_keygen=*/100,   0, /*cggmp21_keygen=*/100, 0, 0 } },
        // large: 200 each of FROST kg, CGGMP21 kg, Ringtail DKG (sized to
        // saturate one round across the arena while leaving headroom).
        { "large",
          { /*frost_keygen=*/200,   0, /*cggmp21_keygen=*/200, 0, /*ringtail_dkg=*/200 } },
        // xlarge: 1000 mixed ceremonies (balanced across schemes; 5 chunks)
        { "xlarge",
          { /*frost_keygen=*/250, /*frost_sign=*/250,
            /*cggmp21_keygen=*/150, /*cggmp21_sign=*/150,
            /*ringtail_dkg=*/200 } },
    };

    for (const auto& sc : micro) run_scenario(sc, metal.get(), wgpu.get());
    for (const auto& sc : macro) run_scenario(sc, metal.get(), wgpu.get());

    std::printf("DONE\n");
    return 0;
}
