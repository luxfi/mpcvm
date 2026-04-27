// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_gpu_engine_wgpu.cpp — WebGPU/wgpu-native driver for MPCVMGPUEngine.
//
// v0.62: per-slot fan-out + parallel leaf reduction (mirrors Metal split).
// One round = four sequential dispatches:
//   1. mpcvm_ceremony_apply  (1x1x1) — Phase 1+2 ops apply, serial
//   2. mpcvm_ceremony_sweep  (kCeremonySlots threads, 1 workgroup) —
//                              Phase 3 sweep with intra-workgroup
//                              prefix-sum for share_id allocation
//   3. mpcvm_compute_leaves  (max(slot counts) threads, 64-thread groups) —
//                              parallel keccak of leaves
//   4. mpcvm_compose_root    (1x1x1) — serial fold + state-root composition
//
// All entry points are byte-equal to the Metal peer; determinism is
// covered by mpcvm_determinism_test.cpp.

#include "lux/mpcvm/mpcvm_gpu_engine.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#if defined(LUX_MPCVM_ENABLE_WGPU)
#  include <webgpu.h>
#  include <wgpu.h>
#endif

namespace mpcvm::gpu {

#if defined(LUX_MPCVM_ENABLE_WGPU)

namespace {

constexpr uint32_t kCeremonySlots     = kDefaultCeremonySlots;
constexpr uint32_t kKeyShareSlots     = kDefaultKeyShareSlots;
constexpr uint32_t kContributionSlots = kDefaultContributionSlots;
constexpr uint32_t kMaxOpsPerRound    = 4096u;
constexpr uint32_t kSweepThreads      = kCeremonySlots;
constexpr uint32_t kLeafThreads       = kContributionSlots;  // ≥ all other slot counts

WGPUStringView mk_sv(const char* s) {
    WGPUStringView v{};
    v.data = s;
    v.length = (s == nullptr) ? 0u : std::strlen(s);
    return v;
}

WGPUStringView mk_sv(const std::string& s) {
    WGPUStringView v{};
    v.data = s.c_str();
    v.length = s.size();
    return v;
}

struct AdapterAwait {
    WGPUAdapter adapter = nullptr;
    bool done = false;
    std::string err;
};

void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                WGPUStringView message, void* userdata1, void* /*ud2*/) {
    auto* a = static_cast<AdapterAwait*>(userdata1);
    a->adapter = (status == WGPURequestAdapterStatus_Success) ? adapter : nullptr;
    if (status != WGPURequestAdapterStatus_Success && message.data && message.length) {
        a->err.assign(message.data, message.length);
    }
    a->done = true;
}

struct DeviceAwait {
    WGPUDevice device = nullptr;
    bool done = false;
    std::string err;
};

void on_device(WGPURequestDeviceStatus status, WGPUDevice device,
               WGPUStringView message, void* userdata1, void* /*ud2*/) {
    auto* a = static_cast<DeviceAwait*>(userdata1);
    a->device = (status == WGPURequestDeviceStatus_Success) ? device : nullptr;
    if (status != WGPURequestDeviceStatus_Success && message.data && message.length) {
        a->err.assign(message.data, message.length);
    }
    a->done = true;
}

struct MapAwait {
    bool done = false;
    bool ok = false;
};

void on_map(WGPUMapAsyncStatus status, WGPUStringView /*msg*/,
            void* userdata1, void* /*ud2*/) {
    auto* a = static_cast<MapAwait*>(userdata1);
    a->ok   = (status == WGPUMapAsyncStatus_Success);
    a->done = true;
}

bool read_file(const std::filesystem::path& p, std::string& out) {
    std::ifstream f(p);
    if (!f.is_open()) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return !out.empty();
}

bool load_wgsl_sources(std::string& ceremony_src, std::string& transition_src) {
    std::filesystem::path here = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path candidates_dir[] = {
        here,
        std::filesystem::current_path(),
        std::filesystem::current_path() / "src",
        std::filesystem::current_path() / "mpcvm" / "src",
        std::filesystem::current_path().parent_path() / "src",
    };
    std::string common, kcer, ktr;
    for (const auto& dir : candidates_dir) {
        if (read_file(dir / "mpcvm_kernels_common.wgsl", common)
            && read_file(dir / "mpcvm_ceremony.wgsl", kcer)
            && read_file(dir / "mpcvm_transition.wgsl", ktr)) {
            ceremony_src.clear();
            ceremony_src += common;
            ceremony_src += "\n";
            ceremony_src += kcer;

            transition_src.clear();
            transition_src += common;
            transition_src += "\n";
            transition_src += ktr;
            return true;
        }
    }
    return false;
}

WGPUShaderModule create_shader_module(WGPUDevice device, const std::string& src,
                                      const char* label) {
    WGPUShaderSourceWGSL wgsl{};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = mk_sv(src);

    WGPUShaderModuleDescriptor desc{};
    desc.nextInChain = &wgsl.chain;
    desc.label = mk_sv(label);

    return wgpuDeviceCreateShaderModule(device, &desc);
}

WGPUComputePipeline create_compute_pipeline(WGPUDevice device, WGPUShaderModule mod,
                                            const char* entry, const char* label) {
    WGPUComputePipelineDescriptor desc{};
    desc.label = mk_sv(label);
    desc.layout = nullptr;  // auto layout
    desc.compute.module = mod;
    desc.compute.entryPoint = mk_sv(entry);
    desc.compute.constantCount = 0;
    desc.compute.constants = nullptr;
    return wgpuDeviceCreateComputePipeline(device, &desc);
}

struct Round {
    MPCVMRoundHandle handle{};
    MPCVMRoundDescriptor desc{};
    uint64_t next_share_id = 1;
    uint64_t next_contribution_id = 1;

    WGPUBuffer desc_buf            = nullptr;
    WGPUBuffer ceremony_ops_buf    = nullptr;
    WGPUBuffer contribution_ops_buf= nullptr;
    WGPUBuffer ceremonies_buf      = nullptr;
    WGPUBuffer key_shares_buf      = nullptr;
    WGPUBuffer contributions_buf   = nullptr;
    WGPUBuffer state_buf           = nullptr;
    WGPUBuffer result_buf          = nullptr;
    WGPUBuffer applied_counts_buf  = nullptr;  // 5 u32 atomics: cer, cnt, advances, finalized, failed
    WGPUBuffer counter_init_buf    = nullptr;  // 4 u32

    // v0.62 leaf-fold scratch.
    WGPUBuffer ceremony_leaves_buf      = nullptr;  // ceremony_count * 8 u32
    WGPUBuffer share_leaves_buf         = nullptr;  // share_count * 8 u32
    WGPUBuffer contribution_leaves_buf  = nullptr;  // contribution_count * 8 u32
    WGPUBuffer ceremony_used_mask_buf   = nullptr;  // ceremony_count u32
    WGPUBuffer share_used_mask_buf      = nullptr;  // share_count u32
    WGPUBuffer contribution_used_mask_buf= nullptr; // contribution_count u32
    WGPUBuffer count_outs_buf           = nullptr;  // 4 atomic u32: active, finalized, failed, share_count

    WGPUBuffer applied_counts_staging = nullptr;
    WGPUBuffer result_staging         = nullptr;

    std::vector<CeremonyOp>     ceremony_ops_host;
    std::vector<ContributionOp> contribution_ops_host;
};

void release_round(Round& r) {
    auto rel = [](WGPUBuffer& b) {
        if (b) { wgpuBufferRelease(b); b = nullptr; }
    };
    rel(r.desc_buf);
    rel(r.ceremony_ops_buf);
    rel(r.contribution_ops_buf);
    rel(r.ceremonies_buf);
    rel(r.key_shares_buf);
    rel(r.contributions_buf);
    rel(r.state_buf);
    rel(r.result_buf);
    rel(r.applied_counts_buf);
    rel(r.counter_init_buf);
    rel(r.ceremony_leaves_buf);
    rel(r.share_leaves_buf);
    rel(r.contribution_leaves_buf);
    rel(r.ceremony_used_mask_buf);
    rel(r.share_used_mask_buf);
    rel(r.contribution_used_mask_buf);
    rel(r.count_outs_buf);
    rel(r.applied_counts_staging);
    rel(r.result_staging);
}

class MPCVMGPUEngineWgpu final : public MPCVMGPUEngine {
public:
    MPCVMGPUEngineWgpu(WGPUInstance instance, WGPUAdapter adapter, WGPUDevice device,
                       WGPUQueue queue,
                       WGPUShaderModule mod_cer, WGPUShaderModule mod_tr,
                       WGPUComputePipeline pso_apply, WGPUComputePipeline pso_sweep,
                       WGPUComputePipeline pso_leaves, WGPUComputePipeline pso_compose,
                       std::string device_name)
        : instance_(instance), adapter_(adapter), device_(device), queue_(queue),
          mod_cer_(mod_cer), mod_tr_(mod_tr),
          pso_apply_(pso_apply), pso_sweep_(pso_sweep),
          pso_leaves_(pso_leaves), pso_compose_(pso_compose),
          device_name_(std::move(device_name)) {}

    ~MPCVMGPUEngineWgpu() override {
        if (round_active()) {
            std::lock_guard<std::mutex> g(mu_);
            release_round(round_);
            round_.handle = MPCVMRoundHandle{};
        }
        if (pso_apply_)   wgpuComputePipelineRelease(pso_apply_);
        if (pso_sweep_)   wgpuComputePipelineRelease(pso_sweep_);
        if (pso_leaves_)  wgpuComputePipelineRelease(pso_leaves_);
        if (pso_compose_) wgpuComputePipelineRelease(pso_compose_);
        if (mod_cer_) wgpuShaderModuleRelease(mod_cer_);
        if (mod_tr_)  wgpuShaderModuleRelease(mod_tr_);
        if (queue_)   wgpuQueueRelease(queue_);
        if (device_)  wgpuDeviceRelease(device_);
        if (adapter_) wgpuAdapterRelease(adapter_);
        if (instance_) wgpuInstanceRelease(instance_);
    }

    const char* device_name() const override { return device_name_.c_str(); }
    bool round_active() const override { return round_.handle.valid(); }

    MPCVMRoundHandle begin_round(const MPCVMRoundDescriptor& desc) override {
        std::lock_guard<std::mutex> g(mu_);
        if (round_.handle.valid()) return MPCVMRoundHandle{0};

        round_ = Round{};
        round_.desc = desc;
        round_.desc.ceremony_op_count = 0;
        round_.desc.contribution_op_count = 0;
        round_.next_share_id = 1;
        round_.next_contribution_id = 1;

        auto mk_buf = [&](size_t size, WGPUBufferUsage usage, const char* label) -> WGPUBuffer {
            WGPUBufferDescriptor bd{};
            bd.label = mk_sv(label);
            bd.size = size;
            bd.usage = usage;
            bd.mappedAtCreation = false;
            return wgpuDeviceCreateBuffer(device_, &bd);
        };

        round_.desc_buf             = mk_buf(sizeof(MPCVMRoundDescriptor),
                                              WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                                              "mpcvm.desc");
        round_.ceremony_ops_buf     = mk_buf(sizeof(CeremonyOp) * kMaxOpsPerRound,
                                              WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                                              "mpcvm.cer_ops");
        round_.contribution_ops_buf = mk_buf(sizeof(ContributionOp) * kMaxOpsPerRound,
                                              WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                                              "mpcvm.cnt_ops");
        round_.ceremonies_buf       = mk_buf(sizeof(Ceremony) * kCeremonySlots,
                                              WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
                                              "mpcvm.ceremonies");
        round_.key_shares_buf       = mk_buf(sizeof(KeyShare) * kKeyShareSlots,
                                              WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
                                              "mpcvm.shares");
        round_.contributions_buf    = mk_buf(sizeof(Contribution) * kContributionSlots,
                                              WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
                                              "mpcvm.contributions");
        round_.state_buf            = mk_buf(sizeof(MPCVMState),
                                              WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
                                              "mpcvm.state");
        round_.result_buf           = mk_buf(sizeof(MPCVMTransitionResult),
                                              WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
                                              "mpcvm.result");
        round_.applied_counts_buf   = mk_buf(sizeof(uint32_t) * 5,
                                              WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
                                              "mpcvm.applied");
        round_.counter_init_buf     = mk_buf(sizeof(uint32_t) * 4,
                                              WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                                              "mpcvm.counter_init");

        round_.ceremony_leaves_buf      = mk_buf(sizeof(uint32_t) * 8 * kCeremonySlots,
                                                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                                                  "mpcvm.cer_leaves");
        round_.share_leaves_buf         = mk_buf(sizeof(uint32_t) * 8 * kKeyShareSlots,
                                                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                                                  "mpcvm.shr_leaves");
        round_.contribution_leaves_buf  = mk_buf(sizeof(uint32_t) * 8 * kContributionSlots,
                                                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                                                  "mpcvm.cnt_leaves");
        round_.ceremony_used_mask_buf   = mk_buf(sizeof(uint32_t) * kCeremonySlots,
                                                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                                                  "mpcvm.cer_mask");
        round_.share_used_mask_buf      = mk_buf(sizeof(uint32_t) * kKeyShareSlots,
                                                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                                                  "mpcvm.shr_mask");
        round_.contribution_used_mask_buf= mk_buf(sizeof(uint32_t) * kContributionSlots,
                                                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                                                  "mpcvm.cnt_mask");
        round_.count_outs_buf           = mk_buf(sizeof(uint32_t) * 4,
                                                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                                                  "mpcvm.count_outs");

        round_.applied_counts_staging = mk_buf(sizeof(uint32_t) * 5,
                                                WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
                                                "mpcvm.applied.stg");
        round_.result_staging         = mk_buf(sizeof(MPCVMTransitionResult),
                                                WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
                                                "mpcvm.result.stg");

        if (!round_.desc_buf || !round_.ceremony_ops_buf || !round_.contribution_ops_buf
            || !round_.ceremonies_buf || !round_.key_shares_buf || !round_.contributions_buf
            || !round_.state_buf || !round_.result_buf
            || !round_.applied_counts_buf || !round_.counter_init_buf
            || !round_.ceremony_leaves_buf || !round_.share_leaves_buf
            || !round_.contribution_leaves_buf
            || !round_.ceremony_used_mask_buf || !round_.share_used_mask_buf
            || !round_.contribution_used_mask_buf
            || !round_.count_outs_buf
            || !round_.applied_counts_staging || !round_.result_staging) {
            release_round(round_);
            return MPCVMRoundHandle{0};
        }

        zero_buffer(round_.ceremonies_buf,    sizeof(Ceremony) * kCeremonySlots);
        zero_buffer(round_.key_shares_buf,    sizeof(KeyShare) * kKeyShareSlots);
        zero_buffer(round_.contributions_buf, sizeof(Contribution) * kContributionSlots);
        zero_buffer(round_.state_buf,         sizeof(MPCVMState));
        zero_buffer(round_.result_buf,        sizeof(MPCVMTransitionResult));
        zero_buffer(round_.applied_counts_buf, sizeof(uint32_t) * 5);
        zero_buffer(round_.ceremony_used_mask_buf,    sizeof(uint32_t) * kCeremonySlots);
        zero_buffer(round_.share_used_mask_buf,       sizeof(uint32_t) * kKeyShareSlots);
        zero_buffer(round_.contribution_used_mask_buf, sizeof(uint32_t) * kContributionSlots);
        zero_buffer(round_.count_outs_buf,            sizeof(uint32_t) * 4);

        round_.ceremony_ops_host.reserve(kMaxOpsPerRound);
        round_.contribution_ops_host.reserve(kMaxOpsPerRound);
        round_.handle = MPCVMRoundHandle{++next_handle_};
        return round_.handle;
    }

    void push_ceremony_ops(MPCVMRoundHandle h, std::span<const CeremonyOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        if (ops.empty()) return;
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.ceremony_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        for (uint32_t i = 0; i < take; ++i)
            round_.ceremony_ops_host.push_back(ops[i]);
        round_.desc.ceremony_op_count += take;
    }

    void push_contribution_ops(MPCVMRoundHandle h, std::span<const ContributionOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        if (ops.empty()) return;
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.contribution_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        for (uint32_t i = 0; i < take; ++i)
            round_.contribution_ops_host.push_back(ops[i]);
        round_.desc.contribution_op_count += take;
    }

    MPCVMTransitionResult run_epoch(MPCVMRoundHandle h) override {
        return run_until_done(h, 1);
    }

    MPCVMTransitionResult run_until_done(MPCVMRoundHandle h, std::size_t /*max*/) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return MPCVMTransitionResult{};

        wgpuQueueWriteBuffer(queue_, round_.desc_buf, 0, &round_.desc, sizeof(round_.desc));
        if (round_.desc.ceremony_op_count > 0) {
            wgpuQueueWriteBuffer(queue_, round_.ceremony_ops_buf, 0,
                                  round_.ceremony_ops_host.data(),
                                  sizeof(CeremonyOp) * round_.desc.ceremony_op_count);
        }
        if (round_.desc.contribution_op_count > 0) {
            wgpuQueueWriteBuffer(queue_, round_.contribution_ops_buf, 0,
                                  round_.contribution_ops_host.data(),
                                  sizeof(ContributionOp) * round_.desc.contribution_op_count);
        }
        uint32_t counter_init[4] = {
            static_cast<uint32_t>(round_.next_contribution_id & 0xFFFFFFFFu),
            static_cast<uint32_t>(round_.next_contribution_id >> 32),
            static_cast<uint32_t>(round_.next_share_id & 0xFFFFFFFFu),
            static_cast<uint32_t>(round_.next_share_id >> 32),
        };
        wgpuQueueWriteBuffer(queue_, round_.counter_init_buf, 0, counter_init, sizeof(counter_init));

        // Bind group layouts come from the auto-layout PSOs.
        WGPUBindGroupLayout layout_apply   = wgpuComputePipelineGetBindGroupLayout(pso_apply_, 0);
        WGPUBindGroupLayout layout_sweep   = wgpuComputePipelineGetBindGroupLayout(pso_sweep_, 0);
        WGPUBindGroupLayout layout_leaves  = wgpuComputePipelineGetBindGroupLayout(pso_leaves_, 0);
        WGPUBindGroupLayout layout_compose = wgpuComputePipelineGetBindGroupLayout(pso_compose_, 0);

        // -- Bind groups --
        // ceremony_apply uses bindings 0..3, 5..7 (skips 4 = key_shares which
        // it doesn't touch). The auto-derived layout for apply has 7 entries.
        WGPUBindGroupEntry e_apply[7]{};
        e_apply[0].binding = 0; e_apply[0].buffer = round_.desc_buf;             e_apply[0].size = sizeof(MPCVMRoundDescriptor);
        e_apply[1].binding = 1; e_apply[1].buffer = round_.ceremony_ops_buf;     e_apply[1].size = sizeof(CeremonyOp) * kMaxOpsPerRound;
        e_apply[2].binding = 2; e_apply[2].buffer = round_.contribution_ops_buf; e_apply[2].size = sizeof(ContributionOp) * kMaxOpsPerRound;
        e_apply[3].binding = 3; e_apply[3].buffer = round_.ceremonies_buf;       e_apply[3].size = sizeof(Ceremony) * kCeremonySlots;
        e_apply[4].binding = 5; e_apply[4].buffer = round_.contributions_buf;    e_apply[4].size = sizeof(Contribution) * kContributionSlots;
        e_apply[5].binding = 6; e_apply[5].buffer = round_.applied_counts_buf;   e_apply[5].size = sizeof(uint32_t) * 5;
        e_apply[6].binding = 7; e_apply[6].buffer = round_.counter_init_buf;     e_apply[6].size = sizeof(uint32_t) * 4;
        WGPUBindGroupDescriptor bg_apply_d{};
        bg_apply_d.label = mk_sv("mpcvm.bg.apply");
        bg_apply_d.layout = layout_apply;
        bg_apply_d.entryCount = 7;
        bg_apply_d.entries = e_apply;
        WGPUBindGroup bg_apply = wgpuDeviceCreateBindGroup(device_, &bg_apply_d);

        // ceremony_sweep uses bindings 0, 3, 4, 5, 6, 7 — skips ceremony_ops
        // (1) and contribution_ops (2) which only the apply kernel reads.
        WGPUBindGroup bg_sweep = nullptr;
        {
            WGPUBindGroupEntry e_sweep[6]{};
            e_sweep[0].binding = 0; e_sweep[0].buffer = round_.desc_buf;          e_sweep[0].size = sizeof(MPCVMRoundDescriptor);
            e_sweep[1].binding = 3; e_sweep[1].buffer = round_.ceremonies_buf;    e_sweep[1].size = sizeof(Ceremony) * kCeremonySlots;
            e_sweep[2].binding = 4; e_sweep[2].buffer = round_.key_shares_buf;    e_sweep[2].size = sizeof(KeyShare) * kKeyShareSlots;
            e_sweep[3].binding = 5; e_sweep[3].buffer = round_.contributions_buf; e_sweep[3].size = sizeof(Contribution) * kContributionSlots;
            e_sweep[4].binding = 6; e_sweep[4].buffer = round_.applied_counts_buf; e_sweep[4].size = sizeof(uint32_t) * 5;
            e_sweep[5].binding = 7; e_sweep[5].buffer = round_.counter_init_buf;  e_sweep[5].size = sizeof(uint32_t) * 4;
            WGPUBindGroupDescriptor d{};
            d.label = mk_sv("mpcvm.bg.sweep");
            d.layout = layout_sweep;
            d.entryCount = 6;
            d.entries = e_sweep;
            bg_sweep = wgpuDeviceCreateBindGroup(device_, &d);
        }

        // compute_leaves uses bindings 1..10 (ceremonies, shares,
        // contributions, leaves x3, masks x3, count_outs).
        WGPUBindGroupEntry e_leaves[10]{};
        e_leaves[0].binding = 1;  e_leaves[0].buffer = round_.ceremonies_buf;    e_leaves[0].size = sizeof(Ceremony) * kCeremonySlots;
        e_leaves[1].binding = 2;  e_leaves[1].buffer = round_.key_shares_buf;    e_leaves[1].size = sizeof(KeyShare) * kKeyShareSlots;
        e_leaves[2].binding = 3;  e_leaves[2].buffer = round_.contributions_buf; e_leaves[2].size = sizeof(Contribution) * kContributionSlots;
        e_leaves[3].binding = 4;  e_leaves[3].buffer = round_.ceremony_leaves_buf;     e_leaves[3].size = sizeof(uint32_t) * 8 * kCeremonySlots;
        e_leaves[4].binding = 5;  e_leaves[4].buffer = round_.share_leaves_buf;        e_leaves[4].size = sizeof(uint32_t) * 8 * kKeyShareSlots;
        e_leaves[5].binding = 6;  e_leaves[5].buffer = round_.contribution_leaves_buf; e_leaves[5].size = sizeof(uint32_t) * 8 * kContributionSlots;
        e_leaves[6].binding = 7;  e_leaves[6].buffer = round_.ceremony_used_mask_buf;     e_leaves[6].size = sizeof(uint32_t) * kCeremonySlots;
        e_leaves[7].binding = 8;  e_leaves[7].buffer = round_.share_used_mask_buf;        e_leaves[7].size = sizeof(uint32_t) * kKeyShareSlots;
        e_leaves[8].binding = 9;  e_leaves[8].buffer = round_.contribution_used_mask_buf; e_leaves[8].size = sizeof(uint32_t) * kContributionSlots;
        e_leaves[9].binding = 10; e_leaves[9].buffer = round_.count_outs_buf;            e_leaves[9].size = sizeof(uint32_t) * 4;

        WGPUBindGroupDescriptor bg_leaves_d{};
        bg_leaves_d.label = mk_sv("mpcvm.bg.leaves");
        bg_leaves_d.layout = layout_leaves;
        bg_leaves_d.entryCount = 10;
        bg_leaves_d.entries = e_leaves;
        WGPUBindGroup bg_leaves = wgpuDeviceCreateBindGroup(device_, &bg_leaves_d);

        // compose_root uses bindings 0 (desc), 4..10 (leaves+masks+counts),
        // 11 (state), 12 (result). It also references ceremonies/shares/
        // contributions for arrayLength so those are part of the auto-layout.
        WGPUBindGroupEntry e_compose[13]{};
        e_compose[0].binding = 0;  e_compose[0].buffer = round_.desc_buf;          e_compose[0].size = sizeof(MPCVMRoundDescriptor);
        e_compose[1].binding = 1;  e_compose[1].buffer = round_.ceremonies_buf;    e_compose[1].size = sizeof(Ceremony) * kCeremonySlots;
        e_compose[2].binding = 2;  e_compose[2].buffer = round_.key_shares_buf;    e_compose[2].size = sizeof(KeyShare) * kKeyShareSlots;
        e_compose[3].binding = 3;  e_compose[3].buffer = round_.contributions_buf; e_compose[3].size = sizeof(Contribution) * kContributionSlots;
        e_compose[4].binding = 4;  e_compose[4].buffer = round_.ceremony_leaves_buf;     e_compose[4].size = sizeof(uint32_t) * 8 * kCeremonySlots;
        e_compose[5].binding = 5;  e_compose[5].buffer = round_.share_leaves_buf;        e_compose[5].size = sizeof(uint32_t) * 8 * kKeyShareSlots;
        e_compose[6].binding = 6;  e_compose[6].buffer = round_.contribution_leaves_buf; e_compose[6].size = sizeof(uint32_t) * 8 * kContributionSlots;
        e_compose[7].binding = 7;  e_compose[7].buffer = round_.ceremony_used_mask_buf;     e_compose[7].size = sizeof(uint32_t) * kCeremonySlots;
        e_compose[8].binding = 8;  e_compose[8].buffer = round_.share_used_mask_buf;        e_compose[8].size = sizeof(uint32_t) * kKeyShareSlots;
        e_compose[9].binding = 9;  e_compose[9].buffer = round_.contribution_used_mask_buf; e_compose[9].size = sizeof(uint32_t) * kContributionSlots;
        e_compose[10].binding = 10; e_compose[10].buffer = round_.count_outs_buf;            e_compose[10].size = sizeof(uint32_t) * 4;
        e_compose[11].binding = 11; e_compose[11].buffer = round_.state_buf;                 e_compose[11].size = sizeof(MPCVMState);
        e_compose[12].binding = 12; e_compose[12].buffer = round_.result_buf;                e_compose[12].size = sizeof(MPCVMTransitionResult);

        WGPUBindGroupDescriptor bg_compose_d{};
        bg_compose_d.label = mk_sv("mpcvm.bg.compose");
        bg_compose_d.layout = layout_compose;
        bg_compose_d.entryCount = 13;
        bg_compose_d.entries = e_compose;
        WGPUBindGroup bg_compose = wgpuDeviceCreateBindGroup(device_, &bg_compose_d);

        // -- Encode dispatches --
        WGPUCommandEncoderDescriptor enc_d{};
        enc_d.label = mk_sv("mpcvm.enc");
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, &enc_d);

        {
            WGPUComputePassDescriptor pd{};
            pd.label = mk_sv("mpcvm.apply.pass");
            WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(enc, &pd);
            wgpuComputePassEncoderSetPipeline(p, pso_apply_);
            wgpuComputePassEncoderSetBindGroup(p, 0, bg_apply, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(p, 1, 1, 1);
            wgpuComputePassEncoderEnd(p);
            wgpuComputePassEncoderRelease(p);
        }
        {
            WGPUComputePassDescriptor pd{};
            pd.label = mk_sv("mpcvm.sweep.pass");
            WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(enc, &pd);
            wgpuComputePassEncoderSetPipeline(p, pso_sweep_);
            wgpuComputePassEncoderSetBindGroup(p, 0, bg_sweep, 0, nullptr);
            // workgroup_size(256) == kSweepThreads; one workgroup covers all
            // ceremony slots.
            wgpuComputePassEncoderDispatchWorkgroups(p, 1, 1, 1);
            wgpuComputePassEncoderEnd(p);
            wgpuComputePassEncoderRelease(p);
        }
        {
            WGPUComputePassDescriptor pd{};
            pd.label = mk_sv("mpcvm.leaves.pass");
            WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(enc, &pd);
            wgpuComputePassEncoderSetPipeline(p, pso_leaves_);
            wgpuComputePassEncoderSetBindGroup(p, 0, bg_leaves, 0, nullptr);
            // workgroup_size(64) — dispatch ceil(kLeafThreads / 64) workgroups.
            uint32_t groups = (kLeafThreads + 63u) / 64u;
            wgpuComputePassEncoderDispatchWorkgroups(p, groups, 1, 1);
            wgpuComputePassEncoderEnd(p);
            wgpuComputePassEncoderRelease(p);
        }
        {
            WGPUComputePassDescriptor pd{};
            pd.label = mk_sv("mpcvm.compose.pass");
            WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(enc, &pd);
            wgpuComputePassEncoderSetPipeline(p, pso_compose_);
            wgpuComputePassEncoderSetBindGroup(p, 0, bg_compose, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(p, 1, 1, 1);
            wgpuComputePassEncoderEnd(p);
            wgpuComputePassEncoderRelease(p);
        }

        wgpuCommandEncoderCopyBufferToBuffer(enc, round_.applied_counts_buf, 0,
                                              round_.applied_counts_staging, 0,
                                              sizeof(uint32_t) * 5);
        wgpuCommandEncoderCopyBufferToBuffer(enc, round_.result_buf, 0,
                                              round_.result_staging, 0,
                                              sizeof(MPCVMTransitionResult));

        WGPUCommandBufferDescriptor cmd_d{};
        cmd_d.label = mk_sv("mpcvm.cmd");
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cmd_d);
        wgpuQueueSubmit(queue_, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        wgpuBindGroupRelease(bg_apply);
        wgpuBindGroupRelease(bg_sweep);
        wgpuBindGroupRelease(bg_leaves);
        wgpuBindGroupRelease(bg_compose);
        wgpuBindGroupLayoutRelease(layout_apply);
        wgpuBindGroupLayoutRelease(layout_sweep);
        wgpuBindGroupLayoutRelease(layout_leaves);
        wgpuBindGroupLayoutRelease(layout_compose);

        wgpuDevicePoll(device_, /*wait=*/true, nullptr);

        uint32_t applied[5]{};
        if (!read_buffer(round_.applied_counts_staging, applied, sizeof(applied))) {
            std::fprintf(stderr, "mpcvm wgpu: applied_counts readback failed\n");
            return MPCVMTransitionResult{};
        }
        MPCVMTransitionResult result{};
        if (!read_buffer(round_.result_staging, &result, sizeof(result))) {
            std::fprintf(stderr, "mpcvm wgpu: result readback failed\n");
            return MPCVMTransitionResult{};
        }

        result.ceremony_apply_count    = applied[0];
        result.contribution_apply_count = applied[1];
        result.round_advance_count      = applied[2];
        result.finalized_this_round     = applied[3];
        result.failed_this_round        = applied[4];

        round_.next_contribution_id += applied[1];
        return result;
    }

    MPCVMTransitionResult poll_round_result(MPCVMRoundHandle h) const override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle_const(h)) return MPCVMTransitionResult{};
        return MPCVMTransitionResult{};
    }

    void end_round(MPCVMRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        release_round(round_);
        round_.handle = MPCVMRoundHandle{};
    }

private:
    bool check_handle(MPCVMRoundHandle h) const {
        return h.valid() && h.opaque == round_.handle.opaque;
    }
    bool check_handle_const(MPCVMRoundHandle h) const { return check_handle(h); }

    void zero_buffer(WGPUBuffer buf, size_t size) {
        std::vector<uint8_t> zeros(size, 0);
        wgpuQueueWriteBuffer(queue_, buf, 0, zeros.data(), zeros.size());
    }

    bool read_buffer(WGPUBuffer staging, void* dst, size_t size) {
        MapAwait await{};
        WGPUBufferMapCallbackInfo cbinfo{};
        cbinfo.mode = WGPUCallbackMode_AllowProcessEvents;
        cbinfo.callback = on_map;
        cbinfo.userdata1 = &await;
        wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, size, cbinfo);
        while (!await.done) {
            wgpuDevicePoll(device_, /*wait=*/true, nullptr);
        }
        if (!await.ok) return false;
        const void* src = wgpuBufferGetConstMappedRange(staging, 0, size);
        if (!src) return false;
        std::memcpy(dst, src, size);
        wgpuBufferUnmap(staging);
        return true;
    }

    WGPUInstance instance_;
    WGPUAdapter  adapter_;
    WGPUDevice   device_;
    WGPUQueue    queue_;
    WGPUShaderModule mod_cer_;
    WGPUShaderModule mod_tr_;
    WGPUComputePipeline pso_apply_;
    WGPUComputePipeline pso_sweep_;
    WGPUComputePipeline pso_leaves_;
    WGPUComputePipeline pso_compose_;
    std::string device_name_;

    Round round_;
    uint64_t next_handle_ = 0;
    mutable std::mutex mu_;
};

}  // namespace

std::unique_ptr<MPCVMGPUEngine> create_mpcvm_wgpu_engine() {
    WGPUInstanceDescriptor idesc{};
    WGPUInstance instance = wgpuCreateInstance(&idesc);
    if (!instance) {
        std::fprintf(stderr, "mpcvm wgpu: wgpuCreateInstance returned null\n");
        return nullptr;
    }

    AdapterAwait aw{};
    WGPURequestAdapterOptions ropt{};
    WGPURequestAdapterCallbackInfo cb{};
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = on_adapter;
    cb.userdata1 = &aw;
    wgpuInstanceRequestAdapter(instance, &ropt, cb);
    while (!aw.done) {
        wgpuInstanceProcessEvents(instance);
    }
    if (!aw.adapter) {
        std::fprintf(stderr, "mpcvm wgpu: no adapter (%s)\n", aw.err.c_str());
        wgpuInstanceRelease(instance);
        return nullptr;
    }

    // Adapter's reported limits — request as much as the adapter supports
    // so the leaves+compose binding group (10 storage buffers) fits.
    WGPULimits adapter_limits{};
    wgpuAdapterGetLimits(aw.adapter, &adapter_limits);

    DeviceAwait dw{};
    WGPUDeviceDescriptor ddesc{};
    ddesc.requiredLimits = &adapter_limits;
    WGPURequestDeviceCallbackInfo dcb{};
    dcb.mode = WGPUCallbackMode_AllowProcessEvents;
    dcb.callback = on_device;
    dcb.userdata1 = &dw;
    wgpuAdapterRequestDevice(aw.adapter, &ddesc, dcb);
    while (!dw.done) {
        wgpuInstanceProcessEvents(instance);
    }
    if (!dw.device) {
        std::fprintf(stderr, "mpcvm wgpu: no device (%s)\n", dw.err.c_str());
        wgpuAdapterRelease(aw.adapter);
        wgpuInstanceRelease(instance);
        return nullptr;
    }

    WGPUQueue queue = wgpuDeviceGetQueue(dw.device);
    if (!queue) {
        std::fprintf(stderr, "mpcvm wgpu: no queue\n");
        wgpuDeviceRelease(dw.device);
        wgpuAdapterRelease(aw.adapter);
        wgpuInstanceRelease(instance);
        return nullptr;
    }

    std::string ceremony_src, transition_src;
    if (!load_wgsl_sources(ceremony_src, transition_src)) {
        std::fprintf(stderr, "mpcvm wgpu: WGSL sources not found near %s\n", __FILE__);
        wgpuQueueRelease(queue);
        wgpuDeviceRelease(dw.device);
        wgpuAdapterRelease(aw.adapter);
        wgpuInstanceRelease(instance);
        return nullptr;
    }

    WGPUShaderModule mod_cer = create_shader_module(dw.device, ceremony_src,  "mpcvm.cer");
    WGPUShaderModule mod_tr  = create_shader_module(dw.device, transition_src, "mpcvm.tr");
    if (!mod_cer || !mod_tr) {
        std::fprintf(stderr, "mpcvm wgpu: shader module create failed\n");
        if (mod_cer) wgpuShaderModuleRelease(mod_cer);
        if (mod_tr)  wgpuShaderModuleRelease(mod_tr);
        wgpuQueueRelease(queue);
        wgpuDeviceRelease(dw.device);
        wgpuAdapterRelease(aw.adapter);
        wgpuInstanceRelease(instance);
        return nullptr;
    }

    WGPUComputePipeline pso_apply   = create_compute_pipeline(dw.device, mod_cer,
                                                                "mpcvm_ceremony_apply", "mpcvm.apply.pso");
    WGPUComputePipeline pso_sweep   = create_compute_pipeline(dw.device, mod_cer,
                                                                "mpcvm_ceremony_sweep", "mpcvm.sweep.pso");
    WGPUComputePipeline pso_leaves  = create_compute_pipeline(dw.device, mod_tr,
                                                                "mpcvm_compute_leaves", "mpcvm.leaves.pso");
    WGPUComputePipeline pso_compose = create_compute_pipeline(dw.device, mod_tr,
                                                                "mpcvm_compose_root", "mpcvm.compose.pso");
    if (!pso_apply || !pso_sweep || !pso_leaves || !pso_compose) {
        std::fprintf(stderr, "mpcvm wgpu: compute pipeline create failed\n");
        if (pso_apply)   wgpuComputePipelineRelease(pso_apply);
        if (pso_sweep)   wgpuComputePipelineRelease(pso_sweep);
        if (pso_leaves)  wgpuComputePipelineRelease(pso_leaves);
        if (pso_compose) wgpuComputePipelineRelease(pso_compose);
        wgpuShaderModuleRelease(mod_cer);
        wgpuShaderModuleRelease(mod_tr);
        wgpuQueueRelease(queue);
        wgpuDeviceRelease(dw.device);
        wgpuAdapterRelease(aw.adapter);
        wgpuInstanceRelease(instance);
        return nullptr;
    }

    std::string name = "wgpu-native";
    WGPUAdapterInfo info{};
    if (wgpuAdapterGetInfo(aw.adapter, &info) == WGPUStatus_Success) {
        if (info.device.data && info.device.length) {
            name.assign(info.device.data, info.device.length);
        }
        wgpuAdapterInfoFreeMembers(info);
    }

    return std::unique_ptr<MPCVMGPUEngine>(
        new MPCVMGPUEngineWgpu(instance, aw.adapter, dw.device, queue,
                                mod_cer, mod_tr,
                                pso_apply, pso_sweep, pso_leaves, pso_compose,
                                name));
}

#if !defined(__APPLE__) && !defined(LUX_MPCVM_HAVE_CUDA)
std::unique_ptr<MPCVMGPUEngine> MPCVMGPUEngine::create() {
    return create_mpcvm_wgpu_engine();
}
#endif

#else  // !LUX_MPCVM_ENABLE_WGPU

#if !defined(__APPLE__) && !defined(LUX_MPCVM_HAVE_CUDA)
std::unique_ptr<MPCVMGPUEngine> MPCVMGPUEngine::create() {
    return nullptr;
}
#endif

std::unique_ptr<MPCVMGPUEngine> create_mpcvm_wgpu_engine() {
    return nullptr;
}

#endif  // LUX_MPCVM_ENABLE_WGPU

}  // namespace mpcvm::gpu
