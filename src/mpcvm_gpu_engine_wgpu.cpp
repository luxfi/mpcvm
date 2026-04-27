// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_gpu_engine_wgpu.cpp — WebGPU/wgpu-native driver for MPCVMGPUEngine.
//
// One round = two sequential dispatches:
//   1. mpcvm_ceremony_step  (begin/cancel + contributions + sweep + share emit)
//   2. mpcvm_transition     (root composition + epoch advance)
//
// Both kernels are workgroup_size(1) — canonical in-order traversal that
// matches CPU/Metal/CUDA byte-for-byte. Determinism contract is covered
// by mpcvm_determinism_test.cpp.
//
// When LUX_MPCVM_ENABLE_WGPU is OFF (default outside this module), the
// MPCVMGPUEngine::create() weak fallback in this TU returns nullptr and the
// determinism test runs CPU-only.

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

// Synchronous adapter request via spinning instance event-pump.
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

// Read kernel source files relative to this TU.
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
            // Concatenate: common header then per-kernel modules. The
            // WGSL files reference struct names defined only in common.
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
    WGPUBuffer applied_counts_buf  = nullptr;  // 5 u32: cer, cnt, advances, finalized, failed
    WGPUBuffer counter_init_buf    = nullptr;  // 4 u32: next_cont_lo/hi, next_share_lo/hi

    // Readback staging for buffers we need to inspect.
    WGPUBuffer applied_counts_staging = nullptr;
    WGPUBuffer result_staging         = nullptr;

    // Host staging for ops (one allocation per buffer; written via QueueWriteBuffer).
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
    rel(r.applied_counts_staging);
    rel(r.result_staging);
}

class MPCVMGPUEngineWgpu final : public MPCVMGPUEngine {
public:
    MPCVMGPUEngineWgpu(WGPUInstance instance, WGPUAdapter adapter, WGPUDevice device,
                       WGPUQueue queue, WGPUShaderModule mod_cer, WGPUShaderModule mod_tr,
                       WGPUComputePipeline pso_cer, WGPUComputePipeline pso_tr,
                       std::string device_name)
        : instance_(instance), adapter_(adapter), device_(device), queue_(queue),
          mod_cer_(mod_cer), mod_tr_(mod_tr),
          pso_cer_(pso_cer), pso_tr_(pso_tr),
          device_name_(std::move(device_name)) {}

    ~MPCVMGPUEngineWgpu() override {
        if (round_active()) {
            std::lock_guard<std::mutex> g(mu_);
            release_round(round_);
            round_.handle = MPCVMRoundHandle{};
        }
        if (pso_cer_) wgpuComputePipelineRelease(pso_cer_);
        if (pso_tr_)  wgpuComputePipelineRelease(pso_tr_);
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
            || !round_.applied_counts_staging || !round_.result_staging) {
            release_round(round_);
            return MPCVMRoundHandle{0};
        }

        // Zero-init all working buffers (Storage + CopyDst).
        zero_buffer(round_.ceremonies_buf,    sizeof(Ceremony) * kCeremonySlots);
        zero_buffer(round_.key_shares_buf,    sizeof(KeyShare) * kKeyShareSlots);
        zero_buffer(round_.contributions_buf, sizeof(Contribution) * kContributionSlots);
        zero_buffer(round_.state_buf,         sizeof(MPCVMState));
        zero_buffer(round_.result_buf,        sizeof(MPCVMTransitionResult));
        zero_buffer(round_.applied_counts_buf, sizeof(uint32_t) * 5);

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

        // Upload descriptor + op streams + counters.
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

        // -- Bind groups --
        WGPUBindGroupLayout layout_cer = wgpuComputePipelineGetBindGroupLayout(pso_cer_, 0);
        WGPUBindGroupLayout layout_tr  = wgpuComputePipelineGetBindGroupLayout(pso_tr_, 0);

        WGPUBindGroupEntry e_cer[8]{};
        e_cer[0].binding = 0; e_cer[0].buffer = round_.desc_buf;             e_cer[0].size = sizeof(MPCVMRoundDescriptor);
        e_cer[1].binding = 1; e_cer[1].buffer = round_.ceremony_ops_buf;     e_cer[1].size = sizeof(CeremonyOp) * kMaxOpsPerRound;
        e_cer[2].binding = 2; e_cer[2].buffer = round_.contribution_ops_buf; e_cer[2].size = sizeof(ContributionOp) * kMaxOpsPerRound;
        e_cer[3].binding = 3; e_cer[3].buffer = round_.ceremonies_buf;       e_cer[3].size = sizeof(Ceremony) * kCeremonySlots;
        e_cer[4].binding = 4; e_cer[4].buffer = round_.key_shares_buf;       e_cer[4].size = sizeof(KeyShare) * kKeyShareSlots;
        e_cer[5].binding = 5; e_cer[5].buffer = round_.contributions_buf;    e_cer[5].size = sizeof(Contribution) * kContributionSlots;
        e_cer[6].binding = 6; e_cer[6].buffer = round_.applied_counts_buf;   e_cer[6].size = sizeof(uint32_t) * 5;
        e_cer[7].binding = 7; e_cer[7].buffer = round_.counter_init_buf;     e_cer[7].size = sizeof(uint32_t) * 4;

        WGPUBindGroupDescriptor bg_cer_d{};
        bg_cer_d.label = mk_sv("mpcvm.bg.cer");
        bg_cer_d.layout = layout_cer;
        bg_cer_d.entryCount = 8;
        bg_cer_d.entries = e_cer;
        WGPUBindGroup bg_cer = wgpuDeviceCreateBindGroup(device_, &bg_cer_d);

        WGPUBindGroupEntry e_tr[6]{};
        e_tr[0].binding = 0; e_tr[0].buffer = round_.desc_buf;          e_tr[0].size = sizeof(MPCVMRoundDescriptor);
        e_tr[1].binding = 1; e_tr[1].buffer = round_.ceremonies_buf;    e_tr[1].size = sizeof(Ceremony) * kCeremonySlots;
        e_tr[2].binding = 2; e_tr[2].buffer = round_.key_shares_buf;    e_tr[2].size = sizeof(KeyShare) * kKeyShareSlots;
        e_tr[3].binding = 3; e_tr[3].buffer = round_.contributions_buf; e_tr[3].size = sizeof(Contribution) * kContributionSlots;
        e_tr[4].binding = 4; e_tr[4].buffer = round_.state_buf;         e_tr[4].size = sizeof(MPCVMState);
        e_tr[5].binding = 5; e_tr[5].buffer = round_.result_buf;        e_tr[5].size = sizeof(MPCVMTransitionResult);

        WGPUBindGroupDescriptor bg_tr_d{};
        bg_tr_d.label = mk_sv("mpcvm.bg.tr");
        bg_tr_d.layout = layout_tr;
        bg_tr_d.entryCount = 6;
        bg_tr_d.entries = e_tr;
        WGPUBindGroup bg_tr = wgpuDeviceCreateBindGroup(device_, &bg_tr_d);

        // -- Encode dispatches --
        WGPUCommandEncoderDescriptor enc_d{};
        enc_d.label = mk_sv("mpcvm.enc");
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, &enc_d);

        {
            WGPUComputePassDescriptor pd{};
            pd.label = mk_sv("mpcvm.cer.pass");
            WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(enc, &pd);
            wgpuComputePassEncoderSetPipeline(p, pso_cer_);
            wgpuComputePassEncoderSetBindGroup(p, 0, bg_cer, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(p, 1, 1, 1);
            wgpuComputePassEncoderEnd(p);
            wgpuComputePassEncoderRelease(p);
        }
        {
            WGPUComputePassDescriptor pd{};
            pd.label = mk_sv("mpcvm.tr.pass");
            WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(enc, &pd);
            wgpuComputePassEncoderSetPipeline(p, pso_tr_);
            wgpuComputePassEncoderSetBindGroup(p, 0, bg_tr, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(p, 1, 1, 1);
            wgpuComputePassEncoderEnd(p);
            wgpuComputePassEncoderRelease(p);
        }

        // Copy result + applied_counts to staging buffers for readback.
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
        wgpuBindGroupRelease(bg_cer);
        wgpuBindGroupRelease(bg_tr);
        wgpuBindGroupLayoutRelease(layout_cer);
        wgpuBindGroupLayoutRelease(layout_tr);

        // Drain async queue work — wgpu-native equivalent of Dawn device.Tick().
        wgpuDevicePoll(device_, /*wait=*/true, nullptr);

        // Map applied counts and result.
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

        // The shader fills root fields + counts. The applied/advance/finalized/failed
        // counts come back via the dedicated buffer.
        result.ceremony_apply_count    = applied[0];
        result.contribution_apply_count = applied[1];
        result.round_advance_count      = applied[2];
        result.finalized_this_round     = applied[3];
        result.failed_this_round        = applied[4];

        // Bump host-side counters by what the shader applied.
        round_.next_contribution_id += applied[1];
        // share IDs: shader writes share_id into the table; host id is monotonic
        // and not strictly continuous within a session — same contract as Metal.

        return result;
    }

    MPCVMTransitionResult poll_round_result(MPCVMRoundHandle h) const override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle_const(h)) return MPCVMTransitionResult{};
        // We don't keep a cached result; re-read via run_until_done is the
        // path. For symmetry with Metal, return a zero result here.
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
        // Host-side zero buffer; uploaded once per round init.
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
        // Poll until the map callback fires.
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
    WGPUComputePipeline pso_cer_;
    WGPUComputePipeline pso_tr_;
    std::string device_name_;

    Round round_;
    uint64_t next_handle_ = 0;
    mutable std::mutex mu_;
};

}  // namespace

// The wgpu-native engine factory. Always available (under
// LUX_MPCVM_ENABLE_WGPU) regardless of Metal/CUDA presence — that lets
// the determinism harness exercise WGSL alongside the platform-canonical
// driver for true 4-way comparison.
std::unique_ptr<MPCVMGPUEngine> create_mpcvm_wgpu_engine() {
    // 1. Instance.
    WGPUInstanceDescriptor idesc{};
    WGPUInstance instance = wgpuCreateInstance(&idesc);
    if (!instance) {
        std::fprintf(stderr, "mpcvm wgpu: wgpuCreateInstance returned null\n");
        return nullptr;
    }

    // 2. Adapter (sync via spin).
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

    // 3. Device (sync via spin).
    DeviceAwait dw{};
    WGPUDeviceDescriptor ddesc{};
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

    // 4. WGSL sources.
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

    WGPUComputePipeline pso_cer = create_compute_pipeline(dw.device, mod_cer,
                                                           "mpcvm_ceremony_step", "mpcvm.cer.pso");
    WGPUComputePipeline pso_tr  = create_compute_pipeline(dw.device, mod_tr,
                                                           "mpcvm_transition", "mpcvm.tr.pso");
    if (!pso_cer || !pso_tr) {
        std::fprintf(stderr, "mpcvm wgpu: compute pipeline create failed\n");
        if (pso_cer) wgpuComputePipelineRelease(pso_cer);
        if (pso_tr)  wgpuComputePipelineRelease(pso_tr);
        wgpuShaderModuleRelease(mod_cer);
        wgpuShaderModuleRelease(mod_tr);
        wgpuQueueRelease(queue);
        wgpuDeviceRelease(dw.device);
        wgpuAdapterRelease(aw.adapter);
        wgpuInstanceRelease(instance);
        return nullptr;
    }

    // Adapter info (best-effort device name).
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
                                mod_cer, mod_tr, pso_cer, pso_tr, name));
}

// On non-Metal / non-CUDA platforms the wgpu engine is the canonical
// MPCVMGPUEngine::create() path. On Apple/Linux+CUDA it's a sibling
// factory used by the test harness for cross-backend equivalence.
#if !defined(__APPLE__) && !defined(LUX_MPCVM_HAVE_CUDA)
std::unique_ptr<MPCVMGPUEngine> MPCVMGPUEngine::create() {
    return create_mpcvm_wgpu_engine();
}
#endif

#else  // !LUX_MPCVM_ENABLE_WGPU

// Without wgpu-native linked, this TU still provides a no-op factory hook
// (weak on Apple/CUDA so the canonical platform driver wins; strong
// otherwise so the linker has a definition for tests).
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
