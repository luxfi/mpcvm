// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_gpu_engine.mm — Metal-backed driver for MPCVMGPUEngine.
//
// v0.62: per-slot fan-out + parallel leaf reduction.
// One round = four kernel dispatches in canonical order:
//   1. mpcvm_ceremony_apply  (1x1x1) — Phase 1+2 ops apply, serial
//   2. mpcvm_ceremony_sweep  (kCeremonySlots threads, 1 threadgroup) —
//                              Phase 3 sweep with intra-threadgroup
//                              prefix-sum for share_id allocation
//   3. mpcvm_compute_leaves  (max(slot counts) threads) — parallel keccak
//                              of leaves into precomputed leaf-hash arenas
//   4. mpcvm_compose_root    (1x1x1) — serial fold + state-root composition
//
// Determinism contract matches the CPU reference (mpcvm_cpu_reference.cpp)
// byte-for-byte; per-slot fan-out preserves canonical ordering because the
// open-addressing hash places each (cid, round, holder) at a deterministic
// slot.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "lux/mpcvm/mpcvm_gpu_engine.hpp"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace mpcvm::gpu {

namespace {

id<MTLLibrary> load_mpcvm_metallib(id<MTLDevice> device)
{
    NSError* error = nil;
    std::filesystem::path here = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path candidates[] = {
        here / "mpcvm.metallib",
        std::filesystem::current_path() / "mpcvm.metallib",
        std::filesystem::current_path() / "src" / "mpcvm.metallib",
        std::filesystem::current_path() / "mpcvm" / "src" / "mpcvm.metallib",
        std::filesystem::current_path().parent_path() / "src" / "mpcvm.metallib",
    };
    for (const auto& p : candidates) {
        if (!std::filesystem::exists(p)) continue;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:p.c_str()]];
        id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
        if (lib) return lib;
        if (error)
            std::fprintf(stderr, "metallib load error %s: %s\n",
                         p.c_str(), [[error localizedDescription] UTF8String]);
    }
    return nil;
}

id<MTLLibrary> compile_mpcvm_library(id<MTLDevice> device)
{
    NSError* error = nil;
    std::filesystem::path here = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path candidates_dir[] = {
        here,
        std::filesystem::current_path(),
        std::filesystem::current_path() / "src",
        std::filesystem::current_path() / "mpcvm" / "src",
        std::filesystem::current_path().parent_path() / "src",
    };
    auto load_file = [&](const std::filesystem::path& p) -> NSString* {
        if (!std::filesystem::exists(p)) return nil;
        NSString* path = [NSString stringWithUTF8String:p.c_str()];
        return [NSString stringWithContentsOfFile:path
                                         encoding:NSUTF8StringEncoding
                                            error:&error];
    };
    NSString* common = nil;
    NSString* k_cer  = nil;
    NSString* k_tr   = nil;
    NSString* k_fr   = nil;
    NSString* k_cg   = nil;
    NSString* k_rt   = nil;
    for (const auto& dir : candidates_dir) {
        common = load_file(dir / "mpcvm_kernels_common.h.metal");
        k_cer  = load_file(dir / "mpcvm_ceremony.metal");
        k_tr   = load_file(dir / "mpcvm_transition.metal");
        k_fr   = load_file(dir / "mpcvm_frost.metal");
        k_cg   = load_file(dir / "mpcvm_cggmp21.metal");
        k_rt   = load_file(dir / "mpcvm_corona.metal");
        if (common && k_cer && k_tr && k_fr && k_cg && k_rt) break;
    }
    if (!common || !k_cer || !k_tr || !k_fr || !k_cg || !k_rt) {
        std::fprintf(stderr, "MPCVM Metal sources not found near %s\n",
                     here.c_str());
        return nil;
    }
    auto strip_include = [](NSString* src) -> NSString* {
        NSMutableString* out = [NSMutableString string];
        NSArray<NSString*>* lines = [src componentsSeparatedByString:@"\n"];
        for (NSString* line in lines) {
            if ([line containsString:@"mpcvm_kernels_common.h.metal"]) continue;
            [out appendString:line];
            [out appendString:@"\n"];
        }
        return out;
    };
    NSMutableString* combined = [NSMutableString string];
    [combined appendString:common];     [combined appendString:@"\n"];
    [combined appendString:strip_include(k_cer)]; [combined appendString:@"\n"];
    [combined appendString:strip_include(k_tr)];  [combined appendString:@"\n"];
    [combined appendString:strip_include(k_fr)];  [combined appendString:@"\n"];
    [combined appendString:strip_include(k_cg)];  [combined appendString:@"\n"];
    [combined appendString:strip_include(k_rt)];

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion3_0;
    id<MTLLibrary> lib = [device newLibraryWithSource:combined
                                              options:opts
                                                error:&error];
    if (!lib && error)
        std::fprintf(stderr, "MPCVM Metal compile error: %s\n",
                     [[error localizedDescription] UTF8String]);
    return lib;
}

constexpr uint32_t kCeremonySlots    = kDefaultCeremonySlots;
constexpr uint32_t kKeyShareSlots    = kDefaultKeyShareSlots;
constexpr uint32_t kContributionSlots= kDefaultContributionSlots;
constexpr uint32_t kMaxOpsPerRound   = 4096u;
constexpr uint32_t kSweepThreads     = kCeremonySlots;  // must match kSweepWorkgroupSize in mpcvm_ceremony.metal

// Maximum slot index covered by mpcvm_compute_leaves (each thread checks
// its tid against ceremony/share/contribution counts independently).
constexpr uint32_t kLeafThreads = (kContributionSlots > kKeyShareSlots
    ? kContributionSlots : kKeyShareSlots) > kCeremonySlots
        ? (kContributionSlots > kKeyShareSlots ? kContributionSlots : kKeyShareSlots)
        : kCeremonySlots;

struct Round {
    MPCVMRoundHandle handle{};
    MPCVMRoundDescriptor desc{};

    uint64_t next_share_id = 1;
    uint64_t next_contribution_id = 1;

    id<MTLBuffer> desc_buf            = nil;
    id<MTLBuffer> ceremony_ops_buf    = nil;
    id<MTLBuffer> contribution_ops_buf= nil;
    id<MTLBuffer> ceremonies_buf      = nil;
    id<MTLBuffer> key_shares_buf      = nil;
    id<MTLBuffer> contributions_buf   = nil;
    id<MTLBuffer> state_buf           = nil;
    id<MTLBuffer> result_buf          = nil;
    id<MTLBuffer> ceremony_applied_buf      = nil;
    id<MTLBuffer> contribution_applied_buf  = nil;
    id<MTLBuffer> round_advance_buf         = nil;
    id<MTLBuffer> finalized_buf             = nil;
    id<MTLBuffer> failed_buf                = nil;

    // v0.62: parallel-fold scratch.
    id<MTLBuffer> ceremony_leaf_hashes      = nil;
    id<MTLBuffer> share_leaf_hashes         = nil;
    id<MTLBuffer> contribution_leaf_hashes  = nil;
    id<MTLBuffer> ceremony_used_mask        = nil;
    id<MTLBuffer> share_used_mask           = nil;
    id<MTLBuffer> contribution_used_mask    = nil;
    id<MTLBuffer> active_count_buf          = nil;
    id<MTLBuffer> finalized_count_buf       = nil;
    id<MTLBuffer> failed_count_buf          = nil;
    id<MTLBuffer> share_count_buf           = nil;
};

class MPCVMGPUEngineMetal final : public MPCVMGPUEngine {
public:
    MPCVMGPUEngineMetal(id<MTLDevice> device,
                        id<MTLCommandQueue> queue,
                        id<MTLComputePipelineState> apply_pso,
                        id<MTLComputePipelineState> sweep_pso,
                        id<MTLComputePipelineState> leaves_pso,
                        id<MTLComputePipelineState> compose_pso,
                        NSString* device_name)
        : device_(device), queue_(queue),
          apply_pso_(apply_pso), sweep_pso_(sweep_pso),
          leaves_pso_(leaves_pso), compose_pso_(compose_pso),
          device_name_str_([device_name UTF8String]) {}

    ~MPCVMGPUEngineMetal() override {
        if (round_active()) end_round(round_.handle);
    }

    const char* device_name() const override { return device_name_str_.c_str(); }
    bool round_active() const override { return round_.handle.valid(); }

    MPCVMRoundHandle begin_round(const MPCVMRoundDescriptor& desc) override {
        std::lock_guard<std::mutex> g(mu_);
        if (round_.handle.valid()) return MPCVMRoundHandle{0};

        round_ = Round{};
        round_.desc = desc;

        round_.desc_buf             = [device_ newBufferWithLength:sizeof(MPCVMRoundDescriptor) options:MTLResourceStorageModeShared];
        round_.ceremony_ops_buf     = [device_ newBufferWithLength:sizeof(CeremonyOp) * kMaxOpsPerRound options:MTLResourceStorageModeShared];
        round_.contribution_ops_buf = [device_ newBufferWithLength:sizeof(ContributionOp) * kMaxOpsPerRound options:MTLResourceStorageModeShared];
        round_.ceremonies_buf       = [device_ newBufferWithLength:sizeof(Ceremony) * kCeremonySlots options:MTLResourceStorageModeShared];
        round_.key_shares_buf       = [device_ newBufferWithLength:sizeof(KeyShare) * kKeyShareSlots options:MTLResourceStorageModeShared];
        round_.contributions_buf    = [device_ newBufferWithLength:sizeof(Contribution) * kContributionSlots options:MTLResourceStorageModeShared];
        round_.state_buf            = [device_ newBufferWithLength:sizeof(MPCVMState) options:MTLResourceStorageModeShared];
        round_.result_buf           = [device_ newBufferWithLength:sizeof(MPCVMTransitionResult) options:MTLResourceStorageModeShared];
        round_.ceremony_applied_buf      = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        round_.contribution_applied_buf  = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        round_.round_advance_buf         = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        round_.finalized_buf             = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        round_.failed_buf                = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        round_.ceremony_leaf_hashes     = [device_ newBufferWithLength:32u * kCeremonySlots options:MTLResourceStorageModeShared];
        round_.share_leaf_hashes        = [device_ newBufferWithLength:32u * kKeyShareSlots options:MTLResourceStorageModeShared];
        round_.contribution_leaf_hashes = [device_ newBufferWithLength:32u * kContributionSlots options:MTLResourceStorageModeShared];
        round_.ceremony_used_mask       = [device_ newBufferWithLength:kCeremonySlots options:MTLResourceStorageModeShared];
        round_.share_used_mask          = [device_ newBufferWithLength:kKeyShareSlots options:MTLResourceStorageModeShared];
        round_.contribution_used_mask   = [device_ newBufferWithLength:kContributionSlots options:MTLResourceStorageModeShared];
        round_.active_count_buf         = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        round_.finalized_count_buf      = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        round_.failed_count_buf         = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        round_.share_count_buf          = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        if (!round_.desc_buf || !round_.ceremony_ops_buf || !round_.contribution_ops_buf
            || !round_.ceremonies_buf || !round_.key_shares_buf || !round_.contributions_buf
            || !round_.state_buf || !round_.result_buf
            || !round_.ceremony_applied_buf || !round_.contribution_applied_buf
            || !round_.round_advance_buf || !round_.finalized_buf || !round_.failed_buf
            || !round_.ceremony_leaf_hashes || !round_.share_leaf_hashes
            || !round_.contribution_leaf_hashes
            || !round_.ceremony_used_mask || !round_.share_used_mask
            || !round_.contribution_used_mask
            || !round_.active_count_buf || !round_.finalized_count_buf
            || !round_.failed_count_buf || !round_.share_count_buf)
            return MPCVMRoundHandle{0};

        std::memset([round_.ceremonies_buf contents],    0, sizeof(Ceremony) * kCeremonySlots);
        std::memset([round_.key_shares_buf contents],    0, sizeof(KeyShare) * kKeyShareSlots);
        std::memset([round_.contributions_buf contents], 0, sizeof(Contribution) * kContributionSlots);
        std::memset([round_.state_buf contents],         0, sizeof(MPCVMState));
        std::memset([round_.result_buf contents],        0, sizeof(MPCVMTransitionResult));
        std::memset([round_.ceremony_used_mask contents],     0, kCeremonySlots);
        std::memset([round_.share_used_mask contents],        0, kKeyShareSlots);
        std::memset([round_.contribution_used_mask contents], 0, kContributionSlots);
        *static_cast<uint32_t*>([round_.ceremony_applied_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.contribution_applied_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.round_advance_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.finalized_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.failed_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.active_count_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.finalized_count_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.failed_count_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.share_count_buf contents]) = 0;

        round_.desc.ceremony_op_count = 0;
        round_.desc.contribution_op_count = 0;
        round_.next_share_id = 1;
        round_.next_contribution_id = 1;

        round_.handle = MPCVMRoundHandle{++next_handle_};
        return round_.handle;
    }

    void push_ceremony_ops(MPCVMRoundHandle h, std::span<const CeremonyOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        if (ops.empty()) return;
        auto* dst = static_cast<CeremonyOp*>([round_.ceremony_ops_buf contents]);
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.ceremony_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        std::memcpy(dst + round_.desc.ceremony_op_count,
                    ops.data(), take * sizeof(CeremonyOp));
        round_.desc.ceremony_op_count += take;
    }

    void push_contribution_ops(MPCVMRoundHandle h, std::span<const ContributionOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        if (ops.empty()) return;
        auto* dst = static_cast<ContributionOp*>([round_.contribution_ops_buf contents]);
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.contribution_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        std::memcpy(dst + round_.desc.contribution_op_count,
                    ops.data(), take * sizeof(ContributionOp));
        round_.desc.contribution_op_count += take;
    }

    MPCVMTransitionResult run_epoch(MPCVMRoundHandle h) override {
        return run_until_done(h, 1);
    }

    MPCVMTransitionResult run_until_done(MPCVMRoundHandle h, std::size_t /*max*/) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return MPCVMTransitionResult{};

        std::memcpy([round_.desc_buf contents], &round_.desc, sizeof(MPCVMRoundDescriptor));

        id<MTLCommandBuffer> cmd = [queue_ commandBuffer];

        uint32_t ceremony_count_v     = kCeremonySlots;
        uint32_t key_share_count_v    = kKeyShareSlots;
        uint32_t contribution_count_v = kContributionSlots;
        uint64_t next_cont_v          = round_.next_contribution_id;
        uint64_t next_share_v         = round_.next_share_id;

        // -- 1. Ceremony apply (Phase 1+2 ops, serial 1x1x1) --
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:apply_pso_];
            [enc setBuffer:round_.desc_buf             offset:0 atIndex:0];
            [enc setBuffer:round_.ceremony_ops_buf     offset:0 atIndex:1];
            [enc setBuffer:round_.contribution_ops_buf offset:0 atIndex:2];
            [enc setBuffer:round_.ceremonies_buf       offset:0 atIndex:3];
            [enc setBuffer:round_.contributions_buf    offset:0 atIndex:4];
            [enc setBuffer:round_.ceremony_applied_buf offset:0 atIndex:5];
            [enc setBuffer:round_.contribution_applied_buf offset:0 atIndex:6];
            [enc setBytes:&ceremony_count_v     length:sizeof(ceremony_count_v)     atIndex:7];
            [enc setBytes:&contribution_count_v length:sizeof(contribution_count_v) atIndex:8];
            [enc setBytes:&next_cont_v          length:sizeof(next_cont_v)          atIndex:9];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
        }
        // -- 2. Ceremony sweep (Phase 3, kSweepThreads in one threadgroup) --
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:sweep_pso_];
            [enc setBuffer:round_.desc_buf             offset:0 atIndex:0];
            [enc setBuffer:round_.ceremonies_buf       offset:0 atIndex:1];
            [enc setBuffer:round_.key_shares_buf       offset:0 atIndex:2];
            [enc setBuffer:round_.contributions_buf    offset:0 atIndex:3];
            [enc setBuffer:round_.round_advance_buf    offset:0 atIndex:4];
            [enc setBuffer:round_.finalized_buf        offset:0 atIndex:5];
            [enc setBuffer:round_.failed_buf           offset:0 atIndex:6];
            [enc setBytes:&ceremony_count_v     length:sizeof(ceremony_count_v)     atIndex:7];
            [enc setBytes:&key_share_count_v    length:sizeof(key_share_count_v)    atIndex:8];
            [enc setBytes:&contribution_count_v length:sizeof(contribution_count_v) atIndex:9];
            [enc setBytes:&next_share_v         length:sizeof(next_share_v)         atIndex:10];
            [enc setThreadgroupMemoryLength:sizeof(uint32_t) * kSweepThreads atIndex:0];
            [enc dispatchThreads:MTLSizeMake(kSweepThreads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kSweepThreads, 1, 1)];
            [enc endEncoding];
        }
        // -- 3. Compute leaves (parallel keccak per slot) --
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:leaves_pso_];
            [enc setBuffer:round_.ceremonies_buf            offset:0 atIndex:0];
            [enc setBuffer:round_.key_shares_buf            offset:0 atIndex:1];
            [enc setBuffer:round_.contributions_buf         offset:0 atIndex:2];
            [enc setBuffer:round_.ceremony_leaf_hashes      offset:0 atIndex:3];
            [enc setBuffer:round_.share_leaf_hashes         offset:0 atIndex:4];
            [enc setBuffer:round_.contribution_leaf_hashes  offset:0 atIndex:5];
            [enc setBuffer:round_.active_count_buf          offset:0 atIndex:6];
            [enc setBuffer:round_.finalized_count_buf       offset:0 atIndex:7];
            [enc setBuffer:round_.failed_count_buf          offset:0 atIndex:8];
            [enc setBuffer:round_.share_count_buf           offset:0 atIndex:9];
            [enc setBuffer:round_.ceremony_used_mask        offset:0 atIndex:10];
            [enc setBuffer:round_.share_used_mask           offset:0 atIndex:11];
            [enc setBuffer:round_.contribution_used_mask    offset:0 atIndex:12];
            [enc setBytes:&ceremony_count_v     length:sizeof(ceremony_count_v)     atIndex:13];
            [enc setBytes:&key_share_count_v    length:sizeof(key_share_count_v)    atIndex:14];
            [enc setBytes:&contribution_count_v length:sizeof(contribution_count_v) atIndex:15];
            // Use a threadgroup sized for the device (Metal recommends 128-256
            // for compute kernels with small per-thread work). Dispatch fewer
            // threads than the union; each thread bounds-checks its tid.
            uint32_t threadgroup = 256u;
            [enc dispatchThreads:MTLSizeMake(kLeafThreads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadgroup, 1, 1)];
            [enc endEncoding];
        }
        // -- 4. Compose root (serial fold + state-root composition) --
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:compose_pso_];
            [enc setBuffer:round_.desc_buf                  offset:0 atIndex:0];
            [enc setBuffer:round_.ceremony_leaf_hashes      offset:0 atIndex:1];
            [enc setBuffer:round_.share_leaf_hashes         offset:0 atIndex:2];
            [enc setBuffer:round_.contribution_leaf_hashes  offset:0 atIndex:3];
            [enc setBuffer:round_.ceremony_used_mask        offset:0 atIndex:4];
            [enc setBuffer:round_.share_used_mask           offset:0 atIndex:5];
            [enc setBuffer:round_.contribution_used_mask    offset:0 atIndex:6];
            [enc setBuffer:round_.active_count_buf          offset:0 atIndex:7];
            [enc setBuffer:round_.finalized_count_buf       offset:0 atIndex:8];
            [enc setBuffer:round_.failed_count_buf          offset:0 atIndex:9];
            [enc setBuffer:round_.share_count_buf           offset:0 atIndex:10];
            [enc setBuffer:round_.state_buf                 offset:0 atIndex:11];
            [enc setBuffer:round_.result_buf                offset:0 atIndex:12];
            [enc setBytes:&ceremony_count_v     length:sizeof(ceremony_count_v)     atIndex:13];
            [enc setBytes:&key_share_count_v    length:sizeof(key_share_count_v)    atIndex:14];
            [enc setBytes:&contribution_count_v length:sizeof(contribution_count_v) atIndex:15];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        auto* result = static_cast<MPCVMTransitionResult*>([round_.result_buf contents]);
        uint32_t cer_app   = *static_cast<uint32_t*>([round_.ceremony_applied_buf contents]);
        uint32_t cnt_app   = *static_cast<uint32_t*>([round_.contribution_applied_buf contents]);
        uint32_t advances  = *static_cast<uint32_t*>([round_.round_advance_buf contents]);
        uint32_t finalized = *static_cast<uint32_t*>([round_.finalized_buf contents]);
        uint32_t failed    = *static_cast<uint32_t*>([round_.failed_buf contents]);
        result->ceremony_apply_count    = cer_app;
        result->contribution_apply_count = cnt_app;
        result->round_advance_count      = advances;
        result->finalized_this_round     = finalized;
        result->failed_this_round        = failed;

        round_.next_contribution_id += cnt_app;
        // share_id allocation lives on-device; the host counter is monotonic
        // but not strictly continuous within a session — same contract as v0.61.
        return *result;
    }

    MPCVMTransitionResult poll_round_result(MPCVMRoundHandle h) const override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle_const(h)) return MPCVMTransitionResult{};
        return *static_cast<const MPCVMTransitionResult*>([round_.result_buf contents]);
    }

    void end_round(MPCVMRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        round_ = Round{};
    }

private:
    bool check_handle(MPCVMRoundHandle h) const {
        return h.valid() && h.opaque == round_.handle.opaque;
    }
    bool check_handle_const(MPCVMRoundHandle h) const { return check_handle(h); }

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> apply_pso_;
    id<MTLComputePipelineState> sweep_pso_;
    id<MTLComputePipelineState> leaves_pso_;
    id<MTLComputePipelineState> compose_pso_;
    std::string device_name_str_;
    Round round_;
    uint64_t next_handle_ = 0;
    mutable std::mutex mu_;
};

}  // namespace

std::unique_ptr<MPCVMGPUEngine> MPCVMGPUEngine::create() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return nullptr;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return nullptr;

        id<MTLLibrary> lib = load_mpcvm_metallib(device);
        if (!lib) lib = compile_mpcvm_library(device);
        if (!lib) return nullptr;

        NSError* err = nil;
        auto fn = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> f = [lib newFunctionWithName:name];
            if (!f) {
                std::fprintf(stderr, "MPCVM kernel %s not found\n", [name UTF8String]);
                return nil;
            }
            id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:f error:&err];
            if (!p && err)
                std::fprintf(stderr, "PSO compile %s: %s\n",
                             [name UTF8String], [[err localizedDescription] UTF8String]);
            return p;
        };
        id<MTLComputePipelineState> apply_pso   = fn(@"mpcvm_ceremony_apply");
        id<MTLComputePipelineState> sweep_pso   = fn(@"mpcvm_ceremony_sweep");
        id<MTLComputePipelineState> leaves_pso  = fn(@"mpcvm_compute_leaves");
        id<MTLComputePipelineState> compose_pso = fn(@"mpcvm_compose_root");
        if (!apply_pso || !sweep_pso || !leaves_pso || !compose_pso) return nullptr;

        return std::unique_ptr<MPCVMGPUEngine>(
            new MPCVMGPUEngineMetal(device, queue, apply_pso, sweep_pso,
                                    leaves_pso, compose_pso, [device name]));
    }
}

}  // namespace mpcvm::gpu
