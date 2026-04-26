// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_gpu_engine.mm — Metal-backed driver for MPCVMGPUEngine.
//
// One round = two sequential kernel dispatches in canonical order:
//   1. mpcvm_ceremony_step   (ceremony begin/cancel + contributions + sweep)
//   2. mpcvm_transition      (root composition + epoch advance)
//
// Each dispatch is a single thread (1x1x1) — the kernels do canonical
// in-order traversal of their op streams. Determinism contract matches
// the CPU reference (mpcvm_cpu_reference.cpp) byte-for-byte.

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
        k_rt   = load_file(dir / "mpcvm_ringtail.metal");
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
};

class MPCVMGPUEngineMetal final : public MPCVMGPUEngine {
public:
    MPCVMGPUEngineMetal(id<MTLDevice> device,
                        id<MTLCommandQueue> queue,
                        id<MTLComputePipelineState> ceremony_pso,
                        id<MTLComputePipelineState> transition_pso,
                        NSString* device_name)
        : device_(device), queue_(queue),
          ceremony_pso_(ceremony_pso), transition_pso_(transition_pso),
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

        if (!round_.desc_buf || !round_.ceremony_ops_buf || !round_.contribution_ops_buf
            || !round_.ceremonies_buf || !round_.key_shares_buf || !round_.contributions_buf
            || !round_.state_buf || !round_.result_buf
            || !round_.ceremony_applied_buf || !round_.contribution_applied_buf
            || !round_.round_advance_buf || !round_.finalized_buf || !round_.failed_buf)
            return MPCVMRoundHandle{0};

        std::memset([round_.ceremonies_buf contents],    0, sizeof(Ceremony) * kCeremonySlots);
        std::memset([round_.key_shares_buf contents],    0, sizeof(KeyShare) * kKeyShareSlots);
        std::memset([round_.contributions_buf contents], 0, sizeof(Contribution) * kContributionSlots);
        std::memset([round_.state_buf contents],         0, sizeof(MPCVMState));
        std::memset([round_.result_buf contents],        0, sizeof(MPCVMTransitionResult));
        *static_cast<uint32_t*>([round_.ceremony_applied_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.contribution_applied_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.round_advance_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.finalized_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.failed_buf contents]) = 0;

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

        // -- ceremony step --
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ceremony_pso_];
            [enc setBuffer:round_.desc_buf             offset:0 atIndex:0];
            [enc setBuffer:round_.ceremony_ops_buf     offset:0 atIndex:1];
            [enc setBuffer:round_.contribution_ops_buf offset:0 atIndex:2];
            [enc setBuffer:round_.ceremonies_buf       offset:0 atIndex:3];
            [enc setBuffer:round_.key_shares_buf       offset:0 atIndex:4];
            [enc setBuffer:round_.contributions_buf    offset:0 atIndex:5];
            [enc setBuffer:round_.ceremony_applied_buf offset:0 atIndex:6];
            [enc setBuffer:round_.contribution_applied_buf offset:0 atIndex:7];
            [enc setBuffer:round_.round_advance_buf    offset:0 atIndex:8];
            [enc setBuffer:round_.finalized_buf        offset:0 atIndex:9];
            [enc setBuffer:round_.failed_buf           offset:0 atIndex:10];
            [enc setBytes:&ceremony_count_v     length:sizeof(ceremony_count_v)     atIndex:11];
            [enc setBytes:&key_share_count_v    length:sizeof(key_share_count_v)    atIndex:12];
            [enc setBytes:&contribution_count_v length:sizeof(contribution_count_v) atIndex:13];
            [enc setBytes:&next_cont_v          length:sizeof(next_cont_v)          atIndex:14];
            [enc setBytes:&next_share_v         length:sizeof(next_share_v)         atIndex:15];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
        }
        // -- transition (root composition) --
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:transition_pso_];
            [enc setBuffer:round_.desc_buf          offset:0 atIndex:0];
            [enc setBuffer:round_.ceremonies_buf    offset:0 atIndex:1];
            [enc setBuffer:round_.key_shares_buf    offset:0 atIndex:2];
            [enc setBuffer:round_.contributions_buf offset:0 atIndex:3];
            [enc setBuffer:round_.state_buf         offset:0 atIndex:4];
            [enc setBuffer:round_.result_buf        offset:0 atIndex:5];
            [enc setBytes:&ceremony_count_v     length:sizeof(ceremony_count_v)     atIndex:6];
            [enc setBytes:&key_share_count_v    length:sizeof(key_share_count_v)    atIndex:7];
            [enc setBytes:&contribution_count_v length:sizeof(contribution_count_v) atIndex:8];
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

        // Bump next_* counters by what was applied this round.
        round_.next_contribution_id += cnt_app;
        // share IDs: each finalized keygen ceremony emits up to total_participants
        // shares — we don't know the exact total without scanning, so the host
        // does not depend on next_share_id continuity within a single engine
        // session for determinism. The leaf encoding uses share_id from the
        // table itself, which the kernel writes.
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
    id<MTLComputePipelineState> ceremony_pso_;
    id<MTLComputePipelineState> transition_pso_;
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
        id<MTLComputePipelineState> cer_pso = fn(@"mpcvm_ceremony_step");
        id<MTLComputePipelineState> tr_pso  = fn(@"mpcvm_transition");
        if (!cer_pso || !tr_pso) return nullptr;

        return std::unique_ptr<MPCVMGPUEngine>(
            new MPCVMGPUEngineMetal(device, queue, cer_pso, tr_pso, [device name]));
    }
}

}  // namespace mpcvm::gpu
