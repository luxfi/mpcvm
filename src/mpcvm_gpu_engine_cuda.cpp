// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_gpu_engine_cuda.cpp — CUDA-backed driver for MPCVMGPUEngine.
//
// Mirrors the Metal driver: one round = two sequential kernel launches.
// Each launch is <<<1, 1>>> — single-thread canonical traversal preserves
// byte-for-byte determinism with the CPU reference and the Metal driver.

#include "lux/mpcvm/mpcvm_gpu_engine.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace mpcvm::gpu {

extern "C" {

void mpcvm_ceremony_step(
    const MPCVMRoundDescriptor* desc,
    const CeremonyOp*           ceremony_ops,
    const ContributionOp*       contribution_ops,
    Ceremony*                   ceremonies,
    KeyShare*                   key_shares,
    Contribution*               contributions,
    uint32_t*                   ceremony_applied_out,
    uint32_t*                   contribution_applied_out,
    uint32_t*                   round_advance_out,
    uint32_t*                   finalized_out,
    uint32_t*                   failed_out,
    uint32_t                    ceremony_count,
    uint32_t                    key_share_count,
    uint32_t                    contribution_count,
    uint64_t                    next_contribution_id_in,
    uint64_t                    next_share_id_in);

void mpcvm_transition(
    const MPCVMRoundDescriptor* desc,
    Ceremony*                   ceremonies,
    KeyShare*                   shares,
    Contribution*               contributions,
    MPCVMState*                 state,
    MPCVMTransitionResult*      result,
    uint32_t                    ceremony_count,
    uint32_t                    share_count,
    uint32_t                    contribution_count);

}  // extern "C"

namespace {

constexpr uint32_t kCeremonySlots    = kDefaultCeremonySlots;
constexpr uint32_t kKeyShareSlots    = kDefaultKeyShareSlots;
constexpr uint32_t kContributionSlots= kDefaultContributionSlots;
constexpr uint32_t kMaxOpsPerRound   = 4096u;

#define CUDA_CHECK(expr) \
    do { cudaError_t err__ = (expr); if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA %s: %s\n", #expr, cudaGetErrorString(err__)); \
        return false; } } while (0)

struct Round {
    MPCVMRoundHandle handle{};
    MPCVMRoundDescriptor desc{};
    uint64_t next_share_id = 1;
    uint64_t next_contribution_id = 1;

    MPCVMRoundDescriptor* d_desc = nullptr;
    CeremonyOp*           d_ceremony_ops = nullptr;
    ContributionOp*       d_contribution_ops = nullptr;
    Ceremony*             d_ceremonies = nullptr;
    KeyShare*             d_key_shares = nullptr;
    Contribution*         d_contributions = nullptr;
    MPCVMState*           d_state = nullptr;
    MPCVMTransitionResult* d_result = nullptr;
    uint32_t* d_ceremony_applied = nullptr;
    uint32_t* d_contribution_applied = nullptr;
    uint32_t* d_round_advance = nullptr;
    uint32_t* d_finalized = nullptr;
    uint32_t* d_failed = nullptr;

    std::vector<CeremonyOp> h_cer_ops;
    std::vector<ContributionOp> h_cnt_ops;
};

class MPCVMGPUEngineCuda final : public MPCVMGPUEngine {
public:
    MPCVMGPUEngineCuda() {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        device_name_str_ = prop.name;
    }

    ~MPCVMGPUEngineCuda() override {
        if (round_active()) end_round(round_.handle);
    }

    const char* device_name() const override { return device_name_str_.c_str(); }
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

        if (!alloc_buffers()) return MPCVMRoundHandle{0};

        // Zero device arenas.
        cudaMemset(round_.d_ceremonies,    0, sizeof(Ceremony) * kCeremonySlots);
        cudaMemset(round_.d_key_shares,    0, sizeof(KeyShare) * kKeyShareSlots);
        cudaMemset(round_.d_contributions, 0, sizeof(Contribution) * kContributionSlots);
        cudaMemset(round_.d_state,         0, sizeof(MPCVMState));
        cudaMemset(round_.d_result,        0, sizeof(MPCVMTransitionResult));

        round_.handle = MPCVMRoundHandle{++next_handle_};
        return round_.handle;
    }

    void push_ceremony_ops(MPCVMRoundHandle h, std::span<const CeremonyOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        if (ops.empty()) return;
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.ceremony_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        round_.h_cer_ops.insert(round_.h_cer_ops.end(),
                                ops.data(), ops.data() + take);
        round_.desc.ceremony_op_count += take;
    }

    void push_contribution_ops(MPCVMRoundHandle h, std::span<const ContributionOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        if (ops.empty()) return;
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.contribution_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        round_.h_cnt_ops.insert(round_.h_cnt_ops.end(),
                                ops.data(), ops.data() + take);
        round_.desc.contribution_op_count += take;
    }

    MPCVMTransitionResult run_epoch(MPCVMRoundHandle h) override {
        return run_until_done(h, 1);
    }

    MPCVMTransitionResult run_until_done(MPCVMRoundHandle h, std::size_t /*max*/) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return MPCVMTransitionResult{};

        cudaMemcpy(round_.d_desc, &round_.desc, sizeof(MPCVMRoundDescriptor), cudaMemcpyHostToDevice);
        if (!round_.h_cer_ops.empty()) {
            cudaMemcpy(round_.d_ceremony_ops, round_.h_cer_ops.data(),
                       round_.h_cer_ops.size() * sizeof(CeremonyOp),
                       cudaMemcpyHostToDevice);
        }
        if (!round_.h_cnt_ops.empty()) {
            cudaMemcpy(round_.d_contribution_ops, round_.h_cnt_ops.data(),
                       round_.h_cnt_ops.size() * sizeof(ContributionOp),
                       cudaMemcpyHostToDevice);
        }

        uint32_t zero = 0;
        cudaMemcpy(round_.d_ceremony_applied,     &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(round_.d_contribution_applied, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(round_.d_round_advance,        &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(round_.d_finalized,            &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(round_.d_failed,               &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);

        mpcvm_ceremony_step<<<1, 1>>>(
            round_.d_desc, round_.d_ceremony_ops, round_.d_contribution_ops,
            round_.d_ceremonies, round_.d_key_shares, round_.d_contributions,
            round_.d_ceremony_applied, round_.d_contribution_applied,
            round_.d_round_advance, round_.d_finalized, round_.d_failed,
            kCeremonySlots, kKeyShareSlots, kContributionSlots,
            round_.next_contribution_id, round_.next_share_id);
        cudaDeviceSynchronize();

        mpcvm_transition<<<1, 1>>>(
            round_.d_desc, round_.d_ceremonies, round_.d_key_shares,
            round_.d_contributions, round_.d_state, round_.d_result,
            kCeremonySlots, kKeyShareSlots, kContributionSlots);
        cudaDeviceSynchronize();

        MPCVMTransitionResult result{};
        cudaMemcpy(&result, round_.d_result, sizeof(result), cudaMemcpyDeviceToHost);

        uint32_t cer_app=0, cnt_app=0, advances=0, finalized=0, failed=0;
        cudaMemcpy(&cer_app,   round_.d_ceremony_applied,     sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cnt_app,   round_.d_contribution_applied, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&advances,  round_.d_round_advance,        sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&finalized, round_.d_finalized,            sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&failed,    round_.d_failed,               sizeof(uint32_t), cudaMemcpyDeviceToHost);
        result.ceremony_apply_count     = cer_app;
        result.contribution_apply_count = cnt_app;
        result.round_advance_count      = advances;
        result.finalized_this_round     = finalized;
        result.failed_this_round        = failed;
        round_.next_contribution_id    += cnt_app;

        // Cache last result on host.
        last_result_ = result;
        return result;
    }

    MPCVMTransitionResult poll_round_result(MPCVMRoundHandle h) const override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle_const(h)) return MPCVMTransitionResult{};
        return last_result_;
    }

    void end_round(MPCVMRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        free_buffers();
        round_ = Round{};
    }

private:
    bool check_handle(MPCVMRoundHandle h) const {
        return h.valid() && h.opaque == round_.handle.opaque;
    }
    bool check_handle_const(MPCVMRoundHandle h) const { return check_handle(h); }

    bool alloc_buffers() {
        CUDA_CHECK(cudaMalloc(&round_.d_desc, sizeof(MPCVMRoundDescriptor)));
        CUDA_CHECK(cudaMalloc(&round_.d_ceremony_ops, sizeof(CeremonyOp) * kMaxOpsPerRound));
        CUDA_CHECK(cudaMalloc(&round_.d_contribution_ops, sizeof(ContributionOp) * kMaxOpsPerRound));
        CUDA_CHECK(cudaMalloc(&round_.d_ceremonies, sizeof(Ceremony) * kCeremonySlots));
        CUDA_CHECK(cudaMalloc(&round_.d_key_shares, sizeof(KeyShare) * kKeyShareSlots));
        CUDA_CHECK(cudaMalloc(&round_.d_contributions, sizeof(Contribution) * kContributionSlots));
        CUDA_CHECK(cudaMalloc(&round_.d_state, sizeof(MPCVMState)));
        CUDA_CHECK(cudaMalloc(&round_.d_result, sizeof(MPCVMTransitionResult)));
        CUDA_CHECK(cudaMalloc(&round_.d_ceremony_applied, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&round_.d_contribution_applied, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&round_.d_round_advance, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&round_.d_finalized, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&round_.d_failed, sizeof(uint32_t)));
        return true;
    }
    void free_buffers() {
        cudaFree(round_.d_desc);
        cudaFree(round_.d_ceremony_ops);
        cudaFree(round_.d_contribution_ops);
        cudaFree(round_.d_ceremonies);
        cudaFree(round_.d_key_shares);
        cudaFree(round_.d_contributions);
        cudaFree(round_.d_state);
        cudaFree(round_.d_result);
        cudaFree(round_.d_ceremony_applied);
        cudaFree(round_.d_contribution_applied);
        cudaFree(round_.d_round_advance);
        cudaFree(round_.d_finalized);
        cudaFree(round_.d_failed);
    }

    std::string device_name_str_;
    Round round_;
    MPCVMTransitionResult last_result_{};
    uint64_t next_handle_ = 0;
    mutable std::mutex mu_;
};

}  // namespace

#if !defined(__APPLE__)
std::unique_ptr<MPCVMGPUEngine> MPCVMGPUEngine::create() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return nullptr;
    return std::make_unique<MPCVMGPUEngineCuda>();
}
#endif

}  // namespace mpcvm::gpu
