// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// mpcvm_gpu_engine_cuda.cpp — CUDA-backed driver for MPCVMGPUEngine.
//
// v0.62: per-slot fan-out + parallel leaf reduction (mirror of Metal).
// One round = four kernel launches in canonical order:
//   1. mpcvm_ceremony_apply  <<<1, 1>>>            — Phase 1+2 ops apply
//   2. mpcvm_ceremony_sweep  <<<1, 256, smem>>>    — Phase 3 sweep + prefix
//                                                    sum + share emission
//   3. mpcvm_compute_leaves  <<<grid, 64>>>        — parallel keccak per leaf
//   4. mpcvm_compose_root    <<<1, 1>>>            — serial fold + state root

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

void mpcvm_ceremony_apply(
    const MPCVMRoundDescriptor* desc,
    const CeremonyOp*           ceremony_ops,
    const ContributionOp*       contribution_ops,
    Ceremony*                   ceremonies,
    Contribution*               contributions,
    uint32_t*                   ceremony_applied_out,
    uint32_t*                   contribution_applied_out,
    uint32_t                    ceremony_count,
    uint32_t                    contribution_count,
    uint64_t                    next_contribution_id_in);

void mpcvm_ceremony_sweep(
    const MPCVMRoundDescriptor* desc,
    Ceremony*                   ceremonies,
    KeyShare*                   key_shares,
    Contribution*               contributions,
    uint32_t*                   round_advance_out,
    uint32_t*                   finalized_out,
    uint32_t*                   failed_out,
    uint32_t                    ceremony_count,
    uint32_t                    key_share_count,
    uint32_t                    contribution_count,
    uint64_t                    next_share_id_in);

void mpcvm_compute_leaves(
    const Ceremony*    ceremonies,
    const KeyShare*    shares,
    const Contribution* contributions,
    uint8_t*           ceremony_leaf_hashes,
    uint8_t*           share_leaf_hashes,
    uint8_t*           contribution_leaf_hashes,
    uint32_t*          active_count_out,
    uint32_t*          finalized_count_out,
    uint32_t*          failed_count_out,
    uint32_t*          share_count_out,
    uint8_t*           ceremony_used_mask,
    uint8_t*           share_used_mask,
    uint8_t*           contribution_used_mask,
    uint32_t           ceremony_count,
    uint32_t           share_count,
    uint32_t           contribution_count);

void mpcvm_compose_root(
    const MPCVMRoundDescriptor* desc,
    const uint8_t*              ceremony_leaf_hashes,
    const uint8_t*              share_leaf_hashes,
    const uint8_t*              contribution_leaf_hashes,
    const uint8_t*              ceremony_used_mask,
    const uint8_t*              share_used_mask,
    const uint8_t*              contribution_used_mask,
    const uint32_t*             active_count_in,
    const uint32_t*             finalized_count_in,
    const uint32_t*             failed_count_in,
    const uint32_t*             share_count_in,
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
constexpr uint32_t kSweepThreads     = 256u;
constexpr uint32_t kLeafThreads      = kContributionSlots;

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

    // v0.62 leaf-fold scratch.
    uint8_t*  d_ceremony_leaf_hashes      = nullptr;
    uint8_t*  d_share_leaf_hashes         = nullptr;
    uint8_t*  d_contribution_leaf_hashes  = nullptr;
    uint8_t*  d_ceremony_used_mask        = nullptr;
    uint8_t*  d_share_used_mask           = nullptr;
    uint8_t*  d_contribution_used_mask    = nullptr;
    uint32_t* d_active_count              = nullptr;
    uint32_t* d_finalized_count           = nullptr;
    uint32_t* d_failed_count              = nullptr;
    uint32_t* d_share_count               = nullptr;

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

        cudaMemset(round_.d_ceremonies,    0, sizeof(Ceremony) * kCeremonySlots);
        cudaMemset(round_.d_key_shares,    0, sizeof(KeyShare) * kKeyShareSlots);
        cudaMemset(round_.d_contributions, 0, sizeof(Contribution) * kContributionSlots);
        cudaMemset(round_.d_state,         0, sizeof(MPCVMState));
        cudaMemset(round_.d_result,        0, sizeof(MPCVMTransitionResult));
        cudaMemset(round_.d_ceremony_used_mask,    0, kCeremonySlots);
        cudaMemset(round_.d_share_used_mask,       0, kKeyShareSlots);
        cudaMemset(round_.d_contribution_used_mask, 0, kContributionSlots);
        cudaMemset(round_.d_active_count,    0, sizeof(uint32_t));
        cudaMemset(round_.d_finalized_count, 0, sizeof(uint32_t));
        cudaMemset(round_.d_failed_count,    0, sizeof(uint32_t));
        cudaMemset(round_.d_share_count,     0, sizeof(uint32_t));

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
        cudaMemcpy(round_.d_active_count,    &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(round_.d_finalized_count, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(round_.d_failed_count,    &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(round_.d_share_count,     &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);

        // -- 1. Apply (1×1) --
        mpcvm_ceremony_apply<<<1, 1>>>(
            round_.d_desc, round_.d_ceremony_ops, round_.d_contribution_ops,
            round_.d_ceremonies, round_.d_contributions,
            round_.d_ceremony_applied, round_.d_contribution_applied,
            kCeremonySlots, kContributionSlots,
            round_.next_contribution_id);
        cudaDeviceSynchronize();

        // -- 2. Sweep (1×256, dynamic shared mem for prefix sum) --
        mpcvm_ceremony_sweep<<<1, kSweepThreads, sizeof(uint32_t) * kSweepThreads>>>(
            round_.d_desc,
            round_.d_ceremonies, round_.d_key_shares, round_.d_contributions,
            round_.d_round_advance, round_.d_finalized, round_.d_failed,
            kCeremonySlots, kKeyShareSlots, kContributionSlots,
            round_.next_share_id);
        cudaDeviceSynchronize();

        // -- 3. Compute leaves (parallel) --
        uint32_t leaf_blocks = (kLeafThreads + 63u) / 64u;
        mpcvm_compute_leaves<<<leaf_blocks, 64>>>(
            round_.d_ceremonies, round_.d_key_shares, round_.d_contributions,
            round_.d_ceremony_leaf_hashes, round_.d_share_leaf_hashes, round_.d_contribution_leaf_hashes,
            round_.d_active_count, round_.d_finalized_count, round_.d_failed_count, round_.d_share_count,
            round_.d_ceremony_used_mask, round_.d_share_used_mask, round_.d_contribution_used_mask,
            kCeremonySlots, kKeyShareSlots, kContributionSlots);
        cudaDeviceSynchronize();

        // -- 4. Compose root (1×1) --
        mpcvm_compose_root<<<1, 1>>>(
            round_.d_desc,
            round_.d_ceremony_leaf_hashes, round_.d_share_leaf_hashes, round_.d_contribution_leaf_hashes,
            round_.d_ceremony_used_mask, round_.d_share_used_mask, round_.d_contribution_used_mask,
            round_.d_active_count, round_.d_finalized_count, round_.d_failed_count, round_.d_share_count,
            round_.d_state, round_.d_result,
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
        CUDA_CHECK(cudaMalloc(&round_.d_ceremony_leaf_hashes,    32u * kCeremonySlots));
        CUDA_CHECK(cudaMalloc(&round_.d_share_leaf_hashes,       32u * kKeyShareSlots));
        CUDA_CHECK(cudaMalloc(&round_.d_contribution_leaf_hashes,32u * kContributionSlots));
        CUDA_CHECK(cudaMalloc(&round_.d_ceremony_used_mask,    kCeremonySlots));
        CUDA_CHECK(cudaMalloc(&round_.d_share_used_mask,       kKeyShareSlots));
        CUDA_CHECK(cudaMalloc(&round_.d_contribution_used_mask, kContributionSlots));
        CUDA_CHECK(cudaMalloc(&round_.d_active_count,    sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&round_.d_finalized_count, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&round_.d_failed_count,    sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&round_.d_share_count,     sizeof(uint32_t)));
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
        cudaFree(round_.d_ceremony_leaf_hashes);
        cudaFree(round_.d_share_leaf_hashes);
        cudaFree(round_.d_contribution_leaf_hashes);
        cudaFree(round_.d_ceremony_used_mask);
        cudaFree(round_.d_share_used_mask);
        cudaFree(round_.d_contribution_used_mask);
        cudaFree(round_.d_active_count);
        cudaFree(round_.d_finalized_count);
        cudaFree(round_.d_failed_count);
        cudaFree(round_.d_share_count);
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
