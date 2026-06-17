# MPCVM Coverage Report

Run: `2026-04-26` — branch `main`, tag target `v0.61.0`.

## Build

```
cmake -S . -B build-cov -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping -O0 -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate" \
  -DLUX_MPCVM_ENABLE_WGPU=ON
cmake --build build-cov -j
cd build-cov && LLVM_PROFILE_FILE="cov.%p.profraw" ctest
xcrun llvm-profdata merge -sparse cov.*.profraw -o cov.profdata
xcrun llvm-cov report -instr-profile=cov.profdata \
    ./mpcvm-determinism-test ./mpcvm-gpu-engine-test ./mpcvm-layout-test \
    --ignore-filename-regex='test/'
```

## Results

```
Filename                                Regions  Cover     Lines  Cover     Branches  Cover
include/lux/mpcvm/mpcvm_gpu_engine.hpp        3  100.00%       3  100.00%         0      -
src/mpcvm_cpu_reference.cpp                 335   94.93%     477   97.90%       248   90.32%
src/mpcvm_gpu_engine.mm                     169   50.30%     296   65.88%       108   32.41%
src/mpcvm_gpu_engine_wgpu.cpp               218   73.85%     471   83.86%       148   47.30%
TOTAL                                       725   78.21%    1247   85.00%       504   65.28%
```

The deterministic CPU oracle (`mpcvm_cpu_reference.cpp`) is the byte-equivalence
target for all four backends. It clears the gating thresholds:

* line ≥ 95% — actual **97.90%**
* branch ≥ 90% — actual **90.32%**

Remaining uncovered paths in the oracle are loop-end safety guards
(open-addressing tables: 256 ceremony slots / 4096 contribution + share slots)
that cannot fire under any of the production-shape workloads exercised by
the test harness, plus dead-arm switch defaults retained for documentary
completeness (`scheme_for_kind` classifies sign and keygen kinds; only keygen
is ever passed by callers).

GPU drivers (`gpu_engine.mm`, `gpu_engine_wgpu.cpp`) sit lower because the
factory-failure / device-acquisition error paths and adapter-info fallbacks
are not exercised on a working host. Their happy paths run end-to-end
through every determinism workload below.

## Test inventory

`mpcvm-layout-test` (28 tests):
* Layout & alignment invariants for all eight host/GPU structs.
* FROST 7-of-10 keygen completes 3 rounds, emits 7 shares.
* FROST 7-of-10 sign completes 2 rounds, no shares.
* CGGMP21 5-of-9 keygen finalises 3 rounds with 5 shares.
* CGGMP21 sign 5 rounds (no shares).
* Corona DKG 2 rounds → 3 lattice shares.
* Corona sign 2 rounds.
* Replay rejected (same `(ceremony, round, holder)` dropped).
* Timeout marks ceremony `failed`.
* 3-way concurrent ceremonies, deterministic across runs.
* Empty round produces deterministic non-zero state root.
* Cancel marks ceremony failed and is idempotent.
* Invalid begin ops rejected (`t=0`, `t>n`, `n>64`).
* Contribution rejection paths (wrong round, bad holder, unknown id, oversized payload).
* Two independent states produce identical roots for the same workload.
* Composed state-root changes when timestamp changes.
* Open-addressing hash-collision linear-probe paths exercised (64 ceremonies).
* Duplicate `(cer, round, holder)` suppressed before bitmap update.
* Table overflow returns 0xFFFFFFFF and op is skipped.
* Unknown ceremony kind falls through to 1-round default.
* All eight `MPCVMTransitionMode` values exercised end-to-end.

`mpcvm-gpu-engine-test` — Metal driver smoke + cross-state-machine path.

`mpcvm-determinism-test` (13 assertions):
* CPU oracle ↔ Metal: FROST 7-of-10 keygen, CGGMP21 5-of-9 keygen, replay
  drop, timeout, 3-way concurrent, empty round, two-engines-bytewise.
* CPU oracle ↔ WGSL/wgpu-native: FROST keygen, CGGMP21 keygen, replay
  drop, timeout, 3-way concurrent, empty round.

All thirteen workloads produce byte-identical
`(ceremony_root, key_share_root, contribution_root, mpcvm_state_root)`
across every enabled backend.

## Cross-backend determinism

Backends present:

* **CPU reference oracle** — `mpcvm_cpu_reference.cpp`.
* **Metal** — Apple silicon canonical via `mpcvm_gpu_engine.mm` +
  `.metal` kernels (auto-built `metallib`).
* **WGSL** via wgpu-native — `mpcvm_gpu_engine_wgpu.cpp` +
  `mpcvm_*.wgsl`. Linked through `LUX_MPCVM_ENABLE_WGPU=ON`.
* **CUDA** — kernels in `mpcvm_*.cu`, off by default; structurally
  identical to Metal/WGSL, validated via the same harness when
  `LUX_MPCVM_ENABLE_CUDA=ON`.

Determinism contract — bytewise equality of all four roots — holds
across CPU, Metal, and WGSL on the macOS run captured here
(Apple M1 Max). Bytes were dumped on diff for each diverging workload
during bring-up; the only divergence found and fixed in v0.61 was a
WGSL `Ceremony` struct stride drift (host = 128 B, WGSL natural pack =
120 B, fixed by adding an explicit 8-byte trailing pad field —
`mpcvm_kernels_common.wgsl`).

## Keccak audit

All four backends use a masked rotation primitive that is defined for
n ∈ [0, 63] and avoids the `x >> 64` UB at n=0:

```c
n &= 63u;
return (x << n) | (x >> ((64u - n) & 63u));
```

* CPU `mpcvm_cpu_reference.cpp:70` — `rotl64`, additionally pinned to
  `__attribute__((optnone))` on Apple Clang so the 136-byte rate path
  through `keccak_f1600` matches the CUDA / Metal / WGSL bit pattern
  under all optimisation levels.
* CUDA `mpcvm_kernels_common.cuh:168` — same masked form.
* Metal `mpcvm_kernels_common.h.metal:192` — same masked form.
* WGSL `mpcvm_kernels_common.wgsl:212` — `rotl64_u32x2` over
  `vec2<u32>` lanes; explicit n=0 / n=32 / n<32 / n>32 fast paths
  verified to match CPU byte-for-byte against every leaf the harness
  emits.

Keccak round constants and rotation table are bit-identical across all
four backends.
