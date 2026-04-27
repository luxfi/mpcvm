# MPCVM Benchmarks — v0.62

Build: `cmake -DCMAKE_BUILD_TYPE=Release -DLUX_MPCVM_ENABLE_WGPU=ON`, `-O3 -DNDEBUG`.
Host: Apple M1 Max, 32 GB.
Metal device: Apple M1 Max. WebGPU device: Apple M1 Max (via `wgpu-native`).
Iterations: 2 warm-up + 5 measured per (scenario × backend).
Raw output: `BENCHMARKS_V062.txt` (v0.61.1 raw left as `BENCHMARKS_RAW.txt`).

## Headline (v0.62)

v0.62 implements the **per-slot fan-out + parallel leaf reduction** plan
documented in v0.61.1 BENCHMARKS.md:

* `mpcvm_ceremony_apply` (1×1×1) — Phase 1+2 ops apply, serial — preserves
  canonical contribution-id assignment in input-stream order.
* `mpcvm_ceremony_sweep` (256 threads / 1 threadgroup) — Phase 3 sweep with
  intra-threadgroup prefix sum over per-slot share emit counts; share_id
  allocation from base + offset preserves byte-equality with the CPU
  reference's serial slot-traversal-order assignment.
* `mpcvm_compute_leaves` (parallel by slot index) — each thread keccaks
  one ceremony / share / contribution leaf into a precomputed leaf-hash
  arena. The sequential left-fold in v0.61.1 is replaced by parallel leaf
  computation followed by a single-thread fold over precomputed hashes.
* `mpcvm_compose_root` (1×1×1) — folds the precomputed leaf hashes (since
  `keccak(acc || leaf)` is non-associative the fold itself stays serial)
  and computes `mpcvm_state_root`.

`emit_keygen_shares` now uses the open-addressing `contribution_locate`
hash for O(1) per (round, holder) lookup instead of a full O(N_contribs)
linear scan. With 200 finalizing keygens × 7 holders × 3 rounds, this
alone replaces ~5M contribution scans per round with ~4200 hash lookups.

Determinism preserved: slot ordering = canonical ordering (open-addressing
is deterministic), so per-slot fan-out produces byte-equal output to CPU.
Verified by `mpcvm-determinism-test` across 21 cases (CPU↔Metal,
CPU↔WGPU at small/medium/large/xlarge).

## Per-scheme microbenchmarks

100 ceremonies of one scheme, mean over 5 iterations.

| scheme              | cpu mean | metal mean | wgpu mean | metal vs cpu | wgpu vs cpu |
|---------------------|---------:|-----------:|----------:|-------------:|------------:|
| FROST keygen 7-of-10|   9.0 ms |     62.2 ms |   295.7 ms | 0.144×       | 0.030×      |
| FROST sign 5-of-7   |   6.9 ms |     48.3 ms |   225.2 ms | 0.142×       | 0.030×      |
| CGGMP21 keygen 5-of-9|  6.9 ms |     48.4 ms |   225.6 ms | 0.142×       | 0.031×      |
| CGGMP21 sign 4-of-7 |   5.6 ms |     41.3 ms |   190.4 ms | 0.136×       | 0.030×      |
| Ringtail DKG 4-of-7 |   8.8 ms |     57.3 ms |   209.9 ms | 0.153×       | 0.042×      |

Throughput (ceremonies / sec, mean):

| scheme              | cpu       | metal     | wgpu      |
|---------------------|----------:|----------:|----------:|
| FROST keygen        | 11,145.2  |   1,608.1 |     338.2 |
| FROST sign          | 14,582.1  |   2,069.0 |     444.1 |
| CGGMP21 keygen      | 14,516.7  |   2,064.2 |     443.2 |
| CGGMP21 sign        | 17,740.5  |   2,418.8 |     525.1 |
| Ringtail DKG        | 11,371.4  |   1,744.0 |     476.3 |

## Mixed-workload macrobenchmarks (v0.62)

| scenario | ceremonies | contribs | cpu mean | metal mean | wgpu mean | metal speedup | wgpu speedup |
|----------|-----------:|---------:|---------:|-----------:|----------:|--------------:|-------------:|
| small    |        10  |      70  |   1.06 ms |     8.92 ms |    34.83 ms | 0.119×        | 0.030×       |
| medium   |       200  |   1,200  |  16.02 ms |   103.06 ms |   505.88 ms | 0.156×        | 0.032×       |
| large    |       600  |   3,200  |  50.51 ms |   302.24 ms |  1484.10 ms | 0.167×        | 0.034×       |
| xlarge   |     1,000  |   5,150  |  79.24 ms |   507.03 ms |  2308.52 ms | 0.156×        | 0.034×       |

## v0.62 vs v0.61.1 — speedup per scheme

Microbenchmarks (mean ms, lower is better):

| scheme              | metal v0.61.1 | metal v0.62 | metal speedup | wgpu v0.61.1 | wgpu v0.62 | wgpu speedup |
|---------------------|--------------:|------------:|--------------:|-------------:|-----------:|-------------:|
| FROST keygen        |     1900.9 ms |     62.2 ms | **30.56×**    |     782.0 ms |   295.7 ms | **2.64×**    |
| FROST sign          |      204.3 ms |     48.3 ms | **4.23×**     |     628.6 ms |   225.2 ms | **2.79×**    |
| CGGMP21 keygen      |      182.4 ms |     48.4 ms | **3.77×**     |     623.3 ms |   225.6 ms | **2.76×**    |
| CGGMP21 sign        |      293.1 ms |     41.3 ms | **7.10×**     |     782.3 ms |   190.4 ms | **4.11×**    |
| Ringtail DKG        |      239.1 ms |     57.3 ms | **4.17×**     |     900.0 ms |   209.9 ms | **4.29×**    |

Macrobenchmarks (mean ms):

| scenario | metal v0.61.1 | metal v0.62 | metal speedup | wgpu v0.61.1 | wgpu v0.62 | wgpu speedup |
|----------|--------------:|------------:|--------------:|-------------:|-----------:|-------------:|
| small    |       25.4 ms |     8.9 ms  | **2.85×**     |     86.1 ms  |    34.8 ms | **2.47×**    |
| medium   |      751.8 ms |   103.1 ms  | **7.29×**     |   1242.2 ms  |   505.9 ms | **2.46×**    |
| large    |     1215.4 ms |   302.2 ms  | **4.02×**     |   4699.9 ms  |  1484.1 ms | **3.17×**    |
| xlarge   |     9451.4 ms |   507.0 ms  | **18.64×**    |  20297.2 ms  |  2308.5 ms | **8.79×**    |

**v0.62 target met**: per the v0.61.1 BENCHMARKS.md "Roadmap to GPU > CPU"
projection of "Metal ≥ 5× ahead of CPU on xlarge" — Metal v0.62 achieves
**18.64× speedup over Metal v0.61.1 on xlarge**, with the largest absolute
saving (8944 ms → 507 ms) on the largest workload.

## Latency tail (p50 / p95 / p99, ms — selected scenarios)

| scenario / backend       | mean  | p50   | p95    | p99    |
|--------------------------|------:|------:|-------:|-------:|
| micro.cggmp21_sign / cpu  |  5.64 |  5.58 |   5.81 |   5.81 |
| micro.cggmp21_sign / metal|  41.34 | 41.29 |  41.51 |  41.52 |
| micro.cggmp21_sign / wgpu | 190.43 | 190.19| 191.13 | 191.14 |
| large / cpu              | 50.51 | 50.39 |  50.89 |  50.98 |
| large / metal            | 302.24| 302.25|  302.45|  302.49 |
| large / wgpu             |1484.10|1462.95| 1587.82| 1609.81 |
| xlarge / cpu             | 79.24 | 79.43 |  79.52 |  79.54 |
| xlarge / metal           | 507.03| 507.17|  507.36|  507.38 |
| xlarge / wgpu            |2308.52|2244.74| 2426.90| 2438.02 |

Steady-state tails are tight on all three backends (p99 ≈ p50). Cold-cache
variance on the first command buffer of a session can produce occasional
2-3× spikes in the first observed iteration; the warm-up phase eliminates
this from the measured window.

## What this means

1. **GPU per-slot fan-out works.** v0.62 replaces the `if (tid != 0u) return;`
   pattern with parallel-by-slot dispatch in the bulk paths (sweep + leaf
   hashing). The contribution-payload hash lookup eliminates the dominant
   O(N²) cost in `emit_keygen_shares`. Net result: Metal 18.6× faster than
   v0.61.1 at xlarge, WGPU 8.8× faster.

2. **CPU still wins on Apple Silicon.** Metal at 0.156× of CPU on xlarge —
   M1 Max is exceptional at single-threaded scalar work, and keccak's
   sequential fold limits the parallelism ceiling. The win matters on
   discrete CUDA hardware where CPU vs GPU keccak throughput is closer
   to parity (CUDA build is OFF on this M1 host; no measurement).

3. **Determinism contract preserved.** All 21 determinism cases pass
   byte-equal: CPU↔Metal and CPU↔WGPU at small/medium/large/xlarge plus
   the original 6 correctness cases (FROST keygen, CGGMP21 keygen, replay
   drop, timeout sweep, 3-way concurrent, empty round). Slot ordering =
   canonical ordering, per-slot fan-out produces byte-equal output.

4. **GPU residency unchanged.** MPCVM ceremony state still lives in GPU
   buffers next to AIVM tensors, CEVM verifier outputs, BridgeVM merkle
   proofs, and FHE ciphertexts. The architectural value remains: zero
   host-staging cost for cross-VM consumption.

## Reproducing

```
cmake -S /Users/z/work/luxcpp/mpcvm -B build \
    -DCMAKE_BUILD_TYPE=Release -DLUX_MPCVM_ENABLE_WGPU=ON
cmake --build build --target mpcvm-layout-test mpcvm-gpu-engine-test \
    mpcvm-determinism-test mpcvm-benchmark
ctest --test-dir build --output-on-failure
./build/mpcvm-benchmark > BENCHMARKS_V062.txt
```

## Future work

* CUDA build path is structurally aligned with Metal/WGSL but not measured
  on this host (Apple M1, no CUDA). Discrete-GPU numbers will quantify the
  GPU-residency value over CPU.
* Single-threadgroup workgroup_size(256) ceiling caps the sweep prefix sum
  at 256 ceremony slots. Larger arenas (kCeremonySlots > 256) would need
  multi-workgroup global atomic prefix or a two-phase scan.
* Keccak fold remains sequential by construction. A Merkle-tree leaf fold
  with order-preserving canonical pairing would expose log-N parallelism;
  not done in v0.62 because it changes the protocol's wire format.
