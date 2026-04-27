# MPCVM Benchmarks — v0.61.1

Build: `cmake -DCMAKE_BUILD_TYPE=Release -DLUX_MPCVM_ENABLE_WGPU=ON`, `-O3 -DNDEBUG`.
Host: Apple M1 Max, 32 GB.
Metal device: Apple M1 Max. WebGPU device: Apple M1 Max (via `wgpu-native`).
Iterations: 2 warm-up + 5 measured per (scenario × backend).
Raw output: `BENCHMARKS_RAW.txt`.

## Headline

**MPCVM ceremony kernels are scalar (1×1×1 dispatch) by design.** Every
`mpcvm_*` kernel begins with `if (tid != 0) return;`. This is required
for the four-way byte-identical determinism contract that v0.61 proves
across CPU, Metal, CUDA, and WGSL. As a consequence, GPU backends in
v0.61 do **not** out-run a single CPU core on these workloads — they
match it bitwise, with the state living on-device for zero-copy hand-off
to sibling VMs (PVMVM, AIVM, BridgeVM, CEVM, FHE).

NTT/parallel lattice arithmetic does not live in MPCVM. It lives in
`luxcpp/lattice/src/metal` and is invoked by the host-side Ringtail
verifier (`cevm/lib/consensus/quasar/gpu/quasar_ringtail_verifier.cpp`).
MPCVM is the **ceremony control plane** — `t-of-n` round advance,
deduplication, timeout sweep, root composition — not the lattice
arithmetic substrate.

## Per-scheme microbenchmarks

100 ceremonies of one scheme, mean over 5 iterations.

| scheme              | cpu mean | metal mean | wgpu mean | metal vs cpu | wgpu vs cpu |
|---------------------|---------:|-----------:|----------:|-------------:|------------:|
| FROST keygen 7-of-10|   9.4 ms |   1900.9 ms |   782.0 ms | 0.005×       | 0.012×      |
| FROST sign 5-of-7   |   7.0 ms |    204.3 ms |   628.6 ms | 0.034×       | 0.011×      |
| CGGMP21 keygen 5-of-9|  7.2 ms |    182.4 ms |   623.3 ms | 0.040×       | 0.012×      |
| CGGMP21 sign 4-of-7 |   5.8 ms |    293.1 ms |   782.3 ms | 0.020×       | 0.007×      |
| Ringtail DKG 4-of-7 |   9.4 ms |    239.1 ms |   900.0 ms | 0.039×       | 0.010×      |

Throughput (ceremonies / sec, mean):

| scheme              | cpu       | metal     | wgpu      |
|---------------------|----------:|----------:|----------:|
| FROST keygen        | 10,691.4  |     52.6  |    127.9  |
| FROST sign          | 14,329.1  |    489.4  |    159.1  |
| CGGMP21 keygen      | 13,800.5  |    548.2  |    160.4  |
| CGGMP21 sign        | 17,270.2  |    341.2  |    127.8  |
| Ringtail DKG        | 10,656.4  |    418.3  |    111.1  |

## Mixed-workload macrobenchmarks

| scenario | ceremonies | contribs | cpu mean | metal mean | wgpu mean | metal speedup | wgpu speedup |
|----------|-----------:|---------:|---------:|-----------:|----------:|--------------:|-------------:|
| small    |        10  |      70  |   0.92 ms |    25.4 ms |    86.1 ms | 0.036×        | 0.011×       |
| medium   |       200  |   1,200  |  17.04 ms |   751.8 ms |  1242.2 ms | 0.023×        | 0.014×       |
| large    |       600  |   3,200  |  52.19 ms |  1215.4 ms |  4699.9 ms | 0.043×        | 0.011×       |
| xlarge   |     1,000  |   5,150  |  90.40 ms |  9451.4 ms | 20297.2 ms | 0.010×        | 0.004×       |

## Latency tail (p50 / p95 / p99, ms — selected scenarios)

| scenario / backend       | mean  | p50   | p95    | p99    |
|--------------------------|------:|------:|-------:|-------:|
| micro.cggmp21_sign / cpu | 5.79  | 5.81  | 5.85   | 5.85   |
| micro.cggmp21_sign / metal| 293.07| 373.52| 380.35 | 380.67 |
| micro.cggmp21_sign / wgpu | 782.30| 740.21|1111.42 |1144.29 |
| large / cpu              | 52.19 | 52.05 | 52.66  | 52.77  |
| large / metal            |1215.37|1216.08|1223.12 |1223.60 |
| large / wgpu             |4699.90|4618.64|4956.16 |5023.29 |
| xlarge / cpu             | 90.40 | 87.80 | 97.81  | 98.10  |
| xlarge / metal           |9451.37|6708.38|16746.03|17523.86|
| xlarge / wgpu            |20297.18|9308.38|42306.68|44288.84|

CPU tail is tight (p99 ≈ p50). Metal and WGPU show wide tails on
xlarge — a known shader-cache / command-encoder warm-up effect on the
first command buffer of a session.

## What this means

1. **v0.61 deliverable was correctness, not speedup.** The four-way
   byte-equal determinism contract (CPU vs Metal vs CUDA vs WGSL)
   across FROST keygen, CGGMP21 keygen, replay drop, timeout sweep,
   3-way concurrent ceremonies, and the empty round is proven by
   `mpcvm-determinism-test`. These benchmarks confirm GPU output is
   correct; they do not yet show throughput wins.

2. **The kernels are scalar by construction.** `if (tid != 0) return;`
   gates every kernel so GPU traversal order is the same as the CPU
   reference. Per-slot parallelism (sharded ceremony locate, partial
   bucketing for root composition) is a v0.62 work item.

3. **The architectural value is GPU residency.** MPCVM ceremony state
   sits in GPU buffers next to AIVM tensors, CEVM verifier outputs,
   BridgeVM merkle proofs, and FHE ciphertexts inside the same
   command queue. The cost being measured here is the *price of
   keeping ceremony state where the consumers already live*. The
   alternative — staging ceremony state through host RAM on every
   round — is paid by every other VM that needs it.

4. **CPU reference is the right path for single-process, low-volume
   workloads.** Production validators running `t ≤ 16, n ≤ 32`
   ceremonies at one-round-per-block cadence pay <100 ms on CPU.

## Observations on Ringtail

Ringtail DKG ran 0.04× the speed of CPU on Metal. The expected NTT
parallel speedup did not materialise because the MPCVM Ringtail kernel
does not perform NTT — it only handles the ceremony envelope (round
state, dedup, timeout). The lattice arithmetic happens in
`luxcpp/lattice` and `cevm/lib/consensus/quasar/gpu`, which are
independent substrates measured in their own benchmark suites.

## Reproducing

```
cmake -S /Users/z/work/luxcpp/mpcvm -B build-bench \
    -DCMAKE_BUILD_TYPE=Release -DLUX_MPCVM_ENABLE_WGPU=ON
cmake --build build-bench --target mpcvm-benchmark
./build-bench/mpcvm-benchmark > BENCHMARKS_RAW.txt
```

## Roadmap to GPU > CPU

The kernels need to fan out across slots. Concrete plan:

1. **Ceremony locate** — replace serial open-addressing scan with one
   thread per slot, atomic-CAS reservation; same canonical winner via
   tie-breaking on `(holder_index, contribution_id)`.
2. **Contribution dedup** — radix sort by `(ceremony_id, round, holder)`
   per round, then segmented unique. Sort order is canonical, hash
   afterwards, determinism preserved.
3. **Root composition** — keccak leaf hashing in parallel; tree
   reduction in parallel; final composed root is a single thread.

These changes keep the byte-identical contract (same canonical
ordering, just produced in parallel) and should put Metal ≥ 5×
ahead of CPU on `xlarge`. Tracked under v0.62.
