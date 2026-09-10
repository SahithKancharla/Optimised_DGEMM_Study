# Dense Matrix Multiplication Benchmark Suite

This repository contains implementations of serial, parallel, and AVX-512-enhanced matrix multiplication algorithms, along with scripts to benchmark their performance and collect system-level metrics.

**This started as a benchmark suite for the paper in
[`Advance_Parallel_Programming_Dense_Analysis.pdf`](./Advance_Parallel_Programming_Dense_Analysis.pdf)
and has since grown into a full audit + follow-on research project**:
real correctness bugs found and fixed in the original code, the paper's
own performance claims independently re-tested and root-caused with
hardware counters, a from-scratch AVX2 DGEMM tuned against production
BLAS, a repeated-measurement case study on whether a big scheduling win
is real or a virtualization artifact, two independently-built Strassen
rebuilds with numerical-accuracy studies (including a 2024 accuracy-
optimized variant, implemented and benchmarked here for what appears to
be the first time), and a literature-grounded write-up strategy for
turning all of it into a citable paper.

- **[`FINDINGS.md`](./FINDINGS.md)** is the living index of everything
  above (Parts A–M) — start there, not here, for the research narrative.
- **[`paper/`](./paper/)** has a working LaTeX outline of the eventual
  paper (Overleaf-ready), built around FINDINGS.md's three headline
  contributions.
- **[`revalidation/`](./revalidation/)** has the durable, locally-
  re-readable evidence backing FINDINGS.md's claims (bug evidence,
  benchmark CSVs, hardware-counter data, the 2024-Strassen-variant
  verification + results) — organized by topic, dated by run.

## Directory Structure

Original benchmark suite (the paper's own algorithms):
- `Serial/` – Serial implementations of matrix multiplication.
- `Parallel/` – OpenMP-based parallel implementations.
- `avx/` – AVX-512 optimized implementations (plus a permanent AVX2 port,
  `kernel_avx2.c`/`strassen_avx2.c`, added during the audit for
  correctness-testing on hardware without AVX-512).
- `Strassen_BreakDown/` – Scripts to analyze Strassen's algorithm recursion and memory usage.
- `Values/` – Temporary directory used during execution (may contain intermediate values).
- `Results/` – Temporary directory used to store output and timing data.
- `visualisations/` – Scripts to generate performance plots.
- `Data/` – Generated runtime data and traces.
- `test/` – Constain the sanity tests to compare the serial implementation output with the other cases.

Audit, harness, and follow-on research (see `FINDINGS.md` for the full map):
- `avx2/` – from-scratch AVX2 DGEMM vs. production BLAS: the microkernel
  sweep (`avx2/sweep/`), hand-written-assembly kernels and their tuning
  logs (`avx2/asm_kernel/`), the portable/generalizable harness that
  auto-detects cache sizes and computes blocking parameters at runtime
  (`avx2/portable/`), and `avx2/ladder/` (portable phase-timed harness,
  classic-Strassen rebuild, Dumas et al. (2024) alternate-basis Strassen).
- `strassen2/` – the rebuilt Strassen used for the paper's headline
  comparison: pooled workspace, fused adds, 1-/2-level recursion, a
  thread-aware task-vs-sequential cutoff, built on the same packed AVX2
  microkernel as `avx2/` rather than a second hand-rolled one.
- `baselines/` – thin drivers calling `cblas_dgemm` from OpenBLAS/BLIS/
  oneMKL with the same CLI/timing contract as `avx2/`, so numbers are
  directly comparable across libraries.
- `bench/` – the shared benchmark harness (CLI, GFLOP/s, reference-check
  comparison) used by every variant above.
- `accuracy/` – numerical-accuracy experiments: uniform / wide-dynamic-
  range / clustered-cancellation input distributions, checked against a
  long-double reference, across classical vs. 1-/2-level Strassen.
- `roofline/` – STREAM-triad bandwidth microbenchmark and the roofline
  operational-intensity model.
- `topology/` – native-Windows P-core/E-core/SMT scaling and core-
  classification tools (used as this single-socket machine's NUMA
  substitute — see `FINDINGS.md` Part B.13).
- `reference/` – curated excerpts of OpenBLAS's and BLIS's actual GEMM
  kernels/blocking parameters (with license + source-commit provenance),
  kept for direct comparison against this repo's own kernel design.
- `setup/` – environment bring-up (`wsl_env_setup.sh`): build toolchain,
  OpenBLAS/BLIS/oneMKL install, Python venv for plotting.
- `harness/` – opt-in call/byte-count instrumentation and a `perf`-
  wrapping sweep driver, used for the hardware-counter root-cause work.
- `revalidation/` – durable evidence: correctness-bug proof, cross-
  machine benchmark reruns, hardware-counter data, the 2024-Strassen
  verification/accuracy/performance data.
- `paper/` – the LaTeX paper outline (see above).
- `FINDINGS.md` – the living research log tying all of the above together.

## Main executables to Run
- `full_run.sh` – Main automation script to run both serial and parallel. Run using `./full_run.sh`.
- `serial_automation.sh` – Script to benchmark serial variants only. Run using `./serial_automation.sh`.
- `parallel_automation.sh` – Script to benchmark parallel variants only. Run using `./parallel_automation.sh`.
- `./avx/run.sh` – Script to benchmark  avx code. Run using `cd ./avx` and then `./run.sh`.
- `./Strassen_BreakDown/run.sh` – Runs the strassen memory and recurssion tests. Run using `cd Strassen_BreakDown` and then `./run.sh`.
- `./test/run.sh` – Runs all the test cases to see if the output of all the code is same to the serial execution. Run using `cd ./test` and then `./run.sh`.

## How to Run

1. **Ensure you are on a Linux system with AVX-512 and OpenMP support**.
2. **Grant execution permissions** if needed to executables:
   ```bash
   chmod +x full_run.sh serial_automation.sh parallel_automation.sh
   chmod +x avx/run.sh Strassen_BreakDown/run.sh test/run.sh 
   ```

You can download or view the PDF here:  
[Download the PDF](./Advance_Parallel_Programming_Dense_Analysis.pdf)