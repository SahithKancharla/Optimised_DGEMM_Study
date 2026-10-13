# Revalidation records

Durable, locally-readable records of correctness and performance testing done
against this codebase, so results from different sessions/hardware can be
compared later without re-deriving them from scratch or trusting memory.

**Start here: [`FINDINGS.md`](../FINDINGS.md)** — the synthesized writeup (bugs
found and fixed, what reproduces vs. doesn't across hardware, and a
hardware-counter-based root-cause analysis of *why*). Everything below this
README is the raw supporting data that FINDINGS.md is built from.

## Structure

```
revalidation/
  ../FINDINGS.md                       -- START HERE: the synthesis
  correctness/
    <date>_<short-description>.md   -- bug evidence + fix verification logs
  benchmarks/
    <date>_<host>-<cpu>/
      metadata.txt   -- hardware, compiler, git commit, build flags, what was
                         (and wasn't) tested, caveats
      raw_output.txt -- exact stdout from the run
      *.csv          -- the same data, parsed for programmatic aggregation
      <driver>.c      -- the exact benchmark driver source used, for repro
  harness/
    <date>_<host>-<cpu>/
      metadata.txt         -- same idea as benchmarks/, plus which perf
                               events were used/unavailable
      harness_results.csv  -- time + hardware counters (cycles, instructions,
                               branches, cache/TLB misses) + recursive call
                               counts and allocation bytes, one row per
                               (algorithm, N, threads)
  dumas2024/
    verify_formula.py            -- sympy derivation/verification of the
                                     Dumas et al. 2024 alternate-basis
                                     Strassen formula (re-run this if
                                     dumas_strassen_rebuild.c's base-case
                                     math is ever edited)
    correctness_sweep.sh          -- N=64..512 x depth 1-3 correctness check
    accuracy_results_N512.csv     -- 5-distribution accuracy comparison vs.
                                     classic Strassen and plain DGEMM
    perf_comparison.sh / perf_comparison_results.csv
                                   -- head-to-head phase-timed performance,
                                     same infrastructure, only the formula
                                     differs -- see its own README.md
```

See `../FINDINGS.md` Part L for the related-work/publication-strategy
writeup this project is being aimed at, and `../paper/` for the resulting
LaTeX paper outline.

## Adding a new run

1. Pick a directory name: `benchmarks/<YYYY-MM-DD>_<host-or-machine-id>-<cpu-short-name>/`.
2. Always write `metadata.txt` first — at minimum: date, git commit hash
   (`git rev-parse HEAD`, plus note any uncommitted changes), hostname, CPU
   model, core count, ISA extensions available (`grep flags /proc/cpuinfo`),
   compiler version, exact build flags, what sizes/threads/reps were used,
   and **what was skipped and why**. Future-you will not remember the
   context otherwise.
3. Save the raw stdout verbatim (`raw_output.txt`) even if you also parse it
   into CSVs — the raw form is the source of truth if a parser bug is ever
   suspected.
4. Save the driver source alongside its own output, not just once globally
   — drivers change between runs and you want to know exactly what produced
   which numbers.
5. If comparing against a previous run, note that explicitly in the new
   `metadata.txt` (which run, what changed) rather than relying on
   directory-name diffing.

## Index of runs so far

- `correctness/2026-09-09_bug_evidence_and_fixes.md` — discovery and
  verification of the four bugs found this session (see summary below),
  ending in a full pass across all 13 implementations.
- `benchmarks/2026-09-09_glogin-e5-2630v3/` — reduced-grid (5 sizes x up to
  5 thread counts, 3 reps) performance spot-check on `glogin.cs.vt.edu`
  (Xeon E5-2630 v3, 8 cores, AVX2 only — **not** the paper's Xeon Gold 5218 /
  AVX-512 / 32-core target; read `metadata.txt` in that directory before
  drawing conclusions from the numbers).
- `benchmarks/2026-09-09_local-i7-13700K/` — same idea, run second on the
  user's own local machine (via WSL2 + Docker, genuine host-CPU execution,
  not a VM sandbox), 6 sizes x up to 16 threads. i7-13700K is a *hybrid*
  P-core/E-core design and a different gcc version than the other two data
  points in this repo (gcc 13 here vs. gcc 11.5 elsewhere) — see
  `metadata.txt` before comparing directly against the glogin run.
- `harness/2026-09-09_glogin-e5-2630v3/` — hardware-counter root-cause data
  (cycles, instructions, branches, cache/TLB misses, plus recursive call
  counts and allocation bytes for the Strassen variants). Only collected on
  glogin — WSL2 blocks all hardware PMU counters entirely, confirmed via
  `perf stat` reporting `<not supported>` for every event even under
  `--privileged` Docker. See `../FINDINGS.md` Part E for the analysis.
- `dumas2024/` — implementation, symbolic-formula verification,
  correctness sweep, and accuracy/performance comparison for Dumas,
  Pernet & Sedoglavic's 2024 accuracy-optimized alternate-basis Strassen
  (arXiv:2402.05630), run 2026-09-09 on the dev machine (i7-13700K, via
  WSL2 gcc 13.3.0). See `../FINDINGS.md` Part M for the full writeup and
  `dumas2024/README.md` for what's in this specific directory.

### Bugs found and fixed (2026-09-09)

1. `audited_submission/avx/matmul.c` computed `C = B·A` instead of `C = A·B` (packing bug).
2. `mm_parallel_blocked.c`/`mm_parallel_blocked_unrolled.c` had a data race
   (parallelized over the reduction dimension without synchronization; only
   "worked" by static-scheduling coincidence).
3. `audited_submission/avx/strassen.c` silently dropped a row/column for non-power-of-2 N > 128
   (no padding, unlike the Serial/Parallel Strassen variants).
4. `audited_submission/test/run.sh`/`audited_submission/test/main.c` segfaulted on a clean checkout (missing
   `Values/`/`Results/` dirs, no `fopen` NULL-check downstream).

See the correctness log for evidence and exact fixes; see
`audited_submission/avx/kernel_avx2.c` / `audited_submission/avx/strassen_avx2.c` / `audited_submission/avx/run_local_avx2.sh` for a
way to re-run correctness checks on AVX2-only hardware going forward.

### Known limitation, not fixed

`strassen()` (`audited_submission/avx/strassen.c`) segfaults if called directly on N ∈ {1, 2}
(or N=4 for the AVX-512 build) — the no-padding fast path feeds a
smaller-than-vector-width N straight into the register-blocked base-case
kernel. Outside the paper's tested range (N≥64); listed here so it isn't
rediscovered from scratch.
