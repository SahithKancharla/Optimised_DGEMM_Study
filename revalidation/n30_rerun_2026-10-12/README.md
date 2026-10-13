# n=30 re-run of the Section 5 core-placement experiment

Upgrades FINDINGS.md Part G.4 / paper `main.tex` Section 5
(`sec:vmcase-repeat`) from n=4-5 repeats with range-reporting to the
paper's own stated target protocol (Section 3.2): n=30 independent
process invocations per condition, bootstrap 95% confidence intervals,
and a Games-Howell pairwise-significance test across all four
conditions.

## Machine / environment

2026-10-12, dev machine (i7-13700K), native Windows, MSVC 19.44.35222
(`cl.exe /O2 /openmp /arch:AVX2`), `src/avx2/ladder/matmul_ladder.c` →
`matmul_ladder.exe`. Same configuration as the original
`pinning_experiment.csv`: N=1024, T=8, 5 internal reps per invocation
(best-of-5), core list `0,2,4,6,16,17,18,19` for the mixed P+E
condition.

## Files

- `run_cond1.ps1` / `cond1_unpinned_static.csv` -- unpinned, static
- `run_cond2.ps1` / `cond2_unpinned_dynamic.csv` -- unpinned, dynamic
- `run_cond3.ps1` / `cond3_mixed_PE_static.csv` -- pinned 4P+4E, static
- `run_cond4.ps1` / `cond4_mixed_PE_dynamic.csv` -- pinned 4P+4E, dynamic
- `analyze.py` / `analysis_output.txt` -- bootstrap CI + Games-Howell
  (needs `numpy`, `scipy`; no `pingouin` dependency, implemented
  directly)
- `rerun_n30.ps1` -- an earlier, hashtable/array-driven version of the
  same sweep, kept for the record because building it is what surfaced
  the anomaly below; **not** what produced the committed CSVs (see
  below for why).

## Headline result

| Condition | n | mean (GFLOP/s) | 95% CI |
|---|---|---|---|
| unpinned, static | 30 | 250.68 | [240.08, 261.82] |
| unpinned, dynamic | 30 | 264.28 | [255.34, 273.30] |
| pinned 4P+4E, static | 30 | 148.16 | [147.59, 148.74] |
| pinned 4P+4E, dynamic | 30 | 148.10 | [147.59, 148.62] |

Games-Howell pairwise test (k=4, Welch df, studentized range):
**static vs. dynamic is not significant in either placement condition**
(unpinned: p=0.249; pinned: p=0.999) — the paper's headline finding
(no detectable OpenMP scheduling-policy effect) replicates cleanly at
n=30 with an actual p-value, not just an eyeballed overlapping range,
and the n=30 means are close to the original n=4-5
`pinning_experiment.csv` means (unpinned static/dynamic ≈269/285 then
vs. 250.68/264.28 now; pinned static/dynamic ≈146.8/147.2 then vs.
148.16/148.10 now — same phenomenon, consistent across a five-week gap).
**Unpinned beats pinned-mixed decisively** in all four cross
comparisons (p < 1e-14) — a sharper, statistically-confirmed version of
a point the paper's prose already made qualitatively (Windows' own
scheduler evidently keeps most unpinned threads on P-cores rather than
splitting them 50/50 onto the slower E-cores the way explicit mixed
pinning forces).

## A real anomaly found while building this, not swept under the rug

Early attempts to drive this sweep via a single PowerShell script using
an array of hashtables (`$conditions = @(@{...}, @{...}, ...)`,
`foreach` + member access like `$c.sched`/`$c.chunk` inside the
exe-invoking line) produced **catastrophically degraded, but still
numerically correct, results**: GFLOP/s readings as low as 0.2-0.7 (vs.
a healthy ~150-300 band), with `relerr` still `0.000e+00` on every one
of them — the computation was correct, just running ~400-1000x slower
than normal. The same script, byte-identical, run moments apart,
sometimes came back fully healthy and sometimes fully degraded from its
very first iteration (not building up gradually). Investigated and
ruled out, each with a direct test: WSL2/Hyper-V interference (killed
WSL2 entirely, degradation persisted), thermal/sustained-load buildup
(a clean single-condition 120-iteration loop with literal arguments
showed zero degradation throughout), argument corruption (the exe's own
echoed-back N/T/schedule/chunk fields were correct even on degraded
runs), and Windows Defender real-time scanning (checked `MsMpEng.exe`
CPU usage concurrently — idle during a degraded run).

What reliably avoided it: replacing every hashtable/array/`foreach`
construct with plain literal arguments passed directly to each
`matmul_ladder.exe` invocation (`run_cond1.ps1` through
`run_cond4.ps1`, used for the committed data above) — tested clean
across 4 separate 30-iteration runs with zero anomalies. The committed
`rerun_n30.ps1` (with its retry-on-outlier safety net, triggering a
re-invocation if any reading falls under a 50 GFLOP/s sanity floor) is
kept as the artifact of that investigation and as a safety net for
future large sweeps, but the actual committed numbers above came from
the simpler, consistently-clean literal-argument scripts.

Raw evidence of one such degraded run is kept verbatim in
`degraded_run_evidence_results.csv` (every one of 120 readings in the
0.2-14 GFLOP/s band) and `degraded_run_evidence_anomalies.csv` (the
retry-attempt log from the safety-net version of the script,
`rerun_n30.ps1`, showing all 5 retries failing to clear the 50 GFLOP/s
floor on every single measurement in that run).

Root cause not identified — most likely some form of resource
governance specific to how this session's shell-automation tooling
spawns long-running or collection-driven child processes, not a defect
in the benchmark, the OS scheduler, or the hardware (the exact same
compute, invoked more simply, never showed it). Recorded here in full
because it is itself a concrete, if unplanned, demonstration of this
paper's own thesis (Section 5): environments can silently and severely
distort a measurement, and the only real defense is checking, not
assuming.
