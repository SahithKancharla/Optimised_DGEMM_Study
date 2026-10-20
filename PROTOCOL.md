# Measurement protocol (applies to every run in this repo)

Single standard for `reproduce.sh`, `reproduce.ps1` and `avx512/run_avx512.sh`:

1. **n = 30 independent process launches** per measured configuration
   (default `--runs 30` / `-Runs 30`). Each launch is a fresh process.
2. **One sample = one launch** = warm-up, then the best of the inner reps
   (`--reps`, default 5; matches `revalidation/n30_rerun_2026-10-12/`).
3. **Interleaving**: in the AVX-512 pipeline the AVX-512 and AVX2 launches
   alternate order every iteration, so thermal/background drift hits both equally.
4. **Correctness in every launch** (relative error checked each time); a run with
   a failing check is invalid, not "slower".
5. **Reported statistics**: mean, SD, CV%, median, bootstrap 95% CI of the mean
   (10,000 resamples, seed 12345), min/max. AVX-512 vs AVX2: ratio of means with a
   bootstrap 95% CI and a permutation test, **Holm-corrected** across all
   comparisons in the run. Groups with CV > 5% are flagged as noisy.
6. **Metadata always recorded**: CPU, ISA flags, OS/kernel (WSL2 noted), compiler,
   flags, git commit + dirty state, power/thermal caveats.
7. `--smoke` / `-Smoke` (n=3) exists only to check the wiring. Its results are
   labelled `SMOKE` and must not be quoted.

What n=30 does and does not buy: it gives usable confidence intervals and
tests across process launches *within one machine and session*. It does not
make a laptop thermally stable, does not remove between-session or
between-machine variation (repeat on different days / systems), and does not
make results from different hardware comparable in absolute GFLOP/s.
Unpinned threads are scheduled by the OS (see paper Section 5).

Runtime: the full AVX-512 grid is ~(sizes x thread counts x 30 x 2) launches
(the script prints the count up front) -- expect from tens of minutes to a few
hours depending on the machine; use `--sizes`/`--threads` to focus.
