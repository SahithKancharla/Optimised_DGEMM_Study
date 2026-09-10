# Dumas, Pernet & Sedoglavic (2024) alternate-basis Strassen -- implementation & measurement

Evidence directory for FINDINGS.md Part M, which reports on implementing
and measuring Dumas, Pernet & Sedoglavic's 2024 accuracy-optimized
alternate-basis 2x2 matrix-multiplication algorithm ("Strassen's
algorithm is not optimally accurate", ISSAC 2024, arXiv:2402.05630,
Equation 36 / Table 3) as a recursive block algorithm, on top of the
exact same pooled-arena / phase-timed / task-parallel infrastructure as
the classic-Strassen rebuild in `avx2/ladder/strassen_rebuild.c` (Part
J.7) -- this is FINDINGS.md Part L.4's highest-priority follow-up
experiment.

Source code lives in `avx2/ladder/`:
- `dumas_strassen_rebuild.c` -- the algorithm itself
- `dumas_main.c` -- standalone CLI driver (correctness + phase-timed perf)
- `accuracy_test.c` -- extended to run `dumas_depth1/2/3` alongside
  `plain_dgemm` and `strassen_depth1/2/3` on the same 5 distributions
- `matmul_ladder2.c` -- given an include guard (`MATMUL_LADDER2_INCLUDED`)
  so it's safe for both `strassen_rebuild.c` and `dumas_strassen_rebuild.c`
  to pull it in within the same translation unit (`accuracy_test.c`)

## Why this needed a verification step before writing any C

The paper states the formula only via a straight-line program (Table 3)
in its own a_ij/b_ij subscript convention. Transcribing that literally
against standard row-major A[i][j]/B[i][j] produced a WRONG result (off
by exactly the antisymmetric a12<->a21, b12<->b21 terms) -- caught by
`verify_formula.py` (sympy), which brute-forced input/output relabelings
until it found the one making the formula symbolically equal `C = A*B`
for arbitrary 2x2 blocks. The corrected formula is what's actually
implemented in `dumas_strassen_rebuild.c`; re-run `verify_formula.py`
(needs `pip install sympy`) if that file's base-case math is ever edited.

## What's in this directory

- `verify_formula.py` -- the sympy derivation/verification above.
- `correctness_sweep.sh` -- runs `dumas_main` at N=64/128/256/512 x
  depth 1/2/3 against small-integer test data with a long-double
  reference check built into the driver (`reference_check` from
  `matmul_ladder2.c`).
- `accuracy_results_N512.csv` -- output of the extended `accuracy_test`
  at N=512 (5 distributions x {plain, strassen_depth1-3, dumas_depth1-3}),
  Frobenius-relative and max-componentwise-relative error vs. an
  independent long-double reference.
- `perf_comparison.sh` / `perf_comparison_results.csv` -- head-to-head
  `strassen_main` vs `dumas_main` phase-timed runs at N=512/1024/2048 x
  depth 1/2/3, T=8, same KC/MC/NC, same leaf kernel -- isolates the
  7-multiplication *formula* as the only difference between the two
  binaries' numbers.

## Machine / environment

Run 2026-09-09, dev machine (i7-13700K, 24 logical CPUs reported), via
WSL2 Ubuntu 24.04 gcc 13.3.0 (`gcc -O2 -mavx2 -mfma -fopenmp`) -- the
same machine FINDINGS.md's revalidation runs label `local-i7-13700K`,
here accessed through WSL2 rather than native Windows/MSVC because this
project's native-Windows MinGW toolchain (MinGW.org gcc 6.3.0, 32-bit)
lacks a usable `-fopenmp`/`pthread` link path; WSL2's real Ubuntu gcc
does not have that limitation. Not yet run on `glogin.cs.vt.edu` or
native Windows -- see FINDINGS.md Part M for what that would add.

## Headline results (see Part M for full discussion)

- **Correctness**: relative error ~1e-15 to ~1e-16 (machine precision)
  across every (N, depth) tested -- the block generalization of the
  2024 formula is genuinely correct, not just the scalar case.
- **Accuracy**: at matched recursion depth, the Dumas variant's
  Frobenius-relative error is consistently *lower* than classic
  Strassen's, across all 5 distributions at N=512 -- qualitatively
  reproducing the paper's own claim, now under real packed/blocked
  execution rather than the paper's own (unblocked, presumably direct)
  test harness.
- **Performance**: the Dumas variant is consistently *slower* than
  classic Strassen at matched (N, depth, thread count) -- e.g. ~20%
  fewer GFLOP/s at N=2048, depth=3 -- driven entirely by higher
  `addsub_s` (its SLP does more, and more expensive, scale-and-combine
  work per level: 24 add/scale ops using irrational sqrt(3) coefficients
  vs. classic Strassen's ~18 pure adds/subs). `leaf_s` (the shared leaf
  kernel) is identical within noise, as expected, confirming the
  slowdown is specifically attributable to the formula, not to anything
  else in the harness. Neither Dumas et al.'s paper nor (as far as the
  Part L.1 literature search found) any other publication reports this
  accuracy/performance trade-off number.
