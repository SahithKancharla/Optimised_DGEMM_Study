# Dense Matrix Multiplication: Audit, Measurement-Bias Study, and a 2024 Strassen Variant

This repository holds two generations of work, kept physically separate
so it's unambiguous which is which:

- **[`audited_submission/`](./audited_submission/)** is the *original*
  classroom paper (`Advance_Parallel_Programming_Dense_Analysis.pdf`)
  and its benchmark suite (serial, OpenMP-parallel, and AVX-512
  implementations) — the thing being studied, not this project's output.
- **Everything else here is this project's own work**, built *on top of*
  that original submission: a correctness audit that found and fixed
  four real bugs in it, the original paper's own performance claims
  independently re-tested and root-caused with hardware counters, a
  from-scratch AVX2 DGEMM tuned against production BLAS, a repeated-
  measurement case study on whether a big scheduling win is real or a
  virtualization artifact, two independently-built Strassen rebuilds
  with numerical-accuracy studies (including a 2024 accuracy-optimized
  variant, implemented and benchmarked here for what appears to be the
  first time), and a paper write-up tying all of it together.

**If you want the paper, go to [`paper/`](./paper/), not
`audited_submission/`** — the PDF in the latter is the input being
audited, not this project's deliverable.

- **[`paper/`](./paper/)** — this project's own paper (Overleaf-ready
  LaTeX), structured around three headline contributions: the
  correctness audit, the virtualization measurement-bias case study,
  and the Dumas et al. (2024) Strassen variant.
- **[`FINDINGS.md`](./FINDINGS.md)** — the living research log behind
  the paper (Parts A–M) — the fastest way to find which script or CSV
  produced any specific number.
- **[`revalidation/`](./revalidation/)** — durable, locally-re-readable
  evidence backing both `FINDINGS.md` and the paper (bug evidence,
  benchmark CSVs, hardware-counter data, the 2024-Strassen-variant
  verification + results), organized by topic, dated by run.
- **[`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md)** — the single
  top-level entry point for re-running any specific claim in the paper
  end-to-end, pinned to a commit. **[`LICENSE`](./LICENSE)** (MIT)
  covers this project's own code/data; `src/reference/`'s curated
  OpenBLAS/BLIS excerpts keep their own upstream BSD-3-Clause terms.
- **Clone and run it yourself**: `./reproduce.sh` (Linux/macOS/WSL) or
  `.\reproduce.ps1` (native Windows) build and run the portable subset
  of this project's claims in a couple of minutes on any machine with
  AVX2+FMA, writing a machine-identified CSV row so results from
  several different machines can be collected and compared directly.

## Directory Structure

```
audited_submission/   Generation 1 -- the original paper + its benchmark
                       suite (Serial/, Parallel/, avx/, test/,
                       Strassen_BreakDown/, visualisations/, Data/, the
                       PDF). See its own README.md before running
                       anything in it -- in particular, its binaries
                       expect to be run FROM this directory, and one
                       real unresolved script-duplication issue is
                       documented there rather than silently fixed.

src/                   Generation 2 -- this project's own new
                       implementation work (avx2/, strassen2/,
                       baselines/, bench/, accuracy/, roofline/,
                       topology/, harness/, reference/). See its own
                       README.md for what each subdirectory does and
                       the one place (harness/) where Generation 2 code
                       is included from Generation 1 files by design.

revalidation/          Evidence tying claims in both generations
                       together: correctness-bug proof, cross-machine
                       benchmark reruns, hardware-counter data, the
                       2024-Strassen verification/accuracy/performance
                       data. See revalidation/README.md.

paper/                 This project's own paper (LaTeX) and its
                       compiled PDF.

setup/                 Environment bring-up (wsl_env_setup.sh): build
                       toolchain, OpenBLAS/BLIS/oneMKL install, Python
                       venv for plotting/formula verification.

FINDINGS.md            The living research log tying everything above
                       together, Parts A-M.
REPRODUCIBILITY.md      How to re-run any specific numbered claim.
reproduce.sh/.ps1       One-command portable-subset reproduction.
LICENSE                 MIT (this project's own code/data).
```

## How to run the original submission

See [`audited_submission/README.md`](./audited_submission/README.md)
for the full detail, including the one open script-duplication issue.
Short version: `cd audited_submission`, then
`./full_run.sh` / `./serial_automation.sh` / `./parallel_automation.sh`,
or `cd avx && ./run.sh`, `cd Strassen_BreakDown && ./run.sh`,
`cd test && ./run.sh` — every one of these expects to be run with
`audited_submission/` (not the repo root) as the working directory,
since `Data/matrix_a.bin`/`matrix_b.bin` are read via a fixed relative
path. Requires a Linux-like environment with OpenMP; the AVX-512 path
additionally requires AVX-512 hardware. The earlier workstreams had none
(`FINDINGS.md` Part A); as of 2026-10-13 it has been run on one such machine
(i5-1145G7, Tiger Lake) — see `FINDINGS.md` Part N and `avx512/README.md`.

## How to run this project's own work

See [`src/README.md`](./src/README.md) for what's in each subdirectory
and the build order, or `REPRODUCIBILITY.md` for the fastest path to
re-running a specific numbered claim from the paper end-to-end.
