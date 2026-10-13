# The audited submission (Generation 1)

**This is not this project's own paper.** Everything in this directory —
the PDF, and the `Serial/`/`Parallel/`/`avx/`/`test/` C code and shell
scripts around it — is the *original* coursework artifact this project
*audits*: a dense-matrix-multiplication paper and its accompanying
benchmark suite, written independently of (and before) the audit,
revalidation, and new-research work that lives in `../src/`,
`../revalidation/`, and `../paper/`.

If you are looking for this project's actual output — the paper that
reports the correctness bugs, the measurement-bias case study, and the
Dumas et al. (2024) Strassen benchmark — that is `../paper/main.tex`
(and `../FINDINGS.md` for the full research log). This directory is the
*subject* of that paper's Section 4 (correctness audit) and part of
Section 7 (supporting material), not a draft or earlier version of it.

## What's here

- `Advance_Parallel_Programming_Dense_Analysis.pdf` — the original paper
  being audited.
- `Serial/`, `Parallel/`, `avx/` — its three implementation tiers
  (serial, OpenMP-parallel, AVX-512), **with this project's correctness
  fixes already applied in place** (see `../revalidation/correctness/`
  for the exact bugs, evidence, and before/after diffs — the pre-fix
  behavior is documented there, not preserved as a separate broken
  copy here, since git history already preserves the original state at
  commit `42386e2` and earlier).
- `test/` — the original repository's own correctness-sanity harness
  (also bug-fixed in place; see `../revalidation/correctness/`).
- `Strassen_BreakDown/`, `visualisations/` — the original paper's own
  recursion/memory-usage breakdown and plotting scripts.
- `Data/` — fixed binary test matrices (`matrix_a.bin`, `matrix_b.bin`)
  that `Serial/`, `Parallel/`, `avx/`, and `test/` read via a relative
  `Data/matrix_a.bin` path — **run their binaries with this directory
  (`audited_submission/`) as the working directory**, not the repo root
  or a subdirectory, or the fixed-path `fopen()` calls will fail.
- `full_run.sh` / `serial_automation.sh` / `parallel_automation.sh` —
  the original orchestration scripts, run from inside this directory.

## A real inconsistency found while reorganizing this, not yet resolved

Each of the three automation scripts above has a same-named sibling
*without* the `.sh` extension (`full_run`, `serial_automation`,
`parallel_automation`). `full_run`/`full_run.sh` are byte-identical, but
**`serial_automation` and `serial_automation.sh` differ** (different
`ALGORITHMS`/`THREAD_COUNTS` arrays), as do `parallel_automation` and
`parallel_automation.sh` — and `full_run.sh` itself actually invokes the
*no-extension* `./serial_automation`/`./parallel_automation`, not the
`.sh` versions the top-level README has always pointed readers to. This
was not silently resolved by picking one and discarding the other,
since it's unclear which version actually produced any specific
committed result — if you depend on these scripts, diff the pair
yourself first and decide which matches what you're trying to
reproduce.
