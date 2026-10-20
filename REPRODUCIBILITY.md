# Reproducibility

This is the single top-level entry point the paper (`paper/main.tex`,
Section 9) points to. It ties together the per-directory `README.md`
files that already document exact provenance for their own data, so a
reader can go from "I have this repo" to "I re-ran a specific numbered
claim in the paper" without guessing.

Pinned reference point for everything below: commit `2b78eaa` on
`main` of https://github.com/SahithKancharla/Optimised_DGEMM_Study
(tag this commit, e.g. `git tag paper-v1 2b78eaa`, before cutting a
Zenodo release -- see "Still open" below).

## What machine(s) this was run on

Full detail in `FINDINGS.md` Part A / `paper/main.tex` Section 3.1.
Summary:

| Machine | Role | Key specs |
|---|---|---|
| Dev machine | Primary -- Parts B, C, D, F, G, H, I, J, M | Intel Core i7-13700K (8 P-cores + 8 E-cores, 24 logical, no AVX-512), 64GB RAM, single NUMA node. Accessed 3 ways: native Windows (MSVC 19.44, no virtualization), WSL2/Hyper-V (gcc 13.3.0), WSL2+Docker. |
| `glogin.cs.vt.edu` | Secondary -- Parts B.12, E, J.4, J.5, M.6 | KVM-virtualized 8-vCPU slice of an Intel Xeon E5-2630 v3 (Haswell-EP), no sudo, gcc 11.5.0 (matches the audited paper's compiler), also has Clang 22.1.3 and Intel oneAPI 2022.1 (icx/MKL/VTune). |

Neither machine is the audited paper's original target (dual-socket
Xeon Gold 5218 / Cascade Lake, AVX-512) -- stated plainly in
`FINDINGS.md` Part A and `paper/main.tex` Section 3.1/8, not glossed
over anywhere.

## Toolchain versions used

- gcc 13.3.0 (WSL2 Ubuntu 24.04, dev machine), gcc 11.5.0 / Clang
  22.1.3 / icx 2022.1.0 (`glogin`), MSVC 19.44 (native Windows).
- OpenBLAS 0.3.26, BLIS (built from source, `--enable-cblas -t
  openmp`), Intel oneMKL 2026.1 (dev machine, apt oneAPI repo) /
  oneMKL from the existing `/opt/intel/oneapi/mkl/2022.1.0` install
  (`glogin`).
- Python venv with numpy/pandas/matplotlib/scipy/sympy for plotting
  and the Dumas-2024 formula verification (`setup/wsl_env_setup.sh`
  installs all of this in one step on a fresh WSL2 Ubuntu 24.04).

## Quickest path: clone and run

`./reproduce.sh` (Linux/macOS/WSL, needs gcc) or `.\reproduce.ps1`
(native Windows -- auto-delegates to WSL if present, else falls back
to a narrower MSVC subset) build and run the portable subset of this
project's claims in a couple of minutes: classical-DGEMM correctness
+ performance, the classic Strassen rebuild (Part H/J.7), the Dumas
et al. 2024 variant (Part M), and the accuracy sweep (Part I/J.8/M.4).
On native Windows without WSL, scope narrows to classical DGEMM plus
the two Windows-only P/E-core tools (Strassen/Dumas need OpenMP tasks,
which MSVC's `/openmp` can't build).

Every run writes `results/reproduce_<timestamp>/system_info.txt` and
`run_info.csv` (OS, kernel, CPU model, physical/logical cores, RAM,
compiler + version, git commit, correctness pass/fail, headline
GFLOP/s numbers), and appends that same row to `results/run_log.csv`.
**To get a real cross-machine comparison table**, run this on as many
machines as you want data from, then collect each one's `run_info.csv`
(same header everywhere) and concatenate them -- one spreadsheet, one
row per machine, directly comparable. `results/` is gitignored on
purpose: it's per-machine output, not something any one machine should
commit on another's behalf.

Absolute GFLOP/s numbers will differ by hardware (expected, not a
failure); what should hold on any AVX2+FMA+OpenMP machine is:
correctness (`relerr` at or near machine epsilon, PASS on both
Strassen variants) and direction (Dumas costs more than classic
Strassen, not less -- magnitude varies, see `paper/main.tex` Section 6.5).

## Re-running a specific claim

Each numbered finding in `paper/main.tex` names the directory and
script/CSV it came from inline. Start there, then:

1. **Section 4 (correctness audit)** -- `revalidation/correctness/2026-09-09_bug_evidence_and_fixes.md`
   has the exact before/after command output for every bug in
   Table 1. `audited_submission/avx/run_local_avx2.sh` re-runs the permanent AVX2
   correctness regression suite (`audited_submission/avx/test_local.c`) from scratch.
2. **Section 5 (scheduling/virtualization case study)** --
   `src/avx2/ladder/sweep.sh` (Linux) / `sweep.ps1` (Windows) regenerate
   the single-run cross-environment sweep CSVs;
   `src/avx2/ladder/pinning_experiment.csv` is the controlled, repeated
   P/E-core experiment -- rerun by calling `matmul_ladder.exe` with an
   explicit core list multiple times (see that file's header comment
   for the exact invocations used).
3. **Section 6 (Dumas et al. 2024 variant)** --
   `revalidation/dumas2024/verify_formula.py` (symbolic formula check,
   run this first if `dumas_strassen_rebuild.c`'s base-case math is
   ever touched), `correctness_sweep.sh`, `perf_comparison.sh` --
   each writes its own timestamped CSV next to the committed one.
4. **Section 7 (supporting material)** -- `src/baselines/build.sh` builds
   the OpenBLAS/BLIS/MKL/naive drivers; `src/avx2/build_sweep.sh` +
   `src/avx2/gen_kernels.py` regenerate the microkernel-shape sweep;
   `src/harness/harness_sweep.py` drives the `perf`-wrapped hardware-counter
   runs behind Part E / Section 7.2.

Every script above is also listed, with its role, in `FINDINGS.md`
Part K (repo map) -- that part is the fastest way to find "which file
produced this number" for anything not listed above.

## What is *not* yet reproducible from a cold clone

- No CI / one-command "run everything" script exists yet -- each
  directory's own `build.sh`/`run.sh` must be run individually, in the
  order implied by its own README (e.g. `src/baselines/` must build before
  anything compares against it).
- `Research/reference-libs/` (full local clones of OpenBLAS/BLIS/
  oneDNN, used for source inspection in Section 7.1/7.3) is
  gitignored and lives one level above this repo -- re-clone those
  upstream projects directly if you need to re-run the `src/reference/`
  comparison from scratch rather than read the curated excerpts
  already checked in.
- oneMKL requires accepting Intel's own license and installing it
  separately (apt oneAPI repo on the dev machine, already present at
  `/opt/intel/oneapi/mkl/2022.1.0` on `glogin`) -- not redistributed
  in this repository.

## Still open (tracked honestly, not hidden)

- **No DOI'd release yet.** Recommended next step: `git tag paper-v1
  2b78eaa && git push --tags`, then connect this GitHub repo to Zenodo
  (or upload a tagged source archive directly) to mint a DOI -- cheap,
  and the citation-compounding benefit (`FINDINGS.md` Part L.3)
  starts accruing the moment it's public, independent of where the
  paper itself is submitted.
- **`reproduce.sh`/`reproduce.ps1` cover the portable subset only.**
  Production-BLAS baselines (`src/baselines/`), the full microkernel sweep
  (`src/avx2/sweep/`), hardware-counter root-cause analysis (`src/harness/`),
  and every multi-hour full sweep behind the paper's tables still need
  their own directory's script run by hand -- see above.
- Statistical rigor is uneven across sections -- `paper/main.tex`
  Section 8 states exactly which claims meet the $n{\geq}4$--$5$-repeat
  protocol (Section 3.2) and which do not yet. Re-running anything
  above a small number of times, per that protocol, before trusting a
  single run is part of using this repository correctly, not optional
  polish.

## Update 2026-10-13: AVX-512 pipeline and the n=30 standard

- **Protocol:** every run now follows [`PROTOCOL.md`](PROTOCOL.md) -- n=30
  independent process launches per measured point (`reproduce.sh --runs`,
  `reproduce.ps1 -Runs`, default 30; `--smoke` = n=3, wiring check only),
  bootstrap 95% CIs, Holm-corrected tests. The earlier "best-of-N, directional"
  numbers elsewhere in this repo predate it.
- **AVX-512 now runs:** `bash avx512/run_avx512.sh` (or `reproduce.ps1 -Avx512`)
  builds the audited AVX-512 kernel, gates on `test_local.c`, benchmarks AVX-512
  vs AVX2 on the same machine, and writes
  `revalidation/benchmarks/<date>_<host>-<cpu>-avx512*/`. First data:
  `2026-10-13_ceburu03-11th-Gen-i5-1145G7-avx512-n30-focused/` (FINDINGS Part N).
- **Cross-machine table:** `python3 tools/combine_results.py [--extra DIR ...]`
  -> `results/combined/` (long CSV, best-by-machine, run_info, optional .xlsx).
- Adds a machine: i5-1145G7 (Tiger Lake, 4C/8T, 16 GB, WSL2 Ubuntu 26.04, gcc 15.2).
- **Full-grid n=30 data (2026-10-19):** `revalidation/benchmarks/2026-10-19_ceburu03-11th-Gen-i5-1145G7-avx512-n30-full/`
  and `results/reproduce_20261019_130645/`, produced by `bash run_full_suite.sh` (sequential: AVX-512 grid, then
  `reproduce.sh --full`, then merge; ~3.7 h on the i5-1145G7). Shell scripts must keep LF line endings
  (`.gitattributes` enforces this); CRLF breaks them under WSL.
