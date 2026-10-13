# This project's own implementation work (Generation 2)

Everything under `src/` is new code written *for* the audit and
follow-on research described in `../FINDINGS.md` and `../paper/`, as
opposed to `../audited_submission/`, which is the original artifact
being studied. Nothing here existed before this project; none of it is
"the paper" either — it's the implementation behind the paper's claims.

| Directory | Role |
|---|---|
| `avx2/` | A from-scratch AVX2 DGEMM built and tuned independently of `audited_submission/avx/`, including the microkernel sweep (`avx2/sweep/`), hand-written-assembly kernels (`avx2/asm_kernel/`), the portable auto-tuning harness (`avx2/portable/`), and `avx2/ladder/` — the phase-timed measurement-bias harness (paper §5) plus the classical and Dumas et al. (2024) Strassen rebuilds (paper §6). |
| `strassen2/` | A second, differently-designed classical-Strassen rebuild (pooled workspace, fused adds, strided quadrant views — paper §7.4's crossover table). |
| `baselines/` | Thin drivers calling `cblas_dgemm` from OpenBLAS/BLIS/oneMKL with the same CLI/timing contract as `avx2/`, so numbers are directly comparable across libraries. |
| `bench/` | The shared benchmark harness (CLI, GFLOP/s, reference-check) used by `avx2/`, `baselines/`, and `strassen2/`. |
| `accuracy/` | Numerical-accuracy experiments (paper §7.4's catastrophic-error finding): uniform/wide-dynamic-range/clustered-cancellation distributions checked against a long-double reference. |
| `roofline/` | STREAM-triad bandwidth microbenchmark and the roofline operational-intensity model. |
| `topology/` | Native-Windows P-core/E-core/SMT classification and scaling tools — used throughout paper §5/§8 as this single-socket machine's NUMA substitute. |
| `harness/` | Opt-in call/byte-count instrumentation (`count_calls.h`) and a `perf`-wrapping sweep driver (`harness_sweep.py`), used for the hardware-counter root-cause analysis in paper §7.2. Included from `../audited_submission/` via `../../src/harness/count_calls.h` — yes, across the generation boundary; see the note below. |
| `reference/` | Curated excerpts of OpenBLAS's and BLIS's actual GEMM kernels/blocking parameters (with license + source-commit provenance), kept for direct comparison against this project's own kernel design — not redistributing either library in full. |

## Why `harness/` lives here, not with the audited code it instruments

`harness/count_calls.h` is included from eight files inside
`../audited_submission/` (`avx/strassen.c`, `avx/strassen_avx2.c`, three
`Parallel/mm_parallel_strassen*.c`, three `Serial/mm_strassen*.c`) via
`#include "../../src/harness/count_calls.h"`. That's real,
compiler-verified coupling across the Generation 1/2 boundary — the
instrumentation is *this project's* opt-in addition (`-DCOUNT_CALLS`,
confirmed to add no behavior/timing change when the flag is absent; see
`FINDINGS.md` Part E), bolted onto the original code without modifying
its own logic, which is exactly why it belongs on this side of the
split rather than being treated as part of the original submission.

## Build order

`bench/` and `baselines/` have no dependencies within this tree and
build first; everything else here (`avx2/`, `strassen2/`, `accuracy/`)
depends on one or both. See each subdirectory's own `build.sh`/`run.sh`
and `../REPRODUCIBILITY.md` for the exact order and commands.
