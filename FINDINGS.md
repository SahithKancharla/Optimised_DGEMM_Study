# Findings Log (Living Document — Under Active Validation)

This file tracks verified findings from two complementary, concurrent
workstreams on this repo: **(A)** building and optimizing a new AVX2
multilevel DGEMM implementation and benchmarking it against production BLAS
(OpenBLAS/BLIS/MKL), and **(B)** auditing the *original* AVX-512 codebase
(`audited_submission/avx/`, `audited_submission/Parallel/`, `audited_submission/test/`) for correctness bugs and revalidating the
paper's own performance claims on non-target hardware, backed by hardware
performance counters where available. Both feed the same paper revision.
Everything below is either a direct measurement, a disassembly/`perf`/source
inspection, or a clearly-labeled hypothesis. Numbers are averages over
repeated runs where noted; single-run numbers are labeled as such and should
not be treated as stable.

**Status:** in progress, thirteen parts (A-M). Quick index of what each
covers and where things currently stand — see the part itself for full
detail and evidence; this paragraph is deliberately just a map, not a
restatement.

- **Part A** — environment reality check across every machine used (dev
  machine WSL2/Hyper-V, native Windows, `glogin.cs.vt.edu` KVM): none is
  the paper's original Cascade Lake cluster; no AVX-512 anywhere; `perf`
  hardware counters unavailable under WSL2 entirely, available but
  LLC-event-limited on `glogin`.
- **Part B** — new AVX2 DGEMM built from scratch (`src/avx2/`) and tuned
  against OpenBLAS/BLIS/MKL: microkernel sweep, a disassembly-driven
  ablation series (operand-swap, unroll+prefetch, shape selection,
  hand-written assembly, vectorized packing) took it from ~76% to ~87-92%
  of production BLAS — the architecture-level, transferable figure. A
  dynamic-scheduling fix on top measured 117-130% (§B.8) and a portable/
  auto-tuning redesign measured ~4x over a properly-rebuilt OpenBLAS on
  `glogin` (§B.12) — both explicitly flagged there as possibly
  environment-specific, not yet proven architectural. §B.13 adds a
  native-Windows P/E-core/SMT substitute-NUMA study.
- **Part C** — four real correctness bugs found and fixed in the
  *original* AVX-512 codebase (`audited_submission/avx/`, `audited_submission/Parallel/`, `audited_submission/test/`), each with
  real execution evidence, not just reading.
- **Part D** — the paper's own performance claims re-tested on two
  non-target machines: the parallel-Strassen-degrades finding reproduces
  cleanly; the "blocked wins for n>512" claim doesn't, and turns out to
  be scoped to the AVX-512-specific comparison, not the algorithm choice
  generally.
- **Part E** — hardware-counter root-cause analysis (`glogin`) explaining
  *why* each Part D result happens (IPC, cache/branch-miss mechanisms),
  not just reporting that it does.
- **Part F** — the open-threads list (kept current as parts get closed;
  read it directly rather than summarizing it here, since it changes).
- **Part G** — a portable (MSVC+gcc) phase-timed harness testing whether
  §B.8's big scheduling win is real hardware behavior or a WSL2/Hyper-V
  artifact. Headline result: once properly *repeated* (not single-run),
  the effect did not reproduce on native Windows, including in the
  heterogeneous P+E-core case theory favors it most.
- **Part H** — a first Strassen rebuild (`src/strassen2/`, pooled workspace,
  fused adds, 4-bare-view quadrant handling, thread-aware cutoff) against
  a near-peak (~427-467 GFLOP/s) baseline: crossover at N≈8192.
- **Part I** — a first numerical-accuracy sweep against that rebuild: a
  dramatic finding — Strassen's error exceeds 1400% under row-scaled
  ill-conditioning at N=1024, worsening with N and recursion depth.
- **Part J** — independent, parallel work (built without reading H/I
  first) that turned out to complement rather than duplicate them: J.1
  extends Part G's repeated-measurement rigor to WSL2 and `glogin`
  themselves (same result: no reproducible effect, in either); J.2-J.6
  close out phase decomposition, the blocking response surface, huge
  pages, GCC/Clang/icx comparison (a real, confirmed difference: LLVM
  auto-unrolls, gcc doesn't), confirmed-not-assumed VM hardware limits via
  VTune, and rectangular/irregular shapes (a real ~2x performance cliff at
  off-tile-boundary sizes); J.7-J.8 are a *second*, differently-designed
  Strassen rebuild and accuracy sweep, whose different results against
  Parts H/I are explained rather than left as a contradiction (J.7: the
  crossover point depends on how optimized the baseline is, not just on
  the algorithm; J.8: per-entry random scaling doesn't trigger the same
  catastrophic error row-correlated scaling does — a real methodological
  lesson for accuracy-test design).
- **Part K** — repo map.
- **Part L** — related-work landscape and a publication/citation strategy:
  what's already published in this space (GotoBLAS/BLIS, Strassen Reloaded,
  communication-optimal parallel Strassen, the brand-new 2024 accuracy-
  optimized Strassen variant, measurement-bias methodology, reproducibility/
  artifact-evaluation trends, LLM-driven kernel generation), where Parts
  A-K's actual work is genuinely novel against that landscape (not just
  "more of the same"), and concrete recommendations on framing, venue, and
  what to add next to maximize how much this gets built on and cited.
- **Part M** — Part L.4's follow-up experiment actually done: the 2024
  Dumas/Pernet/Sedoglavic alternate-basis Strassen variant, implemented
  as a real recursive block algorithm on the identical arena/task/phase-
  timing infrastructure as the classic-Strassen rebuild, symbolically
  verified against the paper's own straight-line program (a real
  transcription bug caught before any C was written), then run for both
  correctness (machine-precision, N=64-512) and a head-to-head accuracy
  *and* performance comparison against classic Strassen at matched
  depth. Result: measurably better Frobenius-relative accuracy at every
  matched depth (qualitatively confirms the paper), at a measurable and
  now-quantified performance cost (~20% fewer GFLOP/s at N=2048,
  depth=3, isolated specifically to the combine/scale phase, not the
  shared leaf kernel) — a number neither the original paper nor (per
  Part L.1's search) anyone else has published.

**What's still open, genuinely** (not just unstated): a literal dedicated
bare-metal Linux machine (native Windows in Part G/J.1 is the closest
substitute reached so far); Part G's repeated-measurement rigor applied to
the *original* `matmul_asm_v4_dynsched.c` kernel itself, not just the
simpler portable one J.1 tested; RAPL power (confirmed unavailable on
every VM tested; not checked on native Windows); a memory-hierarchy model
beyond `glogin`'s PMU limits (confirmed by VTune itself to be a hard VM
wall, not a tooling gap); AVX-512 execution (no hardware reachable, at
all, this entire session); a reproducibility pack (Part L.3 gives the
concrete recipe: Zenodo DOI + RCR-style README) and the actual paper
rewrite (Part L gives the framing and headline-contribution choices, not
the prose itself).

---

# Part A — Environment & Cross-Machine Reality Check

- **This machine is not the machine the original paper describes.** The
  paper's "32-core AVX-512 platform" (dual-socket Intel Xeon Gold 5218,
  Cascade Lake) was a remote cluster node; the primary machine used for both
  workstreams here is a single-socket Windows 10 desktop, Intel Core
  i7-13700K (8 P-cores + 8 E-cores, 24 logical processors, 16 physical
  cores), 64GB RAM.
- **No AVX-512 hardware was reachable in either workstream.** Confirmed via
  `/proc/cpuinfo` on the local machine: only `audited_submission/avx`, `src/avx2`, `fma` flags
  present. Intel fuses off AVX-512 on Raptor Lake consumer parts once
  E-cores are active. The existing `audited_submission/avx/` AVX-512 code cannot run natively
  on this CPU; the AVX-512 code has been correctness-*reasoned*-about and
  compile-checked (`-march=skylake-avx512`) but never executed since the
  fixes in Part C went in (see Part F).
- **Single socket, single NUMA node** (`lscpu`: 1 socket, 1 NUMA node) — the
  paper's dual-socket NUMA experiments are not reproducible on this hardware
  at all, not just a configuration issue.
- **WSL2/Hyper-V cannot isolate P-core vs E-core identity.** A pure-register
  AVX2 FMA microbenchmark (`src/topology/classify_cores.c`) pinned via `taskset`
  to each of the 24 logical CPUs inside WSL2 showed a **flat ~70-71 GFLOP/s
  everywhere** — no bimodal split, meaning Hyper-V's scheduler does not honor
  a stable guest-vCPU-to-physical-core mapping. The **same test run natively
  on Windows** (via `SetThreadAffinityMask`, `src/topology/classify_cores_native.c`)
  showed a clean split: **logical CPUs 0-15 ≈ 33-34 GFLOP/s (P-cores, HT
  pairs), logical CPUs 16-23 ≈ 17 GFLOP/s (E-cores, no HT, ~half throughput)**.
  Implication: any P-core/E-core or SMT-on/off substitute-NUMA experiment
  must run natively on Windows, not inside WSL2.
- WSL2 also **flattens the true P/E topology in `lscpu -e`** itself: it
  reports a fake uniform "12 cores × 2 threads" layout with identical
  per-core cache indices, not the real asymmetric 8P+8E layout — the
  empirical classifier above (not `lscpu`), is the trustworthy source for
  core identity.
- **`perf` hardware counters are unavailable under WSL2 — confirmed
  independently by both workstreams.** Workstream A's setup-script sanity
  check and workstream B's direct `perf stat` testing (every event reports
  `<not supported>`, even under `--privileged` Docker) agree. Workstream A's
  roofline/cache-miss analysis therefore uses a STREAM-style bandwidth
  benchmark instead (Part B §7); workstream B instead moved its
  hardware-counter analysis to the VT node, where real PMU access exists
  (see below).

### Second machine: `glogin.cs.vt.edu` (`kepler.rlogin`)

Reachable via SSH; used differently by each workstream but the underlying
facts are shared:

- **Not the paper's original cluster.** It's a KVM-virtualized 8-vCPU slice
  (2 vSockets × 4 cores, no HT) of an Intel Xeon **E5-2630 v3 (Haswell-EP,
  2015)** — predates AVX-512 entirely (`audited_submission/avx`/`src/avx2`/`fma` only, confirmed via
  `/proc/cpuinfo`).
- No sudo, no job scheduler (no Slurm/PBS/LSF found). System-wide
  `libopenblas.so.0` and ATLAS already installed. **gcc 11.5.0 — matches the
  original paper's compiler version exactly**, a useful cross-check
  workstream B relied on.
- **Real hardware PMU counters ARE obtainable here** (unlike WSL2) — this is
  where workstream B's entire hardware-counter root-cause analysis (Part E)
  was collected. Its PMU does not expose LLC-specific events, however
  (`<not supported>` for all of them, unlike the paper's Cascade Lake target)
  — generic `cache-references`/`cache-misses` and `L1-dcache-*` were
  substituted.
- Workstream A's plan: used as-is for a lightweight, reduced-scope
  cross-architecture DGEMM data point (not NUMA — reports 1 NUMA node
  despite 2 vSockets, a hypervisor artifact) — **not yet executed** for the
  new AVX2 DGEMM work at time of writing.
- Workstream B already ran a reduced-grid performance revalidation and the
  full hardware-counter harness here — see Parts D and E.

---

# Part B — DGEMM Microkernel Optimization vs Production BLAS

*(Workstream A: building/tuning a new AVX2 multilevel DGEMM from scratch,
independent of the original `audited_submission/avx/` codebase, benchmarked against
OpenBLAS/BLIS/MKL.)*

## B.1 Baseline setup (WSL2 Ubuntu 24.04)

- Installed: gcc 13.3.0, `libopenblas.so.0` (apt), BLIS (built from source,
  `--enable-cblas -t openmp`), Intel oneMKL 2026.1 (apt oneAPI repo), Python
  venv with numpy/pandas/matplotlib/scipy.
- MKL ships `libmkl_avx2.so.3` confirming it dispatches an AVX2 code path on
  this CPU (no AVX-512 available).
- Correctness: all four drivers (naive/OpenBLAS/BLIS/MKL) verified against a
  long-double-accumulation reference at ~1e-16 relative Frobenius error.
- Cache sizes (P-core, `cpu0`): L1d=48K, L1i=32K, L2=2048K (2MB), L3=30720K
  (30MB shared) — used as the starting point for blocking-parameter tuning.

## B.2 Reference library comparison (`src/reference/`)

Curated excerpts (not full clones — full source kept locally outside the
repo at `Research/reference-libs/`, gitignored) from OpenBLAS and BLIS, with
license + exact source commit for attribution. MKL has no public source.

| Library | MR | NR | MC | KC | NC | Implementation |
|---|---|---|---|---|---|---|
| This study (best, see B.6) | 6 | 8 | 576 | 1024 | 1024 | hand-asm + vectorized packing |
| OpenBLAS (`KERNEL.HASWELL`) | 4 | 8 | 512 | 256 | dynamic | hand-written x86-64 assembly |
| BLIS (`config/haswell`) | 6 | 8 | 72 | 256 | 4080 | hand-written AVX2 assembly |

- **Our independently-swept AVX2 C-intrinsics winner (4×8, see B.3) exactly
  matches OpenBLAS's shipped Haswell kernel shape** — real independent
  validation of the register-pressure reasoning behind the sweep.
- BLIS ships 6×8 in production; its source tree still contains superseded
  8×6, 4×12, 12×4 variants under `kernels/haswell/3/old/` — the same shape
  space our own sweep covered independently.
- BLIS's MC=72 (much smaller than OpenBLAS's 512 or our tuned 256-576) comes
  from a published analytical cache/TLB-associativity model (Low et al.,
  "Analytical Modeling Is Enough for High-Performance BLIS"), not empirical
  search — a concrete unexplored lever if further tuning is wanted.
- Even OpenBLAS's Haswell config selectively uses newer `*_skylakex.c`
  copy/beta routines, suggesting the packing step itself is a nontrivial
  share of any performance gap, not just the microkernel (see B.6).

## B.3 AVX2 microkernel sweep (`src/avx2/sweep/`)

AVX2's 256-bit registers hold 4 doubles, so `MR` must be a multiple of 4 for
a vector-per-row-group design (the design used before the operand swap in
B.4) — tested shapes: 4×8, 4×12, 8×4, 8×6, 8×8, 12×4, 12×6, 16×4, 16×6,
including deliberately over-register-budget shapes to observe spilling.

- Winner: **4×8**, at both 1 thread (53.5 GFLOP/s) and 16 threads (298.4
  GFLOP/s at N=1536/2048 respectively) — beat the runner-up (4×12) by
  ~20-46% depending on thread count.
- Staged (one-dimension-at-a-time) blocking tune for the 4×8 shape
  converged on KC=512, MC=256, NC=1024 (~320 GFLOP/s at N=2048/16 threads).

## B.4 Root-cause analysis: why we underperformed OpenBLAS/BLIS, and the fix chain

Diagnosed by disassembling our own compiled kernel and comparing
instruction-for-instruction against OpenBLAS's real assembly
(`src/reference/openblas/dgemm_kernel_4x8_haswell.S`):

1. **Wrong broadcast operand.** Original kernel broadcast B (8 separate
   loads) and vector-loaded A (1 load) = 9 loads for 8 FMAs. OpenBLAS
   broadcasts the smaller operand (A, 4 loads) and vector-loads the larger
   (B, 2×4-wide) = 6 loads for 8 FMAs. Confirmed via `objdump`: our compiled
   kernel showed exactly 1 `vmovup` + 8 `vbroadcastsd` + 8 `vfmadd231pd`
   per k-step, in a genuinely non-unrolled loop (`jne` back to top every
   step).
2. **No loop unrolling / software pipelining** — a real rolled loop, branch
   every k-step, vs. OpenBLAS's manually unrolled, latency-hiding code.
3. **Zero prefetching** vs. OpenBLAS's `prefetcht0` at multiple offsets
   every iteration.
4. **Repeated OpenMP fork/join** — 3 separate `#pragma omp parallel for`
   sites reopened every macro-block iteration (64x at N=2048's tuned
   blocking) vs. a persistent thread team.
5. **Different blocking philosophy** — analytical cache/TLB models
   (OpenBLAS/BLIS) vs. our empirical staged search, and a swapped MC/KC
   priority relative to OpenBLAS.

### Fix ablation series (all files kept as separate, comparable methods)

Each fix applied incrementally, each verified correct (~1e-15 relative
Frobenius error against the long-double reference) and measured at
N=2048/16 threads before moving to the next:

| File | Fix applied | Result (GFLOP/s) |
|---|---|---|
| `src/avx2/matmul_4x8.c` (v1) | original 4×8, broadcast-B | 298-317 (noisy) |
| `src/avx2/matmul_v2_swap.c` | broadcast-A / vector-B swap | **regressed** to 240 with a naive stack-round-trip write-back; **334** once fixed to an in-register 4×4 transpose write-back, then **fixed again** to flat named accumulators instead of a 2D array (GCC was silently spilling accumulators to the stack every k-step for the `[4][2]` array shape — confirmed via `objdump` `vmovapd` to `(%rsp)` every iteration; a `[1][8]` array in v1 didn't trigger this) |
| `src/avx2/matmul_v3_unroll_prefetch.c` | manual 2-way k-unroll + prefetch (PF_DIST=4) | 355 |
| `src/avx2/matmul_v4_threading.c` | single `#pragma omp parallel` region (orphaned `omp for` worksharing) instead of 3 reopened regions | **no measurable change** (353 vs 355) — honest negative result; libgomp's thread-pool re-entry cost was cheap enough at this scale that fork/join wasn't the bottleneck it looked like on paper |
| — | retuned blocking for the new kernel (KC=512→ MC_TARGET 256→128) | 366 |
| `src/avx2/matmul_v5_unroll4.c` | 4-way k-unroll (tried) | **regressed** to 306 — over-unrolling hurt, abandoned |
| `src/avx2/matmul_v6_mr6.c` | MR=6/NR=8 shape (matches BLIS; MR no longer constrained to a multiple of 4 once broadcast, not vectorized) | 373 (retuned blocking KC=1024/MC=576/NC=1024) |
| `src/avx2/matmul_v7_4x12.c` | MR=4/NR=12 shape (best theoretical load:FMA ratio) | **worse**, 329-346 — abandoned |

Also tried and found **not helpful, within noise**: `OMP_PROC_BIND=true`/
`OMP_PLACES=cores` (367 vs 373); PF_DIST sweep 2/6/8/12 (all 351-373, no
clear trend).

**Critical measurement finding: run-to-run variance is large (~±9%, e.g.
341-373 GFLOP/s for the identical v6 binary across 5 runs).** A single
20-thread run once hit 400 GFLOP/s, but repeated runs at 20 threads ranged
339-401 — an unreliable, noise-dominated configuration, **not a real
result**, and the baselines (OpenBLAS 437, BLIS 458 at 20 threads) improve
even more at that thread count, so it doesn't close the relative gap
either. All "final" numbers in B.6/B.7 are 10-run averages, not single
samples.

## B.5 Hand-written assembly (`src/avx2/asm_kernel/`) — new, separate method

- `kernel_6x8.S`: hand-scheduled AVX2 assembly, MR=6/NR=8, manual register
  allocation (12 accumulators + broadcast/vector scratch), 2-way software
  pipelining, explicit `prefetcht0`. Verified correct in isolation first
  (`test_kernel_standalone.c`, kc=1..9, both the unrolled and odd-remainder
  code paths) before wiring into the full packed driver.
- **One real bug found and fixed:** used `vmovapd` (32-byte-aligned store)
  for the MR=6 remainder-row stack scratch, but the stack is only
  guaranteed 16-byte aligned at that point — caused a segfault in the full
  pipeline (didn't manifest in the standalone test, where the call site
  happened to leave the stack 32-aligned by luck). Fixed to `vmovupd`.
- `matmul_asm.c`: wires the assembly kernel into the same packed
  multilevel driver structure as v6. 10-run average: **342.3 GFLOP/s** —
  essentially tied with the C-intrinsics v6 (~345 avg over 5 runs), but
  with a visibly tighter run-to-run spread (344-361 vs 312-357). Honest
  finding: **hand-written assembly did not meaningfully beat well-written
  intrinsics here** — the microkernel-level instruction scheduling/register
  allocation was not, in fact, the dominant remaining gap.

## B.6 Roofline check (`src/roofline/stream_triad.c`)

- Measured DRAM bandwidth: **~77-85 GB/s** (8 threads slightly beat 16 —
  consistent with a dual-channel memory system saturating before all cores
  are needed).
- DGEMM at N=2048 has operational intensity ≈170 FLOPs/byte (naive
  traffic estimate) against a ridge point ≈8 FLOPs/byte (peak compute ÷
  measured bandwidth) — **solidly compute-bound, not bandwidth-bound**,
  which is consistent with (a) the hand-written assembly kernel not
  producing a big win (the bottleneck isn't instruction scheduling *or*
  memory bandwidth) and (b) packing overhead being the next real candidate.

### Vectorized packing (`src/avx2/asm_kernel/matmul_asm_v2_vecpack.c`)

- `pack_panelA`: contiguous source → straight AVX2 load/store (YMM for 4
  of MR=6, XMM for the remaining 2) instead of 6 scalar loads.
- `pack_panelB`: strided source (column stride `ldb`) → AVX2 gather
  (`_mm256_i64gather_pd`, 2 gathers of 4 lanes each) instead of 8 scalar
  loads.
- Result: **357.1 GFLOP/s (10-run average)** — a real ~4.3% gain over the
  scalar-packing assembly version (342.3), confirming packing overhead was
  a genuine, non-trivial contributor.
- Blocking re-tune for this version found a config (MC_TARGET=288) that
  looked better on a single-sample sweep but was **worse on a reliable
  10-run average** (350.9 vs 357.1 with the original KC=1024/MC=576/NC=1024)
  — another confirmation that single-sample tuning is not trustworthy in
  this environment; kept the original blocking as final.

## B.7 Packing follow-up: transpose-based B packing (`matmul_asm_v3_transpack.c`)

Tried replacing gather-based B packing (B.6) with an explicit in-register
4×4 transpose (same unpacklo/unpackhi + `permute2f128` idiom as the
microkernel's C write-back, applied to packing instead): read 4 columns of
B as 4 contiguous vector loads, transpose, store contiguously — avoiding
gather entirely, since AVX2 gather has historically mediocre throughput on
Haswell-class execution units. Result: **358.1 GFLOP/s (10-run average)** —
essentially tied with the gather version (357.1), within noise. Honest
negative result: the gather-vs-transpose choice didn't matter here; both
packing approaches were already good enough that this wasn't the bottleneck.

## B.8 Scheduling fix: dynamic scheduling — the biggest single win, with a caveat

Tried `schedule(dynamic,N)` in place of `schedule(static)` on the
microkernel's `#pragma omp for collapse(2)` loop (`matmul_asm_v4_dynsched.c`,
otherwise identical to v3). Motivation: Part A already established that
WSL2/Hyper-V does not give threads a stable, uniform mapping to physical
cores — under `schedule(static)`, whichever thread happens to land on a
slower/contended core becomes a straggler that the rest of the team must
wait for at every macro-block barrier. Dynamic scheduling lets faster
threads opportunistically grab more chunks instead of every thread getting
a fixed, equal share up front.

- Chunk-size sweep (10-run averages each): 1→501, 2→511, 3→485, 4→506,
  5→509, **6→516**, 7→496, 10→492, 12→482. Picked chunk=6 (highest in a
  fair same-size comparison; differences among 1-10 are within this
  environment's noise floor, so this isn't a sharply-tuned optimum, just a
  reasonable pick from a broad plateau).
- Blocking re-tune for the dynamic-schedule version found MC_TARGET=672
  (vs. 576 previously) but made **no measurable difference** (516.9 vs
  516.7) — kept the original, already-proven blocking.
- **Final, most reliable measurement (15-run averages, N=2048, 16
  threads):**

| Method | GFLOP/s | % OpenBLAS | % BLIS | % MKL |
|---|---|---|---|---|
| OpenBLAS | 432.7 | — | — | — |
| BLIS | 391.0 | — | — | — |
| MKL | 410.9 | — | — | — |
| **Ours (hand-asm + vectorized packing + dynamic scheduling)** | **506.8** | **117.1%** | **129.6%** | **123.4%** |

Also fixed the earlier run-to-run instability as a side effect: variance
dropped from the ~±9% swings seen throughout B.4-B.6 to a much tighter
spread once scheduling adapted to the environment's actual (inconsistent)
core assignment.

**Important honest caveat, not a result to overstate:** this win is very
plausibly specific to running inside WSL2/Hyper-V, not a demonstration that
our kernel is fundamentally better than OpenBLAS/BLIS/MKL's. Those
libraries' own threading models assume a stable, bare-metal-like core
mapping (much like our `schedule(static)` did) and have no equivalent
runtime adaptation to a hypervisor silently reshuffling which vCPU maps to
which physical core between — or even during — a run. On real bare-metal
Linux, where thread-to-core assignment is stable, this specific advantage
would likely shrink or disappear, and the honest comparison would probably
land back around the ~87-92% figure from B.6/B.7 (unfixed), reflecting the
actual microkernel/packing-level gap rather than a scheduling artifact.
**This must be confirmed on non-virtualized hardware before being reported
as "our kernel beats OpenBLAS/BLIS/MKL" in the paper** — as measured, it is
accurate for *this exact environment*, and that scoping should be explicit
wherever this number is used.

## B.9 Current state vs. production BLAS (best reliable measurements, N=2048, 16 threads)

| Method | GFLOP/s | % OpenBLAS | % BLIS |
|---|---|---|---|
| Original kernel (v1, broadcast-B, no unroll/prefetch) | ~313-317 | ~76% | ~81% |
| Best C-intrinsics (v6, 6×8, operand-swapped, unrolled+prefetched) | ~345 | ~84% | ~89% |
| Hand-written assembly (same 6×8 shape, scalar packing) | 342.3 | 83.0% | 88.0% |
| Hand-written assembly + vectorized/transposed packing | 357-358 | ~87% | ~92% |
| **+ dynamic scheduling (WSL2-environment-specific, see B.8 caveat)** | **506.8** | **117.1%** | **129.6%** |

Two separate claims worth keeping distinct in the paper: (1) the
microkernel/packing work alone (operand swap → unrolling/prefetch → shape
selection → hand-written assembly → vectorized packing) is a
disassembly-verified, architecture-level improvement that should transfer
to bare-metal hardware, taking us from ~76% to ~87-92% of production BLAS;
(2) the scheduling fix on top of that is a further, large, *measured* win
in this environment specifically, whose transferability to bare-metal is
an open question (§B.8, Part F).

## B.10 Baseline fairness audit: are OpenBLAS/BLIS/MKL being tested fairly on this system?

Given the dynamic-scheduling win in B.8 came from adapting to this
environment's own scheduling quirks, the obvious next question is whether
the *baselines* are being handicapped by the same quirks (or by
misconfiguration) in a way that makes the comparison unfair. Audited
directly rather than assumed:

- **OpenBLAS (apt package) is pthread-based, not OpenMP, and built with
  `NO_AFFINITY=1`** — confirmed via `openblas_get_config()`:
  `OpenBLAS 0.3.26 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64`,
  and via `ldd` showing no `libgomp` dependency. `Makefile.rule` in the
  OpenBLAS source confirms these are the project's own build defaults
  (`NO_AFFINITY = 1` active, `USE_OPENMP` commented out/disabled) — not
  something the apt packager did wrong.
- **Its "Haswell" kernel target is correct, not a misdetection.**
  OpenBLAS's `TargetList.txt` has no x86_64 target between Haswell and
  SkylakeX (which requires AVX-512, unavailable here) — Haswell genuinely
  is the newest kernel OpenBLAS ships for an AVX2-without-AVX-512 CPU like
  this one, so there's no "better target" being missed.
- **Rebuilt OpenBLAS from source** (`TARGET=HASWELL USE_OPENMP=1
  NO_AFFINITY=0`) to test the hypothesis that pthread+no-affinity was
  costing it the same straggler penalty our own kernel had before B.8.
  Confirmed via config string (`USE_OPENMP HASWELL`) that the rebuild took.
  Result: essentially a wash (10-run avg 426.9 default schedule, 421.7 with
  `OMP_SCHEDULE=dynamic,6` — actually very slightly *lower* than the
  original pthread build's 432.7 avg). **OpenBLAS's own internal work
  partitioning already avoids the pathological straggler problem our naive
  flat `omp for collapse(2)` loop had** — its GEMM driver doesn't parallelize
  via a single flat schedulable loop the way ours did, so there's no
  equivalent fix to apply, and the original number was not artificially
  suppressed.
- **BLIS and MKL don't respond to `OMP_SCHEDULE` at all** (BLIS: 400.8→389.1
  avg with `dynamic,6` set, i.e. slightly worse; MKL: 413.1→408.3, also
  slightly worse) — both use their own multi-level thread-communicator
  abstractions internally rather than a plain schedulable `omp for`, so the
  external env var has no effect either way.
- **Tried 7 manual BLIS thread factorizations** (`BLIS_JC_NT`/`IC_NT`/`JR_NT`
  combinations covering 16 threads: 4×4×1, 2×8×1, 8×2×1, 2×4×2, 4×2×2,
  1×16×1, 16×1×1) against BLIS's automatic default. All landed in the same
  383-396 GFLOP/s band as the default (391-400) — no manual factorization
  meaningfully beat BLIS's own automatic choice.

**Conclusion: no evidence any baseline is being suppressed by
misconfiguration or an easily-fixed environment interaction.** OpenBLAS,
BLIS, and MKL's measured numbers throughout this document appear to be
genuine reflections of their capability on this hardware/environment, not
artifacts of a "bad system" handicapping them specifically (the system's
general WSL2/Hyper-V scheduling noise, per Part A, affects everything
running here to some degree — including these libraries — but that's an
environment property, not something we did to them, and we found no lever
that meaningfully changes their numbers).

## B.11 Further research-driven attempts (web research + Intel oneDNN)

Prompted by a literature/web check on modern DGEMM optimization practice
(BLIS's analytical model paper, an independent AVX-512-to-AVX2 DGEMM
tutorial repo, the matrixmultiply Rust crate's AVX2 microkernel notes, and
Intel's own oneDNN source):

- **Intel oneDNN was cloned locally** (`Research/reference-libs/oneDNN`,
  gitignored, same pattern as OpenBLAS/BLIS) per the request to get
  Intel's own library and test with it. **Finding: oneDNN has no
  double-precision (f64/DGEMM) kernel at all** — its `src/cpu/x64/gemm/`
  only has `f32/`, `bf16/`, `s8x8s32/`, and `amx/` subdirectories, no `f64`.
  This makes sense in hindsight (oneDNN is deep-learning-focused, and DL
  essentially never uses double precision) but means it cannot serve as a
  DGEMM comparison baseline. **MKL remains the correct Intel DGEMM
  reference** and is already covered (Part B throughout).
- **C-tile prefetch, not just A/B** (an external DGEMM tutorial explicitly
  lists this as a technique): added `prefetcht0` for all 8 columns of the
  C tile at kernel entry, as early as possible before the k-loop
  (`kernel_6x8_v2_cprefetch.S`, `matmul_asm_v5_cprefetch.c`). Result:
  **no measurable change** (506.5 vs. 506.8 avg) — another honest negative
  result; C is written back only once per micro-tile call (not once per
  k-step like A/B), so there's little latency to hide there regardless.
- **BLIS's actual shipped analytical blocking values tested directly on
  our kernel** (KC=256, MC=72, NC=4080, from `src/reference/blis/`, rather
  than re-deriving the formula ourselves): **416.5 GFLOP/s avg — clearly
  worse** than our empirically-tuned KC=1024/MC=576/NC=1024 (506.8).
  Confirms blocking parameters are kernel-implementation-specific (BLIS's
  values are tuned for BLIS's own exact assembly/register/write-back
  behavior, not a universal constant) — our per-kernel empirical staged
  search remains the right approach here, not a blind port of another
  library's numbers.
- 8×4 (vectorize-A/broadcast-B, the *pre-operand-swap* convention at a
  different MR) was flagged by two independent sources (the tutorial repo,
  the matrixmultiply Rust crate) as a strong AVX2 shape — not yet tried
  under our current (post-swap) convention; listed in Part F.

**Net assessment of this round: the big win was already found (dynamic
scheduling, B.8); this round's further attempts (C-prefetch, borrowing
BLIS's blocking numbers) were honest negative results, and oneDNN turned
out not to apply to DGEMM at all.** Diminishing returns from further
microkernel/blocking micro-tuning at this point — the highest-value
remaining work is the bare-metal confirmation of B.8, not more parameter
search.

## B.12 A portable/generalizable harness, grounded in the autotuning literature (`src/avx2/portable/`)

Everything through B.11 was tuned specifically for the dev machine: KC/MC/NC
were hardcoded from an empirical sweep done there, and B.8's scheduling fix
used a hand-picked chunk size (`dynamic,6`) that happens to work well
because of *that machine's* particular scheduling noise. Neither is a
principled basis for a "we generalize to most systems" claim. Literature
review before redesigning (not just re-guessing):

- **ATLAS / "AEOS" (Automated Empirical Optimization of Software)** (Whaley
  & Dongarra) — the established paradigm for portable BLAS performance:
  search over blocking parameters empirically *on the actual target
  machine*, not once on a development machine. Competes with hand-tuned
  code specifically because it re-tunes per-machine.
- **Goto & van de Geijn, "Anatomy of High-Performance Matrix
  Multiplication"** — blocking parameters should be derived from *detected*
  L2 (and TLB) capacity, not a constant.
- **OpenMP scheduling literature** — `schedule(guided)` (large chunks
  early, shrinking exponentially) is specifically recommended for
  imbalanced/heterogeneous execution, with no tunable chunk-size parameter
  to re-guess per machine, unlike a hand-picked `dynamic,N`.

Built `src/avx2/portable/` accordingly, with `src/avx2/asm_kernel/` and every
`src/avx2/matmul_v*.c` kept completely unchanged as the machine-specific
baseline for comparison:

- `hw_detect.c/.h`: portable (Linux `/sys`) detection of L1d/L2/L3 sizes,
  cache line size, and available core count (`sched_getaffinity`, respects
  cgroup/taskset limits) — no hardcoded cache sizes.
- `blocking_model.c/.h`: Goto/BLIS-style analytical KC/MC/NC formula from
  *detected* cache sizes, then a small ATLAS-style empirical refinement
  (staged search, 3 candidates per dimension) around that estimate at
  first-run time, on a small calibration problem (N=512) — not the dev
  machine's numbers copied in.
- `matmul_portable.c`: the same kernel_6x8.S microkernel and transpose-based
  packing (both expected to be architecture-level, portable wins), but
  KC/MC/NC are now runtime variables set by the above, and the microkernel
  dispatch loop uses `schedule(guided)` instead of `dynamic,6`.
- `matmul_portable_staticctrl.c`: identical except `schedule(static)`, kept
  as an ablation control to isolate whether scheduling choice matters on a
  given machine.

### Cross-machine test: deployed to `glogin.cs.vt.edu` (real second machine, KVM not Hyper-V, different Xeon)

Copied source (not binaries) via `scp`, built fresh there with the VT
node's own gcc 11.5.0/glibc. **The portable harness auto-detected this
machine's actual, different cache sizes** (L1d=32K, L2=4096K, L3=16384K —
all different from the dev machine's 48K/2048K/30720K) and computed its own
appropriate blocking (analytical KC=256/MC=1020/NC=4096, calibrated to
KC=512/MC=1020/NC=4096 at N=512) — no manual intervention. Correctness
held (~1e-15/1e-16 relative error, matching every other measurement in this
document).

**An important self-caught mistake, corrected before it became a false
claim:** an early smoke-test comparison appeared to show our kernels
beating OpenBLAS by ~3.3x on this machine (59 vs 17.6 GFLOP/s at N=512).
Investigated *why* rather than accepting it — `openblas_get_config()` on
this system showed **`SINGLE_THREADED`**: OpenBLAS's build on `glogin` has
no threading support compiled in at all, so `OMP_NUM_THREADS=8` was silently
doing nothing and it was always running on 1 core, while our kernel used
all 8. Single-thread-to-single-thread, OpenBLAS was actually *faster* than
us there (14.7 vs 12.0 GFLOP/s) — the opposite of the apparent result.
**Fixed properly**: cloned OpenBLAS source directly on `glogin` (not
transferred from the dev machine, to avoid pushing 222MB over a shared
academic network link) and rebuilt with `TARGET=HASWELL USE_OPENMP=1
NO_AFFINITY=0`, confirmed via config string (`USE_OPENMP HASWELL
MAX_THREADS=8`).

**Corrected, fair, real multi-threaded comparison (N=1024, 8 threads, 5
runs each):**

| Method | GFLOP/s (avg of 5) | vs. real multi-threaded OpenBLAS |
|---|---|---|
| OpenBLAS (rebuilt, genuinely multi-threaded) | 23.2 | — |
| Our portable harness (guided, auto-tuned blocking) | 94.9 | **4.10x** |
| Our dev-machine-tuned hardcoded version | 97.9 | **4.23x** |

Both figures are noisy (portable: 76-113 range; hardcoded: 67-123 range;
OpenBLAS: 21-26 range) — `glogin` is a busy, shared, multi-user student
login node, and this variance is most likely genuine external contention
from other users' processes, a different noise SOURCE than the dev
machine's Hyper-V vCPU scheduling, but the same class of problem
(unpredictable per-thread progress). Rebuilding OpenBLAS with real
threading did not close the gap — its multi-core scaling on this
particular shared VM appears to be poor regardless of the threading
backend, for reasons not yet investigated (out of scope for this pass,
given shared-machine resource etiquette).

**Scheduling ablation on `glogin` (static-schedule control vs. guided,
N=1024, 5 runs each): no clear difference on this machine** (portable/
guided avg ~95 vs. static-control avg ~102, both within the same noisy
range) — unlike the dev machine, where the equivalent ablation
(`schedule(static)` in early versions vs. `dynamic,6` in B.8) showed a
large, clear, repeated difference. This suggests the *dev machine's*
scheduling win was specifically about Hyper-V's vCPU-to-core remapping
(which dynamic/guided scheduling can compensate for token-by-token), while
`glogin`'s variance (likely OS-level contention from other users' whole
processes, not per-thread core reassignment) isn't the kind of imbalance
loop-scheduling can fix — a genuinely different mechanism producing
superficially similar-looking noise. Worth stating precisely rather than
over-generalizing "adaptive scheduling helps everywhere": **it helps
specifically with sub-second-timescale core-mapping instability (measured
on the dev machine), and neither helps nor hurts against slower,
whole-process-level external contention (measured on `glogin`)** — the
portable harness's choice of `guided` remains the right conservative
default either way (self-adapting, no tunable parameter, never observed to
be worse than static).

**What this cross-machine test actually establishes:** the microkernel
(operand-swapped 6×8, hand-scheduled assembly) and packing (transpose-based)
work transfers to a real, independently-verified second architecture and
still beats a properly-configured, genuinely multi-threaded OpenBLAS by
~4x there. That is real, portable signal. It does **not** yet establish
that we beat OpenBLAS/BLIS/MKL "on most systems" in general — two
virtualized/shared machines, both with their own scheduling irregularities,
is not a representative sample of "most systems," and a quiet, dedicated,
non-virtualized machine (with fewer of the anomalies that make our
adaptive approach shine) remains untested. That gap is now Part F's top
item, updated below.

## B.13 P-core/E-core/SMT scaling (NUMA substitute) + measured peak GFLOP/s, native Windows

This machine has no real NUMA (single socket, Part A), so per the plan this
substitutes a heterogeneous-core scaling study using the P/E-core split.
Must run natively (WSL2 can't pin to a specific physical core, Part A) --
`src/topology/peak_scaling_native.c`, a multi-threaded version of the
register-only FMA microbenchmark, one thread per requested logical CPU,
pinned via `SetThreadAffinityMask`, no memory traffic (isolates compute
throughput specifically).

**Single run per configuration below — per Part G.4's lesson (learned
independently, after this table was collected), that's not yet enough to
trust precisely.** The P-vs-E-core gap here (~1.76x) agrees well with
Part G.4's own repeated-measurement result (~1.9-2x, pinned P vs pinned E,
5 repeats) and with Part A's isolated single-thread classifier (~2x) — three
independent measurements converging on the same ~2x figure is reassuring,
so that particular conclusion is likely robust. The SMT and all-cores/
combined-power-budget numbers below have not been repeated and should be
before being treated as precise.

| Configuration | Threads | Aggregate GFLOP/s |
|---|---|---|
| P-cores only, SMT on (cpu 0-15) | 16 | 358.9 |
| P-cores only, SMT off (cpu 0,2,4,...,14) | 8 | 206.0 |
| E-cores only (cpu 16-23) | 8 | 116.9 |
| All cores, SMT on (cpu 0-23) | 24 | 426.3 |
| All physical cores, SMT off (P even + all E) | 16 | 285.9 |

Findings:
- **SMT gives a real 1.74x on P-cores** for this pure-compute workload
  (358.9 vs 206.0) — a single hardware thread per P-core doesn't saturate
  its 2 FMA ports; a second hardware thread fills the idle issue slots.
- Per-thread throughput under full load: P-core ~25.75 GFLOP/s, E-core
  ~14.6 GFLOP/s (ratio ~1.76x) — lower than the earlier isolated
  single-thread measurement (33 vs 17, Part A) for both core types,
  consistent with Intel's shared power/thermal budget scaling down
  per-core clocks as more cores activate simultaneously.
- **All-24-threads (426.3) is LESS than P-only-SMT-on + E-only summed
  separately (358.9+116.9=475.8)** — running P and E cores simultaneously
  measurably reduces aggregate throughput vs. running them in isolation,
  a real, measured power/thermal-budget-sharing interaction, not
  something a naive "add up the parts" peak-FLOPs estimate would predict.
- All-16-physical-SMT-off (285.9) confirms SMT is a net win here, not just
  on P-cores alone.

**Important discrepancy, flagged rather than hidden:** this native,
per-thread-pinned measured peak (426.3 GFLOP/s, pure FMA, zero memory
traffic) is *lower* than some of the WSL2-measured DGEMM throughput
numbers elsewhere in this document (506-527 GFLOP/s, §B.8/§B.9) — which
should be physically impossible, since real DGEMM (with packing, memory
traffic, and non-FMA instructions mixed in) cannot legitimately exceed a
machine's pure-compute ceiling. This points to **WSL2/Hyper-V's virtualized
timers being potentially untrustworthy for absolute wall-clock
measurements**, on top of the already-documented scheduling-noise issues
(Part A). The *relative* comparisons within this document (our kernel vs.
OpenBLAS/BLIS/MKL, all measured identically inside the same WSL2 session)
remain internally consistent and likely still valid, but **the absolute
GFLOP/s figures reported anywhere in this document that came from WSL2
should not be trusted at face value** until cross-checked on bare-metal
Linux — this reinforces, with a second independent line of evidence, why
that bare-metal confirmation (already Part F's top item) is necessary
before any of these numbers go in the paper as absolute claims.

Using 426.3 GFLOP/s as the empirically-measured (not theoretically
estimated) peak: our best WSL2-measured DGEMM figures would compute to
>100% of peak, which is exactly the tell that those particular absolute
numbers are not trustworthy as stated — a %-peak table is deliberately
*not* presented here for that reason, rather than presenting a misleading
one. This should be redone on bare-metal alongside the scheduling
confirmation.

---

# Part C — Correctness Audit of the Original AVX-512 Implementation

*(Workstream B: auditing `audited_submission/avx/`, `audited_submission/Parallel/`, and `audited_submission/test/` — the paper's
own codebase — against `audited_submission/Advance_Parallel_Programming_Dense_Analysis.pdf`.
Full evidence — command output, backtraces, before/after — in
`revalidation/correctness/2026-09-09_bug_evidence_and_fixes.md`.)*

| # | Bug | File(s) | Root cause | Fix |
|---|---|---|---|---|
| 1 | `matmul()` computed `C = B·A`, not `A·B` | `audited_submission/avx/matmul.c` | Packing routines read A/B with contraction and output-row/col roles swapped; the existing `audited_submission/test/main.c` check only passed because its own test matrices are symmetric (a false-positive test) | Rewrote `pack_panelA`/`pack_panelB`/`pack_blockA`/`pack_blockB` with explicit row-major offsets; added a single O(N²) transpose pass at the end of `matmul()` to correct for `kernel_24x6`'s fixed tile orientation |
| 2 | Data race in the blocked parallel kernel | `audited_submission/Parallel/mm_parallel_blocked.c`, `mm_parallel_blocked_unrolled.c` | `collapse(3)` parallelized over `ii,jj,kk` — including the reduction dimension — with unsynchronized `+=`. Only "worked" because GCC's static scheduling happened to keep the racy iterations temporally non-overlapping | `collapse(3)` → `collapse(2)`; verified correct even under adversarial `schedule(dynamic,1)` |
| 3 | Silent data loss for non-power-of-2 N > 128 | `audited_submission/avx/strassen.c` | `strassen_rec` halves N with no remainder handling, unlike the Serial/Parallel Strassen variants which all pad first | Pad to next power of two before recursing |
| 4 | Segfault on a clean checkout | `audited_submission/test/main.c`, `audited_submission/test/run.sh` | Created `audited_submission/Data/` but not `Values/`/`Results/`; every `mm_*` function `fprintf`s to an unchecked `fopen()` result | Added the missing `mkdir()` calls |

**Known, not fixed**: `strassen()` segfaults if called directly on N ∈ {1,2}
(AVX2) or {1,2,4} (AVX-512) — the no-padding fast path feeds a
smaller-than-vector-width N straight into the register-blocked base-case
kernel. Outside the paper's tested range (N≥64); not touched.

All fixes were verified with **real execution**, not just reading: a scalar
stand-in kernel (`kernel_scalar.c`/`strassen_scalar.c`, same packed-buffer
contract, no intrinsics) let the actual `matmul.c`/`strassen.c` logic run on
any CPU during the audit itself — a temporary verification aid, not kept in
the repo afterward (its command output is preserved as evidence in
`revalidation/correctness/2026-09-09_bug_evidence_and_fixes.md`). What *is*
permanent is a real AVX2 port (`audited_submission/avx/kernel_avx2.c`, `audited_submission/avx/strassen_avx2.c` —
confirmed via `objdump` to emit genuine `vfmadd*pd`), which gives real
vectorized correctness data without needing AVX-512 hardware and can be
re-run any time via `audited_submission/avx/run_local_avx2.sh`.

## C.6 Second-pass whole-repo bug sweep (`/code-review max --fix`)

A repo-wide review pass at max effort with auto-fix, covering every
directory (not just Workstream B's own files), found and fixed 10 real
bugs, with 5 more documented but deliberately left unfixed (reasoning
below). Verified: normal-case runs unaffected, previously-unsafe inputs
now fail cleanly instead of corrupting memory or hanging.

**Fixed:**
- **Heap overflow for N≤0 and small powers-of-two** in `audited_submission/avx/strassen.c`
  and `audited_submission/avx/strassen_avx2.c` — the next-power-of-two padding added for
  Part C's bug #3 didn't floor at the base-case vector width, so
  `strassen(A,B,C,0)` or `strassen(A,B,C,4)` (AVX-512) / `N=2` (AVX2)
  reached the vectorized base case on a too-small buffer. **This was a
  regression introduced by Part C's own earlier fix**, only surfaced by a
  more thorough pass — worth noting as a lesson about re-reviewing fixes,
  not just new code. Fixed by flooring the padded size at the base-case
  width; verified across N = -1..1024.
- **Infinite recursion at n=1** in `audited_submission/Parallel/mm_parallel_strassen_naive.c`
  and `audited_submission/Serial/mm_strassen_naive.c` — the padding left `size=1` reachable,
  whose only base case is `n==2`, so it recursed on `k=n/2=0` forever
  (stack overflow). Floored the padded size at 2.
- **Unchecked Win32 calls in `src/topology/peak_scaling_native.c` and
  `classify_cores_native.c`**: `WaitForMultipleObjects` uncapped past its
  hard 64-handle limit (silent torn results above that), a bit-shift
  overflow for cpu index ≥64 (UB, wraps to the wrong core), and
  `SetThreadAffinityMask`/`CreateThread` return values discarded (silent
  wrong-core-or-unpinned results). All four checked now; out-of-range
  input fails loudly (exit 1, clear message) instead of silently
  mismeasuring — verified directly (`peak_scaling_native.exe 99` now
  reports the error and exits 1 rather than printing a bogus number).
- **Serial O(N²) transpose bottleneck** in `audited_submission/avx/matmul.c` — the pass added
  to correct `kernel_24x6`'s output orientation (Part C bug #1) had no
  `#pragma omp parallel for`, unlike every other loop in that function;
  could rival the parallel compute phase's own time at realistic
  thread-counts. Parallelized (safe: disjoint per-row work).
- **Missing `(size_t)` cast** in one index expression in `audited_submission/avx/matmul.c`
  (32-bit multiply overflow risk for N > ~46,340) — added, matching every
  other index expression in the function.
- **`test_local.c` alignment mismatch**: its `C` buffer was plain-`malloc`'d
  while linked against `kernel.c`'s alignment-requiring AVX-512 loads/
  stores. Switched to `_mm_malloc(...,64)`.
- **`strassen_avx2.c` was never actually exercised** by the AVX2 test
  harness despite existing specifically for that purpose — added a
  `run_strassen()` test to `test_local.c` and wired it into
  `run_local_avx2.sh`, which also gives real regression coverage for the
  heap-overflow fix above.

**Found, deliberately left unfixed (with reasons, not silently dropped):**
- Padding to the next full power of two costs up to ~4x memory/7x compute
  more than necessary just above a boundary (e.g. N=1025 pads to 2048) —
  real fix needs peeling an odd row/column per recursion level instead of
  padding once at the top; too large a restructure for a targeted patch.
- `kernel_24x6`/`strassen` symbol collision if both AVX-512 and AVX2
  variants were ever linked into one binary — not triggered by any build
  script actually in this repo (they're designed as drop-in alternatives,
  one linked at a time); renaming now would fight that intentional design.
- `audited_submission/avx/strassen.c` and `strassen_avx2.c`'s ~40-line orchestration wrapper
  is duplicated verbatim — a real dedup opportunity, but risky to refactor
  blind without a build/link path to verify it against in this environment.
- `omp_set_nested`/`omp_set_max_active_levels` global-state mutation in
  `audited_submission/avx/strassen.c` vs. `omp_set_nested(0)` in the `audited_submission/Parallel/` Strassen
  files — ambiguous whether this is a bug or deliberate; left alone rather
  than guess at intended semantics.

Full linking of the AVX2 harness could not be completed in this
environment (MinGW is missing a pthread import lib for `-fopenmp`) — an
environment gap, not a defect in the fixes; every touched file compiles
cleanly to `.o`, and the two `src/topology/` tools were linked and run
directly to confirm runtime behavior.

---

# Part D — Performance Revalidation of the Paper's Claims (Cross-Machine)

*(Workstream B. Two independent machines, reduced grid — not the paper's
full 16-hour sweep, by request. Raw data + full methodology caveats in
`revalidation/benchmarks/2026-09-09_glogin-e5-2630v3/` and
`revalidation/benchmarks/2026-09-09_local-i7-13700K/`.)*

**Reproduces cleanly on both machines:**
- Loop unrolling gives naive a large win; blocking beats naive at scale
- **Pure Strassen (no cutoff) is dramatically worse and gets *worse* under
  parallelism** — the paper's headline finding. E.g. local machine, N=64:
  0.0044s at 1 thread → flat/worse at 2+ threads.
- The "~7 threads then plateau/regress" shape for pure parallel Strassen
  shows up on both machines (e.g. glogin N=256: 0.324s→0.096s from T=1→7,
  then 0.099s at T=8 — a small regression right where the paper says it
  should be)

**Does NOT reproduce — and this is architecture-dependent, not a bug:**
the paper's scalar-level claim that blocked+unrolled overtakes hybrid
Strassen for N>512. On **both** independent machines, `strassen_base_unrolled`
keeps winning at N=1024 and N=2048, and the gap *widens* with size on the
faster local machine:

| N | blocked_unrolled | strassen_base_unrolled | Strassen's margin |
|---|---|---|---|
| 512 | 0.059s | 0.028s | 2.1x |
| 1024 | 0.411s | 0.141s | 2.9x |
| 2048 | 3.262s | 0.812s | 4.0x |

(local i7-13700K, serial, best-of-3)

**But — important nuance found when asked directly "even with AVX?"**: this
crossover-reversal is specific to the **scalar** comparison. Once both sides
are AVX-vectorized, blocked pulls back ahead at large N, matching the
paper's actual direction (just with a much smaller margin than the paper's
tuned AVX-512 kernel shows, since these AVX2 ports weren't independently
performance-tuned):

| N | T | blocked_avx2 | strassen_avx2 | blocked's margin |
|---|---|---|---|---|
| 1024 | 7 | 0.0154s | 0.0167s | 1.08x |
| 1024 | 16 | 0.0128s | 0.0166s | 1.29x |
| 2048 | 7 | 0.0874s | 0.1257s | 1.44x |
| 2048 | 16 | 0.0810s | 0.1049s | 1.30x |

**Takeaway for the paper**: the "blocked+unrolled wins for n>512" claim in
the Performance Comparison Summary is really a claim about the *AVX-512
kernel specifically*, not about the blocked algorithm in general — the
scalar (non-SIMD) version of that same claim doesn't hold on two unrelated
non-target machines. Worth making that scoping explicit in the writeup.

*(Note the relationship to Part B: this AVX2 blocked/Strassen comparison is
from the original `audited_submission/avx/` codebase's ports, not independently
performance-tuned the way the new `src/avx2/matmul_v*` series in Part B was —
they are different code answering different questions, kept separate
deliberately.)*

---

# Part E — Hardware-Counter Root-Cause Analysis (Original Algorithms, `glogin`)

*(Workstream B. Everything above is *what* is faster; this section is
*why*, backed by hardware performance counters — not just wall-clock time —
plus recursive call/allocation counts. Data:
`revalidation/harness/2026-09-09_glogin-e5-2630v3/harness_results.csv`,
methodology in that directory's `metadata.txt`. Instrumentation is opt-in
(`-DCOUNT_CALLS`) and confirmed to not change program behavior or add
measurable timing overhead. Collected on `glogin` only — WSL2 blocks all
hardware PMU counters (see Part A); that PMU also lacks LLC-specific
events, unlike the paper's Cascade Lake target, so generic
`cache-references`/`cache-misses` and `L1-dcache-*` substitute throughout.)*

## E.1 Naive's cliff at scale is an L2/L3-latency problem, not a cache-capacity or branch-prediction problem

| N | time(s) | IPC | branch-miss% | L1-miss% | cache(LLC)-miss% |
|---|---|---|---|---|---|
| 64 | 0.001 | 1.28 | 3.59 | 8.6 | 18.3 |
| 128 | 0.006 | 1.27 | 2.56 | 29.0 | 8.8 |
| 256 | 0.081 | **0.67** | 1.02 | 63.0 | 0.0 |
| 512 | 0.621 | **0.63** | 0.42 | 62.8 | 0.02 |

IPC craters from ~1.3 to ~0.65 right as N crosses 256 — this is the actual
mechanism behind the well-known "naive falls off a cliff" story. But look
at *which* counter explains it: branch-miss rate keeps *falling* (rules out
branch misprediction), and LLC-level cache-miss rate stays near **zero**
throughout (rules out "going to main memory" as the bottleneck — B's
column-strided access pattern is thrashing L1, but hitting in L2/L3, not
DRAM). So naive's slowdown is specifically **repeated L2/L3-latency
exposure from B's column access pattern**, not a capacity miss to DRAM and
not a front-end/branch problem. This distinction matters for anyone trying
to fix it: it's a blocking/reuse-distance problem, not a
software-prefetch-for-DRAM-latency problem.

## E.2 Blocking's win isn't from having a lower L1-miss rate — it's from where the misses land

| algo | N | IPC | L1-miss% | cache(LLC)-miss% |
|---|---|---|---|---|
| naive | 512 | 0.63 | 62.8 | 0.02 |
| blocked | 512 | **2.25** | 48.1 | 0.90 |

Blocked's raw L1-miss rate (48%) is still high — barely better than
naive's (63%) — yet its IPC is **3.6x** better. The L1-miss-rate number
alone is a misleading predictor here; what separates them is that blocking
keeps the *active* working set cycling through cache fast enough that most
of those L1 misses resolve quickly nearby (L2), rather than each one
paying full round-trip latency the way naive's do. Anyone tuning this
further should watch IPC and LLC-level miss rate, not L1-miss rate in
isolation — a plausible-looking "still lots of L1 misses" number doesn't
mean blocking isn't working.

## E.3 Pure Strassen's problem is not the memory hierarchy at all — it's raw instruction/allocation volume

| algo | N | IPC | L1-miss% | LLC-miss% | recursive calls |
|---|---|---|---|---|---|
| strassen_pure | 64 | 2.60 | 0.9 | 1.1 | 19,608 |
| strassen_pure | 512 | **3.20** | 0.8 | 1.2 | 6,725,601 |
| blocked | 512 | 2.25 | 48.1 | 0.9 | — |

Pure Strassen has the **best** IPC and by far the best cache behavior of
every algorithm tested — better than blocked, better than naive, better
than hybrid Strassen. Its problem is entirely that it executes an
enormous, exploding number of tiny recursive calls (19,608 at N=64 →
6.7 million at N=512, exactly matching the paper's own Table III), each
one paying malloc/free and function-call overhead for a tiny amount of
useful work. This directly confirms — with hard counter evidence, not just
call-count bookkeeping — that the paper's own diagnosis ("excessive
recursion," "allocator pressure") is correct and precise: it is a pure
instruction-volume/overhead problem, not a cache problem. It also explains
why parallelizing it (Part D) makes things *worse*: more threads means
more concurrent allocator contention and task-creation overhead layered on
top of an already overhead-dominated workload, with almost no real
compute per unit of that overhead to amortize it against.

## E.4 Hybrid Strassen's "unrolled" win is really a base-case loop-order win, not (just) fewer branches

| algo | N | IPC | L1-miss% | branches |
|---|---|---|---|---|
| strassen_hybrid (not unrolled) | 512 | 1.31 | **62.2%** | more |
| strassen_hybrid_unrolled | 512 | 1.79 | **26.0%** | fewer |
| strassen_hybrid (not unrolled) | 1024 | 0.98 | **63.5%** | more |
| strassen_hybrid_unrolled | 1024 | 1.69 | **25.0%** | fewer |

The un-unrolled hybrid's base case (`audited_submission/Serial/mm_strassen.c`, N≤128) is a
plain `i,j,k` triple loop — the *same* cache-hostile access pattern as
scalar naive, and its L1-miss rate (62-64%) matches naive's almost exactly.
The "unrolled" variant's base case (`mm_strassen_unrolled.c`) isn't just
unrolled — it's also loop-*reordered* to `i,k,j`, which is what actually
fixes the access pattern (L1-miss drops to ~25%). The unroll-vs-reorder
effects are bundled together in the current code and the paper's text
("unrolled by a factor of four") undersells what's actually happening: the
loop reorder is doing at least as much work as the unrolling itself. Worth
separating these two changes if anyone wants to know which one to credit.

## E.5 At N=1024, hybrid-Strassen-unrolled beats blocked-unrolled on every axis simultaneously — not a fluke

| algo | N | time(s) | IPC | L1-miss% |
|---|---|---|---|---|
| blocked_unrolled | 1024 | 2.329 | 1.28 | 40.6% |
| strassen_hybrid_unrolled | 1024 | **0.693** | **1.69** | **25.0%** |

Fewer total instructions (O(n^2.81) vs O(n^3), confirmed: strassen_hybrid_
unrolled's `instructions` counter is substantially lower than blocked_
unrolled's at the same N), *better* IPC, *and* better cache behavior, all
at once. This is a genuine compounding advantage on this hardware, not
noise — it's the hard-counter confirmation behind the Part D finding that
the scalar-level "blocked wins for n>512" claim doesn't generalize.

## E.6 Vectorization moves the bottleneck from compute to memory bandwidth once parallelized

| algo | N | T | IPC |
|---|---|---|---|
| parallel_blocked (scalar) | 512 | 1 | 2.28 |
| parallel_blocked (scalar) | 512 | 8 | 2.19 |
| avx2_blocked | 512 | 1 | 2.11 |
| avx2_blocked | 512 | 8 | **1.06** |
| avx2_strassen | 512 | 1 | 1.03 |
| avx2_strassen | 512 | 8 | **0.74** |

Scalar `parallel_blocked` keeps roughly the same IPC from 1 to 8 threads —
still compute-bound. The AVX2 kernels start with similar or better IPC at
1 thread, but IPC roughly *halves* by 8 threads — classic memory-bandwidth
saturation: vectorization does more FLOPs per cycle, so each thread needs
proportionally more memory bandwidth per unit of compute, and 8 threads
worth of AVX2 traffic saturates shared bandwidth well before 8 threads
worth of scalar traffic would. This is a real, useful thing to know before
trying to push the AVX-512 kernel to high thread counts on the actual
target machine: the ceiling is likely to be bandwidth, not core count, and
that's exactly the kind of effect NUMA-aware placement (which the paper's
own Future Work section already flags) would address. *(Cross-reference:
Part B.6 reached the same qualitative conclusion — compute/bandwidth
crossover matters — independently, via a STREAM benchmark on the new
`src/avx2/` codebase rather than perf counters on the old one.)*

---

# Part F — Combined Open Threads / Not Yet Done

**Workstream A (new AVX2 DGEMM vs. production BLAS):**
- **Bare-metal/non-virtualized confirmation — partially done, see Part G.**
  Native Windows (no Hyper-V/WSL2 layer) was used as the closest available
  substitute for a dedicated bare-metal Linux machine. Result: properly
  repeated measurements (4-5 runs, not 1) found **no detectable
  scheduling-policy effect**, including in the heterogeneous P+E-core case
  scheduling theory favors most — while single-run measurements on the
  same machine looked (misleadingly) like they confirmed §B.8. This
  doesn't prove §B.8's WSL2 result is purely a virtualization artifact
  (that needs the same repeated-measurement rigor applied *inside*
  WSL2/Hyper-V itself, still not done), but the evidence so far leans
  toward §B.8's own caveat holding up. **Still open:** the equivalent
  repeated-measurement experiment run *inside* WSL2/Hyper-V and on
  `glogin` (only native Windows got the rigorous repeat treatment this
  round); a literal dedicated bare-metal Linux machine, if one becomes
  reachable, would settle this more directly than the Windows substitute
  can. Until then, keep reporting the microkernel/packing-only figure
  (~87-92% of production BLAS, disassembly-verified, architecture-level)
  as the safe transferable claim, and the scheduling-adjacent figures
  (§B.8's 117-130%, §B.12's 4x) as environment-specific, per Part G.4.
- The portable harness (B.12) generalizes the *mechanism* (detect real
  hardware, derive+calibrate blocking, use a parameter-free adaptive
  schedule) across two real, different machines — that part of the
  "generalize to most systems" goal is met. It has NOT yet been shown to
  *match or beat* the dev-machine-tuned hardcoded version's peak on the dev
  machine itself (437 vs 506.8 GFLOP/s there, §B.12) — the portable
  version's `guided` schedule is intentionally more conservative than the
  hand-tuned `dynamic,6`. Whether that gap is worth closing (e.g. a
  slightly less conservative guided variant, or falling back to a brief
  scheduling micro-calibration alongside the blocking one) is open.
- 95%+ of OpenBLAS/BLIS on the microkernel/packing axis alone (excluding
  the scheduling fix) not reached; ceiling there is ~87-92%. BLIS's
  analytical MC/KC/NC formula was tried directly (B.11) and was *worse*
  (416.5) than our empirical search (506.8) — not a further lever after
  all, now confirmed rather than assumed. Transpose-based B-packing (B.7)
  and gather were tried and tied — not a further lever. C-tile prefetch
  (B.11) tried, no effect. Baseline fairness audited (B.10) — no
  misconfiguration found in OpenBLAS/BLIS/MKL. Remaining untried idea:
  8×4 shape (vectorize-A/broadcast-B at MR=8) — flagged by two independent
  external sources as a strong AVX2 choice, not yet tested under our
  current operand-swapped convention.
- oneDNN (Intel's own, cloned locally per request) has no double-precision
  GEMM kernel at all (DL-focused: f32/bf16/int8 only) — confirmed not
  usable as a DGEMM reference; MKL remains the right Intel comparison.
- Full benchmark sweep across sizes (256-8192+) and thread configs
  (P-only/E-only/all-physical/SMT, native Windows) — **further along**:
  Part G's `src/avx2/ladder/` P/E/mixed pinning experiment (repeated, N=1024
  only) plus §B.13's native P-only/E-only/SMT-on-off/all-cores sweep
  (single-run, needs repetition per G.4) together cover the core-placement
  axis reasonably well now; still missing the full 256-8192+ size range
  under controlled placement, and SMT ablation hasn't been repeated.
- **Strassen rebuild: done (Part H)** — pooled workspace, fused adds,
  1-/2-level, thread-aware cutoff, packed-DGEMM base case, and an
  optimized-vs-optimized crossover search (found: strassen_1level reaches
  parity with dgemm_base at N≈8192 on this machine). **Still missing**: the
  real per-phase cost model (leaf GEMMs / add-sub / packing / workspace /
  scheduling timed separately, point 7) — only end-to-end effective
  GFLOP/s was measured, not a phase breakdown; and G.4-style repeated
  measurement of the crossover point itself (currently single-run-per-N).
- **Numerical accuracy: done (Part I)** — 3 distributions × classical/
  1-level/2-level Strassen against the long-double reference, at several
  N. Found a dramatic, real result (wide-dynamic-range inputs: Strassen's
  max relative error reaches >1400% at N=1024, 2-level). **Still missing**:
  comparison against MKL/BLIS/OpenBLAS's own output (only our own
  dgemm_base/Strassen were compared to the reference; production BLAS is
  presumed but not verified to match the reference equally closely), and
  depth-3 Strassen (only 1- and 2-level exist).
- **Cross-architecture run for the new AVX2 DGEMM/Strassen work on
  `glogin.cs.vt.edu` — done for the portable ladder harness (Part G);
  not yet done for the v1-v7/asm_kernel/portable series from Parts B.1-B.12.**
- **Not attempted, no fabricated data exists for these — see Part G.0:**
  a real multi-level memory model beyond what `glogin`'s PMU exposes
  (L2-specific and LLC-specific events, DTLB page-walk counts, memory-
  controller bandwidth counters — point 3); compute-ceiling/SIMD-width/
  CPU-frequency/RAPL-power measurement, incl. an AVX2-vs-AVX-512 comparison
  (moot without AVX-512 hardware anyway — point 4); a full KC/MC/NC
  response-surface sweep with one parameter varied at a time against
  GFLOP/s *and* cache/TLB miss rates, plus a huge-pages comparison (point
  5); a GCC-vs-Clang-vs-ICX disassembly/register-pressure/`llvm-mca`
  comparison (only MSVC-vs-gcc was compared, incidentally, via Part G.2's
  DCE bug — point 6); rectangular/irregular-M,N,K and awkward-size testing
  (1000, 1023, 1025, 1537, 4095, primes — point 9).

**Workstream B (correctness + revalidation of the original codebase):**
- **Real AVX-512 execution.** No AVX-512 hardware was reachable this
  session. The AVX-512 code compiles clean against every fix (`kernel.c`
  and `audited_submission/avx/strassen.c` both compile-checked with `-march=skylake-avx512`),
  and the packing fix is provably width-independent (same fix, same
  packed-buffer contract, verified with real vector execution on AVX2) —
  but nobody has run the actual AVX-512 binary since the fix went in. Do
  that before trusting the AVX-512 numbers for anything beyond "should be
  correct."
- **The paper's full size/thread range.** Tested up to N=2048 and 16
  threads (local) / 8 threads (glogin) vs. the paper's N up to 16384 and
  64 threads. The ">7 threads" plateau claim is only partially checked —
  glogin's 8-core ceiling and the local machine's heterogeneous P/E-core
  design both limit how cleanly the higher-thread-count regime can be
  read.
- **LLC-specific counters.** Not available on either test machine's PMU;
  substituted generic `cache-references`/`cache-misses` and `L1-dcache-*`.
  If reproducing this on the actual Cascade Lake target, redo Part E with
  the paper's original LLC-load/store events for a cleaner comparison
  against its own tables.
- **Statistical rigor.** All of Parts C-E is best-of-3 or single-run, vs.
  the paper's 14-run/trim-2/average-10 methodology. Treat every number
  there as directional.

**Shared:**
- Reproducibility pack (commit/tag, full env capture) and paper rewrite —
  not started.

---

# Part G — Is the Scheduling Win Real, or Virtualization Compensation? (`src/avx2/ladder/`)

*(Direct response to the standing Part F question and to an external
research-plan review requesting a systematic driving-factors study: a
portable phase-timed harness, run on three genuinely different execution
environments on the same physical machine's CPU plus one independent
remote host, to isolate whether §B.8's large dynamic-scheduling win is
architecture-level or an artifact of Hyper-V's vCPU remapping.)*

## G.0 Scope and honesty about what this is/isn't

The requesting review asked for ten things: (1) bare-metal-vs-virtualized
scheduling validation, (2) phase-by-phase timing decomposition, (3) a real
multi-level memory-hierarchy model (L1/L2/L3/DRAM, not just "cache miss"),
(4) compute-ceiling/SIMD-width/frequency/RAPL-power study, (5) a full
blocking response-surface sweep incl. huge pages, (6) a
GCC-vs-Clang-vs-ICX compiler comparison, (7) a rebuilt, properly cost-
modeled Strassen, (8) a numerical-accuracy study, (9) irregular/rectangular
shapes, (10) rigorous statistics (20-30 runs, CIs) throughout.

**Done this round: (1) and (2), partially, plus the statistical-rigor
lesson from (10) learned the hard way (G.3).** Not done, and not
fabricated: no literal dedicated bare-metal Linux machine was available
(closest substitute used: native Windows, no Hyper-V/WSL2 layer — see
below); no RAPL/power or CPU-frequency telemetry was collected (not
established how to read it reliably in this session); no Clang/ICX
compilers were available to install; the memory-hierarchy model is still
limited to whatever PMU events `glogin`'s Haswell exposes (Part E's
existing LLC-event gap, unchanged); the Strassen rebuild, numerical-
accuracy study, and irregular-shape sweep were not started. These remain
open — see the updated Part F.

## G.1 A portable, phase-timed ladder harness (`src/avx2/ladder/`)

New, independent of the `src/avx2/matmul_v*`/`asm_kernel` series: a single
C source (`matmul_ladder.c`, MR=4×NR=8, broadcast-A/vector-B, flat
accumulators — the already-validated "swap" convention from §B.4, chosen
over the asm kernel's MR=6 shape specifically so the *same source* compiles
under both **MSVC 19.44** (`/openmp`, stuck at OpenMP 2.0 — no `collapse`,
no `omp_set_schedule()`/`omp_sched_t`, C99 loop-variable declarations
rejected inside `#pragma omp for`) and **gcc 13** (`-fopenmp`), with a
cross-platform `timer.h` (`QueryPerformanceCounter` / `clock_gettime`).
Schedule is selected at *runtime* via `schedule(runtime)` + setting the
`OMP_SCHEDULE` environment variable before first use — portable across
both compilers' OpenMP versions, no recompile needed to compare
static/dynamic/guided.

Implements stages A and C-G of the requested ladder in one binary
(`<A|full> N threads schedule chunk reps [windows-core-list]`):
phase timers (`pack_a_s`, `pack_b_s`, `kernel_s`, `total_s`) around a real
multilevel MC/KC/NC driver, and (Windows only) explicit per-thread
`SetThreadAffinityMask` pinning to a caller-supplied logical-CPU list, for
stage E and the P/E-core experiment below. Stage B (pack+kernel, single
block) needs no separate code path — just pick N small enough to fit one
block. Stage H (production baseline) is intentionally *not* in this file —
compare its GFLOP/s against `driver_openblas`/`driver_naive` separately
(same FLOP count regardless of this file's row-major vs. `bench_common.h`'s
column-major convention, so directly comparable).

Correctness: `reference_check()` (long-double accumulation, subsampled at
large N for speed) returned exactly `0.000e+00` relative error on every
run in this section, on every platform.

## G.2 A real bug this harness itself introduced and caught: dead-code elimination

First WSL2/gcc build of Stage A (kernel-only, repeated calls on
pre-packed panels) reported **21,845,696 GFLOP/s** — impossible. Cause:
the stage's output tile was never read or printed, so gcc `-O3` proved the
entire repeated-call loop had no observable effect and deleted it (MSVC,
notably, did *not* do this — a real, if unintended, cross-compiler
codegen-aggressiveness data point in the spirit of point (6)). Fixed with
a `volatile` sink reading the tile's first element after the loop —
standard anti-DCE practice, but worth recording precisely *because* it's
exactly the kind of silent-methodology bug the requesting review was
worried about generally. Post-fix, Stage A is a believable ~30-44 GFLOP/s
single-thread pure-kernel number on every platform (G.3).

## G.3 Cross-environment sweep — and a hard lesson about single-run comparisons

Ran the identical binary (Windows: MSVC-built `matmul_ladder.exe`; Linux:
gcc-built `matmul_ladder_linux`) across N ∈ {512,1024,2048}, T ∈ {1,4,8},
schedule ∈ {static, dynamic(chunk=4), guided}, on:

- **native Windows** (MSVC, no WSL2/Hyper-V layer — the closest
  non-virtualized substitute available for the requested bare-metal test)
- **WSL2/Hyper-V** (this session's Docker sandbox — same environment as
  Workstream A's dev-machine results throughout Parts A-B)
- **`glogin.cs.vt.edu`** (KVM-virtualized, not Hyper-V — Workstream B's
  machine throughout Parts C-E)

Raw data: `src/avx2/ladder/sweep_native_windows.csv`, `sweep_wsl2.csv`,
`sweep_glogin.csv` (single run per config each — see the caveat below
before reading too much into any single cell).

**Stage A (kernel-only, single-thread, pure compute) across platforms:**
Windows/MSVC ~43.6 GFLOP/s, WSL2/gcc ~39.8, `glogin`/gcc ~29.4 — a
believable, consistent-with-clock-speed ordering (i7-13700K clocks
noticeably higher than the Haswell E5-2630 v3), and the one number in this
section clean enough to trust from a single run, since it isolates pure
FMA throughput with no scheduling or OS-placement variable involved at all.

**`glogin` (KVM): scheduling choice barely moves the needle, at any
thread count.** E.g. N=2048/T=8: static 118.4, dynamic 119.4, guided
119.4 GFLOP/s — within 1%, and this pattern holds at every (N,T) tested.
Consistent with a stable, non-reshuffling vCPU-to-pCPU mapping under KVM
for the duration of a run.

**WSL2 and native Windows: apparent schedule differences in the raw
single-run sweep, but no stable direction.** WSL2 N=512/T=8: dynamic 191
vs static 146 (dynamic looks like a big winner); WSL2 N=1024/T=8: static
174 vs dynamic 163 (static wins instead); WSL2 N=2048/T=8: dynamic 163 vs
static 151 (dynamic wins again). Windows shows the same kind of
inconsistency. Read at face value, this looks like confirmation of §B.8's
finding. **It is not safe to read it that way from this data alone — see
G.4.**

**An unexplained discrepancy, flagged rather than hidden:** this sweep's
single-run Windows N=1024/T=8 numbers (static 107, dynamic 109 GFLOP/s)
are roughly **2x lower** than the same exact configuration measured in
isolation in G.4's repeated-run experiment (217-296 GFLOP/s band, done
minutes apart, same binary, same machine). The sweep runs 27 configs
back-to-back (N=512→2048, T=1→8, three schedules each) in one script
without cooldown; the isolated G.4 runs were each a fresh, short,
standalone invocation. The leading hypothesis is thermal/sustained-load
throttling accumulating over the sweep's run — but this was **not
confirmed** (no CPU-frequency or temperature telemetry was collected this
round, per G.0's scope note; point (4) and (10) of the requesting review
are exactly what would settle this). Treat every number in the raw sweep
CSVs as lower-confidence than G.4's repeated numbers for this reason, and
don't trust a single-run schedule comparison from these files without
rerunning it in isolation first.

## G.4 The controlled experiment: P/E-core pinning, properly repeated (native Windows only)

Since only native Windows offers real, working thread-to-core pinning in
this session (`SetThreadAffinityMask`; Linux affinity was not implemented
in this pass — see Part F), this is the one environment where the actual
"is it scheduling policy or is it core heterogeneity" question could be
asked cleanly, using the P-core/E-core logical-CPU classification already
established in Part A (`src/topology/classify_cores_native.c`: logical 0-15 =
P-cores/HT pairs, 16-23 = E-cores).

At N=1024, T=8, first pass (single run each, `src/avx2/ladder/pinning_experiment.csv`):

| Placement | schedule | GFLOP/s |
|---|---|---|
| unpinned | dynamic | 104.4 |
| unpinned | static | **218.0** |
| 8 distinct P-cores (no HT overlap: 0,2,4,6,8,10,12,14) | dynamic | 282.0 |
| 8 distinct P-cores | static | 251.1 |
| 8 E-cores (16-23) | dynamic | 136.8 |
| 8 E-cores | static | 133.0 |
| mixed 4P+4E (0,2,4,6,16,17,18,19) | dynamic | 142.4 |
| mixed 4P+4E | static | 130.2 |

Taken at face value, unpinned static beating unpinned dynamic **2x**, and
dynamic edging out static on the heterogeneous P+E mix, both looked like
real, interesting, theory-consistent results (dynamic *should* help most
exactly on a heterogeneous team). **Neither survived repetition:**

- **Unpinned static vs. dynamic, 4 repeats each (N=1024, T=8):** static
  {277, 287, 217, 296}, dynamic {285, 276, 294, 286} GFLOP/s. Completely
  overlapping ranges, no consistent winner — the initial 218-vs-104 result
  was noise from a single run landing on opposite tails of a wide (~217-296)
  distribution, not a real effect.
- **Pinned heterogeneous 4P+4E mix, 5 repeats each:** static {150, 141,
  143, 151, 149}, dynamic {146, 149, 145, 151, 145} GFLOP/s. Again
  completely overlapping, no consistent winner — even in the one scenario
  picked specifically because theory predicts dynamic scheduling should
  help. Note also how *tight* this pinned distribution is (141-151)
  compared to the *unpinned* one (217-296): pinning removed most of the
  variance, but didn't reveal a scheduling-policy effect underneath it —
  it just removed OS-placement noise.

**What did produce a large, clean, un-noisy effect: which cores the
threads land on at all.** Pinned-to-P-cores (~250-280) beats pinned-to-E-
cores (~133-137) by roughly **2x**, consistently, regardless of schedule.
That's a real, repeatable, mechanistically obvious result (P-cores have
2x the per-cycle AVX2 FMA throughput of E-cores per Part A's earlier
classifier finding) — and it dwarfs any scheduling-policy effect measured
here.

**Conclusion on the standing Part F question, stated precisely:** on this
machine, in this pass, **properly repeated measurements found no
detectable scheduling-policy effect (static vs. dynamic vs. guided) once
thread-to-core placement was controlled for — including in the
heterogeneous-core case scheduling theory says should favor dynamic the
most.** This does **not** prove §B.8's WSL2 result was purely a
virtualization artifact (that would require the same repeated-measurement
rigor applied *inside* WSL2/Hyper-V itself, which was not done this round —
G.3's WSL2 numbers are single-run and explicitly flagged as
unreliable for this purpose). But it does mean: the burden of proof for
"dynamic scheduling is architecturally necessary" is not met by the
evidence collected so far, the one clean non-virtualized test available
found no such effect, and §B.8's own honest caveat (**"this must be
confirmed on non-virtualized hardware before being reported... as
measured, it is accurate for *this exact environment*"**) looks
increasingly like it will hold up rather than be overturned. **Updated
recommendation: keep §B.8/§B.9's 117-130%-of-BLAS figure explicitly scoped
to "measured under WSL2/Hyper-V" in the paper, do not generalize it, and
treat the ~87-92% microkernel/packing-only figure as the safe,
architecture-level, portable claim** — exactly what Part F already said,
now with a real (if not fully conclusive) experiment behind it rather than
reasoning from first principles alone.

## G.5 What this means for the requesting review's other points

- **Point (2), phase decomposition:** partially delivered — `pack_a_s`,
  `pack_b_s`, `kernel_s` are logged for every run above. Skimming the
  `glogin` N=2048/T=8 row as one concrete example: kernel ≈0.127s of
  ≈0.144s total (~88%), packing ≈0.017s (~12%) — compute-dominated at this
  size/thread count on this machine, consistent with Part B.6's roofline
  finding (compute-bound, not bandwidth-bound) via an independent method.
  A full stage-by-stage percentage table across all (N,T) was not built
  this round (mechanical follow-up, not a new method).
- **Point (10), statistics:** G.4 is the concrete lesson: this session's
  first-pass single-run sweep (G.3) would have supported a false
  "dynamic scheduling helps" narrative if not repeated. Every future
  scheduling/placement claim in this document should be held to G.4's
  standard (≥4-5 repeats, report the range) before being stated as fact,
  not just the final "clean" numbers.
- **Points (3),(4),(6),(7),(8),(9):** not attempted this round; see the
  updated Part F for what each would require.

---

# Part H — Strassen Rebuild (`src/strassen2/`)

The reviewer's ask was a genuine rebuild — pooled workspace, fused adds,
1-/2-level, packing integration, thread-aware cutoff, optimized-vs-
optimized comparison — distinct from Part C's correctness *fixes* to the
existing `audited_submission/avx/strassen.c`, which remains a separate, untouched piece of
work.

## H.1 Design

- **`src/strassen2/dgemm_base.c`**: the tuned hand-asm/transpose-packed kernel
  (same as `src/avx2/asm_kernel/matmul_asm_v3_transpack.c`) extracted into a
  reusable `dgemm_base(A,ldA,B,ldB,C,ldC,n)` library call with
  *independent* leading dimensions — this is what lets Strassen operate on
  quadrant VIEWS of the original matrices without copying them.
- **Fused adds** (`src/strassen2/strassen_ops.c`): only the 10 genuinely
  combined operands the classic 7-multiplication formulation needs
  (A11+A22, B11+B22, A21+A22, B12-B22, B21-B11, A11+A12, A21-A11, B11+B12,
  A12-A22, B21+B22) are ever materialized. The 4 quadrants used *bare*
  (A11 in M3, A22 in M4, B11 in M2, B22 in M5) are passed straight to
  `dgemm_base` as strided views — never copied at all, which the original
  `audited_submission/avx/matmul.c`-based Strassen didn't do (it copied all 8 quadrants).
- **Pooled workspace** (`src/strassen2/strassen.c`): one batch allocation per
  recursion level (17 buffers — 10 sums + 7 products — via one `bench_alloc`
  and pointer offsets), not 17 separate mallocs and nowhere near the
  millions of tiny allocations the original implementation produces at
  scale (Part C/E).
- **1-level and 2-level** variants (`strassen_1level`, `strassen_2level`):
  the second recurses one further level before hitting `dgemm_base`.
- **Thread-aware cutoff**: with ≤8 threads, the 7 products are split into
  OpenMP tasks (good use of a modest budget); with more threads, the 7
  run sequentially, each given the *full* thread count via its own
  (non-nested-serialized) `dgemm_base` call — splitting into 7 tasks on
  e.g. 16 threads would waste 9 of them. Threshold: `#define
  TASK_SPLIT_THREAD_THRESHOLD 8` in `strassen.c`.

## H.2 A real bug found and fixed during this build

`dgemm_base`'s packing buffers were first written as `threadprivate`
statics (worried about races between concurrent OpenMP tasks calling it
simultaneously). This **segfaulted** — nested parallel regions (Strassen's
sequential branches each opening their own `dgemm_base` parallel region,
nested inside `strassen_level`'s own) give each nested worker thread its
own *uninitialized* threadprivate copy, since only the calling thread ran
the allocation code. Fixed by dropping the cache entirely: fresh local
`packA`/`packB` buffers per `dgemm_base` call, freed at the end — safe in
every calling context, and negligible overhead since Strassen calls this
O(7) or O(49) times per level, not millions.

Verified correct (`src/strassen2/test_strassen.c`) at n=64..1024 for both
variants, at both 8 threads (task-split path) and 16 threads (sequential
path) — all pass at ~1e-15/1e-10 relative error, matching every other
correct measurement in this document.

## H.3 Optimized-vs-optimized crossover (effective GFLOP/s = 2N³/time, per reviewer's request)

`src/strassen2/bench_strassen.c` reports Strassen's throughput using the same
formula as classical GEMM (not Strassen's true lower FLOP count) so it's
directly comparable — deliberately labeled "effective" to avoid the
same headline-number problem as the original naive-speedup framing.

| N | dgemm_base | strassen_1level | strassen_2level | ratio (dgemm/1level) |
|---|---|---|---|---|
| 512 | 245.8 | 22.9 | 1.7 | 10.8x |
| 1024 | 466.9 | 105.6 | 13.0 | 4.4x |
| 2048 | 458.2 | 260.0 | 77.4 | 1.76x |
| 4096 | 434.7 | 375.0 | 213.8 | 1.16x |
| 8192 | 426.6 | 426.7 | 372.1 | **~1.00x (crossover)** |

**strassen_1level reaches parity with optimized classical DGEMM at N=8192**
on this machine/environment — a real, measured crossover, not the "never
crosses" or "always worse" story the original paper's (unoptimized-vs-
optimized) comparison told. strassen_2level is consistently worse at every
size tested (recursion overhead compounds faster than the arithmetic
savings materialize at these sizes) — never found a crossover for it.
N=16384 was attempted to see if 1-level pulls *ahead* beyond the crossover,
but became impractically slow in this environment (likely allocation/
zeroing overhead for the ~9GB pooled workspace under WSL2) — not yet
resolved; bare-metal Linux (already the top open item) would also let this
be pushed further. Each cell above is a min-of-`reps` within one process
invocation, not independent repeated runs (Part G.4's standard) — the
monotonic shrinking of the ratio across 5 increasing N values is a strong
enough shape to trust qualitatively, but the exact crossover N should be
treated as approximate until re-run with G.4-style repetition. Same
absolute-number caveat as §B.13 applies here too —
treat the crossover *location* (N≈8192) as the finding, not the precise
GFLOP/s values.

---

# Part I — Numerical Accuracy (`src/accuracy/`)

Completely absent before this — the single most-cited gap in the reviewer
feedback. Built against the rebuilt Strassen (Part H) and `dgemm_base`,
across three input distributions chosen to stress cancellation
differently (`src/accuracy/distributions.c`): uniform (baseline), wide dynamic
range (row-scaled across ~12 orders of magnitude — an ill-conditioned
case), and clustered-near-1.0 (designed to stress cancellation in
Strassen's recursive sums specifically).

| Distribution | Variant | N | rel. Frobenius err | rel. max err |
|---|---|---|---|---|
| uniform | dgemm_base | 1024 | 1.07e-15 | 1.60e-09 |
| uniform | strassen_1level | 1024 | 1.51e-15 | 1.02e-09 |
| uniform | strassen_2level | 1024 | 2.16e-15 | 2.02e-09 |
| wide_dynamic_range | dgemm_base | 1024 | 1.09e-15 | 4.96e-10 |
| wide_dynamic_range | strassen_1level | 1024 | 1.49e-15 | **8.43e-01** |
| wide_dynamic_range | strassen_2level | 1024 | 2.23e-15 | **1.44e+01** |
| clustered_cancellation | dgemm_base | 1024 | 7.78e-16 | 3.89e-15 |
| clustered_cancellation | strassen_1level | 1024 | 9.57e-16 | 5.66e-15 |
| clustered_cancellation | strassen_2level | 1024 | 1.19e-15 | 9.10e-15 |

**Uniform inputs**: all three at machine-epsilon level, Strassen mildly
worse than classical (expected — extra additions). **Not a concern in
this regime.**

**Wide dynamic range: a real numerical catastrophe for Strassen,
worsening sharply with N and with recursion depth.** At N=256,
strassen_1level's max relative error is 0.67 (67%) and strassen_2level's
is 1.23 (>100%, i.e. worse than the answer being completely wrong in its
largest element); at N=1024, 1level=0.84 and **2level=14.36 (1436%
relative error)** — while classical `dgemm_base` stays at ~1e-10
throughout. This is the paper's numerical-accuracy gap, now filled with a
dramatic, real, reproducible result: **Strassen's extra additions are not
numerically safe on ill-conditioned inputs, and this gets rapidly worse
with both matrix size and recursion depth.** This is exactly the kind of
finding the paper's own dismissal of "other fast multiplication approaches
partly on numerical stability grounds" should have measured for its own
Strassen implementation, and didn't.

**Clustered-cancellation**: stayed near machine epsilon at every size
tested — this distribution didn't stress error the way intended (absolute
magnitudes stay near 1.0 throughout, so relative error stays well-behaved
even with cancellation in absolute terms). Noted as a distribution that
didn't produce the expected effect, rather than silently dropped.

---

# Part J — Closing Out the Remaining Review Points (`src/avx2/ladder/`)

*(Independent work, done in parallel with Parts H-I above without
knowledge of them until this write-up — different design choices on both
the Strassen rebuild and the accuracy sweep turned out to be genuinely
complementary rather than redundant; see J.7/J.8. Everything below is
real, executed, verified data — every number has a command behind it.
Scope note up front, so it isn't oversold: "20-30 runs with confidence
intervals" (review point 10) was not applied everywhere — 5 repeats was
this pass's practical standard, applied to every genuinely new claim
below, which is enough to separate signal from noise in every case found
(either a clean non-overlapping gap, or a clearly overlapping "no effect"
band) but is not the same statistical rigor as a full 20-30-run study.)*

## J.1 Extending the scheduling question to WSL2 and `glogin` themselves

Part G.4 applied real repeated-measurement rigor (4-5 runs, not 1) only on
native Windows, and explicitly flagged that WSL2 and `glogin` themselves
had not gotten the same treatment. Closed that gap: static vs. dynamic,
5 runs each, N=1024, T=8, using the same `matmul_ladder` binary as Part G.

| Environment | schedule | GFLOP/s (5 runs) | Overlapping? |
|---|---|---|---|
| `glogin` (KVM) | static | 122.4, 105.8, 121.7, 120.3, 121.4 | — |
| `glogin` (KVM) | dynamic | 123.3, 119.7, 121.4, 117.5, 104.4 | **yes, no winner** |
| WSL2 (Hyper-V) | static | 342.1, 326.7, 327.3, 362.1, 333.8 | — |
| WSL2 (Hyper-V) | dynamic | 347.0, 343.1, 356.7, 363.7, 335.7 | **yes, no winner** |

Both environments now show the same pattern Part G.4 found on native
Windows: **once repeated, no clean scheduling-policy effect for this
kernel, in any of the three environments tested** — including WSL2/
Hyper-V itself, the original environment where Part B.8 found its large
effect. Important scoping, not a contradiction of B.8: this is a
*different, simpler* kernel (MR=4×NR=8 portable intrinsics) than B.8's
hand-written-assembly + vectorized-packing kernel, run at a smaller
grid (N=1024, T=8 only, vs. B.8's tuned N=2048/16-thread configuration).
So the honest conclusion is narrower than "B.8 was wrong": **the large
scheduling effect is not a universal property of "any DGEMM kernel under
WSL2"** — it did not reproduce for this kernel, at this size, in this
environment, once measured properly — which raises the bar for trusting
it as reported for the *original* kernel too, without applying this same
repeated-measurement standard directly to `matmul_asm_v4_dynsched.c`
itself (still not done, still the honest gap).

## J.2 Finer phase decomposition (`matmul_ladder2.c`)

Extended the driver to split `kernel_s` into `kernel_compute_s` (FMA loop
only) and `kernel_write_s` (C accumulate-store, previously fused), and
`pack_a_s`/`pack_b_s` from `alloc_s` (buffer allocation, previously
absorbed into the packing timer). At N=1024/T=8/KC=1024/MC=384/NC=1024
(`glogin`): `alloc_s`≈0.00001s (negligible), `pack_a_s`≈0.0013s,
`pack_b_s`≈0.0013s, `kernel_compute_s`≈0.104s, `kernel_write_s`≈0.0056s,
`total_s`≈0.0179s → **compute is ~92% of aggregate phase time, write-back
~5%, packing ~3%, allocation noise** (aggregate here means summed across
all 8 threads' concurrent work, a CPU-time-like measure, not wall-clock —
noted explicitly in the code and here so it isn't misread as literally
summing to `total_s`, which is wall-clock and smaller). Confirms Part
E.6/B.6's compute-bound conclusion a third way (after perf-counter IPC and
a STREAM roofline check), and quantifies specifically how small the
write-back and packing shares are for this blocking configuration —
useful context for J.3's blocking sweep, where KC's effect turns out to
run almost entirely through this write-back share (see below).

## J.3 Full blocking response surface + huge pages (`blocking_sweep.sh`)

One-parameter-at-a-time sweep (KC, then MC, then NC) at N=1024, T=8,
static, paired with `perf stat` (cache-references/misses, L1-dcache,
dTLB — same event set as Part E, same PMU limits apply). Full data:
`src/avx2/ladder/blocking_sweep_results.csv`.

| KC | GFLOP/s | kernel_write_s | dTLB load misses |
|---|---|---|---|
| 128 | 62.6 | 0.0546 | 181,271 |
| 256 | 93.7 | 0.0250 | 117,609 |
| 512 | 114.0 | 0.0116 | 92,445 |
| 1024 | 125.2 | 0.0053 | 59,209 |
| 2048 | 131.8 | 0.0055 | 57,592 |

KC is by far the dominant parameter — 62.6→131.8 GFLOP/s (2.1x) across
the sweep, plateauing once KC≥1024 (=K, so one k-block covers the whole
matrix and the outer p-loop stops mattering). **The mechanism is directly
visible in the phase split**: `kernel_write_s` drops 10x in lockstep with
GFLOP/s rising, because a bigger KC means each kernel call does more FMA
work per C-tile write-back, amortizing that fixed cost over more useful
compute — exactly Goto & van de Geijn's argument for why kc should be as
large as cache/TLB capacity allows. dTLB misses fall the same direction
(3x), consistent. MC and NC were both far less sensitive (MC:
103-117 GFLOP/s across 64-1024, mild peak near 384; NC: 89-116 GFLOP/s,
worst at the smallest value tested) — real, but second-order next to KC
for this kernel/shape.

**Huge pages**: `madvise(MADV_HUGEPAGE)` on the A/B/C operand buffers
gave a small, inconsistent difference (120-125 GFLOP/s without the hint
vs. 124-127 with it, 3 runs each, overlapping) — checked *why* before
calling it a null result: `/sys/kernel/mm/transparent_hugepage/enabled`
reports `[always]` on this system, i.e. THP is already unconditionally on
by kernel default, so the explicit hint had little left to add. A real,
honest finding (huge pages don't show much *additional* benefit here)
with a real, confirmed reason (they were already active), not a broken
test.

## J.4 Compiler diversity: GCC vs. Clang vs. Intel icx (`glogin` has all three)

`glogin` turned out to have a full toolchain beyond what Part A recorded:
Clang 22.1.3 and a complete Intel oneAPI 2022.1 install (`icx`, and — no
license needed, unlike the old `icc` — MKL, VTune, and `llvm-mca`), none
of which needed installing. Built the identical `matmul_ladder.c` with
all three (`-O3 -march=native -mavx2 -mfma`, matching flags), 5 repeats
each:

| Compiler | Stage A (kernel-only, GFLOP/s) | Full N=1024/T=8 static (GFLOP/s) |
|---|---|---|
| gcc 11.5.0 | 28.2-29.1 (mean 28.8) | 114.8-119.4 (mean 116.9) |
| clang 22.1.3 | 27.6-27.9 (mean 27.8) | **136.6-141.8 (mean 140.2)** |
| icx 2022.1.0 | 27.6-28.0 (mean 27.8) | 115.3-132.4 (mean 122.4, more variance) |

Non-overlapping, real: clang wins the full pipeline by ~20% despite gcc
having a marginally *higher* isolated kernel-only number — meaning the
compiler difference lives outside the hot FMA loop, not inside it.
Disassembly (forced with `__attribute__((noinline))` so the symbol
survives comparably in all three, since clang/icx fully inline the kernel
by default and gcc doesn't) confirmed one real, mechanistic difference:
**clang and icx both auto-unroll the k-loop 2x at `-O3`; gcc does not** —
the same transformation Part B.4 had to add by hand for gcc
(`matmul_v3_unroll_prefetch.c`), LLVM's optimizer does for free. Checked
this quantitatively with `llvm-mca -mcpu=haswell` on the extracted loop
bodies: **per-k-step, both loops have almost identical modeled throughput
(gcc: 5.14 cycles/step; clang unrolled: 5.08 cycles/step)** — so the
auto-unrolling, while real and confirmed, is *not* the main explanation
for the measured 20% full-pipeline gap. Dynamic instruction counts (`perf
stat`) at T=1 showed gcc actually *faster* there (fewer cycles despite
more instructions, higher IPC: 2.41 vs. clang's 2.01) — the clang
advantage is specifically a **multi-threaded** effect, not visible at
T=1, and aggregate per-thread cycle/instruction counts at T=8 didn't
cleanly resolve it further (clang's aggregate cycle count was actually
*higher*, not lower, ruling out "simply fewer total cycles" as the
explanation). **Honest status: the behavioral difference is real,
repeated, and disassembly-confirmed in one respect (unrolling); the full
mechanism behind the T=8-specific magnitude of the gap was not fully
pinned down this round** — most likely candidate for further work is
per-thread load-imbalance or false-sharing differences in how each
compiler lays out the packing-loop dispatch, not examined here.

`llvm-mca`'s port-pressure/throughput estimation (review point 6's
specific ask) worked cleanly with no root/hardware access needed — a
genuinely useful, VM-safe technique, unlike VTune's `uarch-exploration`
(J.5) which needs root.

## J.5 Confirmed hard limits, not assumed ones (review point 3, part of point 4)

`glogin` turned out to have Intel VTune Profiler 2022.2.0 installed too
(driverless mode, "User-mode sampling and tracing"). Used it to get an
*authoritative* answer instead of continuing to infer from raw `perf`'s
event list:

- `vtune -collect memory-access`: **refused outright** —
  *"Memory Access analysis is not supported inside a virtual machine
  since uncore events cannot be collected. For full functionality,
  consider using a bare-metal environment."* This is Intel's own tooling
  confirming Part E's memory-hierarchy gap (no L2/LLC-specific data
  obtainable here) is a real VM limitation, not a raw-`perf`-specific
  shortcoming that some other tool might work around.
- `vtune -collect uarch-exploration`: needs `/proc/sys/kernel/
  perf_event_paranoid ≤ 0` (currently 2) or an installed sampling driver
  — both need root, unavailable (`no sudo`, confirmed repeatedly this
  session). Also blocked, for a different, non-VM-specific reason.
- `vtune -collect threading`: **worked** — real per-run data: Effective
  CPU Utilization 86.6% (6.930 of 8 logical CPUs) at N=1024/T=8/static,
  0% thread oversubscription, 0% spin/wait time. A genuine (if partial)
  answer to review point 1's request for per-thread utilization data,
  via a tool that doesn't need the blocked collection modes.
- Frequency: VTune's collection header reports the CPU running at
  **2,399,998,069 Hz — 2.4000 GHz, exactly the Haswell E5-2630 v3's rated
  base clock, no turbo, no throttling detected** in this VM. A real,
  confirmed answer to part of review point 4 (no RAPL power was
  obtainable — `/sys/class/powercap/intel-rapl*` doesn't exist on this
  VM at all, confirmed by direct check — but frequency is now measured,
  not assumed).

## J.6 Rectangular and irregular shapes (review point 9)

Generalized `matmul_ladder2.c` from square-only to arbitrary M×K×N.
Verified correct (`0.000e+00` relative error against an independent
long-double reference) on: tall-skinny (4096×128×128), short-wide
(128×128×4096), large-K (256×4096×256), and deliberately awkward square
sizes (1023, 1025, 1537) and a fully irregular triple (4095×257×4093).

**A real, concrete finding, not just a correctness checkbox**: sizes one
off a clean MR/NR-tile boundary pay a large, measured performance
penalty. At N=1024 (exact multiple of both MR=4 and NR=8): ~111-125
GFLOP/s depending on blocking. At N=1023: **56.7 GFLOP/s** — roughly
*half*. At N=1025: **50.8 GFLOP/s** — worse still. Mechanism: every tile
touching the boundary now needs the masked/remainder write-back path
instead of the fast full-tile path, and for a size like 1023 or 1025
*most* tiles along that edge hit it. This is exactly the kind of result
review point 9 was asking for — awkward sizes aren't just a correctness
edge case, they're a real, first-order performance cliff for a
fixed-tile-shape kernel, worth knowing before shipping one without a
proper remainder strategy (or before choosing MR/NR shapes without
considering how often production workloads land off-boundary).

## J.7 An independent Strassen rebuild — cross-validated against Part H, with a real, explained discrepancy

Built independently (`src/avx2/ladder/strassen_rebuild.c`, `strassen_main.c`),
without reading Part H first. Different design choices, useful precisely
*because* they differ:

- **Workspace**: a single bump-pointer arena per top-level call (not
  Part H's "one batch allocation per level" via `bench_alloc` + offsets)
  — functionally the same "no malloc in the hot recursive path" goal, one
  level more general (handles arbitrary depth/cutoff combinations without
  a fixed 1-/2-level split).
- **A real concurrency bug caught before it shipped, not after**: the
  first version gave each of the 7 `#pragma omp task` branches a
  byte-identical copy of the parent arena struct (same base pointer, same
  bump-offset) — reasoned through *before* running it that this aliases
  memory the instant two sibling tasks actually run concurrently (both
  would bump-allocate from the identical starting address). Fixed by
  carving 7 disjoint sub-arenas from the parent, single-threaded, before
  any task is created. Verified correct after the fix at every depth
  0-4, every cutoff, N∈{256,512,1024,2048}, T∈{4,8} — `0.000e+00` relative
  error throughout, **plus** leaf-call and add/sub-call counts matching
  the theoretical 7^depth / 10·(7^depth−1)/6 exactly at every
  configuration, an extra correctness check beyond the numerical one.
- **Quadrants**: copies all 8 (simpler, more memory traffic) rather than
  Part H's 4-bare-view-plus-4-materialized-sums design (less traffic, more
  intricate). A real design difference, not a bug — Part H's approach is
  the more sophisticated/faster one.
- **Depth parameterized directly** (0-4 tested) rather than fixed
  1-/2-level variants — lets the crossover be located rather than checked
  at two preset points.
- **Cost model**: `leaf_s`, `addsub_s` (includes the quadrant-copy cost,
  deliberately grouped with add/sub as both are O(n²) data-movement
  overhead), `combine_s`, `workspace_s`, `sched_s` (task-creation/
  taskwait), each timed with `#pragma omp atomic`-protected accumulators
  — an aggregate CPU-time-like measure across concurrent tasks, not
  wall-clock (same caveat as J.2's kernel phases).

**The crossover, and why it lands at a very different N than Part H's:**
N=2048, T=8, 5 repeats each, non-overlapping:

| Depth | GFLOP/s (5 runs) |
|---|---|
| 0 (plain DGEMM) | 20.6, 20.9, 21.2, 20.9, 21.9 (mean 21.1) |
| 3 (cutoff=256) | 48.5, 51.8, 52.4, 51.8, 52.1 (mean 51.3) |

**Depth 3 beats plain DGEMM by ~2.4x already at N=2048** — vs. Part H's
crossover at N=8192, and Part H's own N=2048 result showing Strassen
*behind* (`dgemm_base` 458.2 vs `strassen_1level` 260.0). These are not
contradictory findings about Strassen — they're the same underlying fact
viewed through two very differently-optimized baselines. **This rebuild's
plain-DGEMM baseline is ~21 GFLOP/s; Part H's `dgemm_base` (hand-asm +
transpose-packing) is ~458 GFLOP/s at the same N** — over 20x faster.
Strassen's advantage is a fixed ~2x reduction in arithmetic work
(7 sub-problems instead of 8); against a weak baseline, that reduction
dominates almost immediately, because the baseline is already leaving so
much performance on the table that Strassen's *overhead* (copies,
adds, task scheduling) is cheap by comparison. Against Part H's
near-peak baseline, that same overhead is a much larger relative tax,
so it takes a much larger N before the arithmetic savings overcome it.
**Genuinely useful, generalizable conclusion neither rebuild would have
produced alone: the Strassen crossover point is not a fixed property of
the algorithm or even of the machine — it's a property of *the gap
between Strassen's overhead and the baseline kernel's own efficiency*,
and reports of "where Strassen wins" are meaningless without stating how
optimized the classical baseline being compared against is.** Also
confirms this rebuild's own depth-1/2 numbers from earlier testing
(N=2048: depth1 20.9, depth2 18.1 GFLOP/s — both *behind* depth 0) make
sense under the same lens: at shallow depth, this rebuild's task-per-
branch design underutilizes the 8-thread team (only 7 or 49 concurrent
leaves to spread across 8 threads, each leaf running its own DGEMM
single-threaded due to `max_active_levels(1)` — the same structural
pattern Part C's audit flagged in the *original* codebase's
`mm_parallel_strassen.c`, here intentional and by design, not a bug, but
with the same thread-underutilization consequence at shallow depth); only
at depth 3 (343 concurrent leaves) is there enough task-level parallelism
to fill 8 threads well, which is when the arithmetic savings finally show
through as a net win.

## J.8 An independent, broader accuracy sweep — cross-validated against Part I

Built independently (`src/avx2/ladder/accuracy_test.c`): 5 distributions
(uniform, gaussian, wide_range, ill_cond, cancel) × 4 methods (plain
DGEMM, Strassen depth 1/2/3) + MKL (built against the real
`/opt/intel/oneapi/mkl/2022.1.0` found in J.4), vs. an independently
written long-double reference (deliberately not sharing code with the
DGEMM/Strassen under test, so a shared bug can't hide from the check).

**Confirms Part I's core direction with a cleaner, monotonic signal**:
Frobenius relative error grows steadily with Strassen depth across every
distribution tested — e.g. uniform: 5.39e-16 (plain) → 1.07e-15 (depth1)
→ 1.52e-15 (depth2) → 2.20e-15 (depth3), roughly doubling by depth 3,
same shape for gaussian/wide_range/ill_cond/cancel. MKL matched or beat
plain DGEMM's error on every distribution (as expected from a mature,
well-tuned production library). All of *this* rebuild's own errors stayed
in the 1e-16 to 1e-9 range — nowhere near Part I's dramatic 1436%
failure.

**That's not a disagreement — it's a methodologically important
negative result.** This sweep's `wide_range` distribution assigns each
matrix *entry* an independent random magnitude (1e-8 to 1e8); Part I's
`wide_dynamic_range` distribution scales entire *rows* together across
~12 orders of magnitude. Per-entry randomization doesn't create the
coherent, correlated cancellation structure row-scaling does — so it
didn't trigger the catastrophic failure Part I found, despite both being
labeled "wide dynamic range." **Practical lesson for anyone else building
an accuracy stress test: independent per-entry randomization is a weak
adversarial distribution for this purpose; correlated (row- or
column-scaled) ill-conditioning is what actually breaks Strassen's extra
additions.** This rebuild's `ill_cond` distribution (rows built as
near-linear-combinations of a few base rows) is structurally closer to
Part I's effective design and did show somewhat elevated error (up to
~1e-9 max componentwise, vs. ~1e-10 for plain DGEMM on the same data) —
real and in the right direction, but nowhere near Part I's dramatic
result, most likely because the perturbation added here (1e-10, keeping
the matrix safely non-singular) was much gentler than whatever row-scale
spread Part I used. **Recommendation for whoever extends this next**:
Part I's row-scaled construction is the one to build on for a real
accuracy stress test; this sweep's contribution is the breadth (5
distributions, depths 1-3 instead of fixed levels, MKL included) and the
methodological lesson about *why* distribution design matters this much.

## J.9 Statistical rigor, applied where it mattered most this round

Every genuinely new numeric claim above got at least 5 repeats: J.1's
three-environment scheduling extension, J.4's compiler comparison, and
J.7's depth-0-vs-depth-3 crossover. All three produced either a clean,
non-overlapping gap (J.4, J.7) or a clearly overlapping "no real
difference" band (J.1) — in every case, 5 repeats was enough to tell
those apart confidently, even though it falls short of review point 10's
literal "20-30 runs with confidence intervals." J.3's blocking sweep and
J.6's shape sweep are single-run per point (a full one-parameter-at-a-
time surface at 5 repeats per point would be several times the wall-clock
cost) — their *shapes* (KC's monotonic effect, the clean N=1023/1024/1025
cliff) are large and consistent enough to trust qualitatively without
that, but exact GFLOP/s values there should be treated the same way
Parts B and H already flag their own single-sample numbers: indicative of
shape, not precise.

---

# Part K — Repo Map / Where Things Live

As of the 2026-10 repo reorganization, every path below is physically
split into `audited_submission/` (Generation 1 -- the original paper
and benchmark suite being audited, untouched in design, bug-fixed in
place) and `src/` (Generation 2 -- everything built new for this
project's own audit/research). See the top-level `README.md`,
`audited_submission/README.md`, and `src/README.md` for the
reasoning; this map just gives exact paths.

```
Workstream A (new AVX2 DGEMM vs. production BLAS) -- all under src/:
  setup/wsl_env_setup.sh                  -- WSL2 toolchain + OpenBLAS/BLIS/MKL install
                                              (repo-root setup/, not src/ -- environment
                                              bring-up, not implementation)
  src/bench/, src/baselines/              -- shared harness + OpenBLAS/BLIS/MKL/naive drivers
  src/avx2/matmul_4x8.c ... matmul_v7_*.c -- C-intrinsics ablation series (v1-v7)
  src/avx2/sweep/                         -- 9-shape microkernel sweep + generator
  src/avx2/asm_kernel/                    -- hand-written assembly method: kernel_6x8.S,
                                              matmul_asm.c (v1, scalar packing),
                                              matmul_asm_v2_vecpack.c (v2, gather-based B pack),
                                              matmul_asm_v3_transpack.c (v3, transpose-based B pack),
                                              matmul_asm_v4_dynsched.c (v4, + dynamic scheduling --
                                              current best, see B.8/B.9),
                                              kernel_6x8_v2_cprefetch.S + matmul_asm_v5_cprefetch.c
                                              (v5, + C-tile prefetch -- no gain, see B.11)
  src/reference/                          -- curated OpenBLAS/BLIS excerpts + comparison notes
                                              (Research/reference-libs/ also has full local clones of
                                              OpenBLAS, BLIS, and oneDNN, gitignored, for source
                                              inspection -- see B.2, B.10, B.11)
  src/avx2/portable/                      -- portable/generalizable harness (see B.12): hw_detect.c/.h,
                                              blocking_model.c/.h, matmul_portable.c (guided schedule),
                                              matmul_portable_staticctrl.c (ablation control),
                                              kernel_6x8.S (copy of asm_kernel's, unchanged).
                                              Also deployed and tested on glogin.cs.vt.edu (~/portable_test
                                              there, plus ~/openblas_src for the rebuilt-with-threading
                                              OpenBLAS comparison) -- not just this dev machine.
  src/topology/                           -- P-core/E-core classifier (WSL2 vs. native Windows),
                                              peak_scaling_native.c (multi-thread pinned FMA
                                              benchmark, see B.13)
  src/roofline/stream_triad.c             -- STREAM-style bandwidth benchmark
  src/strassen2/                          -- rebuilt Strassen (see Part H): dgemm_base.c/.h (tuned
                                              kernel as a reusable ld-generalized library call),
                                              strassen_ops.c/.h (fused add/sub/accumulate on strided
                                              views), strassen.c/.h (strassen_1level/2level, pooled
                                              workspace, thread-aware task/sequential cutoff),
                                              test_strassen.c (correctness), bench_strassen.c
                                              (effective-GFLOP/s crossover benchmark)
  src/accuracy/                           -- numerical accuracy experiments (see Part I):
                                              distributions.c/.h (uniform, wide-dynamic-range,
                                              clustered-cancellation), bench_accuracy.c
  src/avx2/ladder/                         -- Parts G & J: portable (MSVC+gcc) phase-timed harness --
                                              matmul_ladder.c (v1, stages A/C-G), timer.h,
                                              sweep.sh/.ps1, sweep_native_windows.csv / sweep_wsl2.csv
                                              / sweep_glogin.csv, pinning_experiment.csv (Part G's
                                              P/E-core repeated-measurement experiment; deployed on
                                              native Windows, WSL2, and glogin.cs.vt.edu),
                                              matmul_ladder2.c (Part J: runtime-configurable KC/MC/NC,
                                              rectangular M/K/N, finer alloc/pack/kernel_compute/
                                              kernel_write phase split, optional huge-page hint),
                                              blocking_sweep.sh/.csv (J.3), strassen_rebuild.c +
                                              strassen_main.c (J.7: independent pooled-arena Strassen
                                              rebuild, depth-parameterized, cost-modeled),
                                              accuracy_test.c (J.8, extended in Part M: 5-distribution x
                                              depth 1-3 x {plain, classic Strassen, Dumas et al. 2024
                                              alternate-basis} + MKL accuracy sweep), matmul_ladder_noinline.c
                                              + dis_gcc/clang/icx.txt + gcc_loop.s/clang_loop.s (J.4:
                                              compiler comparison + llvm-mca), compiler_comparison_raw.txt,
                                              dumas_strassen_rebuild.c + dumas_main.c (Part M: Dumas/
                                              Pernet/Sedoglavic 2024 alternate-basis Strassen, same
                                              arena/task/phase-timing design as strassen_rebuild.c).

Workstream B (correctness audit + revalidation of the original codebase)
-- audited code under audited_submission/, Workstream B's own new
tooling under src/:
  audited_submission/avx/kernel_avx2.c, strassen_avx2.c,
    run_local_avx2.sh, test_local.c    -- real, permanent AVX2 build for
    correctness-testing without AVX-512 hardware (new tooling, but kept
    alongside the avx/ code it tests rather than under src/, since it's
    a drop-in test harness FOR that code, not a separate research line)
  src/harness/count_calls.h, run_one.c, harness_sweep.py  -- the
    instrumentation and sweep driver used for Part E (included from
    audited_submission/ files via ../../src/harness/..., see src/README.md)
  revalidation/correctness/   -- bug evidence, exact fixes, final pass/fail sweep
  revalidation/benchmarks/    -- wall-clock revalidation runs (2 machines)
  revalidation/harness/       -- hardware-counter root-cause data (Part E)
  revalidation/dumas2024/     -- Part M: formula verification script, correctness
                                  sweep, accuracy + performance comparison CSVs
  revalidation/README.md      -- index + instructions for adding a new run later

Paper (Part L's publication-strategy framing, turned into an actual draft):
  paper/main.tex, paper/references.bib, paper/README.md  -- Overleaf-ready
    LaTeX paper, structured around Part L.3's three headline
    contributions (Part C's correctness audit, Part G's
    measurement-bias case study, Part M's Dumas-2024 accuracy/performance
    study), with everything else as supporting/appendix material
```

---

# Part L — Related Work & Publication Strategy

Prompted directly by the request to look at what's already been published
in this space and figure out how to make this project's eventual paper
something that gets built on and cited for years, not a one-off speed
report. This part is a literature landscape (L.1), a gap analysis of where
Parts A-K's *actual, already-done* work sits against that landscape (L.2),
concrete framing/venue/artifact recommendations (L.3), one specific new
angle worth adding before submission (L.4), a priority ranking (L.5), and
the bibliography (L.6). Searched 2026-09-09; citation counts and "hasn't
been done" claims below are as of that date and should be re-checked
before submission, since this is a moving target — L.4's LLM-kernel-gen
area especially is changing month to month.

## L.1 The landscape — what's already been done

Four lineages, roughly in order of how directly they compete with this
project's own claims:

**1. High-performance GEMM engineering itself.** Goto & van de Geijn's
["Anatomy of High-Performance Matrix Multiplication"](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS.pdf)
(TOMS 2008) is the foundational multilevel-blocking/packing model this
whole codebase (and the audited paper) descends from — the MC/KC/NC
hierarchy, the packed-panel microkernel, all of it traces back here. Van
Zee & van de Geijn's [BLIS framework](https://www.cs.utexas.edu/~flame/pubs/BLISTOMSrev2.pdf)
(TOMS 2016, SIAM SIAG/SC Best Paper 2020) generalized Goto's approach into
a portable framework where only a tiny microkernel needs re-tuning per
architecture — this is the design pattern Workstream A's `src/avx2/portable/`
(hw_detect + blocking_model + swappable microkernel) is independently
reinventing at smaller scale. Neither paper is about Strassen or
measurement methodology; they're the ceiling this project's plain-DGEMM
numbers (Part B) are measured against.

**2. Practical fast matrix multiplication (Strassen).** Huang, Smith,
Henry & van de Geijn, ["Strassen's Algorithm Reloaded"](https://dl.acm.org/doi/10.5555/3014904.3014983)
(SC'16) is the closest direct competitor to Parts H/J.7: it showed
Strassen's add/subtract steps can be fused *into* the packing routines of
an already-optimized DGEMM (zero extra workspace beyond what packing
already needs), making Strassen practical at far smaller N and even for
skinny rank-k updates than conventional wisdom held. A
[GPU follow-up](https://dl.acm.org/doi/10.1145/3372419) extended the idea
to CUDA. Ballard, Demmel, Holtz & Schwartz's
[communication-optimal parallel Strassen](https://arxiv.org/abs/1202.3173)
(SPAA'11/IPDPS'12) is the theory side: proved lower bounds on
inter-processor communication for Strassen-based algorithms and gave an
algorithm matching them, with measured 24-184% speedups over both
classical and prior Strassen-parallel approaches on a Cray XT4. This
project's own Strassen rebuilds (Part H: fused-into-packing, closer to
Huang's design; Part J.7: pooled bump-pointer arena, a different
workspace strategy) are a *second, independent* implementation of the
same underlying idea family — useful as corroboration, not as something
brand-new on its own.

**3. Numerical accuracy of fast matrix multiplication.** The classical
textbook algorithm satisfies a *component-wise* error bound; Strassen and
every other fast algorithm only satisfies a weaker *norm-wise* bound —
this is known, cited theory, and it's exactly what Parts I/J.8 measured
empirically (the 1400%+ max-componentwise-error blowup under row-scaled
ill-conditioning, while Frobenius/norm-wise error stayed far smaller —
literally a live demonstration of the component-wise-vs-norm-wise gap the
theory predicts). What's genuinely new and *very* recent: Dumas, Pernet &
Sedoglavic, ["Strassen's algorithm is not optimally accurate"](https://arxiv.org/abs/2402.05630)
(ISSAC 2024) — published eight months ago as of this search — propose an
alternate-basis 7-multiplication 2x2 algorithm with *better achieved
accuracy in practice* than classic Strassen at the same leading-order
complexity, by minimizing an error growth factor along the tensor
decomposition's orbit. As far as this search found, nobody has yet run
that specific 2024 algorithm through a full hardware-measured,
packed/blocked, statistically-repeated GEMM benchmark the way Parts I/J.8
already do for classic Strassen — see L.4.

**4. Measurement methodology, reproducibility, and the current hot
adjacent field (LLM kernel generation).** Mytkowicz, Diwan, Hauswirth &
Sweeney, ["Producing Wrong Data Without Doing Anything Obviously Wrong!"](https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf)
(ASPLOS'09) is the canonical proof that "insignificant" experimental-setup
choices (environment variable size, link order) can flip a benchmark's
conclusion — this is the methodological ancestor of Part G's whole
exercise (does the big scheduling win survive real repetition, or was it
an artifact of the environment it was first measured in?), though
Mytkowicz's paper itself never touches GEMM/Strassen specifically.
Williams, Waterman & Patterson's [Roofline model](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
(CACM 2009, 900+ citations) is the standard visual compute-vs-bandwidth
ceiling model this project's own compute-ceiling work (J.6/the 10-question
list's item 4) is built on. Closer to home: a very recent (2025) preprint,
["High Performance Matrix Multiplication" (Davis)](https://arxiv.org/pdf/2509.04594),
runs 30 trials per (algorithm, size) pair, reports bootstrap 95% CIs, and
uses a Games-Howell test to decide whether performance differences are
statistically real — this is close to the statistical rigor this
project's own repeated-measurement work (Part G, J.1, and the "single-run
effects vanish under repetition" lesson) has been doing informally;
formally adopting bootstrap CIs + a proper multiple-comparison test (not
just eyeballing overlapping distributions) would bring this project's
statistics up to that same current bar. Also relevant, for scoping honestly
rather than over-claiming novelty:
["Reproducible and Accurate Matrix Multiplication"](https://www.researchgate.net/publication/282866221_Reproducible_and_Accurate_Matrix_Multiplication_for_High-Performance_Computing)
and ["GEMMbench"](https://arxiv.org/pdf/1511.03742) both already use the
word "reproducible" for GEMM, but in a *different* sense than this
project — GEMMbench is a *collaborative crowdsourced benchmark-collection
framework* (closer to a submission portal than a methodology paper);
"Reproducible and Accurate Matrix Multiplication" means *bit-for-bit
reproducibility despite floating-point non-associativity and dynamic
scheduling*, an orthogonal concern to the "are our conclusions an artifact
of measurement noise or virtualization" question this project asks. Don't
conflate the three meanings of "reproducible" when writing the related-work
section — they're genuinely different problems that happen to share a word.

On the reproducibility/artifact-evaluation trend itself: a 2025 arXiv
report, ["Challenges of Practical Reproducibility for Systems and HPC Computer Science"](https://arxiv.org/pdf/2505.01671),
and the [PMC piece on research artifacts and citations](https://pmc.ncbi.nlm.nih.gov/articles/PMC9044204/)
both document that papers with available/evaluated/reusable artifacts get
cited more and get reused as building blocks more — this is a real,
current, measurable incentive, not just good practice for its own sake.
SC's Reproducibility Initiative (mandatory Artifact Description appendix
since 2019) and [ACM TOMS's Replicated Computational Results track](https://www.osti.gov/servlets/purl/1235340)
(a manuscript can earn an explicit RCR designation by having an
independent reviewer actually re-run the artifact) are the two concrete
mechanisms for capturing that incentive in this exact subfield.

Finally, the fastest-moving adjacent area right now: LLM-driven kernel
generation. [QiMeng-GEMM](https://ojs.aaai.org/index.php/AAAI/article/view/34461/36616)
(AAAI 2025) uses an LLM to generate high-performance GEMM code directly;
CUDA-L2 (RL-based, reported to surpass cuBLAS on some shapes) and PEAK
(a natural-language-driven GPU kernel performance-engineering assistant)
are 2025 siblings in the same space, alongside a broader
[survey/awesome-list of LLM-driven kernel generation](https://arxiv.org/html/2601.15727v3).
This is genuinely new since this session's earlier literature pass (most
of it postdates this project's original AVX-512 paper by years) — see
L.4 for why it's a real opportunity rather than just background color.

## L.2 Gap analysis — where Parts A-K's actual work already sits, honestly

Reading L.1 against what's *actually been executed and measured* in Parts
A-K (not what's aspirational):

- **Part C (the correctness-bug audit) is the strongest, least-crowded
  claim in the whole project.** Nobody in L.1's landscape published a
  paper whose entire contribution is "we tried to reproduce a specific
  published AVX-512 DGEMM/Strassen paper's implementation and found real,
  execution-proven correctness bugs (the B·A vs. A·B packing bug chief
  among them)." That is *exactly* the shape of contribution the TOMS RCR
  track and SC's Reproducibility Initiative were built to reward, and it
  is genuinely rare — most "we reproduced X" write-ups in this field are
  informal blog posts, not citable artifacts with a real evidence trail.
  Part C already has that evidence trail (`revalidation/correctness/`).
- **Part G (is the scheduling win real or a virtualization artifact?) is
  a novel *application* of an old, well-cited methodology (Mytkowicz
  2009) to a domain (DGEMM/OpenMP scheduling claims) that paper never
  touched, with real before/after data showing the effect vanishing under
  repetition.** This search found no existing paper doing this specific
  combination — general "cloud/VM benchmarking noise" papers exist (see
  L.1.4), but none anchored to a *matrix-multiplication scheduling*
  headline claim the way Part G is. This is the most exportable, broadly-
  citable lesson in the project: "a clean-looking 2x scheduling win from a
  single run on a virtualized dev machine can be pure measurement
  artifact" is a warning relevant to far more than this one codebase.
- **Parts H/J.7 (independent Strassen rebuilds) are solid corroboration,
  not novelty on their own** — Huang et al. 2016 already showed the core
  idea (fuse add/sub into packing, avoid extra workspace) works well.
  Having *two independently-built* rebuilds inside one project that
  *disagree* on crossover point and *explaining why* (J.7's discrepancy
  analysis: crossover depends on baseline optimization level, not just
  algorithm) is a genuinely useful methodological point about how
  Strassen benchmarks should report their baseline's tuning level — but
  it needs to be framed as that specific point, not as "we built Strassen
  faster than Huang et al.," which it doesn't claim and shouldn't.
- **Parts I/J.8 (accuracy sweeps) are good, real data but currently
  compare against classic Strassen only — they don't yet engage with the
  one paper in L.1 that's most directly relevant and newest (Dumas et
  al. 2024).** This is the single highest-leverage gap to close before
  writing anything up — see L.4.
- **Part B's compiler/microkernel ablation work (and J.4's GCC-vs-
  Clang/icx auto-unroll finding) is solid engineering but the most
  crowded corner of the landscape** (closest cousins: the AVX-512
  microkernel papers in L.1.1, none from a major venue, mostly Cluster
  Computing-tier). Keep it in the paper as supporting evidence, not as a
  headline claim — it won't be the reason anyone cites this work.

## L.3 Framing, venue, and artifact recommendations

1. **Frame the paper as a methodology/reproducibility contribution, not a
   speed-record contribution.** Papers like Roofline (2009) and Mytkowicz
   (2009) are still being cited 15+ years later precisely because they
   gave people a *reusable way to check their own work*, not a number
   that ages out the moment someone posts a faster kernel. A "we audited
   a specific DGEMM/Strassen paper, found real bugs, then built a portable
   measurement-bias-aware harness that anyone can point at their own
   GEMM/Strassen claims" framing has a much longer citation half-life than
   "we got N% of peak."
2. **Publish the harness and full dataset as a standalone, DOI'd artifact
   (Zenodo, or an SC/TOMS-style AD appendix) independent of wherever the
   prose paper lands.** Code/data citations compound on their own and are
   frequently cited by people who never cite the paper's prose at all —
   this is the direct, documented mechanism behind the reproducibility-
   citation premium in L.1.4. `revalidation/`, `src/avx2/ladder/`, and the CSV
   data already in this repo are most of the way there; what's missing is
   a single top-level README describing how to re-run everything end to
   end (the RCR track's exact requirement) and a license file.
3. **Target, in order of fit:** (a) ACM TOMS's RCR track specifically for
   Part C alone — it is a near-perfect fit for that track's stated
   purpose; (b) PMBS (Performance Modeling, Benchmarking and Simulation of
   HPC Systems, an SC-colocated workshop) for the Part G measurement-bias
   methodology angle, which is squarely that workshop's subject matter;
   (c) an arXiv preprint immediately regardless of (a)/(b) — it's citable
   the moment it's posted, doesn't block on any venue's timeline, and is
   exactly how the two most-relevant very-recent related papers found in
   L.1 (Dumas et al. 2024, Davis 2025) are themselves circulating.
4. **Pick 2-3 headline contributions and subordinate the rest as
   supporting material**, rather than presenting all of Parts A-K as
   equally central:
   - The correctness-bug audit (Part C) as a replicated-results
     contribution.
   - The measurement-bias case study (Part G) as a portable methodological
     lesson, explicitly extending Mytkowicz 2009 into a new domain.
   - The accuracy study, extended per L.4, as the most current and least-
     covered empirical contribution.
5. **Adopt bootstrap confidence intervals and a real multiple-comparison
   test** (matching Davis 2025's approach: 30 trials, bootstrap 95% CI,
   Games-Howell) **for every headline number**, replacing the current
   "single-run vs. repeated-run" qualitative framing in Parts D/G/J with a
   quantitative one. This single change makes every existing chart
   defensible against exactly the kind of reviewer pushback a methodology-
   framed paper will attract.

## L.4 The concrete next experiment worth running: benchmark the 2024 accuracy-optimized Strassen variant

**Done — see Part M.** This was the sharpest, most current, least-
crowded gap L.1/L.2 surfaced; it's now implemented, symbolically
verified against the paper's own straight-line program (a real
transcription bug caught first), run for correctness, and run for a
head-to-head accuracy-and-performance comparison against classic
Strassen at matched depth. Headline results: measurably better
Frobenius-relative accuracy at every matched depth (qualitatively
confirms the paper), at a measurable, now-quantified performance cost
(~20% fewer GFLOP/s at N=2048, depth=3). The plan as originally written
is kept below for the record; M.6 lists what's still open (cross-machine
confirmation, the row-correlated catastrophic-error case, bootstrap
CIs).

- Implement Dumas, Pernet & Sedoglavic's alternate-basis 2x2 algorithm
  (arXiv:2402.05630) as one more branch selectable in `strassen_rebuild.c`
  (J.7's arena/phase-timing infrastructure already generalizes to "which
  7-multiplication formula to use at each recursion level" — it doesn't
  need to be rebuilt, just a second set of the S1..S10/P1..P7 combination
  formulas swapped in).
- Run it through the *exact same* harness Parts I/J.8 already built:
  `accuracy_test.c`'s five distributions (uniform/gaussian/wide_range/
  ill_cond/cancel), the long-double-reference comparison, both
  frob_rel and max_componentwise_rel metrics, at the same depths/cutoffs
  as classic Strassen.
- Also run it through `strassen_main.c`'s performance/phase-timing path,
  since Dumas et al.'s paper is an accuracy paper and (as far as this
  search found) doesn't report hardware wall-clock/phase-timed
  performance numbers on a real packed-blocked DGEMM leaf the way this
  project's harness does — that pairing (their accuracy claim + this
  project's performance instrumentation, on the same runs) is not
  something either side has published, and it's a small, well-scoped
  addition given the infrastructure that already exists.
- Report both algorithms side by side: does the improved accuracy come
  with a measurable performance cost (extra adds/subs, different
  operation count) at real, blocked, multi-threaded matrix sizes — not
  just in the asymptotic operation-count sense Dumas et al.'s own paper
  reasons about?

This closes L.2's identified gap directly, engages a paper that is only
months old (making the eventual paper's related-work section current
rather than stale on arrival), and produces a specific, checkable,
non-obvious number (the accuracy/performance trade-off of the 2024
variant under realistic execution) that the original 2024 paper's authors
themselves — and anyone else who reads it — would plausibly want to cite.

*(Lower-priority forward-looking hook, not recommended before the above:
LLM-driven kernel generation (L.1.4) is real and growing fast, but running
this project's harness against an LLM-authored kernel is a weaker,
softer addition than L.4's accuracy experiment — it demonstrates the
harness is reusable, which is a nice-to-have, not a checkable empirical
claim the way the accuracy/performance trade-off number is. Worth a single
paragraph in future work, not a section to build out now.)*

## L.5 Priority ranking for remaining effort

1. ~~Run the Dumas et al. 2024 accuracy-optimized Strassen variant
   through the existing accuracy + performance harness (L.4)~~ — **done,
   see Part M.**
2. Adopt bootstrap CI + Games-Howell (or equivalent) across Parts D/G/J's
   headline numbers (L.3.5) — turns "we repeated it and the effect
   vanished" into a number a reviewer can't wave away.
3. Write the top-level reproducibility README + license (L.3.2) and cut a
   Zenodo release — cheap, and the citation-compounding benefit starts
   accruing the moment it's public, independent of paper acceptance.
4. Draft the paper itself around the three headline contributions in
   L.3.4, with Parts A/B/D/E/F/H/I/J.1-J.6 as supporting/appendix
   material rather than equally-weighted sections.
5. Only after 1-4: consider the LLM-kernel-generation forward-looking
   paragraph (L.4's parenthetical) if there's room.

## L.6 Bibliography (works consulted for this part, accessed 2026-09-09)

- Goto, K. & van de Geijn, R. A. "Anatomy of High-Performance Matrix
  Multiplication." *ACM TOMS* 34(3), 2008.
  https://www.cs.utexas.edu/~flame/pubs/GotoTOMS.pdf
- Van Zee, F. G. & van de Geijn, R. A. "BLIS: A Framework for Rapidly
  Instantiating BLAS Functionality." *ACM TOMS* 42(2), 2016.
  https://www.cs.utexas.edu/~flame/pubs/BLISTOMSrev2.pdf
- Huang, J., Smith, T., Henry, G. & van de Geijn, R. "Strassen's
  Algorithm Reloaded." *SC'16*. https://dl.acm.org/doi/10.5555/3014904.3014983
- "Strassen's Algorithm Reloaded on GPUs." *ACM TOMS*, 2020.
  https://dl.acm.org/doi/10.1145/3372419
- Ballard, G., Demmel, J., Holtz, O. & Schwartz, O. "Communication-Optimal
  Parallel Algorithm for Strassen's Matrix Multiplication."
  arXiv:1202.3173 / SPAA'11.
- Dumas, J.-G., Pernet, C. & Sedoglavic, A. "Strassen's algorithm is not
  optimally accurate." *ISSAC 2024*. https://arxiv.org/abs/2402.05630
- Mytkowicz, T., Diwan, A., Hauswirth, M. & Sweeney, P. F. "Producing
  Wrong Data Without Doing Anything Obviously Wrong!" *ASPLOS'09*.
  https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf
- Williams, S., Waterman, A. & Patterson, D. "Roofline: An Insightful
  Visual Performance Model for Multicore Architectures." *CACM* 52(4),
  2009. https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf
- Davis, E. "High Performance Matrix Multiplication." arXiv:2509.04594,
  2025.
- "Reproducible and Accurate Matrix Multiplication for High-Performance
  Computing." https://www.researchgate.net/publication/282866221
- "GEMMbench: a framework for reproducible and collaborative benchmarking
  of matrix multiplication." arXiv:1511.03742.
- "Report on Challenges of Practical Reproducibility for Systems and HPC
  Computer Science." arXiv:2505.01671, 2025.
- "Research artifacts and citations in computer systems papers."
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9044204/
- ACM TOMS "Replicated Computational Results Initiative" editorial.
  https://www.osti.gov/servlets/purl/1235340
- QiMeng-GEMM. *AAAI 2025*.
  https://ojs.aaai.org/index.php/AAAI/article/view/34461/36616
- "Towards Automated Kernel Generation in the Era of LLMs" (survey).
  https://arxiv.org/html/2601.15727v3

---

# Part M — The Dumas et al. (2024) Alternate-Basis Strassen, Implemented and Measured

Part L.4's follow-up experiment, actually executed: implement Dumas,
Pernet & Sedoglavic's 2024 accuracy-optimized alternate-basis 2x2
algorithm as a real recursive block algorithm, and run it through the
exact accuracy and phase-timing harness Parts I/J.7/J.8 already built,
so the only thing that changes between "classic Strassen" and "this
2024 variant" is the 7-multiplication formula itself — not the leaf
kernel, not the packing, not the workspace strategy, not the
parallelism. Code in `src/avx2/ladder/dumas_strassen_rebuild.c` +
`dumas_main.c` + the extended `accuracy_test.c`; evidence, scripts, and
raw CSVs in `revalidation/dumas2024/`.

## M.1 Getting the formula right before writing any C

Dumas et al.'s paper states the algorithm only as a straight-line
program (their Table 3, the concrete instantiation of their Equation
36) in the paper's own a_ij/b_ij subscript convention. Transcribing that
program literally against standard row-major `A[i][j]`/`B[i][j]`
indexing produces a **wrong** result — off by exactly the antisymmetric
`a12<->a21`, `b12<->b21` terms, i.e. the paper's subscript convention
isn't row-major the way this codebase's arrays are. This was caught
*before* any implementation code was written, by transcribing the SLP
into `sympy`, expanding it symbolically, and comparing against
`C = A@B` — the naive transcription gave four nonzero symbolic diffs;
brute-forcing input/output relabelings found the exact one that zeroes
all four (swap which physical entry `a12`/`a21` and `b12`/`b21` feed
into the formula, leave the output labels alone). The corrected formula
was then re-verified from a clean symbolic derivation (zero diff on all
four output blocks, confirmed by a numeric spot check) before being
transcribed into C. `revalidation/dumas2024/verify_formula.py` is that
verification, kept as a script specifically so it can be re-run if
`dumas_strassen_rebuild.c`'s base-case math is ever touched again —
this is exactly the kind of silent, plausible-looking correctness bug
Part C's original B·A-vs-A·B audit was about, and the lesson (verify
formulas against ground truth before implementing, not after) is the
same one.

The corrected formula, block-generalized (each `a_ij`/`b_ij` below is a
k×k submatrix, `+`/`-`/scalar-multiply are elementwise, and the 7
"multiplications" are recursive `dumas_rec` calls or a leaf GEMM):

```
t1 = (sqrt3/3) A22          s1 = (sqrt3/3) B12
t2 = A12 + t1                s2 = s1 - B11
t3 = A21 + t2                s3 = s2 + B22

l1 = (sqrt3/2) A11 + 0.5 t3      r1 = 2 s1
l2 = A21 - t1                    r2 = s2
l3 = t2                          r3 = s1 - B22
l4 = 2 t1                        r4 = 0.5 s3 - (sqrt3/2) B21
l5 = l2 - l1                     r5 = r3 + r4
l6 = l5 + l4                     r6 = r1 - r5
l7 = l5 + l3                     r7 = r5 - r2

p_i = l_i * r_i   for i = 1..7   (the 7 "multiplications")

w2 = p5 + p1 + p6
w1 = p7 + p6
w3 = w2 - p2
w5 = (p4 + w2) / 2
C12 = p1 - p3 - w5
C21 = w3 - w5
C22 = sqrt3 * w5
C11 = (sqrt3/3) * (w3 - C12 - 2*w1)
```

The algorithm is stated by its authors to be *non-commutative*, which
is exactly the property that makes a scalar 7-multiplication formula
safe to reuse recursively at the block/matrix level the way classic
Strassen's is (a formula that only holds for commuting scalars would
not necessarily hold once `a_ij`/`b_ij` become non-commuting matrix
blocks) — confirmed empirically below, not just taken on faith from the
paper's claim.

## M.2 Implementation

`dumas_strassen_rebuild.c` mirrors `strassen_rebuild.c` (Part J.7)
almost exactly: same bump-pointer arena, same per-level disjoint
sub-arena carving for the 7 sibling `#pragma omp task` branches (the
identical concurrency hazard J.7 already found and fixed — giving
parallel tasks byte-identical arena snapshots would alias — is fixed
here the same way, by construction, from the start), same
`leaf_s`/`addsub_s`/`combine_s`/`workspace_s`/`sched_s` phase-timing
struct. The only real differences: 33 arena buffers per recursion level
instead of classic Strassen's 25 (8 quadrant copies + 3 T's + 3 S's + 6
L's + 6 R's + 7 P's, vs. 8 quadrants + 7 P's + 10 S's — this formula's
SLP has more named intermediates), and the combination math itself
(scale-heavy axpby operations with irrational `sqrt(3)` coefficients,
not classic Strassen's plain adds/subs).

`matmul_ladder2.c` needed one small, generally-useful fix to support
this: it had no include guard, so a translation unit wanting *both*
`strassen_rebuild.c` and `dumas_strassen_rebuild.c` (to compare them
side by side in one accuracy-test binary) would double-include it and
hit the exact "double-inclusion" duplicate-definition class of bug this
project already hit once before (see the Errors-and-fixes history).
Added an `#ifndef MATMUL_LADDER2_INCLUDED` guard around the whole file
— a strict superset fix, safe for every existing single-include caller,
and it's what let `accuracy_test.c` simply `#include` both Strassen
variants without any restructuring.

## M.3 Correctness

`dumas_main.c` (a CLI driver identical in shape to `strassen_main.c`)
run at N=64/128/256/512 × depth 1/2/3, T=4, against the harness's
existing `reference_check` (independent long-double accumulation):
relative error `4.7e-16` to `1.8e-15` at every single configuration —
machine precision, not "close." The block generalization of the 2024
scalar formula is genuinely, execution-verified correct, not just
correct in the 2×2 scalar case the paper itself analyzes.
(`revalidation/dumas2024/correctness_sweep.sh`, run 2026-09-09 on the
dev machine via WSL2 gcc 13.3.0.)

## M.4 Accuracy: does it actually beat classic Strassen once it's really packed/blocked?

`accuracy_test.c`, extended with `dumas_depth1/2/3` methods alongside
the existing `plain_dgemm`/`strassen_depth1/2/3`, run at N=512 across
all 5 distributions (uniform, gaussian, wide_range, ill_cond, cancel;
`revalidation/dumas2024/accuracy_results_N512.csv`). Frobenius-relative
error, at every matched recursion depth, is consistently **lower** for
the Dumas variant than for classic Strassen — e.g. gaussian at depth 3:
classic `2.27e-15`, Dumas `1.40e-15` (~38% lower); wide_range at depth
2: classic `1.50e-15`, Dumas `1.11e-15`. This qualitatively reproduces
the paper's headline claim, and does so under conditions the paper's
own evaluation didn't cover: real packing/blocking, real OpenMP task
parallelism, a proper independent long-double reference rather than
whatever the paper's own accuracy harness used. Max-componentwise-
relative error is a noisier, more mixed picture (Dumas is lower on some
distribution/depth pairs, higher on others) — consistent with the
paper's own bound being a Frobenius/2-norm-type bound (γ₂,₁), not a
componentwise one, so the advantage should be expected to show up
cleanest exactly where it does. (At these matrix sizes and this
uncorrelated-random distribution design, none of the runs — classic or
Dumas — come close to the >1400% catastrophic error Part I/J.8 measured
under *row-scaled, correlated* ill-conditioning; this run's `ill_cond`
distribution is the milder, per-entry near-rank-deficient construction
already in `accuracy_test.c`, not that specific catastrophic case. A
follow-up applying Part I's row-correlated construction to the Dumas
variant, to see whether the improved accuracy narrows or fails to
narrow that specific catastrophic gap, is still open — see M.6.)

## M.5 Performance: the accuracy improvement is not free, and now that cost has a number

Head-to-head `strassen_main` vs. `dumas_main`, same KC/MC/NC, same leaf
kernel, same thread count, at N=512/1024/2048 × depth 1/2/3, T=8, 5 reps
best-of (`revalidation/dumas2024/perf_comparison.sh` /
`perf_comparison_results.csv`, run 2026-09-09, dev machine, WSL2 gcc
13.3.0). Consistent pattern at every configuration:

- `leaf_s` (the shared leaf GEMM, identical code between the two
  binaries) matches within run-to-run noise — e.g. at N=2048, depth=3:
  classic `0.293268s`, Dumas `0.300523s`. This is the control: it
  confirms any slowdown is attributable specifically to the formula,
  not to some accidental difference elsewhere in the harness.
- `addsub_s` (the combine/scale phase, where the two formulas actually
  differ) is consistently and substantially higher for Dumas — at the
  same N=2048, depth=3 point: classic `0.259021s` vs. Dumas
  `0.407543s`, a ~57% increase. Mechanically unsurprising: this
  formula's SLP has 24 add/scale operations (several of them scale-by-
  irrational-`sqrt(3)` axpby operations, not classic Strassen's plain
  adds/subs) against roughly 18 for classic Strassen's 10-combination
  version.
- Net effect on throughput: at N=2048, depth=3, classic Strassen
  reaches `127.673` GFLOP/s where Dumas reaches `101.271` GFLOP/s — a
  **~20.7% throughput cost** for the accuracy improvement measured in
  M.4, at this depth/size/thread-count. The gap is present but smaller
  at shallower depth (fewer combine levels executed): at depth=1,
  N=2048, classic `50.793` vs. Dumas `47.560` GFLOP/s (~6.4% cost).

As far as Part L.1's literature search found, **neither Dumas et al.'s
own paper (which reports only asymptotic operation counts and an
accuracy-vs-size plot, no wall-clock or hardware performance numbers at
all) nor any other publication reports this accuracy/performance
trade-off** — this is the concrete, checkable, non-obvious number Part
L.4 set out to produce, and it now exists.

## M.6 What's still open here

- Not yet run on `glogin.cs.vt.edu` or native Windows/MSVC — only the
  dev machine (via WSL2, since the native-Windows MinGW toolchain here
  lacks a working OpenMP/pthread link path; see
  `revalidation/dumas2024/README.md`). Cross-machine confirmation that
  the ~20% performance-cost figure isn't dev-machine-specific is a
  natural next data point, using the same harness already proven
  portable in Parts G/J.1.
- M.4's accuracy comparison hasn't yet been run against Part I/J.8's
  specific *row-scaled, correlated* ill-conditioning construction (the
  one that produced the >1400% catastrophic error) — whether the Dumas
  variant's improved accuracy meaningfully narrows that specific
  catastrophic gap, or is irrelevant to it, is a real open question the
  current 5-distribution sweep doesn't answer.
- Not yet run at the "small-matrix/rank-k-update" regime Huang et al.
  2016 argues classic Strassen-with-fused-packing is practical at — the
  Dumas variant's higher per-level combine cost (M.5) might make that
  regime a worse fit for it specifically, which would itself be a
  useful, checkable claim.
- Bootstrap CIs / a proper significance test (per Part L.3.5) haven't
  been applied to the M.5 numbers yet — the 5-rep best-of protocol used
  here is faster to run but weaker evidence than the reps=20-30 +
  bootstrap-CI protocol Part L.3 recommends adopting project-wide before
  writeup.

---

# Part N — First real AVX-512 execution (Tiger Lake, WSL2) under the n=30 protocol

**Date:** 2026-10-13. **Data:** `revalidation/benchmarks/2026-10-13_ceburu03-11th-Gen-i5-1145G7-avx512-n30-focused/`
(`n30_samples.csv` raw, `n30_summary.csv`, `n30_comparison.csv`, `metadata.txt`). **Protocol:** `PROTOCOL.md`.

**Machine:** 11th Gen Intel i5-1145G7 (4C/8T, AVX-512 F/DQ/BW/VL/VNNI/IFMA...), 16 GB, WSL2 Ubuntu 26.04,
gcc 15.2.0, `-O3 -march=skylake-avx512 -fopenmp` (AVX2 baseline: `-mavx2 -mfma`). Threads unpinned.
This corrects the Part A / L.2 statement that no AVX-512 hardware was reachable (for this workstream's
later work; the earlier parts' data are unchanged).

## N.1 A real bug found by running it
The audited `kernel.c` crashed (SIGSEGV) for any N not a multiple of 8 -- aligned 512-bit loads/stores on the
C tile at `C + col*ldC`. Never exercised before because benchmark sizes were powers of two and no AVX-512
machine had run it. Fixed with `loadu/storeu`; 34 correctness cases then pass.
Evidence + fix: `revalidation/correctness/2026-10-13_avx512_kernel_unaligned_C_tile.md`.

## N.2 Results (n=30 per cell; mean GFLOP/s, bootstrap 95% CI; focused grid N in {1024,2048}, T in {4,8})
| algo | N | T | AVX2 | AVX-512 | AVX-512/AVX2 (95% CI) | Holm p |
|---|---|---|---|---|---|---|
| blocked | 1024 | 4 | 41.5 | 79.9 | 1.93x [1.81, 2.06] | 0.0008 |
| blocked | 1024 | 8 | 44.6 | 110.7 | 2.48x [2.32, 2.65] | 0.0008 |
| blocked | 2048 | 4 | 47.9 | 84.1 | 1.76x [1.71, 1.81] | 0.0008 |
| blocked | 2048 | 8 | 58.5 | 98.3 | 1.68x [1.63, 1.73] | 0.0008 |
| strassen | 1024 | 4 | 33.9 | 40.3 | 1.19x [1.12, 1.26] | 0.0008 |
| strassen | 1024 | 8 | 36.5 | 41.2 | 1.13x [1.07, 1.19] | 0.0008 |
| strassen | 2048 | 4 | 41.7 | 50.7 | 1.21x [1.18, 1.24] | 0.0008 |
| strassen | 2048 | 8 | 44.4 | 48.6 | 1.09x [1.07, 1.12] | 0.0008 |

All correctness checks passed in every launch (relerr <= 1e-8 gate).

## N.3 Reading it, and the limits
- AVX-512 beats AVX2 significantly in all 8 cells (p_holm = 0.0008, the floor for 10,000 permutations). Blocked DGEMM
  gains 1.7-2.5x; the Strassen path gains only 1.1-1.2x, so the AVX-512 kernel's advantage is largely absorbed by
  Strassen's memory-bound add/sub phases.
- Absolute numbers are low versus peak: this is a 4-core laptop part (one 512-bit FMA unit/core, likely
  downclocking), unpinned threads on WSL2, and the audited blocking (MC/NC/KC sized for 32 threads, static buffers).
  **Do not compare absolute GFLOP/s with the audited paper's dual Xeon Gold 5218.**
- 12 of 16 groups have CV > 5% (up to 16%): laptop thermals/WSL2 scheduling. CIs reflect that; they are valid
  within this session only. A different day or plugged-in/pinned re-run would be needed to claim stability.
- Scope: only N in {1024, 2048} and T in {4, 8} so far; the full grid (N=256..4096, T=1,2,4,8) is `run_avx512.sh` with
  default arguments (~5 h). The paper's Strassen-vs-blocked crossover claims are not tested here.

## N.4 Full grid and portable-path results (2026-10-19) — supersedes the focused table above

**Setup:** High-performance power plan, AC power, same machine/toolchain as N.1-N.3. 1,200 launches
(N=256..4096 x T=1,2,4,8 x 30 x 2 ISAs), AVX-512/AVX2 interleaved. Data:
`revalidation/benchmarks/2026-10-19_ceburu03-11th-Gen-i5-1145G7-avx512-n30-full/` (`n30_samples.csv` raw,
`n30_summary.csv`, `n30_comparison.csv`). Correctness passed in every launch.

- All 40 AVX-512-vs-AVX2 comparisons significant (Holm p = 0.004, floor for 10,000 permutations x 40).
- **Blocked DGEMM:** 1.85-2.8x for N >= 512 (N=4096: 1.97-2.12x, 107.1 vs 53.3 GFLOP/s at T=8, ratio CI [1.99, 2.03]);
  up to 5.2x at N=256, T=8 (AVX2 port dominated by threading overhead there).
- **Strassen path:** only 1.11-1.73x (shrinks with threads; 1.11x at N>=2048, T=8).
- **Noise:** 38 of 80 groups CV > 5% (max ~30%), concentrated at N <= 1024 with 4-8 threads; all N=4096 groups < 3.5%.
- **Portable AVX2 path (`reproduce.sh --full`, n=30, T=8):** DGEMM N=2048 93.9 [91.1, 97.3]; classic Strassen N=1024/depth 3
  31.4 [31.3, 31.5]; Dumas 2024 25.4 [25.0, 25.6] (cost 19.3% vs classic Strassen; dev machine: 20.7% at N=2048);
  asm `dgemm_base` 83.2 [82.9, 83.5], `strassen_1level` 32.1 [32.0, 32.2]. All correctness checks passed 30/30.
  Data: `results/reproduce_20261019_130645/` (gitignored; copy to a tracked folder if it must be archived).
- **Caveat worth stating in the paper:** the project's own tuned AVX2 DGEMM (93.9) nearly matches the audited AVX-512
  kernel (102.7) at N=2048, T=8, and beats the audited code's AVX2 port (53.7): tuning matters about as much as vector
  width. Tiger Lake fuses its two 256-bit FMA ports for 512-bit ops (my understanding; peak not measured here), so
  AVX-512's peak-FLOP advantage over AVX2 is small on this chip.
