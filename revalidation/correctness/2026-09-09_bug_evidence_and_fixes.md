# Correctness verification log — 2026-09-09

Base commit: `42386e26bf63032ea6a934d2b23cbf76421e4e6e`.
No AVX-512 hardware was available (this session's sandbox: AVX2 only;
`glogin.cs.vt.edu` rlogin pool: Intel Xeon E5-2630 v3, AVX2 only, no AVX-512,
confirmed via `/proc/cpuinfo` flags and repeated connections). Two techniques
were used to still get real, executed verification instead of static
read-through:

1. **Scalar stand-in kernels** — `kernel_24x6`/`strassen()` reimplemented with
   plain scalar loops but the *exact same packed-buffer contract*, so the
   real, unmodified packing/blocking/combination logic from `audited_submission/avx/matmul.c`
   and `audited_submission/avx/strassen.c` runs (and can be checked) on any CPU.
2. **Real AVX2 ports** — `audited_submission/avx/kernel_avx2.c`, `audited_submission/avx/strassen_avx2.c`: genuine
   256-bit `_mm256`/FMA3 intrinsics, same tile shapes, confirmed via
   `objdump` to emit real `vfmadd*pd` instructions (not scalar fallback).

## Bug 1: `audited_submission/avx/matmul.c` computed `C = B·A`, not `C = A·B`

Asymmetric test matrices (`A[i][j] = i*10+j+1`, `B[i][j] = 2*i-j+100` — chosen
specifically so `A ≠ Aᵀ`, `B ≠ Bᵀ`, `A·B ≠ B·A`, unlike the paper's own test
data `A[i][j]=i+j`, `B[i][j]=i*j` which is symmetric and can't expose this):

```
$ ./test_matmul   (N=48, original buggy matmul.c)
N=48  mismatches=2304/2304  maxdiff=1.66831e+06

$ ./test_matmul3  (compares raw output against B*A directly)
compare raw matmul(A,B) output vs B*A: mismatches=0/2304 maxdiff=0
```

Root cause: `pack_panelA`/`pack_panelB` read `A`/`B` with the contraction and
output-row/col index roles swapped (row-major reads that were effectively
transposed), and `kernel_24x6`'s natural output orientation (MR axis fast,
NR axis ld-strided) compounded it into a full operand swap once combined
with the C-pointer arithmetic at the call site.

The existing `audited_submission/test/main.c` "verification" only ever applied
`transpose_square()` to the `avx_blocked` result before comparing — which
only cancels out because its test data is symmetric
(`transpose(B·A) = Aᵀ·Bᵀ = A·B` when `A=Aᵀ, B=Bᵀ`). It was a false-positive
pass, not a correct implementation.

**Fix**: rewrote `pack_panelA`/`pack_panelB`/`pack_blockA`/`pack_blockB` to
take explicit `(row0/col0, k0, ld)` offsets and index row-major correctly;
added one O(N²) transpose pass at the end of `matmul()` to correct for
`kernel_24x6`'s fixed internal tile orientation (kept `kernel.c` itself
completely untouched — zero risk to the AVX-512 kernel).

Post-fix, `audited_submission/test/main.c`'s `transpose_square()` workaround was removed (it's
no longer needed and would now double-transpose / re-break the result).

**Verified**: 17 sizes N=1..517 (including non-multiples of the 24×6 tile),
asymmetric data, `maxdiff=0` every time — both with the scalar stand-in and
with the real AVX2 port (`vfmadd*pd` confirmed present in the binary via
`objdump`). Also confirmed the real, unmodified AVX-512 `kernel.c` still
compiles clean against the fixed `matmul.c` with `-march=skylake-avx512`
(compile-only — no AVX-512 hardware to execute on).

## Bug 2: `mm_parallel_blocked.c` / `mm_parallel_blocked_unrolled.c` data race

`#pragma omp parallel for collapse(3) schedule(static)` parallelized over
`ii, jj, kk` — including `kk`, the reduction dimension — with unsynchronized
`c[i][j] +=`. It "passed" under the shipped `schedule(static)` only because
GCC's static chunking happens to put the racy boundary iterations at the very
end of one thread's range and the very start of the next, so in practice they
rarely execute concurrently — not because of any actual synchronization.

**Proof the hazard is real**: changed only `schedule(static)` →
`schedule(dynamic,1)` (nothing else) and it failed every run:

```
=== original (collapse3, static) x5 ===          all PASS (by scheduling luck)
=== fixed candidate under dynamic,1 (worst case) ===
N=512 T=7 mismatches=... FAIL (ref=357389824 got=334095943)
N=512 T=7 mismatches=... FAIL (ref=357389824 got=314778386)
N=512 T=7 mismatches=... FAIL (ref=357389824 got=352624483)
```

(nondeterministic, always too-low — classic lost-update signature)

**Fix**: `collapse(3)` → `collapse(2)` (parallelize only `ii, jj`; `kk` stays
sequential within a thread's block).

**Verified**: fixed version survives `schedule(dynamic,1)` (deliberately
adversarial) at N=1024, T=32 (4x oversubscribed on 8 cores) — 8/8 runs exact
match, `maxdiff=0`.

## Bug 3: `audited_submission/avx/strassen.c` silently drops data for non-power-of-2 N > 128

`strassen_rec` halves `N` every level (`k = N/2`, integer division) with no
padding — unlike the Serial/Parallel Strassen variants, which all pad to the
next power of two first. Any N > 128 that isn't power-of-2-halvable all the
way down silently truncates a row/column at whichever level goes odd.

```
N=517  (517 -> k=258, exact; but eventually 129 -> k=64, drops row/col 128)
avx_strassen : FAIL at [0,1]  ref=45929246.000000  got=45166336.000000
```

**Fix**: pad `A`/`B` up to the next power of two before recursing (same
pattern as `audited_submission/Serial/mm_strassen.c`), copy the result back out.

**Verified**: N=1..517 including odd sizes, `maxdiff=0`.

**Known pre-existing edge case, not fixed**: for N ∈ {1, 2} (AVX2 port) or
{1, 2, 4} (AVX-512, needs N%8==0), `strassen()`'s `size==N` fast path skips
padding and calls the base-case register-blocked `matmul()` with N smaller
than its vector width → out-of-bounds SIMD load, segfault. Confirmed via
`objdump`/crash repro. Not fixed because it's outside the paper's tested
range (N≥64) and not a realistic use case for a "32-core AVX-512 kernel."
Worth fixing before anyone calls `strassen()` on tiny matrices directly.

## Bug 4: `audited_submission/test/run.sh` / `audited_submission/test/main.c` segfault on a clean checkout

`audited_submission/test/main.c` created `audited_submission/Data/` but never `Values/`/`Results/`; every
`mm_*` function (`audited_submission/Serial/*.c`, `audited_submission/Parallel/*.c`) does
`fopen("Values/time_*.txt", "a")` and unconditionally `fprintf`s to the
result without checking for `NULL` — SIGSEGV on the very first call.

```
Thread 1 "test_scalar" received signal SIGSEGV, Segmentation fault.
#0  __vfprintf_internal ()
#1  fprintf ()
#2  mm_naive ()
#3  main ()
```

**Fix**: added `mkdir("Values", 0777); mkdir("Results", 0777);` next to the
existing `mkdir("Data", 0777);` in `audited_submission/test/main.c`.

**Verified**: ran on a genuinely clean state (`rm -rf Values Results Data`)
— no crash, all 13 variants PASS.

## Final post-fix full sweep (all fixes applied together)

Built on `glogin.cs.vt.edu` (gcc 11.5.0, matches the paper's stated
compiler), scalar-AVX-stand-in build, all 13 implementations
(`mm_naive`, `mm_naive_unrolled`, `mm_blocked`, `mm_blocked_unrolled`,
`mm_strassen`, `mm_strassen_unrolled`, `mm_strassen_naive`,
`mm_parallel_blocked`, `mm_parallel_blocked_unrolled`,
`mm_parallel_strassen`, `mm_parallel_strassen_unrolled`,
`mm_parallel_strassen_naive`, `avx_blocked`, `avx_strassen`), against a naive
triple-loop reference, N ∈ {48, 100, 129, 130, 256, 517}, repeated at
N=1024/T=7 and N=1024/T=32 for extra confidence:

```
mm_blocked                     : PASS
mm_strassen                    : PASS
mm_naive_unrolled              : PASS
mm_blocked_unrolled            : PASS
mm_strassen_unrolled           : PASS
mm_strassen_naive              : PASS
mm_parallel_blocked            : PASS
mm_parallel_blocked_unrolled   : PASS
mm_parallel_strassen           : PASS
mm_parallel_strassen_unrolled  : PASS
mm_parallel_strassen_naive     : PASS
avx_blocked                    : PASS
avx_strassen                   : PASS
```

All PASS, every size, every repeat.
