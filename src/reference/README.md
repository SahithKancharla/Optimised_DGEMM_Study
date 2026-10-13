# Reference: OpenBLAS / BLIS Haswell DGEMM internals

Curated excerpts (not full clones) from OpenBLAS and BLIS, pulled for direct
comparison against `avx2/matmul_4x8.c`'s design decisions. Full upstream repos
were shallow-cloned locally for inspection (`~/reference-libs/OpenBLAS`,
`~/blis` in WSL) and are not part of this repo; only the specific
kernel/blocking-parameter files referenced below were copied in, with each
library's LICENSE and the exact source commit (`SOURCE_COMMIT.txt`) for
attribution/reproducibility. OpenBLAS is BSD-3-Clause; BLIS is BSD-3-Clause.

MKL has no public source -- Intel does not distribute its DGEMM kernel
implementation, so there is nothing to copy here. It's treated as a black-box
baseline throughout this study.

## Microkernel shape comparison (double precision, Haswell/AVX2 target)

| Library                  | MR | NR | MC  | KC  | NC      | Implementation |
|---------------------------|----|----|-----|-----|---------|-----------------|
| This study (avx2/matmul_4x8.c) | 4  | 8  | 256 | 512 | 1024    | C + AVX2 intrinsics |
| OpenBLAS (KERNEL.HASWELL) | 4  | 8  | 512 | 256 | dynamic | hand-written x86-64 assembly (`dgemm_kernel_4x8_haswell.S`) |
| BLIS (config/haswell)     | 6  | 8  | 72  | 256 | 4080    | hand-written AVX2 assembly (`bli_gemm_haswell_asm_d6x8.c`) |

**Our empirically-swept 4x8 winner matches OpenBLAS's shipped Haswell shape
exactly** (`param.h`: `DGEMM_DEFAULT_UNROLL_M=4`, `_N=8`) -- independent
validation of the register-pressure reasoning in the microkernel sweep
(avx2/sweep/). BLIS instead ships 6x8; its kernel source directory also still
contains superseded `old/` variants at 8x6, 4x12, and 12x4 -- the same shape
space we swept independently arrived at the same short list.

Open questions this raises for the gap between our GFLOP/s and
OpenBLAS/BLIS's (to investigate in the roofline/analysis phase):

1. **Hand-written assembly vs. compiler-generated intrinsics.** Both
   reference kernels are `.S`/inline-asm, giving explicit control over
   instruction scheduling, prefetch distance, and register allocation that
   `gcc -O3` may not replicate from our C intrinsics.
2. **MC/KC are swapped in priority.** OpenBLAS uses the larger value for MC
   (512) and smaller for KC (256); we tuned to the opposite (MC=256,
   KC=512). BLIS uses a much smaller MC (72) entirely, consistent with the
   BLIS project's published analytical cache/TLB-associativity model (Low et
   al., "Analytical Modeling Is Enough for High-Performance BLIS") rather
   than an empirical sweep -- worth trying that model's formula here
   directly rather than only staged empirical search.
3. **Packing/copy routines.** Even OpenBLAS's `KERNEL.HASWELL` selectively
   pulls in newer `*_skylakex.c` copy/beta routines rather than Haswell-era
   ones, suggesting the packing step itself (not just the microkernel) is a
   nontrivial share of the performance gap.
