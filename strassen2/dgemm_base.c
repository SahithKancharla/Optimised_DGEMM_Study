/* Same microkernel (kernel_6x8_asm) and transpose-based packing as
 * avx2/asm_kernel/matmul_asm_v3_transpack.c, refactored into a reusable
 * dgemm_base(A,ldA,B,ldB,C,ldC,n) library call with independent leading
 * dimensions -- see dgemm_base.h for why. Blocking fixed at this dev
 * machine's proven-good KC=1024/MC=576/NC=1024 (see FINDINGS.md B.8/B.9);
 * Strassen's recursive sub-problems are smaller than the top-level N, and
 * the packing/blocking loops already clamp MC/NC to min(MC,n)/min(NC,n),
 * so smaller n is handled correctly, just with less blocking benefit at
 * very small n (expected and fine -- that's what the base-case cutoff is
 * for). */
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include "dgemm_base.h"
#include "../bench/bench_common.h"

extern void kernel_6x8_asm(const double *A, const double *B, double *C,
                            long kc, long ldC_bytes);

#define MR 6
#define NR 8
#define KC 1024
#define MC 576
#define NC 1024

/* No static/cached pack buffers here (unlike the benchmark-driver versions
 * in avx2/asm_kernel/, which cache across many repeated calls in a timing
 * loop): dgemm_base is called from Strassen's OpenMP tasks (strassen.c),
 * concurrently and/or nested inside other parallel regions. A shared
 * static buffer would race between concurrent tasks; threadprivate breaks
 * under nested parallelism (a nested team's worker threads get their own
 * uninitialized threadprivate copies -- this was tried and segfaulted).
 * Fresh local buffers, freed at the end of each call, are the safe choice
 * here -- Strassen calls this O(7) or O(49) times per level, not the
 * millions of tiny allocations the original avx/strassen.c made, so the
 * extra alloc/free per call is negligible. */

static void pack_panelA(const double *A, double *dst, int mr, int kc, int lda) {
    if (mr == MR) {
        for (int p = 0; p < kc; p++) {
            const double *src = A + (size_t)p * lda;
            _mm256_storeu_pd(dst, _mm256_loadu_pd(src));
            _mm_storeu_pd(dst + 4, _mm_loadu_pd(src + 4));
            dst += MR;
        }
        return;
    }
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) dst[i] = A[(size_t)p * lda + i];
        for (int i = mr; i < MR; i++) dst[i] = 0.0;
        dst += MR;
    }
}

static inline void transpose4x4_pd(__m256d r0, __m256d r1, __m256d r2, __m256d r3,
                                    __m256d *c0, __m256d *c1, __m256d *c2, __m256d *c3) {
    __m256d t0 = _mm256_unpacklo_pd(r0, r1);
    __m256d t1 = _mm256_unpackhi_pd(r0, r1);
    __m256d t2 = _mm256_unpacklo_pd(r2, r3);
    __m256d t3 = _mm256_unpackhi_pd(r2, r3);
    *c0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    *c1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    *c2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    *c3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}

static void pack_panelB(const double *B, double *dst, int nr, int kc, int ldb) {
    if (nr == NR) {
        const double *col0 = B, *col1 = B + ldb, *col2 = B + 2 * (size_t)ldb, *col3 = B + 3 * (size_t)ldb;
        const double *col4 = B + 4 * (size_t)ldb, *col5 = B + 5 * (size_t)ldb, *col6 = B + 6 * (size_t)ldb, *col7 = B + 7 * (size_t)ldb;
        int p = 0;
        for (; p + 3 < kc; p += 4) {
            __m256d r0, r1, r2, r3, c0, c1, c2, c3;
            r0 = _mm256_loadu_pd(col0 + p); r1 = _mm256_loadu_pd(col1 + p);
            r2 = _mm256_loadu_pd(col2 + p); r3 = _mm256_loadu_pd(col3 + p);
            transpose4x4_pd(r0, r1, r2, r3, &c0, &c1, &c2, &c3);
            _mm256_storeu_pd(dst + 0 * NR, c0);
            _mm256_storeu_pd(dst + 1 * NR, c1);
            _mm256_storeu_pd(dst + 2 * NR, c2);
            _mm256_storeu_pd(dst + 3 * NR, c3);

            r0 = _mm256_loadu_pd(col4 + p); r1 = _mm256_loadu_pd(col5 + p);
            r2 = _mm256_loadu_pd(col6 + p); r3 = _mm256_loadu_pd(col7 + p);
            transpose4x4_pd(r0, r1, r2, r3, &c0, &c1, &c2, &c3);
            _mm256_storeu_pd(dst + 0 * NR + 4, c0);
            _mm256_storeu_pd(dst + 1 * NR + 4, c1);
            _mm256_storeu_pd(dst + 2 * NR + 4, c2);
            _mm256_storeu_pd(dst + 3 * NR + 4, c3);
            dst += 4 * NR;
        }
        for (; p < kc; p++) {
            for (int j = 0; j < NR; j++) dst[j] = B[(size_t)j * ldb + p];
            dst += NR;
        }
        return;
    }
    for (int p = 0; p < kc; p++) {
        for (int j = 0; j < nr; j++) dst[j] = B[(size_t)j * ldb + p];
        for (int j = nr; j < NR; j++) dst[j] = 0.0;
        dst += NR;
    }
}

static void pack_blockA(const double *A, double *packA, int mc, int kc, int lda) {
#pragma omp for schedule(static)
    for (int i = 0; i < mc; i += MR) {
        int mr = (mc - i < MR) ? mc - i : MR;
        pack_panelA(A + i, packA + (size_t)(i / MR) * MR * kc, mr, kc, lda);
    }
}

static void pack_blockB(const double *B, double *packB, int nc, int kc, int ldb) {
#pragma omp for schedule(static)
    for (int j = 0; j < nc; j += NR) {
        int nr = (nc - j < NR) ? nc - j : NR;
        pack_panelB(B + (size_t)j * ldb, packB + (size_t)(j / NR) * NR * kc, nr, kc, ldb);
    }
}

static void kernel_MRxNR(const double *A, const double *B, double *C,
                         int mr, int nr, int kc, int ldC) {
    if (mr == MR && nr == NR) {
        kernel_6x8_asm(A, B, C, (long)kc, (long)ldC * (long)sizeof(double));
        return;
    }
    for (int n = 0; n < nr; n++) {
        for (int i = 0; i < mr; i++) {
            double s = 0.0;
            for (int p = 0; p < kc; p++) s += A[(size_t)p * MR + i] * B[(size_t)p * NR + n];
            C[(size_t)n * ldC + i] += s;
        }
    }
}

void dgemm_base(const double *A, int ldA, const double *B, int ldB,
                 double *C, int ldC, int n) {
    int mc_cap = MC < n ? MC : n;
    int nc_cap = NC < n ? NC : n;
    size_t needA = (size_t)mc_cap * KC + MR * KC;
    size_t needB = (size_t)nc_cap * KC + NR * KC;
    double *packA = bench_alloc(needA);
    double *packB = bench_alloc(needB);

    /* Own parallel region: safe to call from a task or top-level. `single`
     * wraps only the memset; all threads then redundantly (cheaply) walk
     * the same jc/pc/ic bounds so they arrive together at pack_blockA/B's
     * and the microkernel's `omp for`, which is what actually work-shares. */
#pragma omp parallel
    {
#pragma omp single
        for (int col = 0; col < n; col++) memset(C + (size_t)col * ldC, 0, (size_t)n * sizeof(double));

        for (int jc = 0; jc < n; jc += NC) {
            int nc = (n - jc < NC) ? n - jc : NC;
            for (int pc = 0; pc < n; pc += KC) {
                int kc = (n - pc < KC) ? n - pc : KC;
                pack_blockB(B + (size_t)jc * ldB + pc, packB, nc, kc, ldB);
                for (int ic = 0; ic < n; ic += MC) {
                    int mc = (n - ic < MC) ? n - ic : MC;
                    pack_blockA(A + (size_t)pc * ldA + ic, packA, mc, kc, ldA);

#pragma omp for collapse(2) schedule(guided)
                    for (int ir = 0; ir < mc; ir += MR) {
                        for (int jr = 0; jr < nc; jr += NR) {
                            int mr = (mc - ir < MR) ? mc - ir : MR;
                            int nr = (nc - jr < NR) ? nc - jr : NR;
                            kernel_MRxNR(packA + (size_t)(ir / MR) * MR * kc,
                                         packB + (size_t)(jr / NR) * NR * kc,
                                         C + (size_t)(jc + jr) * ldC + (ic + ir),
                                         mr, nr, kc, ldC);
                        }
                    }
                }
            }
        }
    }
    bench_free(packA);
    bench_free(packB);
}
