/* v3: builds on matmul_asm_v2_vecpack.c (357.1 GFLOP/s avg, gather-based B
 * packing). AVX2 gather has historically mediocre throughput on Haswell-
 * class execution units (decodes into multiple sequential uops internally),
 * which is a plausible reason it might not be the best choice even though
 * it beat scalar packing. This version replaces the gather with an
 * explicit in-register 4x4 transpose -- the same technique already used
 * for the microkernel's C write-back (kernel_6x8.S) -- applied to packing
 * instead: read 4 columns of B as 4 contiguous vector loads (each column
 * IS contiguous along p, unlike a cross-column read), transpose them into
 * 4 row-vectors, store contiguously. matmul_asm_v2_vecpack.c (gather
 * version) is kept unchanged for comparison. */
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../../bench/bench_common.h"
#include "../../baselines/driver_common.h"

extern void kernel_6x8_asm(const double *A, const double *B, double *C,
                            long kc, long ldC_bytes);

#define MR 6
#define NR 8

#ifndef KC
#define KC 384
#endif
#ifndef MC_TARGET
#define MC_TARGET 384
#endif
#ifndef NC_TARGET
#define NC_TARGET 4096
#endif
#define MC (((MC_TARGET + MR - 1) / MR) * MR)
#define NC (((NC_TARGET + NR - 1) / NR) * NR)

static double *g_packA = NULL, *g_packB = NULL;

static void pack_panelA(const double *A, double *dst, int mr, int kc, int lda) {
    if (mr == MR) {
        /* Source is contiguous (mr consecutive rows of column p) -- a
         * straight vector load/store, MR=6 done as one YMM (4) + one XMM
         * (2) instead of 6 scalar loads. */
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

/* Transpose 4 columns (each a contiguous 4-double vector along p) into 4
 * row-vectors (each a contiguous 4-double vector along column-within-group)
 * -- the same unpacklo/unpackhi + permute2f128 idiom used for the C
 * write-back in kernel_6x8.S, applied here to packing instead of storing. */
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
        /* Source is strided (column j at stride ldb) but EACH column is
         * itself contiguous along p -- so read 4 columns x 4 rows as 4
         * vector loads (not a gather) and transpose in-register instead. */
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
    /* scalar fallback for boundary tiles (mc/nc not multiples of MR/NR) --
     * same as every intrinsics version; the assembly kernel only handles
     * the full MR x NR fast path. */
    for (int n = 0; n < nr; n++) {
        for (int i = 0; i < mr; i++) {
            double s = 0.0;
            for (int p = 0; p < kc; p++) s += A[(size_t)p * MR + i] * B[(size_t)p * NR + n];
            C[(size_t)n * ldC + i] += s;
        }
    }
}

static void avx2_matmul(const double *A, const double *B, double *C, int N) {
    int mc_cap = MC < N ? MC : N;
    int nc_cap = NC < N ? NC : N;
    if (!g_packA) g_packA = bench_alloc((size_t)mc_cap * KC + MR * KC);
    if (!g_packB) g_packB = bench_alloc((size_t)nc_cap * KC + NR * KC);

#pragma omp parallel
    {
#pragma omp single
        memset(C, 0, (size_t)N * N * sizeof(double));

        for (int jc = 0; jc < N; jc += NC) {
            int nc = (N - jc < NC) ? N - jc : NC;
            for (int pc = 0; pc < N; pc += KC) {
                int kc = (N - pc < KC) ? N - pc : KC;
                pack_blockB(B + (size_t)jc * N + pc, g_packB, nc, kc, N);
                for (int ic = 0; ic < N; ic += MC) {
                    int mc = (N - ic < MC) ? N - ic : MC;
                    pack_blockA(A + (size_t)pc * N + ic, g_packA, mc, kc, N);

#pragma omp for collapse(2) schedule(static)
                    for (int ir = 0; ir < mc; ir += MR) {
                        for (int jr = 0; jr < nc; jr += NR) {
                            int mr = (mc - ir < MR) ? mc - ir : MR;
                            int nr = (nc - jr < NR) ? nc - jr : NR;
                            kernel_MRxNR(g_packA + (size_t)(ir / MR) * MR * kc,
                                         g_packB + (size_t)(jr / NR) * NR * kc,
                                         C + (size_t)(jc + jr) * N + (ic + ir),
                                         mr, nr, kc, N);
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    driver_args_t a = parse_driver_args(argc, argv);

    double *A = bench_alloc((size_t)a.N * a.N);
    double *B = bench_alloc((size_t)a.N * a.N);
    double *C = bench_alloc((size_t)a.N * a.N);
    bench_init_random(A, (size_t)a.N * a.N, a.seed);
    bench_init_random(B, (size_t)a.N * a.N, a.seed + 1);

    bench_compare_t cmp = {0.0, 0.0};
    if (a.verify) {
        double *Cref = bench_alloc((size_t)a.N * a.N);
        bench_reference_gemm(A, B, Cref, a.N);
        avx2_matmul(A, B, C, a.N);
        cmp = bench_compare(C, Cref, a.N);
        bench_free(Cref);
    }

    bench_result_t r = bench_run(avx2_matmul, A, B, C, a.N, a.reps);
    char variant[32];
    snprintf(variant, sizeof(variant), "avx2_6x8_asm_transpack");
    bench_print_csv_row(stdout, variant, a.N, omp_get_max_threads(), a.label, a.reps, r, cmp);

    bench_free(A); bench_free(B); bench_free(C);
    return 0;
}
