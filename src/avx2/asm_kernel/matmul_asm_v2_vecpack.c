/* v2: builds on matmul_asm.c (hand-scheduled assembly microkernel, ~342-353
 * GFLOP/s avg) by additionally vectorizing the packing routines, which were
 * plain scalar C in every prior version. Roofline check (roofline/
 * stream_triad.c: ~77-85 GB/s measured bandwidth) confirmed DGEMM at N=2048
 * is solidly compute-bound here (operational intensity ~170 FLOPs/byte vs a
 * ridge point of ~8), and the assembly-vs-intrinsics comparison showed the
 * microkernel itself wasn't the remaining bottleneck -- so packing overhead
 * is the next candidate. pack_panelA's source is contiguous (a straight
 * AVX2 load/store); pack_panelB's source is strided (column j at stride
 * ldb), so it uses AVX2 gather (_mm256_i64gather_pd) instead. matmul_asm.c
 * is kept unchanged as the pre-vectorized-packing reference point. */
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

static void pack_panelB(const double *B, double *dst, int nr, int kc, int ldb) {
    if (nr == NR) {
        /* Source is strided (column j at stride ldb) -- can't be a plain
         * vector load, so use AVX2 gather instead of 8 scalar loads. */
        __m256i idx = _mm256_setr_epi64x(0, (long long)ldb, 2LL * ldb, 3LL * ldb);
        for (int p = 0; p < kc; p++) {
            __m256d g0 = _mm256_i64gather_pd(B + p, idx, 8);
            __m256d g1 = _mm256_i64gather_pd(B + p + 4 * (size_t)ldb, idx, 8);
            _mm256_storeu_pd(dst, g0);
            _mm256_storeu_pd(dst + 4, g1);
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
    snprintf(variant, sizeof(variant), "avx2_6x8_asm_vecpack");
    bench_print_csv_row(stdout, variant, a.N, omp_get_max_threads(), a.label, a.reps, r, cmp);

    bench_free(A); bench_free(B); bench_free(C);
    return 0;
}
