/* Production packed multilevel AVX2 DGEMM. Microkernel shape MR=4 x NR=8,
 * chosen empirically from a 9-shape sweep (avx2/sweep/) -- 4x8 led at both
 * 1 and 16 threads (53.5 and 298.4 GFLOP/s vs. the runner-up's 44.6 and
 * 251.8 at N=1536/2048 respectively). MRV = MR/4 = number of YMM (256-bit,
 * 4 doubles) registers per A column; accumulator registers required =
 * MRV*NR = 1*8 = 8, leaving ample headroom in the 16-register YMM file.
 * KC/MC_TARGET/NC_TARGET are overridable via -D for the staged blocking
 * tuning sweep (see avx2/tune_blocking.sh). */
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../bench/bench_common.h"
#include "../baselines/driver_common.h"

#define MR 4
#define NR 8
#define MRV (4/4)

#ifndef KC
#define KC 384
#endif
/* Fixed targets independent of MR/NR, rounded up to a multiple of MR/NR so
 * the sweep isolates microkernel shape rather than confounding it with a
 * shape-dependent macro-block size. */
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
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) dst[i] = A[(size_t)p * lda + i];
        for (int i = mr; i < MR; i++) dst[i] = 0.0;
        dst += MR;
    }
}

static void pack_panelB(const double *B, double *dst, int nr, int kc, int ldb) {
    for (int p = 0; p < kc; p++) {
        for (int j = 0; j < nr; j++) dst[j] = B[(size_t)j * ldb + p];
        for (int j = nr; j < NR; j++) dst[j] = 0.0;
        dst += NR;
    }
}

static void pack_blockA(const double *A, double *packA, int mc, int kc, int lda) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < mc; i += MR) {
        int mr = (mc - i < MR) ? mc - i : MR;
        pack_panelA(A + i, packA + (size_t)(i / MR) * MR * kc, mr, kc, lda);
    }
}

static void pack_blockB(const double *B, double *packB, int nc, int kc, int ldb) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < nc; j += NR) {
        int nr = (nc - j < NR) ? nc - j : NR;
        pack_panelB(B + (size_t)j * ldb, packB + (size_t)(j / NR) * NR * kc, nr, kc, ldb);
    }
}

static void kernel_MRxNR(const double *A, const double *B, double *C,
                         int mr, int nr, int kc, int ldC) {
    if (mr == MR && nr == NR) {
        __m256d acc[MRV][NR];
        for (int m = 0; m < MRV; m++)
            for (int n = 0; n < NR; n++)
                acc[m][n] = _mm256_setzero_pd();

        for (int p = 0; p < kc; p++) {
            __m256d a[MRV];
            for (int m = 0; m < MRV; m++)
                a[m] = _mm256_load_pd(A + (size_t)p * MR + m * 4);
            for (int n = 0; n < NR; n++) {
                __m256d b = _mm256_broadcast_sd(B + (size_t)p * NR + n);
                for (int m = 0; m < MRV; m++)
                    acc[m][n] = _mm256_fmadd_pd(a[m], b, acc[m][n]);
            }
        }
        for (int n = 0; n < NR; n++) {
            for (int m = 0; m < MRV; m++) {
                double *Cp = C + (size_t)n * ldC + m * 4;
                __m256d c = _mm256_loadu_pd(Cp);
                c = _mm256_add_pd(c, acc[m][n]);
                _mm256_storeu_pd(Cp, c);
            }
        }
        return;
    }
    /* scalar fallback for boundary tiles (mc/nc not multiples of MR/NR) */
    for (int n = 0; n < nr; n++) {
        for (int i = 0; i < mr; i++) {
            double s = 0.0;
            for (int p = 0; p < kc; p++) s += A[(size_t)p * MR + i] * B[(size_t)p * NR + n];
            C[(size_t)n * ldC + i] += s;
        }
    }
}

static void avx2_matmul(const double *A, const double *B, double *C, int N) {
    memset(C, 0, (size_t)N * N * sizeof(double));
    int mc_cap = MC < N ? MC : N;
    int nc_cap = NC < N ? NC : N;
    if (!g_packA) g_packA = bench_alloc((size_t)mc_cap * KC + MR * KC);
    if (!g_packB) g_packB = bench_alloc((size_t)nc_cap * KC + NR * KC);

    for (int jc = 0; jc < N; jc += NC) {
        int nc = (N - jc < NC) ? N - jc : NC;
        for (int pc = 0; pc < N; pc += KC) {
            int kc = (N - pc < KC) ? N - pc : KC;
            pack_blockB(B + (size_t)jc * N + pc, g_packB, nc, kc, N);
            for (int ic = 0; ic < N; ic += MC) {
                int mc = (N - ic < MC) ? N - ic : MC;
                pack_blockA(A + (size_t)pc * N + ic, g_packA, mc, kc, N);

#pragma omp parallel for collapse(2) schedule(static)
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
    snprintf(variant, sizeof(variant), "avx2_4x8");
    bench_print_csv_row(stdout, variant, a.N, omp_get_max_threads(), a.label, a.reps, r, cmp);

    bench_free(A); bench_free(B); bench_free(C);
    return 0;
}
