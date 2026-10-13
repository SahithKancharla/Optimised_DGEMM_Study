/* Ablation v2: same 4x8 tile as v1 (matmul_4x8.c), but with the broadcast
 * operand SWAPPED. v1 vector-loaded A (1 load) and broadcast B eight times
 * (8 loads) = 9 loads per 8 FMAs. Disassembling v1's compiled kernel showed
 * exactly that: 1 vmovup + 8 vbroadcastsd + 8 vfmadd231pd per k-step, in a
 * non-unrolled loop. OpenBLAS's real dgemm_kernel_4x8_haswell.S does the
 * opposite -- broadcast the 4-wide operand (A), vector-load the 8-wide one
 * (B, as two 4-wide chunks) -- 4+2=6 loads per 8 FMAs. This file makes that
 * same swap: accumulators are now acc[MR][NRV] (one scalar row x one 4-wide
 * column group) instead of acc[MRV][NR]. Packed A/B layout is unchanged
 * (both already store NR/MR contiguous doubles per k-step), only the
 * kernel's load pattern changes. Write-back becomes a strided scatter (4
 * separate column stores per row) instead of a contiguous column store, but
 * that only happens once per micro-tile, not once per k-step, so it's
 * negligible next to KC-many fewer loads in the hot loop. */
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../bench/bench_common.h"
#include "../baselines/driver_common.h"

#define MR 4
#define NR 8
#define NRV (NR/4)

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
        /* Flat named accumulators, not an array -- GCC's SROA pass promoted
         * v1's [1][8] array to registers fine, but silently fell back to a
         * real stack spill for this [4][2] shape (confirmed via objdump:
         * vmovapd to (%rsp) after every FMA in the k-loop). Named scalars
         * sidestep that: acc_i_v = row i (broadcast A), column-group v
         * (vector B). 8 accumulators total, same as v1. */
        __m256d acc00 = _mm256_setzero_pd(), acc01 = _mm256_setzero_pd();
        __m256d acc10 = _mm256_setzero_pd(), acc11 = _mm256_setzero_pd();
        __m256d acc20 = _mm256_setzero_pd(), acc21 = _mm256_setzero_pd();
        __m256d acc30 = _mm256_setzero_pd(), acc31 = _mm256_setzero_pd();

        for (int p = 0; p < kc; p++) {
            __m256d b0 = _mm256_load_pd(B + (size_t)p * NR + 0);
            __m256d b1 = _mm256_load_pd(B + (size_t)p * NR + 4);
            __m256d a0 = _mm256_broadcast_sd(A + (size_t)p * MR + 0);
            acc00 = _mm256_fmadd_pd(a0, b0, acc00);
            acc01 = _mm256_fmadd_pd(a0, b1, acc01);
            __m256d a1 = _mm256_broadcast_sd(A + (size_t)p * MR + 1);
            acc10 = _mm256_fmadd_pd(a1, b0, acc10);
            acc11 = _mm256_fmadd_pd(a1, b1, acc11);
            __m256d a2 = _mm256_broadcast_sd(A + (size_t)p * MR + 2);
            acc20 = _mm256_fmadd_pd(a2, b0, acc20);
            acc21 = _mm256_fmadd_pd(a2, b1, acc21);
            __m256d a3 = _mm256_broadcast_sd(A + (size_t)p * MR + 3);
            acc30 = _mm256_fmadd_pd(a3, b0, acc30);
            acc31 = _mm256_fmadd_pd(a3, b1, acc31);
        }

        /* In-register 4x4 transpose per column-group: 4 row-scalar vectors
         * -> 4 column-contiguous vectors, matching v1's write-back. */
        {
            __m256d t0 = _mm256_unpacklo_pd(acc00, acc10);
            __m256d t1 = _mm256_unpackhi_pd(acc00, acc10);
            __m256d t2 = _mm256_unpacklo_pd(acc20, acc30);
            __m256d t3 = _mm256_unpackhi_pd(acc20, acc30);
            __m256d col0 = _mm256_permute2f128_pd(t0, t2, 0x20);
            __m256d col1 = _mm256_permute2f128_pd(t1, t3, 0x20);
            __m256d col2 = _mm256_permute2f128_pd(t0, t2, 0x31);
            __m256d col3 = _mm256_permute2f128_pd(t1, t3, 0x31);
            double *Cp = C;
            _mm256_storeu_pd(Cp + 0 * ldC, _mm256_add_pd(_mm256_loadu_pd(Cp + 0 * ldC), col0));
            _mm256_storeu_pd(Cp + 1 * ldC, _mm256_add_pd(_mm256_loadu_pd(Cp + 1 * ldC), col1));
            _mm256_storeu_pd(Cp + 2 * ldC, _mm256_add_pd(_mm256_loadu_pd(Cp + 2 * ldC), col2));
            _mm256_storeu_pd(Cp + 3 * ldC, _mm256_add_pd(_mm256_loadu_pd(Cp + 3 * ldC), col3));
        }
        {
            __m256d t0 = _mm256_unpacklo_pd(acc01, acc11);
            __m256d t1 = _mm256_unpackhi_pd(acc01, acc11);
            __m256d t2 = _mm256_unpacklo_pd(acc21, acc31);
            __m256d t3 = _mm256_unpackhi_pd(acc21, acc31);
            __m256d col0 = _mm256_permute2f128_pd(t0, t2, 0x20);
            __m256d col1 = _mm256_permute2f128_pd(t1, t3, 0x20);
            __m256d col2 = _mm256_permute2f128_pd(t0, t2, 0x31);
            __m256d col3 = _mm256_permute2f128_pd(t1, t3, 0x31);
            double *Cp = C + (size_t)4 * ldC;
            _mm256_storeu_pd(Cp + 0 * ldC, _mm256_add_pd(_mm256_loadu_pd(Cp + 0 * ldC), col0));
            _mm256_storeu_pd(Cp + 1 * ldC, _mm256_add_pd(_mm256_loadu_pd(Cp + 1 * ldC), col1));
            _mm256_storeu_pd(Cp + 2 * ldC, _mm256_add_pd(_mm256_loadu_pd(Cp + 2 * ldC), col2));
            _mm256_storeu_pd(Cp + 3 * ldC, _mm256_add_pd(_mm256_loadu_pd(Cp + 3 * ldC), col3));
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
    snprintf(variant, sizeof(variant), "avx2_4x8_v2swap");
    bench_print_csv_row(stdout, variant, a.N, omp_get_max_threads(), a.label, a.reps, r, cmp);

    bench_free(A); bench_free(B); bench_free(C);
    return 0;
}
