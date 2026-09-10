/* Ablation v3: builds on v2 (operand swap: broadcast A, vector-load B,
 * flat named accumulators -- 334.0 GFLOP/s vs v1's 317.1 at N=2048/16
 * threads). Adds what OpenBLAS's real dgemm_kernel_4x8_haswell.S does that
 * a plain compiled loop doesn't: the k-loop is manually unrolled by 2 (so
 * loop-branch/compare overhead is paid once per 2 k-steps instead of once
 * per step) and explicit prefetch hints bring the next iteration's A/B
 * panel data into L1 ahead of use, hiding load latency instead of just
 * hoping the hardware prefetcher keeps up with KC=512-deep panels. */
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../bench/bench_common.h"
#include "../baselines/driver_common.h"

/* MR=6 matches BLIS's actual production Haswell shape (reference/blis/
 * bli_family_haswell.h: MR_D=6, NR_D=8). MR is now the broadcast dimension
 * (unconstrained by vector width) since v2's operand swap, so unlike v1
 * it no longer needs to be a multiple of 4 -- this shape was unreachable
 * before that swap. */
#define MR 6
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
/* Orphaned worksharing (`omp for`, no `parallel`) -- these run inside the
 * single `#pragma omp parallel` that now wraps all of avx2_matmul, so the
 * thread team is created once for the whole GEMM call instead of being
 * torn down and rebuilt at every one of these call sites, every macro-block
 * iteration (see avx2_matmul below). */
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
        __m256d acc40 = _mm256_setzero_pd(), acc41 = _mm256_setzero_pd();
        __m256d acc50 = _mm256_setzero_pd(), acc51 = _mm256_setzero_pd();

        /* One k-step, parameterized by offset -- expanded twice per loop
         * iteration below (manual unroll-by-2), each expansion prefetching
         * ahead by PF_DIST steps so the load for that future step is
         * already in flight while the current step's FMAs execute. */
#define KSTEP(p)                                                       \
        do {                                                            \
            __m256d b0 = _mm256_load_pd(B + (size_t)(p) * NR + 0);      \
            __m256d b1 = _mm256_load_pd(B + (size_t)(p) * NR + 4);      \
            __m256d a0 = _mm256_broadcast_sd(A + (size_t)(p) * MR + 0); \
            acc00 = _mm256_fmadd_pd(a0, b0, acc00);                     \
            acc01 = _mm256_fmadd_pd(a0, b1, acc01);                     \
            __m256d a1 = _mm256_broadcast_sd(A + (size_t)(p) * MR + 1); \
            acc10 = _mm256_fmadd_pd(a1, b0, acc10);                     \
            acc11 = _mm256_fmadd_pd(a1, b1, acc11);                     \
            __m256d a2 = _mm256_broadcast_sd(A + (size_t)(p) * MR + 2); \
            acc20 = _mm256_fmadd_pd(a2, b0, acc20);                     \
            acc21 = _mm256_fmadd_pd(a2, b1, acc21);                     \
            __m256d a3 = _mm256_broadcast_sd(A + (size_t)(p) * MR + 3); \
            acc30 = _mm256_fmadd_pd(a3, b0, acc30);                     \
            acc31 = _mm256_fmadd_pd(a3, b1, acc31);                     \
            __m256d a4 = _mm256_broadcast_sd(A + (size_t)(p) * MR + 4); \
            acc40 = _mm256_fmadd_pd(a4, b0, acc40);                     \
            acc41 = _mm256_fmadd_pd(a4, b1, acc41);                     \
            __m256d a5 = _mm256_broadcast_sd(A + (size_t)(p) * MR + 5); \
            acc50 = _mm256_fmadd_pd(a5, b0, acc50);                     \
            acc51 = _mm256_fmadd_pd(a5, b1, acc51);                     \
        } while (0)

#define PF_DIST 12
        int p = 0;
        for (; p + 1 < kc; p += 2) {
            _mm_prefetch((const char *)(A + (size_t)(p + PF_DIST) * MR), _MM_HINT_T0);
            _mm_prefetch((const char *)(B + (size_t)(p + PF_DIST) * NR), _MM_HINT_T0);
            KSTEP(p);
            _mm_prefetch((const char *)(A + (size_t)(p + 1 + PF_DIST) * MR), _MM_HINT_T0);
            _mm_prefetch((const char *)(B + (size_t)(p + 1 + PF_DIST) * NR), _MM_HINT_T0);
            KSTEP(p + 1);
        }
        for (; p < kc; p++) KSTEP(p);
#undef PF_DIST
#undef KSTEP

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
        /* Rows 4-5 (MR=6 isn't a multiple of 4): only 2 leftover rows, not
         * worth a transpose -- plain scalar scatter, once per micro-tile. */
        {
            double t4[4], t5[4];
            _mm256_storeu_pd(t4, acc40); _mm256_storeu_pd(t5, acc50);
            for (int c = 0; c < 4; c++) { C[(size_t)c * ldC + 4] += t4[c]; C[(size_t)c * ldC + 5] += t5[c]; }
            _mm256_storeu_pd(t4, acc41); _mm256_storeu_pd(t5, acc51);
            for (int c = 0; c < 4; c++) { C[(size_t)(4 + c) * ldC + 4] += t4[c]; C[(size_t)(4 + c) * ldC + 5] += t5[c]; }
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
    int mc_cap = MC < N ? MC : N;
    int nc_cap = NC < N ? NC : N;
    if (!g_packA) g_packA = bench_alloc((size_t)mc_cap * KC + MR * KC);
    if (!g_packB) g_packB = bench_alloc((size_t)nc_cap * KC + NR * KC);

    /* Single fork/join for the entire GEMM call: the thread team is created
     * once here and reused via orphaned `#pragma omp for` worksharing at
     * every pack/microkernel call site below, instead of tearing down and
     * rebuilding the team at every one of those sites on every macro-block
     * iteration (v1-v3 opened a fresh `omp parallel for` per call, which at
     * this tuned blocking is 2x4x8=64 macro-block iterations for N=2048 --
     * 64 x 3 parallel-region entries worth of avoidable fork/join). */
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
    snprintf(variant, sizeof(variant), "avx2_6x8_v6");
    bench_print_csv_row(stdout, variant, a.N, omp_get_max_threads(), a.label, a.reps, r, cmp);

    bench_free(A); bench_free(B); bench_free(C);
    return 0;
}
