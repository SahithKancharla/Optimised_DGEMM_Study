/*
 * matmul_ladder2.c -- v2 of matmul_ladder.c, extended for:
 *  - runtime-configurable KC/MC/NC (CLI args, not compile-time #define) --
 *    needed for a real blocking response-surface sweep (one parameter at a
 *    time) without recompiling per point.
 *  - rectangular M x K x N (A is MxK, B is KxN, C is MxN), not just square.
 *  - finer phase split: alloc_s (buffer allocation, separate from the
 *    packing work itself) and kernel_write_s (C accumulate-store, split
 *    out from kernel_compute_s -- the FMA-only inner loop).
 *  - optional Linux transparent-huge-page hint via madvise(MADV_HUGEPAGE)
 *    on the A/B/C operand buffers (not the small packed panels).
 *
 * Same MR=4/NR=8 broadcast-A/vector-B kernel and portability approach
 * (MSVC + gcc, schedule(runtime) + OMP_SCHEDULE env var) as v1.
 *
 * Wrapped in an include guard so it's safe for more than one
 * MATMUL_LADDER2_AS_LIBRARY includer to pull this in within the same
 * translation unit (e.g. an accuracy test that wants both
 * strassen_rebuild.c and dumas_strassen_rebuild.c together) -- a plain
 * repeat #include here previously caused hard duplicate-definition
 * compile errors (see FINDINGS.md's "double-inclusion" entry), which is
 * exactly the failure class this guard exists to prevent.
 */
#ifndef MATMUL_LADDER2_INCLUDED
#define MATMUL_LADDER2_INCLUDED

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "timer.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#define MR 4
#define NR 8

typedef struct {
    double alloc_s, pack_a_s, pack_b_s, kernel_compute_s, kernel_write_s, total_s;
} phase_times_t;

static int g_use_hugepages = 0;

static double *aligned_alloc_d(size_t n) {
#ifdef _WIN32
    return (double*)_aligned_malloc(n * sizeof(double), 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0) return NULL;
    if (g_use_hugepages) {
        madvise(p, n * sizeof(double), MADV_HUGEPAGE); /* hint only; silently
            no-ops if THP isn't available/enabled -- best-effort, not a hard
            guarantee (explicit hugetlbfs reservation needs root, unavailable
            here) */
    }
    return (double*)p;
#endif
}
static void aligned_free_d(double *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static void pack_panelA(const double *A, double *Ap, int mr, int kc, int ld, int row0, int k0) {
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) Ap[p*MR+i] = A[(size_t)(row0+i)*ld + (k0+p)];
        for (int i = mr; i < MR; i++) Ap[p*MR+i] = 0.0;
    }
}
static void pack_panelB(const double *B, double *Bp, int nr, int kc, int ld, int k0, int col0) {
    for (int p = 0; p < kc; p++) {
        for (int j = 0; j < nr; j++) Bp[p*NR+j] = B[(size_t)(k0+p)*ld + (col0+j)];
        for (int j = nr; j < NR; j++) Bp[p*NR+j] = 0.0;
    }
}

/* FMA-only: writes result into caller-supplied tmp[4][8], no C access at all. */
static void kernel_4x8_compute(const double *Ap, const double *Bp, int kc, double tmp[4][8]) {
    __m256d c00, c01, c10, c11, c20, c21, c30, c31;
    c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = _mm256_setzero_pd();
    for (int p = 0; p < kc; p++) {
        __m256d b0 = _mm256_load_pd(Bp + p*NR + 0);
        __m256d b1 = _mm256_load_pd(Bp + p*NR + 4);
        __m256d a0 = _mm256_set1_pd(Ap[p*MR+0]);
        __m256d a1 = _mm256_set1_pd(Ap[p*MR+1]);
        __m256d a2 = _mm256_set1_pd(Ap[p*MR+2]);
        __m256d a3 = _mm256_set1_pd(Ap[p*MR+3]);
        c00 = _mm256_fmadd_pd(a0, b0, c00); c01 = _mm256_fmadd_pd(a0, b1, c01);
        c10 = _mm256_fmadd_pd(a1, b0, c10); c11 = _mm256_fmadd_pd(a1, b1, c11);
        c20 = _mm256_fmadd_pd(a2, b0, c20); c21 = _mm256_fmadd_pd(a2, b1, c21);
        c30 = _mm256_fmadd_pd(a3, b0, c30); c31 = _mm256_fmadd_pd(a3, b1, c31);
    }
    _mm256_storeu_pd(tmp[0]+0, c00); _mm256_storeu_pd(tmp[0]+4, c01);
    _mm256_storeu_pd(tmp[1]+0, c10); _mm256_storeu_pd(tmp[1]+4, c11);
    _mm256_storeu_pd(tmp[2]+0, c20); _mm256_storeu_pd(tmp[2]+4, c21);
    _mm256_storeu_pd(tmp[3]+0, c30); _mm256_storeu_pd(tmp[3]+4, c31);
}
static void kernel_4x8_write(double *C, int mr, int nr, int ldC, double tmp[4][8]) {
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++)
            C[(size_t)i*ldC+j] += tmp[i][j];
}

/* Rectangular M x K x N: A is MxK, B is KxN, C is MxN, all row-major.
   KC/MC/NC now runtime parameters. */
static void matmul_full(const double *A, const double *B, double *C,
                         int M, int K, int N, int KC, int MC, int NC,
                         phase_times_t *pt) {
    memset(C, 0, (size_t)M*N*sizeof(double));
    pt->alloc_s = pt->pack_a_s = pt->pack_b_s = pt->kernel_compute_s = pt->kernel_write_s = 0.0;
    double t_total0 = ladder_now();

    int nblocks_j = (N + NC - 1) / NC;
    int nblocks_p = (K + KC - 1) / KC;
    int nblocks_i = (M + MC - 1) / MC;

    for (int jb = 0; jb < nblocks_j; jb++) {
        int j0 = jb*NC, nc = (N-j0 < NC ? N-j0 : NC);
        for (int pb = 0; pb < nblocks_p; pb++) {
            int p0 = pb*KC, kc = (K-p0 < KC ? K-p0 : KC);

            double ta0 = ladder_now();
            double *Bp_full = aligned_alloc_d((size_t)((nc+NR-1)/NR)*NR*kc);
            pt->alloc_s += ladder_now() - ta0;

            double tb0 = ladder_now();
            {
                int jr;
                int n_jr_here = (nc + NR - 1) / NR;
                #pragma omp parallel for
                for (jr = 0; jr < n_jr_here; jr++) {
                    int j = jr*NR;
                    int nr = (nc-j < NR ? nc-j : NR);
                    pack_panelB(B, Bp_full + (size_t)jr*NR*kc, nr, kc, N, p0, j0+j);
                }
            }
            pt->pack_b_s += ladder_now() - tb0;

            for (int ib = 0; ib < nblocks_i; ib++) {
                int i0 = ib*MC, mc = (M-i0 < MC ? M-i0 : MC);

                double taa0 = ladder_now();
                double *Ap_full = aligned_alloc_d((size_t)((mc+MR-1)/MR)*MR*kc);
                pt->alloc_s += ladder_now() - taa0;

                double tap0 = ladder_now();
                {
                    int ir;
                    int n_ir_here = (mc + MR - 1) / MR;
                    #pragma omp parallel for
                    for (ir = 0; ir < n_ir_here; ir++) {
                        int i = ir*MR;
                        int mr = (mc-i < MR ? mc-i : MR);
                        pack_panelA(A, Ap_full + (size_t)ir*MR*kc, mr, kc, K, i0+i, p0);
                    }
                }
                pt->pack_a_s += ladder_now() - tap0;

                int n_ir_here = (mc + MR - 1) / MR;
                int n_jr_here = (nc + NR - 1) / NR;
                int ntiles = n_ir_here * n_jr_here;
                int t;
                double tkc_sum = 0.0, tkw_sum = 0.0;
                #pragma omp parallel for schedule(runtime) reduction(+:tkc_sum,tkw_sum)
                for (t = 0; t < ntiles; t++) {
                    int ir = t / n_jr_here, jr = t % n_jr_here;
                    int i = ir*MR, j = jr*NR;
                    int mr = (mc-i < MR ? mc-i : MR);
                    int nr = (nc-j < NR ? nc-j : NR);
                    double tmp[4][8];
                    double tc0 = ladder_now();
                    kernel_4x8_compute(Ap_full + (size_t)ir*MR*kc, Bp_full + (size_t)jr*NR*kc, kc, tmp);
                    double tc1 = ladder_now();
                    kernel_4x8_write(&C[(size_t)(i0+i)*N + (j0+j)], mr, nr, N, tmp);
                    double tc2 = ladder_now();
                    tkc_sum += tc1 - tc0;
                    tkw_sum += tc2 - tc1;
                }
                /* sum across threads -- an aggregate CPU-time-like measure,
                   not wall-clock (threads run concurrently); divide by
                   thread count in analysis if a wall-clock-comparable
                   number is wanted */
                pt->kernel_compute_s += tkc_sum;
                pt->kernel_write_s += tkw_sum;
                aligned_free_d(Ap_full);
            }
            aligned_free_d(Bp_full);
        }
    }
    pt->total_s = ladder_now() - t_total0;
}

static double reference_check(const double *A, const double *B, const double *C, int M, int K, int N) {
    double maxerr = 0.0, maxref = 0.0;
    int stepM = M > 64 ? M/32 : 1, stepN = N > 64 ? N/32 : 1;
    for (int i = 0; i < M; i += stepM) {
        for (int j = 0; j < N; j += stepN) {
            long double s = 0.0L;
            for (int k = 0; k < K; k++) s += (long double)A[(size_t)i*K+k] * (long double)B[(size_t)k*N+j];
            double ref = (double)s;
            double got = C[(size_t)i*N+j];
            double e = fabs(got - ref);
            if (e > maxerr) maxerr = e;
            if (fabs(ref) > maxref) maxref = fabs(ref);
        }
    }
    return maxref > 0 ? maxerr/maxref : maxerr;
}

#ifndef MATMUL_LADDER2_AS_LIBRARY
int main(int argc, char **argv) {
    /* usage: matmul_ladder2 M K N threads schedule chunk reps KC MC NC [hugepages] */
    if (argc < 11) {
        fprintf(stderr, "usage: %s <M> <K> <N> <threads> <static|dynamic|guided> <chunk> <reps> <KC> <MC> <NC> [hugepages:0|1]\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    int T = atoi(argv[4]);
    const char *sched_name = argv[5];
    int chunk = atoi(argv[6]);
    int reps = atoi(argv[7]);
    int KC = atoi(argv[8]), MC = atoi(argv[9]), NC = atoi(argv[10]);
    g_use_hugepages = (argc > 11) ? atoi(argv[11]) : 0;

    omp_set_num_threads(T);
    {
        char envval[64];
        snprintf(envval, sizeof(envval), "%s,%d", sched_name, chunk);
#ifdef _WIN32
        _putenv_s("OMP_SCHEDULE", envval);
#else
        setenv("OMP_SCHEDULE", envval, 1);
#endif
    }

    double *A = aligned_alloc_d((size_t)M*K);
    double *B = aligned_alloc_d((size_t)K*N);
    double *C = aligned_alloc_d((size_t)M*N);
    for (size_t i = 0; i < (size_t)M*K; i++) A[i] = (double)((i%13)-6);
    for (size_t i = 0; i < (size_t)K*N; i++) B[i] = (double)((i%11)-5);

    phase_times_t best_pt = {0,0,0,0,0,0};
    double best_total = 1e300;
    matmul_full(A, B, C, M, K, N, KC, MC, NC, &best_pt); /* warmup */
    for (int r = 0; r < reps; r++) {
        phase_times_t pt;
        matmul_full(A, B, C, M, K, N, KC, MC, NC, &pt);
        if (pt.total_s < best_total) { best_total = pt.total_s; best_pt = pt; }
    }
    double relerr = reference_check(A, B, C, M, K, N);
    double gflops = 2.0*(double)M*K*N / best_total / 1e9;

    /* CSV: M,K,N,T,schedule,chunk,reps,KC,MC,NC,hugepages,alloc_s,pack_a_s,pack_b_s,kernel_compute_s,kernel_write_s,total_s,gflops,relerr */
    printf("ladder2,%d,%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3e\n",
           M, K, N, T, sched_name, chunk, reps, KC, MC, NC, g_use_hugepages,
           best_pt.alloc_s, best_pt.pack_a_s, best_pt.pack_b_s,
           best_pt.kernel_compute_s, best_pt.kernel_write_s, best_total, gflops, relerr);

    aligned_free_d(A); aligned_free_d(B); aligned_free_d(C);
    return 0;
}
#endif /* MATMUL_LADDER2_AS_LIBRARY */

#endif /* MATMUL_LADDER2_INCLUDED */
