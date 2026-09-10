/*
 * matmul_ladder.c -- portable (MSVC + gcc, native Windows + Linux),
 * phase-timed DGEMM for the 8-stage isolation ladder:
 *   A: microkernel only, on already-packed panels (pure compute)
 *   B: pack + microkernel, single block (N small enough for one MC/KC/NC
 *      block -- no separate code path needed, just pick N accordingly)
 *   C: full multilevel, 1 thread
 *   D: full multilevel, T threads, schedule(static)
 *   E: same as D + explicit thread-to-core pinning (Windows only)
 *   F: full multilevel, T threads, schedule(dynamic,chunk)
 *   G: full multilevel, T threads, schedule(guided)
 *   H: production baseline -- NOT this file; run driver_openblas /
 *      driver_naive / driver_blis / driver_mkl separately and compare
 *      GFLOP/s (same FLOP count regardless of layout, so directly
 *      comparable even though this driver is row-major and
 *      bench_common.h's drivers are column-major).
 *
 * Row-major throughout (C[i*N+j]), independent of avx2/bench_common.h's
 * column-major convention -- chosen for simplicity in this file, not
 * because it matters for the GFLOP/s comparison (same FLOP count).
 *
 * Kernel: MR=4, NR=8, broadcast-A / vector-B (the "operand-swapped"
 * convention from Part B.4 -- 6 loads per 8 FMAs), flat named
 * accumulators (not a 2D array -- Part B.4 found GCC silently spills a
 * [MR][NR/4] array to the stack every k-step; flat names avoid that on
 * both compilers).
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "timer.h"

#ifdef _WIN32
#include <windows.h>
#endif

#define MR 4
#define NR 8

#ifndef KC
#define KC 512
#endif
#ifndef MC
#define MC 256
#endif
#ifndef NC
#define NC 1024

#endif

typedef struct {
    double pack_a_s, pack_b_s, kernel_s, total_s;
} phase_times_t;

static double *aligned_alloc_d(size_t n) {
#ifdef _WIN32
    return (double*)_aligned_malloc(n * sizeof(double), 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0) return NULL;
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

/* pack an mr(<=MR) x kc panel of A: rows row0..row0+mr-1, cols k0..k0+kc-1,
   row-major A (ld), into Ap as [p=0..kc-1][row=0..MR-1] (padded). */
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

/* MR=4 x NR=8 microkernel: C[row][0..7] += A_panel . B_panel, kc steps.
   C is a plain mr x nr tile, row-major, leading dim ldC. Accumulate. */
static void kernel_4x8(const double *Ap, const double *Bp, double *C, int mr, int nr, int kc, int ldC) {
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
    double tmp[4][8];
    _mm256_storeu_pd(tmp[0]+0, c00); _mm256_storeu_pd(tmp[0]+4, c01);
    _mm256_storeu_pd(tmp[1]+0, c10); _mm256_storeu_pd(tmp[1]+4, c11);
    _mm256_storeu_pd(tmp[2]+0, c20); _mm256_storeu_pd(tmp[2]+4, c21);
    _mm256_storeu_pd(tmp[3]+0, c30); _mm256_storeu_pd(tmp[3]+4, c31);
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++)
            C[(size_t)i*ldC+j] += tmp[i][j];
}

/* Stage A: microkernel only, pre-packed panels reused across `reps` calls
   -- isolates pure compute, no packing/blocking/threading overhead. */
static double stage_A_kernel_only(int reps) {
    double *Ap = aligned_alloc_d((size_t)KC*MR);
    double *Bp = aligned_alloc_d((size_t)KC*NR);
    double *C = aligned_alloc_d((size_t)MR*NR);
    for (int i = 0; i < KC*MR; i++) Ap[i] = (double)(i % 7) - 3.0;
    for (int i = 0; i < KC*NR; i++) Bp[i] = (double)(i % 5) - 2.0;
    memset(C, 0, MR*NR*sizeof(double));
    volatile double sink = 0.0;
    double t0 = ladder_now();
    for (int r = 0; r < reps; r++) {
        kernel_4x8(Ap, Bp, C, MR, NR, KC, NR); /* accumulates into C, deliberately not
                                                    reset between reps -- fine numerically
                                                    (values stay well within double range
                                                    for realistic rep counts), and means
                                                    every rep's result feeds the next, so
                                                    nothing here can be proven dead */
    }
    sink = C[0];
    double t1 = ladder_now();
    double flops = 2.0 * MR * NR * KC * reps;
    if (sink == 0.123456789e300) fprintf(stderr, ""); /* keep sink observably live, never true */
    aligned_free_d(Ap); aligned_free_d(Bp); aligned_free_d(C);
    return flops / (t1 - t0) / 1e9; /* GFLOP/s */
}

/* Full multilevel DGEMM (stages B-G), OMP schedule chosen at RUNTIME via
   schedule(runtime) + omp_set_schedule() -- portable (OpenMP 2.0+, works
   under both MSVC's /openmp and gcc's -fopenmp), no recompile needed to
   switch static/dynamic/guided. */
static void matmul_full(const double *A, const double *B, double *C, int N, phase_times_t *pt) {
    memset(C, 0, (size_t)N*N*sizeof(double));
    pt->pack_a_s = pt->pack_b_s = pt->kernel_s = 0.0;
    double t_total0 = ladder_now();

    int nblocks_j = (N + NC - 1) / NC;
    int nblocks_p = (N + KC - 1) / KC;
    int nblocks_i = (N + MC - 1) / MC;
    int n_ir = 0, n_jr = 0;
    for (int i = 0; i < MC && i < N; i += MR) n_ir++;
    for (int j = 0; j < NC && j < N; j += NR) n_jr++;

    for (int jb = 0; jb < nblocks_j; jb++) {
        int j0 = jb*NC, nc = (N-j0 < NC ? N-j0 : NC);
        for (int pb = 0; pb < nblocks_p; pb++) {
            int p0 = pb*KC, kc = (N-p0 < KC ? N-p0 : KC);
            double *Bp_full = aligned_alloc_d((size_t)((nc+NR-1)/NR)*NR*kc);
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
                int i0 = ib*MC, mc = (N-i0 < MC ? N-i0 : MC);
                double *Ap_full = aligned_alloc_d((size_t)((mc+MR-1)/MR)*MR*kc);
                double ta0 = ladder_now();
                {
                    int ir;
                    int n_ir_here = (mc + MR - 1) / MR;
                    #pragma omp parallel for
                    for (ir = 0; ir < n_ir_here; ir++) {
                        int i = ir*MR;
                        int mr = (mc-i < MR ? mc-i : MR);
                        pack_panelA(A, Ap_full + (size_t)ir*MR*kc, mr, kc, N, i0+i, p0);
                    }
                }
                pt->pack_a_s += ladder_now() - ta0;

                double tk0 = ladder_now();
                {
                    /* single flattened loop over (ir,jr) -- MSVC's OpenMP 2.0
                       supports neither `collapse` nor a nested-loop form here,
                       so flatten by hand for portability */
                    int n_ir_here = (mc + MR - 1) / MR;
                    int n_jr_here = (nc + NR - 1) / NR;
                    int ntiles = n_ir_here * n_jr_here;
                    int t;
                    #pragma omp parallel for schedule(runtime)
                    for (t = 0; t < ntiles; t++) {
                        int ir = t / n_jr_here, jr = t % n_jr_here;
                        int i = ir*MR, j = jr*NR;
                        int mr = (mc-i < MR ? mc-i : MR);
                        int nr = (nc-j < NR ? nc-j : NR);
                        kernel_4x8(Ap_full + (size_t)ir*MR*kc, Bp_full + (size_t)jr*NR*kc,
                                   &C[(size_t)(i0+i)*N + (j0+j)], mr, nr, kc, N);
                    }
                }
                pt->kernel_s += ladder_now() - tk0;
                aligned_free_d(Ap_full);
            }
            aligned_free_d(Bp_full);
        }
    }
    pt->total_s = ladder_now() - t_total0;
}

static double reference_check(const double *A, const double *B, const double *C, int N) {
    /* long-double accumulation reference on a subsample for speed at large N */
    double maxerr = 0.0, maxref = 0.0;
    int step = N > 256 ? N/64 : 1;
    for (int i = 0; i < N; i += step) {
        for (int j = 0; j < N; j += step) {
            long double s = 0.0L;
            for (int k = 0; k < N; k++) s += (long double)A[(size_t)i*N+k] * (long double)B[(size_t)k*N+j];
            double ref = (double)s;
            double got = C[(size_t)i*N+j];
            double e = fabs(got - ref);
            if (e > maxerr) maxerr = e;
            if (fabs(ref) > maxref) maxref = fabs(ref);
        }
    }
    return maxref > 0 ? maxerr/maxref : maxerr;
}

#ifdef _WIN32
/* Pin each OpenMP thread to a specific logical CPU from a caller-supplied
   list (e.g. "0,1,2,...,15" for P-cores on this machine, per
   topology/classify_cores_native.c's earlier empirical classification).
   No-op if core_list is NULL/empty. Linux: rely on OMP_PLACES/
   GOMP_CPU_AFFINITY set externally instead (see run scripts). */
static int g_core_list[64], g_core_count = 0;
static void pin_threads(void) {
    if (g_core_count == 0) return;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int cpu = g_core_list[tid % g_core_count];
        SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu);
    }
}
static void parse_core_list(const char *s) {
    g_core_count = 0;
    if (!s || !*s) return;
    char buf[512]; strncpy(buf, s, sizeof(buf)-1); buf[sizeof(buf)-1] = 0;
    char *tok = strtok(buf, ",");
    while (tok && g_core_count < 64) { g_core_list[g_core_count++] = atoi(tok); tok = strtok(NULL, ","); }
}
#endif

int main(int argc, char **argv) {
    /* usage: matmul_ladder stage N threads schedule chunk reps [core_list]
       stage: A | full   schedule: static|dynamic|guided   core_list: Windows only, e.g. 0,1,2..15 */
    if (argc < 7) {
        fprintf(stderr, "usage: %s <A|full> <N> <threads> <static|dynamic|guided> <chunk> <reps> [core_list]\n", argv[0]);
        return 1;
    }
    const char *stage = argv[1];
    int N = atoi(argv[2]);
    int T = atoi(argv[3]);
    const char *sched_name = argv[4];
    int chunk = atoi(argv[5]);
    int reps = atoi(argv[6]);
#ifdef _WIN32
    parse_core_list(argc > 7 ? argv[7] : NULL);
#endif

    omp_set_num_threads(T);
    /* schedule(runtime) reads OMP_SCHEDULE at first use; set it here rather
       than via omp_set_schedule()/omp_sched_t, which are OpenMP 3.0+ APIs
       MSVC's /openmp (stuck at OpenMP 2.0) doesn't provide. Portable across
       both compilers this way. */
    {
        char envval[64];
        snprintf(envval, sizeof(envval), "%s,%d", sched_name, chunk);
#ifdef _WIN32
        _putenv_s("OMP_SCHEDULE", envval);
#else
        setenv("OMP_SCHEDULE", envval, 1);
#endif
    }

#ifdef _WIN32
    if (g_core_count > 0) pin_threads();
#endif

    if (!strcmp(stage, "A")) {
        double g = stage_A_kernel_only(reps);
        printf("ladder,A,%d,%d,%s,%d,%d,,,,,%.3f,\n", N, T, sched_name, chunk, reps, g);
        return 0;
    }

    double *A = aligned_alloc_d((size_t)N*N);
    double *B = aligned_alloc_d((size_t)N*N);
    double *C = aligned_alloc_d((size_t)N*N);
    for (int i = 0; i < N*N; i++) { A[i] = (double)((i%13)-6); B[i] = (double)((i%11)-5); }

    phase_times_t best = {1e300,1e300,1e300,1e300};
    matmul_full(A, B, C, N, &best); /* warmup */
    phase_times_t sum = {0,0,0,0};
    double best_total = 1e300;
    phase_times_t best_pt = {0,0,0,0};
    for (int r = 0; r < reps; r++) {
        phase_times_t pt;
        matmul_full(A, B, C, N, &pt);
        if (pt.total_s < best_total) { best_total = pt.total_s; best_pt = pt; }
    }
    double relerr = reference_check(A, B, C, N);
    double gflops = 2.0*(double)N*N*N / best_total / 1e9;

    printf("ladder,%s,%d,%d,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.3f,%.3e\n",
           stage, N, T, sched_name, chunk, reps,
           best_pt.pack_a_s, best_pt.pack_b_s, best_pt.kernel_s, best_total, gflops, relerr);

    aligned_free_d(A); aligned_free_d(B); aligned_free_d(C);
    return 0;
}
