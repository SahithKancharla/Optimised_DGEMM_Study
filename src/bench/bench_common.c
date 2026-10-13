#include "bench_common.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

/* Column-major throughout: element (i,j) of an N x N matrix lives at M[j*N + i],
 * matching CblasColMajor so the same buffers feed straight into cblas_dgemm. */

double *bench_alloc(size_t n_doubles) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n_doubles * sizeof(double)) != 0) return NULL;
    return (double *)p;
}

void bench_free(double *p) { free(p); }

#define UINT32_MAX_LOCAL 4294967295u

void bench_init_random(double *M, size_t n, unsigned seed) {
    unsigned state = seed ? seed : 1u;
    for (size_t i = 0; i < n; i++) {
        /* xorshift32 -- deterministic, no external RNG dependency */
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        double u = (double)(state) / (double)UINT32_MAX_LOCAL;
        M[i] = u * 2.0 - 1.0;
    }
}

void bench_reference_gemm(const double *A, const double *B, double *Cref, int N) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            long double acc = 0.0L;
            for (int k = 0; k < N; k++) {
                acc += (long double)A[(size_t)k * N + i] * (long double)B[(size_t)j * N + k];
            }
            Cref[(size_t)j * N + i] = (double)acc;
        }
    }
}

bench_compare_t bench_compare(const double *C, const double *Cref, int N) {
    bench_compare_t r = {0.0, 0.0};
    double num = 0.0, den = 0.0, maxrel = 0.0;
    size_t total = (size_t)N * (size_t)N;
    for (size_t i = 0; i < total; i++) {
        double d = C[i] - Cref[i];
        num += d * d;
        den += Cref[i] * Cref[i];
        double denom = fabs(Cref[i]) > 1e-300 ? fabs(Cref[i]) : 1e-300;
        double rel = fabs(d) / denom;
        if (rel > maxrel) maxrel = rel;
    }
    r.rel_frobenius = den > 0.0 ? sqrt(num / den) : sqrt(num);
    r.rel_max = maxrel;
    return r;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

bench_result_t bench_run(gemm_fn fn, const double *A, const double *B, double *C,
                          int N, int reps) {
    bench_result_t r = {0, 0, 0, 0};
    fn(A, B, C, N); /* untimed warmup: page faults, cold caches, turbo ramp-up */
    double *times = malloc(sizeof(double) * (size_t)reps);
    for (int t = 0; t < reps; t++) {
        double t0 = omp_get_wtime();
        fn(A, B, C, N);
        double t1 = omp_get_wtime();
        times[t] = t1 - t0;
    }
    qsort(times, reps, sizeof(double), cmp_double);
    r.min_s = times[0];
    r.max_s = times[reps - 1];
    r.median_s = times[reps / 2];
    double flops = 2.0 * (double)N * (double)N * (double)N;
    r.gflops_median = flops / r.median_s / 1e9;
    free(times);
    return r;
}

void bench_print_csv_header(FILE *f) {
    fprintf(f, "variant,N,threads,label,reps,median_s,min_s,gflops,rel_frob_err,rel_max_err\n");
}

void bench_print_csv_row(FILE *f, const char *variant, int N, int threads,
                          const char *label, int reps, bench_result_t r,
                          bench_compare_t cmp) {
    fprintf(f, "%s,%d,%d,%s,%d,%.9f,%.9f,%.6f,%.3e,%.3e\n",
            variant, N, threads, label, reps, r.median_s, r.min_s,
            r.gflops_median, cmp.rel_frobenius, cmp.rel_max);
}
