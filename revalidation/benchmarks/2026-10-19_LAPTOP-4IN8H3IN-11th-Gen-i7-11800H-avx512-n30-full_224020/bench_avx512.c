/* bench_avx512.c -- ISA-agnostic benchmark driver for matmul() / strassen().
 *
 * Linked against EITHER audited_submission/avx/{kernel.c,strassen.c} (AVX-512)
 * OR {kernel_avx2.c,strassen_avx2.c} (AVX2) -- both expose the same
 * matmul(A,B,C,N) / strassen(A,B,C,N) entry points -- so one driver gives an
 * apples-to-apples AVX-512 vs AVX2 comparison on the same machine.
 *
 * Usage: bench_avx512 <isa_label> <sizes,comma> <threads,comma> <reps>
 * Prints one CSV header + one row per (N,T) to stdout; progress to stderr.
 *
 * Correctness: Freivalds-style check (C*x vs A*(B*x), random x) at EVERY size,
 * O(N^2), so even N=8192 is verified without an O(N^3) reference. relerr is
 * the inf-norm difference over the inf-norm of A*(B*x). Rows with
 * relerr > 1e-8 are flagged by run_avx512.sh as FAIL.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>
#include <mm_malloc.h>

void matmul(double *A, double *B, double *C, int N);
void strassen(double *A, double *B, double *C, int N);

typedef void (*mm_fn)(double *, double *, double *, int);

static uint64_t rng_state = 88172645463325252ULL;
static double rnd(void) { /* xorshift64*, uniform in [-1,1) */
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    uint64_t r = rng_state * 2685821657736338717ULL;
    return ((double)(r >> 11) / 9007199254740992.0) * 2.0 - 1.0;
}

static int cmp_d(const void *a, const void *b) {
    double d = *(const double *)a - *(const double *)b;
    return (d > 0) - (d < 0);
}

static void matvec(const double *M, const double *x, double *y, int N) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        double s = 0.0;
        for (int k = 0; k < N; k++) s += M[(size_t)i * N + k] * x[k];
        y[i] = s;
    }
}

static double freivalds_relerr(const double *A, const double *B, const double *C, int N) {
    double *x = malloc(N * sizeof(double)), *t = malloc(N * sizeof(double));
    double *z = malloc(N * sizeof(double)), *w = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) x[i] = rnd();
    matvec(B, x, t, N);   /* t = B x     */
    matvec(A, t, z, N);   /* z = A (B x) */
    matvec(C, x, w, N);   /* w = C x     */
    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; i++) {
        double d = fabs(w[i] - z[i]), a = fabs(z[i]);
        if (d > num) num = d;
        if (a > den) den = a;
    }
    free(x); free(t); free(z); free(w);
    return den > 0 ? num / den : num;
}

static void bench(mm_fn fn, double *A, double *B, double *C, int N, int T, int reps,
                  double *best, double *med, double *relerr) {
    omp_set_num_threads(T);
    fn(A, B, C, N); /* untimed warm-up (page faults, static-buffer first touch) */
    double *ts = malloc(reps * sizeof(double));
    for (int r = 0; r < reps; r++) {
        double t0 = omp_get_wtime();
        fn(A, B, C, N);
        ts[r] = omp_get_wtime() - t0;
    }
    *relerr = freivalds_relerr(A, B, C, N);
    qsort(ts, reps, sizeof(double), cmp_d);
    *best = ts[0];
    *med = ts[reps / 2];
    free(ts);
}

static int parse_list(const char *s, int *out, int max) {
    int n = 0;
    char *copy = strdup(s), *save = NULL;
    for (char *tok = strtok_r(copy, ",", &save); tok && n < max; tok = strtok_r(NULL, ",", &save))
        out[n++] = atoi(tok);
    free(copy);
    return n;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <isa_label> <sizes,comma> <threads,comma> <reps>\n", argv[0]);
        return 2;
    }
    const char *isa = argv[1];
    int sizes[64], threads[64];
    int ns = parse_list(argv[2], sizes, 64), nt = parse_list(argv[3], threads, 64);
    int reps = atoi(argv[4]);
    if (ns < 1 || nt < 1 || reps < 1) { fprintf(stderr, "bad arguments\n"); return 2; }

    printf("isa,N,T,reps,blocked_best_s,blocked_median_s,blocked_gflops,blocked_relerr,"
           "strassen_best_s,strassen_median_s,strassen_gflops,strassen_relerr\n");
    for (int si = 0; si < ns; si++) {
        int N = sizes[si];
        size_t bytes = (size_t)N * N * sizeof(double);
        double *A = _mm_malloc(bytes, 64), *B = _mm_malloc(bytes, 64), *C = _mm_malloc(bytes, 64);
        if (!A || !B || !C) { fprintf(stderr, "alloc failed N=%d\n", N); return 1; }
        for (size_t i = 0; i < (size_t)N * N; i++) { A[i] = rnd(); B[i] = rnd(); }
        double flops = 2.0 * N * (double)N * N;
        for (int ti = 0; ti < nt; ti++) {
            int T = threads[ti];
            double bb, bm, be, sb, sm, se;
            bench(matmul, A, B, C, N, T, reps, &bb, &bm, &be);
            bench(strassen, A, B, C, N, T, reps, &sb, &sm, &se);
            printf("%s,%d,%d,%d,%.6f,%.6f,%.3f,%.3e,%.6f,%.6f,%.3f,%.3e\n", isa, N, T, reps,
                   bb, bm, flops / bb / 1e9, be, sb, sm, flops / sb / 1e9, se);
            fflush(stdout);
            fprintf(stderr, "[%s] N=%d T=%d  blocked %.1f GFLOP/s  strassen %.1f GFLOP/s\n",
                    isa, N, T, flops / bb / 1e9, flops / sb / 1e9);
        }
        _mm_free(A); _mm_free(B); _mm_free(C);
    }
    return 0;
}
