#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mm_malloc.h>

/*
 * Single-shot driver: runs exactly one algorithm, once, at one (N, T), and
 * prints one CSV line: algo,N,T,time_sec,calls,bytes_allocated
 *
 * Deliberately does ONE call per process (no internal repetition) so this
 * can be wrapped with `perf stat -- ./run_one <algo> <N> <T>` and have the
 * hardware counters attributed cleanly to that single run, and paired with
 * a `-DCOUNT_CALLS` build to get recursive-call / allocation-byte counts
 * for the same call. Run it three times (plain, under perf, counted build)
 * per (algo,N,T) rather than trying to get everything from one binary.
 *
 * calls/bytes are 0 for non-Strassen algorithms (not applicable) and for
 * Strassen algorithms when built without -DCOUNT_CALLS (counters are 0,
 * not tracked, in that build).
 */

/* Serial */
extern double* mm_naive(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_naive_unrolled(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_blocked(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_blocked_unrolled(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_strassen_naive(int nthreads, int rows, int cols_a, int cols_b); /* pure */
extern double* mm_strassen(int nthreads, int rows, int cols_a, int cols_b);       /* hybrid */
extern double* mm_strassen_unrolled(int nthreads, int rows, int cols_a, int cols_b);

/* Parallel */
extern double* mm_parallel_blocked(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_parallel_blocked_unrolled(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_parallel_strassen(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_parallel_strassen_unrolled(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_parallel_strassen_naive(int nthreads, int rows, int cols_a, int cols_b);

/* AVX2 */
extern void matmul(double *A, double *B, double *C, int N);
extern void strassen(double *A, double *B, double *C, int N);

/* Strassen call/byte counters -- 0 unless built with -DCOUNT_CALLS */
extern unsigned long long g_calls_serial_pure, g_bytes_serial_pure;
extern unsigned long long g_calls_serial_hybrid, g_bytes_serial_hybrid;
extern unsigned long long g_calls_serial_hybrid_unrolled, g_bytes_serial_hybrid_unrolled;
extern unsigned long long g_calls_parallel_pure, g_bytes_parallel_pure;
extern unsigned long long g_calls_parallel_hybrid, g_bytes_parallel_hybrid;
extern unsigned long long g_calls_parallel_hybrid_unrolled, g_bytes_parallel_hybrid_unrolled;
extern unsigned long long g_calls_avx2_hybrid, g_bytes_avx2_hybrid;

static void write_matrices(int N) {
    double *a = malloc((size_t)N * N * sizeof(double));
    double *b = malloc((size_t)N * N * sizeof(double));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i + j;
            b[i * N + j] = (double)i * j;
        }
    FILE *fa = fopen("Data/matrix_a.bin", "wb");
    FILE *fb = fopen("Data/matrix_b.bin", "wb");
    fwrite(a, sizeof(double), (size_t)N * N, fa);
    fwrite(b, sizeof(double), (size_t)N * N, fb);
    fclose(fa); fclose(fb);
    free(a); free(b);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <algo> <N> <T>\n", argv[0]);
        return 1;
    }
    const char *algo = argv[1];
    int N = atoi(argv[2]);
    int T = atoi(argv[3]);

    system("mkdir -p Data Values Results");
    omp_set_num_threads(T);

    unsigned long long calls = 0, bytes = 0;
    double t0, t1;

    if (strcmp(algo, "avx2_blocked") == 0 || strcmp(algo, "avx2_strassen") == 0) {
        double *A = _mm_malloc((size_t)N * N * sizeof(double), 64);
        double *B = _mm_malloc((size_t)N * N * sizeof(double), 64);
        double *C = _mm_malloc((size_t)N * N * sizeof(double), 64);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i * N + j] = i + j;
                B[i * N + j] = (double)i * j;
            }
        t0 = omp_get_wtime();
        if (strcmp(algo, "avx2_blocked") == 0) matmul(A, B, C, N);
        else strassen(A, B, C, N);
        t1 = omp_get_wtime();
        if (strcmp(algo, "avx2_strassen") == 0) { calls = g_calls_avx2_hybrid; bytes = g_bytes_avx2_hybrid; }
        _mm_free(A); _mm_free(B); _mm_free(C);
    } else {
        write_matrices(N);
        double *c = NULL;
        t0 = omp_get_wtime();
        if      (!strcmp(algo, "naive"))                      c = mm_naive(T, N, N, N);
        else if (!strcmp(algo, "naive_unrolled"))              c = mm_naive_unrolled(T, N, N, N);
        else if (!strcmp(algo, "blocked"))                     c = mm_blocked(T, N, N, N);
        else if (!strcmp(algo, "blocked_unrolled"))            c = mm_blocked_unrolled(T, N, N, N);
        else if (!strcmp(algo, "strassen_pure"))               { c = mm_strassen_naive(T, N, N, N); calls = g_calls_serial_pure; bytes = g_bytes_serial_pure; }
        else if (!strcmp(algo, "strassen_hybrid"))             { c = mm_strassen(T, N, N, N); calls = g_calls_serial_hybrid; bytes = g_bytes_serial_hybrid; }
        else if (!strcmp(algo, "strassen_hybrid_unrolled"))    { c = mm_strassen_unrolled(T, N, N, N); calls = g_calls_serial_hybrid_unrolled; bytes = g_bytes_serial_hybrid_unrolled; }
        else if (!strcmp(algo, "parallel_blocked"))            c = mm_parallel_blocked(T, N, N, N);
        else if (!strcmp(algo, "parallel_blocked_unrolled"))   c = mm_parallel_blocked_unrolled(T, N, N, N);
        else if (!strcmp(algo, "parallel_strassen_hybrid"))    { c = mm_parallel_strassen(T, N, N, N); calls = g_calls_parallel_hybrid; bytes = g_bytes_parallel_hybrid; }
        else if (!strcmp(algo, "parallel_strassen_hybrid_unrolled")) { c = mm_parallel_strassen_unrolled(T, N, N, N); calls = g_calls_parallel_hybrid_unrolled; bytes = g_bytes_parallel_hybrid_unrolled; }
        else if (!strcmp(algo, "parallel_strassen_pure"))      { c = mm_parallel_strassen_naive(T, N, N, N); calls = g_calls_parallel_pure; bytes = g_bytes_parallel_pure; }
        else { fprintf(stderr, "unknown algo: %s\n", algo); return 1; }
        t1 = omp_get_wtime();
        free(c);
    }

    printf("%s,%d,%d,%.9f,%llu,%llu\n", algo, N, T, t1 - t0, calls, bytes);
    return 0;
}
