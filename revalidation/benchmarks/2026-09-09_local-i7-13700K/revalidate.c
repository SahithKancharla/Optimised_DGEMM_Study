#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mm_malloc.h>
#include <sys/stat.h>

#define REPS 3

/* Serial */
extern double* mm_naive(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_naive_unrolled(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_blocked(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_blocked_unrolled(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_strassen_naive(int nthreads, int rows, int cols_a, int cols_b); /* pure Strassen, n=2 base */
extern double* mm_strassen(int nthreads, int rows, int cols_a, int cols_b);       /* hybrid, cutoff 128 */
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

typedef double* (*mm_fn)(int, int, int, int);

static double time_it(mm_fn fn, int T, int N) {
    double best = 1e300;
    for (int r = 0; r < REPS; r++) {
        double t0 = omp_get_wtime();
        double *c = fn(T, N, N, N);
        double t1 = omp_get_wtime();
        free(c);
        if (t1 - t0 < best) best = t1 - t0;
    }
    return best;
}

static double time_avx(void (*fn)(double*, double*, double*, int), int T, int N) {
    omp_set_num_threads(T);
    double *A = _mm_malloc((size_t)N * N * sizeof(double), 64);
    double *B = _mm_malloc((size_t)N * N * sizeof(double), 64);
    double *C = _mm_malloc((size_t)N * N * sizeof(double), 64);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i + j;
            B[i * N + j] = (double)i * j;
        }
    double best = 1e300;
    for (int r = 0; r < REPS; r++) {
        double t0 = omp_get_wtime();
        fn(A, B, C, N);
        double t1 = omp_get_wtime();
        if (t1 - t0 < best) best = t1 - t0;
    }
    _mm_free(A); _mm_free(B); _mm_free(C);
    return best;
}

int main(int argc, char **argv) {
    mkdir("Data", 0777);
    mkdir("Values", 0777);
    mkdir("Results", 0777);

    int sizes[] = {64, 128, 256, 512, 1024, 2048};
    int nsizes = sizeof(sizes) / sizeof(*sizes);
    int threads[] = {1, 2, 4, 7, 8, 16};
    int nthreads_list = sizeof(threads) / sizeof(*threads);

    printf("=== SERIAL (T=1) ===\n");
    printf("%-6s %10s %10s %10s %10s %10s %10s %10s\n",
           "N", "naive", "nv_unrl", "blocked", "bl_unrl", "strassen", "str_base", "strb_unrl");
    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        write_matrices(N);
        double t_naive = (N <= 512) ? time_it(mm_naive, 1, N) : -1;
        double t_naive_u = time_it(mm_naive_unrolled, 1, N);
        double t_blocked = time_it(mm_blocked, 1, N);
        double t_blocked_u = time_it(mm_blocked_unrolled, 1, N);
        double t_strassen = (N <= 512) ? time_it(mm_strassen_naive, 1, N) : -1;
        double t_strassen_base = time_it(mm_strassen, 1, N);
        double t_strassen_base_u = time_it(mm_strassen_unrolled, 1, N);
        printf("%-6d %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n",
               N, t_naive, t_naive_u, t_blocked, t_blocked_u,
               t_strassen, t_strassen_base, t_strassen_base_u);
        fflush(stdout);
    }

    printf("\n=== PARALLEL ===\n");
    printf("%-6s %-4s %10s %10s %10s %10s %10s\n",
           "N", "T", "blocked", "bl_unrl", "str_base", "strb_unrl", "str_naive");
    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        write_matrices(N);
        for (int ti = 0; ti < nthreads_list; ti++) {
            int T = threads[ti];
            double t_blocked = time_it(mm_parallel_blocked, T, N);
            double t_blocked_u = time_it(mm_parallel_blocked_unrolled, T, N);
            double t_strassen_base = time_it(mm_parallel_strassen, T, N);
            double t_strassen_base_u = time_it(mm_parallel_strassen_unrolled, T, N);
            double t_strassen_naive = (N <= 512) ? time_it(mm_parallel_strassen_naive, T, N) : -1;
            printf("%-6d %-4d %10.6f %10.6f %10.6f %10.6f %10.6f\n",
                   N, T, t_blocked, t_blocked_u, t_strassen_base, t_strassen_base_u, t_strassen_naive);
            fflush(stdout);
        }
    }

    printf("\n=== AVX2 (matmul.c fixed + kernel_avx2.c / strassen_avx2.c) ===\n");
    printf("%-6s %-4s %12s %12s\n", "N", "T", "blocked_avx2", "strassen_avx2");
    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        for (int ti = 0; ti < nthreads_list; ti++) {
            int T = threads[ti];
            double t_blocked = time_avx(matmul, T, N);
            double t_strassen = time_avx(strassen, T, N);
            printf("%-6d %-4d %12.6f %12.6f\n", N, T, t_blocked, t_strassen);
            fflush(stdout);
        }
    }

    return 0;
}
