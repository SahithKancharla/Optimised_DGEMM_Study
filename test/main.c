#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include <sys/stat.h>

#define MEMALIGN 64
#define TOL 1e-6

// Function declarations
extern double* mm_naive(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_blocked(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_strassen(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_naive_unrolled(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_blocked_unrolled(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_strassen_unrolled(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_strassen_naive(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_parallel_blocked(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_parallel_blocked_unrolled(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_parallel_strassen(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_parallel_strassen_unrolled(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_parallel_strassen_naive(int nthreads, int rows, int cols_a, int cols_b);

// AVX kernel functions
extern void matmul(double* A, double* B, double* C, int n);
extern void strassen(double* A, double* B, double* C, int n);

// AVX wrappers
double* mm_avx_blocked(int nthreads, int rows, int cols_a, int cols_b);
double* mm_avx_strassen(int nthreads, int rows, int cols_a, int cols_b);

void initialize_and_write_matrices(int rows, int cols_a, int cols_b) {
    double *a = malloc(rows * cols_a * sizeof(double));
    double *b = malloc(cols_a * cols_b * sizeof(double));
    if (!a || !b) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols_a; j++)
            a[i * cols_a + j] = i + j;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < cols_a; i++)
        for (int j = 0; j < cols_b; j++)
            b[i * cols_b + j] = i * j;

    mkdir("Data", 0777);
    FILE *fa = fopen("Data/matrix_a.bin", "wb");
    FILE *fb = fopen("Data/matrix_b.bin", "wb");
    if (!fa || !fb) {
        fprintf(stderr, "Failed to write matrices.\n");
        exit(1);
    }

    fwrite(a, sizeof(double), rows * cols_a, fa);
    fwrite(b, sizeof(double), cols_a * cols_b, fb);
    fclose(fa); fclose(fb);
    free(a); free(b);
}

void transpose_square(double *dst, const double *src, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            dst[i * N + j] = src[j * N + i];
}

static int compare_mat(double *ref, double *test, int N) {
    for (int i = 0; i < N * N; i++) {
        if (fabs(ref[i] - test[i]) > TOL)
            return i;
    }
    return -1;
}

// AVX wrappers
double* mm_avx_blocked(int nthreads, int rows, int cols_a, int cols_b) {
    if (rows != cols_a || cols_a != cols_b) return NULL;
    omp_set_num_threads(nthreads);

    size_t sz = (size_t)rows * rows;
    double *A = _mm_malloc(sz * sizeof(double), MEMALIGN);
    double *B = _mm_malloc(sz * sizeof(double), MEMALIGN);
    double *C = _mm_malloc(sz * sizeof(double), MEMALIGN);
    if (!A || !B || !C) return NULL;

    FILE *fa = fopen("Data/matrix_a.bin", "rb");
    FILE *fb = fopen("Data/matrix_b.bin", "rb");
    if (!fa || !fb) return NULL;
    fread(A, sizeof(double), sz, fa);
    fread(B, sizeof(double), sz, fb);
    fclose(fa); fclose(fb);

    memset(C, 0, sz * sizeof(double));
    matmul(A, B, C, rows);

    _mm_free(A); _mm_free(B);
    return C;
}

double* mm_avx_strassen(int nthreads, int rows, int cols_a, int cols_b) {
    if (rows != cols_a || cols_a != cols_b) return NULL;
    omp_set_num_threads(nthreads);

    size_t sz = (size_t)rows * rows;
    double *A = _mm_malloc(sz * sizeof(double), MEMALIGN);
    double *B = _mm_malloc(sz * sizeof(double), MEMALIGN);
    double *C = _mm_malloc(sz * sizeof(double), MEMALIGN);
    if (!A || !B || !C) return NULL;

    FILE *fa = fopen("Data/matrix_a.bin", "rb");
    FILE *fb = fopen("Data/matrix_b.bin", "rb");
    if (!fa || !fb) return NULL;
    fread(A, sizeof(double), sz, fa);
    fread(B, sizeof(double), sz, fb);
    fclose(fa); fclose(fb);

    memset(C, 0, sz * sizeof(double));
    strassen(A, B, C, rows);

    _mm_free(A); _mm_free(B);
    return C;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <matrix_size> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);

    initialize_and_write_matrices(N, N, N);

    double *C_ref = mm_naive(num_threads, N, N, N);
    if (!C_ref) {
        fprintf(stderr, "mm_naive failed.\n");
        return EXIT_FAILURE;
    }

    FILE *fout = fopen("result_naive.txt", "w");
    for (int i = 0; i < N * N; i++)
        fprintf(fout, "%.6f%s", C_ref[i], (i % N == N - 1) ? "\n" : " ");
    fclose(fout);

    typedef double* (*mm_fn)(int, int, int, int);
    struct { const char *name; mm_fn fn; } tests[] = {
        { "mm_blocked", mm_blocked },
        { "mm_strassen", mm_strassen },
        { "mm_naive_unrolled", mm_naive_unrolled },
        { "mm_blocked_unrolled", mm_blocked_unrolled },
        { "mm_strassen_unrolled", mm_strassen_unrolled },
        { "mm_strassen_naive", mm_strassen_naive },
        { "mm_parallel_blocked", mm_parallel_blocked },
        { "mm_parallel_blocked_unrolled", mm_parallel_blocked_unrolled },
        { "mm_parallel_strassen", mm_parallel_strassen },
        { "mm_parallel_strassen_unrolled", mm_parallel_strassen_unrolled },
        { "mm_parallel_strassen_naive", mm_parallel_strassen_naive },
        { "avx_blocked", mm_avx_blocked },
        { "avx_strassen", mm_avx_strassen },
    };

    size_t ntests = sizeof(tests) / sizeof(*tests);
    for (size_t t = 0; t < ntests; t++) {
        double *C = tests[t].fn(num_threads, N, N, N);
        if (!C) {
            printf("%-30s : FAILED (null result)\n", tests[t].name);
            continue;
        }

        double *C_compare = C;
        if (strcmp(tests[t].name, "avx_blocked") == 0) {
            C_compare = _mm_malloc(N * N * sizeof(double), MEMALIGN);
            if (!C_compare) {
                fprintf(stderr, "Memory alloc failed for transpose buffer\n");
                _mm_free(C);
                continue;
            }
            transpose_square(C_compare, C, N);
        }

        int bad = compare_mat(C_ref, C_compare, N);
        if (bad < 0) {
            printf("%-30s : PASS\n", tests[t].name);
        } else {
            int r = bad / N, c = bad % N;
            printf("%-30s : FAIL at [%d,%d]  ref=%.6f  got=%.6f\n",
                tests[t].name, r, c, C_ref[bad], C_compare[bad]);
        }

        if (C_compare != C)
            _mm_free(C_compare);
        _mm_free(C);
    }

    _mm_free(C_ref);
    return 0;
}
