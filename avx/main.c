#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <mm_malloc.h>      // for _mm_malloc / _mm_free
#include "kernel.h"         // declares: void matmul(double* A, double* B, double* C, int n)

#define MEMALIGN 64
#define NUM_RUNS 14

void matmul(double* A, double* B, double* C, int n);
void strassen(double* A, double* B, double* C, int n);

int cmpfunc(const void *a, const void *b) {
    double diff = (*(double*)a) - (*(double*)b);
    return (diff > 0) - (diff < 0);
}

void write_matrix(const char* filename, const double* M, int N) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        perror(filename);
        return;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(f, "%.6f%s", M[i * N + j], (j == N - 1) ? "\n" : " ");
        }
    }
    fclose(f);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <matrix_size> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    if (N <= 0 || num_threads <= 0) {
        fprintf(stderr, "Both matrix_size and num_threads must be positive integers.\n");
        return EXIT_FAILURE;
    }

    omp_set_num_threads(num_threads);

    // Allocate memory
    double *A = (double*) _mm_malloc((size_t)N * N * sizeof(double), MEMALIGN);
    double *B = (double*) _mm_malloc((size_t)N * N * sizeof(double), MEMALIGN);
    double *C_blocked = (double*) _mm_malloc((size_t)N * N * sizeof(double), MEMALIGN);
    double *C_strassen = (double*) _mm_malloc((size_t)N * N * sizeof(double), MEMALIGN);
    if (!A || !B || !C_blocked || !C_strassen) {
        perror("aligned alloc");
        return EXIT_FAILURE;
    }

    // Initialize matrices
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (double)(i + j);
            B[i * N + j] = (double)(i * j);
        }
    }

    double matmul_times[NUM_RUNS];
    double strassen_times[NUM_RUNS];

    // Benchmark matmul
    for (int run = 0; run < NUM_RUNS; run++) {
        memset(C_blocked, 0, (size_t)N * N * sizeof(double));
        double t0 = omp_get_wtime();
        matmul(A, B, C_blocked, N);
        double t1 = omp_get_wtime();
        matmul_times[run] = t1 - t0;
    }

    // Benchmark strassen
    for (int run = 0; run < NUM_RUNS; run++) {
        memset(C_strassen, 0, (size_t)N * N * sizeof(double));
        double t2 = omp_get_wtime();
        strassen(A, B, C_strassen, N);
        double t3 = omp_get_wtime();
        strassen_times[run] = t3 - t2;
    }

    // Sort times
    qsort(matmul_times, NUM_RUNS, sizeof(double), cmpfunc);
    qsort(strassen_times, NUM_RUNS, sizeof(double), cmpfunc);

    // Compute trimmed means
    double matmul_sum = 0.0;
    double strassen_sum = 0.0;
    for (int i = 2; i < NUM_RUNS - 2; i++) {
        matmul_sum += matmul_times[i];
        strassen_sum += strassen_times[i];
    }

    double matmul_avg = matmul_sum / (NUM_RUNS - 4);
    double strassen_avg = strassen_sum / (NUM_RUNS - 4);

    printf("Matrix size: %d×%d, Threads: %d\n", N, N, num_threads);
    printf("Average elapsed time for Blocked (middle 10 runs): %.6f sec\n", matmul_avg);
    printf("Average elapsed time for Strassen's (middle 10 runs): %.6f sec\n", strassen_avg);

    // Write results to file
    // write_matrix("result_blocked.txt", C_blocked, N);
    // write_matrix("result_strassen.txt", C_strassen, N);

    _mm_free(A);
    _mm_free(B);
    _mm_free(C_blocked);
    _mm_free(C_strassen);

    return EXIT_SUCCESS;
}
