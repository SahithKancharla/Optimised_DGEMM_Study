// main.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <mm_malloc.h>      // for _mm_malloc / _mm_free
#include "kernel.h"         // declares: void matmul(double* A, double* B, double* C, int m, int n, int k);

#define MEMALIGN 64

void matmul(double* A, double* B, double* C, int n);
void strassen(double* A, double* B, double* C, int n);

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

    // Open output file
    FILE *out = fopen("result_blocked.txt", "w");
    if (!out) {
        perror("fopen");
        return EXIT_FAILURE;
    }

    // Allocate N×N double matrices, 64‑byte aligned
    double *A = (double*) _mm_malloc((size_t)N * N * sizeof(double), MEMALIGN);
    double *B = (double*) _mm_malloc((size_t)N * N * sizeof(double), MEMALIGN);
    double *C = (double*) _mm_malloc((size_t)N * N * sizeof(double), MEMALIGN);
    if (!A || !B || !C) {
        perror("aligned alloc");
        fclose(out);
        return EXIT_FAILURE;
    }

    // Populate A, B, C
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (double)(i + j);
            B[i * N + j] = (double)(i * j);
            C[i * N + j] = 0.0;
        }
    }


    double t0 = omp_get_wtime();
    matmul(A, B, C, N);
    double t1 = omp_get_wtime();
    double elapsed = t1 - t0;

    // Write summary to file
    printf("Matrix size: %d×%d, Threads: %d\n", N, N, num_threads);
    printf("Elapsed time for Blocked: %.6f sec\n", elapsed);




    double t2 = omp_get_wtime();
    strassen(A, B, C, N);
    double t3 = omp_get_wtime();
    double elapsed1 = t3 - t2;

    printf("Elapsed time for Strassens: %.6f sec\n", elapsed1);
    // Write full C matrix
    // printf("Full C matrix (%d×%d):\n", N, N);
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         fprintf(out, "%f ", C[j * N + i]);
    //     }
    //     fprintf(out, "\n");
    // }

    // Clean up
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    fclose(out);

    return EXIT_SUCCESS;
}
