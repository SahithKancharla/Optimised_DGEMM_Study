#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

double* mm_naive(int nthreads, int NUM_ROWS_A, int NUM_COLS_A, int NUM_COLS_B) {
    omp_set_num_threads(nthreads);

    double *a = (double *)malloc(NUM_ROWS_A * NUM_COLS_A * sizeof(double));
    double *b = (double *)malloc(NUM_COLS_A * NUM_COLS_B * sizeof(double));
    double *c = (double *)malloc(NUM_ROWS_A * NUM_COLS_B * sizeof(double));

    FILE *fa = fopen("Data/matrix_a.bin", "rb");
    FILE *fb = fopen("Data/matrix_b.bin", "rb");

    fread(a, sizeof(double), NUM_ROWS_A * NUM_COLS_A, fa);
    fread(b, sizeof(double), NUM_COLS_A * NUM_COLS_B, fb);

    fclose(fa);
    fclose(fb);

    // Initialize matrix c
    for (int i = 0; i < NUM_ROWS_A * NUM_COLS_B; i++) {
        c[i] = 0.0;
    }

    double start_time = omp_get_wtime();

    // Perform matrix multiplication
    for (int i = 0; i < NUM_ROWS_A; i++) {
        for (int j = 0; j < NUM_COLS_B; j++) {
            for (int k = 0; k < NUM_COLS_A; k++) {
                c[i * NUM_COLS_B + j] += a[i * NUM_COLS_A + k] * b[k * NUM_COLS_B + j];
            }
        }
    }

    double elapsed_time = omp_get_wtime() - start_time;

    FILE *ftime = fopen("Values/time_naive.txt", "a");
    fprintf(ftime, "%f\n", elapsed_time);
    fclose(ftime);

    free(a);
    free(b);
    return c;
}
