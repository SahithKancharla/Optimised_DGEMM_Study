#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

double* mm_naive_unrolled(int nthreads, int NUM_ROWS_A, int NUM_COLS_A, int NUM_COLS_B) {
    omp_set_num_threads(nthreads);

    double *a = (double *)malloc(NUM_ROWS_A * NUM_COLS_A * sizeof(double));
    double *b = (double *)malloc(NUM_COLS_A * NUM_COLS_B * sizeof(double));
    double *c = (double *)calloc(NUM_ROWS_A * NUM_COLS_B, sizeof(double));  // calloc zeroes out

    FILE *fa = fopen("Data/matrix_a.bin", "rb");
    FILE *fb = fopen("Data/matrix_b.bin", "rb");

    fread(a, sizeof(double), NUM_ROWS_A * NUM_COLS_A, fa);
    fread(b, sizeof(double), NUM_COLS_A * NUM_COLS_B, fb);

    fclose(fa);
    fclose(fb);

    double start_time = omp_get_wtime();

    // Loop reorder for better cache behavior + loop unrolling on innermost loop
    for (int i = 0; i < NUM_ROWS_A; i++) {
        for (int k = 0; k < NUM_COLS_A; k++) {
            double a_ik = a[i * NUM_COLS_A + k];
            int j;
            for (j = 0; j <= NUM_COLS_B - 4; j += 4) {  // unroll 4 iterations
                c[i * NUM_COLS_B + j]     += a_ik * b[k * NUM_COLS_B + j];
                c[i * NUM_COLS_B + j + 1] += a_ik * b[k * NUM_COLS_B + j + 1];
                c[i * NUM_COLS_B + j + 2] += a_ik * b[k * NUM_COLS_B + j + 2];
                c[i * NUM_COLS_B + j + 3] += a_ik * b[k * NUM_COLS_B + j + 3];
            }
            // Remainder loop (handle leftovers if not divisible by 4)
            for (; j < NUM_COLS_B; j++) {
                c[i * NUM_COLS_B + j] += a_ik * b[k * NUM_COLS_B + j];
            }
        }
    }

    double elapsed_time = omp_get_wtime() - start_time;

    FILE *ftime = fopen("Values/time_naive_unrolled.txt", "a");
    fprintf(ftime, "%f\n", elapsed_time);
    fclose(ftime);

    free(a);
    free(b);
    return c;
}
