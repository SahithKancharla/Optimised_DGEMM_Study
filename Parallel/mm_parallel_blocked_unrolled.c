#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32

double* mm_parallel_blocked_unrolled(int nthreads, int NUM_ROWS_A, int NUM_COLS_A, int NUM_COLS_B){
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
    #pragma omp parallel for
    for (int i = 0; i < NUM_ROWS_A * NUM_COLS_B; i++) {
        c[i] = 0.0;
    }

    double start_time = omp_get_wtime();

    // Blocked matrix multiplication using 1D arrays
    #pragma omp parallel for collapse(3) schedule(static)
    for (int ii = 0; ii < NUM_ROWS_A; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < NUM_COLS_B; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < NUM_COLS_A; kk += BLOCK_SIZE) {

                
                for (int i = ii; i < ii + BLOCK_SIZE && i < NUM_ROWS_A; i++) {
                    for (int j = jj; j < jj + BLOCK_SIZE && j < NUM_COLS_B; j++) {
                        double temp = 0.0;  // Private variable for each thread

                        int kend = (kk + BLOCK_SIZE < NUM_COLS_A) ? kk + BLOCK_SIZE : NUM_COLS_A;

                        int k;
                        for (k = kk; k + 3 < kend; k += 4) {
                            temp += a[i * NUM_COLS_A + k] * b[k * NUM_COLS_B + j]
                            + a[i * NUM_COLS_A + k+1] * b[(k+1) * NUM_COLS_B + j]
                            + a[i * NUM_COLS_A + k+2] * b[(k+2) * NUM_COLS_B + j]
                            + a[i * NUM_COLS_A + k+3] * b[(k+3) * NUM_COLS_B + j];

                        }

                        for (; k < kend; k++) {
                            temp += a[i * NUM_COLS_A + k] * b[k * NUM_COLS_B + j];
                        }

                        // Critical section to update shared result array
                        c[i * NUM_COLS_B + j] += temp;
                    }
                }
            }
        }
    }

    double time = omp_get_wtime() - start_time;
    FILE *ftime = fopen("Values/parallel.txt", "a");
    fprintf(ftime, "%f\n", time);
    fclose(ftime);

    free(a);
    free(b);
    return c;
}
