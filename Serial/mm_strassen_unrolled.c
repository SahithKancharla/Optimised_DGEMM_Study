#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static double* initializeMatrix1D(int n) {
    return (double*)calloc(n * n, sizeof(double));
}

static void addMatrix(double* A, double* B, double* C, int n) {
    for (int i = 0; i < n * n; i++)
        C[i] = A[i] + B[i];
}

static void subtractMatrix(double* A, double* B, double* C, int n) {
    for (int i = 0; i < n * n; i++)
        C[i] = A[i] - B[i];
}

static double* strassenMultiplyUnrolled(double* A, double* B, int n) {
    if (n <= 128) {
        double* C = initializeMatrix1D(n);
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                double a_ik = A[i * n + k];
                int j;
                for (j = 0; j <= n - 4; j += 4) {
                    C[i * n + j]     += a_ik * B[k * n + j];
                    C[i * n + j + 1] += a_ik * B[k * n + j + 1];
                    C[i * n + j + 2] += a_ik * B[k * n + j + 2];
                    C[i * n + j + 3] += a_ik * B[k * n + j + 3];
                }
                for (; j < n; j++) {
                    C[i * n + j] += a_ik * B[k * n + j];
                }
            }
        }        
        return C;
    }

    int k = n / 2;
    int size = k * k;
    double *A11 = initializeMatrix1D(k), *A12 = initializeMatrix1D(k), *A21 = initializeMatrix1D(k), *A22 = initializeMatrix1D(k);
    double *B11 = initializeMatrix1D(k), *B12 = initializeMatrix1D(k), *B21 = initializeMatrix1D(k), *B22 = initializeMatrix1D(k);

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            A11[i * k + j] = A[i * n + j];
            A12[i * k + j] = A[i * n + j + k];
            A21[i * k + j] = A[(i + k) * n + j];
            A22[i * k + j] = A[(i + k) * n + j + k];
            B11[i * k + j] = B[i * n + j];
            B12[i * k + j] = B[i * n + j + k];
            B21[i * k + j] = B[(i + k) * n + j];
            B22[i * k + j] = B[(i + k) * n + j + k];
        }
    }

    double *S1 = initializeMatrix1D(k), *S2 = initializeMatrix1D(k);
    subtractMatrix(B12, B22, S1, k);
    double* P1 = strassenMultiplyUnrolled(A11, S1, k);
    addMatrix(A11, A12, S1, k);
    double* P2 = strassenMultiplyUnrolled(S1, B22, k);
    addMatrix(A21, A22, S1, k);
    double* P3 = strassenMultiplyUnrolled(S1, B11, k);
    subtractMatrix(B21, B11, S1, k);
    double* P4 = strassenMultiplyUnrolled(A22, S1, k);
    addMatrix(A11, A22, S1, k);
    addMatrix(B11, B22, S2, k);
    double* P5 = strassenMultiplyUnrolled(S1, S2, k);
    subtractMatrix(A12, A22, S1, k);
    addMatrix(B21, B22, S2, k);
    double* P6 = strassenMultiplyUnrolled(S1, S2, k);
    subtractMatrix(A11, A21, S1, k);
    addMatrix(B11, B12, S2, k);
    double* P7 = strassenMultiplyUnrolled(S1, S2, k);

    double *C11 = initializeMatrix1D(k), *C12 = initializeMatrix1D(k), *C21 = initializeMatrix1D(k), *C22 = initializeMatrix1D(k);
    for (int i = 0; i < size; i++) {
        C11[i] = P5[i] + P4[i] - P2[i] + P6[i];
        C12[i] = P1[i] + P2[i];
        C21[i] = P3[i] + P4[i];
        C22[i] = P5[i] + P1[i] - P3[i] - P7[i];
    }

    double* C = initializeMatrix1D(n);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            C[i * n + j] = C11[i * k + j];
            C[i * n + j + k] = C12[i * k + j];
            C[(i + k) * n + j] = C21[i * k + j];
            C[(i + k) * n + j + k] = C22[i * k + j];
        }
    }

    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(P1); free(P2); free(P3); free(P4);
    free(P5); free(P6); free(P7);
    free(C11); free(C12); free(C21); free(C22);
    free(S1); free(S2);

    return C;
}

double* mm_strassen_unrolled(int nthreads, int rows, int cols_a, int cols_b) {
    omp_set_num_threads(nthreads);

    double *A = (double *)malloc(rows * cols_a * sizeof(double));
    double *B = (double *)malloc(cols_a * cols_b * sizeof(double));
    double *C = (double *)malloc(rows * cols_b * sizeof(double));

    FILE *fa = fopen("Data/matrix_a.bin", "rb");
    FILE *fb = fopen("Data/matrix_b.bin", "rb");

    fread(A, sizeof(double), rows * cols_a, fa);
    fread(B, sizeof(double), cols_a * cols_b, fb);

    fclose(fa);
    fclose(fb);

    memset(C, 0, rows * cols_b * sizeof(double));

    int size = 1;
    while (size < rows || size < cols_a || size < cols_b) size <<= 1;

    double* A_mat = initializeMatrix1D(size);
    double* B_mat = initializeMatrix1D(size);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols_a; j++)
            A_mat[i * size + j] = A[i * cols_a + j];

    for (int i = 0; i < cols_a; i++)
        for (int j = 0; j < cols_b; j++)
            B_mat[i * size + j] = B[i * cols_b + j];

    double start_time = omp_get_wtime();
    double* C_mat = strassenMultiplyUnrolled(A_mat, B_mat, size);
    double elapsed_time = omp_get_wtime() - start_time;

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols_b; j++)
            C[i * cols_b + j] = C_mat[i * size + j];

    FILE *ftime = fopen("Values/time_strassen_unrolled.txt", "a");
    fprintf(ftime, "%f\n", elapsed_time);
    fclose(ftime);

    free(A);
    free(B);
    free(A_mat);
    free(B_mat);
    free(C_mat);
    return C;
}
