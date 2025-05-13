#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static double* initializeMatrix1D_n(int n) {
    return (double*)calloc(n * n, sizeof(double));
}

static void addMatrix_N(double* A, double* B, double* C, int n) {
    for (int i = 0; i < n * n; i++)
        C[i] = A[i] + B[i];
}

static void subtractMatrix_N(double* A, double* B, double* C, int n) {
    for (int i = 0; i < n * n; i++)
        C[i] = A[i] - B[i];
}

double* strassenMultiply_P(double* A, double* B, int n, int cutoff) {
    // base case: 2×2 Strassen
    if (n == 2) {
        double* C = initializeMatrix1D_n(n);
        double a = A[0], b = A[1], c = A[2], d = A[3];
        double e = B[0], f = B[1], g = B[2], h = B[3];

        double p1 = a * (f - h);
        double p2 = (a + b) * h;
        double p3 = (c + d) * e;
        double p4 = d * (g - e);
        double p5 = (a + d) * (e + h);
        double p6 = (b - d) * (g + h);
        double p7 = (a - c) * (e + f);

        C[0] = p5 + p4 - p2 + p6;
        C[1] = p1 + p2;
        C[2] = p3 + p4;
        C[3] = p5 + p1 - p3 - p7;
        return C;
    }

    int k = n / 2, size = k * k;
    // allocate and split A, B into quarters
    double *A11 = initializeMatrix1D_n(k), *A12 = initializeMatrix1D_n(k),
           *A21 = initializeMatrix1D_n(k), *A22 = initializeMatrix1D_n(k);
    double *B11 = initializeMatrix1D_n(k), *B12 = initializeMatrix1D_n(k),
           *B21 = initializeMatrix1D_n(k), *B22 = initializeMatrix1D_n(k);

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            A11[i*k + j] = A[i*n + j];
            A12[i*k + j] = A[i*n + j + k];
            A21[i*k + j] = A[(i+k)*n + j];
            A22[i*k + j] = A[(i+k)*n + j + k];
            B11[i*k + j] = B[i*n + j];
            B12[i*k + j] = B[i*n + j + k];
            B21[i*k + j] = B[(i+k)*n + j];
            B22[i*k + j] = B[(i+k)*n + j + k];
        }
    }

    double *P1, *P2, *P3, *P4, *P5, *P6, *P7;

    #pragma omp task shared(P1) if (n > cutoff)
    {
        double* S = initializeMatrix1D_n(k);
        subtractMatrix_N(B12, B22, S, k);
        P1 = strassenMultiply_P(A11, S, k, cutoff);
        free(S);
    }
    #pragma omp task shared(P2) if (n > cutoff)
    {
        double* S = initializeMatrix1D_n(k);
        addMatrix_N(A11, A12, S, k);
        P2 = strassenMultiply_P(S, B22, k, cutoff);
        free(S);
    }
    #pragma omp task shared(P3) if (n > cutoff)
    {
        double* S = initializeMatrix1D_n(k);
        addMatrix_N(A21, A22, S, k);
        P3 = strassenMultiply_P(S, B11, k, cutoff);
        free(S);
    }
    #pragma omp task shared(P4) if (n > cutoff)
    {
        double* S = initializeMatrix1D_n(k);
        subtractMatrix_N(B21, B11, S, k);
        P4 = strassenMultiply_P(A22, S, k, cutoff);
        free(S);
    }
    #pragma omp task shared(P5) if (n > cutoff)
    {
        double* S1 = initializeMatrix1D_n(k);
        double* S2 = initializeMatrix1D_n(k);
        addMatrix_N(A11, A22, S1, k);
        addMatrix_N(B11, B22, S2, k);
        P5 = strassenMultiply_P(S1, S2, k, cutoff);
        free(S1); free(S2);
    }
    #pragma omp task shared(P6) if (n > cutoff)
    {
        double* S1 = initializeMatrix1D_n(k);
        double* S2 = initializeMatrix1D_n(k);
        subtractMatrix_N(A12, A22, S1, k);
        addMatrix_N(B21, B22, S2, k);
        P6 = strassenMultiply_P(S1, S2, k, cutoff);
        free(S1); free(S2);
    }
    #pragma omp task shared(P7) if (n > cutoff)
    {
        double* S1 = initializeMatrix1D_n(k);
        double* S2 = initializeMatrix1D_n(k);
        subtractMatrix_N(A11, A21, S1, k);
        addMatrix_N(B11, B12, S2, k);
        P7 = strassenMultiply_P(S1, S2, k, cutoff);
        free(S1); free(S2);
    }

    #pragma omp taskwait

    // build the four result quadrants
    double *C11 = initializeMatrix1D_n(k), *C12 = initializeMatrix1D_n(k),
           *C21 = initializeMatrix1D_n(k), *C22 = initializeMatrix1D_n(k);
    for (int i = 0; i < size; i++) {
        C11[i] = P5[i] + P4[i] - P2[i] + P6[i];
        C12[i] = P1[i] + P2[i];
        C21[i] = P3[i] + P4[i];
        C22[i] = P5[i] + P1[i] - P3[i] - P7[i];
    }

    // stitch them into one n×n matrix
    double* C = initializeMatrix1D_n(n);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            C[i*n + j]       = C11[i*k + j];
            C[i*n + j + k]   = C12[i*k + j];
            C[(i+k)*n + j]   = C21[i*k + j];
            C[(i+k)*n + j + k] = C22[i*k + j];
        }
    }

    // clean up
    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(P1); free(P2); free(P3); free(P4);
    free(P5); free(P6); free(P7);
    free(C11); free(C12); free(C21); free(C22);

    return C;
}

double* mm_parallel_strassen_naive(int nthreads, int rows, int cols_a, int cols_b) {
    omp_set_num_threads(nthreads);
    omp_set_nested(0);
    omp_set_max_active_levels(1);

    // load A and B
    double *A = malloc(rows * cols_a * sizeof(double));
    double *B = malloc(cols_a * cols_b * sizeof(double));
    double *C = malloc(rows * cols_b * sizeof(double));
    FILE *fa = fopen("Data/matrix_a.bin", "rb");
    FILE *fb = fopen("Data/matrix_b.bin", "rb");
    fread(A, sizeof(double), rows*cols_a, fa);
    fread(B, sizeof(double), cols_a*cols_b, fb);
    fclose(fa); fclose(fb);
    memset(C, 0, rows * cols_b * sizeof(double));

    // pad up to next power‐of‐two
    int size = 1;
    while (size < rows || size < cols_a || size < cols_b) size <<= 1;
    double *A_mat = initializeMatrix1D_n(size);
    double *B_mat = initializeMatrix1D_n(size);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols_a; j++)
            A_mat[i*size + j] = A[i*cols_a + j];
    for (int i = 0; i < cols_a; i++)
        for (int j = 0; j < cols_b; j++)
            B_mat[i*size + j] = B[i*cols_b + j];

    // single parallel region + single initial task
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            double* C_mat = strassenMultiply_P(A_mat, B_mat, size, 128);
            // copy back into C
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols_b; j++)
                    C[i*cols_b + j] = C_mat[i*size + j];
            free(C_mat);
        }
    }
    double elapsed = omp_get_wtime() - start;

    // log time
    FILE *ftime = fopen("Values/parallel.txt", "a");
    fprintf(ftime, "%f\n", elapsed);
    fclose(ftime);

    free(A); free(B); free(A_mat); free(B_mat);
    return C;
}
