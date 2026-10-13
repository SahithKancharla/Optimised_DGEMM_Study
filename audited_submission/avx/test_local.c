#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mm_malloc.h>

/* Correctness check for matmul() (avx/matmul.c) against a naive reference,
 * across a range of sizes including several that are not multiples of the
 * MR=24 / NR=6 micro-kernel tile (to exercise the masked/partial-tile
 * path). Link against either kernel.c (AVX-512) or kernel_avx2.c (AVX2) --
 * see run_local_avx2.sh. Deliberately uses asymmetric A/B so that a
 * transpose or operand-swap bug cannot hide behind symmetric test data.
 * Also exercises strassen() (avx/strassen.c or avx/strassen_avx2.c) the
 * same way -- see run_strassen() below. */

void matmul(double *A, double *B, double *C, int N);
void strassen(double *A, double *B, double *C, int N);

#define MEMALIGN 64

static int run(int N) {
    double *A = malloc((size_t)N * N * sizeof(double));
    double *B = malloc((size_t)N * N * sizeof(double));
    // kernel_24x6 (via matmul()) reads/writes this C tile directly with
    // ALIGNED AVX loads/stores when linked against kernel.c (AVX-512) --
    // plain malloc() only guarantees 16-byte alignment, not the 64 that
    // needs, so allocate it the same way matmul()'s real callers do.
    double *C = _mm_malloc((size_t)N * N * sizeof(double), MEMALIGN);
    double *Ref = calloc((size_t)N * N, sizeof(double));
    if (!A || !B || !C || !Ref) {
        fprintf(stderr, "allocation failed for N=%d\n", N);
        free(A); free(B); _mm_free(C); free(Ref);
        return 1;
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i % 7) * 3.0 + (j % 5) - 2.0;
            B[i * N + j] = (j % 11) - 2.0 * (i % 3) + 0.5;
        }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++) s += A[i * N + k] * B[k * N + j];
            Ref[i * N + j] = s;
        }

    matmul(A, B, C, N);

    double maxdiff = 0.0;
    int mism = 0;
    for (int i = 0; i < N * N; i++) {
        double d = fabs(C[i] - Ref[i]);
        if (d > maxdiff) maxdiff = d;
        if (d > 1e-6) mism++;
    }
    printf("N=%-6d mismatches=%d/%d maxdiff=%g  %s\n",
           N, mism, N * N, maxdiff, mism ? "FAIL" : "PASS");

    free(A); free(B); _mm_free(C); free(Ref);
    return mism != 0;
}

/* Same correctness check as run(), but against strassen() instead of
 * matmul(). strassen()'s base case is a register-blocked matmul just like
 * kernel_24x6's, and once N folds down (via strassen()'s own next-power-
 * of-two padding) to a size <= CUTOFF that's already a power of two, it is
 * called directly on THESE buffers with the same aligned-load requirement
 * -- so A/B/C all need to be 64-byte aligned here too, not just C. */
static int run_strassen(int N) {
    double *A = _mm_malloc((size_t)N * N * sizeof(double), MEMALIGN);
    double *B = _mm_malloc((size_t)N * N * sizeof(double), MEMALIGN);
    double *C = _mm_malloc((size_t)N * N * sizeof(double), MEMALIGN);
    double *Ref = calloc((size_t)N * N, sizeof(double));
    if (!A || !B || !C || !Ref) {
        fprintf(stderr, "allocation failed for N=%d\n", N);
        _mm_free(A); _mm_free(B); _mm_free(C); free(Ref);
        return 1;
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i % 7) * 3.0 + (j % 5) - 2.0;
            B[i * N + j] = (j % 11) - 2.0 * (i % 3) + 0.5;
        }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++) s += A[i * N + k] * B[k * N + j];
            Ref[i * N + j] = s;
        }

    strassen(A, B, C, N);

    double maxdiff = 0.0;
    int mism = 0;
    for (int i = 0; i < N * N; i++) {
        double d = fabs(C[i] - Ref[i]);
        if (d > maxdiff) maxdiff = d;
        if (d > 1e-6) mism++;
    }
    printf("strassen N=%-6d mismatches=%d/%d maxdiff=%g  %s\n",
           N, mism, N * N, maxdiff, mism ? "FAIL" : "PASS");

    _mm_free(A); _mm_free(B); _mm_free(C); free(Ref);
    return mism != 0;
}

int main(void) {
    int sizes[] = {1, 2, 5, 17, 23, 24, 25, 29, 47, 48, 49, 100, 127, 128, 129, 250, 517};
    int bad = 0;
    for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) bad |= run(sizes[i]);
    for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) bad |= run_strassen(sizes[i]);
    if (!bad) printf("ALL PASS\n");
    return bad;
}
