#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include <x86intrin.h>

#define CUTOFF 128
#define MEMALIGN 64

void matmul(double *A, double *B, double *C, int N);

static inline double* aligned_alloc_mat(int n) {
    double *ptr = (double*) _mm_malloc((size_t)n * n * sizeof(double), MEMALIGN);
    if (!ptr) { perror("aligned_alloc_mat"); exit(EXIT_FAILURE); }
    return ptr;
}

static inline void add_mat(const double *A, const double *B, double *C, int n) {
    size_t N2 = (size_t)n * n;
    size_t i = 0;
    const double *A_al = __builtin_assume_aligned(A, 64);
    const double *B_al = __builtin_assume_aligned(B, 64);
    double *C_al = __builtin_assume_aligned(C, 64);
    for (; i + 8 <= N2; i += 8) {
        __m512d a = _mm512_load_pd(&A_al[i]);
        __m512d b = _mm512_load_pd(&B_al[i]);
        _mm512_store_pd(&C_al[i], _mm512_add_pd(a, b));
    }
    for (; i < N2; ++i) {
        C[i] = A[i] + B[i];
    }
}

static inline void sub_mat(const double *A, const double *B, double *C, int n) {
    size_t N2 = (size_t)n * n;
    size_t i = 0;
    const double *A_al = __builtin_assume_aligned(A, 64);
    const double *B_al = __builtin_assume_aligned(B, 64);
    double *C_al = __builtin_assume_aligned(C, 64);
    for (; i + 8 <= N2; i += 8) {
        __m512d a = _mm512_load_pd(&A_al[i]);
        __m512d b = _mm512_load_pd(&B_al[i]);
        _mm512_store_pd(&C_al[i], _mm512_sub_pd(a, b));
    }
    for (; i < N2; ++i) {
        C[i] = A[i] - B[i];
    }
}

static void strassen_rec(const double *A, const double *B, double *C, int n) {
    if (n <= CUTOFF) {
        matmul((double*)A, (double*)B, C, n);
        return;
    }

    int k = n / 2;
    int k2 = k * k;

    double *P1 = aligned_alloc_mat(k);
    double *P2 = aligned_alloc_mat(k);
    double *P3 = aligned_alloc_mat(k);
    double *P4 = aligned_alloc_mat(k);
    double *P5 = aligned_alloc_mat(k);
    double *P6 = aligned_alloc_mat(k);
    double *P7 = aligned_alloc_mat(k);

    #pragma omp task shared(P1)
    {
        double *S = aligned_alloc_mat(k);
        sub_mat(B + k, B + k + k2, S, k);
        strassen_rec(A, S, P1, k);
        _mm_free(S);
    }
    #pragma omp task shared(P2)
    {
        double *S = aligned_alloc_mat(k);
        add_mat(A, A + k, S, k);
        strassen_rec(S, B + k + k2, P2, k);
        _mm_free(S);
    }
    #pragma omp task shared(P3)
    {
        double *S = aligned_alloc_mat(k);
        add_mat(A + k2, A + k + k2, S, k);
        strassen_rec(S, B, P3, k);
        _mm_free(S);
    }
    #pragma omp task shared(P4)
    {
        double *S = aligned_alloc_mat(k);
        sub_mat(B + k2, B, S, k);
        strassen_rec(A + k + k2, S, P4, k);
        _mm_free(S);
    }
    #pragma omp task shared(P5)
    {
        double *S1 = aligned_alloc_mat(k), *S2 = aligned_alloc_mat(k);
        add_mat(A, A + k + k2, S1, k);
        add_mat(B, B + k + k2, S2, k);
        strassen_rec(S1, S2, P5, k);
        _mm_free(S1); _mm_free(S2);
    }
    #pragma omp task shared(P6)
    {
        double *S1 = aligned_alloc_mat(k), *S2 = aligned_alloc_mat(k);
        sub_mat(A + k, A + k + k2, S1, k);
        add_mat(B + k2, B + k + k2, S2, k);
        strassen_rec(S1, S2, P6, k);
        _mm_free(S1); _mm_free(S2);
    }
    #pragma omp task shared(P7)
    {
        double *S1 = aligned_alloc_mat(k), *S2 = aligned_alloc_mat(k);
        sub_mat(A, A + k2, S1, k);
        add_mat(B, B + k, S2, k);
        strassen_rec(S1, S2, P7, k);
        _mm_free(S1); _mm_free(S2);
    }

    #pragma omp taskwait

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            int ij = i * k + j;
            C[i*n + j]       = P5[ij] + P4[ij] - P2[ij] + P6[ij];
            C[i*n + j + k]   = P1[ij] + P2[ij];
            C[(i+k)*n + j]   = P3[ij] + P4[ij];
            C[(i+k)*n + j + k] = P5[ij] + P1[ij] - P3[ij] - P7[ij];
        }
    }

    _mm_free(P1); _mm_free(P2); _mm_free(P3); _mm_free(P4);
    _mm_free(P5); _mm_free(P6); _mm_free(P7);
}

void strassen(double *A, double *B, double *C, int n) {
    omp_set_nested(1);
    omp_set_max_active_levels(1);
    #pragma omp parallel
    #pragma omp single nowait
    {
        strassen_rec(A, B, C, n);
    }
}
