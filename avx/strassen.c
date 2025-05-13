#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <mm_malloc.h>

#define CUTOFF 128
#define MEMALIGN 64

// -----------------------------------------------------------------------------
// 16×8 AVX‑512 register‑blocked matmul for N ≤ CUTOFF (assumes N%8==0)
// unroll the k‑loop by 4 to improve throughput
static void matmul(double *A, double *B, double *C, int N) {
    size_t NN = (size_t)N * N;

    double *Ca = __builtin_assume_aligned(C, MEMALIGN);
    // Zero C in parallel
    #pragma omp parallel for
    for (size_t i = 0; i < NN; i += 8) {
        _mm512_store_pd(Ca + i, _mm512_setzero_pd());
    }

    double *Aa = __builtin_assume_aligned(A, MEMALIGN);
    double *Ba = __builtin_assume_aligned(B, MEMALIGN);

    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < N; i0 += 16) {
        for (int j0 = 0; j0 < N; j0 += 8) {
            for (int ii = 0; ii < 16 && i0 + ii < N; ii += 2) {
                double *Arow0 = Aa + (size_t)(i0 + ii) * N;
                double *Cptr0 = Ca + (size_t)(i0 + ii) * N + j0;
                __m512d cvec0 = _mm512_load_pd(Cptr0);

                double *Arow1 = Aa + (size_t)(i0 + ii + 1) * N;
                double *Cptr1 = Ca + (size_t)(i0 + ii + 1) * N + j0;
                __m512d cvec1 = _mm512_load_pd(Cptr1);

                for (int k = 0; k + 3 < N; k += 4) {
                    __m512d vb0 = _mm512_load_pd(Ba + (size_t)(k    ) * N + j0);
                    __m512d vb1 = _mm512_load_pd(Ba + (size_t)(k + 1) * N + j0);
                    __m512d vb2 = _mm512_load_pd(Ba + (size_t)(k + 2) * N + j0);
                    __m512d vb3 = _mm512_load_pd(Ba + (size_t)(k + 3) * N + j0);

                    __m512d va00 = _mm512_set1_pd(Arow0[k]);
                    __m512d va01 = _mm512_set1_pd(Arow0[k + 1]);
                    __m512d va02 = _mm512_set1_pd(Arow0[k + 2]);
                    __m512d va03 = _mm512_set1_pd(Arow0[k + 3]);

                    __m512d va10 = _mm512_set1_pd(Arow1[k]);
                    __m512d va11 = _mm512_set1_pd(Arow1[k + 1]);
                    __m512d va12 = _mm512_set1_pd(Arow1[k + 2]);
                    __m512d va13 = _mm512_set1_pd(Arow1[k + 3]);

                    cvec0 = _mm512_fmadd_pd(va00, vb0, cvec0);
                    cvec0 = _mm512_fmadd_pd(va01, vb1, cvec0);
                    cvec0 = _mm512_fmadd_pd(va02, vb2, cvec0);
                    cvec0 = _mm512_fmadd_pd(va03, vb3, cvec0);

                    cvec1 = _mm512_fmadd_pd(va10, vb0, cvec1);
                    cvec1 = _mm512_fmadd_pd(va11, vb1, cvec1);
                    cvec1 = _mm512_fmadd_pd(va12, vb2, cvec1);
                    cvec1 = _mm512_fmadd_pd(va13, vb3, cvec1);
                }

                // Remainder for k % 4
                for (int k = (N & ~3); k < N; ++k) {
                    __m512d vb = _mm512_load_pd(Ba + (size_t)k * N + j0);
                    __m512d va0 = _mm512_set1_pd(Arow0[k]);
                    __m512d va1 = _mm512_set1_pd(Arow1[k]);

                    cvec0 = _mm512_fmadd_pd(va0, vb, cvec0);
                    cvec1 = _mm512_fmadd_pd(va1, vb, cvec1);
                }

                _mm512_store_pd(Cptr0, cvec0);
                if (i0 + ii + 1 < N)
                    _mm512_store_pd(Cptr1, cvec1);
            }
        }
    }
}


// -----------------------------------------------------------------------------
// the rest is identical scratch‑copy Strassen…

static inline double* aligned_alloc_mat(int n) {
    double *p = _mm_malloc((size_t)n * n * sizeof(double), MEMALIGN);
    if (!p) { perror("alloc"); exit(1); }
    return p;
}

static inline void add_mat(const double *A, const double *B, double *C, int n) {
    size_t N2 = (size_t)n * n, i = 0;
    const double *Aa = __builtin_assume_aligned(A, MEMALIGN),
                  *Ba = __builtin_assume_aligned(B, MEMALIGN);
    double       *Ca = __builtin_assume_aligned(C, MEMALIGN);
    for (; i + 8 <= N2; i += 8) {
        __m512d va = _mm512_load_pd(Aa + i);
        __m512d vb = _mm512_load_pd(Ba + i);
        _mm512_store_pd(Ca + i, _mm512_add_pd(va, vb));
    }
    for (; i < N2; ++i) {
        Ca[i] = Aa[i] + Ba[i];
    }
}

static inline void sub_mat(const double *A, const double *B, double *C, int n) {
    size_t N2 = (size_t)n * n, i = 0;
    const double *Aa = __builtin_assume_aligned(A, MEMALIGN),
                  *Ba = __builtin_assume_aligned(B, MEMALIGN);
    double       *Ca = __builtin_assume_aligned(C, MEMALIGN);
    for (; i + 8 <= N2; i += 8) {
        __m512d va = _mm512_load_pd(Aa + i);
        __m512d vb = _mm512_load_pd(Ba + i);
        _mm512_store_pd(Ca + i, _mm512_sub_pd(va, vb));
    }
    for (; i < N2; ++i) {
        Ca[i] = Aa[i] - Ba[i];
    }
}

static void copy_block(const double *src, int N, int row0, int col0,
                       double *dst, int n)
{
    for (int i = 0; i < n; ++i) {
        memcpy(dst + (size_t)i * n,
               src + (size_t)(row0 + i) * N + col0,
               n * sizeof(double));
    }
}

static void strassen_rec(const double *A, const double *B,
                         double *C, int N)
{
    if (N <= CUTOFF) {
        matmul((double*)A, (double*)B, C, N);
        return;
    }
    int k = N / 2;

    // scratch copies of quadrants
    double *A11 = aligned_alloc_mat(k), *A12 = aligned_alloc_mat(k),
           *A21 = aligned_alloc_mat(k), *A22 = aligned_alloc_mat(k),
           *B11 = aligned_alloc_mat(k), *B12 = aligned_alloc_mat(k),
           *B21 = aligned_alloc_mat(k), *B22 = aligned_alloc_mat(k);

    copy_block(A, N, 0, 0,     A11, k);
    copy_block(A, N, 0, k,     A12, k);
    copy_block(A, N, k, 0,     A21, k);
    copy_block(A, N, k, k,     A22, k);
    copy_block(B, N, 0, 0,     B11, k);
    copy_block(B, N, 0, k,     B12, k);
    copy_block(B, N, k, 0,     B21, k);
    copy_block(B, N, k, k,     B22, k);

    double *P1 = aligned_alloc_mat(k), *P2 = aligned_alloc_mat(k),
           *P3 = aligned_alloc_mat(k), *P4 = aligned_alloc_mat(k),
           *P5 = aligned_alloc_mat(k), *P6 = aligned_alloc_mat(k),
           *P7 = aligned_alloc_mat(k);

    #pragma omp task shared(P1)
    {
        double *S = aligned_alloc_mat(k);
        sub_mat(B12, B22, S, k);
        strassen_rec(A11, S, P1, k);
        _mm_free(S);
    }
    #pragma omp task shared(P2)
    {
        double *S = aligned_alloc_mat(k);
        add_mat(A11, A12, S, k);
        strassen_rec(S, B22, P2, k);
        _mm_free(S);
    }
    #pragma omp task shared(P3)
    {
        double *S = aligned_alloc_mat(k);
        add_mat(A21, A22, S, k);
        strassen_rec(S, B11, P3, k);
        _mm_free(S);
    }
    #pragma omp task shared(P4)
    {
        double *S = aligned_alloc_mat(k);
        sub_mat(B21, B11, S, k);
        strassen_rec(A22, S, P4, k);
        _mm_free(S);
    }
    #pragma omp task shared(P5)
    {
        double *S1 = aligned_alloc_mat(k), *S2 = aligned_alloc_mat(k);
        add_mat(A11, A22, S1, k);
        add_mat(B11, B22, S2, k);
        strassen_rec(S1, S2, P5, k);
        _mm_free(S1); _mm_free(S2);
    }
    #pragma omp task shared(P6)
    {
        double *S1 = aligned_alloc_mat(k), *S2 = aligned_alloc_mat(k);
        sub_mat(A12, A22, S1, k);
        add_mat(B21, B22, S2, k);
        strassen_rec(S1, S2, P6, k);
        _mm_free(S1); _mm_free(S2);
    }
    #pragma omp task shared(P7)
    {
        double *S1 = aligned_alloc_mat(k), *S2 = aligned_alloc_mat(k);
        sub_mat(A11, A21, S1, k);
        add_mat(B11, B12, S2, k);
        strassen_rec(S1, S2, P7, k);
        _mm_free(S1); _mm_free(S2);
    }
    #pragma omp taskwait

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            int ij = i * k + j;
            C[i * N + j]       = P5[ij] + P4[ij] - P2[ij] + P6[ij];
            C[i * N + j + k]   = P1[ij] + P2[ij];
            C[(i + k) * N + j] = P3[ij] + P4[ij];
            C[(i + k) * N + j + k] = 
                P5[ij] + P1[ij] - P3[ij] - P7[ij];
        }
    }

    // free everything
    _mm_free(A11); _mm_free(A12); _mm_free(A21); _mm_free(A22);
    _mm_free(B11); _mm_free(B12); _mm_free(B21); _mm_free(B22);
    _mm_free(P1);  _mm_free(P2);  _mm_free(P3);  _mm_free(P4);
    _mm_free(P5);  _mm_free(P6);  _mm_free(P7);
}

void strassen(double *A, double *B, double *C, int N) {
    omp_set_nested(1);
    omp_set_max_active_levels(1);
    #pragma omp parallel
    #pragma omp single nowait
    strassen_rec(A, B, C, N);
}

// Don’t forget your main(): allocate A/B/C, init A/B, time strassen(), verify C, free.
