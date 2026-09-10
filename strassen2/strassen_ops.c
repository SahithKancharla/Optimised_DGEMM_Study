#include "strassen_ops.h"
#include <immintrin.h>
#include <omp.h>

void add_view(double *dst, const double *A, int ldA, const double *B, int ldB, int n) {
#pragma omp parallel for schedule(static)
    for (int col = 0; col < n; col++) {
        const double *a = A + (size_t)col * ldA, *b = B + (size_t)col * ldB;
        double *d = dst + (size_t)col * n;
        int i = 0;
        for (; i + 3 < n; i += 4)
            _mm256_storeu_pd(d + i, _mm256_add_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
        for (; i < n; i++) d[i] = a[i] + b[i];
    }
}

void sub_view(double *dst, const double *A, int ldA, const double *B, int ldB, int n) {
#pragma omp parallel for schedule(static)
    for (int col = 0; col < n; col++) {
        const double *a = A + (size_t)col * ldA, *b = B + (size_t)col * ldB;
        double *d = dst + (size_t)col * n;
        int i = 0;
        for (; i + 3 < n; i += 4)
            _mm256_storeu_pd(d + i, _mm256_sub_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
        for (; i < n; i++) d[i] = a[i] - b[i];
    }
}

void accum_add_view(double *dst, int ldDst, const double *P, int n) {
#pragma omp parallel for schedule(static)
    for (int col = 0; col < n; col++) {
        double *d = dst + (size_t)col * ldDst;
        const double *p = P + (size_t)col * n;
        int i = 0;
        for (; i + 3 < n; i += 4)
            _mm256_storeu_pd(d + i, _mm256_add_pd(_mm256_loadu_pd(d + i), _mm256_loadu_pd(p + i)));
        for (; i < n; i++) d[i] += p[i];
    }
}

void copy_to_view(double *dst, int ldDst, const double *P, int n) {
#pragma omp parallel for schedule(static)
    for (int col = 0; col < n; col++) {
        double *d = dst + (size_t)col * ldDst;
        const double *p = P + (size_t)col * n;
        int i = 0;
        for (; i + 3 < n; i += 4) _mm256_storeu_pd(d + i, _mm256_loadu_pd(p + i));
        for (; i < n; i++) d[i] = p[i];
    }
}

void accum_sub_view(double *dst, int ldDst, const double *P, int n) {
#pragma omp parallel for schedule(static)
    for (int col = 0; col < n; col++) {
        double *d = dst + (size_t)col * ldDst;
        const double *p = P + (size_t)col * n;
        int i = 0;
        for (; i + 3 < n; i += 4)
            _mm256_storeu_pd(d + i, _mm256_sub_pd(_mm256_loadu_pd(d + i), _mm256_loadu_pd(p + i)));
        for (; i < n; i++) d[i] -= p[i];
    }
}
