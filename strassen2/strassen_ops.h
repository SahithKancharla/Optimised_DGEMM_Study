#ifndef STRASSEN_OPS_H
#define STRASSEN_OPS_H
#include <stddef.h>

/* dst[i,j] = A[i,j] + B[i,j] (add_view) or A[i,j] - B[i,j] (sub_view), for
 * an n x n block where A/B are strided VIEWS (leading dimensions ldA/ldB,
 * e.g. a quadrant of a larger matrix) and dst is a freshly allocated
 * contiguous n x n buffer (ld = n). This is the "fused addition" step:
 * only combined (summed/subtracted) operands are ever materialized this
 * way; quadrants used bare in the classic Strassen formulation (e.g. A11
 * alone) are passed straight to dgemm_base as strided views and never
 * touch this function at all. */
void add_view(double *dst, const double *A, int ldA, const double *B, int ldB, int n);
void sub_view(double *dst, const double *A, int ldA, const double *B, int ldB, int n);

/* dst[i,j] (a view with leading dimension ldDst, e.g. a C quadrant) +=
 * or -= P[i,j] (contiguous, ld = n) -- used for the final combination
 * step (C11 = M1+M4-M5+M7 etc.), accumulating directly into the caller's
 * output matrix without a separate temporary. */
void accum_add_view(double *dst, int ldDst, const double *P, int n);
void accum_sub_view(double *dst, int ldDst, const double *P, int n);

/* dst[i,j] = P[i,j]: the first term of a quadrant combination must SET,
 * not accumulate into, the (uninitialized) output view. */
void copy_to_view(double *dst, int ldDst, const double *P, int n);

#endif
