#ifndef DGEMM_BASE_H
#define DGEMM_BASE_H

/* Reusable library entry point for our tuned packed AVX2 DGEMM (hand-
 * written assembly microkernel + transpose-based packing, the same design
 * as avx2/asm_kernel/matmul_asm_v3_transpack.c), extracted here as a
 * callable function instead of a benchmark driver with its own main().
 *
 * Generalized to accept independent leading dimensions for A, B, and C
 * (ldA/ldB/ldC), not just square N x N contiguous buffers -- this is what
 * lets the Strassen rebuild (strassen2/) operate directly on sub-matrix
 * VIEWS of the original A/B/C (e.g. the A11 quadrant, ld = original N)
 * without copying them into contiguous scratch first. Only the quadrants
 * that are genuinely combined (added/subtracted) need to be materialized
 * into a scratch buffer; bare quadrants used as-is in the classic Strassen
 * formulation (e.g. A11 alone in M3, B11 alone in M2) are passed straight
 * through as strided views -- the "fused" part of "fused matrix additions
 * where possible". Computes C = A*B (overwrites C, does not accumulate). */
void dgemm_base(const double *A, int ldA, const double *B, int ldB,
                 double *C, int ldC, int n);

#endif
