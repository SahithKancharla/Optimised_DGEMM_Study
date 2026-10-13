#ifndef STRASSEN_H
#define STRASSEN_H

/* Rebuilt Strassen: pooled workspace (one batch allocation per recursion
 * level, not 17 separate mallocs and definitely not the thousands-to-
 * millions of small allocations the ORIGINAL avx/strassen.c produces --
 * see FINDINGS.md Part C/E for that implementation's allocator-pressure
 * problem), fused adds (bare quadrants passed as strided views straight
 * into dgemm_base, never copied; only genuinely combined operands are
 * materialized), a packed-DGEMM base case (dgemm_base.c, our tuned
 * hand-asm kernel), and a thread-aware task/sequential cutoff (see
 * strassen.c for the policy). n must be even (odd n is handled by the
 * caller padding, matching the correctness fix already applied to
 * avx/strassen.c -- see FINDINGS.md Part C bug #3).
 *
 * strassen_1level: one level of the 7-multiplication decomposition, base
 * case (each n/2 x n/2 subproblem) computed directly by dgemm_base.
 * strassen_2level: recurses one further level before hitting dgemm_base
 * (49 leaf multiplications of size n/4 instead of 7 of size n/2). */
void strassen_1level(const double *A, int ldA, const double *B, int ldB,
                      double *C, int ldC, int n);
void strassen_2level(const double *A, int ldA, const double *B, int ldB,
                      double *C, int ldC, int n);

#endif
