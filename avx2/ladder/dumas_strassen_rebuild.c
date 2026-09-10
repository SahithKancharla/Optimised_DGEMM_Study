/*
 * dumas_strassen_rebuild.c -- Dumas, Pernet & Sedoglavic's 2024
 * accuracy-optimized alternate-basis 2x2 (7-multiplication) matrix
 * multiplication algorithm ("Strassen's algorithm is not optimally
 * accurate", ISSAC 2024, arXiv:2402.05630, Equation 36 / Table 3),
 * rebuilt as a recursive block algorithm on top of the exact same
 * pooled-arena / phase-timed / OpenMP-task infrastructure as
 * strassen_rebuild.c (Part J.7's classic-Strassen rebuild), so that
 * the *only* variable between the two is the 7-multiplication formula
 * itself -- everything else (leaf kernel, packing, blocking, workspace
 * strategy, task-parallelism, phase timers) is identical. This is
 * FINDINGS.md Part L.4's highest-priority follow-up experiment: as of
 * the 2026-09-09 literature search behind Part L, nobody had yet run
 * this specific 2024 algorithm through a real, hardware-measured,
 * packed/blocked, statistically-repeated GEMM benchmark -- Dumas et
 * al.'s own paper reports only asymptotic operation counts and a
 * numerical-accuracy plot, no wall-clock/phase-timed performance data.
 *
 * WHY THIS FORMULA, EXACTLY (do not "simplify" the constants below
 * without re-deriving): the paper's own subscript convention for a_ij
 * puts a12 where standard row-major indexing puts a21 and vice versa
 * (confirmed by symbolic verification against Table 3's straight-line
 * program with sympy -- see the commit/session notes; the raw formula
 * transcribed literally against row-major A/B produces WRONG results,
 * off by exactly the antisymmetric a12<->a21, b12<->b21 terms). The
 * formula below has already been adjusted for that and was verified,
 * symbolically (sympy.simplify to exactly 0 for all four C blocks) and
 * numerically, to equal standard C = A*B for arbitrary 2x2 blocks
 * before being transcribed here. If this file is ever edited, that
 * verification must be redone -- a sign error here would silently
 * corrupt the accuracy study this exists to run.
 *
 * The algorithm is stated by its authors to be non-commutative, which
 * is exactly the property that makes a scalar 7-multiplication formula
 * safe to reuse at the *block* (matrix) level via recursion the way
 * classic Strassen's is -- the "multiplications" below are recursive
 * calls (or a leaf GEMM), and the constant-times-block terms (sqrt(3)/3
 * etc.) are elementwise scalar-times-matrix scaling, not scalar algebra
 * that would only hold for actual scalars.
 *
 * Square N x N only, same convention as strassen_rebuild.c: N must
 * already be a power of two down to cutoff (caller's responsibility).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "timer.h"
#define MATMUL_LADDER2_AS_LIBRARY
#include "matmul_ladder2.c"

#define DUMAS_SQRT3 1.7320508075688772935274463415059

typedef struct {
    double leaf_s, addsub_s, combine_s, workspace_s, sched_s, total_s;
    long leaf_calls, addsub_calls;
} dumas_times_t;

typedef struct {
    double *base;
    size_t used, cap;
} dumas_arena_t;

static double *dumas_arena_alloc(dumas_arena_t *a, size_t n_doubles, dumas_times_t *st) {
    double tw0 = ladder_now();
    if (a->used + n_doubles > a->cap) {
        fprintf(stderr, "dumas arena overflow: used=%zu + req=%zu > cap=%zu\n", a->used, n_doubles, a->cap);
        exit(1);
    }
    double *p = a->base + a->used;
    a->used += n_doubles;
    #pragma omp atomic
    st->workspace_s += ladder_now() - tw0;
    return p;
}

/* C = alpha*A + beta*B, elementwise over an n x n block. Covers every
   add/sub/scale in the SLP below (scale_mat is axpby with beta=0 and B
   unused; classic add/sub is axpby with alpha=beta=+-1). */
static void dumas_axpby(double alpha, const double *A, double beta, const double *B,
                         double *C, int n, dumas_times_t *st) {
    double t0 = ladder_now();
    size_t N2 = (size_t)n * n;
    for (size_t i = 0; i < N2; i++) C[i] = alpha*A[i] + beta*B[i];
    double dt = ladder_now() - t0;
    #pragma omp atomic
    st->addsub_s += dt;
    #pragma omp atomic
    st->addsub_calls += 1;
}
static void dumas_scale(const double *A, double alpha, double *C, int n, dumas_times_t *st) {
    double t0 = ladder_now();
    size_t N2 = (size_t)n * n;
    for (size_t i = 0; i < N2; i++) C[i] = alpha*A[i];
    double dt = ladder_now() - t0;
    #pragma omp atomic
    st->addsub_s += dt;
    #pragma omp atomic
    st->addsub_calls += 1;
}
static void dumas_copy_block(const double *src, int N, int row0, int col0, double *dst, int n) {
    for (int i = 0; i < n; ++i)
        memcpy(dst + (size_t)i*n, src + (size_t)(row0+i)*N + col0, n*sizeof(double));
}

static void dumas_leaf_gemm(const double *A, const double *B, double *C, int n,
                             int KC, int MC, int NC, dumas_times_t *st) {
    double t0 = ladder_now();
    phase_times_t pt;
    matmul_full(A, B, C, n, n, n, KC, MC, NC, &pt);
    double dt = ladder_now() - t0;
    #pragma omp atomic
    st->leaf_s += dt;
    #pragma omp atomic
    st->leaf_calls += 1;
}

static int dumas_g_cutoff, dumas_g_max_depth, dumas_g_KC, dumas_g_MC, dumas_g_NC;
static size_t dumas_workspace_doubles_needed(int N, int cutoff, int max_depth);

static void dumas_rec(const double *A, const double *B, double *C, int N,
                       int depth, dumas_arena_t *arena, dumas_times_t *st) {
    if (N <= dumas_g_cutoff || depth >= dumas_g_max_depth) {
        dumas_leaf_gemm(A, B, C, N, dumas_g_KC, dumas_g_MC, dumas_g_NC, st);
        return;
    }
    int k = N / 2, size = k * k;
    const double SQ3 = DUMAS_SQRT3; /* named SQ3, not S3 -- S3 is already the s3 buffer pointer below */

    double *A11 = dumas_arena_alloc(arena, size, st), *A12 = dumas_arena_alloc(arena, size, st),
           *A21 = dumas_arena_alloc(arena, size, st), *A22 = dumas_arena_alloc(arena, size, st),
           *B11 = dumas_arena_alloc(arena, size, st), *B12 = dumas_arena_alloc(arena, size, st),
           *B21 = dumas_arena_alloc(arena, size, st), *B22 = dumas_arena_alloc(arena, size, st);

    double tcp0 = ladder_now();
    dumas_copy_block(A, N, 0, 0, A11, k); dumas_copy_block(A, N, 0, k, A12, k);
    dumas_copy_block(A, N, k, 0, A21, k); dumas_copy_block(A, N, k, k, A22, k);
    dumas_copy_block(B, N, 0, 0, B11, k); dumas_copy_block(B, N, 0, k, B12, k);
    dumas_copy_block(B, N, k, 0, B21, k); dumas_copy_block(B, N, k, k, B22, k);
    #pragma omp atomic
    st->addsub_s += ladder_now() - tcp0;

    /* Table 3's slp (24 add., 12 mul./div.), adapted from the paper's
       a_ij/b_ij subscript convention to standard row-major indexing --
       see the file header. T1..T3/S1..S3 are shared intermediates; L3
       and R2 are pure aliases of T2/S2 in the paper's own slp (no extra
       buffer or copy needed for them). */
    double *T1 = dumas_arena_alloc(arena, size, st), *T2 = dumas_arena_alloc(arena, size, st),
           *T3 = dumas_arena_alloc(arena, size, st);
    double *S1 = dumas_arena_alloc(arena, size, st), *S2 = dumas_arena_alloc(arena, size, st),
           *S3 = dumas_arena_alloc(arena, size, st);
    double *L1 = dumas_arena_alloc(arena, size, st), *L2 = dumas_arena_alloc(arena, size, st),
           *L4 = dumas_arena_alloc(arena, size, st), *L5 = dumas_arena_alloc(arena, size, st),
           *L6 = dumas_arena_alloc(arena, size, st), *L7 = dumas_arena_alloc(arena, size, st);
    double *R1 = dumas_arena_alloc(arena, size, st), *R3 = dumas_arena_alloc(arena, size, st),
           *R4 = dumas_arena_alloc(arena, size, st), *R5 = dumas_arena_alloc(arena, size, st),
           *R6 = dumas_arena_alloc(arena, size, st), *R7 = dumas_arena_alloc(arena, size, st);
    double *P1 = dumas_arena_alloc(arena, size, st), *P2 = dumas_arena_alloc(arena, size, st),
           *P3 = dumas_arena_alloc(arena, size, st), *P4 = dumas_arena_alloc(arena, size, st),
           *P5 = dumas_arena_alloc(arena, size, st), *P6 = dumas_arena_alloc(arena, size, st),
           *P7 = dumas_arena_alloc(arena, size, st);
    double *L3 = T2, *R2 = S2; /* aliases, not separate allocations */

    double tsc0 = ladder_now();
    dumas_scale(A22, SQ3/3.0, T1, k, st);              /* t1 = (sqrt3/3) A22 */
    dumas_axpby(1.0, A12, 1.0, T1, T2, k, st);         /* t2 = A12 + t1 */
    dumas_axpby(1.0, A21, 1.0, T2, T3, k, st);         /* t3 = A21 + t2 */
    dumas_scale(B12, SQ3/3.0, S1, k, st);              /* s1 = (sqrt3/3) B12 */
    dumas_axpby(1.0, S1, -1.0, B11, S2, k, st);        /* s2 = s1 - B11 */
    dumas_axpby(1.0, S2, 1.0, B22, S3, k, st);         /* s3 = s2 + B22 (S3 is the s3 buffer) */

    dumas_axpby(SQ3/2.0, A11, 0.5, T3, L1, k, st);     /* l1 = (sqrt3/2) A11 + 0.5 t3 */
    dumas_axpby(1.0, A21, -1.0, T1, L2, k, st);        /* l2 = A21 - t1 */
    dumas_scale(T1, 2.0, L4, k, st);                   /* l4 = 2 t1 */
    dumas_axpby(1.0, L2, -1.0, L1, L5, k, st);         /* l5 = l2 - l1 */
    dumas_axpby(1.0, L5, 1.0, L4, L6, k, st);          /* l6 = l5 + l4 */
    dumas_axpby(1.0, L5, 1.0, L3, L7, k, st);          /* l7 = l5 + l3 */

    dumas_scale(S1, 2.0, R1, k, st);                   /* r1 = 2 s1 */
    dumas_axpby(1.0, S1, -1.0, B22, R3, k, st);        /* r3 = s1 - B22 */
    dumas_axpby(0.5, S3, -SQ3/2.0, B21, R4, k, st);    /* r4 = 0.5 s3(buffer) - (sqrt3/2) B21 */
    dumas_axpby(1.0, R3, 1.0, R4, R5, k, st);          /* r5 = r3 + r4 */
    dumas_axpby(1.0, R1, -1.0, R5, R6, k, st);         /* r6 = r1 - r5 */
    dumas_axpby(1.0, R5, -1.0, R2, R7, k, st);         /* r7 = r5 - r2 */

    /* Each of the 7 (L_i, R_i) pairs needs its own disjoint sub-arena
       for genuinely concurrent task execution -- identical reasoning
       and identical fix to strassen_rebuild.c's arena-aliasing bug
       (see that file's comment; caught there before ever running). */
    size_t child_ws = dumas_workspace_doubles_needed(k, dumas_g_cutoff, dumas_g_max_depth - depth - 1);
    double *sub_base[7];
    for (int b = 0; b < 7; b++) sub_base[b] = dumas_arena_alloc(arena, child_ws, st);
    #pragma omp atomic
    st->sched_s += ladder_now() - tsc0;

    double ttask0 = ladder_now();
    #pragma omp task shared(P1) if (N > dumas_g_cutoff*4)
    { dumas_arena_t sub = { sub_base[0], 0, child_ws }; dumas_rec(L1, R1, P1, k, depth+1, &sub, st); }
    #pragma omp task shared(P2) if (N > dumas_g_cutoff*4)
    { dumas_arena_t sub = { sub_base[1], 0, child_ws }; dumas_rec(L2, R2, P2, k, depth+1, &sub, st); }
    #pragma omp task shared(P3) if (N > dumas_g_cutoff*4)
    { dumas_arena_t sub = { sub_base[2], 0, child_ws }; dumas_rec(L3, R3, P3, k, depth+1, &sub, st); }
    #pragma omp task shared(P4) if (N > dumas_g_cutoff*4)
    { dumas_arena_t sub = { sub_base[3], 0, child_ws }; dumas_rec(L4, R4, P4, k, depth+1, &sub, st); }
    #pragma omp task shared(P5) if (N > dumas_g_cutoff*4)
    { dumas_arena_t sub = { sub_base[4], 0, child_ws }; dumas_rec(L5, R5, P5, k, depth+1, &sub, st); }
    #pragma omp task shared(P6) if (N > dumas_g_cutoff*4)
    { dumas_arena_t sub = { sub_base[5], 0, child_ws }; dumas_rec(L6, R6, P6, k, depth+1, &sub, st); }
    #pragma omp task shared(P7) if (N > dumas_g_cutoff*4)
    { dumas_arena_t sub = { sub_base[6], 0, child_ws }; dumas_rec(L7, R7, P7, k, depth+1, &sub, st); }
    #pragma omp taskwait
    #pragma omp atomic
    st->sched_s += ladder_now() - ttask0;

    /* Combine: w2=p5+p1+p6, w1=p7+p6, w3=w2-p2, w5=(p4+w2)/2,
       C12=p1-p3-w5, C21=w3-w5, C22=sqrt3*w5, C11=(sqrt3/3)(w3-C12-2*w1),
       done directly per element (like strassen_rebuild.c's combine) --
       no separate arena buffers needed for w1/w2/w3/w5. */
    double tcomb0 = ladder_now();
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            int ij = i*k+j;
            double p1=P1[ij], p2=P2[ij], p3=P3[ij], p4=P4[ij], p5=P5[ij], p6=P6[ij], p7=P7[ij];
            double w2 = p5 + p1 + p6;
            double w1v = p7 + p6;
            double w3 = w2 - p2;
            double w5 = (p4 + w2) * 0.5;
            double c12 = p1 - p3 - w5;
            double c21 = w3 - w5;
            double c22 = SQ3 * w5;
            double c11 = (SQ3/3.0) * (w3 - c12 - 2.0*w1v);
            C[(size_t)i*N+j]         = c11;
            C[(size_t)i*N+j+k]       = c12;
            C[(size_t)(i+k)*N+j]     = c21;
            C[(size_t)(i+k)*N+j+k]   = c22;
        }
    }
    #pragma omp atomic
    st->combine_s += ladder_now() - tcomb0;
}

/* Per level: 8 quadrant copies + 25 T/S/L/R/P buffers (3+3+6+6+7) = 33
   buffers of size (N/2)^2, same worst-case-7-independent-subtrees
   reasoning as strassen_rebuild.c's workspace_doubles_needed. */
static size_t dumas_workspace_doubles_needed(int N, int cutoff, int max_depth) {
    if (N <= cutoff || max_depth <= 0) return 0;
    int k = N/2;
    size_t here = 33 * (size_t)k * k;
    size_t child = dumas_workspace_doubles_needed(k, cutoff, max_depth - 1);
    return here + 7 * child;
}

double* dumas_strassen_rebuild(const double *A, const double *B, int N, int cutoff, int max_depth,
                                int KC, int MC, int NC, int nthreads, dumas_times_t *st) {
    memset(st, 0, sizeof(*st));
    dumas_g_cutoff = cutoff; dumas_g_max_depth = max_depth;
    dumas_g_KC = KC; dumas_g_MC = MC; dumas_g_NC = NC;

    size_t arena_size = dumas_workspace_doubles_needed(N, cutoff, max_depth) + 1024;
    double *arena_base = aligned_alloc_d(arena_size);
    dumas_arena_t arena = { arena_base, 0, arena_size };

    double *C = aligned_alloc_d((size_t)N*N);

    omp_set_num_threads(nthreads);
    omp_set_max_active_levels(1);
    double t0 = ladder_now();
    #pragma omp parallel
    #pragma omp single nowait
    dumas_rec(A, B, C, N, 0, &arena, st);
    st->total_s = ladder_now() - t0;

    aligned_free_d(arena_base);
    return C;
}
