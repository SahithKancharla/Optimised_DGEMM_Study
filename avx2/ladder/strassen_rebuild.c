/*
 * strassen_rebuild.c -- Strassen rebuilt on top of the ladder's packed
 * MR=4xNR=8 DGEMM (matmul_full from matmul_ladder2.c, #included directly
 * as the leaf kernel), addressing the review's point 7:
 *   - pooled workspace: one big arena allocated once per top-level call,
 *     carved with a bump-pointer allocator during recursion -- no
 *     malloc/free anywhere in the recursive path.
 *   - configurable depth (stop recursing at max_depth OR at cutoff size,
 *     whichever comes first) and cutoff.
 *   - separately timed phases: leaf GEMMs, add/sub, combine (the final
 *     C11/C12/C21/C22 assembly), workspace bump-alloc bookkeeping,
 *     task creation/sync (OpenMP task overhead).
 *   - cost model: T_total = T_leaf + T_addsub + T_combine + T_workspace +
 *     T_scheduling, reported per run so the crossover vs. plain DGEMM can
 *     be explained, not just observed.
 *
 * Square N x N only (Strassen's split assumes A11..B22 same shape);
 * N padded to the next power of two by the caller if needed (not done
 * here -- caller's responsibility, matches the audited original code's
 * now-fixed convention).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "timer.h"
#define MATMUL_LADDER2_AS_LIBRARY
#include "matmul_ladder2.c"

typedef struct {
    double leaf_s, addsub_s, combine_s, workspace_s, sched_s, total_s;
    long leaf_calls, addsub_calls;
} strassen_times_t;

/* simple bump-pointer arena, reset per top-level call */
typedef struct {
    double *base;
    size_t used, cap;
} arena_t;

static double *arena_alloc(arena_t *a, size_t n_doubles, strassen_times_t *st) {
    double tw0 = ladder_now();
    if (a->used + n_doubles > a->cap) {
        fprintf(stderr, "arena overflow: used=%zu + req=%zu > cap=%zu\n", a->used, n_doubles, a->cap);
        exit(1);
    }
    double *p = a->base + a->used;
    a->used += n_doubles;
    #pragma omp atomic
    st->workspace_s += ladder_now() - tw0;
    return p;
}

static void add_mat(const double *A, const double *B, double *C, int n, strassen_times_t *st) {
    double t0 = ladder_now();
    size_t N2 = (size_t)n * n;
    for (size_t i = 0; i < N2; i++) C[i] = A[i] + B[i];
    double dt = ladder_now() - t0;
    #pragma omp atomic
    st->addsub_s += dt;
    #pragma omp atomic
    st->addsub_calls += 1;
}
static void sub_mat(const double *A, const double *B, double *C, int n, strassen_times_t *st) {
    double t0 = ladder_now();
    size_t N2 = (size_t)n * n;
    for (size_t i = 0; i < N2; i++) C[i] = A[i] - B[i];
    double dt = ladder_now() - t0;
    #pragma omp atomic
    st->addsub_s += dt;
    #pragma omp atomic
    st->addsub_calls += 1;
}
static void copy_block(const double *src, int N, int row0, int col0, double *dst, int n) {
    for (int i = 0; i < n; ++i)
        memcpy(dst + (size_t)i*n, src + (size_t)(row0+i)*N + col0, n*sizeof(double));
}

static void leaf_gemm(const double *A, const double *B, double *C, int n,
                       int KC, int MC, int NC, strassen_times_t *st) {
    double t0 = ladder_now();
    phase_times_t pt;
    matmul_full(A, B, C, n, n, n, KC, MC, NC, &pt);
    double dt = ladder_now() - t0;
    #pragma omp atomic
    st->leaf_s += dt;
    #pragma omp atomic
    st->leaf_calls += 1;
}

static int g_cutoff, g_max_depth, g_KC, g_MC, g_NC;
static size_t workspace_doubles_needed(int N, int cutoff, int max_depth);

static void strassen_rec(const double *A, const double *B, double *C, int N,
                          int depth, arena_t *arena, strassen_times_t *st) {
    if (N <= g_cutoff || depth >= g_max_depth) {
        leaf_gemm(A, B, C, N, g_KC, g_MC, g_NC, st);
        return;
    }
    int k = N / 2, size = k * k;

    double *A11 = arena_alloc(arena, size, st), *A12 = arena_alloc(arena, size, st),
           *A21 = arena_alloc(arena, size, st), *A22 = arena_alloc(arena, size, st),
           *B11 = arena_alloc(arena, size, st), *B12 = arena_alloc(arena, size, st),
           *B21 = arena_alloc(arena, size, st), *B22 = arena_alloc(arena, size, st);

    double tcp0 = ladder_now();
    copy_block(A, N, 0, 0, A11, k); copy_block(A, N, 0, k, A12, k);
    copy_block(A, N, k, 0, A21, k); copy_block(A, N, k, k, A22, k);
    copy_block(B, N, 0, 0, B11, k); copy_block(B, N, 0, k, B12, k);
    copy_block(B, N, k, 0, B21, k); copy_block(B, N, k, k, B22, k);
    #pragma omp atomic
    st->addsub_s += ladder_now() - tcp0; /* copy counted with add/sub -- both are O(n^2) data-movement overhead */

    double *P1 = arena_alloc(arena, size, st), *P2 = arena_alloc(arena, size, st),
           *P3 = arena_alloc(arena, size, st), *P4 = arena_alloc(arena, size, st),
           *P5 = arena_alloc(arena, size, st), *P6 = arena_alloc(arena, size, st),
           *P7 = arena_alloc(arena, size, st);
    double *S1 = arena_alloc(arena, size, st), *S2 = arena_alloc(arena, size, st),
           *S3 = arena_alloc(arena, size, st), *S4 = arena_alloc(arena, size, st),
           *S5 = arena_alloc(arena, size, st), *S6 = arena_alloc(arena, size, st),
           *S7 = arena_alloc(arena, size, st), *S8 = arena_alloc(arena, size, st),
           *S9 = arena_alloc(arena, size, st), *S10 = arena_alloc(arena, size, st);

    double tsc0 = ladder_now();
    sub_mat(B12, B22, S1, k, st);      add_mat(A11, A12, S2, k, st);
    add_mat(A21, A22, S3, k, st);      sub_mat(B21, B11, S4, k, st);
    add_mat(A11, A22, S5, k, st);      add_mat(B11, B22, S6, k, st);
    sub_mat(A12, A22, S7, k, st);      add_mat(B21, B22, S8, k, st);
    sub_mat(A11, A21, S9, k, st);      add_mat(B11, B12, S10, k, st);

    /* Each of the 7 subtrees needs its OWN, non-overlapping slice of the
       arena when they run as genuinely concurrent tasks -- giving them all
       a byte-copy of the SAME arena_t (same base pointer, same starting
       `used` offset) would mean every subtree's bump-pointer allocator
       starts from the identical address, so two tasks running in parallel
       would alias and corrupt each other's workspace. Carve 7 disjoint,
       single-threaded (safe) allocations from the parent arena up front,
       each sized for a full child subtree, before any task is created. */
    size_t child_ws = workspace_doubles_needed(k, g_cutoff, g_max_depth - depth - 1);
    double *sub_base[7];
    for (int b = 0; b < 7; b++) sub_base[b] = arena_alloc(arena, child_ws, st);
    #pragma omp atomic
    st->sched_s += ladder_now() - tsc0;

    /* recursion: 7 subproblems. #pragma omp task per call (depth-limited
       via max_active_levels so we don't explode thread creation, same
       principle as the audited fix in Part C's mm_parallel_strassen_naive.c) */
    double ttask0 = ladder_now();
    #pragma omp task shared(P1) if (N > g_cutoff*4)
    { arena_t sub = { sub_base[0], 0, child_ws }; strassen_rec(A11, S1, P1, k, depth+1, &sub, st); }
    #pragma omp task shared(P2) if (N > g_cutoff*4)
    { arena_t sub = { sub_base[1], 0, child_ws }; strassen_rec(S2, B22, P2, k, depth+1, &sub, st); }
    #pragma omp task shared(P3) if (N > g_cutoff*4)
    { arena_t sub = { sub_base[2], 0, child_ws }; strassen_rec(S3, B11, P3, k, depth+1, &sub, st); }
    #pragma omp task shared(P4) if (N > g_cutoff*4)
    { arena_t sub = { sub_base[3], 0, child_ws }; strassen_rec(A22, S4, P4, k, depth+1, &sub, st); }
    #pragma omp task shared(P5) if (N > g_cutoff*4)
    { arena_t sub = { sub_base[4], 0, child_ws }; strassen_rec(S5, S6, P5, k, depth+1, &sub, st); }
    #pragma omp task shared(P6) if (N > g_cutoff*4)
    { arena_t sub = { sub_base[5], 0, child_ws }; strassen_rec(S7, S8, P6, k, depth+1, &sub, st); }
    #pragma omp task shared(P7) if (N > g_cutoff*4)
    { arena_t sub = { sub_base[6], 0, child_ws }; strassen_rec(S9, S10, P7, k, depth+1, &sub, st); }
    #pragma omp taskwait
    #pragma omp atomic
    st->sched_s += ladder_now() - ttask0;

    double tcomb0 = ladder_now();
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            int ij = i*k+j;
            C[(size_t)i*N+j]         = P5[ij] + P4[ij] - P2[ij] + P6[ij];
            C[(size_t)i*N+j+k]       = P1[ij] + P2[ij];
            C[(size_t)(i+k)*N+j]     = P3[ij] + P4[ij];
            C[(size_t)(i+k)*N+j+k]   = P5[ij] + P1[ij] - P3[ij] - P7[ij];
        }
    }
    #pragma omp atomic
    st->combine_s += ladder_now() - tcomb0;
}

/* Workspace needed (in doubles) for a full recursion from N down to
   cutoff/max_depth: each level allocates 8 (quadrants) + 7 (P) + 10 (S) =
   25 buffers of size (N/2)^2, plus, since each of the 7 subtrees gets its
   OWN disjoint sub-arena (see the carving logic above -- required for
   correctness under real task-parallel execution, not just a convenience),
   7 times whatever the next level down needs. Worst-case (assumes every
   branch recurses to its own full depth; some branches may bottom out
   earlier at the cutoff, in which case the arena is merely oversized, not
   wrong). */
static size_t workspace_doubles_needed(int N, int cutoff, int max_depth) {
    if (N <= cutoff || max_depth <= 0) return 0;
    int k = N/2;
    size_t here = 25 * (size_t)k * k;
    size_t child = workspace_doubles_needed(k, cutoff, max_depth - 1);
    return here + 7 * child; /* worst case: 7 independent subtrees each need their own space */
}

double* strassen_rebuild(const double *A, const double *B, int N, int cutoff, int max_depth,
                          int KC, int MC, int NC, int nthreads, strassen_times_t *st) {
    memset(st, 0, sizeof(*st));
    g_cutoff = cutoff; g_max_depth = max_depth; g_KC = KC; g_MC = MC; g_NC = NC;

    size_t arena_size = workspace_doubles_needed(N, cutoff, max_depth) + 1024;
    double *arena_base = aligned_alloc_d(arena_size);
    arena_t arena = { arena_base, 0, arena_size };

    double *C = aligned_alloc_d((size_t)N*N);

    omp_set_num_threads(nthreads);
    omp_set_max_active_levels(1);
    double t0 = ladder_now();
    #pragma omp parallel
    #pragma omp single nowait
    strassen_rec(A, B, C, N, 0, &arena, st);
    st->total_s = ladder_now() - t0;

    aligned_free_d(arena_base);
    return C;
}
