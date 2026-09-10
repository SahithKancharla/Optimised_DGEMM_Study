#include "strassen.h"
#include "strassen_ops.h"
#include "dgemm_base.h"
#include <omp.h>
#include "../bench/bench_common.h"

typedef void (*mm_fn)(const double *A, int ldA, const double *B, int ldB,
                       double *C, int ldC, int n);

/* Thread-aware cutoff policy: with <=8 threads, splitting the 7 products
 * into OpenMP tasks (each running dgemm_base single-threaded, since
 * nested parallel regions serialize to 1 thread by default -- see
 * dgemm_base.c's comment) is a good use of a modest thread budget. With
 * more threads than that, 7-way task splitting wastes most of them (7
 * single-threaded branches on e.g. 16 threads leaves 9 idle); better to
 * run the 7 products SEQUENTIALLY and let each one's dgemm_base call use
 * the FULL thread count via its own (non-nested, so not serialized)
 * parallel region. This is exactly the "thread-aware recursion cutoff"
 * the reviewer feedback asked for, made concrete and measurable rather
 * than a fixed constant. */
#define TASK_SPLIT_THREAD_THRESHOLD 8

static void strassen_level(const double *A, int ldA, const double *B, int ldB,
                            double *C, int ldC, int n, mm_fn sub_mm) {
    int k = n / 2;
    int use_tasks = omp_get_max_threads() <= TASK_SPLIT_THREAD_THRESHOLD;

    /* Pooled workspace: ONE allocation for all 10 combined-operand
     * temporaries (S0..S9) and 7 product results (P0..P6) at this
     * recursion level, not 17 separate mallocs and absolutely not one
     * malloc per tiny recursive call the way the original implementation
     * did (millions of allocations at scale -- FINDINGS.md Part C/E). */
    size_t kk = (size_t)k * k;
    double *pool = bench_alloc(17 * kk);
    double *S[10];
    double *P[7];
    for (int i = 0; i < 10; i++) S[i] = pool + (size_t)i * kk;
    for (int i = 0; i < 7; i++) P[i] = pool + (size_t)(10 + i) * kk;

    const double *A11 = A, *A12 = A + (size_t)k * ldA, *A21 = A + k, *A22 = A + (size_t)k * ldA + k;
    const double *B11 = B, *B12 = B + (size_t)k * ldB, *B21 = B + k, *B22 = B + (size_t)k * ldB + k;

    /* Fused adds: only the 10 GENUINELY combined operands are
     * materialized (Winograd/Strassen's classic formulation needs exactly
     * these 10, no more); the 4 bare quadrants used alone (A11 in M3, A22
     * in M4, B11 in M2, B22 in M5) are never copied -- they're passed to
     * sub_mm directly below as strided views with their original leading
     * dimension. */
    add_view(S[0], A11, ldA, A22, ldA, k);  /* A11+A22 */
    add_view(S[1], B11, ldB, B22, ldB, k);  /* B11+B22 */
    add_view(S[2], A21, ldA, A22, ldA, k);  /* A21+A22 */
    sub_view(S[3], B12, ldB, B22, ldB, k);  /* B12-B22 */
    sub_view(S[4], B21, ldB, B11, ldB, k);  /* B21-B11 */
    add_view(S[5], A11, ldA, A12, ldA, k);  /* A11+A12 */
    sub_view(S[6], A21, ldA, A11, ldA, k);  /* A21-A11 */
    add_view(S[7], B11, ldB, B12, ldB, k);  /* B11+B12 */
    sub_view(S[8], A12, ldA, A22, ldA, k);  /* A12-A22 */
    add_view(S[9], B21, ldB, B22, ldB, k);  /* B21+B22 */

#pragma omp parallel if(use_tasks) num_threads(omp_get_max_threads())
#pragma omp single nowait
    {
#pragma omp task if(use_tasks)
        sub_mm(S[0], k, S[1], k, P[0], k, k);   /* M1 = (A11+A22)(B11+B22) */
#pragma omp task if(use_tasks)
        sub_mm(S[2], k, B11, ldB, P[1], k, k);  /* M2 = (A21+A22) B11 */
#pragma omp task if(use_tasks)
        sub_mm(A11, ldA, S[3], k, P[2], k, k);  /* M3 = A11 (B12-B22) */
#pragma omp task if(use_tasks)
        sub_mm(A22, ldA, S[4], k, P[3], k, k);  /* M4 = A22 (B21-B11) */
#pragma omp task if(use_tasks)
        sub_mm(S[5], k, B22, ldB, P[4], k, k);  /* M5 = (A11+A12) B22 */
#pragma omp task if(use_tasks)
        sub_mm(S[6], k, S[7], k, P[5], k, k);   /* M6 = (A21-A11)(B11+B12) */
#pragma omp task if(use_tasks)
        sub_mm(S[8], k, S[9], k, P[6], k, k);   /* M7 = (A12-A22)(B21+B22) */
#pragma omp taskwait
    }

    double *C11 = C, *C12 = C + (size_t)k * ldC, *C21 = C + k, *C22 = C + (size_t)k * ldC + k;
    /* C11 = M1+M4-M5+M7 ; C12 = M3+M5 ; C21 = M2+M4 ; C22 = M1-M2+M3+M6 */
    copy_to_view(C11, ldC, P[0], k);
    accum_add_view(C11, ldC, P[3], k);
    accum_sub_view(C11, ldC, P[4], k);
    accum_add_view(C11, ldC, P[6], k);

    copy_to_view(C12, ldC, P[2], k);
    accum_add_view(C12, ldC, P[4], k);

    copy_to_view(C21, ldC, P[1], k);
    accum_add_view(C21, ldC, P[3], k);

    copy_to_view(C22, ldC, P[0], k);
    accum_sub_view(C22, ldC, P[1], k);
    accum_add_view(C22, ldC, P[2], k);
    accum_add_view(C22, ldC, P[5], k);

    bench_free(pool);
}

void strassen_1level(const double *A, int ldA, const double *B, int ldB,
                      double *C, int ldC, int n) {
    strassen_level(A, ldA, B, ldB, C, ldC, n, dgemm_base);
}

static void strassen_inner_wrapper(const double *A, int ldA, const double *B, int ldB,
                                    double *C, int ldC, int n) {
    strassen_level(A, ldA, B, ldB, C, ldC, n, dgemm_base);
}

void strassen_2level(const double *A, int ldA, const double *B, int ldB,
                      double *C, int ldC, int n) {
    strassen_level(A, ldA, B, ldB, C, ldC, n, strassen_inner_wrapper);
}
