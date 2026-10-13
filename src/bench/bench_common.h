#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include <stddef.h>
#include <stdio.h>

/* Column-major, C = A*B, all N x N, doubles, 64-byte aligned. */
typedef void (*gemm_fn)(const double *A, const double *B, double *C, int N);

double *bench_alloc(size_t n_doubles);
void bench_free(double *p);
void bench_init_random(double *M, size_t n, unsigned seed);

/* Reference C = A*B computed with long-double accumulation (slow, used only
 * for correctness verification, never timed / never part of a headline number). */
void bench_reference_gemm(const double *A, const double *B, double *Cref, int N);

typedef struct {
    double rel_frobenius;
    double rel_max;
} bench_compare_t;

bench_compare_t bench_compare(const double *C, const double *Cref, int N);

typedef struct {
    double median_s;
    double min_s;
    double max_s;
    double gflops_median;
} bench_result_t;

/* Runs fn(A,B,C,N) `reps` times, returns timing stats. C is not re-zeroed
 * between reps by this function -- caller's fn must overwrite C fully. */
bench_result_t bench_run(gemm_fn fn, const double *A, const double *B, double *C,
                          int N, int reps);

void bench_print_csv_header(FILE *f);

/* One CSV row: variant,N,threads,label,reps,median_s,min_s,gflops,rel_frob_err,rel_max_err */
void bench_print_csv_row(FILE *f, const char *variant, int N, int threads,
                          const char *label, int reps, bench_result_t r,
                          bench_compare_t cmp);

#endif
