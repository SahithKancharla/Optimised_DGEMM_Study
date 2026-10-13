/* Numerical accuracy comparison: classical (dgemm_base) vs strassen_1level
 * vs strassen_2level, against the long-double reference, across 3 input
 * distributions chosen to stress cancellation differently. This is the
 * numerical-accuracy experiment the paper was missing entirely. */
#include <stdio.h>
#include <stdlib.h>
#include "distributions.h"
#include "../strassen2/strassen.h"
#include "../strassen2/dgemm_base.h"
#include "../bench/bench_common.h"

typedef void (*mm3)(const double *, int, const double *, int, double *, int, int);
typedef void (*dist_fn)(double *, size_t, unsigned);

static void run(const char *dist_name, dist_fn dist, const char *variant, mm3 fn,
                 int N, unsigned seed) {
    double *A = bench_alloc((size_t)N * N);
    double *B = bench_alloc((size_t)N * N);
    double *C = bench_alloc((size_t)N * N);
    double *Cref = bench_alloc((size_t)N * N);
    dist(A, (size_t)N * N, seed);
    dist(B, (size_t)N * N, seed + 1);
    bench_reference_gemm(A, B, Cref, N);
    fn(A, N, B, N, C, N, N);
    bench_compare_t cmp = bench_compare(C, Cref, N);
    printf("%s,%s,%d,%.3e,%.3e\n", dist_name, variant, N, cmp.rel_frobenius, cmp.rel_max);
    bench_free(A); bench_free(B); bench_free(C); bench_free(Cref);
}

static void wide_range_wrapper(double *M, size_t total, unsigned seed) {
    /* distributions.h's wide-dynamic-range wants n_rows/n_cols separately;
     * reconstruct from total for this square-matrix-only harness. */
    size_t n = 1;
    while (n * n < total) n++;
    dist_wide_dynamic_range(M, n, n, seed);
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 512;
    unsigned seed = argc > 2 ? (unsigned)strtoul(argv[2], NULL, 10) : 7u;

    printf("distribution,variant,N,rel_frob_err,rel_max_err\n");
    struct { const char *name; dist_fn fn; } dists[] = {
        {"uniform", dist_uniform},
        {"wide_dynamic_range", wide_range_wrapper},
        {"clustered_cancellation", dist_clustered_cancellation},
    };
    for (size_t d = 0; d < sizeof(dists) / sizeof(dists[0]); d++) {
        run(dists[d].name, dists[d].fn, "dgemm_base", dgemm_base, N, seed);
        run(dists[d].name, dists[d].fn, "strassen_1level", strassen_1level, N, seed);
        if (N % 4 == 0) run(dists[d].name, dists[d].fn, "strassen_2level", strassen_2level, N, seed);
    }
    return 0;
}
