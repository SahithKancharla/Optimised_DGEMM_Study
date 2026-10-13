/* Effective GFLOP/s = 2*N^3/time for ALL variants here, including
 * Strassen -- deliberately the same formula as classical GEMM, not
 * Strassen's true (lower) flop count, so the number is directly
 * comparable to dgemm_base/OpenBLAS/BLIS/MKL at the same N. This is
 * exactly what the reviewer asked for: Strassen does less real
 * arithmetic, so a naive flop-counted GFLOP/s would look artificially
 * high; "effective" GFLOP/s answers "how does this compare to classical
 * DGEMM at this size" instead. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "strassen.h"
#include "dgemm_base.h"
#include "../bench/bench_common.h"
#include "../baselines/driver_common.h"

typedef void (*mm3)(const double *, int, const double *, int, double *, int, int);

static void run_one(const char *label, mm3 fn, int N, int reps, unsigned seed, int verify) {
    double *A = bench_alloc((size_t)N * N);
    double *B = bench_alloc((size_t)N * N);
    double *C = bench_alloc((size_t)N * N);
    bench_init_random(A, (size_t)N * N, seed);
    bench_init_random(B, (size_t)N * N, seed + 1);

    bench_compare_t cmp = {0.0, 0.0};
    if (verify) {
        double *Cref = bench_alloc((size_t)N * N);
        bench_reference_gemm(A, B, Cref, N);
        fn(A, N, B, N, C, N, N);
        cmp = bench_compare(C, Cref, N);
        bench_free(Cref);
    }

    fn(A, N, B, N, C, N, N); /* warmup */
    double best = 1e300;
    for (int r = 0; r < reps; r++) {
        double t0 = omp_get_wtime();
        fn(A, N, B, N, C, N, N);
        double t1 = omp_get_wtime();
        if (t1 - t0 < best) best = t1 - t0;
    }
    double gflops = 2.0 * (double)N * (double)N * (double)N / best / 1e9;
    printf("%s,%d,%d,%.9f,%.6f,%.3e,%.3e\n", label, N, omp_get_max_threads(), best, gflops,
           cmp.rel_frobenius, cmp.rel_max);
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1024;
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    unsigned seed = argc > 3 ? (unsigned)strtoul(argv[3], NULL, 10) : 42u;
    int verify = argc > 4 ? atoi(argv[4]) : 1;

    printf("variant,N,threads,best_s,gflops,rel_frob_err,rel_max_err\n");
    run_one("dgemm_base", dgemm_base, N, reps, seed, verify);
    run_one("strassen_1level", strassen_1level, N, reps, seed, verify);
    if (N % 4 == 0) run_one("strassen_2level", strassen_2level, N, reps, seed, verify);
    return 0;
}
