/* Pedagogical baseline only: unblocked, unvectorized, single-threaded triple
 * loop. Used solely as the secondary "vs naive" reference point -- never as
 * the headline metric (see bench/README.md). */
#include "driver_common.h"
#include <stdio.h>

static void naive_gemm(const double *A, const double *B, double *C, int N) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            double acc = 0.0;
            for (int k = 0; k < N; k++) {
                acc += A[(size_t)k * N + i] * B[(size_t)j * N + k];
            }
            C[(size_t)j * N + i] = acc;
        }
    }
}

int main(int argc, char **argv) {
    driver_args_t a = parse_driver_args(argc, argv);

    double *A = bench_alloc((size_t)a.N * a.N);
    double *B = bench_alloc((size_t)a.N * a.N);
    double *C = bench_alloc((size_t)a.N * a.N);
    bench_init_random(A, (size_t)a.N * a.N, a.seed);
    bench_init_random(B, (size_t)a.N * a.N, a.seed + 1);

    bench_compare_t cmp = {0.0, 0.0};
    if (a.verify) {
        double *Cref = bench_alloc((size_t)a.N * a.N);
        bench_reference_gemm(A, B, Cref, a.N);
        naive_gemm(A, B, C, a.N);
        cmp = bench_compare(C, Cref, a.N);
        bench_free(Cref);
    }

    bench_result_t r = bench_run(naive_gemm, A, B, C, a.N, a.reps);
    bench_print_csv_row(stdout, "naive", a.N, 1, a.label, a.reps, r, cmp);

    bench_free(A); bench_free(B); bench_free(C);
    return 0;
}
