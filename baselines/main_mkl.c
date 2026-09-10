#include "driver_common.h"
#include <mkl_cblas.h>
#include <stdio.h>

static void mkl_gemm(const double *A, const double *B, double *C, int N) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                1.0, A, N, B, N, 0.0, C, N);
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
        mkl_gemm(A, B, C, a.N);
        cmp = bench_compare(C, Cref, a.N);
        bench_free(Cref);
    }

    bench_result_t r = bench_run(mkl_gemm, A, B, C, a.N, a.reps);
    int threads = driver_reported_threads("MKL_NUM_THREADS");
    bench_print_csv_row(stdout, "mkl", a.N, threads, a.label, a.reps, r, cmp);

    bench_free(A); bench_free(B); bench_free(C);
    return 0;
}
