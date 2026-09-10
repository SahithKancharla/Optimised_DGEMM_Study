/* Correctness test for the rebuilt Strassen (strassen_1level/2level)
 * against the long-double reference used everywhere else in this study. */
#include <stdio.h>
#include "strassen.h"
#include "dgemm_base.h"
#include "../bench/bench_common.h"

static int check(const char *name, void (*fn)(const double *, int, const double *, int, double *, int, int),
                  int n, unsigned seed) {
    double *A = bench_alloc((size_t)n * n);
    double *B = bench_alloc((size_t)n * n);
    double *C = bench_alloc((size_t)n * n);
    double *Cref = bench_alloc((size_t)n * n);
    bench_init_random(A, (size_t)n * n, seed);
    bench_init_random(B, (size_t)n * n, seed + 1);
    bench_reference_gemm(A, B, Cref, n);
    fn(A, n, B, n, C, n, n);
    bench_compare_t cmp = bench_compare(C, Cref, n);
    printf("%-16s n=%-5d rel_frob=%.3e rel_max=%.3e  %s\n",
           name, n, cmp.rel_frobenius, cmp.rel_max, cmp.rel_frobenius < 1e-10 ? "OK" : "FAIL");
    int ok = cmp.rel_frobenius < 1e-10;
    bench_free(A); bench_free(B); bench_free(C); bench_free(Cref);
    return ok;
}

int main(void) {
    int fails = 0;
    int sizes_1level[] = {64, 128, 256, 512, 1024};
    int sizes_2level[] = {128, 256, 512, 1024};

    printf("=== dgemm_base sanity (n odd-ish sizes too) ===\n");
    fails += !check("dgemm_base", dgemm_base, 500, 100);
    fails += !check("dgemm_base", dgemm_base, 517, 101); /* not a multiple of MR/NR: exercises boundary path */

    printf("=== strassen_1level ===\n");
    for (size_t i = 0; i < sizeof(sizes_1level) / sizeof(int); i++)
        fails += !check("strassen_1level", strassen_1level, sizes_1level[i], 42 + i);

    printf("=== strassen_2level ===\n");
    for (size_t i = 0; i < sizeof(sizes_2level) / sizeof(int); i++)
        fails += !check("strassen_2level", strassen_2level, sizes_2level[i], 42 + i);

    printf(fails == 0 ? "\nALL PASS\n" : "\n%d FAILURES\n", fails);
    return fails != 0;
}
