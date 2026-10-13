/* Pure register-only AVX2 FMA microbenchmark, single-threaded. Meant to be
 * run once per logical CPU via `taskset -c N ./classify_cores` so the
 * measured sustained GFLOP/s can empirically separate P-cores (2x 256-bit
 * FMA/cycle) from E-cores (1x 256-bit FMA/cycle) -- needed because WSL2/
 * Hyper-V reports a flattened, uniform vCPU topology that hides which
 * logical CPU is which (see topology/README.md). No memory traffic: all
 * operands live in registers so this measures compute throughput only. */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ITERS 2000000000LL

int main(int argc, char **argv) {
    long long iters = argc > 1 ? atoll(argv[1]) : ITERS;

    /* 8 independent accumulator chains: FMA latency is ~4-5 cycles and a
     * P-core has 2 FMA-capable ports, so >=8 independent in-flight chains
     * are needed to fully hide latency and expose true 2-port throughput
     * (too few chains would under-measure P-cores and make them look like
     * E-cores, defeating the point of this classifier). */
    __m256d b = _mm256_set1_pd(0.9999999);
    __m256d a[8], c[8];
    for (int i = 0; i < 8; i++) {
        a[i] = _mm256_set1_pd(1.0000001 + 0.0000001 * i);
        c[i] = _mm256_set1_pd(0.0);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long long i = 0; i < iters; i++) {
        c[0] = _mm256_fmadd_pd(a[0], b, c[0]);
        c[1] = _mm256_fmadd_pd(a[1], b, c[1]);
        c[2] = _mm256_fmadd_pd(a[2], b, c[2]);
        c[3] = _mm256_fmadd_pd(a[3], b, c[3]);
        c[4] = _mm256_fmadd_pd(a[4], b, c[4]);
        c[5] = _mm256_fmadd_pd(a[5], b, c[5]);
        c[6] = _mm256_fmadd_pd(a[6], b, c[6]);
        c[7] = _mm256_fmadd_pd(a[7], b, c[7]);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    /* 8 independent FMA chains x 4 doubles/reg x 2 flops(mul+add)/FMA */
    double gflops = (double)iters * 8.0 * 4.0 * 2.0 / secs / 1e9;

    double sink = 0.0;
    for (int i = 0; i < 8; i++) {
        double tmp[4];
        _mm256_storeu_pd(tmp, c[i]);
        sink += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
    fprintf(stderr, "sink=%f (prevents dead-code elim)\n", sink);

    printf("%.3f\n", gflops);
    return 0;
}
