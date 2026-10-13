/* STREAM-triad-style bandwidth microbenchmark: c[i] = a[i] + scalar*b[i].
 * Used as the roofline model's achieved-bandwidth measurement in place of
 * perf hardware counters, which are unavailable under WSL2/Hyper-V (see
 * setup/wsl_env_setup.sh's sanity check). Arrays sized well beyond LLC so
 * this measures real DRAM bandwidth, not cache-resident bandwidth. */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv) {
    /* Default: 64M doubles/array = 512MB/array, 1.5GB total -- comfortably
     * beyond this CPU's 30MB L3. */
    size_t n = argc > 1 ? (size_t)atoll(argv[1]) : (64ull << 20);
    int reps = argc > 2 ? atoi(argv[2]) : 10;

    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *c = malloc(n * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }

    double scalar = 3.0;
    double best_s = 1e300;
    for (int r = 0; r < reps; r++) {
        double t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) c[i] = a[i] + scalar * b[i];
        double t1 = omp_get_wtime();
        if (t1 - t0 < best_s) best_s = t1 - t0;
    }

    /* Triad touches 2 reads + 1 write per element = 3*8 bytes */
    double gb = (double)n * 3.0 * sizeof(double) / 1e9;
    double gbps = gb / best_s;
    printf("n=%zu threads=%d best_time_s=%.6f bandwidth_GBps=%.2f\n",
           n, omp_get_max_threads(), best_s, gbps);
    printf("sink=%f\n", c[0] + c[n / 2] + c[n - 1]);

    free(a); free(b); free(c);
    return 0;
}
