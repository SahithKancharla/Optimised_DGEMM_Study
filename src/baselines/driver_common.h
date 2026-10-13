#ifndef DRIVER_COMMON_H
#define DRIVER_COMMON_H

#include "../bench/bench_common.h"

/* Shared CLI for every baseline/variant driver:
 *   ./driver N reps label seed verify
 * verify: "1" runs bench_reference_gemm and reports error columns, "0" skips
 * (skip for large N where the O(N^3) long-double reference would dominate runtime). */
typedef struct {
    int N;
    int reps;
    const char *label;
    unsigned seed;
    int verify;
} driver_args_t;

driver_args_t parse_driver_args(int argc, char **argv);

/* Best-effort reported thread count: checks the library-specific env var first
 * (so the number logged matches what actually governs the library), then
 * OMP_NUM_THREADS, then falls back to omp_get_max_threads(). */
int driver_reported_threads(const char *lib_env_var);

#endif
