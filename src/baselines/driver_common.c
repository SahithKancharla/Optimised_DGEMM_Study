#include "driver_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int driver_reported_threads(const char *lib_env_var) {
    const char *v = lib_env_var ? getenv(lib_env_var) : NULL;
    if (v && *v) return atoi(v);
    v = getenv("OMP_NUM_THREADS");
    if (v && *v) return atoi(v);
    return omp_get_max_threads();
}

driver_args_t parse_driver_args(int argc, char **argv) {
    driver_args_t a;
    a.N = argc > 1 ? atoi(argv[1]) : 512;
    a.reps = argc > 2 ? atoi(argv[2]) : 5;
    a.label = argc > 3 ? argv[3] : "default";
    a.seed = argc > 4 ? (unsigned)strtoul(argv[4], NULL, 10) : 42u;
    a.verify = argc > 5 ? atoi(argv[5]) : 1;
    if (a.N <= 0 || a.reps <= 0) {
        fprintf(stderr, "usage: %s N reps [label] [seed] [verify:0|1]\n", argv[0]);
        exit(1);
    }
    return a;
}
