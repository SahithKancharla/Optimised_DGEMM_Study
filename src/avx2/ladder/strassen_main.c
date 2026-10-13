#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "timer.h"
#include "strassen_rebuild.c"

int main(int argc, char **argv) {
    /* usage: strassen_main N cutoff max_depth threads reps KC MC NC */
    if (argc < 9) {
        fprintf(stderr, "usage: %s <N> <cutoff> <max_depth> <threads> <reps> <KC> <MC> <NC>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int cutoff = atoi(argv[2]);
    int max_depth = atoi(argv[3]);
    int T = atoi(argv[4]);
    int reps = atoi(argv[5]);
    int KC = atoi(argv[6]), MC = atoi(argv[7]), NC = atoi(argv[8]);

    double *A = aligned_alloc_d((size_t)N*N);
    double *B = aligned_alloc_d((size_t)N*N);
    for (size_t i = 0; i < (size_t)N*N; i++) { A[i] = (double)((i%13)-6); B[i] = (double)((i%11)-5); }

    strassen_times_t best_st; memset(&best_st, 0, sizeof(best_st));
    double best_total = 1e300;
    double *C = NULL;
    for (int r = 0; r < reps; r++) {
        strassen_times_t st;
        double *Cr = strassen_rebuild(A, B, N, cutoff, max_depth, KC, MC, NC, T, &st);
        if (st.total_s < best_total) {
            best_total = st.total_s; best_st = st;
            if (C) aligned_free_d(C);
            C = Cr;
        } else {
            aligned_free_d(Cr);
        }
    }
    double relerr = reference_check(A, B, C, N, N, N);
    double gflops = 2.0*(double)N*N*N / best_total / 1e9;

    /* CSV: N,cutoff,max_depth,T,KC,MC,NC,reps,leaf_s,addsub_s,combine_s,workspace_s,sched_s,total_s,gflops,relerr,leaf_calls,addsub_calls */
    printf("strassen,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3e,%ld,%ld\n",
           N, cutoff, max_depth, T, KC, MC, NC, reps,
           best_st.leaf_s, best_st.addsub_s, best_st.combine_s, best_st.workspace_s, best_st.sched_s,
           best_total, gflops, relerr, best_st.leaf_calls, best_st.addsub_calls);

    aligned_free_d(A); aligned_free_d(B); aligned_free_d(C);
    return relerr > 1e-6 ? 1 : 0;
}
