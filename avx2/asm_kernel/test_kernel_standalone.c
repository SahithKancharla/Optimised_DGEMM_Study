/* Standalone correctness test for kernel_6x8_asm, BEFORE wiring it into the
 * full packed multilevel driver -- isolates assembly bugs from packing/
 * blocking bugs. Builds a small packed A (kc x MR) / B (kc x NR) panel,
 * computes the MR x NR result both via kernel_6x8_asm and via a naive
 * reference loop, and compares. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MR 6
#define NR 8

extern void kernel_6x8_asm(const double *A, const double *B, double *C,
                            long kc, long ldC_bytes);

static double frand(unsigned *state) {
    *state ^= *state << 13; *state ^= *state >> 17; *state ^= *state << 5;
    return ((double)*state / 4294967295.0) * 2.0 - 1.0;
}

int main(void) {
    int fails = 0;
    for (int kc = 1; kc <= 9; kc++) {
        double *A = aligned_alloc(64, (size_t)kc * MR * sizeof(double));
        double *B = aligned_alloc(64, (size_t)kc * NR * sizeof(double));
        unsigned seed = 12345u + kc;
        for (int i = 0; i < kc * MR; i++) A[i] = frand(&seed);
        for (int i = 0; i < kc * NR; i++) B[i] = frand(&seed);

        int ldC = MR; /* col-major, ldC rows per column, only need >= MR */
        double *C_asm = aligned_alloc(64, (size_t)ldC * NR * sizeof(double));
        double *C_ref = calloc((size_t)ldC * NR, sizeof(double));
        memset(C_asm, 0, (size_t)ldC * NR * sizeof(double));

        kernel_6x8_asm(A, B, C_asm, (long)kc, (long)(ldC * sizeof(double)));

        for (int p = 0; p < kc; p++)
            for (int n = 0; n < NR; n++)
                for (int i = 0; i < MR; i++)
                    C_ref[(size_t)n * ldC + i] += A[p * MR + i] * B[p * NR + n];

        double maxerr = 0.0;
        for (int i = 0; i < ldC * NR; i++) {
            double d = fabs(C_asm[i] - C_ref[i]);
            if (d > maxerr) maxerr = d;
        }
        printf("kc=%d  max_abs_err=%.3e  %s\n", kc, maxerr, maxerr < 1e-9 ? "OK" : "FAIL");
        if (maxerr >= 1e-9) fails++;

        free(A); free(B); free(C_asm); free(C_ref);
    }
    printf(fails == 0 ? "ALL PASS\n" : "%d FAILURES\n", fails);
    return fails != 0;
}
