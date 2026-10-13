#include "kernel.h"
#include <immintrin.h>

/*
 * 24x6 micro-kernel -- AVX2+FMA3 (double), port of kernel.c's AVX-512
 * version for hardware without AVX-512. Same MR=24/NR=6 tile and the same
 * packed-panel contract as kernel.c (A panel: [p*MR+row], B panel:
 * [p*NR+col]), so it is a drop-in replacement for kernel_24x6 in matmul.c
 * -- only the register width differs (six 256-bit ymm registers per
 * column instead of three 512-bit zmm registers, since MR=24 doubles
 * needs 24/4=6 ymm registers vs. 24/8=3 zmm registers). Use this build to
 * validate matmul.c/kernel.c changes on machines that only have AVX2.
 *
 *   Fast path : mr == 24 -> unmasked loads/stores
 *   Slow path : mr <  24 -> scalar fallback (correctness, not tuned)
 */

#define MR 24
#define NR  6

void kernel_24x6(const double *restrict A,
                  const double *restrict B,
                  double *restrict       C,
                  int                    mr,
                  int                    nr,
                  int                    kc,
                  int                    ldC)
{
    if (mr == MR) {
        __m256d C0[6], C1[6], C2[6], C3[6], C4[6], C5[6];
        __m256d *Ccols[6] = { C0, C1, C2, C3, C4, C5 };

        for (int col = 0; col < nr; col++) {
            for (int s = 0; s < 6; s++) {
                Ccols[col][s] = _mm256_loadu_pd(C + (size_t)col * ldC + s * 4);
            }
        }

        for (int p = 0; p < kc; p++) {
            __m256d a[6];
            for (int s = 0; s < 6; s++) {
                a[s] = _mm256_load_pd(A + p * MR + s * 4);
            }
            for (int col = 0; col < nr; col++) {
                __m256d b = _mm256_set1_pd(B[p * NR + col]);
                for (int s = 0; s < 6; s++) {
                    Ccols[col][s] = _mm256_fmadd_pd(a[s], b, Ccols[col][s]);
                }
            }
        }

        for (int col = 0; col < nr; col++) {
            for (int s = 0; s < 6; s++) {
                _mm256_storeu_pd(C + (size_t)col * ldC + s * 4, Ccols[col][s]);
            }
        }
        return;
    }

    /* slow path: partial tile, plain scalar accumulation */
    for (int col = 0; col < nr; col++) {
        for (int row = 0; row < mr; row++) {
            double acc = C[(size_t)col * ldC + row];
            for (int p = 0; p < kc; p++) {
                acc += A[p * MR + row] * B[p * NR + col];
            }
            C[(size_t)col * ldC + row] = acc;
        }
    }
}
