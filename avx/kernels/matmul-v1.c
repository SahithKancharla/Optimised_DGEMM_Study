// matmul_double.c

#include "kernel.h"
#include <string.h>
#include <omp.h>
#include <immintrin.h>

#define min(x,y)      ((x) < (y) ? (x) : (y))

#ifndef NTHREADS
  #define NTHREADS 32
#endif

// micro‑kernel dimensions
#define MR 24
#define NR 6

// blocking parameters
#define MC (MR * NTHREADS * 5)
#define NC (NR * NTHREADS * 50)
#define KC 1000

// packed buffers (double now)
static double blockA_packed[MC * KC] __attribute__((aligned(64)));
static double blockB_packed[NC * KC] __attribute__((aligned(64)));

void pack_panelB(const double* B, double* blockB_p, int nr, int kc, int K) {
    for (int p = 0; p < kc; p++) {
        if (p + 2 < kc) {
            _mm_prefetch((const char*)(B + (p+2)), _MM_HINT_T0);
        }
        // copy the nr real columns
        for (int j = 0; j < nr; j++) {
            *blockB_p++ = B[j*K + p];
        }
        // pad out to NR
        for (int j = nr; j < NR; j++) {
            *blockB_p++ = 0.0;
        }
    }
}

void pack_blockB(const double* B, double* blockB_p, int nc, int kc, int K) {
#pragma omp parallel for
    for (int j = 0; j < nc; j += NR) {
        int nr = min(NR, nc - j);
        pack_panelB(&B[j*K], &blockB_p[j*kc], nr, kc, K);
    }
}

void pack_panelA(const double* A, double* blockA_p, int mr, int kc, int M) {
    for (int p = 0; p < kc; p++) {
        if (p + 2 < kc) {
            _mm_prefetch((const char*)(A + (p+2)*M), _MM_HINT_T0);
        }
        // copy the mr real rows
        for (int i = 0; i < mr; i++) {
            *blockA_p++ = A[p*M + i];
        }
        // pad out to MR
        for (int i = mr; i < MR; i++) {
            *blockA_p++ = 0.0;
        }
    }
}

void pack_blockA(const double* A, double* blockA_p, int mc, int kc, int M) {
#pragma omp parallel for
    for (int i = 0; i < mc; i += MR) {
        int mr = min(MR, mc - i);
        pack_panelA(&A[i], &blockA_p[i*kc], mr, kc, M);
    }
}

void matmul(double* A, double* B, double* C, int m, int n, int k) {
    // zero C
    memset(C, 0, (size_t)m * n * sizeof(double));

    for (int j = 0; j < n; j += NC) {
        int nc = min(NC, n - j);
        for (int p = 0; p < k; p += KC) {
            int kc = min(KC, k - p);

            // pack B-panel
            pack_blockB(&B[j * k + p], blockB_packed, nc, kc, k);

            for (int i = 0; i < m; i += MC) {
                int mc = min(MC, m - i);

                // pack A-block
                pack_blockA(&A[p * m + i], blockA_packed, mc, kc, m);

                #pragma omp parallel for collapse(2)
                for (int ir = 0; ir < mc; ir += MR) {
                    for (int jr = 0; jr < nc; jr += NR) {
                        int mr = min(MR, mc - ir);
                        int nr = min(NR, nc - jr);

                        kernel_24x6(
                          &blockA_packed[ir * kc],
                          &blockB_packed[jr * kc],
                          &C[(j + jr)*m + (i + ir)],
                          mr,
                          nr,
                          kc,
                          m
                        );
                    }
                }
            }
        }
    }
}
