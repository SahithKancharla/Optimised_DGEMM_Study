// matmul_double.c
#include <immintrin.h>  // for _mm_prefetch, _MM_HINT_T0, AVX-512 intrinsics
#include "kernel.h"
#include <string.h>
#include <omp.h>
#include <x86intrin.h>

#define min(x,y)      ((x) < (y) ? (x) : (y))

#ifndef NTHREADS
  #define NTHREADS 32
#endif

// micro‑kernel dimensions
#define MR 24
#define NR 6

// blocking parameters
#define MC (MR * NTHREADS * 10)
#define NC (NR * NTHREADS * 100)
#define KC 2048

// packed buffers (double now)
static double blockA_packed[MC * KC] __attribute__((aligned(64)));
static double blockB_packed[NC * KC] __attribute__((aligned(64)));
static inline int round_up_24(int n) { return (n + MR - 1) / MR * MR; }

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

void matmul(double *A, double *B, double *C, int N) {
    // zero C
    memset(C, 0, (size_t)N * N * sizeof(double));

    for (int j = 0; j < N; j += NC) {
        int nc = (N - j < NC ? N - j : NC);
        for (int p = 0; p < N; p += KC) {
            int kc = (N - p < KC ? N - p : KC);

            // pack B-panel (nc × kc) from B[j..j+nc-1][p..p+kc-1]
            pack_blockB(&B[j * N + p], blockB_packed, nc, kc, N);

            for (int i = 0; i < N; i += MC) {
                int mc = (N - i < MC ? N - i : MC);

                // pack A-block (mc × kc) from A[p..p+kc-1][i..i+mc-1]
                pack_blockA(&A[p * N + i], blockA_packed, mc, kc, N);

                // now launch micro‑kernels over the mc×nc block
                #pragma omp parallel for collapse(2) schedule(static)
                for (int ir = 0; ir < mc; ir += MR) {
                    for (int jr = 0; jr < nc; jr += NR) {
                        int mr = (mc - ir < MR ? mc - ir : MR);
                        int nr = (nc - jr < NR ? nc - jr : NR);

                        kernel_24x6(
                          &blockA_packed[ir * kc],      // A micro‑panel
                          &blockB_packed[jr * kc],      // B micro‑panel
                          &C[(j + jr) * N + (i + ir)],   
                          mr, nr, kc, N
                        );
                    }
                }
            }
        }
    }
}

