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

// pack an nr x kc panel of B: rows k0..k0+kc-1, cols col0..col0+nr-1 of the
// full row-major B (leading dimension ld), into blockB_p laid out as
// [p = 0..kc-1][col = 0..NR-1] (col varies fastest, padded to NR).
void pack_panelB(const double* B, double* blockB_p, int nr, int kc, int ld, int k0, int col0) {
    for (int p = 0; p < kc; p++) {
        if (p + 2 < kc) {
            _mm_prefetch((const char*)(B + (size_t)(k0 + p + 2) * ld + col0), _MM_HINT_T0);
        }
        // copy the nr real columns
        for (int j = 0; j < nr; j++) {
            *blockB_p++ = B[(size_t)(k0 + p) * ld + (col0 + j)];
        }
        // pad out to NR
        for (int j = nr; j < NR; j++) {
            *blockB_p++ = 0.0;
        }
    }
}

void pack_blockB(const double* B, double* blockB_p, int nc, int kc, int ld, int k0, int col0) {
#pragma omp parallel for
    for (int j = 0; j < nc; j += NR) {
        int nr = min(NR, nc - j);
        pack_panelB(B, &blockB_p[j*kc], nr, kc, ld, k0, col0 + j);
    }
}

// pack an mr x kc panel of A: rows row0..row0+mr-1, cols k0..k0+kc-1 of the
// full row-major A (leading dimension ld), into blockA_p laid out as
// [p = 0..kc-1][row = 0..MR-1] (row varies fastest, padded to MR).
void pack_panelA(const double* A, double* blockA_p, int mr, int kc, int ld, int row0, int k0) {
    for (int p = 0; p < kc; p++) {
        if (p + 2 < kc) {
            _mm_prefetch((const char*)(A + (size_t)row0 * ld + (k0 + p + 2)), _MM_HINT_T0);
        }
        // copy the mr real rows
        for (int i = 0; i < mr; i++) {
            *blockA_p++ = A[(size_t)(row0 + i) * ld + (k0 + p)];
        }
        // pad out to MR
        for (int i = mr; i < MR; i++) {
            *blockA_p++ = 0.0;
        }
    }
}

void pack_blockA(const double* A, double* blockA_p, int mc, int kc, int ld, int row0, int k0) {
#pragma omp parallel for
    for (int i = 0; i < mc; i += MR) {
        int mr = min(MR, mc - i);
        pack_panelA(A, &blockA_p[i*kc], mr, kc, ld, row0 + i, k0);
    }
}

void matmul(double *A, double *B, double *C, int N) {
    // zero C
    memset(C, 0, (size_t)N * N * sizeof(double));

    for (int j = 0; j < N; j += NC) {
        int nc = (N - j < NC ? N - j : NC);
        for (int p = 0; p < N; p += KC) {
            int kc = (N - p < KC ? N - p : KC);

            // pack B-panel (nc x kc) from B[p..p+kc-1][j..j+nc-1]
            pack_blockB(B, blockB_packed, nc, kc, N, p, j);

            for (int i = 0; i < N; i += MC) {
                int mc = (N - i < MC ? N - i : MC);

                // pack A-block (mc x kc) from A[i..i+mc-1][p..p+kc-1]
                pack_blockA(A, blockA_packed, mc, kc, N, i, p);

                // now launch micro‑kernels over the mc×nc block
                #pragma omp parallel for collapse(2) schedule(static)
                for (int ir = 0; ir < mc; ir += MR) {
                    for (int jr = 0; jr < nc; jr += NR) {
                        int mr = (mc - ir < MR ? mc - ir : MR);
                        int nr = (nc - jr < NR ? nc - jr : NR);

                        // kernel_24x6 fills its C tile in "MR fastest, NR
                        // ld-strided" order -- i.e. transposed relative to
                        // standard row-major C. Write to the transposed
                        // location here; the whole buffer is un-transposed
                        // once below.
                        kernel_24x6(
                          &blockA_packed[ir * kc],      // A micro‑panel
                          &blockB_packed[jr * kc],      // B micro‑panel
                          &C[(size_t)(j + jr) * N + (i + ir)],
                          mr, nr, kc, N
                        );
                    }
                }
            }
        }
    }

    // kernel_24x6 wrote C transposed (see note above); flip it back to
    // standard row-major C = A*B. Independent per r (disjoint (r,c) pairs
    // across different r), so this is safe to parallelize like every other
    // loop nest in this function.
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < N; r++) {
        for (int c = r + 1; c < N; c++) {
            double t = C[(size_t)r * N + c];
            C[(size_t)r * N + c] = C[(size_t)c * N + r];
            C[(size_t)c * N + r] = t;
        }
    }
}
