/* Portable/generalizable harness: everything in avx2/asm_kernel and
 * avx2/matmul_v*.c was tuned FOR THIS ONE MACHINE (hardcoded KC/MC/NC from
 * an empirical sweep done here, and a scheduling chunk size, "dynamic,6",
 * that happens to work well specifically because of THIS environment's
 * particular WSL2/Hyper-V scheduling noise -- see FINDINGS.md B.8's
 * caveat). This file is a deliberately different design, grounded in the
 * actual HPC autotuning/portability literature rather than one machine's
 * numbers:
 *
 *   - Blocking (KC/MC/NC): derived from ACTUALLY DETECTED cache sizes
 *     (hw_detect.c, portable via /sys on any Linux box) using the
 *     Goto/van de Geijn + BLIS analytical model (blocking_model.c), then
 *     refined by a brief on-first-run empirical search around that
 *     estimate -- the ATLAS/AEOS "automated empirical optimization of
 *     software" paradigm (Whaley & Dongarra), done at runtime on whatever
 *     machine this actually executes on, not baked in from development.
 *   - Threading: `schedule(static)` instead of a hand-picked dynamic chunk
 *     size. Guided scheduling starts with large chunks (low overhead) and
 *     shrinks them over the loop's remaining iterations (fine-grained load
 *     balance near the end) -- a principled, parameter-free choice the
 *     OpenMP scheduling literature specifically recommends for imbalanced/
 *     heterogeneous execution (whether that heterogeneity is real
 *     asymmetric cores on bare metal, or a hypervisor silently reshuffling
 *     vCPU-to-core mapping, as on this dev machine -- same mechanism,
 *     same fix, no magic number to re-tune per environment).
 *
 * kernel_6x8.S (this directory's copy) is unchanged from asm_kernel/ --
 * the microkernel-level work (operand swap, unrolling, prefetch, shape
 * selection) is architecture-level and already expected to transfer; only
 * the blocking and scheduling decisions above are being made portable
 * here. */
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "hw_detect.h"
#include "blocking_model.h"
#include "../../bench/bench_common.h"
#include "../../baselines/driver_common.h"

extern void kernel_6x8_asm(const double *A, const double *B, double *C,
                            long kc, long ldC_bytes);

#define MR 6
#define NR 8

/* Runtime-configurable blocking (was compile-time #define KC/MC/NC in
 * every prior version) -- set once at startup by portable_init_blocking()
 * below, not hardcoded. */
static int g_KC = 512, g_MC = 288, g_NC = 2048;
static double *g_packA = NULL, *g_packB = NULL;
static size_t g_packA_cap = 0, g_packB_cap = 0;

static void pack_panelA(const double *A, double *dst, int mr, int kc, int lda) {
    if (mr == MR) {
        for (int p = 0; p < kc; p++) {
            const double *src = A + (size_t)p * lda;
            _mm256_storeu_pd(dst, _mm256_loadu_pd(src));
            _mm_storeu_pd(dst + 4, _mm_loadu_pd(src + 4));
            dst += MR;
        }
        return;
    }
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) dst[i] = A[(size_t)p * lda + i];
        for (int i = mr; i < MR; i++) dst[i] = 0.0;
        dst += MR;
    }
}

static inline void transpose4x4_pd(__m256d r0, __m256d r1, __m256d r2, __m256d r3,
                                    __m256d *c0, __m256d *c1, __m256d *c2, __m256d *c3) {
    __m256d t0 = _mm256_unpacklo_pd(r0, r1);
    __m256d t1 = _mm256_unpackhi_pd(r0, r1);
    __m256d t2 = _mm256_unpacklo_pd(r2, r3);
    __m256d t3 = _mm256_unpackhi_pd(r2, r3);
    *c0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    *c1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    *c2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    *c3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}

static void pack_panelB(const double *B, double *dst, int nr, int kc, int ldb) {
    if (nr == NR) {
        /* Transpose-based, not gather: gather instruction throughput
         * varies a lot across microarchitectures (historically weak on
         * Haswell-class cores), while unpack+permute2f128 is a stable,
         * universally-available AVX2 idiom -- the more portable choice. */
        const double *col0 = B, *col1 = B + ldb, *col2 = B + 2 * (size_t)ldb, *col3 = B + 3 * (size_t)ldb;
        const double *col4 = B + 4 * (size_t)ldb, *col5 = B + 5 * (size_t)ldb, *col6 = B + 6 * (size_t)ldb, *col7 = B + 7 * (size_t)ldb;
        int p = 0;
        for (; p + 3 < kc; p += 4) {
            __m256d r0, r1, r2, r3, c0, c1, c2, c3;
            r0 = _mm256_loadu_pd(col0 + p); r1 = _mm256_loadu_pd(col1 + p);
            r2 = _mm256_loadu_pd(col2 + p); r3 = _mm256_loadu_pd(col3 + p);
            transpose4x4_pd(r0, r1, r2, r3, &c0, &c1, &c2, &c3);
            _mm256_storeu_pd(dst + 0 * NR, c0);
            _mm256_storeu_pd(dst + 1 * NR, c1);
            _mm256_storeu_pd(dst + 2 * NR, c2);
            _mm256_storeu_pd(dst + 3 * NR, c3);

            r0 = _mm256_loadu_pd(col4 + p); r1 = _mm256_loadu_pd(col5 + p);
            r2 = _mm256_loadu_pd(col6 + p); r3 = _mm256_loadu_pd(col7 + p);
            transpose4x4_pd(r0, r1, r2, r3, &c0, &c1, &c2, &c3);
            _mm256_storeu_pd(dst + 0 * NR + 4, c0);
            _mm256_storeu_pd(dst + 1 * NR + 4, c1);
            _mm256_storeu_pd(dst + 2 * NR + 4, c2);
            _mm256_storeu_pd(dst + 3 * NR + 4, c3);
            dst += 4 * NR;
        }
        for (; p < kc; p++) {
            for (int j = 0; j < NR; j++) dst[j] = B[(size_t)j * ldb + p];
            dst += NR;
        }
        return;
    }
    for (int p = 0; p < kc; p++) {
        for (int j = 0; j < nr; j++) dst[j] = B[(size_t)j * ldb + p];
        for (int j = nr; j < NR; j++) dst[j] = 0.0;
        dst += NR;
    }
}

static void pack_blockA(const double *A, double *packA, int mc, int kc, int lda) {
#pragma omp for schedule(static)
    for (int i = 0; i < mc; i += MR) {
        int mr = (mc - i < MR) ? mc - i : MR;
        pack_panelA(A + i, packA + (size_t)(i / MR) * MR * kc, mr, kc, lda);
    }
}

static void pack_blockB(const double *B, double *packB, int nc, int kc, int ldb) {
#pragma omp for schedule(static)
    for (int j = 0; j < nc; j += NR) {
        int nr = (nc - j < NR) ? nc - j : NR;
        pack_panelB(B + (size_t)j * ldb, packB + (size_t)(j / NR) * NR * kc, nr, kc, ldb);
    }
}

static void kernel_MRxNR(const double *A, const double *B, double *C,
                         int mr, int nr, int kc, int ldC) {
    if (mr == MR && nr == NR) {
        kernel_6x8_asm(A, B, C, (long)kc, (long)ldC * (long)sizeof(double));
        return;
    }
    for (int n = 0; n < nr; n++) {
        for (int i = 0; i < mr; i++) {
            double s = 0.0;
            for (int p = 0; p < kc; p++) s += A[(size_t)p * MR + i] * B[(size_t)p * NR + n];
            C[(size_t)n * ldC + i] += s;
        }
    }
}

static void avx2_matmul(const double *A, const double *B, double *C, int N) {
    int KC = g_KC, MC = g_MC, NC = g_NC;
    int mc_cap = MC < N ? MC : N;
    int nc_cap = NC < N ? NC : N;
    size_t needA = (size_t)mc_cap * KC + MR * KC;
    size_t needB = (size_t)nc_cap * KC + NR * KC;
    if (needA > g_packA_cap) { bench_free(g_packA); g_packA = bench_alloc(needA); g_packA_cap = needA; }
    if (needB > g_packB_cap) { bench_free(g_packB); g_packB = bench_alloc(needB); g_packB_cap = needB; }

#pragma omp parallel
    {
#pragma omp single
        memset(C, 0, (size_t)N * N * sizeof(double));

        for (int jc = 0; jc < N; jc += NC) {
            int nc = (N - jc < NC) ? N - jc : NC;
            for (int pc = 0; pc < N; pc += KC) {
                int kc = (N - pc < KC) ? N - pc : KC;
                pack_blockB(B + (size_t)jc * N + pc, g_packB, nc, kc, N);
                for (int ic = 0; ic < N; ic += MC) {
                    int mc = (N - ic < MC) ? N - ic : MC;
                    pack_blockA(A + (size_t)pc * N + ic, g_packA, mc, kc, N);

                    /* schedule(static), not a hand-picked dynamic chunk size:
                     * see the file header for why this is the portable
                     * choice rather than "dynamic,6". */
#pragma omp for collapse(2) schedule(static)
                    for (int ir = 0; ir < mc; ir += MR) {
                        for (int jr = 0; jr < nc; jr += NR) {
                            int mr = (mc - ir < MR) ? mc - ir : MR;
                            int nr = (nc - jr < NR) ? nc - jr : NR;
                            kernel_MRxNR(g_packA + (size_t)(ir / MR) * MR * kc,
                                         g_packB + (size_t)(jr / NR) * NR * kc,
                                         C + (size_t)(jc + jr) * N + (ic + ir),
                                         mr, nr, kc, N);
                        }
                    }
                }
            }
        }
    }
}

/* --- portable_init_blocking: hw_detect + analytical model + calibration --- */

static void set_blocking_cb(blocking_t b) { g_KC = b.kc; g_MC = b.mc; g_NC = b.nc; }

static double time_at_cb(int N) {
    double *A = bench_alloc((size_t)N * N);
    double *B = bench_alloc((size_t)N * N);
    double *C = bench_alloc((size_t)N * N);
    bench_init_random(A, (size_t)N * N, 1);
    bench_init_random(B, (size_t)N * N, 2);
    avx2_matmul(A, B, C, N); /* untimed warmup at this config */
    double t0 = omp_get_wtime();
    avx2_matmul(A, B, C, N);
    double t1 = omp_get_wtime();
    bench_free(A); bench_free(B); bench_free(C);
    return t1 - t0;
}

static blocking_t portable_init_blocking(int verbose) {
    hw_info_t hw = hw_detect();
    blocking_t analytical = blocking_analytical(hw, MR, NR);
    set_blocking_cb(analytical);
    blocking_t tuned = blocking_calibrate(analytical, MR, NR, set_blocking_cb, time_at_cb, 512);
    if (verbose) {
        fprintf(stderr, "# portable harness: CPU=\"%s\" cores=%d L1d=%zuK L2=%zuK L3=%zuK line=%zuB\n",
                hw.cpu_model, hw.num_cores, hw.l1d_bytes / 1024, hw.l2_bytes / 1024,
                hw.l3_bytes / 1024, hw.line_bytes);
        fprintf(stderr, "# analytical KC=%d MC=%d NC=%d -> calibrated KC=%d MC=%d NC=%d\n",
                analytical.kc, analytical.mc, analytical.nc, tuned.kc, tuned.mc, tuned.nc);
    }
    return tuned;
}

int main(int argc, char **argv) {
    driver_args_t a = parse_driver_args(argc, argv);

    portable_init_blocking(1);

    double *A = bench_alloc((size_t)a.N * a.N);
    double *B = bench_alloc((size_t)a.N * a.N);
    double *C = bench_alloc((size_t)a.N * a.N);
    bench_init_random(A, (size_t)a.N * a.N, a.seed);
    bench_init_random(B, (size_t)a.N * a.N, a.seed + 1);

    bench_compare_t cmp = {0.0, 0.0};
    if (a.verify) {
        double *Cref = bench_alloc((size_t)a.N * a.N);
        bench_reference_gemm(A, B, Cref, a.N);
        avx2_matmul(A, B, C, a.N);
        cmp = bench_compare(C, Cref, a.N);
        bench_free(Cref);
    }

    bench_result_t r = bench_run(avx2_matmul, A, B, C, a.N, a.reps);
    char variant[32];
    snprintf(variant, sizeof(variant), "avx2_6x8_portable_staticctrl");
    bench_print_csv_row(stdout, variant, a.N, omp_get_max_threads(), a.label, a.reps, r, cmp);

    bench_free(A); bench_free(B); bench_free(C);
    return 0;
}
