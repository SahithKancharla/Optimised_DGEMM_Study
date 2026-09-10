/*
 * accuracy_test.c -- numerical accuracy study (review point 8): compare
 * plain DGEMM (Strassen depth 0), Strassen depth 1/2/3, and MKL's
 * cblas_dgemm, across several input distributions, against a long-double
 * accumulation reference computed independently (not reusing any of the
 * code under test).
 *
 * Distributions:
 *   uniform     -- U(-1,1)
 *   gaussian    -- Box-Muller, mean 0 stddev 1
 *   wide_range  -- mix of ~1e-8 .. ~1e8 magnitude entries (tests scaling)
 *   ill_cond    -- rows built as near-linear-combinations of a few base
 *                  rows plus a tiny perturbation (tests near rank
 *                  deficiency)
 *   cancel      -- alternating-sign, nearly-equal-magnitude entries
 *                  specifically constructed so partial dot-product sums
 *                  pass through large intermediate values before mostly
 *                  cancelling (stresses summation-order sensitivity,
 *                  which is exactly where Strassen's extra add/subtract
 *                  steps could matter, since it changes the order and
 *                  grouping of the underlying scalar sums relative to
 *                  the textbook triple loop)
 *
 * Metrics: relative Frobenius error, max absolute componentwise relative
 * error (vs. the long-double reference), both -- the review specifically
 * asked for both, since they can disagree (Frobenius can look fine while
 * a few entries are badly wrong).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define MATMUL_LADDER2_AS_LIBRARY
#include "strassen_rebuild.c" /* pulls in matmul_ladder2.c itself */
#include "dumas_strassen_rebuild.c" /* Part L.4: Dumas/Pernet/Sedoglavic 2024
    accuracy-optimized alternate-basis Strassen (arXiv:2402.05630, Eq. 36 /
    Table 3) -- re-includes matmul_ladder2.c, safe now that it's guarded
    (see that file's header comment) */

static unsigned long rng_state;
static double rnd01(void) {
    rng_state = rng_state * 6364136223846793005UL + 1442695040888963407UL;
    return (double)((rng_state >> 11) & ((1UL<<53)-1)) / (double)(1UL<<53);
}
static double rnd_uniform(void) { return rnd01()*2.0 - 1.0; }
static double rnd_gaussian(void) {
    double u1 = rnd01() + 1e-300, u2 = rnd01();
    return sqrt(-2.0*log(u1)) * cos(2.0*M_PI*u2);
}

typedef void (*fill_fn)(double *M, int N);

static void fill_uniform(double *M, int N) {
    for (size_t i = 0; i < (size_t)N*N; i++) M[i] = rnd_uniform();
}
static void fill_gaussian(double *M, int N) {
    for (size_t i = 0; i < (size_t)N*N; i++) M[i] = rnd_gaussian();
}
static void fill_wide_range(double *M, int N) {
    for (size_t i = 0; i < (size_t)N*N; i++) {
        double mag = pow(10.0, rnd01()*16.0 - 8.0); /* 1e-8 .. 1e8 */
        M[i] = rnd_uniform() * mag;
    }
}
static void fill_ill_cond(double *M, int N) {
    int nbase = 3 < N ? 3 : N;
    double *base = malloc((size_t)nbase*N*sizeof(double));
    for (int b = 0; b < nbase; b++)
        for (int j = 0; j < N; j++) base[b*N+j] = rnd_uniform();
    for (int i = 0; i < N; i++) {
        double coeff[3];
        for (int b = 0; b < nbase; b++) coeff[b] = rnd_uniform();
        for (int j = 0; j < N; j++) {
            double v = 0.0;
            for (int b = 0; b < nbase; b++) v += coeff[b]*base[b*N+j];
            M[(size_t)i*N+j] = v + 1e-10*rnd_uniform(); /* tiny perturbation off the low-rank subspace */
        }
    }
    free(base);
}
static void fill_cancel(double *M, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            M[(size_t)i*N+j] = ((i+j) % 2 == 0 ? 1.0 : -1.0) * (1e6 + rnd_uniform());
}

typedef struct { const char *name; fill_fn fn; } dist_t;
static dist_t DISTS[] = {
    {"uniform", fill_uniform}, {"gaussian", fill_gaussian},
    {"wide_range", fill_wide_range}, {"ill_cond", fill_ill_cond}, {"cancel", fill_cancel},
};

/* independent long-double reference -- deliberately NOT calling anything
   from matmul_ladder2.c/strassen_rebuild.c, so a shared bug can't hide
   from this check */
static void reference_ld(const double *A, const double *B, double *Cref, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            long double s = 0.0L;
            for (int k = 0; k < N; k++)
                s += (long double)A[(size_t)i*N+k] * (long double)B[(size_t)k*N+j];
            Cref[(size_t)i*N+j] = (double)s;
        }
    }
}

typedef struct { double frob_rel, max_rel; } errs_t;
static errs_t compare(const double *C, const double *Cref, int N) {
    double num = 0.0, den = 0.0, maxrel = 0.0;
    for (size_t i = 0; i < (size_t)N*N; i++) {
        double d = C[i] - Cref[i];
        num += d*d;
        den += Cref[i]*Cref[i];
        double denom = fabs(Cref[i]) > 1e-300 ? fabs(Cref[i]) : 1e-300;
        double rel = fabs(d) / denom;
        if (fabs(Cref[i]) > 1e-6 && rel > maxrel) maxrel = rel; /* ignore near-zero refs, dominated by noise */
    }
    errs_t e = { sqrt(num) / (sqrt(den) > 0 ? sqrt(den) : 1.0), maxrel };
    return e;
}

#ifdef WITH_MKL
#include <mkl_cblas.h>
static void mkl_gemm(const double *A, const double *B, double *C, int N) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);
}
#endif

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 512;
    int T = argc > 2 ? atoi(argv[2]) : 8;
    unsigned seed = argc > 3 ? (unsigned)atoi(argv[3]) : 12345;

    double *A = aligned_alloc_d((size_t)N*N);
    double *B = aligned_alloc_d((size_t)N*N);
    double *Cref = aligned_alloc_d((size_t)N*N);
    double *C = aligned_alloc_d((size_t)N*N);

    printf("distribution,method,frob_rel_err,max_componentwise_rel_err\n");

    for (size_t d = 0; d < sizeof(DISTS)/sizeof(*DISTS); d++) {
        rng_state = seed;
        DISTS[d].fn(A, N);
        rng_state = seed ^ 0x9e3779b97f4a7c15UL;
        DISTS[d].fn(B, N);

        reference_ld(A, B, Cref, N);

        /* plain DGEMM (Strassen depth 0) */
        {
            strassen_times_t st;
            double *Cr = strassen_rebuild(A, B, N, N, 0, 256, 256, 1024, T, &st);
            errs_t e = compare(Cr, Cref, N);
            printf("%s,plain_dgemm,%.6e,%.6e\n", DISTS[d].name, e.frob_rel, e.max_rel);
            aligned_free_d(Cr);
        }
        /* Strassen depth 1,2,3 */
        for (int depth = 1; depth <= 3; depth++) {
            int cutoff = N; for (int x=0;x<depth;x++) cutoff/=2;
            if (cutoff < 4) continue;
            strassen_times_t st;
            double *Cr = strassen_rebuild(A, B, N, cutoff, depth, 256, 256, 1024, T, &st);
            errs_t e = compare(Cr, Cref, N);
            printf("%s,strassen_depth%d,%.6e,%.6e\n", DISTS[d].name, depth, e.frob_rel, e.max_rel);
            aligned_free_d(Cr);
        }
        /* Dumas et al. 2024 alternate-basis variant, depth 1,2,3 -- same
           cutoffs as classic Strassen above, so the two are compared at
           matched recursion depths, not just matched leaf size. */
        for (int depth = 1; depth <= 3; depth++) {
            int cutoff = N; for (int x=0;x<depth;x++) cutoff/=2;
            if (cutoff < 4) continue;
            dumas_times_t st;
            double *Cr = dumas_strassen_rebuild(A, B, N, cutoff, depth, 256, 256, 1024, T, &st);
            errs_t e = compare(Cr, Cref, N);
            printf("%s,dumas_depth%d,%.6e,%.6e\n", DISTS[d].name, depth, e.frob_rel, e.max_rel);
            aligned_free_d(Cr);
        }
#ifdef WITH_MKL
        {
            mkl_gemm(A, B, C, N);
            errs_t e = compare(C, Cref, N);
            printf("%s,mkl,%.6e,%.6e\n", DISTS[d].name, e.frob_rel, e.max_rel);
        }
#endif
    }
    aligned_free_d(A); aligned_free_d(B); aligned_free_d(Cref); aligned_free_d(C);
    return 0;
}
