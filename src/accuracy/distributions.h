#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H
#include <stddef.h>

/* Three input distributions to stress numerical accuracy differently.
 * Strassen does extra additions/subtractions beyond classical GEMM, so
 * its error behavior can diverge from classical DGEMM specifically under
 * cancellation-prone inputs -- these distributions are chosen to expose
 * that rather than just re-testing the easy uniform case. */

/* Uniform in [-1, 1]. Baseline, well-conditioned case. */
void dist_uniform(double *M, size_t n, unsigned seed);

/* Each row scaled by a random power-of-10 factor spanning ~12 orders of
 * magnitude (1e-6 .. 1e6) -- simulates a poorly-scaled/ill-conditioned
 * input, the kind of matrix where summation order matters a lot. */
void dist_wide_dynamic_range(double *M, size_t n_rows, size_t n_cols, unsigned seed);

/* Values clustered tightly around 1.0 with small perturbations -- designed
 * to produce heavy cancellation in the +/- combinations Strassen's
 * recursive step introduces (e.g. A11+A22 when both are close to a
 * shared value), which classical GEMM's single accumulation loop never
 * does. */
void dist_clustered_cancellation(double *M, size_t n, unsigned seed);

#endif
