#include "distributions.h"
#include <math.h>

static double xrand(unsigned *state) {
    *state ^= *state << 13; *state ^= *state >> 17; *state ^= *state << 5;
    return (double)(*state) / 4294967295.0;
}

void dist_uniform(double *M, size_t n, unsigned seed) {
    unsigned s = seed ? seed : 1u;
    for (size_t i = 0; i < n; i++) M[i] = xrand(&s) * 2.0 - 1.0;
}

void dist_wide_dynamic_range(double *M, size_t n_rows, size_t n_cols, unsigned seed) {
    unsigned s = seed ? seed : 1u;
    for (size_t r = 0; r < n_rows; r++) {
        double exponent = (xrand(&s) * 12.0) - 6.0; /* -6 .. +6 */
        double scale = pow(10.0, exponent);
        for (size_t c = 0; c < n_cols; c++) {
            double v = xrand(&s) * 2.0 - 1.0;
            M[c * n_rows + r] = v * scale; /* column-major */
        }
    }
}

void dist_clustered_cancellation(double *M, size_t n, unsigned seed) {
    unsigned s = seed ? seed : 1u;
    for (size_t i = 0; i < n; i++) {
        double perturb = (xrand(&s) * 2.0 - 1.0) * 1e-6;
        M[i] = 1.0 + perturb;
    }
}
