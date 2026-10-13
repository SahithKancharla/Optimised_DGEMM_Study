#ifndef BLOCKING_MODEL_H
#define BLOCKING_MODEL_H

#include "hw_detect.h"

typedef struct {
    int kc, mc, nc;
} blocking_t;

/* Goto/van de Geijn + BLIS-style analytical blocking: derive KC/MC/NC from
 * DETECTED cache sizes (hw_detect.h) rather than a constant copied from one
 * machine's empirical sweep. KC sized so a KC x NR micro-panel of B uses
 * about half of L1 (leaving room for A's micro-panel); MC sized so an MC x
 * KC block of A uses about half of L2; NC sized so an NC x KC block of B
 * uses about half of L3. Rounded to multiples of mr/nr. This is a portable
 * STARTING POINT, refined empirically by blocking_calibrate() below. */
blocking_t blocking_analytical(hw_info_t hw, int mr, int nr);

/* gemm_fn signature matches bench_common.h's gemm_fn: void(const double*A,
 * const double*B, double*C, int N). Calibration applies a candidate
 * blocking via set_blocking (caller-supplied, since KC/MC/NC live as
 * runtime globals in the matmul translation unit, not here), times a
 * small problem with it, and keeps the best of a handful of candidates
 * around the analytical starting point -- the ATLAS/AEOS empirical-search
 * paradigm, done at first-run time on whatever machine this actually
 * executes on, not baked in from one development machine. */
typedef void (*set_blocking_fn)(blocking_t b);
typedef void (*timed_gemm_fn)(int N); /* runs+times matmul at size N, returns nothing;
                                          caller reads elapsed via its own clock */

blocking_t blocking_calibrate(blocking_t start, int mr, int nr,
                               set_blocking_fn set_blocking,
                               double (*time_at)(int N), int calib_N);

#endif
