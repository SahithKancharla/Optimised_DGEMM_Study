#include "blocking_model.h"
#include <stdlib.h>

static int round_to_multiple(int v, int m) {
    int r = (v / m) * m;
    return r < m ? m : r;
}

blocking_t blocking_analytical(hw_info_t hw, int mr, int nr) {
    blocking_t b;
    size_t kc = (hw.l1d_bytes / 2) / ((size_t)nr * sizeof(double));
    if (kc < (size_t)mr) kc = mr; /* degenerate-cache-size guard */
    size_t mc = (hw.l2_bytes / 2) / (kc * sizeof(double));
    size_t nc = (hw.l3_bytes / 2) / (kc * sizeof(double));

    b.kc = (int)kc;
    b.mc = round_to_multiple((int)mc, mr);
    b.nc = round_to_multiple((int)nc, nr);
    return b;
}

static double try_config(blocking_t cfg, set_blocking_fn set_blocking,
                          double (*time_at)(int N), int calib_N) {
    set_blocking(cfg);
    return time_at(calib_N);
}

blocking_t blocking_calibrate(blocking_t start, int mr, int nr,
                               set_blocking_fn set_blocking,
                               double (*time_at)(int N), int calib_N) {
    /* Staged (one-dimension-at-a-time) empirical refinement around the
     * analytical starting point -- ATLAS/AEOS-style search, but scoped
     * small (3 candidates per stage) to keep first-run calibration cheap
     * (a handful of small-N GEMM calls, not a full sweep). */
    blocking_t best = start;
    double best_t;

    int kc_candidates[3] = {start.kc / 2 > 0 ? start.kc / 2 : start.kc, start.kc, start.kc * 2};
    best_t = try_config(best, set_blocking, time_at, calib_N);
    for (int i = 0; i < 3; i++) {
        if (kc_candidates[i] == best.kc) continue;
        blocking_t cand = best;
        cand.kc = kc_candidates[i] > 0 ? kc_candidates[i] : 1;
        double t = try_config(cand, set_blocking, time_at, calib_N);
        if (t < best_t) { best_t = t; best.kc = cand.kc; }
    }

    int mc_candidates[3] = {
        round_to_multiple(best.mc / 2 > 0 ? best.mc / 2 : best.mc, mr),
        best.mc,
        round_to_multiple(best.mc * 2, mr)
    };
    for (int i = 0; i < 3; i++) {
        if (mc_candidates[i] == best.mc) continue;
        blocking_t cand = best;
        cand.mc = mc_candidates[i] > 0 ? mc_candidates[i] : mr;
        double t = try_config(cand, set_blocking, time_at, calib_N);
        if (t < best_t) { best_t = t; best.mc = cand.mc; }
    }

    int nc_candidates[3] = {
        round_to_multiple(best.nc / 2 > 0 ? best.nc / 2 : best.nc, nr),
        best.nc,
        round_to_multiple(best.nc * 2, nr)
    };
    for (int i = 0; i < 3; i++) {
        if (nc_candidates[i] == best.nc) continue;
        blocking_t cand = best;
        cand.nc = nc_candidates[i] > 0 ? nc_candidates[i] : nr;
        double t = try_config(cand, set_blocking, time_at, calib_N);
        if (t < best_t) { best_t = t; best.nc = cand.nc; }
    }

    set_blocking(best);
    return best;
}
