#!/usr/bin/env bash
# run_n30.sh -- n=30 independent launches per PROTOCOL.md for each
# production-BLAS baseline (naive, OpenBLAS, BLIS, MKL), at the same
# N/T as reproduce.sh --full's classical-DGEMM leg, so the two are
# directly comparable. Thread count comes from each library's own env
# var (see driver_common.c); CPU affinity/placement is left to the OS,
# same as reproduce.sh's own classical-DGEMM leg.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# MKL's rpath is baked into driver_mkl at build time (see build.sh), so no
# vars.sh sourcing needed -- that would also clobber $1/$2/$3 below.

N="${1:-2048}"
T="${2:-$(nproc)}"
RUNS="${3:-30}"
OUT="results_n30_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

run_variant () {
  local label="$1" exe="$2" envvar="$3"
  echo "== $label (N=$N T=$T n=$RUNS) =="
  echo "variant,N,threads,label,reps,median_s,min_s,gflops,rel_frob,rel_max" > "$OUT/${label}.csv"
  for ((r = 1; r <= RUNS; r++)); do
    env "$envvar=$T" "./bin/$exe" "$N" 5 "run$r" "$((42 + r))" 0 >> "$OUT/${label}.csv"
  done
  python3 ../../tools/sample_stats.py "$OUT/${label}.csv" gflops \
    | awk -v l="$label" '{printf "  %-10s mean=%-8s 95%%CI=[%s, %s] CV=%s%% n=%s\n", l, $1, $2, $3, $4, $5}'
}

run_variant naive    driver_naive    OMP_NUM_THREADS
run_variant openblas driver_openblas OPENBLAS_NUM_THREADS
run_variant blis     driver_blis     BLIS_NUM_THREADS
run_variant mkl      driver_mkl      MKL_NUM_THREADS

echo "Done. Raw CSVs + stats in $OUT/"
