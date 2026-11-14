#!/usr/bin/env bash
# rerun_n30_crossover.sh -- n=30/bootstrap-CI refresh of
# tab:strassen-crossover (paper/main.tex): dgemm_base vs strassen_1level,
# effective GFLOP/s, N in {512,1024,2048,4096,8192}. verify=0 (correctness
# already established elsewhere at small N; large-N long-double reference
# here would dominate runtime for no new information).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT="results_n30_crossover_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

gcc -O2 -mavx2 -mfma -fopenmp dgemm_base.c strassen.c strassen_ops.c \
  kernel_6x8.S bench_strassen.c ../bench/bench_common.c \
  -o "$OUT/bench_strassen" -lm -I.

RUNS=30
for N in 512 1024 2048 4096 8192; do
  echo "== N=$N (n=$RUNS, dgemm_base + strassen_1level) =="
  echo "variant,N,threads,best_s,gflops,rel_frob,rel_max" > "$OUT/dgemm_${N}.csv"
  echo "variant,N,threads,best_s,gflops,rel_frob,rel_max" > "$OUT/strassen1_${N}.csv"
  for ((r = 1; r <= RUNS; r++)); do
    line=$("$OUT/bench_strassen" "$N" 5 "$((100 + r))" 0)
    echo "$line" | grep '^dgemm_base,' >> "$OUT/dgemm_${N}.csv"
    echo "$line" | grep '^strassen_1level,' >> "$OUT/strassen1_${N}.csv"
  done
  python3 ../../tools/sample_stats.py "$OUT/dgemm_${N}.csv" gflops \
    | awk -v n="$N" '{printf "  N=%-5s dgemm_base      mean=%-8s 95%%CI=[%s, %s] CV=%s%% n=%s\n", n, $1, $2, $3, $4, $5}'
  python3 ../../tools/sample_stats.py "$OUT/strassen1_${N}.csv" gflops \
    | awk -v n="$N" '{printf "  N=%-5s strassen_1level mean=%-8s 95%%CI=[%s, %s] CV=%s%% n=%s\n", n, $1, $2, $3, $4, $5}'
done
echo "Done. Raw CSVs in $OUT/"
