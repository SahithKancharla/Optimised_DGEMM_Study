#!/usr/bin/env bash
# rerun_n30_dumasperf.sh -- n=30/bootstrap-CI refresh of tab:dumas-perf
# (paper/main.tex): classical Strassen vs Dumas 2024, N=2048, T=8,
# depth in {1,2,3}, matching the original table's exact configuration.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="$(pwd)"
OUT="revalidation/dumas2024/n30_rerun_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT/bin"

CFLAGS="-O2 -mavx2 -mfma -fopenmp"
gcc $CFLAGS src/avx2/ladder/strassen_main.c -o "$OUT/bin/strassen_main" -lm
gcc $CFLAGS src/avx2/ladder/dumas_main.c    -o "$OUT/bin/dumas_main"    -lm

N=2048
T=8
KC=256; MC=256; NC=1024
RUNS=30

for depth in 1 2 3; do
  cutoff=$N
  for ((d = 0; d < depth; d++)); do cutoff=$((cutoff / 2)); done

  echo "== strassen depth=$depth cutoff=$cutoff =="
  echo "depth,gflops" > "$OUT/strassen_depth${depth}.csv"
  for ((r = 1; r <= RUNS; r++)); do
    line=$("$OUT/bin/strassen_main" "$N" "$cutoff" "$depth" "$T" 5 "$KC" "$MC" "$NC")
    gflops=$(echo "$line" | awk -F, '{print $(NF-3)}')
    echo "$depth,$gflops" >> "$OUT/strassen_depth${depth}.csv"
  done
  python3 tools/sample_stats.py "$OUT/strassen_depth${depth}.csv" gflops \
    | awk -v d="$depth" '{printf "  strassen depth=%s mean=%-8s 95%%CI=[%s, %s] CV=%s%% n=%s\n", d, $1, $2, $3, $4, $5}'

  echo "== dumas depth=$depth cutoff=$cutoff =="
  echo "depth,gflops" > "$OUT/dumas_depth${depth}.csv"
  for ((r = 1; r <= RUNS; r++)); do
    line=$("$OUT/bin/dumas_main" "$N" "$cutoff" "$depth" "$T" 5 "$KC" "$MC" "$NC")
    gflops=$(echo "$line" | awk -F, '{print $(NF-3)}')
    echo "$depth,$gflops" >> "$OUT/dumas_depth${depth}.csv"
  done
  python3 tools/sample_stats.py "$OUT/dumas_depth${depth}.csv" gflops \
    | awk -v d="$depth" '{printf "  dumas    depth=%s mean=%-8s 95%%CI=[%s, %s] CV=%s%% n=%s\n", d, $1, $2, $3, $4, $5}'
done
echo "Done. Raw CSVs in $OUT/"
