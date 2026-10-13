#!/usr/bin/env bash
# Staged (one-dimension-at-a-time) KC/MC/NC tuning around cache-derived
# starting points for this CPU (P-core: L1d=48K, L2=2048K, L3=30720K shared).
# Not a full grid search -- greedy staged sweep, as scoped in the plan.
set -euo pipefail
cd "$(dirname "$0")"

CC=gcc
CFLAGS="-O3 -march=native -mavx2 -mfma -fopenmp -Wall -Wno-unused-function"
COMMON="../bench/bench_common.c ../baselines/driver_common.c"
N=2048
REPS=7
THREADS=16
mkdir -p tune_bin
RESULTS=tune_results.csv
echo "stage,KC,MC_TARGET,NC_TARGET,gflops" > "$RESULTS"

build_run() {
    local kc=$1 mc=$2 nc=$3 stage=$4
    local tag="k${kc}_m${mc}_n${nc}"
    $CC $CFLAGS -DKC=$kc -DMC_TARGET=$mc -DNC_TARGET=$nc \
        matmul_4x8.c $COMMON -o "tune_bin/$tag" -lm 2>/tmp/tune_err.log \
        || { echo "BUILD FAIL $tag"; cat /tmp/tune_err.log; return; }
    local out
    out=$(OMP_NUM_THREADS=$THREADS "tune_bin/$tag" $N $REPS tune 42 0)
    local gflops
    gflops=$(echo "$out" | awk -F, '{print $8}')
    echo "$stage,$kc,$mc,$nc,$gflops" >> "$RESULTS"
    echo "[$stage] KC=$kc MC_TARGET=$mc NC_TARGET=$nc -> $gflops GFLOP/s"
}

echo "=== Stage 1: sweep KC (MC=384, NC=4096 fixed) ==="
BEST_KC=384
for kc in 192 256 384 512 768 1024; do
    build_run $kc 384 4096 kc_sweep
done
BEST_KC=$(awk -F, '$1=="kc_sweep"{if($5+0>best){best=$5+0; k=$2}} END{print k}' "$RESULTS")
[ -z "$BEST_KC" ] && { echo "FATAL: no successful KC builds, aborting"; exit 1; }
echo ">>> best KC = $BEST_KC"

echo "=== Stage 2: sweep MC (KC=$BEST_KC, NC=4096 fixed) ==="
for mc in 128 192 256 384 480 576 672 768; do
    build_run $BEST_KC $mc 4096 mc_sweep
done
BEST_MC=$(awk -F, -v kc="$BEST_KC" '$1=="mc_sweep"{if($5+0>best){best=$5+0; m=$3}} END{print m}' "$RESULTS")
[ -z "$BEST_MC" ] && { echo "FATAL: no successful MC builds, aborting"; exit 1; }
echo ">>> best MC_TARGET = $BEST_MC"

echo "=== Stage 3: sweep NC (KC=$BEST_KC, MC_TARGET=$BEST_MC fixed) ==="
for nc in 1024 2048 3072 4096 6144 8192; do
    build_run $BEST_KC $BEST_MC $nc nc_sweep
done
BEST_NC=$(awk -F, -v kc="$BEST_KC" -v mc="$BEST_MC" '$1=="nc_sweep"{if($5+0>best){best=$5+0; n=$4}} END{print n}' "$RESULTS")
echo ">>> best NC_TARGET = $BEST_NC"

echo ""
echo "=== FINAL: KC=$BEST_KC MC_TARGET=$BEST_MC NC_TARGET=$BEST_NC ==="
echo "$BEST_KC,$BEST_MC,$BEST_NC" > best_blocking.txt
