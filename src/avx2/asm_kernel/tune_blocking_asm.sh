#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
N=2048
REPS=7
THREADS=16
mkdir -p tune_bin
RESULTS=tune_results_asm.csv
echo "stage,KC,MC_TARGET,NC_TARGET,gflops" > "$RESULTS"
build_run() {
    local kc=$1 mc=$2 nc=$3 stage=$4
    local tag="k${kc}_m${mc}_n${nc}"
    gcc -O3 -march=native -mavx2 -mfma -fopenmp -DKC=$kc -DMC_TARGET=$mc -DNC_TARGET=$nc \
        matmul_asm.c kernel_6x8.o ../../bench/bench_common.c ../../baselines/driver_common.c \
        -o "tune_bin/$tag" -lm 2>/tmp/tune_asm_err.log || { echo "BUILD FAIL $tag"; cat /tmp/tune_asm_err.log; return; }
    local out gflops
    out=$(OMP_NUM_THREADS=$THREADS "tune_bin/$tag" $N $REPS tune 42 0)
    gflops=$(echo "$out" | awk -F, '{print $8}')
    echo "$stage,$kc,$mc,$nc,$gflops" >> "$RESULTS"
}
for kc in 384 512 768 1024 1536; do build_run $kc 384 1024 kc_sweep; done
BEST_KC=$(awk -F, '$1=="kc_sweep"{if($5+0>best){best=$5+0;k=$2}} END{print k}' "$RESULTS")
for mc in 192 288 384 480 576 672; do build_run $BEST_KC $mc 1024 mc_sweep; done
BEST_MC=$(awk -F, '$1=="mc_sweep"{if($5+0>best){best=$5+0;m=$3}} END{print m}' "$RESULTS")
for nc in 512 1024 2048 4096; do build_run $BEST_KC $BEST_MC $nc nc_sweep; done
BEST_NC=$(awk -F, '$1=="nc_sweep"{if($5+0>best){best=$5+0;n=$4}} END{print n}' "$RESULTS")
echo "$BEST_KC,$BEST_MC,$BEST_NC" > best_blocking_asm.txt
echo "FINAL: KC=$BEST_KC MC=$BEST_MC NC=$BEST_NC"
