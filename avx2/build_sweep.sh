#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 gen_kernels.py

CC=gcc
CFLAGS="-O3 -march=native -mavx2 -mfma -fopenmp -Wall -Wno-unused-function"
COMMON="../bench/bench_common.c ../baselines/driver_common.c"

mkdir -p sweep/bin
tail -n +2 sweep/manifest.csv | while IFS=, read -r mr nr mrv acc regs fname; do
    name="${fname%.c}"
    echo "== building $name (accumulators=$acc, est. total regs=$regs) =="
    $CC $CFLAGS "sweep/$fname" $COMMON -o "sweep/bin/$name" -lm
done
echo "All sweep binaries built in avx2/sweep/bin/"
