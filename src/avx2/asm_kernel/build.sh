#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CC=gcc
CFLAGS="-O3 -march=native -mavx2 -mfma -fopenmp -Wall -Wno-unused-function"
gcc -c kernel_6x8.S -o kernel_6x8.o
$CC $CFLAGS ${KC:+-DKC=$KC} ${MC_TARGET:+-DMC_TARGET=$MC_TARGET} ${NC_TARGET:+-DNC_TARGET=$NC_TARGET} \
    matmul_asm.c kernel_6x8.o ../../bench/bench_common.c ../../baselines/driver_common.c \
    -o matmul_asm -lm
echo "built avx2/asm_kernel/matmul_asm"
