#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

CC=gcc
CFLAGS="-O3 -march=native -fopenmp -Wall"
COMMON="../bench/bench_common.c driver_common.c"
MKLROOT=/opt/intel/oneapi/mkl/latest

mkdir -p bin

echo "== naive =="
$CC $CFLAGS $COMMON main_naive.c -o bin/driver_naive -lm

echo "== openblas =="
$CC $CFLAGS $COMMON main_openblas.c -o bin/driver_openblas -lopenblas -lm

echo "== blis =="
$CC $CFLAGS -I/usr/local/include $COMMON main_blis.c -o bin/driver_blis \
    -L/usr/local/lib -Wl,-rpath,/usr/local/lib -lblis -lm -lpthread

echo "== mkl =="
$CC $CFLAGS -I"$MKLROOT/include" $COMMON main_mkl.c -o bin/driver_mkl \
    -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" -lmkl_rt -lm -lpthread -ldl

echo "All drivers built in baselines/bin/"
