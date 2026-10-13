#!/usr/bin/env bash
set -euo pipefail

# Builds and runs test_local.c against the real matmul.c (packing/blocking
# logic) using the AVX2 micro-kernel port (kernel_avx2.c), so matmul.c
# changes can be validated for correctness on machines without AVX-512 --
# e.g. before trusting them on the AVX-512 target machine. Same packed
# buffer contract, so a pass here is strong evidence the AVX-512 build
# (kernel.c, run.sh) is correct too. Also links strassen_avx2.c so
# test_local.c's run_strassen() actually exercises the Strassen+AVX2 path
# (previously this build never referenced strassen_avx2.c at all, so
# "ALL PASS" gave no evidence the Strassen path worked).

CFLAGS="-O3 -mavx2 -mfma -fopenmp"
gcc $CFLAGS matmul.c kernel_avx2.c strassen_avx2.c test_local.c -o test_local_avx2 -lm
./test_local_avx2
