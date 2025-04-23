#!/usr/bin/env bash
set -euo pipefail

# build_and_run.sh
# Compile the 24×6 blocked‑GEMM kernel and main, then benchmark for
# various matrix sizes and thread counts with thread affinity.

# Compilation flags
CFLAGS="-O3 -fno-math-errno -mavx512f -mavx512dq -march=skylake-avx512 -fopenmp"

# Source files
SRCS=(matmul.c kernel.c main.c strassen.c)

# Executable name
EXE=matmul_bench

# Matrix sizes to test
SIZES=(64 128 256 512 1024 2048 4096 8192 16384)

# Thread counts to test (use one per physical core)
THREADS=(8 16 32 64)

# Affinity policy: bind each thread to a core, scattering across sockets
export KMP_AFFINITY="verbose,granularity=core,scatter"

echo "Compiling sources: ${SRCS[*]}"
gcc $CFLAGS "${SRCS[@]}" -o $EXE

echo "Running benchmarks with affinity: $KMP_AFFINITY"
for N in "${SIZES[@]}"; do
  for T in "${THREADS[@]}"; do
    echo
    echo "===== Size: ${N}×${N}, Threads: ${T} ====="
    # set thread count for this run
    export OMP_NUM_THREADS=${T}
    # run benchmark
    OMP_NUM_THREADS=${T} ./${EXE} "${N}" "${T}"
  done
done
