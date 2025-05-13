#!/usr/bin/env bash
set -euo pipefail

# Compilation flags
CFLAGS="-O3 -fno-math-errno -march=skylake-avx512 -fopenmp"

# Source files
SRCS=(matmul.c kernel.c main.c strassen.c)

# Executable name
EXE=matmul_bench

# Matrix sizes to test
SIZES=(64 128 256 512 1024 2048 4096 8192 16384)
# SIZES=( 256 )

# Thread counts to test
THREADS=(8 16 32 64)
# THREADS=(64)

# Affinity policy: bind each thread to a core, scattering across sockets
export KMP_AFFINITY="verbose,granularity=core,scatter"

# Compile the executable
echo "Compiling sources: ${SRCS[*]}"
gcc $CFLAGS "${SRCS[@]}" -o $EXE

# Loop over thread counts
for T in "${THREADS[@]}"; do
    RESULT_FILE="results_avx_${T}.txt"
    CACHE_FILE="cache_avx_${T}.csv"

    # Clear previous files
    > "$RESULT_FILE"
    > "$CACHE_FILE"

    # CSV header for cache metrics
    echo "Size,Threads,Cache Misses,Cache References,LLC Loads,LLC Load Misses,LLC Stores,LLC Store Misses" >> "$CACHE_FILE"

    echo "=========================================" | tee -a "$RESULT_FILE"
    echo "Running experiments with $T threads" | tee -a "$RESULT_FILE"
    echo "=========================================" | tee -a "$RESULT_FILE"

    for N in "${SIZES[@]}"; do
        echo "Matrix size: ${N}x${N}" | tee -a "$RESULT_FILE"

        # Set thread count
        export OMP_NUM_THREADS=${T}

        # Run benchmark and capture output
        OMP_NUM_THREADS=${T} ./${EXE} "$N" "$T" | tee -a "$RESULT_FILE"

        # Use perf to gather cache stats
        PERF_OUTPUT=$(perf stat -e cache-misses,cache-references,\
LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses \
-x, -- ./${EXE} "$N" "$T" 2>&1 >/dev/null)

        # Extract metrics
        CACHE_MISSES=$(echo "$PERF_OUTPUT" | awk -F, '/cache-misses/ {gsub(/ /, "", $1); print $1}')
        CACHE_REFS=$(echo "$PERF_OUTPUT" | awk -F, '/cache-references/ {gsub(/ /, "", $1); print $1}')
        LLC_LOADS=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-loads/ {gsub(/ /, "", $1); print $1}')
        LLC_LOAD_MISSES=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-load-misses/ {gsub(/ /, "", $1); print $1}')
        LLC_STORES=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-stores/ {gsub(/ /, "", $1); print $1}')
        LLC_STORE_MISSES=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-store-misses/ {gsub(/ /, "", $1); print $1}')

        # Append to cache CSV
        echo "$N,$T,$CACHE_MISSES,$CACHE_REFS,$LLC_LOADS,$LLC_LOAD_MISSES,$LLC_STORES,$LLC_STORE_MISSES" >> "$CACHE_FILE"
        echo "---------------------------------------------" | tee -a "$RESULT_FILE"
    done

    echo "Results saved to $RESULT_FILE"
    echo "Cache stats saved to $CACHE_FILE"
    echo ""
done
