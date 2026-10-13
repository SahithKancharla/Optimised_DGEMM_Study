#!/bin/bash

SIZES=(64 128 256 512 1024 2048)
ALGORITHMS=("blocked" "blocked_unrolled" "strassen" "strassen_base" "strassen_base_unrolled")
THREAD_COUNTS=(6 7 8 16 32 48 49 50 64)

# Compile the parallel implementation
echo "Compiling parallel version..."
if gcc -fopenmp -O3 -march=native -ffast-math \
    Parallel/main.c \
    Parallel/mm_parallel_blocked.c \
    Parallel/mm_parallel_blocked_unrolled.c \
    Parallel/mm_parallel_strassen.c \
    Parallel/mm_parallel_strassen_unrolled.c \
    Parallel/mm_parallel_strassen_naive.c \
    -o parallel_mult; then
    echo "Compilation successful."
else
    echo "Compilation failed."
    exit 1
fi

# Loop over thread counts
for THREADS in "${THREAD_COUNTS[@]}"; do
    RESULT_FILE="results_parallel_${THREADS}.txt"
    CACHE_FILE="cache_matrix_parallel_${THREADS}.csv"
    
    # Clear previous files
    > "$RESULT_FILE"
    > "$CACHE_FILE"
    
    # CSV header
    echo "Algorithm,Size,Threads,Cache Misses,Cache References,LLC Loads,LLC Load Misses,LLC Stores,LLC Store Misses" >> "$CACHE_FILE"
    
    echo "==========================================" | tee -a "$RESULT_FILE"
    echo "Running experiments with $THREADS threads" | tee -a "$RESULT_FILE"
    echo "==========================================" | tee -a "$RESULT_FILE"

    for SIZE in "${SIZES[@]}"; do
        echo "Matrix size: ${SIZE}x${SIZE}" | tee -a "$RESULT_FILE"
        for ALG in "${ALGORITHMS[@]}"; do

            echo "Running $ALG algorithm with $THREADS threads..." | tee -a "$RESULT_FILE"

            # Run the algorithm and collect runtime
            ./parallel_mult "$THREADS" "$SIZE" "$SIZE" "$SIZE" "$ALG" >> "$RESULT_FILE"

            # Use perf to gather cache stats
            PERF_OUTPUT=$(perf stat -e cache-misses,cache-references,\
LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses \
-x, -- ./parallel_mult "$THREADS" "$SIZE" "$SIZE" "$SIZE" "$ALG" 2>&1 >/dev/null)

            # Extract metrics
            CACHE_MISSES=$(echo "$PERF_OUTPUT" | awk -F, '/cache-misses/ {gsub(/ /, "", $1); print $1}')
            CACHE_REFS=$(echo "$PERF_OUTPUT" | awk -F, '/cache-references/ {gsub(/ /, "", $1); print $1}')
            LLC_LOADS=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-loads/ {gsub(/ /, "", $1); print $1}')
            LLC_LOAD_MISSES=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-load-misses/ {gsub(/ /, "", $1); print $1}')
            LLC_STORES=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-stores/ {gsub(/ /, "", $1); print $1}')
            LLC_STORE_MISSES=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-store-misses/ {gsub(/ /, "", $1); print $1}')

            # Append to cache CSV
            echo "$ALG,$SIZE,$THREADS,$CACHE_MISSES,$CACHE_REFS,$LLC_LOADS,$LLC_LOAD_MISSES,$LLC_STORES,$LLC_STORE_MISSES" >> "$CACHE_FILE"
            echo "--------------------------------------------" | tee -a "$RESULT_FILE"
        done
        echo "" | tee -a "$RESULT_FILE"
    done

    echo "Results saved to $RESULT_FILE"
    echo "Cache stats saved to $CACHE_FILE"
    echo ""
done
