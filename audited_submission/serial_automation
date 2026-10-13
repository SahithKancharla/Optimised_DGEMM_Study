#!/bin/bash

# Output files
RESULT_FILE="results.txt"
CACHE_FILE="cache_metrics.csv"
> "$RESULT_FILE"    # Clear previous result logs
> "$CACHE_FILE"     # Clear previous cache stats

# CSV Header for perf
echo "Algorithm,Size,Cache Misses,Cache References,LLC Loads,LLC Load Misses,LLC Stores,LLC Store Misses" >> "$CACHE_FILE"

# Compile the serial implementation
echo "Compiling serial version..." | tee -a "$RESULT_FILE"
if gcc -fopenmp -O3 -march=native -ffast-math \
    Serial/main.c \
    Serial/mm_naive.c Serial/mm_naive_unrolled.c \
    Serial/mm_blocked.c Serial/mm_blocked_unrolled.c \
    Serial/mm_strassen.c Serial/mm_strassen_unrolled.c \
    Serial/mm_strassen_naive.c \
    -o serial_mult; then
    echo "Compilation successful." | tee -a "$RESULT_FILE"
else
    echo "Compilation failed." | tee -a "$RESULT_FILE"
    exit 1
fi

SIZES=(64 128 256 512 1024 2048)
ALGORITHMS=("naive" "naive_unrolled" "blocked" "blocked_unrolled" "strassen" "strassen_base" "strassen_base_unrolled")
# ALGORITHMS=("naive" "naive_unrolled" "blocked" "blocked_unrolled" "strassen_naive")
# ALGORITHMS=("strassen" "strassen_unrolled")

# Run experiments
echo "------------------------------------------------------------------------" | tee -a "$RESULT_FILE"
for SIZE in "${SIZES[@]}"; do
    echo "Matrix size: ${SIZE}x${SIZE}" | tee -a "$RESULT_FILE"
    for ALG in "${ALGORITHMS[@]}"; do
        echo "Running $ALG algorithm..." | tee -a "$RESULT_FILE"

        # Run the algorithm and collect runtime
        ./serial_mult 1 "$SIZE" "$SIZE" "$SIZE" "$ALG" >> "$RESULT_FILE"

        # Use perf to gather cache stats
        PERF_OUTPUT=$(perf stat -e cache-misses,cache-references,\
LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses \
-x, -- ./serial_mult 1 "$SIZE" "$SIZE" "$SIZE" "$ALG" 2>&1 >/dev/null)

        # Extract metrics from perf output
        CACHE_MISSES=$(echo "$PERF_OUTPUT" | awk -F, '/cache-misses/ {gsub(/ /, "", $1); print $1}')
        CACHE_REFS=$(echo "$PERF_OUTPUT" | awk -F, '/cache-references/ {gsub(/ /, "", $1); print $1}')
        LLC_LOADS=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-loads/ {gsub(/ /, "", $1); print $1}')
        LLC_LOAD_MISSES=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-load-misses/ {gsub(/ /, "", $1); print $1}')
        LLC_STORES=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-stores/ {gsub(/ /, "", $1); print $1}')
        LLC_STORE_MISSES=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-store-misses/ {gsub(/ /, "", $1); print $1}')

        # Append to cache CSV
        echo "$ALG,$SIZE,$CACHE_MISSES,$CACHE_REFS,$LLC_LOADS,$LLC_LOAD_MISSES,$LLC_STORES,$LLC_STORE_MISSES" >> "$CACHE_FILE"

        echo "--------------------------------------------" | tee -a "$RESULT_FILE"
    done
    echo "" | tee -a "$RESULT_FILE"
done

echo "All results saved to $RESULT_FILE"
echo "Cache stats saved to $CACHE_FILE"





# echo "------------------------------------------------------------------------" | tee -a "$RESULT_FILE"
# for SIZE in "${SIZES[@]}"; do
#     echo "Running Naive on ${SIZE}x${SIZE}"
#     ./serial_mult 1 "$SIZE" "$SIZE" "$SIZE" "naive" >> "$RESULT_FILE"
#     echo "--------------------------------------------" | tee -a "$RESULT_FILE"
# done

# echo "------------------------------------------------------------------------" | tee -a "$RESULT_FILE"
# for SIZE in "${SIZES[@]}"; do
#     echo "Running Naive on ${SIZE}x${SIZE}"
#     ./serial_mult 1 "$SIZE" "$SIZE" "$SIZE" "naive_unrolled" >> "$RESULT_FILE"
#     echo "--------------------------------------------" | tee -a "$RESULT_FILE"
# done

# echo "------------------------------------------------------------------------" | tee -a "$RESULT_FILE"
# for SIZE in "${SIZES[@]}"; do
#     echo "Running Naive on ${SIZE}x${SIZE}"
#     ./serial_mult 1 "$SIZE" "$SIZE" "$SIZE" "blocked" >> "$RESULT_FILE"
#     echo "--------------------------------------------" | tee -a "$RESULT_FILE"
# done

# echo "------------------------------------------------------------------------" | tee -a "$RESULT_FILE"
# for SIZE in "${SIZES[@]}"; do
#     echo "Running Naive on ${SIZE}x${SIZE}"
#     ./serial_mult 1 "$SIZE" "$SIZE" "$SIZE" "strassen" >> "$RESULT_FILE"
#     echo "--------------------------------------------" | tee -a "$RESULT_FILE"
# done




# SIZES=(2048 4096 8192)
# ALGORITHMS=("strassen" "strassen_unrolled")

# # Run experiments
# echo "------------------------------------------------------------------------" | tee -a "$RESULT_FILE"
# for SIZE in "${SIZES[@]}"; do
#     echo "Matrix size: ${SIZE}x${SIZE}" | tee -a "$RESULT_FILE"
#     for ALG in "${ALGORITHMS[@]}"; do
#         echo "Running $ALG algorithm..." | tee -a "$RESULT_FILE"

#         # Run the algorithm and collect runtime
#         ./serial_mult 1 "$SIZE" "$SIZE" "$SIZE" "$ALG" >> "$RESULT_FILE"

#         # Use perf to gather cache stats
#         PERF_OUTPUT=$(perf stat -e cache-misses,cache-references,\
# LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses \
# -x, -- ./serial_mult 1 "$SIZE" "$SIZE" "$SIZE" "$ALG" 2>&1 >/dev/null)

#         # Extract metrics from perf output
#         CACHE_MISSES=$(echo "$PERF_OUTPUT" | awk -F, '/cache-misses/ {gsub(/ /, "", $1); print $1}')
#         CACHE_REFS=$(echo "$PERF_OUTPUT" | awk -F, '/cache-references/ {gsub(/ /, "", $1); print $1}')
#         LLC_LOADS=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-loads/ {gsub(/ /, "", $1); print $1}')
#         LLC_LOAD_MISSES=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-load-misses/ {gsub(/ /, "", $1); print $1}')
#         LLC_STORES=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-stores/ {gsub(/ /, "", $1); print $1}')
#         LLC_STORE_MISSES=$(echo "$PERF_OUTPUT" | awk -F, '/LLC-store-misses/ {gsub(/ /, "", $1); print $1}')

#         # Append to cache CSV
#         echo "$ALG,$SIZE,$CACHE_MISSES,$CACHE_REFS,$LLC_LOADS,$LLC_LOAD_MISSES,$LLC_STORES,$LLC_STORE_MISSES" >> "$CACHE_FILE"

#         echo "--------------------------------------------" | tee -a "$RESULT_FILE"
#     done
#     echo "" | tee -a "$RESULT_FILE"
# done

# echo "All results saved to $RESULT_FILE"
# echo "Cache stats saved to $CACHE_FILE"