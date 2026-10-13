#!/bin/bash

# Compilation step
echo "Compiling main.c with your implementations..."
gcc -O3 -march=native -mavx512f -fopenmp main.c \
    ../Parallel/mm_parallel_blocked.c \
    ../Parallel/mm_parallel_blocked_unrolled.c \
    ../Parallel/mm_parallel_strassen.c \
    ../Parallel/mm_parallel_strassen_unrolled.c \
    ../Parallel/mm_parallel_strassen_naive.c \
    ../Serial/mm_naive.c \
    ../Serial/mm_naive_unrolled.c \
    ../Serial/mm_blocked.c \
    ../Serial/mm_blocked_unrolled.c \
    ../Serial/mm_strassen.c \
    ../Serial/mm_strassen_unrolled.c \
    ../Serial/mm_strassen_naive.c \
    ../avx/matmul.c \
    ../avx/kernel.c \
    ../avx/strassen.c \
 -o matmul_tester
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi
echo "Compilation successful."

# Parameters to test
MATRIX_SIZES=(256 512 1024)
THREAD_COUNTS=(1 4 8 16 32)

# Create output directory
mkdir -p results

# Run tests
for N in "${MATRIX_SIZES[@]}"; do
    for T in "${THREAD_COUNTS[@]}"; do
        echo "Running: matrix_size=$N threads=$T"
        ./matmul_tester "$N" "$T" > "results/output_${N}_${T}.txt"
    done
done

echo "All tests completed. See the 'results' directory for output."