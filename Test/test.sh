#!/usr/bin/env bash
# file: run_counts.sh

set -e

# Compile the two counters
gcc -O2 -fopenmp -fstack-usage -o strassen_recusion strassen_recusion.c
gcc -O2 -fopenmp -fstack-usage -o strassen_memory   strassen_memory.c

# Run the recursion‐call counter
echo "=== Strassen Recursive Call Counts ==="
./strassen_recusion
echo

# Run the allocation counter
echo "=== Strassen Allocation & Memory Counts ==="
./strassen_memory
