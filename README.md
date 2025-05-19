# Dense Matrix Multiplication Benchmark Suite

This repository contains implementations of serial, parallel, and AVX-512-enhanced matrix multiplication algorithms, along with scripts to benchmark their performance and collect system-level metrics.

## Directory Structure

- `Serial/` – Serial implementations of matrix multiplication.
- `Parallel/` – OpenMP-based parallel implementations.
- `avx/` – AVX-512 optimized implementations.
- `Strassen_BreakDown/` – Scripts to analyze Strassen's algorithm recursion and memory usage.
- `Values/` – Temporary directory used during execution (may contain intermediate values).
- `Results/` – Temporary directory used to store output and timing data.
- `visualisations/` – Scripts to generate performance plots.
- `Data/` – Generated runtime data and traces.
- `test/` – Constain the sanity tests to compare the serial implementation output with the other cases.

## Main executables to Run
- `full_run.sh` – Main automation script to run both serial and parallel. Run using `./full_run.sh`.
- `serial_automation.sh` – Script to benchmark serial variants only. Run using `./serial_automation.sh`.
- `parallel_automation.sh` – Script to benchmark parallel variants only. Run using `./parallel_automation.sh`.
- `./avx/run.sh` – Script to benchmark  avx code. Run using `cd ./avx` and then `./run.sh`.
- `./Strassen_BreakDown/run.sh` – Runs the strassen memory and recurssion tests. Run using `cd Strassen_BreakDown` and then `./run.sh`.
- `./test/run.sh` – Runs all the test cases to see if the output of all the code is same to the serial execution. Run using `cd ./test` and then `./run.sh`.

## How to Run

1. **Ensure you are on a Linux system with AVX-512 and OpenMP support**.
2. **Grant execution permissions** if needed to executables:
   ```bash
   chmod +x full_run.sh serial_automation.sh parallel_automation.sh
   chmod +x avx/run.sh Strassen_BreakDown/run.sh test/run.sh 
   ```

You can download or view the PDF here:  
[Download the PDF](./Advance_Parallel_Programming_Dense_Analysis.pdf)