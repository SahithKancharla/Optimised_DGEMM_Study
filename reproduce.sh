#!/usr/bin/env bash
# reproduce.sh -- clone-and-run harness for Linux/macOS/WSL (gcc,
# AVX2+FMA+OpenMP): classical DGEMM, classic Strassen (Part H/J.7),
# Dumas et al. 2024 Strassen (Part M), accuracy sweep (Part I/J.8/M.4).
# See REPRODUCIBILITY.md for scope/limits.
#
# Usage: ./reproduce.sh [--full] [--runs 30] [--smoke]   (default n=30 per PROTOCOL.md)
# Needs gcc (`apt install build-essential`). No WSL/gcc on Windows:
# use reproduce.ps1 instead (narrower coverage, see its header).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="quick"
RUNS=30   # project standard (PROTOCOL.md): n=30 independent process launches per measurement
while [ $# -gt 0 ]; do
  case "$1" in
    --full) MODE="full" ;;
    --runs) RUNS="$2"; shift ;;
    --smoke) RUNS=3 ;;   # wiring check only -- NOT publication-grade
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 needed for the n=$RUNS statistics (sudo apt install python3)"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUT="results/reproduce_${TS}"
mkdir -p "$OUT/bin"

CC="${CC:-gcc}"
NPROC=$(command -v nproc >/dev/null 2>&1 && nproc || sysctl -n hw.ncpu 2>/dev/null || echo 4)
# Was capped at min(NPROC,8): measured real throughput left on the table
# above 8 threads on a 24-thread machine (336->447 GFLOP/s at T=8->16).
T="${THREADS:-$NPROC}"
[ "$T" -lt 1 ] && T=1

if [ "$MODE" = "full" ]; then
  DGEMM_N=2048; DGEMM_REPS=10
  STRASSEN_N=1024; STRASSEN_DEPTH=3; STRASSEN_REPS=5
  ACC_N=512
else
  DGEMM_N=1024; DGEMM_REPS=5
  STRASSEN_N=512; STRASSEN_DEPTH=2; STRASSEN_REPS=3
  ACC_N=256
fi
# cutoff = N >> depth (matches revalidation/dumas2024/perf_comparison.sh);
# a fixed cutoff truncates recursion to a shallower effective depth.
STRASSEN_CUTOFF=$STRASSEN_N
for ((i = 0; i < STRASSEN_DEPTH; i++)); do STRASSEN_CUTOFF=$((STRASSEN_CUTOFF / 2)); done
# KC/MC/NC: the validated values from revalidation/dumas2024, not a guess.
STRASSEN_KC=256; STRASSEN_MC=256; STRASSEN_NC=1024

echo "================================================================"
echo " Optimised_DGEMM_Study reproducibility harness (mode: $MODE)"
echo " CPU threads available: $NPROC, using T=$T"
echo " Output: $OUT/"
echo "================================================================"

# System/run metadata, so N runs across different machines can be
# concatenated into one comparison table (run_info.csv / run_log.csv below).
UNAME_S=$(uname -s 2>/dev/null || echo unknown)
HOSTNAME_V=$(hostname 2>/dev/null || echo unknown)
KERNEL_V=$(uname -r 2>/dev/null || echo unknown)
IS_WSL="no"
if [ -f /proc/version ] && grep -qi microsoft /proc/version 2>/dev/null; then IS_WSL="yes (WSL)"; fi

if [ "$UNAME_S" = "Darwin" ]; then
  OS_NAME="$(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null)"
  CPU_MODEL=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)
  PHYS_CORES=$(sysctl -n hw.physicalcpu 2>/dev/null || echo unknown)
  RAM_GB=$(awk "BEGIN{printf \"%.0f\", $(sysctl -n hw.memsize 2>/dev/null || echo 0)/1073741824}")
else
  OS_NAME=$( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || echo "Linux (unknown distro)")
  CPU_MODEL=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | sed 's/.*: //' || echo unknown)
  PHYS_CORES=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l || echo unknown)
  RAM_KB=$(grep -m1 MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
  RAM_GB=$(awk "BEGIN{printf \"%.0f\", ${RAM_KB:-0}/1048576}")
fi
[ "$IS_WSL" != "no" ] && OS_NAME="$OS_NAME [$IS_WSL -- see FINDINGS.md Part A on WSL2 timer trustworthiness]"

GIT_COMMIT="n/a (not a git checkout)"
GIT_DIRTY="n/a"
if git -C "$SCRIPT_DIR" rev-parse --short HEAD >/dev/null 2>&1; then
  GIT_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse --short HEAD)
  GIT_DIRTY=$([ -z "$(git -C "$SCRIPT_DIR" status --porcelain 2>/dev/null)" ] && echo clean || echo "dirty (local edits present)")
fi

{
  echo "run_timestamp: $TS"
  echo "hostname: $HOSTNAME_V"
  echo "os: $OS_NAME"
  echo "kernel: $KERNEL_V"
  echo "cpu_model: $CPU_MODEL"
  echo "physical_cores: $PHYS_CORES"
  echo "logical_cores: $NPROC"
  echo "ram_gb: $RAM_GB"
  echo "compiler: ${CC}"
  echo "git_commit: $GIT_COMMIT"
  echo "git_status: $GIT_DIRTY"
  echo "mode: $MODE"
} | tee "$OUT/system_info.txt"

echo
echo "== Preflight =="
if ! command -v "$CC" >/dev/null 2>&1; then
  echo "ERROR: '$CC' not found."
  echo "  Linux: sudo apt install build-essential  (or your distro's equivalent)"
  echo "  macOS: this project's OpenMP/task usage needs real GNU gcc, not"
  echo "         Apple's clang shim -- 'brew install gcc' and set CC=gcc-14"
  echo "         (or whatever version brew installs) before re-running."
  echo "  Windows without WSL: use reproduce.ps1 instead."
  exit 1
fi
COMPILER_V=$("$CC" --version | head -1)
echo "Using $COMPILER_V"
echo "compiler_version: $COMPILER_V" >> "$OUT/system_info.txt"

CFLAGS="-O2 -mavx2 -mfma -fopenmp"

echo
echo "== Building (gcc $CFLAGS) =="
"$CC" $CFLAGS src/avx2/ladder/matmul_ladder.c   -o "$OUT/bin/matmul_ladder"  -lm
"$CC" $CFLAGS src/avx2/ladder/strassen_main.c   -o "$OUT/bin/strassen_main" -lm
"$CC" $CFLAGS src/avx2/ladder/dumas_main.c      -o "$OUT/bin/dumas_main"    -lm
"$CC" $CFLAGS src/avx2/ladder/accuracy_test.c   -o "$OUT/bin/accuracy_test" -lm
# Hand-tuned asm kernel (dgemm_base/strassen_1level/2level): measured
# faster than matmul_ladder's kernel above (478 vs 447 GFLOP/s, N=2048
# T=16 on the dev machine) -- built separately so both are reported.
"$CC" $CFLAGS src/strassen2/dgemm_base.c src/strassen2/strassen.c \
  src/strassen2/strassen_ops.c src/strassen2/kernel_6x8.S \
  src/strassen2/bench_strassen.c src/bench/bench_common.c \
  -o "$OUT/bin/bench_strassen" -lm
echo "Build OK."

echo
echo "== Hardware sanity check (smallest possible AVX2+FMA+OpenMP run) =="
set +e
PROBE=$("$OUT/bin/matmul_ladder" full 128 1 static 0 1 2>&1)
PROBE_RC=$?
set -e
if [ $PROBE_RC -ne 0 ]; then
  echo "ERROR: smoke run failed (exit $PROBE_RC). Output:"
  echo "$PROBE"
  echo "Most likely cause: this CPU does not support AVX2+FMA (required"
  echo "everywhere in this project -- see FINDINGS.md Part A), or gcc's"
  echo "OpenMP runtime (libgomp) is missing."
  exit 1
fi
echo "OK -- AVX2+FMA+OpenMP all work on this machine."

run_step () {
  echo
  echo "== $1 =="
}

# ---------------------------------------------------------------------
run_step "1/4 Classical DGEMM: correctness + performance (matmul_ladder.c)"
echo "run,gflops,relerr" > "$OUT/matmul_ladder_samples.csv"
for ((r = 1; r <= RUNS; r++)); do
  LINE=$("$OUT/bin/matmul_ladder" full "$DGEMM_N" "$T" guided 0 "$DGEMM_REPS")
  echo "$LINE" > "$OUT/matmul_ladder.csv"   # last raw line kept for reference
  echo "$r,$(echo "$LINE" | awk -F, '{print $(NF-1)","$NF}')" >> "$OUT/matmul_ladder_samples.csv"
done
read -r DGEMM_GFLOPS DGEMM_LO DGEMM_HI DGEMM_CV _ DGEMM_RELERR < <(python3 tools/sample_stats.py "$OUT/matmul_ladder_samples.csv" gflops relerr)
echo "  N=$DGEMM_N T=$T  n=$RUNS  GFLOP/s: $DGEMM_GFLOPS [95% CI $DGEMM_LO, $DGEMM_HI] CV ${DGEMM_CV}%   worst relerr: $DGEMM_RELERR"

# ---------------------------------------------------------------------
run_step "2/4 Classic Strassen rebuild: correctness + performance (strassen_main.c, Part H/J.7)"
STRASSEN_RC=0
echo "run,gflops,relerr" > "$OUT/strassen_samples.csv"
for ((r = 1; r <= RUNS; r++)); do
  set +e
  LINE=$("$OUT/bin/strassen_main" "$STRASSEN_N" "$STRASSEN_CUTOFF" "$STRASSEN_DEPTH" "$T" "$STRASSEN_REPS" "$STRASSEN_KC" "$STRASSEN_MC" "$STRASSEN_NC")
  RC=$?
  set -e
  [ $RC -ne 0 ] && STRASSEN_RC=$RC
  echo "$LINE" > "$OUT/strassen.csv"
  echo "$r,$(echo "$LINE" | awk -F, '{print $(NF-3)","$(NF-2)}')" >> "$OUT/strassen_samples.csv"
done
read -r STRASSEN_GFLOPS STRASSEN_LO STRASSEN_HI STRASSEN_CV _ STRASSEN_RELERR < <(python3 tools/sample_stats.py "$OUT/strassen_samples.csv" gflops relerr)
if [ $STRASSEN_RC -eq 0 ]; then
  echo "  correctness: PASS in all $RUNS runs (worst relerr=$STRASSEN_RELERR)   GFLOP/s: $STRASSEN_GFLOPS [95% CI $STRASSEN_LO, $STRASSEN_HI] CV ${STRASSEN_CV}%"
else
  echo "  correctness: FAIL in >=1 of $RUNS runs (worst relerr=$STRASSEN_RELERR exceeds 1e-6) -- see FINDINGS.md Part H.2"
fi

# ---------------------------------------------------------------------
run_step "3/4 Dumas et al. (2024) alternate-basis Strassen: correctness + performance (dumas_main.c, Part M)"
DUMAS_RC=0
echo "run,gflops,relerr" > "$OUT/dumas_samples.csv"
for ((r = 1; r <= RUNS; r++)); do
  set +e
  LINE=$("$OUT/bin/dumas_main" "$STRASSEN_N" "$STRASSEN_CUTOFF" "$STRASSEN_DEPTH" "$T" "$STRASSEN_REPS" "$STRASSEN_KC" "$STRASSEN_MC" "$STRASSEN_NC")
  RC=$?
  set -e
  [ $RC -ne 0 ] && DUMAS_RC=$RC
  echo "$LINE" > "$OUT/dumas.csv"
  echo "$r,$(echo "$LINE" | awk -F, '{print $(NF-3)","$(NF-2)}')" >> "$OUT/dumas_samples.csv"
done
read -r DUMAS_GFLOPS DUMAS_LO DUMAS_HI DUMAS_CV _ DUMAS_RELERR < <(python3 tools/sample_stats.py "$OUT/dumas_samples.csv" gflops relerr)
if [ $DUMAS_RC -eq 0 ]; then
  echo "  correctness: PASS in all $RUNS runs (worst relerr=$DUMAS_RELERR)   GFLOP/s: $DUMAS_GFLOPS [95% CI $DUMAS_LO, $DUMAS_HI] CV ${DUMAS_CV}%"
else
  echo "  correctness: FAIL in >=1 of $RUNS runs (worst relerr=$DUMAS_RELERR exceeds 1e-6) -- see FINDINGS.md Part M.1"
fi

# ---------------------------------------------------------------------
run_step "4/5 Numerical accuracy sweep, 5 distributions x {plain,Strassen depth1-3,Dumas depth1-3} (accuracy_test.c)"
"$OUT/bin/accuracy_test" "$ACC_N" "$T" 12345 | tee "$OUT/accuracy.csv"

# ---------------------------------------------------------------------
run_step "5/5 Best-known kernel: dgemm_base + strassen_1level/2level (bench_strassen.c)"
echo "run,gflops" > "$OUT/bs_dgemm_samples.csv"; echo "run,gflops" > "$OUT/bs_s1_samples.csv"
for ((r = 1; r <= RUNS; r++)); do
  OMP_NUM_THREADS="$T" "$OUT/bin/bench_strassen" "$DGEMM_N" "$DGEMM_REPS" 42 1 > "$OUT/bench_strassen.csv"
  awk -F, -v r="$r" '$1=="dgemm_base"{print r","$5}' "$OUT/bench_strassen.csv" >> "$OUT/bs_dgemm_samples.csv"
  awk -F, -v r="$r" '$1=="strassen_1level"{print r","$5}' "$OUT/bench_strassen.csv" >> "$OUT/bs_s1_samples.csv"
done
cat "$OUT/bench_strassen.csv"   # last of $RUNS raw outputs; the n=$RUNS stats are in the summary

# ---------------------------------------------------------------------
echo
echo "================================================================"
echo " SUMMARY  (full detail + raw CSVs in $OUT/)"
echo "================================================================"
echo "Classical DGEMM   : relerr=$DGEMM_RELERR   GFLOP/s=$DGEMM_GFLOPS"
echo "Classic Strassen  : relerr=$STRASSEN_RELERR   GFLOP/s=$STRASSEN_GFLOPS   $([ $STRASSEN_RC -eq 0 ] && echo PASS || echo FAIL)"
echo "Dumas (2024)      : relerr=$DUMAS_RELERR   GFLOP/s=$DUMAS_GFLOPS   $([ $DUMAS_RC -eq 0 ] && echo PASS || echo FAIL)"
read -r BS_DGEMM_GFLOPS BS_D_LO BS_D_HI _ < <(python3 tools/sample_stats.py "$OUT/bs_dgemm_samples.csv" gflops)
read -r BS_S1_GFLOPS BS_S_LO BS_S_HI _ < <(python3 tools/sample_stats.py "$OUT/bs_s1_samples.csv" gflops)
echo "Best-known kernel : dgemm_base=${BS_DGEMM_GFLOPS} GFLOP/s [${BS_D_LO}, ${BS_D_HI}]   strassen_1level=${BS_S1_GFLOPS} GFLOP/s [${BS_S_LO}, ${BS_S_HI}]   (n=$RUNS, mean + 95% CI)"
if command -v awk >/dev/null 2>&1 && [ "$STRASSEN_RC" -eq 0 ] && [ "$DUMAS_RC" -eq 0 ]; then
  COST=$(awk -v s="$STRASSEN_GFLOPS" -v d="$DUMAS_GFLOPS" 'BEGIN{ if (s>0) printf "%.1f", (1-d/s)*100; else print "n/a" }')
  echo
  echo "Dumas-vs-classic-Strassen throughput cost on this machine: ${COST}%"
  echo "  (paper Section 6.5: ~20.7% at N=2048/depth=3 on the dev machine;"
  echo "   this run used N=$STRASSEN_N/depth=$STRASSEN_DEPTH, so expect a"
  echo "   different magnitude -- the DIRECTION is what reproduces.)"
else
  COST="n/a"
fi
echo

# Comparison-ready row: collect $OUT/run_info.csv from every machine you
# run this on and concatenate them (same header) into one table.
RUN_INFO_HEADER="timestamp,hostname,os,kernel,cpu_model,physical_cores,logical_cores,ram_gb,compiler,compiler_version,git_commit,git_status,mode,dgemm_N,dgemm_T,dgemm_gflops,dgemm_relerr,strassen_gflops,strassen_relerr,strassen_pass,dumas_gflops,dumas_relerr,dumas_pass,dumas_cost_pct,runs_per_point,dgemm_ci95_lo,dgemm_ci95_hi,dgemm_cv_pct,strassen_ci95_lo,strassen_ci95_hi,dumas_ci95_lo,dumas_ci95_hi"
RUN_INFO_ROW="$TS,$HOSTNAME_V,\"$OS_NAME\",$KERNEL_V,\"$CPU_MODEL\",$PHYS_CORES,$NPROC,$RAM_GB,$CC,\"$COMPILER_V\",$GIT_COMMIT,$GIT_DIRTY,$MODE,$DGEMM_N,$T,$DGEMM_GFLOPS,$DGEMM_RELERR,$STRASSEN_GFLOPS,$STRASSEN_RELERR,$([ $STRASSEN_RC -eq 0 ] && echo pass || echo fail),$DUMAS_GFLOPS,$DUMAS_RELERR,$([ $DUMAS_RC -eq 0 ] && echo pass || echo fail),$COST,$RUNS,$DGEMM_LO,$DGEMM_HI,$DGEMM_CV,$STRASSEN_LO,$STRASSEN_HI,$DUMAS_LO,$DUMAS_HI"
{ echo "$RUN_INFO_HEADER"; echo "$RUN_INFO_ROW"; } | tee "$OUT/run_info.csv" > /dev/null
# header grew (n=30 columns appended): don't mix with an older-format log
if [ -f results/run_log.csv ] && [ "$(head -1 results/run_log.csv)" != "$RUN_INFO_HEADER" ]; then
  mv results/run_log.csv "results/run_log_pre_n30_$TS.csv"
fi
[ -f results/run_log.csv ] || echo "$RUN_INFO_HEADER" > results/run_log.csv
echo "$RUN_INFO_ROW" >> results/run_log.csv
echo "Run info: $OUT/run_info.csv (also appended to results/run_log.csv)"
echo
echo "What this script does NOT reproduce (by design -- see REPRODUCIBILITY.md):"
echo "  - the full statistical-rigor (n=4-5+ repeat) scheduling/placement"
echo "    experiment behind paper Section 5 (needs Windows thread pinning"
echo "    -- see src/avx2/ladder/pinning_experiment.csv and reproduce.ps1)"
echo "  - the original paper's AVX-512 kernel (separate pipeline: bash avx512/run_avx512.sh,"
echo "    or reproduce.ps1 -Avx512; needs a CPU with avx512f -- see avx512/README.md)"
echo "  - production-BLAS baseline comparisons (needs OpenBLAS/BLIS/MKL"
echo "    installed separately -- see src/baselines/build.sh)"
echo "  - the full multi-hour N/thread-count sweeps behind every table"
echo "    in the paper (this is a fast smoke test of the same binaries,"
echo "    not a re-run of every data point)"
