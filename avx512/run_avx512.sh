#!/usr/bin/env bash
# run_avx512.sh -- AVX-512 pipeline: correctness + n=30 benchmark + results folder.
#
# Builds the audited paper's real AVX-512 path (audited_submission/avx/
# kernel.c + matmul.c + strassen.c), checks correctness, then benchmarks it
# AND the AVX2 port on the same machine under the project's standard protocol
# (PROTOCOL.md): n=30 independent process launches per (N,T) point, each
# launch = best of --reps inner reps after a warm-up, AVX-512/AVX2 launches
# interleaved (alternating order) to cancel thermal/background drift.
# Statistics: mean, sd, CV, bootstrap 95% CI, AVX-512/AVX2 ratio CI,
# Holm-adjusted permutation test (tools/analyze_n30.py).
#
# Output: revalidation/benchmarks/<date>_<host>-<cpu>-avx512/
#   metadata.txt n30_samples.csv n30_summary.csv n30_comparison.csv
#   avx512.csv avx2.csv (mean seconds, legacy schema) correctness_avx512.txt
#   run_info.csv raw_output.txt lscpu.txt bench_avx512.c
# plus one row appended to results/avx512_run_log.csv.
#
# Usage: bash avx512/run_avx512.sh [--runs 30] [--reps 5] [--sizes 256,512,..]
#                                  [--threads 1,4,8] [--tag label] [--smoke]
#   --smoke = 3 runs, small grid: a wiring check, NOT publication-grade.
# Needs: Linux/WSL2, gcc + OpenMP, python3, a CPU with avx512f.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AVX="$ROOT/audited_submission/avx"
HERE="$ROOT/avx512"
CC="${CC:-gcc}"

RUNS=30; REPS=5; SIZES=""; THREADS=""; TAG=""; SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --full) ;;   # accepted for reproduce.ps1 compatibility; the grid is already full
    --runs) RUNS="$2"; shift ;;
    --reps) REPS="$2"; shift ;;
    --sizes) SIZES="$2"; shift ;;
    --threads) THREADS="$2"; shift ;;
    --tag) TAG="$2"; shift ;;
    --smoke) SMOKE=1; RUNS=3; REPS=2; SIZES="${SIZES:-256,512}" ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
MODE="n${RUNS}"; [ "$SMOKE" = 1 ] && MODE="SMOKE-n${RUNS}(not-publication-grade)"

# ---------------------------------------------------------------- preflight
if [ "$(uname -s)" != "Linux" ]; then
  echo "ERROR: run this on Linux or inside WSL2 (found $(uname -s))." >&2; exit 1
fi
command -v "$CC" >/dev/null || { echo "ERROR: $CC not found (sudo apt install build-essential)" >&2; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 not found (sudo apt install python3)" >&2; exit 1; }
if ! grep -qw avx512f /proc/cpuinfo; then
  echo "ERROR: this CPU/VM does not expose avx512f -- nothing to run here." >&2
  echo "       (On a VM/WSL, the host may hide it. Use ./reproduce.sh for the AVX2 path.)" >&2
  exit 3
fi
for f in kernel.c matmul.c strassen.c kernel_avx2.c strassen_avx2.c test_local.c; do
  [ -f "$AVX/$f" ] || { echo "ERROR: missing $AVX/$f" >&2; exit 1; }
done

NPROC=$(nproc)
[ -z "$SIZES" ] && SIZES="256,512,1024,2048,4096"
if [ -z "$THREADS" ]; then   # 1, powers of two below nproc, then nproc
  THREADS=1; p=2
  while [ "$p" -lt "$NPROC" ]; do THREADS="$THREADS,$p"; p=$((p * 2)); done
  [ "$NPROC" -gt 1 ] && THREADS="$THREADS,$NPROC"
fi
IFS=, read -ra SZ <<< "$SIZES"; IFS=, read -ra TH <<< "$THREADS"
LAUNCHES=$(( ${#SZ[@]} * ${#TH[@]} * RUNS * 2 ))

# ---------------------------------------------------------------- machine id
CPU_MODEL=$(grep -m1 "model name" /proc/cpuinfo | sed 's/.*: //')
CPU_SLUG=$(echo "$CPU_MODEL" | sed -E 's/\((R|TM)\)//g; s/@.*//; s/\b(CPU|Processor|Intel|Core|AMD)\b//g; s/[^A-Za-z0-9]+/-/g; s/^-+|-+$//g')
HOST=$(hostname | tr -cd 'A-Za-z0-9-')
OUT="$ROOT/revalidation/benchmarks/$(date +%F)_${HOST}-${CPU_SLUG}-avx512${TAG:+-$TAG}"
[ -e "$OUT" ] && OUT="${OUT}_$(date +%H%M%S)"
BUILD="$ROOT/results/avx512_build"
mkdir -p "$OUT" "$BUILD"
TS=$(date +%Y%m%d_%H%M%S)

PHYS=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
RAM_GB=$(awk '/MemTotal/{printf "%.0f", $2/1048576}' /proc/meminfo)
OS_NAME=$( (. /etc/os-release && echo "$PRETTY_NAME") 2>/dev/null || echo Linux)
grep -qi microsoft /proc/version && OS_NAME="$OS_NAME [WSL2]"
KERNEL=$(uname -r)
CC_V=$("$CC" --version | head -1)
AVX512_FLAGS=$(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep '^avx512' | sort -u | tr '\n' ' ')
GIT_COMMIT="n/a"; GIT_DIRTY="n/a"
if git -C "$ROOT" rev-parse --short HEAD >/dev/null 2>&1; then
  GIT_COMMIT=$(git -C "$ROOT" rev-parse --short HEAD)
  GIT_DIRTY=$([ -z "$(git -C "$ROOT" status --porcelain)" ] && echo clean || echo "dirty (local edits present)")
fi
lscpu > "$OUT/lscpu.txt" 2>&1 || true
cp "$HERE/bench_avx512.c" "$OUT/bench_avx512.c"

CF512="${CF512:--O3 -fno-math-errno -march=skylake-avx512 -fopenmp}"   # = audited run.sh
CF2="${CF2:--O3 -fno-math-errno -mavx2 -mfma -fopenmp}"

write_metadata () {  # $1 = correctness verdict
cat > "$OUT/metadata.txt" <<EOF
run_date: $(date +%F) ($TS)
repo_commit: $GIT_COMMIT ($GIT_DIRTY)
host: $HOST  ($OS_NAME, kernel $KERNEL)
cpu: $CPU_MODEL
cores: $PHYS physical / $NPROC logical, ${RAM_GB} GB RAM
isa_available: $AVX512_FLAGS
compiler: $CC_V
build_flags_avx512: $CF512
build_flags_avx2: $CF2
driver: bench_avx512.c (same driver for both ISAs; Freivalds correctness check in every launch)
protocol: $MODE -- $RUNS independent process launches per (N,T) point, each = best of $REPS inner reps after 1 warm-up; ISA order alternated per launch (PROTOCOL.md)
sizes_tested: $SIZES
threads_tested: $THREADS
total_launches: $LAUNCHES
correctness_suite: $1  (test_local.c against kernel.c, see correctness_avx512.txt)
statistics: tools/analyze_n30.py -- bootstrap 95% CI (10000 resamples, seed 12345), permutation test, Holm correction
caveats: laptop/VM runs throttle -- plug in, high-performance power mode, close other apps (CV>5% points are flagged).
  WSL2/VM timers and core scheduling add noise (FINDINGS.md Part A). Threads are not pinned (OS scheduler decides).
  Not the audited paper's machine (dual Xeon Gold 5218) -- absolute GFLOP/s will differ.
EOF
}

echo "== Machine: $CPU_MODEL  ($NPROC threads)"
echo "== AVX-512 flags: $AVX512_FLAGS"
echo "== Protocol: $MODE; $LAUNCHES process launches; sizes $SIZES; threads $THREADS; inner reps $REPS"
echo "== Output: $OUT"

# ---------------------------------------------------------------- build
echo; echo "== Build =="
"$CC" $CF512 "$AVX/matmul.c" "$AVX/kernel.c" "$AVX/strassen.c" "$AVX/test_local.c" -o "$BUILD/test_avx512" -lm
"$CC" $CF512 "$AVX/matmul.c" "$AVX/kernel.c" "$AVX/strassen.c" "$HERE/bench_avx512.c" -o "$BUILD/bench_avx512" -lm
"$CC" $CF2   "$AVX/matmul.c" "$AVX/kernel_avx2.c" "$AVX/strassen_avx2.c" "$HERE/bench_avx512.c" -o "$BUILD/bench_avx2" -lm
echo "Build OK."

# ---------------------------------------------------------------- correctness
echo; echo "== Correctness (test_local.c on the AVX-512 kernel) =="
set +e
"$BUILD/test_avx512" 2>&1 | tee "$OUT/correctness_avx512.txt"
RC=${PIPESTATUS[0]}
set -e
if [ "$RC" -ne 0 ]; then
  write_metadata FAIL
  echo "ERROR: correctness suite FAILED -- not benchmarking a wrong kernel. See $OUT/correctness_avx512.txt" >&2
  exit 1
fi
write_metadata PASS

# ---------------------------------------------------------------- n=30 sampling
echo; echo "== Sampling ($LAUNCHES launches) =="
SAMPLES="$OUT/n30_samples.csv"
HEADER_DONE=0; DONE=0
for N in "${SZ[@]}"; do
  for T in "${TH[@]}"; do
    for ((r = 1; r <= RUNS; r++)); do
      if (( r % 2 )); then ORDER="avx512 avx2"; else ORDER="avx2 avx512"; fi
      for isa in $ORDER; do
        OUTTXT=$("$BUILD/bench_$isa" "$isa" "$N" "$T" "$REPS" 2>>"$OUT/raw_output.txt")
        if [ "$HEADER_DONE" = 0 ]; then echo "run,$(echo "$OUTTXT" | head -1)" > "$SAMPLES"; HEADER_DONE=1; fi
        echo "$r,$(echo "$OUTTXT" | tail -1)" >> "$SAMPLES"
        DONE=$((DONE + 1))
      done
    done
    echo "  N=$N T=$T done ($DONE/$LAUNCHES launches)"
  done
done

# ---------------------------------------------------------------- statistics
echo; echo "== Statistics =="
python3 "$ROOT/tools/analyze_n30.py" "$OUT" | tee -a "$OUT/raw_output.txt"
# shellcheck disable=SC1090
source "$OUT/n30_headline.env"
VERDICT=PASS; [ "${BAD:-0}" -ne 0 ] && VERDICT="FAIL(${BAD} groups relerr>1e-8)"
[ "$SMOKE" = 1 ] && VERDICT="$VERDICT;SMOKE"

HDR="timestamp,hostname,os,kernel,cpu_model,physical_cores,logical_cores,ram_gb,compiler,compiler_version,git_commit,git_status,protocol,runs_per_point,inner_reps,avx512_flags,headline_N,headline_T,blocked_avx512_gflops_mean,blocked_avx512_ci95_lo,blocked_avx512_ci95_hi,strassen_avx512_gflops_mean,blocked_avx2_gflops_mean,strassen_avx2_gflops_mean,avx512_over_avx2_blocked,ratio_ci_lo,ratio_ci_hi,ratio_p_holm,noisy_groups,correctness"
ROW="$TS,$HOST,\"$OS_NAME\",$KERNEL,\"$CPU_MODEL\",$PHYS,$NPROC,$RAM_GB,$CC,\"$CC_V\",$GIT_COMMIT,$GIT_DIRTY,$MODE,$RUNS,$REPS,\"$AVX512_FLAGS\",$HN,$HT,$B512,$B512_LO,$B512_HI,$S512,$B2,$S2,${RATIO:-n/a},${RATIO_LO:-n/a},${RATIO_HI:-n/a},${PHOLM:-n/a},$NOISY,$VERDICT"
{ echo "$HDR"; echo "$ROW"; } > "$OUT/run_info.csv"
[ -f "$ROOT/results/avx512_run_log.csv" ] || { mkdir -p "$ROOT/results"; echo "$HDR" > "$ROOT/results/avx512_run_log.csv"; }
echo "$ROW" >> "$ROOT/results/avx512_run_log.csv"

echo
echo "================================================================"
echo " SUMMARY  (N=$HN, T=$HT, n=$RUNS)    correctness: $VERDICT"
echo "   AVX-512 : blocked ${B512} GFLOP/s [95% CI ${B512_LO}, ${B512_HI}]   strassen ${S512}"
echo "   AVX2    : blocked ${B2} GFLOP/s   strassen ${S2}"
echo "   AVX-512 / AVX2 (blocked): ${RATIO:-n/a}x [${RATIO_LO:-?}, ${RATIO_HI:-?}]  Holm p=${PHOLM:-n/a}"
echo "   noisy groups (CV>5%): $NOISY"
echo " Run folder : $OUT"
echo " Run log    : results/avx512_run_log.csv"
echo " Merge machines: python3 tools/combine_results.py"
echo "================================================================"
