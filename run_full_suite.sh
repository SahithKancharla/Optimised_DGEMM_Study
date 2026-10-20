#!/usr/bin/env bash
# run_full_suite.sh -- the complete benchmark set, under PROTOCOL.md (n=30), sequentially:
#   1. AVX-512 vs AVX2, full grid (N=256..4096 x T=1,2,4,8), 1200 launches
#   2. Portable AVX2 path (reproduce.sh --full): classical DGEMM, classic Strassen,
#      Dumas 2024, best-known kernel -- 30 launches each; accuracy sweep once
#   3. Merge all machines' data into results/combined/
# Sequential on purpose: concurrent runs would contaminate each other's timings.
# Progress for monitoring: results/suite_progress.txt (one line per event) and
# results/suite_logs/*.log.   Usage (inside WSL, from anywhere): bash run_full_suite.sh
# Before starting: plug in, Windows power mode = High performance, close other apps.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p results/suite_logs
PROG=results/suite_progress.txt
log() { echo "$(date +%F_%T) $*" | tee -a "$PROG"; }

: > "$PROG"
log "SUITE START host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //')"

if grep -qw avx512f /proc/cpuinfo; then
  log "STAGE 1/3 START avx512 full grid (1200 launches)"
  bash avx512/run_avx512.sh --tag n30-full > results/suite_logs/1_avx512_full.log 2>&1
  rc=$?
  log "STAGE 1/3 END rc=$rc"
else
  log "STAGE 1/3 SKIPPED (no avx512f on this CPU)"
fi

log "STAGE 2/3 START reproduce.sh --full (n=30)"
bash reproduce.sh --full > results/suite_logs/2_reproduce_full.log 2>&1
rc=$?
log "STAGE 2/3 END rc=$rc"

log "STAGE 3/3 START combine"
python3 tools/combine_results.py > results/suite_logs/3_combine.log 2>&1
rc=$?
log "STAGE 3/3 END rc=$rc"
log "SUITE DONE"
