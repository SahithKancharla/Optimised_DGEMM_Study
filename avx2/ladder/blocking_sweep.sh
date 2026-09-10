#!/bin/bash
# One-parameter-at-a-time blocking response surface: KC, then MC, then NC,
# holding the other two at a reasonable default, at N=1024, T=8, static.
BIN=./matmul_ladder2_gcc
N=1024
T=8
echo "param,value,gflops,total_s,alloc_s,pack_a_s,pack_b_s,kernel_compute_s,kernel_write_s,cache_refs,cache_misses,L1_loads,L1_load_misses,dTLB_loads,dTLB_load_misses"

run_and_perf() {
  local param="$1" val="$2" kc="$3" mc="$4" nc="$5"
  local line=$("$BIN" $N $N $N $T static 0 5 $kc $mc $nc 0)
  # field:  1       2 3 4 5 6     7     8    9  10 11 12 13    14     15     16       17      18    19     20
  # ladder2,M,      K,N,T,sched,chunk,reps,KC,MC,NC,hp,alloc,pack_a,pack_b,kcompute,kwrite,total,gflops,relerr
  local gflops=$(echo "$line" | cut -d, -f19)
  local total=$(echo "$line" | cut -d, -f18)
  local alloc=$(echo "$line" | cut -d, -f13)
  local pa=$(echo "$line" | cut -d, -f14)
  local pb=$(echo "$line" | cut -d, -f15)
  local kc_t=$(echo "$line" | cut -d, -f16)
  local kw_t=$(echo "$line" | cut -d, -f17)
  local perfout=$(perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,dTLB-loads,dTLB-load-misses -x, -- "$BIN" $N $N $N $T static 0 3 $kc $mc $nc 0 2>&1 >/dev/null)
  local cr=$(echo "$perfout" | awk -F, '/cache-references/{print $1}')
  local cm=$(echo "$perfout" | awk -F, '/cache-misses/{print $1}')
  local l1l=$(echo "$perfout" | awk -F, '/L1-dcache-loads:/{print $1}')
  local l1m=$(echo "$perfout" | awk -F, '/L1-dcache-load-misses/{print $1}')
  local tl=$(echo "$perfout" | awk -F, '/dTLB-loads:/{print $1}')
  local tm=$(echo "$perfout" | awk -F, '/dTLB-load-misses/{print $1}')
  echo "$param,$val,$gflops,$total,$alloc,$pa,$pb,$kc_t,$kw_t,$cr,$cm,$l1l,$l1m,$tl,$tm"
}

for kc in 128 256 384 512 768 1024 2048; do
  run_and_perf KC $kc $kc 256 1024
done
for mc in 64 128 256 384 512 768 1024; do
  run_and_perf MC $mc 512 $mc 1024
done
for nc in 256 512 1024 2048 4096; do
  run_and_perf NC $nc 512 256 $nc
done
