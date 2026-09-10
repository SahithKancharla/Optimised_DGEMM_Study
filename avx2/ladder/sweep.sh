#!/bin/bash
# Usage: bash sweep.sh <binary> <env_label> > out.csv
BIN="$1"
ENV="$2"
echo "env,binary,stage,N,T,schedule,chunk,reps,pack_a_s,pack_b_s,kernel_s,total_s,gflops,relerr"
for N in 512 1024 2048; do
  for T in 1 4 8; do
    for SCHED in static dynamic guided; do
      CHUNK=4
      [ "$SCHED" = "static" ] && CHUNK=0
      [ "$SCHED" = "guided" ] && CHUNK=0
      LINE=$("$BIN" full "$N" "$T" "$SCHED" "$CHUNK" 5)
      echo "$ENV,$LINE"
    done
  done
done
