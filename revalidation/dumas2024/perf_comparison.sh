#!/bin/bash
OUT=/tmp/dumas_vs_strassen_perf.csv
echo "algo,N,cutoff,max_depth,T,KC,MC,NC,reps,leaf_s,addsub_s,combine_s,workspace_s,sched_s,total_s,gflops,relerr,leaf_calls,addsub_calls" > "$OUT"
for N in 512 1024 2048; do
  for depth in 1 2 3; do
    cutoff=$N
    for x in $(seq 1 $depth); do cutoff=$((cutoff/2)); done
    if [ "$cutoff" -lt 4 ]; then continue; fi
    /tmp/strassen_main "$N" "$cutoff" "$depth" 8 5 256 256 1024 >> "$OUT"
    /tmp/dumas_main "$N" "$cutoff" "$depth" 8 5 256 256 1024 >> "$OUT"
  done
done
cat "$OUT"
