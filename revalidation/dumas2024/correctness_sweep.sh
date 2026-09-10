#!/bin/bash
for N in 64 128 256 512; do
  for depth in 1 2 3; do
    cutoff=$((N/2))
    /tmp/dumas_main "$N" "$cutoff" "$depth" 4 3 256 256 1024
  done
done
