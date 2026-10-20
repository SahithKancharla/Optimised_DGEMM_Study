#!/usr/bin/env python3
"""sample_stats.py <csv> <value_col> [max_col] -- n=30 stats for one sample column.

Prints one line: mean ci95_lo ci95_hi cv_pct n [max_of_max_col]
(bootstrap 95% CI of the mean, 10000 resamples, seed 12345; stdlib only).
Used by reproduce.sh / avx512 pipeline so every run follows PROTOCOL.md.
"""
import csv, math, random, statistics, sys

path, col = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(path)))
x = [float(r[col]) for r in rows]
n, mean = len(x), statistics.mean(x)
sd = statistics.stdev(x) if n > 1 else 0.0
rng = random.Random(12345)
bm = sorted(sum(rng.choices(x, k=n)) / n for _ in range(10000))
lo, hi = bm[int(0.025 * 9999)], bm[int(0.975 * 9999)]
out = [f"{mean:.3f}", f"{lo:.3f}", f"{hi:.3f}", f"{100 * sd / mean if mean else 0:.2f}", str(n)]
if len(sys.argv) > 3:
    out.append(f"{max(float(r[sys.argv[3]]) for r in rows):.3e}")
print(" ".join(out))
