"""Win-count + mean/median summary for row_scaled_seeds_raw.csv."""
import csv
from collections import defaultdict

rows = list(csv.DictReader(open('row_scaled_seeds_raw.csv')))
by_depth = defaultdict(lambda: defaultdict(dict))
for r in rows:
    by_depth[int(r['depth'])][r['method']][r['seed']] = (float(r['frob_rel']), float(r['max_rel']))

for depth in [1, 2, 3]:
    s, d = by_depth[depth]['strassen'], by_depth[depth]['dumas']
    seeds = list(s.keys())
    n = len(seeds)
    frob_wins = sum(1 for sd in seeds if d[sd][0] < s[sd][0])
    max_wins = sum(1 for sd in seeds if d[sd][1] < s[sd][1])
    maxs_s = sorted(s[sd][1] for sd in seeds)
    maxs_d = sorted(d[sd][1] for sd in seeds)
    print(f"depth {depth}: dumas lower on frob_rel {frob_wins}/{n}, on max_rel {max_wins}/{n}")
    print(f"  max_rel mean: strassen={sum(maxs_s)/n:.2f} dumas={sum(maxs_d)/n:.2f}  "
          f"median: strassen={maxs_s[n//2]:.2f} dumas={maxs_d[n//2]:.2f}")
