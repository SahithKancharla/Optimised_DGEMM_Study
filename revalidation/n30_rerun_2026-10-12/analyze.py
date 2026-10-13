"""
Bootstrap 95% CIs + Games-Howell pairwise test for the n=30 re-run
(see README.md). Games-Howell implemented directly (no pingouin):
Welch t -> studentized range q=t*sqrt(2) -> p via
scipy.stats.studentized_range.sf, Welch-Satterthwaite df per pair.
"""
import csv
import numpy as np
from scipy.stats import studentized_range

np.random.seed(0)

files = {
    'unpinned_static':  'cond1_unpinned_static.csv',
    'unpinned_dynamic': 'cond2_unpinned_dynamic.csv',
    'mixed_PE_static':  'cond3_mixed_PE_static.csv',
    'mixed_PE_dynamic': 'cond4_mixed_PE_dynamic.csv',
}

data = {}
for label, fn in files.items():
    with open(fn, encoding='utf-8-sig') as f:
        r = csv.DictReader(f)
        vals = np.array([float(row['gflops']) for row in r])
    data[label] = vals
    print(f"{label}: n={len(vals)} mean={vals.mean():.2f} sd={vals.std(ddof=1):.2f} "
          f"min={vals.min():.2f} max={vals.max():.2f}")

print()
print("=== Bootstrap 95% CI for the mean (10,000 resamples) ===")
B = 10000
for label, vals in data.items():
    n = len(vals)
    boots = np.array([np.mean(np.random.choice(vals, size=n, replace=True)) for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{label}: mean={vals.mean():.2f}  95% CI [{lo:.2f}, {hi:.2f}]")

print()
print("=== Games-Howell pairwise test (k=4 groups, Welch df, studentized range) ===")
labels = list(data.keys())
k = len(labels)
for i in range(k):
    for j in range(i + 1, k):
        a, b = data[labels[i]], data[labels[j]]
        na, nb = len(a), len(b)
        va, vb = a.var(ddof=1), b.var(ddof=1)
        se = np.sqrt(va / na + vb / nb)
        t = abs(a.mean() - b.mean()) / se
        q = t * np.sqrt(2)
        df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        p = studentized_range.sf(q, k, df)
        sig = "***" if p < 0.001 else ("*" if p < 0.05 else "ns")
        print(f"{labels[i]:20s} vs {labels[j]:20s}: diff={a.mean() - b.mean():8.2f}  "
              f"q={q:7.3f}  df={df:6.1f}  p={p:.2e}  {sig}")
