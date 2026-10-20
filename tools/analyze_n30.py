#!/usr/bin/env python3
"""analyze_n30.py -- statistics for an n=30 run folder (stdlib only, no numpy/scipy).

Input : <run_dir>/n30_samples.csv  (written by avx512/run_avx512.sh; one row per
        independent process launch, per ISA, per (N,T))
Output: <run_dir>/n30_summary.csv     mean, sd, CV%, median, bootstrap 95% CI,
                                      min/max, trimmed mean, worst rel. error
        <run_dir>/n30_comparison.csv  AVX-512 vs AVX2 per (algorithm,N,T): mean ratio,
                                      bootstrap 95% CI of the ratio, permutation-test p,
                                      Holm-adjusted p across ALL comparisons in the run
        <run_dir>/avx512.csv, avx2.csv  mean seconds per (N,T), same schema as older runs
        <run_dir>/n30_headline.env    key=value summary read by run_avx512.sh

Sample unit = GFLOP/s of one process launch (best of the inner reps in that launch),
matching revalidation/n30_rerun_2026-10-12/. Resampling is seeded -> reproducible.
Usage: python3 tools/analyze_n30.py <run_dir> [--boot 10000] [--seed 12345]
"""
import argparse, csv, math, os, random, statistics, sys
from collections import defaultdict

ALGOS = ("blocked", "strassen")


def pct(sorted_x, q):
    i = (len(sorted_x) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return sorted_x[lo] + (sorted_x[hi] - sorted_x[lo]) * (i - lo)


def boot_means(x, B, rng):
    n = len(x)
    return sorted(sum(rng.choices(x, k=n)) / n for _ in range(B))


def trimmed(x, k=2):
    s = sorted(x)
    return statistics.mean(s[k:-k]) if len(s) > 2 * k else statistics.mean(s)


def perm_p(a, b, B, rng):
    obs = abs(statistics.mean(a) - statistics.mean(b))
    pool, na, hits = a + b, len(a), 0
    for _ in range(B):
        rng.shuffle(pool)
        if abs(statistics.mean(pool[:na]) - statistics.mean(pool[na:])) >= obs:
            hits += 1
    return (hits + 1) / (B + 1)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    path = os.path.join(a.run_dir, "n30_samples.csv")
    g = defaultdict(lambda: {"gf": [], "sec": [], "err": []})
    for r in csv.DictReader(open(path, encoding="utf-8")):
        for al in ALGOS:
            k = (r["isa"], al, int(r["N"]), int(r["T"]))
            g[k]["gf"].append(float(r[f"{al}_gflops"]))
            g[k]["sec"].append(float(r[f"{al}_best_s"]))
            g[k]["err"].append(float(r[f"{al}_relerr"]))

    summary = []
    for (isa, al, N, T), d in sorted(g.items()):
        x = d["gf"]
        n = len(x)
        bm = boot_means(x, a.boot, rng)
        mean = statistics.mean(x)
        sd = statistics.stdev(x) if n > 1 else 0.0
        summary.append({"isa": isa, "algorithm": al, "N": N, "T": T, "n": n,
                        "mean_gflops": round(mean, 3), "sd": round(sd, 3),
                        "cv_pct": round(100 * sd / mean, 2) if mean else 0,
                        "median": round(statistics.median(x), 3),
                        "ci95_lo": round(pct(bm, 0.025), 3), "ci95_hi": round(pct(bm, 0.975), 3),
                        "min": round(min(x), 3), "max": round(max(x), 3),
                        "trimmed_mean_drop2": round(trimmed(x), 3),
                        "max_relerr": max(d["err"]), "mean_seconds": statistics.mean(d["sec"])})

    comp = []
    for (isa, al, N, T) in sorted(g):
        if isa != "avx512" or ("avx2", al, N, T) not in g:
            continue
        x, y = g[(isa, al, N, T)]["gf"], g[("avx2", al, N, T)]["gf"]
        ratios = sorted(
            (sum(rng.choices(x, k=len(x))) / len(x)) / (sum(rng.choices(y, k=len(y))) / len(y))
            for _ in range(a.boot))
        comp.append({"algorithm": al, "N": N, "T": T, "n_avx512": len(x), "n_avx2": len(y),
                     "mean_avx512": round(statistics.mean(x), 3), "mean_avx2": round(statistics.mean(y), 3),
                     "ratio": round(statistics.mean(x) / statistics.mean(y), 3),
                     "ratio_ci_lo": round(pct(ratios, 0.025), 3), "ratio_ci_hi": round(pct(ratios, 0.975), 3),
                     "perm_p": perm_p(list(x), list(y), a.boot, rng)})
    for c, p in zip(comp, holm([c["perm_p"] for c in comp])):
        c["holm_p"] = round(p, 5)
        c["perm_p"] = round(c["perm_p"], 5)
        c["significant_0.05"] = "yes" if p < 0.05 else "no"

    def write(name, rows, cols=None):
        cols = cols or list(rows[0].keys())
        with open(os.path.join(a.run_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    scols = ["isa", "algorithm", "N", "T", "n", "mean_gflops", "sd", "cv_pct", "median", "ci95_lo",
             "ci95_hi", "min", "max", "trimmed_mean_drop2", "max_relerr"]
    write("n30_summary.csv", summary, scols)
    if comp:
        write("n30_comparison.csv", comp)

    # compat CSVs (mean seconds), same schema as older avx2.csv / avx512.csv
    for isa in ("avx512", "avx2"):
        pts = sorted({(N, T) for (i, _, N, T) in g if i == isa})
        with open(os.path.join(a.run_dir, f"{isa}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["N", "T", f"blocked_{isa}", f"strassen_{isa}"])
            for N, T in pts:
                w.writerow([N, T, f"{statistics.mean(g[(isa, 'blocked', N, T)]['sec']):.6f}",
                            f"{statistics.mean(g[(isa, 'strassen', N, T)]['sec']):.6f}"])

    # headline = largest N, then largest T
    HN = max(N for (_, _, N, _) in g)
    HT = max(T for (_, _, N, T) in g if N == HN)
    S = {(s["isa"], s["algorithm"]): s for s in summary if s["N"] == HN and s["T"] == HT}
    C = {c["algorithm"]: c for c in comp if c["N"] == HN and c["T"] == HT}
    bad = sum(1 for s in summary if s["max_relerr"] > 1e-8)
    noisy = sum(1 for s in summary if s["cv_pct"] > 5.0)
    env = {"HN": HN, "HT": HT, "RUNS": min(s["n"] for s in summary), "BAD": bad, "NOISY": noisy,
           "B512": S[("avx512", "blocked")]["mean_gflops"], "B512_LO": S[("avx512", "blocked")]["ci95_lo"],
           "B512_HI": S[("avx512", "blocked")]["ci95_hi"], "S512": S[("avx512", "strassen")]["mean_gflops"],
           "B2": S[("avx2", "blocked")]["mean_gflops"], "S2": S[("avx2", "strassen")]["mean_gflops"]}
    if "blocked" in C:
        env.update(RATIO=C["blocked"]["ratio"], RATIO_LO=C["blocked"]["ratio_ci_lo"],
                   RATIO_HI=C["blocked"]["ratio_ci_hi"], PHOLM=C["blocked"]["holm_p"])
    with open(os.path.join(a.run_dir, "n30_headline.env"), "w") as f:
        for k, v in env.items():
            f.write(f"{k}={v}\n")

    print(f"{'isa':7}{'algo':9}{'N':>6}{'T':>4}{'n':>4}{'mean':>9}{'95% CI':>20}{'CV%':>7}")
    for s in summary:
        print(f"{s['isa']:7}{s['algorithm']:9}{s['N']:>6}{s['T']:>4}{s['n']:>4}{s['mean_gflops']:>9.1f}"
              f"{'[%.1f, %.1f]' % (s['ci95_lo'], s['ci95_hi']):>20}{s['cv_pct']:>7.1f}")
    if comp:
        print("\nAVX-512 / AVX2 (mean GFLOP/s ratio, bootstrap 95% CI, Holm-adjusted permutation p):")
        for c in comp:
            print(f"  {c['algorithm']:9} N={c['N']:<5} T={c['T']:<3} {c['ratio']:.2f}x "
                  f"[{c['ratio_ci_lo']:.2f}, {c['ratio_ci_hi']:.2f}]  p_holm={c['holm_p']}  {c['significant_0.05']}")
    if bad:
        print(f"\n!! {bad} (isa,algo,N,T) groups had relative error > 1e-8 -- results INVALID")
    if noisy:
        print(f"\nWARNING: {noisy} groups have CV > 5% (noisy: throttling/background load?) -- "
              "consider re-running plugged in with other apps closed.")


if __name__ == "__main__":
    sys.exit(main())
