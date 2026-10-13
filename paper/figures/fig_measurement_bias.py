"""
Backs paper/main.tex Section 5. The four scattered groups are the
n=30 dataset from revalidation/n30_rerun_2026-10-12/; the P/E-only
range bars are the original n=5 data (not re-measured at n=30).
"""
import csv
import matplotlib.pyplot as plt
import numpy as np

def _load(fn):
    with open(fn, encoding='utf-8-sig') as f:
        return [float(row['gflops']) for row in csv.DictReader(f)]

_base = "../../revalidation/n30_rerun_2026-10-12/"
unpinned_static = _load(_base + "cond1_unpinned_static.csv")
unpinned_dynamic = _load(_base + "cond2_unpinned_dynamic.csv")
mixed_static = _load(_base + "cond3_mixed_PE_static.csv")
mixed_dynamic = _load(_base + "cond4_mixed_PE_dynamic.csv")

# Ranges only (Section 5.3 reports a range, not individual repeats here).
p_cores_range = (250, 280)
e_cores_range = (133, 137)

fig, ax = plt.subplots(figsize=(6.6, 4.2))

rng = np.random.default_rng(0)
groups = [
    ("Unpinned\nstatic", unpinned_static, "#4477AA"),
    ("Unpinned\ndynamic", unpinned_dynamic, "#EE6677"),
    ("Pinned 4P+4E\nstatic", mixed_static, "#4477AA"),
    ("Pinned 4P+4E\ndynamic", mixed_dynamic, "#EE6677"),
]

positions = list(range(len(groups)))
for pos, (label, values, color) in zip(positions, groups):
    jitter = rng.uniform(-0.15, 0.15, size=len(values))
    ax.scatter([pos + j for j in jitter], values, color=color, s=22,
               zorder=3, edgecolor="white", linewidth=0.4, alpha=0.85)
    lo, hi = min(values), max(values)
    ax.plot([pos - 0.22, pos + 0.22], [lo, lo], color=color, lw=1.2, alpha=0.5)
    ax.plot([pos - 0.22, pos + 0.22], [hi, hi], color=color, lw=1.2, alpha=0.5)
    ax.fill_between([pos - 0.22, pos + 0.22], lo, hi, color=color, alpha=0.08)

# Core-placement range bars (policy-independent; range only, not individual repeats).
for pos_label, (lo, hi) in [(4, p_cores_range), (5, e_cores_range)]:
    ax.add_patch(plt.Rectangle((pos_label - 0.28, lo), 0.56, hi - lo,
                                facecolor="#228833", alpha=0.55, zorder=2))
    ax.text(pos_label, hi + 6, f"{lo}-{hi}", ha="center", fontsize=8.5, color="#1b6b2c")

positions += [4, 5]
labels = [g[0] for g in groups] + ["Pinned\nP-cores only", "Pinned\nE-cores only"]
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("GFLOP/s  (N=1024, T=8)")
ax.set_ylim(90, 335)
ax.axvline(3.5, color="gray", lw=0.8, linestyle=":")
ax.text(1.5, 322, "scheduling policy (n=30 each): overlapping, p=0.25/p=0.999 (ns)",
        ha="center", fontsize=8.5, style="italic", color="#333333")
ax.text(4.5, 322, "core placement:\nclean separation",
        ha="center", fontsize=9, style="italic", color="#1b6b2c")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig_measurement_bias.pdf")
print("wrote fig_measurement_bias.pdf")
