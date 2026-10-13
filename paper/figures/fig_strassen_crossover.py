"""
Figure: optimized classical DGEMM vs. optimized 1-level Strassen,
effective GFLOP/s vs N -- backs paper/main.tex Section 7.4
(Table strassen-crossover). Numbers copied verbatim from that table.
Run: python3 fig_strassen_crossover.py
Output: fig_strassen_crossover.pdf
"""
import matplotlib.pyplot as plt

N = [512, 1024, 2048, 4096, 8192]
dgemm = [245.8, 466.9, 458.2, 434.7, 426.6]
strassen = [22.9, 105.6, 260.0, 375.0, 426.7]

fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.plot(N, dgemm, marker="o", color="#4477AA", label="Classical DGEMM (baseline)", lw=2)
ax.plot(N, strassen, marker="s", color="#EE6677", label="Strassen (1-level)", lw=2)

ax.annotate("crossover\n$N{=}8192$", xy=(8192, 426.6), xytext=(4600, 330),
            arrowprops=dict(arrowstyle="->", color="#333333"), fontsize=9,
            ha="center")

ax.set_xscale("log", base=2)
ax.set_xticks(N)
ax.set_xticklabels([str(n) for n in N])
ax.set_xlabel("$N$")
ax.set_ylabel("Effective GFLOP/s  ($=2N^3/\\mathrm{time}$)")
ax.set_ylim(0, 500)
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#dddddd", lw=0.7, zorder=0)
fig.tight_layout()
fig.savefig("fig_strassen_crossover.pdf")
print("wrote fig_strassen_crossover.pdf")
