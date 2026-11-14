"""
Figure: optimized classical DGEMM vs. optimized 1-level Strassen,
effective GFLOP/s vs N, mean + bootstrap 95% CI band -- backs
paper/main.tex Table strassen-crossover. n=30 independent launches per
point; numbers copied verbatim from that table (and from
src/strassen2/results_n30_crossover_20261020_152612/).
Run: python3 fig_strassen_crossover.py
Output: fig_strassen_crossover.pdf
"""
import matplotlib.pyplot as plt

N = [512, 1024, 2048, 4096, 8192]
dgemm = [272.8, 501.7, 486.1, 391.9, 402.9]
dgemm_lo = [252.5, 454.1, 465.9, 360.2, 386.7]
dgemm_hi = [287.5, 543.0, 500.5, 420.2, 416.9]
strassen = [14.5, 73.0, 239.1, 379.0, 472.0]
strassen_lo = [13.8, 69.6, 234.0, 369.3, 469.7]
strassen_hi = [15.0, 76.2, 243.8, 388.5, 474.2]

fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.plot(N, dgemm, marker="o", color="#4477AA", label="Classical DGEMM (baseline)", lw=2)
ax.fill_between(N, dgemm_lo, dgemm_hi, color="#4477AA", alpha=0.2, zorder=1)
ax.plot(N, strassen, marker="s", color="#EE6677", label="Strassen (1-level)", lw=2)
ax.fill_between(N, strassen_lo, strassen_hi, color="#EE6677", alpha=0.2, zorder=1)

ax.annotate("CIs overlap\n(not distinguishable)", xy=(4096, 385), xytext=(1800, 150),
            arrowprops=dict(arrowstyle="->", color="#333333"), fontsize=8.5, ha="center")
ax.annotate("Strassen\nsignificantly ahead", xy=(8192, 472), xytext=(4600, 560),
            arrowprops=dict(arrowstyle="->", color="#333333"), fontsize=8.5, ha="center")

ax.set_xscale("log", base=2)
ax.set_xticks(N)
ax.set_xticklabels([str(n) for n in N])
ax.set_xlabel("$N$")
ax.set_ylabel("Effective GFLOP/s  ($=2N^3/\\mathrm{time}$)")
ax.set_ylim(0, 620)
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#dddddd", lw=0.7, zorder=0)
fig.tight_layout()
fig.savefig("fig_strassen_crossover.pdf")
print("wrote fig_strassen_crossover.pdf")
