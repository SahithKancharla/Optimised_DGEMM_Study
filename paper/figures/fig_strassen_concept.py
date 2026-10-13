"""
Figure: conceptual diagram of classical block multiplication (8
quadrant products) vs. Strassen's recursion (7 products). Purely
explanatory -- no benchmark data -- to give a reader new to the topic
a mental picture before Section 2's math. Run: python3 fig_strassen_concept.py
Output: fig_strassen_concept.pdf
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))


def box(ax, xy, w, h, text, color="#4477AA", fontsize=9, fc_alpha=0.15):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                        linewidth=1.3, edgecolor=color,
                        facecolor=color, alpha=fc_alpha, zorder=2)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=3)


def arrow(ax, p1, p2, color="#555555"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=10,
                                  color=color, lw=1.1, zorder=2))


# --- Left panel: classical quadrant multiplication ---
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Classical: 8 quadrant products", fontsize=10)
box(ax, (0.3, 7.5), 2.6, 1.6, "$A$\nsplit into\n$A_{11..22}$", "#4477AA")
box(ax, (7.1, 7.5), 2.6, 1.6, "$B$\nsplit into\n$B_{11..22}$", "#4477AA")
box(ax, (2.2, 3.8), 5.6, 1.8, "8 products:\n$A_{11}B_{11},\\,A_{11}B_{12},\\,\\dots,\\,A_{22}B_{22}$",
    "#EE6677", fontsize=8.5)
box(ax, (3.2, 0.4), 3.6, 1.8, "$C$: sum pairs\nof products per\nquadrant", "#228833")
arrow(ax, (1.6, 7.5), (3.8, 5.6))
arrow(ax, (8.4, 7.5), (6.2, 5.6))
arrow(ax, (5.0, 3.8), (5.0, 2.2))

# --- Right panel: Strassen's 7 products ---
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Strassen: 7 products ($M_1 \\ldots M_7$)", fontsize=10)
box(ax, (0.3, 7.5), 2.6, 1.6, "$A$\nsplit into\n$A_{11..22}$", "#4477AA")
box(ax, (7.1, 7.5), 2.6, 1.6, "$B$\nsplit into\n$B_{11..22}$", "#4477AA")
box(ax, (1.6, 3.8), 6.8, 1.8,
    "7 recursive products, each a\nfixed $+/-$ combination of quadrants:\n$M_1=(A_{11}{+}A_{22})(B_{11}{+}B_{22}),\\;\\ldots,\\;M_7$",
    "#EE6677", fontsize=8)
box(ax, (2.6, 0.4), 4.8, 1.8,
    "$C$: combine $M_1..M_7$\nvia $+/-$ only\n(one fewer multiply)", "#228833")
arrow(ax, (1.6, 7.5), (3.8, 5.6))
arrow(ax, (8.4, 7.5), (6.2, 5.6))
arrow(ax, (5.0, 3.8), (5.0, 2.2))

fig.tight_layout()
fig.savefig("fig_strassen_concept.pdf")
print("wrote fig_strassen_concept.pdf")
