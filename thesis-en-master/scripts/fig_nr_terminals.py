"""Generates img/nr-terminals.png for chap04.tex: illustrates the two
nonrenewable-resource-aware GP terminals introduced in Section 4.2 --
NR_STOCK_RATIO (global scarcity signal, left panel) and
NR_MODE_DEMAND_RATIO (per-mode demand signal, right panel).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = "serif"

RED = "#b03a2e"
RED_FILL = "#f5d6d1"
BLUE = "#34495e"
BLUE_FILL = "#dde3ea"
GOLD = "#c8960c"
GOLD_FILL = "#f5e7c4"
GRAY = "#7a7a7a"

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))

# ---------------------------------------------------------------------------
# Left panel: NR_STOCK_RATIO over the course of decoding one schedule
# ---------------------------------------------------------------------------
ax = axes[0]
ax.set_title("NR_STOCK_RATIO (activity tree)", fontsize=12,
             fontweight="bold", pad=12)

steps = np.arange(0, 11)
ratio = np.array([1.00, 0.93, 0.88, 0.80, 0.71, 0.71, 0.60, 0.47, 0.40, 0.30, 0.22])

ax.step(steps, ratio, where="post", color=BLUE, linewidth=2.0, zorder=3)
ax.fill_between(steps, 0, ratio, step="post", color=BLUE_FILL, alpha=0.7,
                 zorder=2)

ax.axhline(1.0, color=GRAY, linewidth=0.8, linestyle=":")
ax.annotate("ample budget\n(value close to 1)", xy=(1, 0.93),
            xytext=(2.2, 1.18), fontsize=9, color=GOLD,
            arrowprops=dict(arrowstyle="-", color=GOLD, lw=1.0))
ax.annotate("budget nearly\nexhausted (value\nclose to 0)", xy=(9.3, 0.27),
            xytext=(6.3, 0.10), fontsize=9, color=RED,
            arrowprops=dict(arrowstyle="-", color=RED, lw=1.0))

ax.set_xlim(0, 10.3)
ax.set_ylim(0, 1.32)
ax.set_xlabel("SGS decision step (activities scheduled so far)", fontsize=10)
ax.set_ylabel("Remaining nonrenewable\nbudget (fraction)", fontsize=10)
ax.set_xticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ---------------------------------------------------------------------------
# Right panel: NR_MODE_DEMAND_RATIO for three candidate modes
# ---------------------------------------------------------------------------
ax = axes[1]
ax.set_title("NR_MODE_DEMAND_RATIO (mode tree)", fontsize=12,
             fontweight="bold", pad=12)

modes = ["Mode 1\n(fast, heavy)", "Mode 2\n(balanced)", "Mode 3\n(slow, light)"]
remaining_stock = 10.0
consumption = [9.5, 4.0, 1.5]
ratios = [c / remaining_stock for c in consumption]
colors = [RED, GOLD, BLUE]
fills = [RED_FILL, GOLD_FILL, BLUE_FILL]

xs = np.arange(3)
bars = ax.bar(xs, ratios, width=0.55, color=fills, edgecolor=colors,
              linewidth=1.8, zorder=3)
for x, r, c in zip(xs, ratios, colors):
    ax.text(x, r + 0.04, f"{r:.2f}", ha="center", va="bottom", fontsize=10,
            fontweight="bold", color=c)

ax.axhline(1.0, color=RED, linewidth=1.4, linestyle="--", zorder=2)
ax.text(2.62, 1.0, "ratio $>1$:\ninfeasible", fontsize=8.6, color=RED,
        va="center", ha="left")

ax.set_xticks(xs)
ax.set_xticklabels(modes, fontsize=9.3)
ax.set_ylabel("Demand / remaining stock", fontsize=10)
ax.set_ylim(0, 1.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("img/nr-terminals.png", dpi=200, bbox_inches="tight")
print("done")
