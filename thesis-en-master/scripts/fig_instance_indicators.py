"""Generates img/instance-indicators.png for chap03.tex: a two-panel
illustration of the Order Strength (OS) and Resource Strength (RS) instance
indicators described in Section 3.1.4 ("Instance characterisation
indicators"), contrasting a loosely- and a tightly-constrained instance for
each indicator.
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

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

# ---------------------------------------------------------------------------
# Left panel: Order Strength (OS) -- two small precedence graphs
# ---------------------------------------------------------------------------
ax = axes[0]
ax.set_title("Order strength (OS)", fontsize=13, fontweight="bold", pad=14)

def draw_dag(ax, x0, nodes, edges, label, os_value, color, fill):
    pos = {n: (x0 + p[0], p[1]) for n, p in nodes.items()}
    for a, b in edges:
        xa, ya = pos[a]
        xb, yb = pos[b]
        ax.annotate(
            "", xy=(xb, yb), xytext=(xa, ya),
            arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.3,
                             shrinkA=10, shrinkB=10),
        )
    for n, (x, y) in pos.items():
        ax.add_patch(mpatches.Circle((x, y), 0.22, facecolor=fill,
                                      edgecolor=color, linewidth=1.6, zorder=3))
        ax.text(x, y, n, ha="center", va="center", fontsize=9,
                fontweight="bold", color=color, zorder=4)
    cx = x0 + 0.9
    ax.text(cx, -0.85, label, ha="center", va="center", fontsize=10.5)
    ax.text(cx, -1.25, f"OS $\\approx$ {os_value}", ha="center", va="center",
             fontsize=10, color=GRAY, style="italic")

# Low OS: sparse precedence (many parallel chains)
low_nodes = {"1": (0, 1), "2": (0.9, 1.6), "3": (0.9, 0.4), "4": (1.8, 1)}
low_edges = [("1", "2"), ("1", "3")]
draw_dag(ax, 0.0, low_nodes, low_edges, "Low OS: mostly parallel", "0.2",
         BLUE, BLUE_FILL)

# High OS: dense precedence (mostly serial chain)
high_nodes = {"1": (0, 1), "2": (0.9, 1.3), "3": (0.9, 0.7), "4": (1.8, 1)}
high_edges = [("1", "2"), ("1", "3"), ("2", "3"), ("2", "4"), ("3", "4"),
              ("1", "4")]
draw_dag(ax, 3.4, high_nodes, high_edges, "High OS: mostly serial", "0.8",
         RED, RED_FILL)

ax.set_xlim(-0.6, 5.8)
ax.set_ylim(-1.6, 2.1)
ax.set_aspect("equal")
ax.axis("off")

# ---------------------------------------------------------------------------
# Right panel: Resource Strength (RS) -- demand profile vs. availability
# ---------------------------------------------------------------------------
ax = axes[1]
ax.set_title("Resource strength (RS)", fontsize=13, fontweight="bold", pad=14)

t = np.linspace(0, 10, 200)
demand = 3 + 1.6 * np.sin(t) + 0.6 * np.sin(2.3 * t + 1)
demand = np.clip(demand, 0.5, None)
k_min, k_max = demand.max() * 0.55, demand.max()

ax.fill_between(t, 0, demand, color=BLUE_FILL, alpha=0.7, zorder=2)
ax.plot(t, demand, color=BLUE, linewidth=1.8, zorder=3, label="resource demand")

a_tight = k_min + 0.12 * (k_max - k_min)   # RS close to 0
a_loose = k_min + 0.85 * (k_max - k_min)   # RS close to 1

ax.axhline(a_tight, color=RED, linewidth=1.8, linestyle="--", zorder=4)
ax.axhline(a_loose, color=GOLD, linewidth=1.8, linestyle="--", zorder=4)
ax.text(10.2, a_tight, "RS $\\approx$ 0\n(tight)", color=RED, fontsize=9.5,
        va="center", ha="left")
ax.text(10.2, a_loose, "RS $\\approx$ 1\n(loose)", color=GOLD, fontsize=9.5,
        va="center", ha="left")

ax.set_xlim(0, 10)
ax.set_ylim(0, k_max * 1.15)
ax.set_xlabel("Time", fontsize=11)
ax.set_ylabel("Resource units", fontsize=11)
ax.set_xticks([])
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("")
fig.tight_layout()
fig.savefig("img/instance-indicators.png", dpi=200, bbox_inches="tight")
print("done")
