"""Generates img/ea-loop.png for chap02.tex: the general evolutionary loop
(Section 2.4.1) -- population, selection, variation (crossover and mutation)
and replacement -- shared by every evolutionary algorithm discussed in this
thesis, regardless of whether the individual is a GA chromosome or a GP tree.
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


def node(ax, x, y, text, edge, face, w=2.6, h=1.0, fontsize=10.5):
    b = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.7, edgecolor=edge, facecolor=face, zorder=3)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=edge, zorder=4, linespacing=1.3)
    return x, y


fig, ax = plt.subplots(figsize=(7.4, 7.4))
ax.set_xlim(-4.4, 4.4)
ax.set_ylim(-4.4, 4.4)
ax.set_aspect("equal")
ax.axis("off")

R = 3.0
positions = {
    "population":  (0, R),
    "selection":   (R * np.sin(np.radians(72)), R * np.cos(np.radians(72))),
    "crossover":   (R * np.sin(np.radians(144)), R * np.cos(np.radians(144))),
    "mutation":    (R * np.sin(np.radians(216)), R * np.cos(np.radians(216))),
    "replacement": (R * np.sin(np.radians(288)), R * np.cos(np.radians(288))),
}
labels = {
    "population":  "Population\nof individuals",
    "selection":   "Selection\n(favour fitter\nindividuals as parents)",
    "crossover":   "Crossover\n(recombine\ntwo parents)",
    "mutation":    "Mutation\n(perturb an\noffspring)",
    "replacement": "Replacement\n(form the next\ngeneration)",
}
colors = {
    "population":  (BLUE, BLUE_FILL),
    "selection":   (GOLD, GOLD_FILL),
    "crossover":   (GOLD, GOLD_FILL),
    "mutation":    (GOLD, GOLD_FILL),
    "replacement": (RED, RED_FILL),
}

pts = {}
for key, (x, y) in positions.items():
    edge, face = colors[key]
    pts[key] = node(ax, x, y, labels[key], edge, face, w=2.5, h=1.35, fontsize=9.8)

order = ["population", "selection", "crossover", "mutation", "replacement"]
for i in range(len(order)):
    a, b = order[i], order[(i + 1) % len(order)]
    pa, pb = np.array(pts[a]), np.array(pts[b])
    direction = pb - pa
    direction = direction / np.linalg.norm(direction)
    start = pa + direction * 1.15
    end = pb - direction * 1.15
    ax.annotate("", xy=tuple(end), xytext=tuple(start),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.8,
                                 shrinkA=0, shrinkB=0), zorder=2)

ax.text(0, 0, "Evolutionary\nloop", ha="center", va="center", fontsize=13,
        fontweight="bold", color=GRAY, style="italic")

ax.text(0, -4.0,
        "Representation-specific: a GA varies a fixed-length chromosome; "
        "a GP varies a variable-size tree (Sections 2.4.2-2.4.3).",
        ha="center", va="center", fontsize=9.0, color=GRAY, style="italic")

fig.tight_layout()
fig.savefig("img/ea-loop.png", dpi=200, bbox_inches="tight")
print("done")
