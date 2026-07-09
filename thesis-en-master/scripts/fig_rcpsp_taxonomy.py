"""Generates img/rcpsp-taxonomy.png for chap02.tex: a taxonomy of RCPSP
extensions, highlighting the multi-mode branch (MRCPSP) pursued by this thesis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "serif"

RED = "#b03a2e"
RED_FILL = "#f5d6d1"
BLUE = "#34495e"
BLUE_FILL = "#dde3ea"
GRAY_FILL = "#eef0f2"
GRAY = "#7a7a7a"


def box(ax, x, y, w, h, text, edge, face, fontsize=11, fontweight="bold"):
    b = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=3)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=edge, zorder=4,
            linespacing=1.3)
    return (x + w / 2, y), (x + w / 2, y + h)


def connect(ax, p_bottom, p_top, color="gray"):
    (x0, y0) = p_bottom
    (x1, y1) = p_top
    ax.plot([x0, x0, x1, x1], [y0, (y0 + y1) / 2, (y0 + y1) / 2, y1],
            color=color, linewidth=1.1, zorder=1)


fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6.4)
ax.axis("off")

# root
root_bottom, root_top = box(ax, 4.4, 5.3, 2.2, 0.85, "RCPSP", BLUE, BLUE_FILL, 13)

# second level
children = [
    ("Multi-mode\n(MRCPSP)", 0.1, True),
    ("Preemptive", 2.35, False),
    ("Multi-skill", 4.6, False),
    ("Stochastic /\nrobust", 6.85, False),
    ("Multi-objective", 9.1, False),
]
mid_tops = []
for label, x, highlight in children:
    edge, face = (RED, RED_FILL) if highlight else (BLUE, BLUE_FILL)
    b_bot, b_top = box(ax, x, 3.55, 1.85, 0.95, label, edge, face, 10.5)
    connect(ax, root_bottom, b_top)
    mid_tops.append((label, x, b_bot, highlight))

# leaves under "Multi-mode" — the branch this thesis targets
mm_bot = (0.1 + 1.85 / 2, 3.55)
leaves = [
    "Renewable +\nnon-renewable\nresources",
    "Generalized\nprecedence\n(MRCPSP/max)",
    "Dynamic /\ndisrupted",
]
leaf_xs = [0.1, 2.1, 4.1]
for label, lx in zip(leaves, leaf_xs):
    l_bot, l_top = box(ax, lx, 1.65, 1.85, 1.05, label, RED, "white", 9.5,
                        fontweight="normal")
    connect(ax, mm_bot, l_top, color=RED)

ax.text(5.5, 0.55,
        "Highlighted boxes mark the branch addressed by this thesis: the\n"
        "multi-mode RCPSP (MRCPSP) with renewable and non-renewable resources.",
        ha="center", va="center", fontsize=10, color=GRAY, style="italic")

fig.tight_layout()
fig.savefig("img/rcpsp-taxonomy.png", dpi=200, bbox_inches="tight")
print("done")
