"""Generates img/gphh-architecture.png for chap04.tex: the multi-tree GPHH
pipeline used as the baseline algorithm -- from a GP individual (activity
tree + mode tree), through serial-SGS decoding, to the fitness value that
drives the evolutionary loop.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "serif"

RED = "#b03a2e"
RED_FILL = "#f5d6d1"
BLUE = "#34495e"
BLUE_FILL = "#dde3ea"
GOLD = "#c8960c"
GOLD_FILL = "#f5e7c4"
GRAY = "#7a7a7a"


def box(ax, x, y, w, h, text, edge, face, fontsize=10.5, fontweight="bold"):
    b = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=3)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=edge, zorder=4,
            linespacing=1.35)
    return (x + w / 2, y + h), (x + w / 2, y), (x, y + h / 2), (x + w, y + h / 2)


def varrow(ax, p_from, p_to, color=GRAY, **kw):
    ax.annotate("", xy=p_to, xytext=p_from,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                shrinkA=2, shrinkB=2), zorder=2, **kw)


fig, ax = plt.subplots(figsize=(7.6, 9.6))

# 1. GP individual
top1, bot1, l1, r1 = box(ax, 1.2, 11.2, 4.6, 1.0,
                          "GP individual\n(activity tree, mode tree)",
                          BLUE, BLUE_FILL)

# 2. Serial SGS decoding (composite box with sub-steps)
sgs_x, sgs_y, sgs_w, sgs_h = 0.5, 7.6, 6.0, 2.9
ax.add_patch(mpatches.FancyBboxPatch(
    (sgs_x, sgs_y), sgs_w, sgs_h, boxstyle="round,pad=0.03,rounding_size=0.08",
    linewidth=1.8, edgecolor=GOLD, facecolor="white", zorder=2))
ax.text(sgs_x + sgs_w / 2, sgs_y + sgs_h - 0.32, "Serial SGS decoding",
        ha="center", va="center", fontsize=11, fontweight="bold", color=GOLD,
        zorder=3)

steps = [
    "for the eligible-activity set:\nevaluate the activity tree,\nselect the minimum-score activity",
    "for that activity's modes:\nevaluate the mode tree,\nselect the minimum-score mode",
    "insert the (activity, mode) pair\nat the earliest resource-feasible time",
]
sy = sgs_y + sgs_h - 0.85
step_boxes = []
for s in steps:
    t, b, l, r = box(ax, sgs_x + 0.35, sy - 0.62, sgs_w - 0.7, 0.62, s,
                      GOLD, GOLD_FILL, fontsize=8.7, fontweight="normal")
    step_boxes.append((t, b, l, r))
    sy -= 0.78
for i in range(len(step_boxes) - 1):
    varrow(ax, step_boxes[i][1], step_boxes[i + 1][0], color=GOLD)
ax.annotate(
    "repeat until every\nactivity is scheduled", xy=(sgs_x + sgs_w - 0.1, sgs_y + 1.1),
    xytext=(sgs_x + sgs_w + 0.55, sgs_y + 1.1), fontsize=8.3, color=GRAY,
    va="center", ha="left", style="italic",
    arrowprops=dict(arrowstyle="-", color=GRAY, lw=1.0,
                     connectionstyle="arc3,rad=-0.6"))

varrow(ax, bot1, (sgs_x + sgs_w / 2, sgs_y + sgs_h), color=GRAY)

# 3. Schedule / makespan
top3, bot3, l3, r3 = box(ax, 1.2, 6.2, 4.6, 1.0,
                          "Complete schedule\nmakespan $C_{\\max}(i)$",
                          BLUE, BLUE_FILL)
varrow(ax, (sgs_x + sgs_w / 2, sgs_y), top3, color=GRAY)

# 4. Deviation
top4, bot4, l4, r4 = box(ax, 1.0, 4.7, 5.0, 1.05,
                          "$\\mathrm{dev}(i) = \\dfrac{C_{\\max}(i) - "
                          "\\mathrm{CPM}_{\\mathrm{EF}}(i)}"
                          "{\\mathrm{CPM}_{\\mathrm{EF}}(i)} \\times 100\\%$",
                          RED, RED_FILL, fontsize=10)
varrow(ax, bot3, top4, color=GRAY)

# 5. Fitness (mean over training set)
top5, bot5, l5, r5 = box(ax, 1.2, 3.2, 4.6, 1.0,
                          "Fitness: mean $\\mathrm{dev}(i)$\nover the training set",
                          RED, RED_FILL)
varrow(ax, bot4, top5, color=GRAY)
ax.text(6.0, 4.05, "repeated for every\ntraining instance $i$", fontsize=8.3,
        color=GRAY, style="italic", ha="left", va="center")

# 6. Evolutionary loop
top6, bot6, l6, r6 = box(ax, 1.2, 1.5, 4.6, 1.1,
                          "Selection, crossover, mutation\n(next generation)",
                          BLUE, BLUE_FILL)
varrow(ax, bot5, top6, color=GRAY)

# feedback arrow back to the GP individual box
ax.annotate(
    "", xy=(r1[0] + 0.05, r1[1]), xytext=(r6[0] + 0.05, r6[1]),
    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6,
                     connectionstyle="arc3,rad=0.55"), zorder=2)
ax.text(6.9, 6.6, "evolved over\n50 generations", fontsize=8.3, color=GRAY,
        style="italic", ha="center", va="center")

ax.set_xlim(-0.3, 7.8)
ax.set_ylim(1.0, 12.5)
ax.set_aspect("equal")
ax.axis("off")

fig.tight_layout()
fig.savefig("img/gphh-architecture.png", dpi=200, bbox_inches="tight")
print("done")
