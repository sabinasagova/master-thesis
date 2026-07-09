"""Generates img/representation-contrast.png for chap02.tex: contrasts the
fixed-length GA activity-list chromosome with the variable-size GP priority-rule
tree, motivating the move to genetic-programming hyper-heuristics in section 2.4.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "serif"

RED = "#b03a2e"
RED_FILL = "#f5d6d1"
BLUE = "#34495e"
BLUE_FILL = "#dde3ea"
GRAY = "#7a7a7a"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.0))

# ---------------------------------------------------------------- left: GA --
axL.set_xlim(0, 6)
axL.set_ylim(0, 4)
axL.axis("off")
axL.set_title("Genetic algorithm:\nactivity-list chromosome", fontsize=12.5, pad=14)

ACT = ["A", "C", "E", "B", "D", "F"]
for i, act in enumerate(ACT):
    b = mpatches.FancyBboxPatch(
        (i + 0.07, 2.2), 0.86, 0.8, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=BLUE, facecolor=BLUE_FILL)
    axL.add_patch(b)
    axL.text(i + 0.5, 2.6, act, ha="center", va="center", fontsize=14,
              fontweight="bold", color=BLUE)
axL.annotate("", xy=(3, 2.0), xytext=(3, 1.55),
             arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4))
axL.text(3, 1.3, "decode with one SGS pass", ha="center", fontsize=9.5, color=GRAY)
axL.annotate("", xy=(3, 1.05), xytext=(3, 0.7),
             arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4))
b = mpatches.FancyBboxPatch((1.3, 0.05), 3.4, 0.55,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             linewidth=1.6, edgecolor=RED, facecolor=RED_FILL)
axL.add_patch(b)
axL.text(3, 0.33, "one schedule, for one instance", ha="center", va="center",
         fontsize=10.5, color=RED, fontweight="bold")

# ---------------------------------------------------------------- right: GP --
axR.set_xlim(0, 6.4)
axR.set_ylim(0, 4)
axR.axis("off")
axR.set_title("Genetic programming:\npriority-rule tree", fontsize=12.5, pad=14)


def node(ax, x, y, text, edge, face, fontsize=12, w=0.95, h=0.55):
    b = mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                 boxstyle="round,pad=0.02,rounding_size=0.08",
                                 linewidth=1.6, edgecolor=edge, facecolor=face,
                                 zorder=3)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=edge, zorder=4)
    return x, y


def edge(ax, p, q, color=GRAY):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linewidth=1.2, zorder=1)


root = node(axR, 3.1, 3.35, "min", BLUE, BLUE_FILL, 13, w=1.0, h=0.6)
lst = node(axR, 1.5, 2.3, "LST", BLUE, BLUE_FILL, 11.5)
sub = node(axR, 4.7, 2.3, "$-$", BLUE, BLUE_FILL, 13)
edge(axR, root, lst)
edge(axR, root, sub)

grpw = node(axR, 3.8, 1.15, "GRPW", BLUE, BLUE_FILL, 10.5, w=1.05)
dur = node(axR, 5.55, 1.15, "Duration", BLUE, BLUE_FILL, 9.5, w=1.1)
edge(axR, sub, grpw)
edge(axR, sub, dur)

b = mpatches.FancyBboxPatch((0.55, 0.05), 5.3, 0.55,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             linewidth=1.6, edgecolor=RED, facecolor=RED_FILL,
                             zorder=0)
axR.add_patch(b)
axR.text(3.2, 0.32, "one rule, compiled by SGS on any instance",
         ha="center", va="center", fontsize=10.2, color=RED, fontweight="bold",
         zorder=2)

fig.tight_layout()
fig.savefig("img/representation-contrast.png", dpi=200, bbox_inches="tight")
print("done")
