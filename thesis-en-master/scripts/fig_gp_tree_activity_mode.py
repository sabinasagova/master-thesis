"""Generates img/gp-tree-activity-mode.png for chap04.tex: a concrete
multi-tree GP individual -- one activity-tree and one mode-tree priority
rule -- using the exact two examples already discussed in the text
(Section 4.3's if_then_else(IS_ON_CRITICAL_PATH, ...) example and Section
4.4's if_else(NR_STOCK_RATIO, ...) example), so the abstract "two cooperating
trees" description of the baseline's representation has a concrete picture
to anchor it.
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


def node(ax, x, y, text, edge, face, fontsize=10, w=1.5, h=0.62):
    b = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=3)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=edge, zorder=4, linespacing=1.2)
    return x, y


def edge(ax, p, q, color=GRAY):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linewidth=1.3, zorder=1)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.6))

# --------------------------------------------------------------- activity --
axL.set_xlim(0, 6.4)
axL.set_ylim(0, 4.6)
axL.axis("off")
axL.set_title("Activity tree (selects which\nactivity to schedule next)",
              fontsize=12, pad=10)

root = node(axL, 3.2, 3.7, "if_then_else", BLUE, BLUE_FILL, 10.5, w=2.0)
cond = node(axL, 1.3, 2.5, "IS_ON_CRITICAL\n_PATH", GOLD, GOLD_FILL, 8.6, w=1.7)
fa = node(axL, 3.2, 2.5, "$f_A$", RED, RED_FILL, 11, w=0.95)
fb = node(axL, 5.1, 2.5, "$f_B$", RED, RED_FILL, 11, w=0.95)
edge(axL, root, cond)
edge(axL, root, fa)
edge(axL, root, fb)

esd = node(axL, 2.4, 1.2, "EST", BLUE, BLUE_FILL, 9.5, w=0.9, h=0.5)
edge(axL, fa, esd)
sub = node(axL, 4.2, 1.2, "$-$", BLUE, BLUE_FILL, 11, w=0.8, h=0.5)
edge(axL, fb, sub)
grpw = node(axL, 3.4, 0.35, "GRPW", BLUE, BLUE_FILL, 8.6, w=1.0, h=0.45)
slack = node(axL, 5.0, 0.35, "Slack", BLUE, BLUE_FILL, 8.6, w=1.0, h=0.45)
edge(axL, sub, grpw)
edge(axL, sub, slack)

axL.text(3.2, 4.25,
         "critical activities ($f_A$): rush them as early as possible\n"
         "non-critical activities ($f_B$): trade slack for resource savings",
         ha="center", va="center", fontsize=8.4, color=GRAY, style="italic")

# -------------------------------------------------------------------- mode --
axR.set_xlim(0, 6.4)
axR.set_ylim(0, 4.6)
axR.axis("off")
axR.set_title("Mode tree (selects which\nexecution mode to assign)",
              fontsize=12, pad=10)

root2 = node(axR, 3.2, 3.7, "if_else", BLUE, BLUE_FILL, 11, w=1.8)
cond2 = node(axR, 1.3, 2.5, "NR_STOCK\n_RATIO", GOLD, GOLD_FILL, 8.6, w=1.6)
out1 = node(axR, 3.2, 2.5, "NR_MODE_\nDEMAND_RATIO", RED, RED_FILL, 7.6, w=1.9)
out2 = node(axR, 5.2, 2.5, "TASK_\nDURATION", BLUE, BLUE_FILL, 8.6, w=1.5)
edge(axR, root2, cond2)
edge(axR, root2, out1)
edge(axR, root2, out2)

axR.text(3.2, 1.55,
         "low remaining NR budget: penalise modes\nthat would exhaust it "
         "(NR_MODE_DEMAND_RATIO)",
         ha="center", va="center", fontsize=8.4, color=GRAY, style="italic")
axR.text(3.2, 0.75,
         "ample remaining NR budget: minimise\nduration instead "
         "(TASK_DURATION)",
         ha="center", va="center", fontsize=8.4, color=GRAY, style="italic")

fig.tight_layout()
fig.savefig("img/gp-tree-activity-mode.png", dpi=200, bbox_inches="tight")
print("done")
