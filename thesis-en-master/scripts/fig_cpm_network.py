"""Generates img/cpm-network.png for chap01.tex.

Activity-on-node network for the six-activity worked example (Table 1.1):
each node is a three-row box showing (EST | d | EFT) on top, the activity
label in the middle, and (LST | TS | LFT) on the bottom -- using the
EST/EFT/LST/LFT/TS notation of Demeulemeester & Herroelen's forward/backward
pass and total-slack formulas (chap01.tex, Section 1.1.2).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "serif"

RED = "#b03a2e"
RED_FILL = "#f5d6d1"
BLUE = "#34495e"
BLUE_FILL = "#dde3ea"
GOLD = "#c8960c"

# ---- problem data and CPM results (chapter 1, Table 1.1 / worked example) --
DUR = {"A": 3, "B": 4, "C": 2, "D": 5, "E": 2, "F": 3}
PRED = {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"], "E": ["C"], "F": ["D", "E"]}

EST = {"A": 0, "B": 3, "C": 3, "D": 7, "E": 5, "F": 12}
EFT = {"A": 3, "B": 7, "C": 5, "D": 12, "E": 7, "F": 15}
LST = {"A": 0, "B": 3, "C": 5, "D": 7, "E": 10, "F": 12}
LFT = {"A": 3, "B": 7, "C": 7, "D": 12, "E": 12, "F": 15}
TS = {a: LST[a] - EST[a] for a in DUR}
CRITICAL = {a for a in DUR if TS[a] == 0}

POS = {
    "A": (0.175, 0.6),
    "B": (1.875, 1.15),
    "C": (1.875, 0.05),
    "D": (3.575, 0.6),
    "E": (3.575, -0.5),
    "F": (5.275, 0.6),
}

BOX_W, BOX_H = 1.10, 0.75
ROW_H = BOX_H / 3


def draw_node(ax, act):
    x, y = POS[act]
    crit = act in CRITICAL
    edge = RED if crit else BLUE
    face = RED_FILL if crit else BLUE_FILL
    box = mpatches.FancyBboxPatch(
        (x, y), BOX_W, BOX_H,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=1.8, edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(box)
    # row separators
    ax.plot([x, x + BOX_W], [y + 2 * ROW_H, y + 2 * ROW_H], color=edge, linewidth=1.1, zorder=3)
    ax.plot([x, x + BOX_W], [y + ROW_H, y + ROW_H], color=edge, linewidth=1.1, zorder=3)
    # column separators on top/bottom rows only
    for yy in (y + 2 * ROW_H, y):
        ax.plot([x + BOX_W / 3, x + BOX_W / 3], [yy, yy + ROW_H], color=edge, linewidth=0.9, zorder=3)
        ax.plot([x + 2 * BOX_W / 3, x + 2 * BOX_W / 3], [yy, yy + ROW_H], color=edge, linewidth=0.9, zorder=3)

    cx = x + BOX_W / 2
    ax.text(x + BOX_W / 6, y + 2.5 * ROW_H, str(EST[act]), ha="center", va="center", fontsize=9)
    ax.text(cx, y + 2.5 * ROW_H, str(DUR[act]), ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(x + 5 * BOX_W / 6, y + 2.5 * ROW_H, str(EFT[act]), ha="center", va="center", fontsize=9)

    ax.text(cx, y + 1.5 * ROW_H, act, ha="center", va="center", fontsize=15, fontweight="bold", color=edge)

    ts_color = GOLD if TS[act] > 0 else "black"
    ax.text(x + BOX_W / 6, y + 0.5 * ROW_H, str(LST[act]), ha="center", va="center", fontsize=9)
    ax.text(cx, y + 0.5 * ROW_H, str(TS[act]), ha="center", va="center", fontsize=10, fontweight="bold", color=ts_color)
    ax.text(x + 5 * BOX_W / 6, y + 0.5 * ROW_H, str(LFT[act]), ha="center", va="center", fontsize=9)


def edge_anchor(act, side):
    x, y = POS[act]
    if side == "out":
        return (x + BOX_W, y + BOX_H / 2)
    return (x, y + BOX_H / 2)


fig, ax = plt.subplots(figsize=(11, 5.3))

for act in DUR:
    for p in PRED[act]:
        crit = act in CRITICAL and p in CRITICAL
        color = RED if crit else "#7f8c8d"
        lw = 2.6 if crit else 1.4
        x0, y0 = edge_anchor(p, "out")
        x1, y1 = edge_anchor(act, "in")
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                     shrinkA=2, shrinkB=2), zorder=1)

for act in DUR:
    draw_node(ax, act)

# ---- legend box (top-left): row layout key --------------------------------
lx, ly = -0.05, 2.1
lw_, lh_ = 1.7, 0.8
lrow = lh_ / 3
ax.add_patch(mpatches.Rectangle((lx, ly), lw_, lh_, linewidth=1.2,
                                  edgecolor="0.4", facecolor="none", zorder=2))
ax.plot([lx, lx + lw_], [ly + 2 * lrow, ly + 2 * lrow], color="0.4", linewidth=1.0)
ax.plot([lx, lx + lw_], [ly + lrow, ly + lrow], color="0.4", linewidth=1.0)
for yy in (ly + 2 * lrow, ly):
    ax.plot([lx + lw_ / 3, lx + lw_ / 3], [yy, yy + lrow], color="0.4", linewidth=0.8)
    ax.plot([lx + 2 * lw_ / 3, lx + 2 * lw_ / 3], [yy, yy + lrow], color="0.4", linewidth=0.8)
ax.text(lx + lw_ / 6, ly + 2.5 * lrow, "EST", ha="center", va="center", fontsize=9.5, color="0.3")
ax.text(lx + lw_ / 2, ly + 2.5 * lrow, "d", ha="center", va="center", fontsize=9.5, color="0.3")
ax.text(lx + 5 * lw_ / 6, ly + 2.5 * lrow, "EFT", ha="center", va="center", fontsize=9.5, color="0.3")
ax.text(lx + lw_ / 2, ly + 1.5 * lrow, "ID", ha="center", va="center", fontsize=9.5, color="0.3")
ax.text(lx + lw_ / 6, ly + 0.5 * lrow, "LST", ha="center", va="center", fontsize=9.5, color="0.3")
ax.text(lx + lw_ / 2, ly + 0.5 * lrow, "TS", ha="center", va="center", fontsize=9.5, color="0.3")
ax.text(lx + 5 * lw_ / 6, ly + 0.5 * lrow, "LFT", ha="center", va="center", fontsize=9.5, color="0.3")

ax.text(5.0, 2.45, "Critical path:\nA → B → D → F\n(makespan = 15)",
        ha="center", va="center", fontsize=12, fontweight="bold", color=RED)

ax.set_xlim(-0.3, 6.7)
ax.set_ylim(-0.8, 3.05)
ax.axis("off")
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("img/cpm-network.png", dpi=200, bbox_inches="tight")
print("wrote img/cpm-network.png")
