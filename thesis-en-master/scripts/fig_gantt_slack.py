"""Generates img/gantt-slack.png for chap01.tex.

Early-start Gantt chart for the six-activity worked example (Table 1.1):
solid bars at each activity's EST, with the total slack TS_j of the
non-critical activities (C, E) drawn as a hatched reserve to their right,
using the TS notation of Demeulemeester & Herroelen's Eq. (4.3)
(chap01.tex, Section 1.1.3), matching img/cpm-network.png.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "serif"

RED = "#b03a2e"
RED_FILL = "#f5d6d1"
BLUE = "#34495e"
BLUE_FILL = "#dde3ea"
GOLD = "#c8960c"

ACT = ["A", "B", "C", "D", "E", "F"]
DUR = {"A": 3, "B": 4, "C": 2, "D": 5, "E": 2, "F": 3}
EST = {"A": 0, "B": 3, "C": 3, "D": 7, "E": 5, "F": 12}
LST = {"A": 0, "B": 3, "C": 5, "D": 7, "E": 10, "F": 12}
TS = {a: LST[a] - EST[a] for a in DUR}
CRITICAL = {a for a in DUR if TS[a] == 0}
MAKESPAN = 15

fig, ax = plt.subplots(figsize=(9.5, 4.6))

for i, act in enumerate(ACT):
    y = len(ACT) - i - 1
    crit = act in CRITICAL
    face = RED_FILL if crit else BLUE_FILL
    edge = RED if crit else BLUE
    ax.barh(y, DUR[act], left=EST[act], height=0.6,
            color=face, edgecolor=edge, linewidth=1.8, zorder=3)
    ax.text(EST[act] + DUR[act] / 2, y, act, ha="center", va="center",
            fontsize=13, fontweight="bold", color=edge, zorder=4)
    if TS[act] > 0:
        ax.barh(y, TS[act], left=EST[act] + DUR[act], height=0.6,
                facecolor="none", edgecolor=GOLD, hatch="//", linewidth=1.6, zorder=2)
        ax.text(EST[act] + DUR[act] + TS[act] / 2, y - 0.55, f"TS={TS[act]}",
                ha="center", va="center", fontsize=9.5, color=GOLD)

ax.axvline(MAKESPAN, color=RED, linestyle="--", linewidth=1.4)
ax.text(MAKESPAN - 0.15, len(ACT) - 0.3, f"makespan = {MAKESPAN}",
        color=RED, fontsize=11, va="center", ha="right")

ax.set_yticks(range(len(ACT)))
ax.set_yticklabels(reversed(ACT), fontsize=12)
ax.set_xlabel("Time", fontsize=12)
ax.set_xticks(range(0, MAKESPAN + 1))
ax.set_xlim(-0.3, MAKESPAN + 0.3)
ax.set_ylim(-1, len(ACT))
ax.grid(axis="x", color="0.85", linewidth=0.8, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

handles = [
    mpatches.Patch(facecolor=RED_FILL, edgecolor=RED, label="Critical activity (TS = 0)"),
    mpatches.Patch(facecolor=BLUE_FILL, edgecolor=BLUE, label="Non-critical activity"),
    mpatches.Patch(facecolor="none", edgecolor=GOLD, hatch="//", label="Total slack (float)"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
          ncol=3, frameon=False, fontsize=10)

fig.tight_layout()
fig.savefig("img/gantt-slack.png", dpi=200, bbox_inches="tight")
print("wrote img/gantt-slack.png")
