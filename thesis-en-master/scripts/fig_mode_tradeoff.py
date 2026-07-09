"""Generates img/mode-tradeoff.png for chap03.tex: visualizes the duration
vs. resource-consumption trade-off across the three execution modes of
activity 2 in the MMLIB50 instance J50100_1.mm, whose REQUESTS/DURATIONS
block is quoted verbatim in Section 3.2 (Data instance structure):

    jobnr. mode  dur  R1  R2  N1  N2
    2      1     2    0   5   9   9
           2     5    5   0   3   5
           3     6    5   0   2   1

Left panel: a small Gantt-style bar per mode, bar length = duration.
Right panel: renewable (R1, R2) and nonrenewable (N1, N2) demand per mode,
showing that the fastest mode is also the heaviest nonrenewable consumer.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"

RED = "#b03a2e"
GOLD = "#c8960c"
BLUE = "#34495e"
BLUE_FILL = "#dde3ea"
GRAY = "#7a7a7a"

MODES = ["Mode 1", "Mode 2", "Mode 3"]
DUR = [2, 5, 6]
R1 = [0, 5, 5]
R2 = [5, 0, 0]
N1 = [9, 3, 2]
N2 = [9, 5, 1]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.2))

# ---------------------------------------------------------- left: duration --
axL.set_title("Duration $d_{2,m}$ by mode", fontsize=12, fontweight="bold",
              pad=10)
ys = np.arange(len(MODES))
colors = [RED, GOLD, BLUE]
for y, d, c, name in zip(ys, DUR, colors, MODES):
    axL.barh(y, d, height=0.55, left=0, color=c, alpha=0.85, zorder=3)
    axL.text(d + 0.15, y, f"{d} periods", va="center", fontsize=9.5, color=c,
              fontweight="bold")
axL.set_yticks(ys)
axL.set_yticklabels(MODES, fontsize=10.5)
axL.set_xlabel("Duration (time periods)", fontsize=10)
axL.set_xlim(0, 8.5)
axL.invert_yaxis()
axL.spines["top"].set_visible(False)
axL.spines["right"].set_visible(False)

# --------------------------------------------------- right: resource demand --
axR.set_title("Resource demand $r_{2,m,k}$ by mode", fontsize=12,
              fontweight="bold", pad=10)
resources = ["R1\n(renewable)", "R2\n(renewable)", "N1\n(nonrenewable)",
             "N2\n(nonrenewable)"]
data = np.array([R1, R2, N1, N2])  # shape (4 resources, 3 modes)

x = np.arange(len(resources))
width = 0.25
for i, (name, c) in enumerate(zip(MODES, colors)):
    axR.bar(x + (i - 1) * width, data[:, i], width=width, color=c, alpha=0.85,
            label=name, zorder=3)

axR.set_xticks(x)
axR.set_xticklabels(resources, fontsize=9.5)
axR.set_ylabel("Units requested", fontsize=10)
axR.legend(fontsize=9.5, frameon=False)
axR.spines["top"].set_visible(False)
axR.spines["right"].set_visible(False)
axR.axvline(1.5, color=GRAY, linewidth=0.8, linestyle=":")
axR.text(1.5, axR.get_ylim()[1] * 0.95, "renewable $\\vert$ nonrenewable",
          ha="center", va="top", fontsize=8.2, color=GRAY, style="italic")

fig.suptitle(
    "Activity 2, instance J50100_1.mm: the fastest mode (Mode 1) is also "
    "the heaviest\nconsumer of both nonrenewable resources — the central "
    "trade-off of the MRCPSP.",
    fontsize=9.6, y=1.04, color=GRAY)

fig.tight_layout()
fig.savefig("img/mode-tradeoff.png", dpi=200, bbox_inches="tight")
print("done")
