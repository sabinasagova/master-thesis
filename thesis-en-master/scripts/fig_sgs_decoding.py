"""Generates img/sgs-decoding.png for chap02.tex.

Extends the six-activity example of chapter 1 (Table 1.1 / Figure 1.1-1.2) with a
single renewable resource R of capacity 2, so that the previously unconstrained
CPM schedule is no longer feasible. A serial SGS is run on the activity-list
chromosome [A, C, E, B, D, F] and the resulting resource-feasible schedule is
plotted as a Gantt chart, in the same visual style as gantt-slack.png.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "serif"

RED = "#b03a2e"
RED_FILL = "#f5d6d1"
BLUE = "#34495e"
BLUE_FILL = "#dde3ea"
GOLD = "#c8960c"

# ---- problem data (durations and precedence from chapter 1, Table 1.1) -----
ACT = ["A", "B", "C", "D", "E", "F"]
DUR = {"A": 3, "B": 4, "C": 2, "D": 5, "E": 2, "F": 3}
PRED = {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"], "E": ["C"], "F": ["D", "E"]}
DEMAND = {"A": 1, "B": 2, "C": 1, "D": 2, "E": 1, "F": 1}   # units of resource R
CAPACITY = 2

# Activity-list chromosome (priority order used by the serial SGS)
CHROMOSOME = ["A", "C", "E", "B", "D", "F"]


def serial_sgs(activity_list):
    """Minimal serial SGS: schedule activities one at a time, in the order given
    by the chromosome, each at the earliest time that respects precedence AND
    the resource capacity (scanned in unit time steps)."""
    start, finish = {}, {}
    usage = {}  # time -> total resource usage already committed

    def demand_at(t, act):
        s = start[act]
        f = finish[act]
        return DEMAND[act] if s <= t < f else 0

    for act in activity_list:
        earliest = max((finish[p] for p in PRED[act]), default=0)
        t = earliest
        while True:
            # check resource feasibility for the whole duration [t, t+dur)
            ok = True
            for tau in range(t, t + DUR[act]):
                used = sum(demand_at(tau, a) for a in start)
                if used + DEMAND[act] > CAPACITY:
                    ok = False
                    break
            if ok:
                start[act], finish[act] = t, t + DUR[act]
                break
            t += 1
    return start, finish


start, finish = serial_sgs(CHROMOSOME)
makespan = max(finish.values())

fig, (ax_chrom, ax_gantt) = plt.subplots(
    2, 1, figsize=(9.5, 5.6), gridspec_kw={"height_ratios": [1, 3.2]}
)

# ---- top panel: the chromosome ---------------------------------------------
ax_chrom.set_xlim(0, len(CHROMOSOME))
ax_chrom.set_ylim(0, 1)
ax_chrom.axis("off")
ax_chrom.text(-0.05, 0.5, "Chromosome:", ha="right", va="center",
              fontsize=12, transform=ax_chrom.transData)
for i, act in enumerate(CHROMOSOME):
    box = mpatches.FancyBboxPatch(
        (i + 0.08, 0.12), 0.84, 0.76,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=BLUE, facecolor=BLUE_FILL)
    ax_chrom.add_patch(box)
    ax_chrom.text(i + 0.5, 0.5, act, ha="center", va="center",
                  fontsize=15, fontweight="bold", color=BLUE)
    if i < len(CHROMOSOME) - 1:
        ax_chrom.annotate("", xy=(i + 1.08, 0.5), xytext=(i + 0.92, 0.5),
                           arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.2))
ax_chrom.text(len(CHROMOSOME) + 0.3, 0.5,
              f"  resource $R$: capacity {CAPACITY}",
              ha="left", va="center", fontsize=11, color="gray")

# ---- bottom panel: resulting Gantt chart -----------------------------------
ax = ax_gantt
CRITICAL_LOOKING = {"A", "B", "D", "F"}  # same critical chain as ch.1 (precedence-critical)
for i, act in enumerate(ACT):
    y = len(ACT) - i - 1
    is_long_chain = act in CRITICAL_LOOKING
    face = RED_FILL if is_long_chain else BLUE_FILL
    edge = RED if is_long_chain else BLUE
    ax.barh(y, DUR[act], left=start[act], height=0.6,
            color=face, edgecolor=edge, linewidth=1.8, zorder=3)
    ax.text(start[act] + DUR[act] / 2, y, act, ha="center", va="center",
            fontsize=13, fontweight="bold", color=edge, zorder=4)
    ax.text(start[act] + DUR[act] / 2, y - 0.42, f"R={DEMAND[act]}",
            ha="center", va="center", fontsize=8.5, color="gray")

ax.axvline(makespan, color=RED, linestyle="--", linewidth=1.4)
ax.text(makespan + 0.15, len(ACT) - 0.3, f"makespan = {makespan}",
        color=RED, fontsize=11, va="center")
ax.axvline(15, color="gray", linestyle=":", linewidth=1.2)
ax.text(15 + 0.15, len(ACT) - 1.3, "unconstrained\nCPM makespan = 15",
        color="gray", fontsize=8.5, va="center")

ax.set_yticks(range(len(ACT)))
ax.set_yticklabels(reversed(ACT), fontsize=12)
ax.set_xlabel("Time", fontsize=12)
ax.set_xlim(0, makespan + 3)
ax.set_ylim(-1, len(ACT))
ax.grid(axis="x", color="0.85", linewidth=0.8, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

handles = [
    mpatches.Patch(facecolor=RED_FILL, edgecolor=RED, label="Activity on the precedence-critical chain"),
    mpatches.Patch(facecolor=BLUE_FILL, edgecolor=BLUE, label="Other activity"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
          ncol=2, frameon=False, fontsize=10)

fig.tight_layout()
fig.savefig("img/sgs-decoding.png", dpi=200, bbox_inches="tight")
print(f"start={start}\nfinish={finish}\nmakespan={makespan}")
