"""Aggregate full108_eval.py's per-cell records into a per-(dataset, sgs,
strategy, condition) summary and emit a LaTeX table comparing the 10-class
test fitness the matrix experiment reported against the full 108-class
re-evaluation of the same evolved heuristics.

Run: PYTHONPATH=$(pwd) python yuantian/experiments/analyze_full108.py
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results" / "full108_evaluation" / "all_runs.json"

COND_ORDER = ["baseline", "lexicase", "local_search", "hybrid",
              "baseline_nr", "nr"]
COND_LABEL = {"baseline": "baseline", "lexicase": "lexicase",
              "local_search": "local search", "hybrid": "hybrid",
              "baseline_nr": r"baseline\_nr", "nr": "NR terminals"}
STRAT_ORDER = ["AF", "MF", "S"]
STRAT_LABEL = {"AF": "AF", "MF": "MF", "S": "S"}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def aggregate(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["dataset"], r["sgs"], r["strategy"], r["condition"])].append(r)
    summary = {}
    for key, rs in groups.items():
        summary[key] = {
            "n": len(rs),
            "ard10": _mean([r["test_ard_10"] for r in rs]),
            "ard108": _mean([r["test_ard_108"] for r in rs]),
            "feas10": _mean([r["feasibility_10"] for r in rs]),
            "feas108": _mean([r["feasibility_108"] for r in rs]),
        }
    return summary


def _fmt(x, pct=False):
    if x is None:
        return "--"
    return f"{100*x:.0f}\\%" if pct else f"{x:.1f}"


def latex_table(summary, dataset):
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        r"    & & \multicolumn{2}{c}{\textbf{Test ARD\%}} "
        r"& \multicolumn{2}{c}{\textbf{Feasible}} \\",
        r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"    \textbf{SGS} & \textbf{Cond.} & \textbf{10 cls} "
        r"& \textbf{108 cls} & \textbf{10 cls} & \textbf{108 cls} \\",
        r"    \midrule",
    ]
    for sgs in ("serial", "parallel"):
        first = True
        for strat in STRAT_ORDER:
            for cond in COND_ORDER:
                key = (dataset, sgs, strat, cond)
                if key not in summary:
                    continue
                s = summary[key]
                sgs_cell = f"{sgs}/{STRAT_LABEL[strat]}" if first else ""
                first = False
                lines.append(
                    f"    {sgs_cell} & {COND_LABEL[cond]} "
                    f"& {_fmt(s['ard10'])} & {_fmt(s['ard108'])} "
                    f"& {_fmt(s['feas10'], True)} & {_fmt(s['feas108'], True)} \\\\")
            lines.append(r"    \addlinespace")
        lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines += [
        r"  \end{tabular}",
        rf"  \caption{{Full 108-class re-evaluation, {dataset}.}}",
        rf"  \label{{tab:full108-{dataset.lower()}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    records = json.load(open(RESULTS))
    summary = aggregate(records)

    # console summary
    print(f"{'dataset':9} {'sgs':8} {'strat':5} {'cond':13} "
          f"{'n':>3} {'ard10':>7} {'ard108':>7} {'feas10':>7} {'feas108':>7}")
    for key in sorted(summary):
        s = summary[key]
        ds, sgs, strat, cond = key
        print(f"{ds:9} {sgs:8} {strat:5} {cond:13} {s['n']:3} "
              f"{(s['ard10'] or 0):7.1f} {(s['ard108'] or 0):7.1f} "
              f"{(s['feas10'] or 0):7.2f} {(s['feas108'] or 0):7.2f}")

    for dataset in ("MMLIB50", "MMLIB100"):
        print("\n" + latex_table(summary, dataset))


if __name__ == "__main__":
    main()
