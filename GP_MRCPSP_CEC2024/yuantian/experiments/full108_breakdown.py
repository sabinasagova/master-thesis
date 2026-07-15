"""Per-instance-characteristic breakdown of the full-108-class re-evaluation.

full108_eval.py re-evaluated every matrix champion on the complete 108-class
held-out test set but stored only per-cell aggregates. This script repeats
that (cheap) re-evaluation while keeping the per-instance records, computes
each test class's OS/RF/RS indicators directly from the instance (the same
computation that produced the thesis's per-class table for the ten stratified
classes), and aggregates ARD% and feasibility separately for every value of
each indicator -- one breakdown table per indicator (OS, RF, RS).

Outputs (results/full108_evaluation/):
    class_indicators.json   {dataset: {class_id: {"OS":..,"RF":..,"RS":..}}}
    per_instance.json       one record per cell with 108-long per-instance
                            feasibility and (feasible-only) fitness lists
    breakdown_report.txt    the three aggregated tables, plain text + LaTeX

Run (from the GP_MRCPSP_CEC2024 repo root):
    PYTHONPATH=$(pwd) FULL108_WORKERS=4 python -O \
        yuantian/experiments/full108_breakdown.py
"""
import glob
import json
import os
import sys
from collections import deque
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discrete_optimization.rcpsp.rcpsp_parser import parse_file

from yuantian.experiments.full108_eval import (MATRIX_DIR, OUT_DIR,
                                               TEST_FILE_GETTER, evaluate_file)

sys.setrecursionlimit(10000)


# ---------------------------------------------------------------- indicators
def indicators(model):
    """(OS, RF, RS) of one instance: OS is the transitive-closure density of
    the precedence graph over non-dummy activities, RF the per-activity mean
    fraction of renewable resource types a mode requests, RS the mean over
    the renewable resources of (a_k - r_min)/(r_max - r_min) with r_min the
    largest unavoidable per-period demand and r_max the earliest-start-
    schedule peak (min-duration modes). Matches Section 3.1.4 of the thesis
    and reproduces its Table 5.22 values exactly."""
    jobs = model.mode_details
    source, sink = model.source_task, model.sink_task
    nondummy = [j for j in jobs if j not in (source, sink)]
    n = len(nondummy)
    succ = {j: set(model.successors.get(j, [])) for j in jobs}

    reach = {}

    def dfs(j):
        if j in reach:
            return reach[j]
        acc = set()
        for s in succ[j]:
            acc.add(s)
            acc |= dfs(s)
        reach[j] = acc
        return acc

    for j in jobs:
        dfs(j)
    npairs = sum(1 for j in nondummy for s in reach[j] if s in nondummy)
    os_val = npairs / (n * (n - 1) / 2)

    renew = [k for k in model.resources_list
             if k not in model.non_renewable_resources_list]

    tot = 0.0
    for j in nondummy:
        modes = jobs[j]
        acc = sum(sum(1 for k in renew if det.get(k, 0) > 0)
                  for det in modes.values())
        tot += acc / (len(modes) * len(renew))
    rf = tot / n

    # earliest-start schedule with min-duration modes, resource-relaxed
    dmin, mode_pick = {}, {}
    for j in jobs:
        m, det = min(jobs[j].items(), key=lambda kv: kv[1]["duration"])
        mode_pick[j], dmin[j] = m, det["duration"]
    est = {j: 0 for j in jobs}
    indeg = {j: 0 for j in jobs}
    for j in jobs:
        for s in succ[j]:
            indeg[s] += 1
    q = deque([j for j in jobs if indeg[j] == 0])
    while q:
        j = q.popleft()
        for s in succ[j]:
            est[s] = max(est[s], est[j] + dmin[j])
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)

    horizon = max(est[j] + dmin[j] for j in jobs) + 1
    rs_vals = []
    for k in renew:
        cap = model.resources[k]
        if isinstance(cap, list):
            cap = max(cap)
        rmin = max(min(det.get(k, 0) for det in jobs[j].values())
                   for j in nondummy)
        profile = [0] * horizon
        for j in nondummy:
            r = jobs[j][mode_pick[j]].get(k, 0)
            for t in range(est[j], est[j] + dmin[j]):
                profile[t] += r
        rmax = max(profile)
        rs_vals.append(1.0 if rmax == rmin else (cap - rmin) / (rmax - rmin))
    rs = sum(rs_vals) / len(rs_vals)
    return os_val, rf, rs


def compute_class_indicators():
    out = {}
    for ds, getter in TEST_FILE_GETTER.items():
        files = getter(5, 6)  # case 5 of every class 1..108
        out[ds] = {}
        for cls, f in enumerate(files, 1):
            o, rf, rs = indicators(parse_file(f))
            out[ds][str(cls)] = {"OS": round(o, 4), "RF": round(rf, 4),
                                 "RS": round(rs, 4)}
        print(f"indicators computed for {ds}", flush=True)
    return out


# --------------------------------------------------------------- evaluation
def evaluate_file_per_instance(path):
    """full108_eval.evaluate_file, but keeping the per-instance vectors.
    Relies on evaluate_file leaving case_feasible/case_fitness on the
    individual it builds; we re-run the same evaluation and read them."""
    # evaluate_file already computes everything; re-doing its internals here
    # would duplicate its conventions, so instead we call the same building
    # blocks it does.
    from deap import gp

    from yuantian.experiments.full108_eval import (_params, _test_set,
                                                   _wrap_individual,
                                                   NR_CONDITIONS,
                                                   STRATEGY_TO_DECISION)
    from yuantian.gphh_solver import evaluate_heuristic

    d = json.load(open(path))
    cell = d["matrix_cell"]
    cond, strat, sgs, ds = (cell["condition"], cell["strategy"],
                            cell["sgs"], cell["dataset"])
    params = _params(sgs, strat, cond)
    test = _test_set(ds, cond in NR_CONDITIONS)
    decision = STRATEGY_TO_DECISION[strat]

    ind = _wrap_individual(d["best_heuristic_validation"]["tree"])
    evaluate_heuristic(individual=ind, domains=test, compile_func=gp.compile,
                       pset=params.set_primitves, decision_type=decision,
                       simulator=params.simulator)

    return {
        "dataset": ds, "sgs": sgs, "strategy": strat,
        "condition": cond, "seed": cell["seed"],
        "feasible": [bool(x) for x in ind.case_feasible],
        "fitness": [round(v, 4) if ok else None
                    for v, ok in zip(ind.case_fitness, ind.case_feasible)],
    }


N_WORKERS = int(os.environ.get("FULL108_WORKERS", "4"))


# -------------------------------------------------------------- aggregation
CONDITIONS = ["baseline", "lexicase", "local_search", "hybrid",
              "baseline_nr", "nr"]
COND_LABEL = {"baseline": "Baseline", "lexicase": "Lexicase",
              "local_search": "Local search", "hybrid": "Hybrid",
              "baseline_nr": r"\texttt{baseline\_nr}", "nr": "NR terminals"}

RS_BUCKETS = [("$\\mathrm{RS} \\le 0.5$", lambda r: r <= 0.5),
              ("$0.5 < \\mathrm{RS} \\le 1$", lambda r: 0.5 < r <= 1.0),
              ("$\\mathrm{RS} > 1$", lambda r: r > 1.0)]


def _grouping(cls_ind, ds):
    """{indicator: [(column label, set of 0-based instance idx)]}"""
    ind = cls_ind[ds]
    groups = {}
    os_levels = sorted({round(v["OS"], 2) for v in ind.values()})
    groups["OS"] = [(f"$\\mathrm{{OS}} = {lv:.2f}$",
                     {int(c) - 1 for c, v in ind.items()
                      if round(v["OS"], 2) == lv}) for lv in os_levels]
    rf_levels = sorted({round(v["RF"], 2) for v in ind.values()})
    groups["RF"] = [(f"$\\mathrm{{RF}} = {lv:.2f}$",
                     {int(c) - 1 for c, v in ind.items()
                      if round(v["RF"], 2) == lv}) for lv in rf_levels]
    groups["RS"] = [(lab, {int(c) - 1 for c, v in ind.items()
                           if pred(v["RS"])}) for lab, pred in RS_BUCKETS]
    return groups


def aggregate():
    cls_ind = json.load(open(OUT_DIR / "class_indicators.json"))
    recs = json.load(open(OUT_DIR / "per_instance.json"))

    lines = []
    for indicator in ("OS", "RF", "RS"):
        lines.append(f"===== breakdown by {indicator} =====")
        for ds in ("MMLIB50", "MMLIB100"):
            groups = _grouping(cls_ind, ds)[indicator]
            lines.append(f"--- {ds} ({', '.join(f'{lab}: {len(idx)} classes' for lab, idx in groups)})")
            for cond in CONDITIONS:
                cells = [r for r in recs
                         if r["dataset"] == ds and r["condition"] == cond]
                if not cells:
                    lines.append(f"  {cond:12s} (no records yet)")
                    continue
                row = []
                for lab, idx in groups:
                    feas_flags, fits = [], []
                    for r in cells:
                        for i in idx:
                            feas_flags.append(r["feasible"][i])
                            if r["feasible"][i]:
                                fits.append(r["fitness"][i])
                    ard = sum(fits) / len(fits) if fits else float("nan")
                    feas = sum(feas_flags) / len(feas_flags)
                    row.append((ard, feas))
                cellstr = "  ".join(
                    f"{a:6.1f} ({f*100:3.0f}%)" for a, f in row)
                lines.append(f"  {cond:12s} {cellstr}")
        lines.append("")
    report = "\n".join(lines)
    (OUT_DIR / "breakdown_report.txt").write_text(report)
    print(report)


def main():
    if "--aggregate-only" in sys.argv:
        aggregate()
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ind_path = OUT_DIR / "class_indicators.json"
    if ind_path.exists():
        cls_ind = json.load(open(ind_path))
    else:
        cls_ind = compute_class_indicators()
        json.dump(cls_ind, open(ind_path, "w"))
        print("wrote", ind_path, flush=True)

    out = OUT_DIR / "per_instance.json"
    files = sorted(glob.glob(str(MATRIX_DIR / "*.json")))
    n = len(files)
    print(f"re-evaluating {n} cells with per-instance records "
          f"({N_WORKERS} workers)", flush=True)
    results = []
    with Pool(processes=N_WORKERS) as pool:
        for i, rec in enumerate(
                pool.imap_unordered(evaluate_file_per_instance, files,
                                    chunksize=1), 1):
            results.append(rec)
            if i % 50 == 0 or i == n:
                print(f"  {i}/{n}", flush=True)
                json.dump(results, open(out, "w"))
    json.dump(results, open(out, "w"))
    print("wrote", out, flush=True)
    aggregate()


if __name__ == "__main__":
    main()
