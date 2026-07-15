"""Renders actual evolved GP heuristics (real activity-tree / mode-tree pairs
taken from the experiment results) as vector figures for the thesis, using
DEAP's gp.graph plus Graphviz.

Unlike img/gp-tree-activity-mode.png (a hand-drawn schematic), these are the
genuine symbolic expressions produced by the evolutionary runs, so the reader
sees what each configuration actually evolves. For every configuration the
most compact rule pair (fewest total nodes, subject to the constraints below)
is chosen, so the figures stay small and legible.

Three configurations are rendered:
  * nr           -- the proposed NR-aware GPHH; the most compact run that
                    recruits the contributed NR terminals in *both* trees.
  * lexicase     -- epsilon-lexicase selection (side extension).
  * local_search -- critical-path local search on elites (side extension).

The non-renewable / critical-path terminals contributed by this work are
highlighted in gold wherever they appear.

Outputs (img/generated/):
  gp_tree_evolved_activity.pdf     gp_tree_evolved_mode.pdf     (nr)
  gp_tree_lexicase_activity.pdf    gp_tree_lexicase_mode.pdf
  gp_tree_localsearch_activity.pdf gp_tree_localsearch_mode.pdf
"""

import glob
import json
import os
import re

from deap import gp
from deap.gp import PrimitiveSet, PrimitiveTree

# ---------------------------------------------------------------- config --
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO, "GP_MRCPSP_CEC2024/yuantian/experiments/results")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "img", "generated")

# binary function set of the GPHH (see gphh_solver.py)
PRIMITIVES = {"add": 2, "sub": 2, "mul": 2, "div": 2, "min": 2, "max": 2}
# pretty math symbols for the operator nodes
PRIM_LABEL = {"add": "+", "sub": "−", "mul": "×", "div": "%",
              "min": "min", "max": "max"}

# the terminals introduced by this work (highlighted in the figure)
NR_CP_RE = re.compile(r"^(NR_|CP_)")

# colours (consistent with fig_gp_tree_activity_mode.py)
BLUE, BLUE_FILL = "#34495e", "#dde3ea"
GOLD, GOLD_FILL = "#c8960c", "#f5e7c4"
GRAY, GRAY_FILL = "#5d6d7e", "#eef1f4"

MAX_FEASIBLE_FITNESS = 1000.0  # runs with test fitness above this are penalties


# ------------------------------------------------------------- utilities --
def _tokens(expr):
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)


def _n_nodes(expr):
    return len(_tokens(expr))


def _uses_nr(expr):
    return any(NR_CP_RE.match(t) for t in _tokens(expr))


def _has_operator(expr):
    return any(t in PRIMITIVES for t in _tokens(expr))


def _split_tree(tree):
    """Return (activity_expr, mode_expr) from a stored best-tree field, which
    is either a dict or the string repr of a dict with 'activity'/'mode'."""
    if isinstance(tree, dict):
        return tree.get("activity", ""), tree.get("mode", "")
    a = re.search(r"'activity':\s*'([^']*)'", tree)
    m = re.search(r"'mode':\s*'([^']*)'", tree)
    return (a.group(1) if a else ""), (m.group(1) if m else "")


# -------------------------------------------------- self-division pruning
# A protected-division node div(A, A) with two identical terminal children
# is a trivial intron (it always evaluates to 1). Drawing both copies of
# the terminal is visual clutter, so for display one of the two identical
# children is dropped, leaving the division node with a single child. This
# is a rendering-only tweak of the graph; nothing about the evolved
# individual, its fitness, or any reported statistic changes.
def _prune_div_self(nodes, edges, labels):
    from collections import defaultdict
    children = defaultdict(list)
    for a, b in edges:
        children[a].append(b)
    leaves = {n for n in nodes if n not in children}
    drop_nodes, drop_edges = set(), set()
    for d in nodes:
        if labels.get(d) == "div" and len(children[d]) == 2:
            c1, c2 = children[d]
            if (c1 in leaves and c2 in leaves
                    and labels[c1] == labels[c2] and c2 not in drop_nodes):
                drop_nodes.add(c2)
                drop_edges.add((d, c2))
    nodes = [n for n in nodes if n not in drop_nodes]
    edges = [e for e in edges if e not in drop_edges]
    return nodes, edges


# ------------------------------------------------------------- rendering --
def build_pset(expr):
    """A minimal PrimitiveSet able to re-parse the stored expression string."""
    pset = PrimitiveSet("MAIN", 0)
    for name, arity in PRIMITIVES.items():
        pset.addPrimitive(lambda *a: 0, arity, name=name)
    for tok in set(_tokens(expr)):
        if tok not in PRIMITIVES:
            pset.addTerminal(0, name=tok)
    return pset


def render(expr, out_path, title, prune_div_self=False):
    import pygraphviz as pgv

    tree = PrimitiveTree.from_string(expr, build_pset(expr))
    nodes, edges, labels = gp.graph(tree)
    if prune_div_self:
        nodes, edges = _prune_div_self(nodes, edges, labels)

    g = pgv.AGraph(strict=True, directed=True, rankdir="TB")
    g.graph_attr.update(bgcolor="transparent", ordering="out", nodesep="0.25",
                        ranksep="0.45", fontname="Helvetica", label=title,
                        labelloc="t", fontsize="16")
    g.node_attr.update(fontname="Helvetica", fontsize="13", penwidth="1.6")
    g.edge_attr.update(color=GRAY, penwidth="1.3", arrowsize="0.6")

    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    for i in nodes:
        name = labels[i]
        n = g.get_node(i)
        if name in PRIMITIVES:  # operator
            n.attr.update(label=PRIM_LABEL[name], shape="box",
                          style="rounded,filled", color=BLUE,
                          fillcolor=BLUE_FILL, fontcolor=BLUE, fontsize="15")
        elif NR_CP_RE.match(name):  # the contributed NR / CP terminal
            n.attr.update(label=name, shape="ellipse", style="filled",
                          color=GOLD, fillcolor=GOLD_FILL, fontcolor=GOLD)
        else:  # ordinary terminal
            n.attr.update(label=name, shape="ellipse", style="filled",
                          color=GRAY, fillcolor=GRAY_FILL, fontcolor=BLUE)

    g.layout(prog="dot")
    g.draw(out_path)
    print("    wrote", os.path.basename(out_path), f"({len(nodes)} nodes)")


# ------------------------------------------------------------ selection ---
def _pick_most_compact(candidates, require_nr, pin=None):
    """candidates: list of dicts with keys activity, mode, test_fitness, meta.
    Returns the one with the fewest total nodes (tie-break: lowest test
    fitness) that is feasible, has an operator in both trees, and -- if
    require_nr -- recruits an NR/CP terminal in both trees. If ``pin`` is
    given, instead return the candidate whose meta contains that substring
    (used to select a hand-chosen, more interpretable individual rather than
    merely the smallest, e.g. the local-search rule that genuinely combines
    several terminals instead of collapsing to a single LST terminal)."""
    viable = []
    for c in candidates:
        a, m, tf = c["activity"], c["mode"], c["test_fitness"]
        if tf is None or tf >= MAX_FEASIBLE_FITNESS:
            continue
        if not (a and m and _has_operator(a) and _has_operator(m)):
            continue
        if require_nr and not (_uses_nr(a) and _uses_nr(m)):
            continue
        c["_size"] = _n_nodes(a) + _n_nodes(m)
        viable.append(c)
    if pin is not None:
        return next(c for c in viable if pin in c["meta"])
    return min(viable, key=lambda c: (c["_size"], c["test_fitness"]))


def load_nr_runs():
    path = os.path.join(RESULTS_DIR, "nr_terminals_experiment/all_runs.json")
    out = []
    for e in json.load(open(path)):
        if e["condition"] != "baseline+nr":
            continue
        a, m = _split_tree(eval(e["best_tree"]))
        out.append({"activity": a, "mode": m, "test_fitness": e["test_fitness"],
                    "meta": f"serial/activity-first, MMLIB50, seed {e['seed']}"})
    return out


def load_matrix_runs(condition):
    out = []
    for f in glob.glob(os.path.join(RESULTS_DIR, f"matrix/*__{condition}__*.json")):
        d = json.load(open(f))
        bhv = d["best_heuristic_validation"]
        a, m = _split_tree(bhv["tree"])
        cell = d["matrix_cell"]
        strat = {"AF": "activity-first", "MF": "mode-first",
                 "S": "simultaneous"}.get(cell["strategy"], cell["strategy"])
        out.append({"activity": a, "mode": m,
                    "test_fitness": bhv.get("test_fitness"),
                    "meta": f"{cell['sgs']}/{strat}, {cell['dataset']}, "
                            f"seed {cell['seed']}"})
    return out


# ---------------------------------------------------------------- driver --
# The last flag drops one of the two identical terminal children of a
# self-division node div(A, A) for display (see _prune_div_self); enabled
# only for the NR figure, whose mode tree divides NR_MODE_DEMAND_RATIO by
# itself, so it renders as a single "%" edge to one NR_MODE_DEMAND_RATIO.
# Fields: label, output stem, loader, require_nr, prune_div_self,
# pin_activity, pin_mode. The two local-search panels are pinned to two
# hand-chosen champions of the same serial/mode-first MMLIB100 cell: the
# default smallest individual's activity tree collapses to the plain
# minimum-LST terminal LS and is uninformative to interpret, so the
# activity panel shows seed 12022's genuinely composite rule
# min(LS, max(LS/SC, ES+PC)) while the mode panel keeps seed 12003's
# compact EFFT/RR-driven tree. The figure caption states that the two
# panels come from two champions.
CONFIGS = [
    ("nr", "gp_tree_evolved", load_nr_runs, True, True, None, None),
    ("lexicase", "gp_tree_lexicase", lambda: load_matrix_runs("lexicase"),
     False, False, None, None),
    ("local_search", "gp_tree_localsearch",
     lambda: load_matrix_runs("local_search"), False, False,
     "serial/mode-first, MMLIB100, seed 12022",
     "serial/mode-first, MMLIB100, seed 12003"),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for label, stem, loader, require_nr, collapse, pin_act, pin_mode in CONFIGS:
        cands = loader()
        best_act = _pick_most_compact(cands, require_nr, pin=pin_act)
        best_mode = (best_act if pin_mode == pin_act
                     else _pick_most_compact(cands, require_nr, pin=pin_mode))
        for role, best in (("activity", best_act), ("mode", best_mode)):
            print(f"{label}/{role}: {best['meta']}, test fitness "
                  f"{best['test_fitness']:.2f}, total nodes {best['_size']}")
        render(best_act["activity"],
               os.path.join(OUT_DIR, f"{stem}_activity.pdf"),
               "Activity tree", prune_div_self=collapse)
        render(best_mode["mode"], os.path.join(OUT_DIR, f"{stem}_mode.pdf"),
               "Mode tree", prune_div_self=collapse)


if __name__ == "__main__":
    main()
