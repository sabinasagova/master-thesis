"""What do the evolved champions actually decide by?

The thesis shows three hand-picked compact rule pairs (Figures 5.5, 5.10,
5.15) and interprets them. This script extends that inspection to *every*
comprehensive-stage champion (best_heuristic_validation of all matrix
cells), answering two questions:

1. Vocabulary: which terminals does each condition/SGS actually recruit,
   per tree (activity/mode/integrated)?
2. Behavioural simplicity: how many activity trees are, despite their
   size, order-equivalent to a single terminal (e.g. the LS collapse of
   Figure 5.10), tested by sampling random terminal assignments and
   checking whether the tree induces the same candidate ordering as the
   bare terminal? (A sampling test: equivalence is confirmed up to the
   sampled contexts, the same way Figure 5.10's collapse was first
   noticed, then proved by hand.)

Run (from the GP_MRCPSP_CEC2024 repo root):
    PYTHONPATH=$(pwd) python yuantian/experiments/champion_tree_analysis.py
"""
import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

MATRIX_DIR = Path(__file__).resolve().parent / "results" / "matrix"
PRIMITIVES = {"add", "sub", "mul", "div", "min", "max"}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

N_PAIRS = 400  # sampled candidate pairs per equivalence test
random.seed(12000)


def tokens(expr):
    return TOKEN_RE.findall(expr)


def terminals(expr):
    return [t for t in tokens(expr) if t not in PRIMITIVES]


def split_tree(tree_str):
    if tree_str.lstrip().startswith("{"):
        d = eval(tree_str)
        return d.get("activity", ""), d.get("mode", "")
    return None, None  # integrated (simultaneous)


def compile_expr(expr):
    """Compile a stored prefix expression into a python function of a
    terminal-value dict, with protected division."""
    def div(a, b):
        return a / b if abs(b) > 1e-9 else 1.0

    env = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b,
           "mul": lambda a, b: a * b, "div": div, "min": min, "max": max}
    names = sorted(set(terminals(expr)))
    src = "lambda ctx: " + re.sub(
        r"\b(?!(?:add|sub|mul|div|min|max)\b)([A-Za-z_][A-Za-z0-9_]*)\b",
        r"ctx['\1']", expr)
    return eval(src, env), names


def order_equivalent(expr, ref_terminal):
    """True iff, over N_PAIRS sampled candidate pairs, the tree always
    ranks candidates in the same order as ref_terminal alone."""
    f, names = compile_expr(expr)
    if ref_terminal not in names:
        return False
    for _ in range(N_PAIRS):
        a = {n: random.uniform(0.0, 10.0) for n in names}
        b = {n: random.uniform(0.0, 10.0) for n in names}
        da = a[ref_terminal] - b[ref_terminal]
        if abs(da) < 1e-6:
            continue
        try:
            df = f(a) - f(b)
        except (ZeroDivisionError, OverflowError):
            return False
        if df * da < 0 or abs(df) < 1e-12:
            return False
    return True


def main():
    files = sorted(glob.glob(str(MATRIX_DIR / "*.json")))
    vocab = defaultdict(Counter)        # (condition, sgs, tree) -> Counter
    n_champ = Counter()                 # (condition, sgs) -> count
    collapse = defaultdict(Counter)     # (condition, sgs) -> Counter of ref
    for path in files:
        d = json.load(open(path))
        cell = d["matrix_cell"]
        cond, sgs, strat = cell["condition"], cell["sgs"], cell["strategy"]
        tree = d["best_heuristic_validation"]["tree"]
        act, mode = split_tree(tree)
        key = (cond, sgs)
        n_champ[key] += 1
        if act is None:  # simultaneous: one integrated tree
            vocab[(cond, sgs, "integrated")].update(set(terminals(tree)))
            continue
        vocab[(cond, sgs, "activity")].update(set(terminals(act)))
        vocab[(cond, sgs, "mode")].update(set(terminals(mode)))
        for ref in ("LS", "LF", "ES", "EF", "GRPW", "TSC"):
            if order_equivalent(act, ref):
                collapse[key][ref] += 1
                break
        for ref in ("EFFT", "task_duration", "RR", "GRD"):
            if order_equivalent(mode, ref):
                collapse[key]["mode:" + ref] += 1
                break

    print("=== champions per (condition, sgs); two-tree strategies only "
          "feed the collapse test ===")
    for key in sorted(n_champ):
        print(key, n_champ[key])
    print("\n=== activity/mode trees order-equivalent to a single terminal "
          f"(sampling test, {N_PAIRS} pairs) ===")
    for key in sorted(collapse):
        total_twotree = n_champ[key] * 2 // 3  # AF+MF of the 3 strategies
        print(f"{key[0]:12s} {key[1]:8s} of {total_twotree:3d} two-tree "
              f"champions: {dict(collapse[key])}")
    print("\n=== most-recruited terminals (fraction of champions whose tree "
          "uses the terminal at least once) ===")
    for (cond, sgs, tree), cnt in sorted(vocab.items()):
        denom = n_champ[(cond, sgs)] * (2 if tree != "integrated" else 1) / 3
        denom = max(denom, 1)
        top = ", ".join(f"{t}:{c/denom:.0%}" for t, c in cnt.most_common(5))
        print(f"{cond:12s} {sgs:8s} {tree:10s} {top}")


if __name__ == "__main__":
    main()
