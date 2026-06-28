"""
Single source of truth for the full paper-spec experiment matrix: axes,
presets (paper_full / smoke), the Table VII cost model, and the paper's
reference numbers (Tables V/VI + manual-rule baselines) for the
reproduction report in analyze_matrix.py.

Tables IV, V, VI, and VII are transcribed in full directly from the paper
text (Tian, Mei & Zhang, "Learning Heuristics via Genetic Programming for
Multi-mode Resource-constrained Project Scheduling," CEC 2024).
PAPER_TABLE_V_SERIAL/PAPER_TABLE_VI_PARALLEL include both the GP mean+std
and the per-dataset manual-rule baseline (the manual rule's name and value
both vary by dataset -- see the comment above those dicts).
"""
from yuantian.rcpsp_dataset import RCPSPDatabase
from yuantian.rcpsp_simulation import DecisionTypeEnum, SimulatorTypeEnum

# ---------------------------------------------------------------------------
# Axes
#
# "baseline_nr" is the same algorithm as "baseline" (plain standard_gp, no
# NR terminals) but loaded on the NR-preserving instance set "nr" uses
# (keep_non_renewable=True, see matrix_runner.load_split) instead of the
# paper's renewable-only conversion -- it exists purely to give "nr" a
# same-instance paired control inside this matrix (analyze_matrix.py pairs
# nr against baseline_nr, not against the renewable-only "baseline" row).
# ---------------------------------------------------------------------------
CONDITIONS = ["baseline", "baseline_nr", "nr", "lexicase", "local_search", "hybrid"]
STRATEGIES = ["AF", "MF", "S"]  # activity-first / mode-first / simultaneous
SGS_TYPES = ["serial", "parallel"]
DATASETS = ["MMLIB50", "MMLIB100", "MMLIBPLUS_50", "MMLIBPLUS_100"]

STRATEGY_TO_DECISION_TYPE = {
    "AF": DecisionTypeEnum.ACTIVITY_THEN_MODE,
    "MF": DecisionTypeEnum.MODE_THEN_ACTIVITY,
    "S": DecisionTypeEnum.SIMULTANEOUS,
}
SGS_TO_SIMULATOR_TYPE = {
    "serial": SimulatorTypeEnum.SERIAL_SGS,
    "parallel": SimulatorTypeEnum.PARALLEL_SGS,
}

# dataset -> (directory, filename prefix, class id range [inclusive])
# MMLIB+'s "50"/"100" split is a class-id range within the SAME directory
# (Jall1..Jall324 = 50-activity instances, Jall325..Jall648 = 100-activity),
# not a separate folder -- confirmed against the actual instance files,
# same convention nr_terminals_mmlib_plus_experiment.py already uses.
DATASET_SPEC = {
    "MMLIB50": (RCPSPDatabase.MMLIB_50_DIR, "J50", (1, 108)),
    "MMLIB100": (RCPSPDatabase.MMLIB_100_DIR, "J100", (1, 108)),
    "MMLIBPLUS_50": (RCPSPDatabase.MMLIB_PLUS_DIR, "Jall", (1, 324)),
    "MMLIBPLUS_100": (RCPSPDatabase.MMLIB_PLUS_DIR, "Jall", (325, 648)),
}

# Paper's case-index convention (5 cases per class): cases 1-3 = train,
# case 4 = validation, case 5 = test. Same 60/20/20 split
# full_mmlib_experiment.py already uses for MMLIB50, generalized here to
# all 4 datasets (all of them have exactly 5 cases per class).
TRAIN_CASES = (1, 2, 3)
VAL_CASE = 4
TEST_CASE = 5


def stratified_classes(dataset: str, n_classes: int) -> list:
    """n_classes evenly spaced class numbers within dataset's id range."""
    lo, hi = DATASET_SPEC[dataset][2]
    n_total = hi - lo + 1
    n_classes = min(n_classes, n_total)
    step = max(1, n_total // n_classes)
    return [lo + i for i in range(0, n_total, step)][:n_classes]


def dataset_files(dataset: str, classes: list) -> tuple:
    """(train_files, val_files, test_files) full paths for the given classes."""
    directory, prefix, _ = DATASET_SPEC[dataset]
    train = [f"{directory}{prefix}{c}_{case}.mm" for c in classes for case in TRAIN_CASES]
    val = [f"{directory}{prefix}{c}_{VAL_CASE}.mm" for c in classes]
    test = [f"{directory}{prefix}{c}_{TEST_CASE}.mm" for c in classes]
    return train, val, test


# ---------------------------------------------------------------------------
# Paper Table IV params. ParametersGPHH.default() already matches these
# exactly (verified directly against gphh_solver.py before writing this
# file): pop=1000, n_gen=50, n_elite=10, n_tournament=7, crossover_rate=0.8,
# mutation_rate=0.15 (reproduction=0.05 is implicit in gp_algorithms.varOr's
# "else: apply reproduction" branch, since cxpb+mutpb=0.95), max_program_
# depth=8, init_min/max_tree_depth=2/6 via genHalfAndHalf (ramped
# half-and-half), function set {add,sub,mul,protected-div,min,max}. So the
# baseline condition needs no parameter override beyond decision_type and
# simulator_type -- matrix_runner.py calls ParametersGPHH.default(...)
# unmodified for it.
# ---------------------------------------------------------------------------
PAPER_POP_SIZE = 1000
PAPER_N_GEN = 50
PAPER_N_SEEDS = 30
PAPER_N_CLASSES = 10  # full_mmlib_experiment.py's own moderate-tier choice; paper itself uses all 108/324 classes, see SCALE NOTE in that script for why a full split is intractable

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "paper_full": dict(
        conditions=CONDITIONS, strategies=STRATEGIES, sgs_types=SGS_TYPES,
        datasets=DATASETS, n_seeds=PAPER_N_SEEDS, seed_base=11000,
        pop_size=PAPER_POP_SIZE, n_gen=PAPER_N_GEN, n_classes=PAPER_N_CLASSES,
    ),
    "smoke": dict(
        conditions=CONDITIONS, strategies=STRATEGIES, sgs_types=SGS_TYPES,
        datasets=["MMLIB50"], n_seeds=1, seed_base=99000,
        pop_size=20, n_gen=3, n_classes=3,
    ),
    # Focused subset: all 6 conditions (baseline/baseline_nr/nr/lexicase/
    # local_search/hybrid) x all 3 strategies x both SGS x MMLIB50/MMLIB100
    # only (no MMLIB+ -- those are the cells that need the 168h+ oven queue
    # in paper_full, see CELL_EXCLUSIONS/SEED_OVERRIDES above). n_seeds=30
    # matches PAPER_N_SEEDS, so this preset's own seed count lines up with
    # Tables V/VI's 30-run convention -- 6*3*2*2*30 = 2160 cells. Pairs with
    # generate_pbs_jobs.py's THESIS_CORE_WALLTIME_HOURS (fixed per-dataset
    # walltime, not the dynamic cost-tiering paper_full uses) and
    # --multiprocess --cpu_cores 8 baked into its PBS template for this
    # preset specifically -- see that module's docstring for why multiprocess
    # is required here, not just a
    # speedup.
    "thesis_core": dict(
        conditions=CONDITIONS, strategies=STRATEGIES, sgs_types=SGS_TYPES,
        datasets=["MMLIB50", "MMLIB100"], n_seeds=30, seed_base=12000,
        pop_size=PAPER_POP_SIZE, n_gen=PAPER_N_GEN, n_classes=PAPER_N_CLASSES,
    ),
}


# Cells dropped from the matrix entirely -- not a seed-count or queue
# question, genuinely out of scope. hybrid+MF/serial/MMLIBPLUS_100: even at
# a single seed, this combination's safety-adjusted estimate is ~178.5h
# (TABLE_VII_SECONDS=244763s x CONDITION_COST_MULTIPLIER["hybrid"]=1.75 x
# SAFETY_FACTOR=1.5), over the 168h ceiling actually available -- no seed
# count or queue choice fixes a per-job walltime problem. Flag this gap
# explicitly in the thesis write-up rather than silently having a missing
# row in the results table.
CELL_EXCLUSIONS = {
    ("hybrid", "MF", "serial", "MMLIBPLUS_100"),
}


def matrix_cells(preset: dict):
    """Yield every (condition, strategy, sgs, dataset) cell for a preset
    dict, skipping anything in CELL_EXCLUSIONS."""
    for dataset in preset["datasets"]:
        for sgs in preset["sgs_types"]:
            for strategy in preset["strategies"]:
                for condition in preset["conditions"]:
                    if (condition, strategy, sgs, dataset) in CELL_EXCLUSIONS:
                        continue
                    yield condition, strategy, sgs, dataset


# ---------------------------------------------------------------------------
# Per-(sgs, dataset, strategy) seed-count override, applied across all
# conditions in that slice. Added because a 168h walltime ceiling (not
# MetaCentrum's 720h oven max -- a tighter limit on the account actually
# being used) makes the slowest corner of the matrix infeasible as written.
#
# IMPORTANT, read before relying on this alone: reducing n_seeds does NOT
# reduce any single job's walltime -- one PBS array element is one seed, so
# per-job cost is set by pop_size/n_gen/condition/dataset/sgs/strategy, not
# by how many seeds are queued in total. Of the 12 (strategy x condition)
# combinations in (serial, MMLIBPLUS_100), only hybrid+MF actually exceeds
# 168h on its own: estimate_cell_seconds("hybrid", "MMLIBPLUS_100",
# "serial", "MF", 1000, 50) * SAFETY_FACTOR / 3600 = ~178.5h (computed from
# TABLE_VII_SECONDS=244763s x CONDITION_COST_MULTIPLIER["hybrid"]=1.75 x
# SAFETY_FACTOR=1.5). The other 11 combinations already fit the 168h tier
# (see generate_pbs_jobs.py's own tier printout). So this override controls
# how much TOTAL core-hours / queue time this expensive slice consumes --
# it does NOT by itself bring hybrid+MF under the 168h ceiling. To actually
# fit that one combination in a single job, pair this with --multiprocess
# on those specific manifest rows (cuts wall-clock, not CPU-seconds) or
# accept dropping hybrid+MF/serial/MMLIBPLUS_100 from this run and noting
# the gap in the thesis -- generate_pbs_jobs.py doesn't (yet) flip
# --multiprocess on automatically for over-budget cells.
SEED_OVERRIDES = {
    ("serial", "MMLIBPLUS_100", "MF"): 10,
    ("serial", "MMLIBPLUS_100", "S"): 10,
}


def n_seeds_for(preset: dict, sgs: str, dataset: str, strategy: str) -> int:
    """preset's global n_seeds, unless (sgs, dataset, strategy) has an
    override in SEED_OVERRIDES -- see its comment for why this exists and
    what it does/doesn't solve."""
    return SEED_OVERRIDES.get((sgs, dataset, strategy), preset["n_seeds"])


# ---------------------------------------------------------------------------
# Table VII cost model: average training time (seconds) over 30 independent
# runs at the paper's pop=1000/gen=50, by dataset x sgs x strategy.
# Transcribed in full directly from the paper text (Tian, Mei & Zhang, CEC
# 2024) -- exact per-strategy values, not the midpoint-of-range
# approximation an earlier pass here used before the paper text was
# available.
# ---------------------------------------------------------------------------
TABLE_VII_SECONDS = {
    ("MMLIB50", "serial", "AF"): 940,
    ("MMLIB50", "serial", "MF"): 2275,
    ("MMLIB50", "serial", "S"): 1691,
    ("MMLIB50", "parallel", "AF"): 655,
    ("MMLIB50", "parallel", "MF"): 778,
    ("MMLIB50", "parallel", "S"): 782,
    ("MMLIB100", "serial", "AF"): 3756,
    ("MMLIB100", "serial", "MF"): 9178,
    ("MMLIB100", "serial", "S"): 7805,
    ("MMLIB100", "parallel", "AF"): 2851,
    ("MMLIB100", "parallel", "MF"): 3388,
    ("MMLIB100", "parallel", "S"): 3246,
    ("MMLIBPLUS_50", "serial", "AF"): 12769,
    ("MMLIBPLUS_50", "serial", "MF"): 67776,
    ("MMLIBPLUS_50", "serial", "S"): 53916,
    ("MMLIBPLUS_50", "parallel", "AF"): 2310,
    ("MMLIBPLUS_50", "parallel", "MF"): 3072,
    ("MMLIBPLUS_50", "parallel", "S"): 3249,
    ("MMLIBPLUS_100", "serial", "AF"): 30753,
    ("MMLIBPLUS_100", "serial", "MF"): 244763,
    ("MMLIBPLUS_100", "serial", "S"): 198455,
    ("MMLIBPLUS_100", "parallel", "AF"): 9350,
    ("MMLIBPLUS_100", "parallel", "MF"): 11976,
    ("MMLIBPLUS_100", "parallel", "S"): 16801,
}

# Rough multipliers on top of TABLE_VII_SECONDS's baseline cost, used only
# to pick an INITIAL walltime/queue guess before any real measurement --
# "nr" only changes the terminal set (negligible extra cost per call),
# "lexicase" adds selection bookkeeping, "local_search"/"hybrid" run CP
# repair on the elite fraction every generation on top of evaluation. Not
# measured per-cell; matrix_runner.py's own --dry_run mode is what actually
# measures real cost before submitting at scale -- see generate_pbs_jobs.py.
CONDITION_COST_MULTIPLIER = {
    "baseline": 1.0,
    "baseline_nr": 1.0,
    "nr": 1.05,
    "lexicase": 1.15,
    "local_search": 1.6,
    "hybrid": 1.75,
}

SAFETY_FACTOR = 1.5  # walltime request = estimated cost x this, for queueing slop
OVEN_QUEUE_THRESHOLD_HOURS = 24  # cells estimated above this go to `oven` (720h), not `default`


def estimate_cell_seconds(condition: str, dataset: str, sgs: str, strategy: str,
                           pop_size: int, n_gen: int) -> float:
    """Rough scaled estimate (seconds) for ONE single-seed run of this cell,
    derived from TABLE_VII_SECONDS' pop=1000/gen=50 baseline number, scaled
    linearly by (pop_size/1000)*(n_gen/50) -- matching how GP evaluation
    cost scales with population x generations -- times a condition
    multiplier. This is a planning number for generate_pbs_jobs.py to pick
    an initial walltime/queue; the trustworthy number is whatever
    matrix_runner.py's own --dry_run mode actually measures for that cell.
    """
    base = TABLE_VII_SECONDS[(dataset, sgs, strategy)]
    scale = (pop_size / PAPER_POP_SIZE) * (n_gen / PAPER_N_GEN)
    return base * scale * CONDITION_COST_MULTIPLIER[condition]


def queue_for_hours(hours: float) -> str:
    return "oven" if hours > OVEN_QUEUE_THRESHOLD_HOURS else "default"


# ---------------------------------------------------------------------------
# Paper reference numbers for the reproduction report (analyze_matrix.py).
# dataset -> strategy -> (manual_rule_name, manual_value, paper_gp_mean,
# paper_gp_std). Transcribed in full directly from the paper text (Tables V
# and VI, Tian, Mei & Zhang, CEC 2024).
#
# The manual-rule reference ISN'T a single fixed heuristic across the whole
# table -- it's the paper's best-performing hand-crafted rule PER dataset,
# which differs (LSTLFT-EFFT for MMLIB50/MMLIB+50 under serial but
# LS-EFFT for MMLIB100/MMLIB+100; LSTLFT-SFM/EFFT for MMLIB50/MMLIB100
# under parallel but LF-SFM/EFFT for MMLIB+50/MMLIB+100), hence the name
# is stored per-cell rather than as one global constant.
# ---------------------------------------------------------------------------
PAPER_TABLE_V_SERIAL = {
    "MMLIB50": {
        "AF": ("LSTLFT-EFFT", 13.80, 12.87, 0.22),
        "MF": ("LSTLFT-EFFT", 13.80, 12.85, 0.26),
        "S": ("LSTLFT-EFFT", 13.80, 13.00, 0.24),
    },
    "MMLIB100": {
        "AF": ("LS-EFFT", 12.25, 11.24, 0.18),
        "MF": ("LS-EFFT", 12.25, 11.32, 0.17),
        "S": ("LS-EFFT", 12.25, 11.56, 0.13),
    },
    "MMLIBPLUS_50": {
        "AF": ("LSTLFT-EFFT", 23.79, 23.50, 0.19),
        "MF": ("LSTLFT-EFFT", 23.79, 23.48, 0.18),
        "S": ("LSTLFT-EFFT", 23.79, 23.08, 0.13),
    },
    "MMLIBPLUS_100": {
        "AF": ("LS-EFFT", 24.73, 24.28, 0.18),
        "MF": ("LS-EFFT", 24.73, 24.33, 0.15),
        "S": ("LS-EFFT", 24.73, 24.15, 0.23),
    },
}

PAPER_TABLE_VI_PARALLEL = {
    "MMLIB50": {
        "AF": ("LSTLFT-SFM/EFFT", 24.77, 22.16, 0.79),
        "MF": ("LSTLFT-SFM/EFFT", 24.77, 23.67, 0.63),
        "S": ("LSTLFT-SFM/EFFT", 24.77, 19.63, 0.76),
    },
    "MMLIB100": {
        "AF": ("LSTLFT-SFM/EFFT", 20.54, 18.58, 0.56),
        "MF": ("LSTLFT-SFM/EFFT", 20.54, 20.01, 0.48),
        "S": ("LSTLFT-SFM/EFFT", 20.54, 16.34, 0.41),
    },
    "MMLIBPLUS_50": {
        "AF": ("LF-SFM/EFFT", 68.88, 66.71, 1.41),
        "MF": ("LF-SFM/EFFT", 68.88, 69.71, 1.42),
        "S": ("LF-SFM/EFFT", 68.88, 55.31, 1.16),
    },
    "MMLIBPLUS_100": {
        "AF": ("LF-SFM/EFFT", 67.13, 61.46, 1.27),
        "MF": ("LF-SFM/EFFT", 67.13, 64.38, 0.80),
        "S": ("LF-SFM/EFFT", 67.13, 47.57, 0.89),
    },
}
