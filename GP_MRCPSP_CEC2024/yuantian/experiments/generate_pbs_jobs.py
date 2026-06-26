"""
Emit cost-tiered PBS array job scripts (MetaCentrum) for a chosen
matrix_config preset.

    python -m yuantian.experiments.generate_pbs_jobs --preset paper_full --out_dir pbs_jobs/

Why cost-tiered, not one script: matrix_config.TABLE_VII_SECONDS shows a
~100x cost range across cells (a parallel-MMLIB50 single run is minutes; a
serial-MMLIB+100-MF/S single run is tens of hours). A single PBS array
needs ONE walltime directive shared by every element in it -- bundling the
whole range into one array means either every cheap element waits behind a
720h walltime request (terrible for scheduling priority) or the expensive
elements get killed by a walltime too short for them. So this script:

  1. Computes every (condition, strategy, sgs, dataset, seed) cell's
     estimated cost via matrix_config.estimate_cell_seconds (the Table-VII-
     based pre-submission guess, NOT a measurement -- see matrix_runner.py's
     --dry_run for the real, measured number to sanity-check this against
     before actually submitting anything).
  2. Applies SAFETY_FACTOR and buckets each cell into a walltime tier.
  3. Routes tiers <=24h to the `default` queue, tiers >24h to `oven` (the
     720h-walltime queue).
  4. Emits one .pbs script + one .manifest.csv per (queue, walltime bucket)
     combination that actually has cells in it -- so a script's array size
     exactly matches its cell count, no wasted array slots.

Each PBS script reads its own manifest with $PBS_ARRAY_INDEX as the line
number (1-based, matching PBS's own 1-based array indexing), follows the
scratch copy / run / copy-result-back / clean_scratch pattern, and checks
for the result file at the PERSISTENT output location before doing any
scratch setup at all -- so a resubmitted array (e.g. after a partial
failure) skips already-finished cells without even copying data to scratch
for them, not just without rerunning them.

Storage paths (PROJECT_DIR, VENV_PATH, PERSISTENT_OUT) are placeholders at
the top of each generated script -- they depend on your own MetaCentrum
account's storage layout, so they need to be filled in once per cluster
account, not per script. See the module-level SETUP NOTE below for what
each one is and how to find it on your account.
"""

# ---------------------------------------------------------------------------
# SETUP NOTE -- what PROJECT_DIR / VENV_PATH / PERSISTENT_OUT actually are,
# and how to find them (one-time, per MetaCentrum account):
#
# PROJECT_DIR: wherever you put a copy of this repo on MetaCentrum's
#   storage (NOT your Mac's path -- nodes don't see your Mac's filesystem).
#   Log in (ssh username@frontend.metacentrum.cz or similar), `git clone`
#   this repo into your home/storage area there, then PROJECT_DIR is just
#   that path, e.g. /storage/praha1/home/USERNAME/master-thesis/GP_MRCPSP_CEC2024.
#   `pwd` after cd-ing into it on the cluster gives you the exact string.
#
# VENV_PATH: a Python virtualenv created FRESH on MetaCentrum, not copied
#   from your Mac (different OS/architecture, won't work). On a frontend
#   node: load a Python module (e.g. `module add python/3.11...` -- run
#   `module avail python` to see what's offered), then
#   `python3 -m venv /storage/.../master-thesis/.venv` and
#   `pip install -r requirements.txt` inside it. VENV_PATH is that venv's
#   directory (the one containing bin/activate).
#
# PERSISTENT_OUT: any directory under your own storage quota where you want
#   result JSONs kept permanently. There's no "correct" value -- scratch
#   storage gets wiped after each job, so this just needs to be a real,
#   writable path you control and intend to keep checking back on, e.g.
#   PROJECT_DIR + "/yuantian/experiments/results/matrix".
#
# None of this can be filled in from outside your account -- MetaCentrum
# storage allocations are per-user/per-group, not something inferable from
# this codebase.
# ---------------------------------------------------------------------------
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yuantian.experiments.matrix_config import (
    OVEN_QUEUE_THRESHOLD_HOURS,
    PRESETS,
    SAFETY_FACTOR,
    estimate_cell_seconds,
    matrix_cells,
    n_seeds_for,
)

# walltime tiers (hours) -- a cell goes into the smallest tier >= its own
# safety-adjusted estimate. default queue only ever uses the <=24h tiers;
# oven only the >24h ones (matches OVEN_QUEUE_THRESHOLD_HOURS=24).
DEFAULT_QUEUE_TIERS = [2, 6, 24]
OVEN_QUEUE_TIERS = [48, 168, 720]  # 2 days, 1 week, the 720h max

# Placeholders -- fill in once per MetaCentrum account, not per script. No
# DATA_HOME here on purpose: that env var is only read by
# rcpsp_dataset.py's get_instance_list_from_txt, which this pipeline never
# calls. The dataset paths actually used (RCPSPDatabase.MMLIB_50_DIR etc.)
# are plain relative paths resolved against the working directory, and the
# script already cd's into the scratch-copied repo (which includes
# discrete_optimization_data/) before running -- nothing extra to set.
PROJECT_DIR_PLACEHOLDER = "/storage/REPLACE_ME/master-thesis/GP_MRCPSP_CEC2024"
VENV_PLACEHOLDER = "/storage/REPLACE_ME/master-thesis/.env"
PERSISTENT_OUT_PLACEHOLDER = "/storage/REPLACE_ME/master-thesis/GP_MRCPSP_CEC2024/yuantian/experiments/results/matrix"


def bucket_for(safe_hours: float):
    """(queue, walltime_hours) for a safety-adjusted hour estimate."""
    if safe_hours <= OVEN_QUEUE_THRESHOLD_HOURS:
        for tier in DEFAULT_QUEUE_TIERS:
            if safe_hours <= tier:
                return "default", tier
        return "default", DEFAULT_QUEUE_TIERS[-1]
    for tier in OVEN_QUEUE_TIERS:
        if safe_hours <= tier:
            return "oven", tier
    return "oven", OVEN_QUEUE_TIERS[-1]


def build_manifest(preset: dict):
    """List of (condition, strategy, sgs, dataset, seed, queue, walltime_h,
    estimate_h) for every cell x seed in preset, one entry per array element."""
    rows = []
    for condition, strategy, sgs, dataset in matrix_cells(preset):
        estimate_s = estimate_cell_seconds(
            condition, dataset, sgs, strategy, preset["pop_size"], preset["n_gen"]
        )
        safe_hours = (estimate_s * SAFETY_FACTOR) / 3600
        queue, walltime_h = bucket_for(safe_hours)
        for i in range(n_seeds_for(preset, sgs, dataset, strategy)):
            seed = preset["seed_base"] + i
            rows.append(dict(
                condition=condition, strategy=strategy, sgs=sgs, dataset=dataset,
                seed=seed, pop=preset["pop_size"], gen=preset["n_gen"],
                n_classes=preset["n_classes"], queue=queue, walltime_h=walltime_h,
                estimate_h=estimate_s / 3600,
            ))
    return rows


PBS_TEMPLATE = """#!/bin/bash
#PBS -N matrix_{queue}_{walltime_h}h
#PBS -l walltime={walltime_h}:00:00
#PBS -l select=1:ncpus=1:mem=4gb:scratch_local=4gb
#PBS -q {queue}
#PBS -j oe

# --- fill these in once per MetaCentrum account, not per script ---
PROJECT_DIR="{project_dir}"
VENV_PATH="{venv_path}"
PERSISTENT_OUT="{persistent_out}"
MANIFEST="$PROJECT_DIR/yuantian/experiments/pbs_jobs/{manifest_name}"
# --------------------------------------------------------------------

set -e
trap 'clean_scratch' TERM EXIT

LINE=$(sed -n "${{PBS_ARRAY_INDEX}}p" "$MANIFEST")
IFS=',' read -r CONDITION STRATEGY SGS DATASET SEED POP GEN NCLASSES <<< "$LINE"
RESULT_NAME="${{DATASET}}__${{SGS}}__${{STRATEGY}}__${{CONDITION}}__seed${{SEED}}.json"

# idempotent: bail before touching scratch at all if the persistent result
# already exists -- a resubmitted/relaunched array element shouldn't redo
# finished cells, and shouldn't even pay for the scratch copy to find that out.
if [ -f "$PERSISTENT_OUT/$RESULT_NAME" ]; then
    echo "SKIP (already exists): $PERSISTENT_OUT/$RESULT_NAME"
    exit 0
fi

cp -r "$PROJECT_DIR" "$SCRATCHDIR/repo"
cd "$SCRATCHDIR/repo"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$SCRATCHDIR/repo:$SCRATCHDIR/repo/yuantian:$PYTHONPATH"
mkdir -p "$SCRATCHDIR/results"

python3 -O -m yuantian.experiments.matrix_runner \\
    --condition "$CONDITION" --strategy "$STRATEGY" --sgs "$SGS" --dataset "$DATASET" \\
    --pop "$POP" --gen "$GEN" --seed "$SEED" --n_classes "$NCLASSES" \\
    --out "$SCRATCHDIR/results"

mkdir -p "$PERSISTENT_OUT"
cp "$SCRATCHDIR/results/"*.json "$PERSISTENT_OUT/" 2>/dev/null || true
"""


def write_tier_files(queue: str, walltime_h: int, rows: list, out_dir: Path):
    manifest_name = f"matrix_{queue}_{walltime_h}h.manifest.csv"
    pbs_name = f"matrix_{queue}_{walltime_h}h.pbs"

    with open(out_dir / manifest_name, "w", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([r["condition"], r["strategy"], r["sgs"], r["dataset"],
                        r["seed"], r["pop"], r["gen"], r["n_classes"]])

    script = PBS_TEMPLATE.format(
        queue=queue, walltime_h=walltime_h, manifest_name=manifest_name,
        project_dir=PROJECT_DIR_PLACEHOLDER, venv_path=VENV_PLACEHOLDER,
        persistent_out=PERSISTENT_OUT_PLACEHOLDER,
    )
    with open(out_dir / pbs_name, "w") as f:
        f.write(script)
    return pbs_name, manifest_name, len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", required=True, choices=list(PRESETS))
    p.add_argument("--out_dir", type=str, default="yuantian/experiments/pbs_jobs")
    args = p.parse_args()

    preset = PRESETS[args.preset]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear previously generated tier files first -- otherwise a tier that
    # no longer has any cells (e.g. after a CELL_EXCLUSIONS change) leaves
    # its stale .pbs/.manifest.csv sitting in out_dir from the last run,
    # which someone could submit by mistake. Scoped to this generator's own
    # naming pattern, not a blind directory wipe.
    for stale in out_dir.glob("matrix_*.pbs"):
        stale.unlink()
    for stale in out_dir.glob("matrix_*.manifest.csv"):
        stale.unlink()

    rows = build_manifest(preset)
    by_tier = {}
    for r in rows:
        by_tier.setdefault((r["queue"], r["walltime_h"]), []).append(r)

    print(f"Preset '{args.preset}': {len(rows)} total cell x seed runs, {len(by_tier)} cost tiers.\n")
    total_core_hours = 0.0
    for (queue, walltime_h), tier_rows in sorted(by_tier.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        pbs_name, manifest_name, n = write_tier_files(queue, walltime_h, tier_rows, out_dir)
        tier_core_hours = sum(r["estimate_h"] for r in tier_rows)  # ncpus=1, so core-hours == wallclock-hours of actual estimated work
        total_core_hours += tier_core_hours
        print(
            f"  queue={queue:<8} walltime={walltime_h:>3}h  n_elements={n:<5} "
            f"est_core_hours={tier_core_hours:>9.1f}  -> {pbs_name} / {manifest_name}"
        )
    print(f"\nTOTAL estimated core-hours across all tiers: {total_core_hours:.1f} "
          f"(~{total_core_hours / 24:.1f} core-days)")
    print(
        "\nFill in PROJECT_DIR / VENV_PATH / PERSISTENT_OUT at the top of "
        "each .pbs script before submitting (qsub -J 1-N script.pbs, N = n_elements above)."
    )


if __name__ == "__main__":
    main()
