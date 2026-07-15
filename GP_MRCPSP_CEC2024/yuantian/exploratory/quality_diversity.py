"""
MAP-Elites / Quality-Diversity (exploratory strategy "map_elites";
see yuantian/exploratory/README.md).

Adapted-method baseline (Mouret & Clune, 2015): illuminates a CP-reliance x
NR-reliance behaviour grid rather than optimising mean fitness alone.

Descriptor adaptation note: the original grid used the modification-era
terminal names {"Is_On_Critical_Path", "Slack", "Dynamic_Slack", "CP_Ext"}
(CP axis) and {"NR_Stock_Ratio", "NR_Mode_Demand_Ratio"} (NR axis), which
lived in the now-also-deleted yuantian/modifications.py. This port targets
the current repo's analogous terminal sets instead: CP_FORWARD/CP_BACKWARD/
CP_SLACK_SCORE/CP_PROB (yuantian/cp_propagation.py, enabled via
--cp_propagation) for the CP axis, and NR_STOCK_RATIO/NR_MODE_DEMAND_RATIO/
NR_BUDGET_PRESSURE (yuantian/nr_terminals.py, enabled via --nr_terminals)
for the NR axis. The grid is only non-degenerate (every individual landing
in the same cell otherwise) when the GPHH instance passed to this driver
was built with both flags on.
"""
import random
import time
from typing import Optional

from deap import tools

from yuantian.exploratory.shared import (
    DataProvider,
    Toolbox,
    _eval_full,
    _new_logbook,
    _record,
    _terminal_reliance,
    _Timer,
)
from yuantian.gp_algorithms import varOr

# See module docstring: current repo's closest analogue to the deleted
# modifications.py terminal names used for the behaviour descriptor.
CP_DESCRIPTOR_NAMES = {"CP_FORWARD", "CP_BACKWARD", "CP_SLACK_SCORE", "CP_PROB"}
NR_DESCRIPTOR_NAMES = {"NR_STOCK_RATIO", "NR_MODE_DEMAND_RATIO", "NR_BUDGET_PRESSURE"}


def map_elites_gp(
    population: list,
    toolbox: Toolbox,
    cxpb: float,
    mutpb: float,
    n_elite: int,
    ngen: int,
    training_data_provider: DataProvider,
    validation_data_provider: DataProvider,
    stats: Optional[tools.Statistics] = None,
    halloffame: Optional[tools.HallOfFame] = None,
    pop_archive: Optional[list] = None,
    verbose: bool = __debug__,
    grid: int = 8,
) -> tuple:
    """MAP-Elites over a CP-reliance x NR-reliance (genotype) behaviour grid."""
    logbook = _new_logbook(stats)
    timer = _Timer(ngen)
    pop_size = len(population)

    def cell(ind):
        cp = min(grid - 1, int(_terminal_reliance(ind, CP_DESCRIPTOR_NAMES) * grid))
        nr = min(grid - 1, int(_terminal_reliance(ind, NR_DESCRIPTOR_NAMES) * grid))
        return cp, nr

    archive: dict = {}

    def deposit(ind):
        key = cell(ind)
        cur = archive.get(key)
        if cur is None or ind.fitness.values[0] < cur.fitness.values[0]:
            archive[key] = ind

    t0 = time.time()
    training = training_data_provider.next()
    _eval_full(population, training, toolbox)
    for ind in population:
        deposit(ind)
    _record(
        0,
        len(population),
        list(archive.values()),
        toolbox,
        halloffame,
        stats,
        logbook,
        validation_data_provider,
        pop_archive,
        timer,
        t0,
        extra=f"  filled={len(archive)}/{grid * grid}",
    )

    for gen in range(1, ngen + 1):
        t0 = time.time()
        elites = list(archive.values())
        parents = [toolbox.clone(random.choice(elites)) for _ in range(pop_size)]
        offspring = varOr(parents, toolbox, cxpb, mutpb)
        training = training_data_provider.next()
        _eval_full(offspring, training, toolbox)
        for child in offspring:
            deposit(child)
        _record(
            gen,
            len(offspring),
            list(archive.values()),
            toolbox,
            halloffame,
            stats,
            logbook,
            validation_data_provider,
            pop_archive,
            timer,
            t0,
            extra=f"  filled={len(archive)}/{grid * grid}",
        )

    return list(archive.values()), logbook
