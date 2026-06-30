"""Baseline regime construction package.

Builds the 18-regime model decomposition programmatically from four
structural dimensions: HIS x Medicare x SS x Work.

Pre-65 regimes (mc=nomc/dimc) use HealthWithDisability (3-state).
Post-65 regimes (mc=oamc) use Health (2-state).

Each HIS type (retiree, tied, nongroup) has its own transition logic in a
dedicated submodule. Shared definitions and builders live in _common.
"""

from collections.abc import Mapping
from typing import Any

from lcm import DiscreteGrid, Regime
from lcm.solvers import BQSEGM, DCEGM
from lcm.typing import UserParams

from aca_model.baseline.regimes import _nongroup as nongroup
from aca_model.baseline.regimes import _retiree as retiree
from aca_model.baseline.regimes import _tied as tied
from aca_model.baseline.regimes._bqsegm import build_bqsegm_solver
from aca_model.baseline.regimes._common import (
    REGIME_SPECS,
    Grids,
    RegimeId,
    SolverName,
    build_dead_regime,
    build_grids,
    build_model_constraints,
    build_model_functions,
    build_model_state_transitions,
    build_model_states,
)
from aca_model.baseline.regimes._dcegm import build_dcegm_solver
from aca_model.config import GridConfig

# BQSEGM is a per-regime (not global) solver: it solves one 1-D
# consumption/savings regime with at most one discrete action, so it attaches
# only to the M1 vertical-slice regime.
_BQSEGM_REGIME = "nongroup_nomc_inelig_canwork"

__all__ = [
    "REGIME_SPECS",
    "RegimeId",
    "build_all_regimes",
    "build_model_slots",
    "build_regime",
    "nongroup",
    "retiree",
    "tied",
]

_HIS_BUILDERS = {
    "retiree": retiree.build_regime,
    "tied": tied.build_regime,
    "nongroup": nongroup.build_regime,
}


def build_regime(
    name: str,
    grids: Grids,
    *,
    dcegm_solver: DCEGM | None = None,
    bqsegm_solver: BQSEGM | None = None,
) -> Regime:
    """Build a single baseline Regime object for the given regime name."""
    if name == "dead":
        return build_dead_regime(
            solver="brute_force" if dcegm_solver is None else "dcegm"
        )

    spec = REGIME_SPECS[name]
    builder = _HIS_BUILDERS.get(spec["his"])
    if builder is None:
        msg = f"Unknown HIS type: {spec['his']}"
        raise ValueError(msg)
    return builder(name, grids, dcegm_solver=dcegm_solver, bqsegm_solver=bqsegm_solver)


def build_all_regimes(
    *,
    grid_config: GridConfig,
    fixed_params: UserParams,
    wage_params: Mapping[str, Any],
    pref_type_grid: DiscreteGrid,
    solver: SolverName = "brute_force",
    consumption_dollars_points: tuple[float, ...] | None = None,
) -> dict[str, Regime]:
    """Build all 19 baseline regimes (18 non-terminal + dead).

    `fixed_params` carries the PIA bends for the AIME piecewise grid;
    `wage_params` sizes the assets-floor to `-max_annual_labor_income`;
    `pref_type_grid` selects the pref-type cardinality (production
    `DiscreteGrid(PrefType)` or the benchmark's 2-type variant);
    `solver` selects brute force or DC-EGM for every living regime
    (`dead` is terminal and keeps the default);
    `consumption_dollars_points` fixes the consumption action grid at
    construction (required under DC-EGM).
    """
    grids = build_grids(
        grid_config=grid_config,
        fixed_params=fixed_params,
        wage_params=wage_params,
        pref_type_grid=pref_type_grid,
        consumption_dollars_points=consumption_dollars_points,
    )
    dcegm_solver = build_dcegm_solver(grids) if solver == "dcegm" else None
    bqsegm_solver = build_bqsegm_solver(grids) if solver == "bqsegm" else None
    regimes = {}
    for name in REGIME_SPECS:
        regimes[name] = build_regime(
            name,
            grids,
            dcegm_solver=dcegm_solver,
            bqsegm_solver=bqsegm_solver if name == _BQSEGM_REGIME else None,
        )
    regimes["dead"] = build_dead_regime(solver=solver)
    return regimes


def build_model_slots(
    *,
    grid_config: GridConfig,
    fixed_params: UserParams,
    wage_params: Mapping[str, Any],
    pref_type_grid: DiscreteGrid,
    solver: SolverName = "brute_force",
) -> dict[str, Any]:
    """Build the model-level regime slots broadcast into every regime.

    Returns keyword arguments for `lcm.Model(...)`: the functions,
    constraint, states, and laws of motion shared by all living regimes.
    Both the baseline and the ACA `create_model` consume this — the ACA
    overlay swaps only regime-level functions, so the broadcast slots are
    policy-invariant. Under DC-EGM the solver-contract functions join the
    broadcast set and no borrowing constraint is declared.
    """
    grids = build_grids(
        grid_config=grid_config,
        fixed_params=fixed_params,
        wage_params=wage_params,
        pref_type_grid=pref_type_grid,
    )
    return {
        "functions": build_model_functions(solver=solver),
        "constraints": build_model_constraints(solver=solver),
        "states": build_model_states(grids),
        "state_transitions": build_model_state_transitions(),
    }
