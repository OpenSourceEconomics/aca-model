"""Baseline structural retirement model using pylcm.

This is the main model specification, ported from struct-ret/. The baseline
represents pre-ACA rules (no individual mandate, no ACA subsidies).

Usage:
    from aca_model.baseline.model import create_model
    model = create_model(n_subjects=..., fixed_params=..., wage_params=..., ...)
    params = get_default_params()
    V = model.solve(params)
"""

from collections.abc import Mapping
from typing import Any

from lcm import AgeGrid, DiscreteGrid, Model
from lcm.typing import UserParams

from aca_model.baseline.regimes import (
    RegimeId,
    SolverName,
    build_all_regimes,
    build_model_slots,
)
from aca_model.config import MODEL_CONFIG, GridConfig
from aca_model.environment.pensions import with_nongroup_imputation_slices


def create_model(
    *,
    n_subjects: int,
    fixed_params: UserParams,
    wage_params: Mapping[str, Any],
    derived_categoricals: Mapping[str, DiscreteGrid],
    grid_config: GridConfig,
    pref_type_grid: DiscreteGrid,
    solver: SolverName = "brute_force",
    consumption_dollars_points: tuple[float, ...] | None = None,
) -> Model:
    """Create the baseline structural retirement model.

    Args:
        n_subjects: Forwarded to `lcm.Model(n_subjects=...)`.
        fixed_params: Parameters to fix at model creation time. Fixed
            params are partialled into compiled functions and removed
            from the params template. Pass data-derived constants here;
            only estimation parameters should go through
            `model.simulate(params=...)`.
        wage_params: Data-derived wage profile dict (`log_ft_wage_mean`,
            `log_ft_wage_std`, `adj_wage_hours_*`) used only at grid-build
            time to size the assets-floor to `-max_annual_labor_income`.
            Not routed to the pylcm Model.
        derived_categoricals: Categorical mappings for `pd.Series`
            fixed_params index levels that aren't model state/action
            grids — `target_his`, `his`, `good_health`, `is_married`,
            `pref_type`.
        grid_config: Continuous-grid point counts. Pass `GRID_CONFIG` for
            production values or `BENCHMARK_GRID_CONFIG` for the
            fast-but-structurally-faithful benchmark.
        pref_type_grid: Pref-type `DiscreteGrid`. Pass
            `DiscreteGrid(PrefType)` for the production 3-type layout,
            or a compact variant (e.g. `DiscreteGrid(BenchmarkPrefType)`).
        solver: `"brute_force"` (the default) or `"dcegm"`. DC-EGM swaps
            the spec to its post-decision form: savings-form assets laws,
            broadcast `resources`/`savings`/`inverse_marginal_utility`
            functions, no borrowing constraint, and a `DCEGM` solver
            config on every living regime.
        consumption_dollars_points: Construction-time consumption action
            gridpoints. Required under DC-EGM (the kernel needs the
            continuous-action grid at model construction); `None` keeps
            the runtime-points grid completed per iteration via
            `inject_consumption_dollars_points`.

    Returns:
        A pylcm Model with 19 regimes (18 non-terminal + dead) spanning
        ages 51-95. Regime names follow the `<his>_<medicare>_<ss>_<work>` scheme.

    """
    ages = AgeGrid(
        start=MODEL_CONFIG.start_age,
        stop=MODEL_CONFIG.end_age - 1,
        step="Y",
    )
    _fail_if_dcegm_without_consumption_points(
        solver=solver, consumption_dollars_points=consumption_dollars_points
    )
    fixed_params = with_nongroup_imputation_slices(fixed_params)
    regimes = build_all_regimes(
        grid_config=grid_config,
        fixed_params=fixed_params,
        wage_params=wage_params,
        pref_type_grid=pref_type_grid,
        solver=solver,
        consumption_dollars_points=consumption_dollars_points,
    )
    model_slots = build_model_slots(
        grid_config=grid_config,
        fixed_params=fixed_params,
        wage_params=wage_params,
        pref_type_grid=pref_type_grid,
        solver=solver,
    )

    return Model(
        regimes=regimes,
        ages=ages,
        regime_id_class=RegimeId,
        description="Baseline structural retirement model (pre-ACA)",
        fixed_params=fixed_params,
        derived_categoricals=derived_categoricals,
        n_subjects=n_subjects,
        **model_slots,
    )


def _fail_if_dcegm_without_consumption_points(
    *,
    solver: SolverName,
    consumption_dollars_points: tuple[float, ...] | None,
) -> None:
    if solver == "dcegm" and consumption_dollars_points is None:
        msg = (
            "solver='dcegm' requires `consumption_dollars_points`: the "
            "DC-EGM kernel needs the continuous-action grid at model "
            "construction, so the runtime params injection "
            "(`inject_consumption_dollars_points`) cannot supply it. "
            "Compute the points with `compute_consumption_dollars_points` "
            "(or `get_benchmark_consumption_dollars_points` for the frozen "
            "benchmark snapshot)."
        )
        raise ValueError(msg)
