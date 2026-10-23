"""Baseline structural retirement model using pylcm.

This is the main model specification, ported from struct-ret/. The baseline
represents pre-ACA rules (no individual mandate, no ACA subsidies).

Usage:
    from aca_model.baseline.model import create_model
    model = create_model(fixed_params=..., wage_params=..., ...)
    params = get_default_params()
    V = model.solve(params)
"""

from collections.abc import Mapping
from typing import Any

from lcm import AgeGrid, DiscreteGrid, ExecutionConfig, InitialNodes, Model
from lcm.typing import UserParams

from aca_model.baseline.regimes import (
    RegimeId,
    SolverName,
    build_all_regimes,
    build_entry_initial_nodes,
    build_model_slots,
)
from aca_model.config import MODEL_CONFIG, GridConfig, model_ages
from aca_model.execution import execution_config_for_devices


def create_model(
    *,
    fixed_params: UserParams,
    wage_params: Mapping[str, Any],
    derived_categoricals: Mapping[str, DiscreteGrid],
    grid_config: GridConfig,
    pref_type_grid: DiscreteGrid,
    solver: SolverName = "brute_force",
    consumption_dollars_points: tuple[float, ...] | None = None,
    execution_config: ExecutionConfig | None = None,
    initial_nodes: InitialNodes | None = None,
    final_age: int = MODEL_CONFIG.end_age,
) -> Model:
    """Create the baseline structural retirement model.

    Args:
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
        execution_config: Explicit hardware-local policy forwarded unchanged.
            None uses the smallest selected accelerator allocator limit as the
            device-memory budget; CPU construction remains unbudgeted.
        initial_nodes: Admissible starting `(age, regime)` pairs. None admits
            the estimation sample's entry ages and regimes that lie before
            `final_age`; pass `build_restart_initial_nodes` to restart
            simulations mid-life.
        final_age: First age at which everybody is dead; the model spans
            `MODEL_CONFIG.start_age` through `final_age`. The default is the
            full horizon. A shorter horizon reads a prefix of the age-indexed
            data, and `fixed_params` must hold only what it reads (see
            `select_horizon_params`).

    Returns:
        A pylcm Model with 19 regimes (18 non-terminal + dead) spanning
        ages 51 through `final_age` (96 by default); `final_age - 1` is the
        last age anyone is alive. Regime names follow
        the `<his>_<medicare>_<ss>_<work>` scheme.

    """
    ages = model_ages(final_age=final_age)
    _fail_if_dcegm_without_consumption_points(
        solver=solver, consumption_dollars_points=consumption_dollars_points
    )
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
        ages=ages,
        solver=solver,
    )

    return Model(
        regimes=regimes,
        initial_nodes=(
            build_entry_initial_nodes(ages=ages)
            if initial_nodes is None
            else initial_nodes
        ),
        ages=AgeGrid(exact_values=ages),
        regime_id_class=RegimeId,
        description="Baseline structural retirement model (pre-ACA)",
        fixed_params=fixed_params,
        derived_categoricals=derived_categoricals,
        execution_config=(
            execution_config_for_devices()
            if execution_config is None
            else execution_config
        ),
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
