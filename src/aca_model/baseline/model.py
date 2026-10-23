"""Structural retirement model specification and construction using pylcm.

The baseline represents pre-ACA rules (no individual mandate, no ACA
subsidies); an ACA `PolicyVariant` overlays its schedules on the baseline
regimes.

Usage:
    from aca_model.baseline.model import build_model_spec, create_model
    spec = build_model_spec(grid_config=..., pia_aime_grid=..., ...)
    model = create_model(spec=spec, fixed_params=...)
"""

from collections.abc import Mapping
from typing import Any

from jax.typing import ArrayLike
from lcm import AgeGrid, DiscreteGrid, Model
from lcm.typing import UserParams

from aca_model.aca.health_insurance import PolicyVariant
from aca_model.aca.regimes import build_all_regimes as build_all_aca_regimes
from aca_model.agent.health import GoodHealth
from aca_model.agent.labor_market import IsMarried
from aca_model.baseline.health_insurance import HealthInsuranceState
from aca_model.baseline.regimes import (
    RegimeId,
    SolverName,
    build_all_regimes,
    build_entry_initial_nodes,
    build_model_slots,
)
from aca_model.config import MODEL_CONFIG, GridConfig, model_ages


def build_model_spec(
    *,
    policy: PolicyVariant | None = None,
    grid_config: GridConfig,
    pia_aime_grid: ArrayLike,
    wage_params: Mapping[str, Any],
    pref_type_grid: DiscreteGrid,
    solver: SolverName = "brute_force",
    consumption_dollars_points: tuple[float, ...] | None = None,
    final_age: int = MODEL_CONFIG.end_age,
) -> dict[str, Any]:
    """Return the domain keyword arguments of `lcm.Model` for the retirement model.

    The spec holds `regimes`, `ages`, `regime_id_class`, `description`,
    `derived_categoricals`, the entry `initial_nodes`, `edges`, `functions`,
    `constraints`, `states` and `state_transitions`. It holds no generic
    `Model` option; pass those to `create_model`. Merge an override with `|`,
    e.g. `spec | {"initial_nodes": build_restart_initial_nodes(ages=...)}`.

    Args:
        policy: ACA policy whose schedules overlay the baseline regimes. None
            builds the baseline (pre-ACA) model.
        grid_config: Continuous-grid point counts. Pass `GRID_CONFIG` for
            production values or `BENCHMARK_GRID_CONFIG` for the
            fast-but-structurally-faithful benchmark.
        pia_aime_grid: PIA bend points in AIME, which break the AIME grid
            into its segments.
        wage_params: Data-derived wage profile dict (`log_ft_wage_mean`,
            `log_ft_wage_std`, `adj_wage_hours_*`) used only at grid-build
            time to size the assets-floor to `-max_annual_labor_income`.
        pref_type_grid: Pref-type `DiscreteGrid`. Pass
            `DiscreteGrid(PrefType)` for the production 3-type layout,
            or a compact variant (e.g. `DiscreteGrid(BenchmarkPrefType)`).
        solver: `"brute_force"` (the default), `"dcegm"` or `"nbegm"`. DC-EGM
            swaps the spec to its post-decision form: savings-form assets laws,
            broadcast `resources`/`savings`/`inverse_marginal_utility`
            functions, and a `DCEGM` solver config on every living regime.
        consumption_dollars_points: Construction-time consumption action
            gridpoints. Required under DC-EGM (the kernel needs the
            continuous-action grid at model construction); `None` keeps
            the runtime-points grid completed per iteration via
            `inject_consumption_dollars_points`.
        final_age: First age at which everybody is dead; the model spans
            `MODEL_CONFIG.start_age` through `final_age`. The default is the
            full horizon. A shorter horizon reads a prefix of the age-indexed
            data, and its fixed params must hold only what it reads (see
            `select_horizon_params`).

    Returns:
        Keyword arguments for `lcm.Model` describing 19 regimes (18
        non-terminal + dead) over ages 51 through `final_age`; `final_age - 1`
        is the last age anyone is alive. Regime names follow the
        `<his>_<medicare>_<ss>_<work>` scheme.

    """
    _fail_if_dcegm_without_consumption_points(
        solver=solver, consumption_dollars_points=consumption_dollars_points
    )
    ages = model_ages(final_age=final_age)
    regimes = (
        build_all_regimes(
            grid_config=grid_config,
            pia_aime_grid=pia_aime_grid,
            wage_params=wage_params,
            pref_type_grid=pref_type_grid,
            solver=solver,
            consumption_dollars_points=consumption_dollars_points,
        )
        if policy is None
        else build_all_aca_regimes(
            policy=policy,
            grid_config=grid_config,
            pia_aime_grid=pia_aime_grid,
            wage_params=wage_params,
            pref_type_grid=pref_type_grid,
            solver=solver,
            consumption_dollars_points=consumption_dollars_points,
        )
    )
    # The ACA overlay swaps only regime-level functions; the broadcast slots
    # are policy-invariant.
    model_slots = build_model_slots(
        grid_config=grid_config,
        pia_aime_grid=pia_aime_grid,
        wage_params=wage_params,
        pref_type_grid=pref_type_grid,
        ages=ages,
        solver=solver,
    )
    return {
        "regimes": regimes,
        "ages": AgeGrid(exact_values=ages),
        "regime_id_class": RegimeId,
        "description": (
            "Baseline structural retirement model (pre-ACA)"
            if policy is None
            else f"Structural retirement model ({policy.name})"
        ),
        "derived_categoricals": build_derived_categoricals(
            pref_type_grid=pref_type_grid
        ),
        "initial_nodes": build_entry_initial_nodes(ages=ages),
        **model_slots,
    }


def create_model(
    *, spec: Mapping[str, Any], fixed_params: UserParams, **model_options: Any
) -> Model:
    """Build the pylcm Model from `spec`, `fixed_params` and generic options.

    Args:
        spec: Domain keyword arguments from `build_model_spec`.
        fixed_params: Parameters to fix at model creation time. Fixed params
            are partialled into compiled functions and removed from the params
            template. They must hold exactly what the spec's horizon reads.
        **model_options: Any other `lcm.Model` keyword argument, forwarded
            unchanged, e.g. `execution_config`. A key the spec also holds,
            such as `initial_nodes`, overrides the spec's value.

    Returns:
        The pylcm Model.

    """
    return Model(**{**spec, **model_options}, fixed_params=fixed_params)


def build_derived_categoricals(
    *, pref_type_grid: DiscreteGrid
) -> dict[str, DiscreteGrid]:
    """Return the labels of params index levels that are not grids of every regime.

    `pd.Series` params are indexed by `target_his`, `his`, `good_health`,
    `is_married` and `pref_type`.
    """
    return {
        "good_health": DiscreteGrid(GoodHealth),
        "is_married": DiscreteGrid(IsMarried),
        "his": DiscreteGrid(HealthInsuranceState),
        "target_his": DiscreteGrid(HealthInsuranceState),
        "pref_type": pref_type_grid,
    }


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
