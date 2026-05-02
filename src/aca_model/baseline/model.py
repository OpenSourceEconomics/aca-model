"""Baseline structural retirement model using pylcm.

This is the main model specification, ported from struct-ret/. The baseline
represents pre-ACA rules (no individual mandate, no ACA subsidies).

Usage:
    from aca_model.baseline.model import create_model
    model = create_model()
    params = get_default_params()
    V = model.solve(params)
"""

from collections.abc import Mapping
from typing import Any

from lcm import AgeGrid, DiscreteGrid, Model

from aca_model.baseline.regimes import RegimeId, build_all_regimes
from aca_model.config import GRID_CONFIG, MODEL_CONFIG, GridConfig

_DEFAULT_MAX_CONSUMPTION: float = 300_000.0
"""Upper bound of the consumption grid in $/year. Brackets the unconstrained
CRRA optimum for the highest-asset, highest-income agents in the state space.
Callers can override by passing `max_consumption` in `fixed_params`."""


def create_model(
    *,
    fixed_params: Mapping[str, Any] | None = None,
    wage_params: Mapping[str, Any] | None = None,
    derived_categoricals: Mapping[str, DiscreteGrid | Mapping[str, DiscreteGrid]]
    | None = None,
    grid_config: GridConfig = GRID_CONFIG,
    pref_type_grid: DiscreteGrid | None = None,
    n_subjects: int | None = None,
) -> Model:
    """Create the baseline structural retirement model.

    Args:
        fixed_params: Parameters to fix at model creation time. These are
            partialled into compiled functions and removed from the params
            template. Pass data-derived constants here; only estimation
            parameters should go through `model.simulate(params=...)`.
        wage_params: Data-derived wage profile dict (`log_ft_wage_mean`,
            `log_ft_wage_std`, `adj_wage_hours_*`) used only at grid-build
            time to size the assets-floor to `-max_annual_labor_income`.
            Not routed to the pylcm Model.
        derived_categoricals: Extra categorical mappings for derived variables
            not in the model's state/action grids. Needed when `fixed_params`
            contains `pd.Series` indexed by DAG function outputs.
        grid_config: Continuous-grid point counts. Defaults to production
            values; pass `BENCHMARK_GRID_CONFIG` for a fast-but-structurally-
            faithful benchmark.
        pref_type_grid: Optional override for the `pref_type` `DiscreteGrid`.
            Defaults to `DiscreteGrid(PrefType)`. Used by the benchmark to
            substitute a 2-type variant with `DispatchStrategy.PARTITION_SCAN`.

    Returns:
        A pylcm Model with 19 regimes (18 non-terminal + dead) spanning
        ages 51-95. Regime names follow the `<his>_<medicare>_<ss>_<work>` scheme.

    """
    ages = AgeGrid(
        start=MODEL_CONFIG.start_age,
        stop=MODEL_CONFIG.end_age - 1,
        step="Y",
    )
    fixed_params = _with_max_consumption_default(fixed_params)
    regimes = build_all_regimes(
        grid_config,
        fixed_params=fixed_params,
        wage_params=wage_params,
        pref_type_grid=pref_type_grid,
    )

    return Model(
        regimes=regimes,
        ages=ages,
        regime_id_class=RegimeId,
        description="Baseline structural retirement model (pre-ACA)",
        fixed_params=fixed_params,
        derived_categoricals=derived_categoricals,
        n_subjects=n_subjects,
    )


def _with_max_consumption_default(
    fixed_params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a copy of `fixed_params` with `max_consumption` defaulted."""
    out = dict(fixed_params or {})
    out.setdefault("max_consumption", _DEFAULT_MAX_CONSUMPTION)
    return out
