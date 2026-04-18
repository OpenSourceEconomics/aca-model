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


def create_model(
    *,
    fixed_params: Mapping[str, Any] | None = None,
    derived_categoricals: Mapping[str, DiscreteGrid | Mapping[str, DiscreteGrid]]
    | None = None,
    grid_config: GridConfig = GRID_CONFIG,
) -> Model:
    """Create the baseline structural retirement model.

    Args:
        fixed_params: Parameters to fix at model creation time. These are
            partialled into compiled functions and removed from the params
            template. Pass data-derived constants here; only estimation
            parameters should go through `model.simulate(params=...)`.
        derived_categoricals: Extra categorical mappings for derived variables
            not in the model's state/action grids. Needed when `fixed_params`
            contains `pd.Series` indexed by DAG function outputs.
        grid_config: Continuous-grid point counts. Defaults to production
            values; pass `BENCHMARK_GRID_CONFIG` for a fast-but-structurally-
            faithful benchmark.

    Returns:
        A pylcm Model with 19 regimes (18 non-terminal + dead) spanning
        ages 51-95. Regime names follow the `<his>_<medicare>_<ss>_<work>` scheme.

    """
    ages = AgeGrid(
        start=MODEL_CONFIG.start_age,
        stop=MODEL_CONFIG.end_age - 1,
        step="Y",
    )
    regimes = build_all_regimes(grid_config)

    return Model(
        regimes=regimes,
        ages=ages,
        regime_id_class=RegimeId,
        description="Baseline structural retirement model (pre-ACA)",
        fixed_params=fixed_params or {},
        derived_categoricals=derived_categoricals,
    )
