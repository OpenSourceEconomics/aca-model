"""Baseline structural retirement model using pylcm.

This is the main model specification, ported from struct-ret/. The baseline
represents pre-ACA rules (no individual mandate, no ACA subsidies).

Usage:
    from aca_model.baseline.model import create_model
    model = create_model(n_subjects=...)
    params = get_default_params()
    V = model.solve(params)
"""

from collections.abc import Mapping
from typing import Any

from lcm import AgeGrid, DiscreteGrid, Model

from aca_model.baseline.health_insurance import HealthInsuranceState
from aca_model.baseline.regimes import RegimeId, build_all_regimes
from aca_model.baseline.regimes._common import MAX_CONSUMPTION_UNEQUIV
from aca_model.config import MODEL_CONFIG, GridConfig


def create_model(
    *,
    n_subjects: int,
    fixed_params: Mapping[str, Any] | None,
    wage_params: Mapping[str, Any] | None,
    derived_categoricals: Mapping[str, DiscreteGrid] | None,
    grid_config: GridConfig,
    pref_type_grid: DiscreteGrid | None,
) -> Model:
    """Create the baseline structural retirement model.

    Args:
        n_subjects: Forwarded to `lcm.Model(n_subjects=...)`.
        fixed_params: Parameters to fix at model creation time, or `None`
            to skip. Fixed params are partialled into compiled functions
            and removed from the params template. Pass data-derived
            constants here; only estimation parameters should go through
            `model.simulate(params=...)`.
        wage_params: Data-derived wage profile dict (`log_ft_wage_mean`,
            `log_ft_wage_std`, `adj_wage_hours_*`) used only at grid-build
            time to size the assets-floor to `-max_annual_labor_income`.
            Not routed to the pylcm Model. `None` skips the floor sizing.
        derived_categoricals: Extra categorical mappings for derived
            variables not in the model's state/action grids, or `None`.
            Needed when `fixed_params` contains `pd.Series` indexed by DAG
            function outputs.
        grid_config: Continuous-grid point counts. Pass `GRID_CONFIG` for
            production values or `BENCHMARK_GRID_CONFIG` for the
            fast-but-structurally-faithful benchmark.
        pref_type_grid: Pref-type `DiscreteGrid`, or `None` to use
            `DiscreteGrid(PrefType)`. Pass a custom grid to substitute
            the production layout (e.g. the 2-type benchmark variant).

    Returns:
        A pylcm Model with 19 regimes (18 non-terminal + dead) spanning
        ages 51-95. Regime names follow the `<his>_<medicare>_<ss>_<work>` scheme.

    """
    ages = AgeGrid(
        start=MODEL_CONFIG.start_age,
        stop=MODEL_CONFIG.end_age - 1,
        step="Y",
    )
    regimes = build_all_regimes(
        grid_config=grid_config,
        fixed_params=fixed_params,
        wage_params=wage_params,
        pref_type_grid=pref_type_grid,
    )

    # `target_his` is a DAG output of `health_insurance.target_his` (set on
    # nongroup/tied/retiree regimes). The pension imputation correction
    # (`imputed_pension_wealth_next_period`) indexes shifted arrays by
    # `arr[period, target_his]`; pylcm needs the categorical declared so
    # `pd.Series` fixed_params with a `target_his` index level resolve.
    base_derived: dict[str, DiscreteGrid] = {
        "target_his": DiscreteGrid(HealthInsuranceState),
    }
    if derived_categoricals is not None:
        base_derived.update(derived_categoricals)

    model = Model(
        regimes=regimes,
        ages=ages,
        regime_id_class=RegimeId,
        description="Baseline structural retirement model (pre-ACA)",
        fixed_params=fixed_params or {},
        derived_categoricals=base_derived,
        n_subjects=n_subjects,
    )
    # See `MAX_CONSUMPTION_UNEQUIV` in `baseline.regimes._common` for why this
    # rides on the Model instance instead of `fixed_params`.
    model.max_consumption_unequiv = MAX_CONSUMPTION_UNEQUIV
    return model
