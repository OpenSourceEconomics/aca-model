"""ACA structural retirement model variants.

Creates model variants for counterfactual ACA policy analysis by applying
function overrides on top of baseline regimes.
"""

from collections.abc import Mapping
from typing import Any

from lcm import AgeGrid, DiscreteGrid, Model

from aca_model.aca import PolicyVariant
from aca_model.aca.regimes import build_all_regimes
from aca_model.baseline.regimes import RegimeId
from aca_model.baseline.regimes._common import MAX_CONSUMPTION
from aca_model.config import GRID_CONFIG, MODEL_CONFIG, GridConfig


def create_model(
    *,
    n_subjects: int,
    policy: PolicyVariant = PolicyVariant.ACA,
    fixed_params: Mapping[str, Any] | None = None,
    wage_params: Mapping[str, Any] | None = None,
    derived_categoricals: Mapping[str, DiscreteGrid | Mapping[str, DiscreteGrid]]
    | None = None,
    grid_config: GridConfig = GRID_CONFIG,
) -> Model:
    """Create an ACA policy variant model.

    Args:
        n_subjects: Forwarded to `lcm.Model(n_subjects=...)`.
        policy: Which ACA policy combination to apply.
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
            values.

    Returns:
        pylcm Model with ACA-specific function overrides.

    """
    ages = AgeGrid(
        start=MODEL_CONFIG.start_age,
        stop=MODEL_CONFIG.end_age - 1,
        step="Y",
    )
    regimes = build_all_regimes(
        policy=policy,
        grid_config=grid_config,
        fixed_params=fixed_params,
        wage_params=wage_params,
    )

    model = Model(
        regimes=regimes,
        ages=ages,
        regime_id_class=RegimeId,
        description=f"Structural retirement model ({policy.name})",
        fixed_params=fixed_params or {},
        derived_categoricals=derived_categoricals,
        n_subjects=n_subjects,
    )
    model.max_consumption = MAX_CONSUMPTION
    return model
