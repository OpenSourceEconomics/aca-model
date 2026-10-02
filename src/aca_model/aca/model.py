"""ACA structural retirement model variants.

Creates model variants for counterfactual ACA policy analysis by applying
function overrides on top of baseline regimes.
"""

from collections.abc import Mapping
from typing import Any

from lcm import AgeGrid, DiscreteGrid, ExecutionConfig, Model
from lcm.typing import UserParams

from aca_model.aca import PolicyVariant
from aca_model.aca.regimes import build_all_regimes
from aca_model.baseline.model import _fail_if_dcegm_without_consumption_points
from aca_model.baseline.regimes import (
    INITIAL_REGIMES,
    RegimeId,
    SolverName,
    build_model_slots,
)
from aca_model.config import MODEL_AGES, GridConfig
from aca_model.execution import execution_config_for_devices


def create_model(
    *,
    policy: PolicyVariant,
    fixed_params: UserParams,
    wage_params: Mapping[str, Any],
    derived_categoricals: Mapping[str, DiscreteGrid],
    grid_config: GridConfig,
    pref_type_grid: DiscreteGrid,
    solver: SolverName = "brute_force",
    consumption_dollars_points: tuple[float, ...] | None = None,
    execution_config: ExecutionConfig | None = None,
) -> Model:
    """Create an ACA policy variant model.

    Args:
        policy: Which ACA policy combination to apply (e.g.
            `PolicyVariant.ACA`).
        fixed_params: Parameters to fix at model creation time. Pass
            data-derived constants here; only estimation parameters
            should go through `model.simulate(params=...)`.
        wage_params: Data-derived wage profile dict (`log_ft_wage_mean`,
            `log_ft_wage_std`, `adj_wage_hours_*`) used only at grid-build
            time to size the assets-floor to `-max_annual_labor_income`.
            Not routed to the pylcm Model.
        derived_categoricals: Categorical mappings for `pd.Series`
            fixed_params index levels that aren't model state/action
            grids — `target_his`, `his`, `good_health`, `is_married`,
            `pref_type`.
        grid_config: Continuous-grid point counts.
        pref_type_grid: Pref-type `DiscreteGrid`.
        solver: `"brute_force"` (the default) or `"dcegm"`; see
            `aca_model.baseline.model.create_model`.
        consumption_dollars_points: Construction-time consumption action
            gridpoints; required under DC-EGM. See
            `aca_model.baseline.model.create_model`.
        execution_config: Explicit hardware-local policy forwarded unchanged.
            None uses the smallest selected accelerator allocator limit as the
            device-memory budget; CPU construction remains unbudgeted.

    Returns:
        pylcm Model.

    """
    ages = AgeGrid(exact_values=MODEL_AGES)
    _fail_if_dcegm_without_consumption_points(
        solver=solver, consumption_dollars_points=consumption_dollars_points
    )
    regimes = build_all_regimes(
        policy=policy,
        grid_config=grid_config,
        fixed_params=fixed_params,
        wage_params=wage_params,
        pref_type_grid=pref_type_grid,
        solver=solver,
        consumption_dollars_points=consumption_dollars_points,
    )
    # The overlay swaps only regime-level functions; the broadcast slots
    # are policy-invariant.
    model_slots = build_model_slots(
        grid_config=grid_config,
        fixed_params=fixed_params,
        wage_params=wage_params,
        pref_type_grid=pref_type_grid,
        solver=solver,
    )

    return Model(
        regimes=regimes,
        initial_regimes=INITIAL_REGIMES,
        ages=ages,
        regime_id_class=RegimeId,
        description=f"Structural retirement model ({policy.name})",
        fixed_params=fixed_params,
        derived_categoricals=derived_categoricals,
        execution_config=(
            execution_config_for_devices()
            if execution_config is None
            else execution_config
        ),
        **model_slots,
    )
