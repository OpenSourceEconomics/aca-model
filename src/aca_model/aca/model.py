"""ACA structural retirement model variants.

Creates model variants for counterfactual ACA policy analysis by applying
function overrides on top of baseline regimes.
"""

from collections.abc import Mapping
from typing import Any

from lcm import AgeGrid, DiscreteGrid, Model
from lcm.typing import UserParams

from aca_model.aca import PolicyVariant
from aca_model.aca.regimes import build_all_regimes
from aca_model.baseline.model import _fail_if_dcegm_without_consumption_points
from aca_model.baseline.regimes import RegimeId, SolverName, build_model_slots
from aca_model.config import MODEL_CONFIG, GridConfig


def create_model(
    *,
    n_subjects: int,
    policy: PolicyVariant,
    fixed_params: UserParams,
    wage_params: Mapping[str, Any],
    derived_categoricals: Mapping[str, DiscreteGrid],
    grid_config: GridConfig,
    pref_type_grid: DiscreteGrid,
    solver: SolverName = "brute_force",
    consumption_dollars_points: tuple[float, ...] | None = None,
) -> Model:
    """Create an ACA policy variant model.

    Args:
        n_subjects: Forwarded to `lcm.Model(n_subjects=...)`.
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

    Returns:
        pylcm Model.

    """
    ages = AgeGrid(
        start=MODEL_CONFIG.start_age,
        stop=MODEL_CONFIG.end_age - 1,
        step="Y",
    )
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
        ages=ages,
        regime_id_class=RegimeId,
        description=f"Structural retirement model ({policy.name})",
        fixed_params=fixed_params,
        derived_categoricals=derived_categoricals,
        n_subjects=n_subjects,
        **model_slots,
    )
