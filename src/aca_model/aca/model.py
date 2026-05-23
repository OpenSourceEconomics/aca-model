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
from aca_model.baseline.regimes import RegimeId
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
    subjects_batch_size: int = 0,
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
        subjects_batch_size: Per-device chunk size for the simulate-side
            per-subject dispatch. `0` (default) keeps a single vmap over
            all subjects; `>0` chunks each device's local shard via
            `jax.lax.map`. Tune via `grid_config.get_subjects_batch_size(log_level)`.

    Returns:
        pylcm Model.

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
        pref_type_grid=pref_type_grid,
    )

    return Model(
        regimes=regimes,
        ages=ages,
        regime_id_class=RegimeId,
        description=f"Structural retirement model ({policy.name})",
        fixed_params=fixed_params,
        derived_categoricals=derived_categoricals,
        n_subjects=n_subjects,
        subjects_batch_size=subjects_batch_size,
    )
