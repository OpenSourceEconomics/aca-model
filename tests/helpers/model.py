"""Tiny factories that wrap `create_model` with `None` for every optional input.

Used by tests that don't need fixed params, wage params, or a custom pref-type
grid. These helpers exist so production `create_model` factories can stay
default-free without forcing every test call site to spell out
`fixed_params=None, wage_params=None, ...` six times.
"""

from lcm import Model

from aca_model.aca.health_insurance import PolicyVariant
from aca_model.aca.model import create_model as _create_aca_model
from aca_model.baseline.model import create_model as _create_baseline_model
from aca_model.config import GRID_CONFIG


def make_baseline_model(*, n_subjects: int) -> Model:
    """Baseline model with `GRID_CONFIG` and no fixed/wage/derived params."""
    return _create_baseline_model(
        n_subjects=n_subjects,
        fixed_params=None,
        wage_params=None,
        derived_categoricals=None,
        grid_config=GRID_CONFIG,
        pref_type_grid=None,
    )


def make_aca_model(*, n_subjects: int, policy: PolicyVariant) -> Model:
    """ACA model with `GRID_CONFIG` and no fixed/wage/derived params."""
    return _create_aca_model(
        n_subjects=n_subjects,
        policy=policy,
        fixed_params=None,
        wage_params=None,
        derived_categoricals=None,
        grid_config=GRID_CONFIG,
    )
