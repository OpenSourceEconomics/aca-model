"""Tiny factories that wrap `create_model` with the benchmark snapshot.

Used by tests that need a structurally faithful model without spelling
out fixed_params, wage_params, and a pref-type grid at every call site.
Production callers (aca-estimation, scripts) assemble these explicitly.
"""

from lcm import DiscreteGrid, Model

from aca_model.aca.health_insurance import PolicyVariant
from aca_model.aca.model import create_model as _create_aca_model
from aca_model.agent.health import GoodHealth
from aca_model.agent.labor_market import IsMarried
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline.health_insurance import HealthInsuranceState
from aca_model.baseline.model import create_model as _create_baseline_model
from aca_model.benchmark import get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG

_DERIVED_CATEGORICALS = {
    "good_health": DiscreteGrid(GoodHealth),
    "is_married": DiscreteGrid(IsMarried),
    "his": DiscreteGrid(HealthInsuranceState),
    "pref_type": DiscreteGrid(BenchmarkPrefType),
}


def make_baseline_model(*, n_subjects: int) -> Model:
    """Baseline model on `BENCHMARK_GRID_CONFIG` with the benchmark snapshot params."""
    fixed_params, wage_params, _ = get_benchmark_params(model=None)
    return _create_baseline_model(
        n_subjects=n_subjects,
        fixed_params=fixed_params,
        wage_params=wage_params,
        derived_categoricals=_DERIVED_CATEGORICALS,
        grid_config=BENCHMARK_GRID_CONFIG,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )


def make_aca_model(*, n_subjects: int, policy: PolicyVariant) -> Model:
    """ACA model on `BENCHMARK_GRID_CONFIG` with the benchmark snapshot params."""
    fixed_params, wage_params, _ = get_benchmark_params(model=None)
    return _create_aca_model(
        n_subjects=n_subjects,
        policy=policy,
        fixed_params=fixed_params,
        wage_params=wage_params,
        derived_categoricals=_DERIVED_CATEGORICALS,
        grid_config=BENCHMARK_GRID_CONFIG,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )
