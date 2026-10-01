"""Tiny factories that wrap `create_model` with the benchmark snapshot.

Used by tests that need a structurally faithful model without spelling
out fixed_params, wage_params, and a pref-type grid at every call site.
Production callers (aca-slurm, scripts) assemble these explicitly.
"""

from lcm import DiscreteGrid, Model

from aca_model.aca.health_insurance import PolicyVariant
from aca_model.aca.model import create_model as _create_aca_model
from aca_model.agent.health import GoodHealth
from aca_model.agent.labor_market import IsMarried
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline.health_insurance import HealthInsuranceState
from aca_model.baseline.model import create_model as _create_baseline_model
from aca_model.baseline.regimes import REGIME_SPECS
from aca_model.benchmark import get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG

_NONGROUP_BEFORE_MEDICARE = frozenset(
    name
    for name, spec in REGIME_SPECS.items()
    if spec["his"] == "nongroup" and spec["mc"] == "nomc"
)

_DERIVED_CATEGORICALS = {
    "good_health": DiscreteGrid(GoodHealth),
    "is_married": DiscreteGrid(IsMarried),
    "his": DiscreteGrid(HealthInsuranceState),
    "target_his": DiscreteGrid(HealthInsuranceState),
    "pref_type": DiscreteGrid(BenchmarkPrefType),
}


def make_baseline_model() -> Model:
    """Baseline model on `BENCHMARK_GRID_CONFIG` with the benchmark snapshot params."""
    fixed_params, wage_params, _ = get_benchmark_params(model=None)
    return _create_baseline_model(
        fixed_params=fixed_params,
        wage_params=wage_params,
        derived_categoricals=_DERIVED_CATEGORICALS,
        grid_config=BENCHMARK_GRID_CONFIG,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )


def make_aca_model(*, policy: PolicyVariant) -> Model:
    """ACA model on `BENCHMARK_GRID_CONFIG` with the benchmark snapshot params.

    Under a variant with the reformed non-group market, the non-group regimes
    before Medicare price cover at the community-rated ACA premium, so the
    snapshot's risk-rated premium parameters (prefix `premium_`) are dropped
    from those regimes, as aca-slurm does.
    """
    fixed_params, wage_params, _ = get_benchmark_params(model=None)
    if policy != PolicyVariant.ACA_ONLY_MEDICAID_EXPANSION:
        fixed_params = {
            name: (
                {k: v for k, v in value.items() if not k.startswith("premium_")}
                if name in _NONGROUP_BEFORE_MEDICARE
                else value
            )
            for name, value in fixed_params.items()
        }
    return _create_aca_model(
        policy=policy,
        fixed_params=fixed_params,
        wage_params=wage_params,
        derived_categoricals=_DERIVED_CATEGORICALS,
        grid_config=BENCHMARK_GRID_CONFIG,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )
