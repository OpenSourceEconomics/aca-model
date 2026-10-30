"""The ACA benchmark integrates the transitory medical bill out of every state."""

from collections.abc import Mapping

import lcm
import pytest
from lcm import DiscreteGrid

from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.benchmark import create_benchmark_model, get_benchmark_initial_conditions


@pytest.fixture(scope="module")
def benchmark_model() -> lcm.Model:
    return create_benchmark_model(pref_type_grid=DiscreteGrid(BenchmarkPrefType))


def test_benchmark_folds_the_transitory_medical_bill(
    benchmark_model: lcm.Model,
) -> None:
    model = benchmark_model
    living = {
        name: regime for name, regime in model.user_regimes.items() if name != "dead"
    }
    assert living
    assert all(
        "hcc_transitory" not in model.state_names(regime_name=name) for name in living
    )
    for regime in living.values():
        laws = regime.state_transitions["assets"]
        assert isinstance(laws, Mapping)
        assert all(
            isinstance(law, lcm.AdditiveShockTransition) for law in laws.values()
        )
    initial = get_benchmark_initial_conditions(model=model, n_subjects=8, seed=42)
    assert "hcc_transitory" not in initial
