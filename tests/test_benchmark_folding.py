"""The ACA benchmark integrates the transitory medical bill out of every state."""

from collections.abc import Mapping

import lcm

from aca_model.benchmark import get_benchmark_initial_conditions


def test_benchmark_folds_the_transitory_medical_bill(
    benchmark_model: lcm.Model,
) -> None:
    model = benchmark_model
    living = {
        name: regime for name, regime in model.user_regimes.items() if name != "dead"
    }
    assert living
    assert all(
        "lagged_hcc_transitory" not in model.state_names(regime_name=name)
        for name in living
    )
    for regime in living.values():
        laws = regime.state_transitions["assets"]
        assert isinstance(laws, Mapping)
        assert all(
            isinstance(law, lcm.AdditiveShockTransition) for law in laws.values()
        )
    initial = get_benchmark_initial_conditions(model=model, n_agents=8, seed=42)
    assert "lagged_hcc_transitory" not in initial
