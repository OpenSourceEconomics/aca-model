"""The ACA benchmark activates transitory-shock folding explicitly."""

from collections.abc import Mapping
from dataclasses import replace

import lcm
import numpy as np

from aca_model.benchmark import get_benchmark_initial_conditions
from aca_model.config import BENCHMARK_GRID_CONFIG


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


def test_unfolded_control_has_identical_shared_initial_conditions(
    benchmark_model: lcm.Model, baseline_model: lcm.Model
) -> None:
    """The unfolded control is the benchmark built on unfolded grids."""
    control_grid = replace(
        BENCHMARK_GRID_CONFIG, fold_hcc_transitory=False, subtract_hcc_bill=False
    )
    control = baseline_model
    assert all(
        "hcc_transitory" in control.state_names(regime_name=name)
        for name in control.user_regimes
        if name != "dead"
    )
    folded_initial = get_benchmark_initial_conditions(
        model=benchmark_model, n_subjects=8, seed=42
    )
    control_initial = get_benchmark_initial_conditions(
        model=control, n_subjects=8, seed=42, grid_config=control_grid
    )
    assert set(control_initial) == set(folded_initial) | {"hcc_transitory"}
    for name, values in folded_initial.items():
        np.testing.assert_array_equal(values, control_initial[name])
