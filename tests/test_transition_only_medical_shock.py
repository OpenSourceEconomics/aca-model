"""Folding the transitory medical-cost shock takes it out of every state space.

Nothing a period computes reads last year's transitory shock: this year's draw
enters the out-of-pocket bill inside the end-of-period asset law, and the draw is
IID. With `fold_hcc_transitory`, the shock is therefore drawn on the transition
only. Living regimes keep the lagged persistent shock, which prices the private
premium, and lose the transitory axis; values agree with the unfolded model on
every transitory slice.
"""

from dataclasses import replace

import numpy as np
import pytest
from helpers.model import make_baseline_model
from lcm import Model

from aca_model.benchmark import get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG

_FOLDED = replace(
    BENCHMARK_GRID_CONFIG, fold_hcc_transitory=True, subtract_hcc_bill=False
)


@pytest.fixture(scope="module")
def folded_model() -> Model:
    return make_baseline_model(grid_config=_FOLDED)


def test_living_regimes_carry_no_lagged_transitory_shock(folded_model: Model) -> None:
    assert not any(
        "hcc_transitory" in folded_model.state_names(regime_name=name)
        for name in folded_model.user_regimes
    )


def test_living_regimes_keep_the_lagged_persistent_shock(folded_model: Model) -> None:
    assert all(
        "hcc_persistent" in folded_model.state_names(regime_name=name)
        for name in folded_model.user_regimes
        if name != "dead"
    )


@pytest.mark.long_running
def test_folded_value_equals_every_transitory_slice_of_the_unfolded_value(
    folded_model: Model,
) -> None:
    """The two models' sums differ only in order, so they agree to rounding."""
    unfolded_model = make_baseline_model(
        grid_config=replace(
            BENCHMARK_GRID_CONFIG, fold_hcc_transitory=False, subtract_hcc_bill=False
        )
    )
    _, _, params = get_benchmark_params(model=unfolded_model)
    unfolded_values = unfolded_model.solve(params=params, log_level="off").values
    folded_values = folded_model.solve(params=params, log_level="off").values
    assert set(folded_values) == set(unfolded_values)
    for period, regimes in unfolded_values.items():
        assert set(folded_values[period]) == set(regimes)
        for name, values in regimes.items():
            unfolded = np.asarray(values)
            folded = np.asarray(folded_values[period][name])
            states = unfolded_model.state_names(regime_name=name)
            by_slice = (
                np.moveaxis(unfolded, states.index("hcc_transitory"), 0)
                if "hcc_transitory" in states
                else unfolded
            )
            expected = np.broadcast_to(folded, by_slice.shape)
            np.testing.assert_array_equal(np.isfinite(by_slice), np.isfinite(expected))
            np.testing.assert_allclose(
                by_slice,
                expected,
                rtol=1e-6,
                atol=0,
                equal_nan=False,
                err_msg=f"period={period}, regime={name}",
            )
