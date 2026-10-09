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

from aca_model.benchmark import get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG

_FOLDED = replace(BENCHMARK_GRID_CONFIG, fold_hcc_transitory=True)
_REGIME = "retiree_nomc_inelig_canwork"


@pytest.fixture(scope="module")
def folded_model():
    return make_baseline_model(grid_config=_FOLDED)


def test_living_regimes_carry_no_lagged_transitory_shock(folded_model):
    assert not any(
        "hcc_transitory" in folded_model.state_names(regime_name=name)
        for name in folded_model.user_regimes
    )


def test_living_regimes_keep_the_lagged_persistent_shock(folded_model):
    assert all(
        "hcc_persistent" in folded_model.state_names(regime_name=name)
        for name in folded_model.user_regimes
        if name != "dead"
    )


@pytest.mark.long_running
def test_folded_value_equals_every_transitory_slice_of_the_unfolded_value(
    folded_model,
):
    """The two models' sums differ only in order, so they agree to rounding."""
    unfolded_model = make_baseline_model()
    _, _, params = get_benchmark_params(model=unfolded_model)
    unfolded = np.asarray(
        unfolded_model.solve(params=params, log_level="off").values[0][_REGIME]
    )
    folded = np.asarray(
        folded_model.solve(params=params, log_level="off").values[0][_REGIME]
    )
    axis = unfolded_model.state_names(regime_name=_REGIME).index("hcc_transitory")
    by_slice = np.moveaxis(unfolded, axis, 0)

    np.testing.assert_allclose(
        by_slice, np.broadcast_to(folded, by_slice.shape), rtol=1e-6, atol=0
    )
