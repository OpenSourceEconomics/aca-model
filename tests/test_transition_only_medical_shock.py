"""The transitory medical-cost shock is in no state space.

Nothing a period computes reads last year's transitory shock: this year's draw
enters the out-of-pocket bill inside the end-of-period asset law as
`next_hcc_transitory`, and the draw is IID. pylcm therefore draws it on the
transition only and integrates it in the continuation. Living regimes keep the
lagged persistent shock, which prices the private premium.
"""

from dataclasses import replace

import pytest
from helpers.model import make_baseline_model
from lcm import Model

from aca_model.config import BENCHMARK_GRID_CONFIG

_GRID_CONFIG = replace(BENCHMARK_GRID_CONFIG, subtract_hcc_bill=False)


@pytest.fixture(scope="module")
def model() -> Model:
    return make_baseline_model(grid_config=_GRID_CONFIG)


def test_living_regimes_carry_no_lagged_transitory_shock(model: Model) -> None:
    assert not any(
        "hcc_transitory" in model.state_names(regime_name=name)
        for name in model.user_regimes
    )


def test_living_regimes_keep_the_lagged_persistent_shock(model: Model) -> None:
    assert all(
        "hcc_persistent" in model.state_names(regime_name=name)
        for name in model.user_regimes
        if name != "dead"
    )
