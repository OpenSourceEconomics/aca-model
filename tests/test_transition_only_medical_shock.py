"""The transitory medical-cost shock is in no state space.

Nothing a period computes reads last year's transitory shock: this year's draw
enters the out-of-pocket bill inside the end-of-period asset law as
`next_lagged_hcc_transitory`, and the draw is IID. pylcm therefore draws it on the
transition only and integrates it in the continuation. Living regimes keep the
lagged persistent shock, which prices the private premium.
"""

import pytest
from lcm import Model


@pytest.fixture(scope="module")
def model(baseline_model: Model) -> Model:
    return baseline_model


def test_living_regimes_carry_no_lagged_transitory_shock(model: Model) -> None:
    assert not any(
        "lagged_hcc_transitory" in model.state_names(regime_name=name)
        for name in model.user_regimes
    )


def test_living_regimes_keep_the_lagged_persistent_shock(model: Model) -> None:
    assert all(
        "lagged_hcc_persistent" in model.state_names(regime_name=name)
        for name in model.user_regimes
        if name != "dead"
    )
