"""Declaring the asset law as resources plus the negated medical bill is exact.

With `subtract_hcc_bill`, the asset law toward every target is
`lcm.AdditiveShockTransition`: end-of-period resources before the out-of-pocket
bill, plus the shock `negative_oop_costs`. pylcm then averages each target's value
over the transitory medical-cost shock once per period, on the merged points
`{a_j - shock_k}`, rather than interpolating it at every shock node of every source
point. With linear interpolation in assets that average is exact, so the values
agree with the plain asset law to rounding.
"""

from dataclasses import replace

import jax.numpy as jnp
import lcm
import numpy as np
import pytest
from helpers.model import make_baseline_model

from aca_model.baseline.health_insurance import negative_oop_costs
from aca_model.benchmark import get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG

_FOLDED = replace(BENCHMARK_GRID_CONFIG, fold_hcc_transitory=True)
_ADDITIVE = replace(_FOLDED, subtract_hcc_bill=True)
_REGIME = "retiree_nomc_inelig_canwork"


@pytest.fixture(scope="module")
def additive_model():
    return make_baseline_model(grid_config=_ADDITIVE)


def test_negative_oop_costs_is_the_negated_bill():
    np.testing.assert_array_equal(
        negative_oop_costs(oop_costs=jnp.array([0.0, 1.5, 250.0])),
        np.array([0.0, -1.5, -250.0]),
    )


def test_every_target_adds_the_negated_bill(additive_model):
    laws = [
        (type(law), law.shock)
        for name, regime in additive_model.user_regimes.items()
        if name != "dead"
        for law in regime.state_transitions["assets"].values()
    ]

    assert laws
    assert set(laws) == {(lcm.AdditiveShockTransition, "negative_oop_costs")}


@pytest.mark.long_running
def test_additive_shock_value_equals_the_plain_law_value(additive_model):
    """The two routes sum the same terms in a different order."""
    folded_model = make_baseline_model(grid_config=_FOLDED)
    _, _, params = get_benchmark_params(model=folded_model)
    folded = np.asarray(
        folded_model.solve(params=params, log_level="off").values[0][_REGIME]
    )
    additive = np.asarray(
        additive_model.solve(params=params, log_level="off").values[0][_REGIME]
    )

    np.testing.assert_allclose(additive, folded, rtol=1e-9, atol=0)
