"""Declaring the asset law as resources plus the negated medical bill is exact.

With `subtract_hcc_bill`, the asset law toward every target is
`lcm.AdditiveShockTransition`: end-of-period resources before the out-of-pocket
bill, plus the shock `negative_oop_costs`. pylcm then averages each target's value
over the transitory medical-cost shock once per period, on the merged points
`{a_j - shock_k}`, rather than interpolating it at every shock node of every source
point. With linear interpolation in assets that average is exact, so the values
agree with the plain asset law to rounding.
"""

from collections.abc import Mapping
from dataclasses import replace

import jax.numpy as jnp
import lcm
import numpy as np
import pytest
from helpers.model import make_baseline_model

from aca_model.agent import assets_and_income
from aca_model.baseline.health_insurance import negative_oop_costs
from aca_model.benchmark import get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG

_FOLDED = replace(
    BENCHMARK_GRID_CONFIG, fold_hcc_transitory=True, subtract_hcc_bill=False
)
_ADDITIVE = replace(_FOLDED, subtract_hcc_bill=True)
_REGIME = "retiree_nomc_inelig_canwork"


@pytest.fixture(scope="module")
def additive_model() -> lcm.Model:
    return make_baseline_model(grid_config=_ADDITIVE)


def test_negative_oop_costs_is_the_negated_bill() -> None:
    np.testing.assert_array_equal(
        negative_oop_costs(oop_costs=jnp.array([0.0, 1.5, 250.0])),
        np.array([0.0, -1.5, -250.0]),
    )


def test_every_target_adds_the_negated_bill(additive_model: lcm.Model) -> None:
    for name, regime in additive_model.user_regimes.items():
        if name == "dead":
            continue
        laws = regime.state_transitions["assets"]
        assert isinstance(laws, Mapping)
        assert laws
        for law in laws.values():
            assert isinstance(law, lcm.AdditiveShockTransition)
            assert law.shock == "negative_oop_costs"


@pytest.mark.long_running
def test_additive_shock_value_equals_the_plain_law_value(
    additive_model: lcm.Model,
) -> None:
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


@pytest.mark.parametrize(
    ("bill", "expected_living", "expected_dead"),
    [(0.0, 85.0, 75.0), (20.0, 65.0, 55.0), (250.0, -165.0, -175.0)],
)
def test_asset_law_preserves_resources_minus_medical_costs(
    bill: float, expected_living: float, expected_dead: float
) -> None:
    common = {
        "cash_on_hand": jnp.array(100.0),
        "transfers": jnp.array(5.0),
        "consumption_dollars": jnp.array(30.0),
    }
    shock = negative_oop_costs(oop_costs=jnp.array(bill))
    living = (
        assets_and_income.assets_before_bill(
            **common, pension_assets_adjustment=jnp.array(10.0)
        )
        + shock
    )
    dead = assets_and_income.assets_before_bill_when_dead(**common) + shock
    np.testing.assert_allclose(living, expected_living, rtol=0, atol=0)
    np.testing.assert_allclose(dead, expected_dead, rtol=0, atol=0)
