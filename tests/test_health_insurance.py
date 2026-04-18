"""Tests for health insurance / SSI functions, ported from struct-ret."""

import jax.numpy as jnp
import pytest

from aca_model.agent.labor_market import LaborSupply
from aca_model.baseline import health_insurance
from aca_model.baseline.health_insurance import BuyPrivate

ATOL = 0.01

SSI_ASSETS_TEST = jnp.array([2000.0, 3000.0, 3000.0])
SSI_MAX_BENEFIT = jnp.array([8000.0, 12000.0, 12000.0])


def test_ssi_eligible_assets_too_high() -> None:
    result = health_insurance.is_ssi_eligible(
        assets=jnp.array(5000.0),
        countable_income=jnp.array(1000.0),
        spousal_income=jnp.array(0),
        gets_medicare=True,
        ssi_assets_test=SSI_ASSETS_TEST,
        ssi_maximum_benefit=SSI_MAX_BENEFIT,
    )
    assert not result


def test_ssi_eligible_income_too_high() -> None:
    result = health_insurance.is_ssi_eligible(
        assets=jnp.array(1000.0),
        countable_income=jnp.array(9000.0),
        spousal_income=jnp.array(0),
        gets_medicare=True,
        ssi_assets_test=SSI_ASSETS_TEST,
        ssi_maximum_benefit=SSI_MAX_BENEFIT,
    )
    assert not result


def test_ssi_eligible_no_medicare() -> None:
    result = health_insurance.is_ssi_eligible(
        assets=jnp.array(1000.0),
        countable_income=jnp.array(1000.0),
        spousal_income=jnp.array(0),
        gets_medicare=False,
        ssi_assets_test=SSI_ASSETS_TEST,
        ssi_maximum_benefit=SSI_MAX_BENEFIT,
    )
    assert not result


def test_ssi_eligible_all_pass() -> None:
    result = health_insurance.is_ssi_eligible(
        assets=jnp.array(1000.0),
        countable_income=jnp.array(1000.0),
        spousal_income=jnp.array(0),
        gets_medicare=True,
        ssi_assets_test=SSI_ASSETS_TEST,
        ssi_maximum_benefit=SSI_MAX_BENEFIT,
    )
    assert result


def test_ssi_benefit_eligible() -> None:
    result = health_insurance.ssi_benefit(
        countable_income=jnp.array(3000.0),
        spousal_income=jnp.array(0),
        is_ssi_eligible=jnp.array(True),
        ssi_maximum_benefit=SSI_MAX_BENEFIT,
    )
    assert jnp.isclose(result, 8000.0 - 3000.0, atol=ATOL)


def test_ssi_benefit_not_eligible() -> None:
    result = health_insurance.ssi_benefit(
        countable_income=jnp.array(3000.0),
        spousal_income=jnp.array(0),
        is_ssi_eligible=jnp.array(False),
        ssi_maximum_benefit=SSI_MAX_BENEFIT,
    )
    assert jnp.isclose(result, 0.0, atol=ATOL)


# --- predicted_hcc_insurer (DAG interpolation function) ---

GRID = jnp.array([-1.0, 0.0, 1.0])
TABLE = jnp.array([100.0, 200.0, 400.0])


def test_predicted_hcc_insurer_on_grid() -> None:
    result = health_insurance.hcc_insurer_predicted(
        hcc_persistent=jnp.array(0.0),
        predicted_hcc_insurer_table=TABLE,
        hcc_persistent_grid=GRID,
    )
    assert jnp.isclose(result, 200.0, atol=ATOL)


def test_predicted_hcc_insurer_off_grid() -> None:
    result = health_insurance.hcc_insurer_predicted(
        hcc_persistent=jnp.array(-0.5),
        predicted_hcc_insurer_table=TABLE,
        hcc_persistent_grid=GRID,
    )
    # Linear interpolation: midpoint of 100 and 200
    assert jnp.isclose(result, 150.0, atol=ATOL)


# --- compute_predicted_hcc_insurer_table ---


@pytest.fixture
def table_inputs() -> dict:
    n_p = 3
    return {
        "hcc_persistent_grid": jnp.array([-1.0, 0.0, 1.0]),
        "hcc_persistent_trans_probs": jnp.eye(n_p),  # identity = no mixing
        "hcc_transitory_grid": jnp.array([-0.5, 0.5]),
        "hcc_transitory_weights": jnp.array([0.5, 0.5]),
        "log_mean": 8.0,
        "log_std": 1.0,
        "std_xsect_persistent": 0.8,
        "deductible": 500.0,
        "coinsurance_rate": 0.2,
        "oop_max": 5000.0,
    }


def test_compute_table_shape(table_inputs: dict) -> None:
    result = health_insurance.compute_hcc_insurer_table(**table_inputs)
    assert result.shape == (3,)


def test_compute_table_nonnegative(table_inputs: dict) -> None:
    result = health_insurance.compute_hcc_insurer_table(**table_inputs)
    assert jnp.all(result >= 0.0)


def test_compute_table_uniform_transition(table_inputs: dict) -> None:
    """With uniform transition probs, all table entries are equal."""
    n_p = 3
    table_inputs["hcc_persistent_trans_probs"] = jnp.ones((n_p, n_p)) / n_p
    result = health_insurance.compute_hcc_insurer_table(**table_inputs)
    assert jnp.allclose(result, result[0], atol=ATOL)


# --- premium: buy_private conditioning ---


_PREMIUM_KWARGS: dict = {
    "age": 60,
    "good_health": jnp.array(True),
    "is_married": jnp.array(False),
    "labor_supply": jnp.array(LaborSupply.h2000),
    "premium_intercept": 1000.0,
    "premium_age": 0,
    "premium_age_sq": 0.0,
    "premium_age_cub": 0.0,
    "premium_predicted_hcc": 0.0,
    "premium_good_health": 0.0,
    "premium_married": 0.0,
    "premium_works": 0.0,
    "premium_married_works": 0.0,
    "premium_minimum": 500.0,
    "predicted_hcc_insurer": jnp.array(0.0),
}


def test_premium_insured_positive() -> None:
    result = health_insurance.premium(
        buy_private=jnp.array(BuyPrivate.yes), **_PREMIUM_KWARGS
    )
    assert result > 0.0


def test_premium_uninsured_zero() -> None:
    result = health_insurance.premium(
        buy_private=jnp.array(BuyPrivate.no), **_PREMIUM_KWARGS
    )
    assert jnp.isclose(result, 0.0)


# --- primary_oop: buy_private conditioning ---


def test_primary_oop_insured_applies_deductible_coinsurance() -> None:
    result = health_insurance.primary_oop(
        total_health_costs=jnp.array(10000.0),
        buy_private=jnp.array(BuyPrivate.yes),
        deductible=500.0,
        coinsurance_rate=0.2,
        oop_max=5000.0,
    )
    expected = 500.0 + (10000.0 - 500.0) * 0.2  # 2400
    assert jnp.isclose(result, expected, atol=ATOL)


def test_primary_oop_uninsured_equals_total_costs() -> None:
    total = jnp.array(10000.0)
    result = health_insurance.primary_oop(
        total_health_costs=total,
        buy_private=jnp.array(BuyPrivate.no),
        deductible=500.0,
        coinsurance_rate=0.2,
        oop_max=5000.0,
    )
    assert jnp.isclose(result, total)
