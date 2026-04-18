"""Tests that baseline functions have clean interfaces without ACA parameters.

Baseline `cash_on_hand` should not take subsidy/penalty params, and baseline
`primary_oop` should not take a cost_sharing_scale param. The ACA-aware
versions with those extra params live in `aca.health_insurance`.
"""

import jax.numpy as jnp

from aca_model.aca import health_insurance as aca_hi
from aca_model.agent import assets_and_income
from aca_model.baseline import health_insurance
from aca_model.baseline.health_insurance import BuyPrivate


def test_baseline_cash_on_hand_no_aca_params() -> None:
    """Baseline cash_on_hand has no subsidy or mandate params."""
    assets = jnp.array(100000.0)
    ati = jnp.array(30000.0)
    ssi = jnp.array(5000.0)
    premium = jnp.array(3000.0)
    result = assets_and_income.cash_on_hand(
        assets=assets,
        after_tax_income=ati,
        ssi_benefit=ssi,
        hic_premium=premium,
    )
    expected = assets + ati + ssi - premium
    assert jnp.isclose(result, expected)


def test_aca_cash_on_hand_with_subsidy_and_penalty() -> None:
    """ACA cash_on_hand adds subsidy and subtracts penalty."""
    assets = jnp.array(100000.0)
    ati = jnp.array(30000.0)
    ssi = jnp.array(5000.0)
    premium = jnp.array(3000.0)
    subsidy = jnp.array(1000.0)
    penalty = jnp.array(500.0)
    result = aca_hi.cash_on_hand(
        assets=assets,
        after_tax_income=ati,
        ssi_benefit=ssi,
        hic_premium=premium,
        hic_premium_subsidy=subsidy,
        mandate_penalty=penalty,
    )
    expected = assets + ati + ssi - premium + subsidy - penalty
    assert jnp.isclose(result, expected)


def test_aca_cash_on_hand_matches_baseline_when_neutral() -> None:
    """ACA cash_on_hand with zero subsidy/penalty equals baseline."""
    assets = jnp.array(100000.0)
    ati = jnp.array(30000.0)
    ssi = jnp.array(5000.0)
    premium = jnp.array(3000.0)
    baseline_result = assets_and_income.cash_on_hand(
        assets=assets,
        after_tax_income=ati,
        ssi_benefit=ssi,
        hic_premium=premium,
    )
    aca_result = aca_hi.cash_on_hand(
        assets=assets,
        after_tax_income=ati,
        ssi_benefit=ssi,
        hic_premium=premium,
        hic_premium_subsidy=jnp.array(0.0),
        mandate_penalty=jnp.array(0.0),
    )
    assert jnp.isclose(baseline_result, aca_result)


def test_baseline_primary_oop_no_cost_sharing_scale() -> None:
    """Baseline primary_oop applies raw deductible/coinsurance/oop_max."""
    costs = jnp.array(5000.0)
    deductible = 500.0
    coinsurance = 0.2
    oop_max_val = 3000.0
    result = health_insurance.primary_oop(
        total_health_costs=costs,
        buy_private=jnp.array(BuyPrivate.yes),
        deductible=deductible,
        coinsurance_rate=coinsurance,
        oop_max=oop_max_val,
    )
    expected = health_insurance.oop_costs(
        total_health_costs=costs,
        deductible=deductible,
        coinsurance_rate=coinsurance,
        oop_max=oop_max_val,
    )
    assert jnp.isclose(result, expected)


def test_aca_primary_oop_scaled_reduces_costs() -> None:
    """ACA primary_oop with scale < 1.0 reduces OOP costs."""
    costs = jnp.array(5000.0)
    deductible = 500.0
    coinsurance = 0.2
    oop_max_val = 3000.0
    oop_full = aca_hi.primary_oop(
        total_health_costs=costs,
        cost_sharing_scale=jnp.array(1.0),
        buy_private=jnp.array(BuyPrivate.yes),
        deductible=deductible,
        coinsurance_rate=coinsurance,
        oop_max=oop_max_val,
    )
    oop_reduced = aca_hi.primary_oop(
        total_health_costs=costs,
        cost_sharing_scale=jnp.array(0.3),
        buy_private=jnp.array(BuyPrivate.yes),
        deductible=deductible,
        coinsurance_rate=coinsurance,
        oop_max=oop_max_val,
    )
    assert oop_reduced < oop_full


def test_medicaid_eligible_baseline_delegates_to_ssi_true() -> None:
    """is_medicaid_eligible_baseline returns True when is_ssi_eligible is True."""
    result = health_insurance.is_medicaid_eligible(is_ssi_eligible=jnp.array(True))
    assert jnp.array_equal(result, jnp.array(True))


def test_medicaid_eligible_baseline_delegates_to_ssi_false() -> None:
    """is_medicaid_eligible_baseline returns False when is_ssi_eligible is False."""
    result = health_insurance.is_medicaid_eligible(is_ssi_eligible=jnp.array(False))
    assert jnp.array_equal(result, jnp.array(False))
