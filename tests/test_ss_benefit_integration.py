"""Integration tests for Social Security benefit determination.

Test the earnings test behavior: benefits are reduced when working before
full retirement age, and unreduced after.
"""

import jax.numpy as jnp
from aca_data.social_security import compute_pia_table

from aca_model.agent.labor_market import LaborSupply
from aca_model.environment import social_security
from aca_model.environment.social_security import ClaimedSS

ATOL = 0.01

# SS formula parameters
AIME_KINK_0 = 816.0
AIME_KINK_1 = 4917.0
PIA_CONV_0 = 0.9
PIA_CONV_1 = 0.32
PIA_CONV_2 = 0.15
MAX_AIME = 50000.0

_pia_grid_np, _pia_table_np = compute_pia_table(
    AIME_KINK_0,
    AIME_KINK_1,
    PIA_CONV_0,
    PIA_CONV_1,
    PIA_CONV_2,
    MAX_AIME,
)
PIA_AIME_GRID = jnp.asarray(_pia_grid_np)
PIA_TABLE = jnp.asarray(_pia_table_np)


def test_earnings_test_reduces_benefit_before_fra() -> None:
    """Working claimant before FRA: earnings test reduces SS benefit."""
    pia_val = social_security.pia(
        aime=jnp.array(3000.0),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
    )
    n_periods = 45
    ssdi_pia_val = social_security.ssdi_pia(
        aime=jnp.array(3000.0),
        period=12,
        di_dropout_scale=jnp.ones(n_periods + 1),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
    )

    benefit_working = social_security.benefit_choose_pre65(
        pia=pia_val,
        ssdi_pia=ssdi_pia_val,
        age=63,
        period=0,
        claim_ss=jnp.array(ClaimedSS.yes),
        claimed_ss=jnp.array(ClaimedSS.no),
        health=jnp.array(2),
        labor_supply=jnp.array(LaborSupply.h2000),
        labor_income=jnp.array(30000.0),
        early_ret_adjustment=jnp.array([0.75]),
        normal_retirement_age=66,
        earnings_test_threshold=jnp.array([17640.0]),
        earnings_test_fraction=jnp.array([0.5]),
        earnings_test_repealed_age=66,
        ssdi_substantial_gainful_activity=13560.0,
    )

    benefit_not_working = social_security.benefit_choose_pre65(
        pia=pia_val,
        ssdi_pia=ssdi_pia_val,
        age=63,
        period=0,
        claim_ss=jnp.array(ClaimedSS.yes),
        claimed_ss=jnp.array(ClaimedSS.no),
        health=jnp.array(2),
        labor_supply=jnp.array(LaborSupply.do_not_work),
        labor_income=jnp.array(0.0),
        early_ret_adjustment=jnp.array([0.75]),
        normal_retirement_age=66,
        earnings_test_threshold=jnp.array([17640.0]),
        earnings_test_fraction=jnp.array([0.5]),
        earnings_test_repealed_age=66,
        ssdi_substantial_gainful_activity=13560.0,
    )

    assert benefit_working < benefit_not_working
    assert benefit_working >= 0.0


def test_earnings_test_not_applied_after_fra() -> None:
    """After FRA, earnings test is repealed — full benefit regardless of work."""
    pia_val = social_security.pia(
        aime=jnp.array(3000.0),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
    )

    benefit_post65 = social_security.benefit_choose_post65(
        pia=pia_val,
        age=67,
        period=0,
        claim_ss=jnp.array(ClaimedSS.yes),
        claimed_ss=jnp.array(ClaimedSS.no),
        labor_supply=jnp.array(LaborSupply.h2000),
        labor_income=jnp.array(50000.0),
        early_ret_adjustment=jnp.array([1.0]),
        normal_retirement_age=66,
        earnings_test_threshold=jnp.array([17640.0]),
        earnings_test_fraction=jnp.array([0.5]),
        earnings_test_repealed_age=66,
    )

    benefit_not_working = social_security.benefit_choose_post65(
        pia=pia_val,
        age=67,
        period=0,
        claim_ss=jnp.array(ClaimedSS.yes),
        claimed_ss=jnp.array(ClaimedSS.no),
        labor_supply=jnp.array(LaborSupply.do_not_work),
        labor_income=jnp.array(0.0),
        early_ret_adjustment=jnp.array([1.0]),
        normal_retirement_age=66,
        earnings_test_threshold=jnp.array([17640.0]),
        earnings_test_fraction=jnp.array([0.5]),
        earnings_test_repealed_age=66,
    )

    assert jnp.isclose(benefit_post65, benefit_not_working, atol=ATOL)
    assert benefit_post65 > 0.0


def test_aime_to_pia_progressive() -> None:
    """PIA formula is progressive: higher AIME → lower marginal conversion rate."""
    low = social_security.pia(
        aime=jnp.array(500.0),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
    )
    high = social_security.pia(
        aime=jnp.array(6000.0),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
    )

    # Low AIME gets 90% conversion, high AIME blends 90%/32%/15%
    assert low / 500.0 > high / 6000.0  # Average rate decreases
    assert high > low  # But absolute PIA still higher
