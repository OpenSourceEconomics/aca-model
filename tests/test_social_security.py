"""Tests for social security functions, ported from struct-ret.

Parameter values from French & Jones (2011) Appendix C.
"""

import jax.numpy as jnp
import numpy as np

from aca_model.agent.labor_market import LaborSupply
from aca_model.environment import social_security
from aca_model.environment.social_security import ClaimedSS
from tests.helpers.social_security import compute_di_dropout_scale, compute_pia_table

ATOL = 0.01

# French & Jones (2011) Appendix C
AIME_KINK_0 = 5724.0
AIME_KINK_1 = 34500.0
AIME_KINK_2 = 39000.0
PIA_CONVERSION_RATE_0 = 0.9
PIA_CONVERSION_RATE_1 = 0.32
PIA_CONVERSION_RATE_2 = 0.15
PIA_KINK_0 = 5151.6
PIA_KINK_1 = 14359.9
AIME_ACCRUAL_FACTOR = 0.025
AGGREGATE_WAGE_GROWTH = 0.03
AIME_LAST_AGE_WITH_INDEXING = 59
SSDI_SGA = 12840.0

PIA_PARAMS = {
    "aime_kink_0": AIME_KINK_0,
    "aime_kink_1": AIME_KINK_1,
    "pia_conversion_rate_0": PIA_CONVERSION_RATE_0,
    "pia_conversion_rate_1": PIA_CONVERSION_RATE_1,
    "pia_conversion_rate_2": PIA_CONVERSION_RATE_2,
    "pia_kink_0": PIA_KINK_0,
    "pia_kink_1": PIA_KINK_1,
}

# ratio_lowest_earnings indexed by period (start_age=0 so period==age)
_RATIO_NP = np.zeros(100)
_RATIO_NP[56] = 0.1
_RATIO_NP[57] = 0.15
_RATIO_NP[58] = 0.2
_RATIO_NP[59] = 0.25
_RATIO_NP[60] = 0.3
_RATIO_NP[61] = 0.35
_RATIO_NP[62] = 0.4
_RATIO_NP[63] = 0.45
_RATIO_NP[64] = 0.5
_RATIO_NP[65] = 0.55
_RATIO_NP[66] = 0.6
_RATIO_NP[67] = 0.65
_RATIO_NP[68] = 0.7
_RATIO_NP[69] = 0.7
RATIO = jnp.array(_RATIO_NP)

DI_SCALE = jnp.array(
    compute_di_dropout_scale(_RATIO_NP, AIME_ACCRUAL_FACTOR, start_age=0, n_periods=100)
)

# Pre-computed PIA lookup table (4-point exact grid)
MAX_AIME = AIME_KINK_2 * float(DI_SCALE.max()) * 1.1
_pia_grid_np, _pia_table_np = compute_pia_table(
    AIME_KINK_0,
    AIME_KINK_1,
    PIA_CONVERSION_RATE_0,
    PIA_CONVERSION_RATE_1,
    PIA_CONVERSION_RATE_2,
    MAX_AIME,
)
PIA_AIME_GRID = jnp.asarray(_pia_grid_np)
PIA_TABLE = jnp.asarray(_pia_table_np)


# --- aime_to_pia (analytics function, still available) ---


def test_aime_to_pia_below_kink_0() -> None:
    result = social_security.aime_to_pia(aime=jnp.array(2000.0), **PIA_PARAMS)
    assert jnp.isclose(result, 0.9 * 2000, atol=ATOL)


def test_aime_to_pia_between_kinks() -> None:
    result = social_security.aime_to_pia(aime=jnp.array(6724.0), **PIA_PARAMS)
    assert jnp.isclose(result, 5151.6 + 0.32 * 1000, atol=ATOL)


def test_aime_to_pia_above_kink_1() -> None:
    result = social_security.aime_to_pia(aime=jnp.array(40000.0), **PIA_PARAMS)
    assert jnp.isclose(result, 14359.9 + 0.15 * 5500, atol=ATOL)


# --- pia_to_aime ---


def test_pia_to_aime_below_kink_0() -> None:
    pia = 0.9 * 2000
    result = social_security.pia_to_aime(pia=jnp.array(pia), **PIA_PARAMS)
    assert jnp.isclose(result, 2000, atol=ATOL)


def test_pia_to_aime_between_kinks() -> None:
    pia = 5151.6 + 0.32 * 1000
    result = social_security.pia_to_aime(pia=jnp.array(pia), **PIA_PARAMS)
    assert jnp.isclose(result, 6724, atol=ATOL)


def test_pia_to_aime_above_kink_1() -> None:
    pia = 14359.9 + 0.15 * 5500
    result = social_security.pia_to_aime(pia=jnp.array(pia), **PIA_PARAMS)
    assert jnp.isclose(result, 40000, atol=ATOL)


# --- next_aime ---


def test_next_aime_indexing_high_income() -> None:
    result = social_security.next_aime(
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=70,
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime=jnp.array(1000.0),
        labor_income=jnp.array(20000.0),
        period=58,
        age=58,
        aime_accrual_factor=AIME_ACCRUAL_FACTOR,
        aggregate_wage_growth=AGGREGATE_WAGE_GROWTH,
        aime_last_age_with_indexing=AIME_LAST_AGE_WITH_INDEXING,
        aime_kink_2=AIME_KINK_2,
        ratio_lowest_earnings=RATIO,
    )
    expected = 1000 * 1.03 + (20000 - 0.2 * 1000 * 1.03) * 0.025
    assert jnp.isclose(result, expected, atol=ATOL)


def test_next_aime_indexing_low_income() -> None:
    result = social_security.next_aime(
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=70,
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime=jnp.array(10000.0),
        labor_income=jnp.array(510.0),
        period=58,
        age=58,
        aime_accrual_factor=AIME_ACCRUAL_FACTOR,
        aggregate_wage_growth=AGGREGATE_WAGE_GROWTH,
        aime_last_age_with_indexing=AIME_LAST_AGE_WITH_INDEXING,
        aime_kink_2=AIME_KINK_2,
        ratio_lowest_earnings=RATIO,
    )
    assert jnp.isclose(result, 10000 * 1.03, atol=ATOL)


def test_next_aime_no_indexing_high_income() -> None:
    result = social_security.next_aime(
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=70,
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime=jnp.array(1000.0),
        labor_income=jnp.array(20000.0),
        period=62,
        age=62,
        aime_accrual_factor=AIME_ACCRUAL_FACTOR,
        aggregate_wage_growth=AGGREGATE_WAGE_GROWTH,
        aime_last_age_with_indexing=AIME_LAST_AGE_WITH_INDEXING,
        aime_kink_2=AIME_KINK_2,
        ratio_lowest_earnings=RATIO,
    )
    expected = 1000 + (20000 - 0.4 * 1000) * 0.025
    assert jnp.isclose(result, expected, atol=ATOL)


def test_next_aime_no_indexing_low_income() -> None:
    result = social_security.next_aime(
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=70,
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime=jnp.array(1000.0),
        labor_income=jnp.array(99.0),
        period=62,
        age=62,
        aime_accrual_factor=AIME_ACCRUAL_FACTOR,
        aggregate_wage_growth=AGGREGATE_WAGE_GROWTH,
        aime_last_age_with_indexing=AIME_LAST_AGE_WITH_INDEXING,
        aime_kink_2=AIME_KINK_2,
        ratio_lowest_earnings=RATIO,
    )
    assert jnp.isclose(result, 1000, atol=ATOL)


def test_next_aime_cap_high_aime_high_income() -> None:
    result = social_security.next_aime(
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=70,
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime=jnp.array(40000.0),
        labor_income=jnp.array(20000.0),
        period=62,
        age=62,
        aime_accrual_factor=AIME_ACCRUAL_FACTOR,
        aggregate_wage_growth=AGGREGATE_WAGE_GROWTH,
        aime_last_age_with_indexing=AIME_LAST_AGE_WITH_INDEXING,
        aime_kink_2=AIME_KINK_2,
        ratio_lowest_earnings=RATIO,
    )
    assert jnp.isclose(result, 39000, atol=ATOL)


def test_next_aime_cap_high_aime_low_income() -> None:
    result = social_security.next_aime(
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=70,
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime=jnp.array(40000.0),
        labor_income=jnp.array(3500.0),
        period=62,
        age=62,
        aime_accrual_factor=AIME_ACCRUAL_FACTOR,
        aggregate_wage_growth=AGGREGATE_WAGE_GROWTH,
        aime_last_age_with_indexing=AIME_LAST_AGE_WITH_INDEXING,
        aime_kink_2=AIME_KINK_2,
        ratio_lowest_earnings=RATIO,
    )
    assert jnp.isclose(result, 39000, atol=ATOL)


# --- pia DAG function (lookup table) ---


def test_pia_lookup_matches_formula() -> None:
    """PIA lookup via interp matches the aime_to_pia formula exactly."""
    test_aime = jnp.array([0.0, 2000.0, 5724.0, 10000.0, 34500.0, 39000.0])
    for aime_val in test_aime:
        lookup = social_security.pia(
            aime=aime_val,
            pia_table=PIA_TABLE,
            pia_aime_grid=PIA_AIME_GRID,
        )
        formula = social_security.aime_to_pia(aime=aime_val, **PIA_PARAMS)
        assert jnp.isclose(lookup, formula, atol=ATOL)


def test_ssdi_pia_matches_dropout_adjusted() -> None:
    """ssdi_pia lookup matches aime_to_pia(aime * di_dropout_scale[period])."""
    aime = jnp.array(5000.0)
    period = 55
    adjusted_aime = aime * DI_SCALE[period]

    lookup = social_security.ssdi_pia(
        aime=aime,
        period=period,
        di_dropout_scale=DI_SCALE,
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
    )
    formula = social_security.aime_to_pia(aime=adjusted_aime, **PIA_PARAMS)
    assert jnp.isclose(lookup, formula, atol=ATOL)


# --- benefit functions (new simplified signatures) ---


def test_benefit_forced_equals_pia() -> None:
    pia_val = social_security.pia(
        aime=jnp.array(5000.0),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
    )
    result = social_security.benefit_forced(pia=pia_val)
    assert jnp.isclose(result, pia_val, atol=ATOL)


def test_benefit_choose_post65_below_et_threshold() -> None:
    """Below earnings test threshold: benefit = PIA (with early ret adj = 1)."""
    pia_val = social_security.pia(
        aime=jnp.array(5000.0),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
    )
    result = social_security.benefit_choose_post65(
        pia=pia_val,
        age=67,
        period=0,
        claim_ss=jnp.array(ClaimedSS.yes),
        claimed_ss=jnp.array(ClaimedSS.no),
        labor_supply=jnp.array(LaborSupply.h2000),
        labor_income=jnp.array(4000.0),
        early_ret_adjustment=jnp.array([1.0]),
        normal_retirement_age=66,
        earnings_test_threshold=jnp.array([10000.0]),
        earnings_test_fraction=jnp.array([0.0]),
        earnings_test_repealed_age=70,
    )
    assert jnp.isclose(result, pia_val, atol=ATOL)


def test_benefit_choose_post65_partially_reduced() -> None:
    """Earnings test reduces benefit by fraction of excess earnings."""
    pia_val = social_security.pia(
        aime=jnp.array(5000.0),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
    )
    result = social_security.benefit_choose_post65(
        pia=pia_val,
        age=60,
        period=0,
        claim_ss=jnp.array(ClaimedSS.yes),
        claimed_ss=jnp.array(ClaimedSS.no),
        labor_supply=jnp.array(LaborSupply.h2000),
        labor_income=jnp.array(6000.0),
        early_ret_adjustment=jnp.array([1.0]),
        normal_retirement_age=66,
        earnings_test_threshold=jnp.array([2000.0]),
        earnings_test_fraction=jnp.array([0.2]),
        earnings_test_repealed_age=70,
    )
    expected = pia_val - (6000 - 2000) * 0.2
    assert jnp.isclose(result, expected, atol=ATOL)


def test_benefit_inelig_pre65_disabled_below_sga() -> None:
    """Disabled agent below SGA: benefit = ssdi_pia."""
    ssdi_val = social_security.ssdi_pia(
        aime=jnp.array(5000.0),
        period=55,
        di_dropout_scale=DI_SCALE,
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
    )
    result = social_security.benefit_inelig_pre65(
        ssdi_pia=ssdi_val,
        health=jnp.array(0),  # disabled
        labor_income=jnp.array(0.0),
        ssdi_substantial_gainful_activity=SSDI_SGA,
    )
    assert jnp.isclose(result, ssdi_val, atol=ATOL)


def test_benefit_inelig_pre65_disabled_above_sga() -> None:
    """Disabled agent above SGA: benefit = 0."""
    ssdi_val = social_security.ssdi_pia(
        aime=jnp.array(5000.0),
        period=55,
        di_dropout_scale=DI_SCALE,
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
    )
    result = social_security.benefit_inelig_pre65(
        ssdi_pia=ssdi_val,
        health=jnp.array(0),  # disabled
        labor_income=jnp.array(20000.0),
        ssdi_substantial_gainful_activity=SSDI_SGA,
    )
    assert jnp.isclose(result, 0.0, atol=ATOL)


def test_benefit_inelig_pre65_not_disabled() -> None:
    """Non-disabled agent: benefit = 0."""
    result = social_security.benefit_inelig_pre65(
        ssdi_pia=jnp.array(1000.0),
        health=jnp.array(2),  # good health
        labor_income=jnp.array(0.0),
        ssdi_substantial_gainful_activity=SSDI_SGA,
    )
    assert jnp.isclose(result, 0.0, atol=ATOL)


# --- DI dropout inverse (retained) ---


def test_di_dropout_round_trip_zero_years() -> None:
    aime = jnp.array(10000.0)
    scaled = aime * DI_SCALE[52]
    round_tripped = social_security.adjust_aime_di_dropout_inv(52, scaled, DI_SCALE)
    assert jnp.isclose(aime, round_tripped, atol=ATOL)


def test_di_dropout_round_trip_positive_years() -> None:
    aime = jnp.array(10000.0)
    scaled = aime * DI_SCALE[62]
    round_tripped = social_security.adjust_aime_di_dropout_inv(62, scaled, DI_SCALE)
    assert jnp.isclose(aime, round_tripped, rtol=0.0002)
