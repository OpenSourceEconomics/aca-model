"""Tests for social security functions, ported from struct-ret.

Parameter values from French & Jones (2011) Appendix C.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
from helpers.social_security import (  # ty: ignore[unresolved-import]
    compute_di_dropout_scale,
    compute_pia_table,
)
from lcm.typing import ScalarInt

from aca_model.agent.labor_market import LaborSupply
from aca_model.environment import social_security
from aca_model.environment.social_security import ClaimedSS

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
AIME_ACCRUAL_FACTOR = jnp.asarray(0.025)
AGGREGATE_WAGE_GROWTH = jnp.asarray(0.03)
AIME_LAST_AGE_WITH_INDEXING = jnp.int32(59)
AIME_KINK_2_SCALAR = jnp.asarray(AIME_KINK_2)
SSDI_SGA = jnp.asarray(12840.0)

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

# Cumulative SSA actuarial factors indexed by age (period==age here): the
# reduction for claiming early and the delayed-retirement credit for claiming
# late. Ages below 62 hold the age-62 factor; ages above 70 hold the age-70
# factor, matching the data-prep clamp.
_ADJ_FACTORS = {
    62: 0.75,
    63: 0.8,
    64: 0.866666666667,
    65: 0.933333333333,
    66: 1.0,
    67: 1.08,
    68: 1.16,
    69: 1.24,
    70: 1.32,
}
_EARLY_RET_ADJ_NP = np.empty(100)
for _age in range(100):
    _clamped = max(62, min(70, _age))
    _EARLY_RET_ADJ_NP[_age] = _ADJ_FACTORS[_clamped]
EARLY_RET_ADJ = jnp.array(_EARLY_RET_ADJ_NP)

NORMAL_RETIREMENT_AGE = jnp.int32(66)

DI_SCALE = jnp.array(
    compute_di_dropout_scale(
        pd.Series(_RATIO_NP),
        AIME_ACCRUAL_FACTOR.item(),
        start_age=0,
        n_periods=100,
    )
)

# Pre-computed PIA lookup table (5-point exact grid). The fourth point is
# the taxable-max AIME; the fifth extends AIME above it so the largest
# delayed-retirement credit (max of the cumulative factors) baked into the
# carried AIME survives the round-trip instead of clamping at the taxable max.
MAX_DELAYED_FACTOR = float(EARLY_RET_ADJ.max())
_pia_grid_np, _pia_table_np = compute_pia_table(
    AIME_KINK_0,
    AIME_KINK_1,
    PIA_CONVERSION_RATE_0,
    PIA_CONVERSION_RATE_1,
    PIA_CONVERSION_RATE_2,
    AIME_KINK_2,
    MAX_DELAYED_FACTOR,
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
        claim_ss=jnp.array(ClaimedSS.no),
        claimed_ss=jnp.array(ClaimedSS.no),
        normal_retirement_age=NORMAL_RETIREMENT_AGE,
        early_ret_adjustment=EARLY_RET_ADJ,
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=jnp.int32(70),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime=jnp.array(1000.0),
        labor_income=jnp.array(20000.0),
        period=jnp.int32(58),
        age=jnp.int32(58),
        aime_accrual_factor=AIME_ACCRUAL_FACTOR,
        aggregate_wage_growth=AGGREGATE_WAGE_GROWTH,
        aime_last_age_with_indexing=AIME_LAST_AGE_WITH_INDEXING,
        aime_kink_2=AIME_KINK_2_SCALAR,
        ratio_lowest_earnings=RATIO,
    )
    expected = 1000 * 1.03 + (20000 - 0.2 * 1000 * 1.03) * 0.025
    assert jnp.isclose(result, expected, atol=ATOL)


def test_next_aime_indexing_low_income() -> None:
    result = social_security.next_aime(
        claim_ss=jnp.array(ClaimedSS.no),
        claimed_ss=jnp.array(ClaimedSS.no),
        normal_retirement_age=NORMAL_RETIREMENT_AGE,
        early_ret_adjustment=EARLY_RET_ADJ,
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=jnp.int32(70),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime=jnp.array(10000.0),
        labor_income=jnp.array(510.0),
        period=jnp.int32(58),
        age=jnp.int32(58),
        aime_accrual_factor=AIME_ACCRUAL_FACTOR,
        aggregate_wage_growth=AGGREGATE_WAGE_GROWTH,
        aime_last_age_with_indexing=AIME_LAST_AGE_WITH_INDEXING,
        aime_kink_2=AIME_KINK_2_SCALAR,
        ratio_lowest_earnings=RATIO,
    )
    assert jnp.isclose(result, 10000 * 1.03, atol=ATOL)


def test_next_aime_no_indexing_high_income() -> None:
    result = social_security.next_aime(
        claim_ss=jnp.array(ClaimedSS.no),
        claimed_ss=jnp.array(ClaimedSS.no),
        normal_retirement_age=NORMAL_RETIREMENT_AGE,
        early_ret_adjustment=EARLY_RET_ADJ,
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=jnp.int32(70),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime=jnp.array(1000.0),
        labor_income=jnp.array(20000.0),
        period=jnp.int32(62),
        age=jnp.int32(62),
        aime_accrual_factor=AIME_ACCRUAL_FACTOR,
        aggregate_wage_growth=AGGREGATE_WAGE_GROWTH,
        aime_last_age_with_indexing=AIME_LAST_AGE_WITH_INDEXING,
        aime_kink_2=AIME_KINK_2_SCALAR,
        ratio_lowest_earnings=RATIO,
    )
    expected = 1000 + (20000 - 0.4 * 1000) * 0.025
    assert jnp.isclose(result, expected, atol=ATOL)


def test_next_aime_no_indexing_low_income() -> None:
    result = social_security.next_aime(
        claim_ss=jnp.array(ClaimedSS.no),
        claimed_ss=jnp.array(ClaimedSS.no),
        normal_retirement_age=NORMAL_RETIREMENT_AGE,
        early_ret_adjustment=EARLY_RET_ADJ,
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=jnp.int32(70),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime=jnp.array(1000.0),
        labor_income=jnp.array(99.0),
        period=jnp.int32(62),
        age=jnp.int32(62),
        aime_accrual_factor=AIME_ACCRUAL_FACTOR,
        aggregate_wage_growth=AGGREGATE_WAGE_GROWTH,
        aime_last_age_with_indexing=AIME_LAST_AGE_WITH_INDEXING,
        aime_kink_2=AIME_KINK_2_SCALAR,
        ratio_lowest_earnings=RATIO,
    )
    assert jnp.isclose(result, 1000, atol=ATOL)


def test_next_aime_high_aime_high_income_accrues_above_taxable_max() -> None:
    """Within-period labor accrual above the taxable-max-indexed base is preserved.

    `_accrue_aime` caps the *indexed* base at the taxable max; the small extra
    accrual from current labor earnings then rides on top. The carried AIME is
    the round-trip of that accrued PIA — not re-clamped at the taxable max — so
    it lands just above `aime_kink_2`.
    """
    result = social_security.next_aime(
        claim_ss=jnp.array(ClaimedSS.no),
        claimed_ss=jnp.array(ClaimedSS.no),
        normal_retirement_age=NORMAL_RETIREMENT_AGE,
        early_ret_adjustment=EARLY_RET_ADJ,
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=jnp.int32(70),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime=jnp.array(40000.0),
        labor_income=jnp.array(20000.0),
        period=jnp.int32(62),
        age=jnp.int32(62),
        aime_accrual_factor=AIME_ACCRUAL_FACTOR,
        aggregate_wage_growth=AGGREGATE_WAGE_GROWTH,
        aime_last_age_with_indexing=AIME_LAST_AGE_WITH_INDEXING,
        aime_kink_2=AIME_KINK_2_SCALAR,
        ratio_lowest_earnings=RATIO,
    )
    capped_base = AIME_KINK_2
    lowest_year = _RATIO_NP[62] * capped_base
    expected = capped_base + (20000.0 - lowest_year) * float(AIME_ACCRUAL_FACTOR)
    assert jnp.isclose(result, expected, atol=ATOL)


def test_next_aime_cap_high_aime_low_income() -> None:
    result = social_security.next_aime(
        claim_ss=jnp.array(ClaimedSS.no),
        claimed_ss=jnp.array(ClaimedSS.no),
        normal_retirement_age=NORMAL_RETIREMENT_AGE,
        early_ret_adjustment=EARLY_RET_ADJ,
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=jnp.int32(70),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime=jnp.array(40000.0),
        labor_income=jnp.array(3500.0),
        period=jnp.int32(62),
        age=jnp.int32(62),
        aime_accrual_factor=AIME_ACCRUAL_FACTOR,
        aggregate_wage_growth=AGGREGATE_WAGE_GROWTH,
        aime_last_age_with_indexing=AIME_LAST_AGE_WITH_INDEXING,
        aime_kink_2=AIME_KINK_2_SCALAR,
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
    period = jnp.int32(55)
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
        age=jnp.int32(67),
        period=jnp.int32(0),
        claim_ss=jnp.array(ClaimedSS.yes),
        claimed_ss=jnp.array(ClaimedSS.no),
        labor_supply=jnp.array(LaborSupply.h2000),
        labor_income=jnp.array(4000.0),
        early_ret_adjustment=jnp.array([1.0]),
        normal_retirement_age=jnp.int32(66),
        earnings_test_threshold=jnp.array([10000.0]),
        earnings_test_fraction=jnp.array([0.0]),
        earnings_test_repealed_age=jnp.int32(70),
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
        age=jnp.int32(60),
        period=jnp.int32(0),
        claim_ss=jnp.array(ClaimedSS.yes),
        claimed_ss=jnp.array(ClaimedSS.no),
        labor_supply=jnp.array(LaborSupply.h2000),
        labor_income=jnp.array(6000.0),
        early_ret_adjustment=jnp.array([1.0]),
        normal_retirement_age=jnp.int32(66),
        earnings_test_threshold=jnp.array([2000.0]),
        earnings_test_fraction=jnp.array([0.2]),
        earnings_test_repealed_age=jnp.int32(70),
    )
    expected = pia_val - (6000 - 2000) * 0.2
    assert jnp.isclose(result, expected, atol=ATOL)


def test_benefit_inelig_pre65_disabled_below_sga() -> None:
    """Disabled agent below SGA: benefit = ssdi_pia."""
    ssdi_val = social_security.ssdi_pia(
        aime=jnp.array(5000.0),
        period=jnp.int32(55),
        di_dropout_scale=DI_SCALE,
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
    )
    result = social_security.benefit_inelig_pre65(
        ssdi_pia=ssdi_val,
        health=jnp.int32(0),  # disabled
        labor_income=jnp.array(0.0),
        ssdi_substantial_gainful_activity=SSDI_SGA,
    )
    assert jnp.isclose(result, ssdi_val, atol=ATOL)


def test_benefit_inelig_pre65_disabled_above_sga() -> None:
    """Disabled agent above SGA: benefit = 0."""
    ssdi_val = social_security.ssdi_pia(
        aime=jnp.array(5000.0),
        period=jnp.int32(55),
        di_dropout_scale=DI_SCALE,
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
    )
    result = social_security.benefit_inelig_pre65(
        ssdi_pia=ssdi_val,
        health=jnp.int32(0),  # disabled
        labor_income=jnp.array(20000.0),
        ssdi_substantial_gainful_activity=SSDI_SGA,
    )
    assert jnp.isclose(result, 0.0, atol=ATOL)


def test_benefit_inelig_pre65_not_disabled() -> None:
    """Non-disabled agent: benefit = 0."""
    result = social_security.benefit_inelig_pre65(
        ssdi_pia=jnp.array(1000.0),
        health=jnp.int32(2),  # good health
        labor_income=jnp.array(0.0),
        ssdi_substantial_gainful_activity=SSDI_SGA,
    )
    assert jnp.isclose(result, 0.0, atol=ATOL)


# --- DI dropout inverse (retained) ---


def test_di_dropout_round_trip_zero_years() -> None:
    aime = jnp.array(10000.0)
    scaled = aime * DI_SCALE[52]
    round_tripped = social_security.adjust_aime_di_dropout_inv(
        jnp.int32(52), scaled, DI_SCALE
    )
    assert jnp.isclose(aime, round_tripped, atol=ATOL)


def test_di_dropout_round_trip_positive_years() -> None:
    aime = jnp.array(10000.0)
    scaled = aime * DI_SCALE[62]
    round_tripped = social_security.adjust_aime_di_dropout_inv(
        jnp.int32(62), scaled, DI_SCALE
    )
    assert jnp.isclose(aime, round_tripped, rtol=0.0002)


# --- find_aime (exact inverse of the PIA lookup) ---


def test_find_aime_round_trips_pia_on_grid_kinks() -> None:
    """`find_aime` recovers the AIME that produced a given PIA at the bend points."""
    for aime_val in PIA_AIME_GRID:
        pia_val = social_security.pia(
            aime=aime_val,
            pia_table=PIA_TABLE,
            pia_aime_grid=PIA_AIME_GRID,
        )
        recovered = social_security.find_aime(
            pia=pia_val,
            pia_table=PIA_TABLE,
            pia_aime_grid=PIA_AIME_GRID,
        )
        assert jnp.isclose(recovered, aime_val, atol=ATOL)


# --- claim-age actuarial adjustment baked into next_aime ---


def _next_aime_claiming(
    *,
    aime: jnp.ndarray,
    age: int,
    claim_ss: ScalarInt,
    claimed_ss: ScalarInt,
    labor_income: float = 0.0,
) -> jnp.ndarray:
    """Advance AIME one period under a given claim state, no labor accrual."""
    return social_security.next_aime(
        aime=aime,
        labor_income=jnp.array(labor_income),
        period=jnp.int32(age),
        age=jnp.int32(age),
        claim_ss=jnp.array(claim_ss),
        claimed_ss=jnp.array(claimed_ss),
        normal_retirement_age=NORMAL_RETIREMENT_AGE,
        early_ret_adjustment=EARLY_RET_ADJ,
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=jnp.int32(70),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime_accrual_factor=AIME_ACCRUAL_FACTOR,
        aggregate_wage_growth=AGGREGATE_WAGE_GROWTH,
        aime_last_age_with_indexing=AIME_LAST_AGE_WITH_INDEXING,
        aime_kink_2=AIME_KINK_2_SCALAR,
        ratio_lowest_earnings=RATIO,
    )


def _pia_of(aime: jnp.ndarray) -> jnp.ndarray:
    return social_security.pia(
        aime=aime, pia_table=PIA_TABLE, pia_aime_grid=PIA_AIME_GRID
    )


def test_early_claim_holds_yields_permanent_reduction() -> None:
    """Claiming at 62 and holding leaves a benefit permanently reduced to 0.75·PIA.

    Iterating the AIME law from the claim age to 70 (no labor income) bakes the
    per-year early-retirement reduction into AIME. The benefit at 70, read off
    the carried AIME, is the unreduced PIA scaled by the cumulative 0.75 factor.
    """
    aime = jnp.array(20_000.0)
    pia_unreduced = _pia_of(aime)

    carried = aime
    for age in range(62, 70):
        carried = _next_aime_claiming(
            aime=carried, age=age, claim_ss=ClaimedSS.yes, claimed_ss=ClaimedSS.yes
        )

    benefit_at_70 = social_security.benefit_forced(pia=_pia_of(carried))
    np.testing.assert_allclose(benefit_at_70, 0.75 * pia_unreduced, rtol=1e-4)


def test_delayed_claim_yields_credit_increased_benefit() -> None:
    """Deferring to 68 raises the benefit above PIA by the delayed-retirement credit.

    Holding off on claiming from the normal retirement age to 68 bakes the
    per-year credit into AIME. Claiming at 68 then pays the unreduced PIA scaled
    up by the cumulative 1.16 factor.
    """
    aime = jnp.array(20_000.0)
    pia_unreduced = _pia_of(aime)

    carried = aime
    for age in range(66, 68):
        carried = _next_aime_claiming(
            aime=carried, age=age, claim_ss=ClaimedSS.no, claimed_ss=ClaimedSS.no
        )

    benefit_at_68 = social_security.benefit_forced(pia=_pia_of(carried))
    np.testing.assert_allclose(benefit_at_68, 1.16 * pia_unreduced, rtol=1e-4)


def test_delayed_claim_at_taxable_max_carries_credit_above_max() -> None:
    """A top earner who defers does not lose the delayed credit at the taxable max.

    For an agent at the taxable-max AIME who is unclaimed at the normal
    retirement age, the one-year delayed-retirement credit scales PIA above the
    maximum PIA. Converting that credited PIA back to AIME lands above the
    taxable max instead of clamping there, so the credit is carried forward.
    """
    aime_at_max = AIME_KINK_2_SCALAR
    result = social_security.next_aime(
        claim_ss=jnp.array(ClaimedSS.no),
        claimed_ss=jnp.array(ClaimedSS.no),
        normal_retirement_age=NORMAL_RETIREMENT_AGE,
        early_ret_adjustment=EARLY_RET_ADJ,
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=jnp.int32(70),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime=aime_at_max,
        labor_income=jnp.array(0.0),
        period=jnp.int32(66),
        age=jnp.int32(66),
        aime_accrual_factor=AIME_ACCRUAL_FACTOR,
        aggregate_wage_growth=AGGREGATE_WAGE_GROWTH,
        aime_last_age_with_indexing=AIME_LAST_AGE_WITH_INDEXING,
        aime_kink_2=AIME_KINK_2_SCALAR,
        ratio_lowest_earnings=RATIO,
    )

    max_pia = float(_pia_table_np[3])
    pia_kink_1 = float(_pia_table_np[2])
    delayed_factor = float(EARLY_RET_ADJ[67]) / float(EARLY_RET_ADJ[66])
    credited_pia = delayed_factor * max_pia
    expected_aime = AIME_KINK_1 + (credited_pia - pia_kink_1) / PIA_CONVERSION_RATE_2

    assert result > AIME_KINK_2
    np.testing.assert_allclose(result, expected_aime, rtol=1e-4)


def test_forced_claim_without_early_claim_pays_full_pia() -> None:
    """A never-early claimant forced to claim at 70 receives the full unreduced PIA.

    Holding from the normal retirement age to 70 without claiming and without
    earnings bakes the full delayed-retirement credit; that is the deferral path,
    not the focus here. The focus: never claiming early keeps AIME unbaked while
    age < NRA, so an agent who reaches the forced regime at 70 having never
    claimed early carries the unreduced AIME.
    """
    aime = jnp.array(20_000.0)
    pia_unreduced = _pia_of(aime)

    # Pre-NRA, never claiming: no bake, AIME unchanged (no labor income).
    carried = aime
    for age in range(62, 66):
        carried = _next_aime_claiming(
            aime=carried, age=age, claim_ss=ClaimedSS.no, claimed_ss=ClaimedSS.no
        )

    np.testing.assert_allclose(_pia_of(carried), pia_unreduced, rtol=1e-4)


def test_unclaimed_pre_nra_aime_is_not_baked() -> None:
    """Before NRA, an agent who has not claimed carries unbaked AIME.

    The early-reduction bake fires only once the agent has claimed; an agent
    still deciding (claim_ss == claimed_ss == no) keeps the unreduced AIME.
    """
    aime = jnp.array(20_000.0)
    result = _next_aime_claiming(
        aime=aime, age=63, claim_ss=ClaimedSS.no, claimed_ss=ClaimedSS.no
    )
    np.testing.assert_allclose(_pia_of(result), _pia_of(aime), rtol=1e-4)


def test_disabled_never_claimer_keeps_unadjusted_ssdi_pia() -> None:
    """A disabled never-claimer's SSDI benefit ignores the claim-age machinery.

    `ssdi_pia` reads the un-baked AIME (the DI branch is routed only when
    `ss == 0`), so the disability benefit equals the dropout-adjusted PIA
    regardless of the early/delayed actuarial factors.
    """
    aime = jnp.array(5000.0)
    period = jnp.int32(55)
    ssdi_val = social_security.ssdi_pia(
        aime=aime,
        period=period,
        di_dropout_scale=DI_SCALE,
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
    )
    result = social_security.benefit_inelig_pre65(
        ssdi_pia=ssdi_val,
        health=jnp.int32(0),
        labor_income=jnp.array(0.0),
        ssdi_substantial_gainful_activity=SSDI_SGA,
    )
    expected = social_security.aime_to_pia(aime=aime * DI_SCALE[55], **PIA_PARAMS)
    np.testing.assert_allclose(result, expected, atol=ATOL)


def test_pia_unadjusted_next_period_ignores_claim_bake() -> None:
    """The unadjusted next-period PIA channel reflects pure labor accrual only.

    Pension imputation reads `pia_unadjusted_next_period`, which equals the PIA of the
    labor-income-accrued AIME — never the claim-baked one. For an early claimant
    it therefore exceeds the (reduced) PIA the SS benefit would pay.
    """
    aime = jnp.array(20_000.0)
    unadjusted = social_security.pia_unadjusted_next_period(
        aime=aime,
        labor_income=jnp.array(0.0),
        period=jnp.int32(62),
        age=jnp.int32(62),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime_accrual_factor=AIME_ACCRUAL_FACTOR,
        aggregate_wage_growth=AGGREGATE_WAGE_GROWTH,
        aime_last_age_with_indexing=AIME_LAST_AGE_WITH_INDEXING,
        aime_kink_2=AIME_KINK_2_SCALAR,
        ratio_lowest_earnings=RATIO,
    )
    baked = _next_aime_claiming(
        aime=aime, age=62, claim_ss=ClaimedSS.yes, claimed_ss=ClaimedSS.yes
    )
    # With no labor income at age 62, the unadjusted PIA is the unbaked PIA.
    np.testing.assert_allclose(unadjusted, _pia_of(aime), rtol=1e-4)
    # The baked AIME's PIA is strictly lower (early reduction applied).
    assert _pia_of(baked) < unadjusted
