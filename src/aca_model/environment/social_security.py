"""Social Security types and functions: AIME, PIA, earnings test, benefits.

Ported from struct-ret/src/model/baseline/soc_sec_pensions_taxes.py.

PIA is pre-computed on a 4-point grid (the piecewise-linear kink points) in aca-data
and looked up via `jnp.interp` — same pattern as `predicted_hcc_insurer`. This
eliminates 7 PIA formula constants from the DAG.
"""

import jax.numpy as jnp
from lcm import categorical
from lcm.typing import ContinuousState, DiscreteAction, DiscreteState, FloatND, Period

from aca_model.agent.labor_market import LaborSupply


@categorical(ordered=False)
class ClaimedSS:
    no: int
    yes: int


def next_claimed_ss(
    claim_ss: DiscreteAction,
    claimed_ss: DiscreteState,
) -> DiscreteState:
    """Absorbing transition: once claimed, always claimed."""
    return jnp.maximum(claim_ss, claimed_ss)


def enter_claimed_ss() -> DiscreteState:
    """Initial claimed_ss when entering the SS eligibility window."""
    return ClaimedSS.no


# --- PIA lookup (DAG functions) ---


def pia(
    aime: ContinuousState,
    pia_table: FloatND,
    pia_aime_grid: FloatND,
) -> FloatND:
    """Look up Primary Insurance Amount from pre-computed table.

    The table has 4 points (0, kink_0, kink_1, max_aime) with exact PIA values.
    Linear interpolation reproduces the piecewise-linear formula exactly.
    """
    return jnp.interp(aime, pia_aime_grid, pia_table)


def ssdi_pia(
    aime: ContinuousState,
    period: Period,
    di_dropout_scale: FloatND,
    pia_table: FloatND,
    pia_aime_grid: FloatND,
) -> FloatND:
    """Look up PIA for disabled agents with dropout-year AIME adjustment.

    Adjusts AIME for DI dropout years before interpolating from the
    pre-computed PIA table.
    """
    adjusted_aime = aime * di_dropout_scale[period]
    return jnp.interp(adjusted_aime, pia_aime_grid, pia_table)


# --- Benefit functions (DAG functions, take `pia` / `ssdi_pia` from DAG) ---


def benefit_forced(
    pia: FloatND,
) -> FloatND:
    """SS benefit when claiming is forced: benefit equals PIA."""
    return pia


def benefit_choose_post65(
    pia: FloatND,
    age: int,
    period: Period,
    claim_ss: DiscreteAction,
    claimed_ss: DiscreteState,
    labor_supply: DiscreteAction,
    labor_income: FloatND,
    early_ret_adjustment: FloatND,
    normal_retirement_age: int,
    earnings_test_threshold: FloatND,
    earnings_test_fraction: FloatND,
    earnings_test_repealed_age: int,
) -> FloatND:
    """SS benefit for post-65, ss=choose: SS if claiming, 0 otherwise."""
    ss = jnp.maximum(claim_ss, claimed_ss)
    work = labor_supply != LaborSupply.do_not_work
    return _apply_benefit_rules(
        pia=pia,
        age=age,
        period=period,
        ss=ss,
        work=work,
        labor_income=labor_income,
        early_ret_adjustment=early_ret_adjustment,
        normal_retirement_age=normal_retirement_age,
        earnings_test_threshold=earnings_test_threshold,
        earnings_test_fraction=earnings_test_fraction,
        earnings_test_repealed_age=earnings_test_repealed_age,
    )


def benefit_choose_pre65(
    pia: FloatND,
    ssdi_pia: FloatND,
    age: int,
    period: Period,
    claim_ss: DiscreteAction,
    claimed_ss: DiscreteState,
    health: DiscreteState,
    labor_supply: DiscreteAction,
    labor_income: FloatND,
    early_ret_adjustment: FloatND,
    normal_retirement_age: int,
    earnings_test_threshold: FloatND,
    earnings_test_fraction: FloatND,
    earnings_test_repealed_age: int,
    ssdi_substantial_gainful_activity: float,
) -> FloatND:
    """SS benefit for pre-65, ss=choose: SS if claiming, SSDI if disabled, else 0."""
    ss = jnp.maximum(claim_ss, claimed_ss)
    work = labor_supply != LaborSupply.do_not_work
    is_disabled = health == 0

    regular = _apply_benefit_rules(
        pia=pia,
        age=age,
        period=period,
        ss=ss,
        work=work,
        labor_income=labor_income,
        early_ret_adjustment=early_ret_adjustment,
        normal_retirement_age=normal_retirement_age,
        earnings_test_threshold=earnings_test_threshold,
        earnings_test_fraction=earnings_test_fraction,
        earnings_test_repealed_age=earnings_test_repealed_age,
    )
    ssdi = jnp.where(
        labor_income > ssdi_substantial_gainful_activity,
        0.0,
        ssdi_pia,
    )

    not_claiming = ss == 0
    return jnp.where(
        ss > 0,
        regular,
        jnp.where(not_claiming & is_disabled, ssdi, 0.0),
    )


def benefit_inelig_pre65(
    ssdi_pia: FloatND,
    health: DiscreteState,
    labor_income: FloatND,
    ssdi_substantial_gainful_activity: float,
) -> FloatND:
    """SS benefit for pre-65, ss=inelig: SSDI if disabled, else 0."""
    is_disabled = health == 0
    ssdi = jnp.where(
        labor_income > ssdi_substantial_gainful_activity,
        0.0,
        ssdi_pia,
    )
    return jnp.where(is_disabled, ssdi, 0.0)


# --- Benefit withholding (DAG function for credit-back) ---


def benefit_withheld_fraction(
    pia: FloatND,
    ss_benefit: FloatND,
    claim_ss: DiscreteAction,
    claimed_ss: DiscreteState,
) -> FloatND:
    """Fraction of raw PIA withheld (early retirement + earnings test).

    Zero when not claiming regular SS. Used by `next_aime` to credit back
    future AIME for benefit withholding during the earnings test.
    """
    is_claiming = jnp.maximum(claim_ss, claimed_ss) > 0
    return jnp.where(
        is_claiming & (pia > 0),
        jnp.maximum(0.0, 1.0 - ss_benefit / pia),
        0.0,
    )


# --- Private helper (NOT a DAG function) ---


def _apply_benefit_rules(
    *,
    pia: FloatND,
    age: int,
    period: Period,
    ss: FloatND,
    work: FloatND,
    labor_income: FloatND,
    early_ret_adjustment: FloatND,
    normal_retirement_age: int,
    earnings_test_threshold: FloatND,
    earnings_test_fraction: FloatND,
    earnings_test_repealed_age: int,
) -> FloatND:
    """Apply early retirement adjustment and earnings test to PIA.

    Returns 0 if not claiming (ss == 0).
    """
    # Apply early retirement adjustment
    raw_benefit = jnp.where(
        age < normal_retirement_age,
        pia * early_ret_adjustment[period],
        pia,
    )

    # Apply earnings test (only if working and below repealed age)
    excess_earnings = jnp.maximum(0.0, labor_income - earnings_test_threshold[period])
    reduction = excess_earnings * earnings_test_fraction[period]
    post_et_benefit = jnp.maximum(0.0, raw_benefit - reduction)

    benefit_if_working = jnp.where(
        age >= earnings_test_repealed_age,
        raw_benefit,
        post_et_benefit,
    )

    result = jnp.where(work > 0, benefit_if_working, raw_benefit)

    # Zero if not claiming
    return jnp.where(ss > 0, result, 0.0)


# --- AIME transition functions (state transitions, no aime_to_pia calls) ---


def next_aime(
    aime: ContinuousState,
    labor_income: FloatND,
    period: Period,
    age: int,
    benefit_withheld_fraction: FloatND,
    earnings_test_credited_back: FloatND,
    earnings_test_repealed_age: int,
    pia_table: FloatND,
    pia_aime_grid: FloatND,
    aime_accrual_factor: float,
    aggregate_wage_growth: float,
    aime_last_age_with_indexing: int,
    aime_kink_2: float,
    ratio_lowest_earnings: FloatND,
) -> ContinuousState:
    """Compute next period's AIME given labor earnings.

    Steps:
    1. Apply wage indexing if age <= indexing cutoff
    2. Cap AIME and labor income at maximum taxable earnings
    3. Replace lowest earnings year if current earnings exceed it
    4. Credit back for earnings test withholding (PIA round-trip)
    """
    # Apply aggregate wage growth for indexing
    indexed_aime = jnp.where(
        age <= aime_last_age_with_indexing,
        aime * (1.0 + aggregate_wage_growth),
        aime,
    )

    # Cap at maximum taxable earnings
    capped_aime = jnp.minimum(indexed_aime, aime_kink_2)
    capped_labor = jnp.minimum(labor_income, aime_kink_2)

    # Earnings from lowest year
    lowest_year_earnings = ratio_lowest_earnings[period] * capped_aime

    # Only accrue if labor income exceeds lowest year earnings
    accrual = (
        jnp.maximum(0.0, capped_labor - lowest_year_earnings) * aime_accrual_factor
    )
    new_aime = capped_aime + accrual

    # Credit back: increase AIME to compensate for benefit withholding
    credit = jnp.where(
        age < earnings_test_repealed_age,
        earnings_test_credited_back[period] * benefit_withheld_fraction,
        0.0,
    )
    new_pia = jnp.interp(new_aime, pia_aime_grid, pia_table)
    desired_pia = new_pia * (1.0 + credit)
    credited_aime = jnp.interp(desired_pia, pia_table, pia_aime_grid)

    return jnp.minimum(
        jnp.where(credit > 0, credited_aime, new_aime),
        aime_kink_2,
    )


def next_aime_disabled(
    aime: ContinuousState,
    labor_income: FloatND,
    period: Period,
    age: int,
    health: DiscreteState,
    benefit_withheld_fraction: FloatND,
    earnings_test_credited_back: FloatND,
    earnings_test_repealed_age: int,
    pia_table: FloatND,
    pia_aime_grid: FloatND,
    aime_accrual_factor: float,
    aggregate_wage_growth: float,
    aime_last_age_with_indexing: int,
    aime_kink_2: float,
    ratio_lowest_earnings: FloatND,
    medicare_age: int,
    di_dropout_scale: FloatND,
    di_dropout_next_period_ratio: FloatND,
) -> ContinuousState:
    """AIME transition for pre-65 regimes handling both disabled and non-disabled.

    Non-disabled: standard AIME accrual from labor income + credit-back.
    Disabled: maintain PIA continuity across ages by adjusting AIME so the
    DI dropout-year scale factor change doesn't alter the benefit.
    At Medicare transition, stores the dropout-adjusted AIME (switching to OA).
    """
    # --- Regular path (non-disabled) ---
    indexed_aime = jnp.where(
        age <= aime_last_age_with_indexing,
        aime * (1.0 + aggregate_wage_growth),
        aime,
    )
    capped_aime = jnp.minimum(indexed_aime, aime_kink_2)
    capped_labor = jnp.minimum(labor_income, aime_kink_2)
    lowest_year_earnings = ratio_lowest_earnings[period] * capped_aime
    accrual = (
        jnp.maximum(0.0, capped_labor - lowest_year_earnings) * aime_accrual_factor
    )
    new_aime = capped_aime + accrual

    # Credit back for earnings test withholding
    credit = jnp.where(
        age < earnings_test_repealed_age,
        earnings_test_credited_back[period] * benefit_withheld_fraction,
        0.0,
    )
    new_pia = jnp.interp(new_aime, pia_aime_grid, pia_table)
    desired_pia = new_pia * (1.0 + credit)
    credited_aime = jnp.interp(desired_pia, pia_table, pia_aime_grid)
    regular = jnp.minimum(
        jnp.where(credit > 0, credited_aime, new_aime),
        aime_kink_2,
    )

    # --- Disabled path ---
    disabled_next = jnp.where(
        age + 1 < medicare_age,
        aime * di_dropout_next_period_ratio[period],
        aime * di_dropout_scale[period],
    )

    is_disabled = health == 0
    return jnp.where(is_disabled, disabled_next, regular)


# --- Analytics functions (NOT DAG functions, used for post-estimation analysis) ---


def aime_to_pia(
    aime: ContinuousState,
    aime_kink_0: float,
    aime_kink_1: float,
    pia_conversion_rate_0: float,
    pia_conversion_rate_1: float,
    pia_conversion_rate_2: float,
    pia_kink_0: float,
    pia_kink_1: float,
) -> FloatND:
    """Convert Average Indexed Monthly Earnings to Primary Insurance Amount.

    Three-bracket progressive formula. Used for analytics and tests;
    the DAG uses pre-computed lookup tables via `pia()` instead.
    """
    pia_bracket_0 = pia_conversion_rate_0 * aime
    pia_bracket_1 = pia_kink_0 + pia_conversion_rate_1 * (aime - aime_kink_0)
    pia_bracket_2 = pia_kink_1 + pia_conversion_rate_2 * (aime - aime_kink_1)

    return jnp.where(
        aime < aime_kink_0,
        pia_bracket_0,
        jnp.where(aime < aime_kink_1, pia_bracket_1, pia_bracket_2),
    )


def pia_to_aime(
    pia: FloatND,
    aime_kink_0: float,
    aime_kink_1: float,
    pia_conversion_rate_0: float,
    pia_conversion_rate_1: float,
    pia_conversion_rate_2: float,
    pia_kink_0: float,
    pia_kink_1: float,
) -> FloatND:
    """Inverse of aime_to_pia: convert PIA back to AIME.

    Three-bracket piecewise inverse.
    """
    aime_bracket_0 = pia / pia_conversion_rate_0
    aime_bracket_1 = aime_kink_0 + (pia - pia_kink_0) / pia_conversion_rate_1
    aime_bracket_2 = aime_kink_1 + (pia - pia_kink_1) / pia_conversion_rate_2

    return jnp.where(
        pia < pia_kink_0,
        aime_bracket_0,
        jnp.where(pia < pia_kink_1, aime_bracket_1, aime_bracket_2),
    )


def adjust_aime_di_dropout_inv(
    period: Period,
    aime: FloatND,
    di_dropout_scale: FloatND,
) -> FloatND:
    """Inverse of DI dropout adjustment."""
    return aime / di_dropout_scale[period]
