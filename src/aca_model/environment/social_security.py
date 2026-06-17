"""Social Security types and functions: AIME, PIA, earnings test, benefits.

Ported from struct-ret/src/model/baseline/soc_sec_pensions_taxes.py.

PIA is pre-computed on a 4-point grid (the piecewise-linear kink points) in aca-data
and looked up via `jnp.interp` — same pattern as `predicted_hcc_insurer`. This
eliminates 7 PIA formula constants from the DAG.
"""

import jax.numpy as jnp
from lcm import categorical
from lcm.typing import (
    Age,
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    IntND,
    Period,
    ScalarFloat,
    ScalarInt,
)

from aca_model.agent.labor_market import LaborSupply


@categorical(ordered=False)
class ClaimedSS:
    no: ScalarInt
    yes: ScalarInt


def next_claimed_ss(
    claim_ss: DiscreteAction,
    claimed_ss: DiscreteState,
) -> DiscreteState:
    """Absorbing transition: once claimed, always claimed."""
    return jnp.maximum(claim_ss, claimed_ss)


def enter_claimed_ss() -> DiscreteState:
    """Initial claimed_ss when entering the SS eligibility window."""
    return jnp.int32(ClaimedSS.no)


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


def find_aime(
    pia: FloatND,
    pia_table: FloatND,
    pia_aime_grid: FloatND,
) -> ContinuousState:
    """Recover the AIME that produces a given PIA — the exact inverse of `pia`.

    `pia(aime) = jnp.interp(aime, pia_aime_grid, pia_table)` is piecewise-linear
    and strictly increasing, so its inverse interpolates the swapped table.
    """
    return jnp.interp(pia, pia_table, pia_aime_grid)


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
    age: Age,
    period: Period,
    claim_ss: DiscreteAction,
    claimed_ss: DiscreteState,
    labor_supply: DiscreteAction,
    labor_income: FloatND,
    early_ret_adjustment: FloatND,
    normal_retirement_age: ScalarInt,
    earnings_test_threshold: FloatND,
    earnings_test_fraction: FloatND,
    earnings_test_repealed_age: ScalarInt,
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
    age: Age,
    period: Period,
    claim_ss: DiscreteAction,
    claimed_ss: DiscreteState,
    health: DiscreteState,
    labor_supply: DiscreteAction,
    labor_income: FloatND,
    early_ret_adjustment: FloatND,
    normal_retirement_age: ScalarInt,
    earnings_test_threshold: FloatND,
    earnings_test_fraction: FloatND,
    earnings_test_repealed_age: ScalarInt,
    ssdi_substantial_gainful_activity: ScalarFloat,
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
    ssdi_substantial_gainful_activity: ScalarFloat,
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
    age: Age,
    period: Period,
    ss: IntND,
    work: BoolND,
    labor_income: FloatND,
    early_ret_adjustment: FloatND,
    normal_retirement_age: ScalarInt,
    earnings_test_threshold: FloatND,
    earnings_test_fraction: FloatND,
    earnings_test_repealed_age: ScalarInt,
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
    age: Age,
    claim_ss: DiscreteAction,
    claimed_ss: DiscreteState,
    normal_retirement_age: ScalarInt,
    early_ret_adjustment: FloatND,
    benefit_withheld_fraction: FloatND,
    earnings_test_credited_back: FloatND,
    earnings_test_repealed_age: ScalarInt,
    pia_table: FloatND,
    pia_aime_grid: FloatND,
    aime_accrual_factor: ScalarFloat,
    aggregate_wage_growth: ScalarFloat,
    aime_last_age_with_indexing: ScalarInt,
    aime_kink_2: ScalarFloat,
    ratio_lowest_earnings: FloatND,
) -> ContinuousState:
    """Compute next period's AIME given labor earnings and the claim decision.

    Steps:
    1. Accrue AIME from labor income (indexing, taxable cap, lowest-year drop).
    2. Credit back for earnings-test withholding (PIA round-trip).
    3. Bake the claim-age actuarial factor into AIME (`_apply_claim_adjustment`),
       so the carried AIME permanently encodes the early-retirement reduction or
       the delayed-retirement credit. The flat-PIA benefit read off this AIME is
       then correct at every later age, including the forced regimes.
    """
    credited_pia = _accrue_and_credit_back_pia(
        aime=aime,
        labor_income=labor_income,
        period=period,
        age=age,
        benefit_withheld_fraction=benefit_withheld_fraction,
        earnings_test_credited_back=earnings_test_credited_back,
        earnings_test_repealed_age=earnings_test_repealed_age,
        pia_table=pia_table,
        pia_aime_grid=pia_aime_grid,
        aime_accrual_factor=aime_accrual_factor,
        aggregate_wage_growth=aggregate_wage_growth,
        aime_last_age_with_indexing=aime_last_age_with_indexing,
        aime_kink_2=aime_kink_2,
        ratio_lowest_earnings=ratio_lowest_earnings,
    )

    adjusted_pia = _apply_claim_adjustment(
        pia=credited_pia,
        period=period,
        age=age,
        claim_ss=claim_ss,
        claimed_ss=claimed_ss,
        normal_retirement_age=normal_retirement_age,
        early_ret_adjustment=early_ret_adjustment,
    )
    # The extended `pia_table`/`pia_aime_grid` reach above the taxable max so a
    # delayed-retirement credit on a top earner's PIA round-trips to an AIME
    # above `aime_kink_2` rather than clamping there. `_accrue_aime` already
    # capped the labor-earnings base at the taxable max; the actuarial credit is
    # the only thing carried beyond it.
    return jnp.interp(adjusted_pia, pia_table, pia_aime_grid)


def next_aime_plain(
    aime: ContinuousState,
    labor_income: FloatND,
    period: Period,
    age: Age,
    benefit_withheld_fraction: FloatND,
    earnings_test_credited_back: FloatND,
    earnings_test_repealed_age: ScalarInt,
    pia_table: FloatND,
    pia_aime_grid: FloatND,
    aime_accrual_factor: ScalarFloat,
    aggregate_wage_growth: ScalarFloat,
    aime_last_age_with_indexing: ScalarInt,
    aime_kink_2: ScalarFloat,
    ratio_lowest_earnings: FloatND,
) -> ContinuousState:
    """Compute next period's AIME without any claim-age adjustment.

    Used by post-65 `ss=forced` regimes, where the agent cannot choose when to
    claim and so carries no `claim_ss` action / `claimed_ss` state. A forced
    claimant who claimed early already has the actuarial reduction baked into
    the AIME carried in from the `ss=choose` regime; plain accrual preserves it.
    A forced claimant who never claimed early keeps a pristine AIME, so the
    flat-PIA benefit equals the full PIA, which is correct.
    """
    credited_pia = _accrue_and_credit_back_pia(
        aime=aime,
        labor_income=labor_income,
        period=period,
        age=age,
        benefit_withheld_fraction=benefit_withheld_fraction,
        earnings_test_credited_back=earnings_test_credited_back,
        earnings_test_repealed_age=earnings_test_repealed_age,
        pia_table=pia_table,
        pia_aime_grid=pia_aime_grid,
        aime_accrual_factor=aime_accrual_factor,
        aggregate_wage_growth=aggregate_wage_growth,
        aime_last_age_with_indexing=aime_last_age_with_indexing,
        aime_kink_2=aime_kink_2,
        ratio_lowest_earnings=ratio_lowest_earnings,
    )
    accrued_aime = jnp.interp(credited_pia, pia_table, pia_aime_grid)
    return jnp.minimum(accrued_aime, aime_kink_2)


def pia_unadjusted_next_period(
    aime: ContinuousState,
    labor_income: FloatND,
    period: Period,
    age: Age,
    pia_table: FloatND,
    pia_aime_grid: FloatND,
    aime_accrual_factor: ScalarFloat,
    aggregate_wage_growth: ScalarFloat,
    aime_last_age_with_indexing: ScalarInt,
    aime_kink_2: ScalarFloat,
    ratio_lowest_earnings: FloatND,
) -> FloatND:
    """Next-period PIA from pure labor accrual, before any claim adjustment.

    Pension imputation reads this channel: French & Jones (2011) deliberately
    impute pension wealth from the unadjusted PIA, so the claim-age reduction
    or credit baked into `next_aime` must not feed the pension node.
    """
    accrued_aime = _accrue_aime(
        aime=aime,
        labor_income=labor_income,
        period=period,
        age=age,
        aime_accrual_factor=aime_accrual_factor,
        aggregate_wage_growth=aggregate_wage_growth,
        aime_last_age_with_indexing=aime_last_age_with_indexing,
        aime_kink_2=aime_kink_2,
        ratio_lowest_earnings=ratio_lowest_earnings,
    )
    return jnp.interp(accrued_aime, pia_aime_grid, pia_table)


def _accrue_aime(
    *,
    aime: ContinuousState,
    labor_income: FloatND,
    period: Period,
    age: Age,
    aime_accrual_factor: ScalarFloat,
    aggregate_wage_growth: ScalarFloat,
    aime_last_age_with_indexing: ScalarInt,
    aime_kink_2: ScalarFloat,
    ratio_lowest_earnings: FloatND,
) -> FloatND:
    """Accrue AIME from labor income (indexing, taxable cap, lowest-year drop)."""
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
    return capped_aime + accrual


def _accrue_and_credit_back_pia(
    *,
    aime: ContinuousState,
    labor_income: FloatND,
    period: Period,
    age: Age,
    benefit_withheld_fraction: FloatND,
    earnings_test_credited_back: FloatND,
    earnings_test_repealed_age: ScalarInt,
    pia_table: FloatND,
    pia_aime_grid: FloatND,
    aime_accrual_factor: ScalarFloat,
    aggregate_wage_growth: ScalarFloat,
    aime_last_age_with_indexing: ScalarInt,
    aime_kink_2: ScalarFloat,
    ratio_lowest_earnings: FloatND,
) -> FloatND:
    """Accrue AIME from labor income, then credit it back for earnings-test withholding.

    Returns the PIA of the accrued, credited-back AIME. The credit-back raises
    the PIA to compensate for benefits withheld under the earnings test, before
    any claim-age actuarial bake. The shared prefix of every AIME law of motion.
    """
    new_aime = _accrue_aime(
        aime=aime,
        labor_income=labor_income,
        period=period,
        age=age,
        aime_accrual_factor=aime_accrual_factor,
        aggregate_wage_growth=aggregate_wage_growth,
        aime_last_age_with_indexing=aime_last_age_with_indexing,
        aime_kink_2=aime_kink_2,
        ratio_lowest_earnings=ratio_lowest_earnings,
    )

    credit = jnp.where(
        age < earnings_test_repealed_age,
        earnings_test_credited_back[period] * benefit_withheld_fraction,
        0.0,
    )
    accrued_pia = jnp.interp(new_aime, pia_aime_grid, pia_table)
    return accrued_pia * (1.0 + credit)


def _apply_claim_adjustment(
    *,
    pia: FloatND,
    period: Period,
    age: Age,
    claim_ss: DiscreteAction,
    claimed_ss: DiscreteState,
    normal_retirement_age: ScalarInt,
    early_ret_adjustment: FloatND,
) -> FloatND:
    """Bake the per-year claim-age actuarial factor into the PIA.

    `early_ret_adjustment` holds the cumulative SSA factors (0.75 at 62 … 1.0 at
    NRA … 1.32 at 70). The per-year factor that compounds to those cumulative
    levels is the ratio of neighbouring entries; applying it each period from the
    claim age toward the cumulative target reconstructs the full permanent
    adjustment (French & Jones 2011). The cases:

    - early: already claimed and below NRA — multiply by
      `early_ret_adjustment[age] / early_ret_adjustment[age + 1]` (< 1).
    - delayed: not yet claimed and at/above NRA — multiply by
      `early_ret_adjustment[age + 1] / early_ret_adjustment[age]` (≥ 1; the
      data-prep clamp makes it 1.0 once age reaches the delayed-credit ceiling).
    - otherwise: factor 1.0 (no adjustment).
    """
    claimed = jnp.maximum(claim_ss, claimed_ss) > 0
    cumulative_this = early_ret_adjustment[period]
    cumulative_next = early_ret_adjustment[period + 1]

    is_early = claimed & (age < normal_retirement_age)
    is_delayed = (~claimed) & (age >= normal_retirement_age)

    factor = jnp.where(
        is_early,
        cumulative_this / cumulative_next,
        jnp.where(is_delayed, cumulative_next / cumulative_this, 1.0),
    )
    return pia * factor


def next_aime_disabled(
    aime: ContinuousState,
    labor_income: FloatND,
    period: Period,
    age: Age,
    health: DiscreteState,
    claim_ss: DiscreteAction,
    claimed_ss: DiscreteState,
    normal_retirement_age: ScalarInt,
    early_ret_adjustment: FloatND,
    benefit_withheld_fraction: FloatND,
    earnings_test_credited_back: FloatND,
    earnings_test_repealed_age: ScalarInt,
    pia_table: FloatND,
    pia_aime_grid: FloatND,
    aime_accrual_factor: ScalarFloat,
    aggregate_wage_growth: ScalarFloat,
    aime_last_age_with_indexing: ScalarInt,
    aime_kink_2: ScalarFloat,
    ratio_lowest_earnings: FloatND,
    medicare_age: ScalarInt,
    di_dropout_scale: FloatND,
    di_dropout_next_period_ratio: FloatND,
) -> ContinuousState:
    """AIME transition for pre-65 regimes handling both disabled and non-disabled.

    Non-disabled: standard AIME accrual from labor income, earnings-test
    credit-back, and the claim-age actuarial bake (early reduction for claimants
    below NRA — pre-65 never reaches the delayed-credit window).
    Disabled: maintain PIA continuity across ages by adjusting AIME so the DI
    dropout-year scale factor change doesn't alter the benefit. The disabled path
    reads the un-baked AIME — DI benefits are routed only when never claimed, so
    the claim adjustment never touches them.
    At Medicare transition, stores the dropout-adjusted AIME (switching to OA).
    """
    credited_pia = _accrue_and_credit_back_pia(
        aime=aime,
        labor_income=labor_income,
        period=period,
        age=age,
        benefit_withheld_fraction=benefit_withheld_fraction,
        earnings_test_credited_back=earnings_test_credited_back,
        earnings_test_repealed_age=earnings_test_repealed_age,
        pia_table=pia_table,
        pia_aime_grid=pia_aime_grid,
        aime_accrual_factor=aime_accrual_factor,
        aggregate_wage_growth=aggregate_wage_growth,
        aime_last_age_with_indexing=aime_last_age_with_indexing,
        aime_kink_2=aime_kink_2,
        ratio_lowest_earnings=ratio_lowest_earnings,
    )

    adjusted_pia = _apply_claim_adjustment(
        pia=credited_pia,
        period=period,
        age=age,
        claim_ss=claim_ss,
        claimed_ss=claimed_ss,
        normal_retirement_age=normal_retirement_age,
        early_ret_adjustment=early_ret_adjustment,
    )
    regular = jnp.minimum(
        jnp.interp(adjusted_pia, pia_table, pia_aime_grid),
        aime_kink_2,
    )

    return _select_disabled_or_regular(
        aime=aime,
        regular=regular,
        period=period,
        age=age,
        health=health,
        medicare_age=medicare_age,
        di_dropout_scale=di_dropout_scale,
        di_dropout_next_period_ratio=di_dropout_next_period_ratio,
    )


def next_aime_disabled_plain(
    aime: ContinuousState,
    labor_income: FloatND,
    period: Period,
    age: Age,
    health: DiscreteState,
    benefit_withheld_fraction: FloatND,
    earnings_test_credited_back: FloatND,
    earnings_test_repealed_age: ScalarInt,
    pia_table: FloatND,
    pia_aime_grid: FloatND,
    aime_accrual_factor: ScalarFloat,
    aggregate_wage_growth: ScalarFloat,
    aime_last_age_with_indexing: ScalarInt,
    aime_kink_2: ScalarFloat,
    ratio_lowest_earnings: FloatND,
    medicare_age: ScalarInt,
    di_dropout_scale: FloatND,
    di_dropout_next_period_ratio: FloatND,
) -> ContinuousState:
    """AIME transition for pre-65 `ss=inelig` regimes, without claim adjustment.

    Non-disabled: standard AIME accrual from labor income and earnings-test
    credit-back. SS-ineligible agents cannot claim, so the regime carries no
    `claim_ss` action / `claimed_ss` state and the AIME stays unbaked.
    Disabled: maintain PIA continuity across ages by adjusting AIME so the DI
    dropout-year scale factor change doesn't alter the benefit. At the Medicare
    transition, stores the dropout-adjusted AIME (switching to OA).
    """
    credited_pia = _accrue_and_credit_back_pia(
        aime=aime,
        labor_income=labor_income,
        period=period,
        age=age,
        benefit_withheld_fraction=benefit_withheld_fraction,
        earnings_test_credited_back=earnings_test_credited_back,
        earnings_test_repealed_age=earnings_test_repealed_age,
        pia_table=pia_table,
        pia_aime_grid=pia_aime_grid,
        aime_accrual_factor=aime_accrual_factor,
        aggregate_wage_growth=aggregate_wage_growth,
        aime_last_age_with_indexing=aime_last_age_with_indexing,
        aime_kink_2=aime_kink_2,
        ratio_lowest_earnings=ratio_lowest_earnings,
    )
    regular = jnp.minimum(
        jnp.interp(credited_pia, pia_table, pia_aime_grid),
        aime_kink_2,
    )

    return _select_disabled_or_regular(
        aime=aime,
        regular=regular,
        period=period,
        age=age,
        health=health,
        medicare_age=medicare_age,
        di_dropout_scale=di_dropout_scale,
        di_dropout_next_period_ratio=di_dropout_next_period_ratio,
    )


def _select_disabled_or_regular(
    *,
    aime: ContinuousState,
    regular: FloatND,
    period: Period,
    age: Age,
    health: DiscreteState,
    medicare_age: ScalarInt,
    di_dropout_scale: FloatND,
    di_dropout_next_period_ratio: FloatND,
) -> ContinuousState:
    """Route the disabled DI-continuity AIME against the non-disabled AIME.

    The disabled path reads the un-baked `aime`, scaling it so the DI dropout-year
    factor change leaves the benefit unchanged. At the Medicare transition it
    switches to the dropout-adjusted AIME (OA from then on).
    """
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
