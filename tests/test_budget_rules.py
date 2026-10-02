"""Budget, Social Security and pension rules at their production values.

Expected values come from a literal scalar transcription of struct-ret's
formulas (or of the draft's formula where struct-ret departs from it),
evaluated at the same inputs.
"""

import inspect
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from helpers import production_rules as pr

from aca_model.agent import labor_market
from aca_model.agent.labor_market import LaborSupply, SpousalIncome
from aca_model.baseline.regimes._common import (
    REGIME_SPECS,
    _select_aime_law,
    build_model_constraints,
)
from aca_model.environment import pensions, social_security, taxes
from aca_model.environment.social_security import ClaimedSS

PIA_AIME_GRID = jnp.asarray(pr.PIA_AIME_GRID)
PIA_TABLE = jnp.asarray(pr.PIA_TABLE)
EARLY_RET_ADJUSTMENT = jnp.asarray(pr.EARLY_RET_ADJUSTMENT)
EARNINGS_TEST_CREDITED_BACK = jnp.asarray(pr.EARNINGS_TEST_CREDITED_BACK)
RATIO_LOWEST_EARNINGS = jnp.asarray(pr.RATIO_LOWEST_EARNINGS)
DI_DROPOUT_SCALE = jnp.asarray(pr.DI_DROPOUT_SCALE)
DI_DROPOUT_NEXT_PERIOD_RATIO = jnp.asarray(pr.DI_DROPOUT_NEXT_PERIOD_RATIO)


def _shift_one_period_forward(values: tuple) -> jnp.ndarray:
    """Return the period-indexed array whose row `t` carries the value at `t + 1`."""
    arr = jnp.asarray(values)
    return jnp.concatenate([arr[1:], arr[-1:]], axis=0)


IMP_NEXT_PERIOD = {
    "imp_intercept_next_period": _shift_one_period_forward(pr.IMP_INTERCEPT),
    "imp_pia_coeff_next_period": _shift_one_period_forward(pr.IMP_PIA_COEFF),
    "imp_pia_kink_0_coeff_next_period": _shift_one_period_forward(
        pr.IMP_PIA_KINK_0_COEFF
    ),
    "imp_pia_kink_1_coeff_next_period": _shift_one_period_forward(
        pr.IMP_PIA_KINK_1_COEFF
    ),
    "imp_kink_0_next_period": _shift_one_period_forward(pr.IMP_KINK_0),
    "imp_kink_1_next_period": _shift_one_period_forward(pr.IMP_KINK_1),
}

SS_RULES = {
    "pia_table": PIA_TABLE,
    "pia_aime_grid": PIA_AIME_GRID,
    "normal_retirement_age": jnp.int32(pr.NORMAL_RETIREMENT_AGE),
    "early_ret_adjustment": EARLY_RET_ADJUSTMENT,
    "earnings_test_credited_back": EARNINGS_TEST_CREDITED_BACK,
    "earnings_test_repealed_age": jnp.int32(pr.EARNINGS_TEST_REPEALED_AGE),
    "aime_accrual_factor": jnp.asarray(pr.AIME_ACCRUAL_FACTOR),
    "aggregate_wage_growth": jnp.asarray(pr.AGGREGATE_WAGE_GROWTH),
    "aime_last_age_with_indexing": jnp.int32(pr.AIME_LAST_AGE_WITH_INDEXING),
    "aime_kink_2": jnp.asarray(pr.AIME_KINK_2),
    "ratio_lowest_earnings": RATIO_LOWEST_EARNINGS,
    "medicare_age": jnp.int32(pr.MEDICARE_AGE),
    "di_dropout_scale": DI_DROPOUT_SCALE,
    "di_dropout_next_period_ratio": DI_DROPOUT_NEXT_PERIOD_RATIO,
}


def _period(age: int) -> jnp.ndarray:
    return jnp.int32(age - pr.START_AGE)


def _call(func: Callable[..., Any], **kwargs: Any) -> Any:
    """Call `func` with the subset of `kwargs` its signature names."""
    names = inspect.signature(func).parameters
    return func(**{k: v for k, v in kwargs.items() if k in names})


@pytest.mark.parametrize(
    ("age", "pia_unadjusted", "pia_adjusted", "full_benefit_next", "expected"),
    [
        (62, 18479.36, 17324.4, 35680.75116959574, 18117.163555158222),
        (63, 18479.36, 17057.87076922421, 37436.31858689668, 18035.38135778553),
        (66, 18479.36, 19957.7088, 44113.51778285309, 19000.883559951508),
        (68, 24710.04, 26414.180689655175, 58432.219638177565, 25466.839034926863),
    ],
)
def test_total_to_pia_keeps_social_security_plus_imputed_pension_total(
    age: int,
    pia_unadjusted: float,
    pia_adjusted: float,
    full_benefit_next: float,
    expected: float,
) -> None:
    """The PIA carried into the next period reproduces the claim-adjusted total.

    For a retiree-HIS household at a 18.19% marginal tax rate,
    `PIA* + (1 - τ) pbmax_{t+1}(PIA*) = PIA_adj + (1 - τ) pbmax_{t+1}(PIA_unadj)`.
    """
    result = pensions.total_to_pia(
        pia_adjusted_next_period=jnp.asarray(pia_adjusted),
        pia_unadjusted_next_period=jnp.asarray(pia_unadjusted),
        full_benefit_next_period=jnp.asarray(full_benefit_next),
        target_his=jnp.int32(0),
        period=_period(age),
        marginal_tax_rate=jnp.asarray(0.1819),
        **IMP_NEXT_PERIOD,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_total_to_pia_without_claim_adjustment_returns_unadjusted_pia() -> None:
    """Without a claim adjustment or credit the carried PIA is the accrued PIA."""
    result = pensions.total_to_pia(
        pia_adjusted_next_period=jnp.asarray(5000.0),
        pia_unadjusted_next_period=jnp.asarray(5000.0),
        full_benefit_next_period=jnp.asarray(0.0),
        target_his=jnp.int32(0),
        period=_period(62),
        marginal_tax_rate=jnp.asarray(0.1819),
        **IMP_NEXT_PERIOD,
    )
    np.testing.assert_allclose(result, 5000.0, rtol=1e-12)


@pytest.mark.parametrize(
    ("labor_supply", "ss_benefit", "expected"),
    [
        # Claims at 62 and does not work: nothing is withheld.
        (LaborSupply.do_not_work, 13859.52, 0.0),
        # Claims at 62 and earns 10,000, below the 15,480 threshold.
        (LaborSupply.h1000, 13859.52, 0.0),
        # Claims at 62 and earns 30,000: 7,260 of the 13,859.52 benefit withheld.
        (LaborSupply.h1000, 6599.52, 0.5238276650273602),
    ],
)
def test_benefit_withheld_fraction_counts_only_earnings_test_withholding(
    labor_supply: int, ss_benefit: float, expected: float
) -> None:
    """The withheld fraction is the earnings-test share of the claimed benefit.

    AIME is credited back only for benefits the earnings test withheld, not
    for the early-claiming reduction. AIME 40,000 (PIA 18,479.36), age 62.
    """
    result = _call(
        social_security.benefit_withheld_fraction,
        pia=jnp.asarray(18479.36),
        ss_benefit=jnp.asarray(ss_benefit),
        age=jnp.int32(62),
        period=_period(62),
        claim_ss=jnp.int32(ClaimedSS.yes),
        claimed_ss=jnp.int32(ClaimedSS.no),
        labor_supply=jnp.int32(labor_supply),
        normal_retirement_age=jnp.int32(pr.NORMAL_RETIREMENT_AGE),
        early_ret_adjustment=EARLY_RET_ADJUSTMENT,
    )
    np.testing.assert_allclose(result, expected, atol=1e-6)


SPOUSAL_INCOME_TABLE = jnp.zeros((45, 2, 3))
SPOUSAL_INCOME_TABLE = SPOUSAL_INCOME_TABLE.at[4, :, SpousalIncome.married_has_inc].set(
    jnp.asarray([20299.103515625, 21513.8984375])
)
SPOUSAL_INCOME_TABLE = SPOUSAL_INCOME_TABLE.at[
    11, :, SpousalIncome.married_has_inc
].set(jnp.asarray([22837.513671875, 23848.978515625]))


@pytest.mark.parametrize(
    ("age", "good_health", "spousal_income", "expected"),
    [
        (55, 0, SpousalIncome.married_has_inc, 20299.103515625),
        (55, 1, SpousalIncome.married_has_inc, 21513.8984375),
        (62, 1, SpousalIncome.married_has_inc, 23848.978515625),
        (62, 1, SpousalIncome.married_no_inc, 0.0),
        (62, 0, SpousalIncome.single, 0.0),
    ],
)
def test_spousal_income_amount_varies_with_age_and_own_health(
    age: int, good_health: int, spousal_income: int, expected: float
) -> None:
    """Spousal income is read from the age × own-health × spousal-income table."""
    result = labor_market.spousal_income_amount(
        period=_period(age),
        good_health=jnp.int32(good_health),
        spousal_income=jnp.int32(spousal_income),
        spousal_income_amounts=SPOUSAL_INCOME_TABLE,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize(
    ("labor_income", "expected"),
    [
        # Indexed by 0.8%, no earnings: 40,320 times the DI continuity ratio.
        (0.0, 40531.764705882364),
        # Indexed and accrues 12,000/35 of earnings, below the SGA threshold.
        (12000.0, 40876.42256902762),
    ],
)
def test_disabled_aime_accrues_indexing_and_earnings(
    labor_income: float, expected: float
) -> None:
    """A disabled agent's AIME is indexed and accrues earnings like anyone's.

    The DI continuity ratio then scales the accrued AIME. Age 55, AIME 40,000.
    """
    result = social_security.next_aime_disabled_plain(
        aime=jnp.asarray(40000.0),
        labor_income=jnp.asarray(labor_income),
        period=_period(55),
        age=jnp.int32(55),
        health=jnp.int32(0),
        benefit_withheld_fraction=jnp.asarray(0.0),
        earnings_test_credited_back=EARNINGS_TEST_CREDITED_BACK,
        earnings_test_repealed_age=jnp.int32(pr.EARNINGS_TEST_REPEALED_AGE),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime_accrual_factor=jnp.asarray(pr.AIME_ACCRUAL_FACTOR),
        aggregate_wage_growth=jnp.asarray(pr.AGGREGATE_WAGE_GROWTH),
        aime_last_age_with_indexing=jnp.int32(pr.AIME_LAST_AGE_WITH_INDEXING),
        aime_kink_2=jnp.asarray(pr.AIME_KINK_2),
        ratio_lowest_earnings=RATIO_LOWEST_EARNINGS,
        medicare_age=jnp.int32(pr.MEDICARE_AGE),
        di_dropout_scale=DI_DROPOUT_SCALE,
        di_dropout_next_period_ratio=DI_DROPOUT_NEXT_PERIOD_RATIO,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-10)


@pytest.mark.parametrize(
    ("age", "aime", "labor_income"),
    [
        (70, 40000.0, 60000.0),
        (71, 131305.0, 0.0),
    ],
)
def test_aime_is_frozen_once_claiming_is_forced(
    age: int, aime: float, labor_income: float
) -> None:
    """From the forced-claim age on, AIME neither accrues nor is capped."""
    law = _select_aime_law(REGIME_SPECS["retiree_oamc_forced_canwork"])
    result = _call(
        law,
        aime=jnp.asarray(aime),
        labor_income=jnp.asarray(labor_income),
        period=_period(age),
        age=jnp.int32(age),
        benefit_withheld_fraction=jnp.asarray(0.0),
        **SS_RULES,
    )
    np.testing.assert_allclose(result, aime, rtol=1e-12)


def test_accrued_aime_is_capped_at_the_taxable_maximum() -> None:
    """Accrual from earnings cannot lift AIME above the taxable maximum.

    Age 66, AIME 116,000, earnings 117,000: AIME accrues to the 117,000 cap,
    whose PIA is 33,260.04.
    """
    result = social_security.pia_unadjusted_next_period(
        aime=jnp.asarray(116000.0),
        labor_income=jnp.asarray(117000.0),
        period=_period(66),
        age=jnp.int32(66),
        pia_table=PIA_TABLE,
        pia_aime_grid=PIA_AIME_GRID,
        aime_accrual_factor=jnp.asarray(pr.AIME_ACCRUAL_FACTOR),
        aggregate_wage_growth=jnp.asarray(pr.AGGREGATE_WAGE_GROWTH),
        aime_last_age_with_indexing=jnp.int32(pr.AIME_LAST_AGE_WITH_INDEXING),
        aime_kink_2=jnp.asarray(pr.AIME_KINK_2),
        ratio_lowest_earnings=RATIO_LOWEST_EARNINGS,
    )
    np.testing.assert_allclose(result, 33260.04, atol=0.01)


@pytest.mark.parametrize(
    ("age", "his", "expected"),
    [
        (70, 0, -190.86435823930114),
        (70, 1, 696.7748524892276),
        (71, 0, 0.0),
        (71, 1, 0.0),
    ],
)
def test_pension_accrual_stops_at_the_pension_claiming_age(
    age: int, his: int, expected: float
) -> None:
    """Pension wealth accrues from earnings through age 70 and not at 71.

    Earnings 40,000.
    """
    result = _call(
        pensions.accrual,
        labor_income=jnp.asarray(40000.0),
        period=_period(age),
        age=jnp.int32(age),
        his=jnp.int32(his),
        accrual_intercept=jnp.asarray(pr.ACCRUAL_INTERCEPT),
        accrual_log_earnings=jnp.asarray(pr.ACCRUAL_LOG_EARNINGS),
        accrual_prob_intercept=jnp.asarray(pr.ACCRUAL_PROB_INTERCEPT),
        accrual_prob_log_earnings=jnp.asarray(pr.ACCRUAL_PROB_LOG_EARNINGS),
        accrual_prob_log_earnings_sq=jnp.asarray(pr.ACCRUAL_PROB_LOG_EARNINGS_SQ),
    )
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-8)


SINGLE_FLOOR = 1597.09
MARRIED_FLOOR = SINGLE_FLOOR * 2.0**0.7


@pytest.mark.parametrize(
    ("consumption", "expected"),
    [
        (SINGLE_FLOOR, False),
        (MARRIED_FLOOR, True),
    ],
)
def test_married_household_cannot_consume_below_its_own_floor(
    consumption: float, expected: bool
) -> None:
    """A married household at the transfer floor consumes exactly its own floor.

    Cash on hand 500 is topped up to the married floor; consuming the lower
    single floor is infeasible.
    """
    constraints = build_model_constraints(solver="brute_force")
    feasible = all(
        bool(
            _call(
                constraint,
                consumption_dollars=jnp.asarray(consumption),
                cash_on_hand=jnp.asarray(500.0),
                consumption_dollars_floor=jnp.asarray(MARRIED_FLOOR),
                hcc_persistent=jnp.asarray(0.0),
                hcc_transitory=jnp.asarray(0.0),
            )
        )
        for constraint in constraints.values()
    )
    assert feasible is expected


@pytest.mark.parametrize(
    ("gross_income", "labor_income", "expected_surtax"),
    [
        # Unearned income 250,000: 3.8% on the 50,000 above 200,000.
        (250000.0, 0.0, 1900.0),
        # Earnings 250,000: 0.9% on the 50,000 above 200,000.
        (250000.0, 250000.0, 450.0),
        # Earnings 250,000 and unearned income 250,000.
        (500000.0, 250000.0, 2350.0),
        # Unearned income 150,000 and earnings 150,000: both below the thresholds.
        (300000.0, 150000.0, 0.0),
    ],
)
def test_aca_surtaxes_reduce_after_tax_income(
    gross_income: float, labor_income: float, expected_surtax: float
) -> None:
    """The ACA adds a 3.8% tax on unearned and 0.9% on earned income above 200,000."""
    result = taxes.after_tax_income_with_aca_surtaxes(
        after_tax_income_before_aca_surtaxes=jnp.asarray(100000.0),
        gross_income=jnp.asarray(gross_income),
        labor_income=jnp.asarray(labor_income),
    )
    np.testing.assert_allclose(result, 100000.0 - expected_surtax, rtol=1e-12)


@pytest.mark.parametrize(
    ("gross_income", "labor_income", "expected_surtax"),
    [
        (250000.0, 0.0, 1900.0),
        (250000.0, 250000.0, 0.0),
    ],
)
def test_aca_investment_surtax_alone_spares_earnings(
    gross_income: float, labor_income: float, expected_surtax: float
) -> None:
    """Without the payroll surtax only unearned income above 200,000 is taxed."""
    result = taxes.after_tax_income_with_aca_investment_surtax(
        after_tax_income_before_aca_surtaxes=jnp.asarray(100000.0),
        gross_income=jnp.asarray(gross_income),
        labor_income=jnp.asarray(labor_income),
    )
    np.testing.assert_allclose(result, 100000.0 - expected_surtax, rtol=1e-12)
