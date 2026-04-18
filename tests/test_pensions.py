"""Tests for pension functions, ported from struct-ret."""

import math

import jax.numpy as jnp

from aca_model.environment import pensions

ATOL = 0.01

# Struct-ret fixture: pension imputation coefficients at period 29, his=0.
# All other period/his entries are zero except period 29.
PW_IMP_INTERCEPT = jnp.zeros((30, 1))
PW_IMP_INTERCEPT = PW_IMP_INTERCEPT.at[29, 0].set(-50.0)

PW_IMP_PIA = jnp.zeros((30, 1))
PW_IMP_PIA = PW_IMP_PIA.at[29, 0].set(0.2)

PW_IMP_PIA_KINK_0_COEFF = jnp.zeros((30, 1))
PW_IMP_PIA_KINK_0_COEFF = PW_IMP_PIA_KINK_0_COEFF.at[29, 0].set(0.1)

PW_IMP_PIA_KINK_1_COEFF = jnp.zeros((30, 1))
PW_IMP_PIA_KINK_1_COEFF = PW_IMP_PIA_KINK_1_COEFF.at[29, 0].set(0.05)

PIA_IMP_KINK_0 = 9999.6 + jnp.zeros(30)
PIA_IMP_KINK_1 = 14359.9 + jnp.zeros(30)

FRACTION_RECEIVING = jnp.ones(30)

# Pension accrual coefficients
ACCRUAL_INTERCEPT = jnp.zeros((30, 1))
ACCRUAL_LOG_EARNINGS = jnp.full((30, 1), 0.5)
ACCRUAL_PROB_INTERCEPT = jnp.full(1, 0.1)
ACCRUAL_PROB_LOG_EARNINGS = jnp.zeros(1)
ACCRUAL_PROB_LOG_EARNINGS_SQ = jnp.zeros(1)

SURVIVAL_PROBS = jnp.ones(30) * 0.99
SURVIVAL_PROBS = SURVIVAL_PROBS.at[28].set(0.99)
SURVIVAL_PROBS = SURVIVAL_PROBS.at[29].set(0.98)


def test_pension_benefit_zero_pia() -> None:
    result = pensions.benefit(
        pia=jnp.array(0.0),
        period=29,
        his=0,
        imp_intercept=PW_IMP_INTERCEPT,
        imp_pia_coeff=PW_IMP_PIA,
        imp_pia_kink_0_coeff=PW_IMP_PIA_KINK_0_COEFF,
        imp_pia_kink_1_coeff=PW_IMP_PIA_KINK_1_COEFF,
        imp_kink_0=PIA_IMP_KINK_0,
        imp_kink_1=PIA_IMP_KINK_1,
        imp_fraction_receiving=FRACTION_RECEIVING,
    )
    # intercept=-50, pia_pred=0, kinks=0 → max(0, -50) = 0
    assert jnp.isclose(result, 0.0, atol=ATOL)


def test_pension_benefit_below_kink_0() -> None:
    result = pensions.benefit(
        pia=jnp.array(500.0),
        period=29,
        his=0,
        imp_intercept=PW_IMP_INTERCEPT,
        imp_pia_coeff=PW_IMP_PIA,
        imp_pia_kink_0_coeff=PW_IMP_PIA_KINK_0_COEFF,
        imp_pia_kink_1_coeff=PW_IMP_PIA_KINK_1_COEFF,
        imp_kink_0=PIA_IMP_KINK_0,
        imp_kink_1=PIA_IMP_KINK_1,
        imp_fraction_receiving=FRACTION_RECEIVING,
    )
    assert jnp.isclose(result, 500 * 0.2 - 50, atol=ATOL)


def test_pension_benefit_between_kinks() -> None:
    result = pensions.benefit(
        pia=jnp.array(12000.0),
        period=29,
        his=0,
        imp_intercept=PW_IMP_INTERCEPT,
        imp_pia_coeff=PW_IMP_PIA,
        imp_pia_kink_0_coeff=PW_IMP_PIA_KINK_0_COEFF,
        imp_pia_kink_1_coeff=PW_IMP_PIA_KINK_1_COEFF,
        imp_kink_0=PIA_IMP_KINK_0,
        imp_kink_1=PIA_IMP_KINK_1,
        imp_fraction_receiving=FRACTION_RECEIVING,
    )
    expected = 12000 * 0.2 + (12000 - 9999.6) * 0.1 - 50
    assert jnp.isclose(result, expected, atol=ATOL)


def test_pension_benefit_above_kink_1() -> None:
    result = pensions.benefit(
        pia=jnp.array(20000.0),
        period=29,
        his=0,
        imp_intercept=PW_IMP_INTERCEPT,
        imp_pia_coeff=PW_IMP_PIA,
        imp_pia_kink_0_coeff=PW_IMP_PIA_KINK_0_COEFF,
        imp_pia_kink_1_coeff=PW_IMP_PIA_KINK_1_COEFF,
        imp_kink_0=PIA_IMP_KINK_0,
        imp_kink_1=PIA_IMP_KINK_1,
        imp_fraction_receiving=FRACTION_RECEIVING,
    )
    expected = 20000 * 0.2 + (20000 - 9999.6) * 0.1 + (20000 - 14359.9) * 0.05 - 50
    assert jnp.isclose(result, expected, atol=ATOL)


def test_pension_accrual_no_income() -> None:
    result = pensions.accrual(
        labor_income=jnp.array(-1000.0),
        period=20,
        his=0,
        accrual_intercept=ACCRUAL_INTERCEPT,
        accrual_log_earnings=ACCRUAL_LOG_EARNINGS,
        accrual_prob_intercept=ACCRUAL_PROB_INTERCEPT,
        accrual_prob_log_earnings=ACCRUAL_PROB_LOG_EARNINGS,
        accrual_prob_log_earnings_sq=ACCRUAL_PROB_LOG_EARNINGS_SQ,
    )
    assert jnp.isclose(result, 0.0, atol=ATOL)


def test_pension_accrual_positive() -> None:
    result = pensions.accrual(
        labor_income=jnp.array(10000.0),
        period=20,
        his=0,
        accrual_intercept=ACCRUAL_INTERCEPT,
        accrual_log_earnings=ACCRUAL_LOG_EARNINGS,
        accrual_prob_intercept=ACCRUAL_PROB_INTERCEPT,
        accrual_prob_log_earnings=ACCRUAL_PROB_LOG_EARNINGS,
        accrual_prob_log_earnings_sq=ACCRUAL_PROB_LOG_EARNINGS_SQ,
    )
    lli = math.log(10000)
    prob = math.exp(0.1) / (1 + math.exp(0.1))
    expected = lli * 0.5 * prob * 10000
    assert jnp.isclose(result, expected, atol=ATOL)


def test_pension_wealth_next_accrual_only() -> None:
    lli = math.log(10000)
    prob = math.exp(0.1) / (1 + math.exp(0.1))
    accrual = lli * 0.5 * prob * 10000
    r = 0.03
    result = pensions.wealth_next_before_adjustment(
        pension_wealth=jnp.array(0.0),
        pension_benefit=jnp.array(0.0),
        pension_accrual=jnp.array(accrual),
        rate_of_return=r,
        unconditional_survival_prob=SURVIVAL_PROBS,
        period=28,
    )
    assert jnp.isclose(result, accrual / 0.99, atol=ATOL)


def test_pension_wealth_next_with_benefit() -> None:
    lli = math.log(10000)
    prob = math.exp(0.1) / (1 + math.exp(0.1))
    accrual = lli * 0.5 * prob * 10000
    r = 0.03
    result = pensions.wealth_next_before_adjustment(
        pension_wealth=jnp.array(3000.0),
        pension_benefit=jnp.array(2000.0),
        pension_accrual=jnp.array(accrual),
        rate_of_return=r,
        unconditional_survival_prob=SURVIVAL_PROBS,
        period=29,
    )
    expected = ((1 + r) * 3000 + accrual - 2000) / 0.98
    assert jnp.isclose(result, expected, atol=ATOL)


def test_convert_total_ben_to_pia_below_kink_0() -> None:
    """Round-trip: PIA below first kink recovers original PIA."""
    pia_input = jnp.array(500.0)
    mtr = jnp.array(0.2)

    pb = pensions.benefit(
        pia=pia_input,
        period=29,
        his=0,
        imp_intercept=PW_IMP_INTERCEPT,
        imp_pia_coeff=PW_IMP_PIA,
        imp_pia_kink_0_coeff=PW_IMP_PIA_KINK_0_COEFF,
        imp_pia_kink_1_coeff=PW_IMP_PIA_KINK_1_COEFF,
        imp_kink_0=PIA_IMP_KINK_0,
        imp_kink_1=PIA_IMP_KINK_1,
        imp_fraction_receiving=FRACTION_RECEIVING,
    )
    recovered = pensions.total_to_pia(
        pension_benefit=pb,
        pia=pia_input,
        period=29,
        his=0,
        marginal_tax_rate=mtr,
        imp_intercept=PW_IMP_INTERCEPT,
        imp_pia_coeff=PW_IMP_PIA,
        imp_pia_kink_0_coeff=PW_IMP_PIA_KINK_0_COEFF,
        imp_pia_kink_1_coeff=PW_IMP_PIA_KINK_1_COEFF,
        imp_kink_0=PIA_IMP_KINK_0,
        imp_kink_1=PIA_IMP_KINK_1,
    )
    assert jnp.isclose(recovered, pia_input, atol=ATOL)


def test_convert_total_ben_to_pia_between_kinks() -> None:
    """Round-trip: PIA between kinks recovers original PIA."""
    pia_input = jnp.array(12000.0)
    mtr = jnp.array(0.2)

    pb = pensions.benefit(
        pia=pia_input,
        period=29,
        his=0,
        imp_intercept=PW_IMP_INTERCEPT,
        imp_pia_coeff=PW_IMP_PIA,
        imp_pia_kink_0_coeff=PW_IMP_PIA_KINK_0_COEFF,
        imp_pia_kink_1_coeff=PW_IMP_PIA_KINK_1_COEFF,
        imp_kink_0=PIA_IMP_KINK_0,
        imp_kink_1=PIA_IMP_KINK_1,
        imp_fraction_receiving=FRACTION_RECEIVING,
    )
    recovered = pensions.total_to_pia(
        pension_benefit=pb,
        pia=pia_input,
        period=29,
        his=0,
        marginal_tax_rate=mtr,
        imp_intercept=PW_IMP_INTERCEPT,
        imp_pia_coeff=PW_IMP_PIA,
        imp_pia_kink_0_coeff=PW_IMP_PIA_KINK_0_COEFF,
        imp_pia_kink_1_coeff=PW_IMP_PIA_KINK_1_COEFF,
        imp_kink_0=PIA_IMP_KINK_0,
        imp_kink_1=PIA_IMP_KINK_1,
    )
    assert jnp.isclose(recovered, pia_input, atol=ATOL)


def test_convert_total_ben_to_pia_above_kink_1() -> None:
    """Round-trip: PIA above both kinks recovers original PIA."""
    pia_input = jnp.array(20000.0)
    mtr = jnp.array(0.2)

    pb = pensions.benefit(
        pia=pia_input,
        period=29,
        his=0,
        imp_intercept=PW_IMP_INTERCEPT,
        imp_pia_coeff=PW_IMP_PIA,
        imp_pia_kink_0_coeff=PW_IMP_PIA_KINK_0_COEFF,
        imp_pia_kink_1_coeff=PW_IMP_PIA_KINK_1_COEFF,
        imp_kink_0=PIA_IMP_KINK_0,
        imp_kink_1=PIA_IMP_KINK_1,
        imp_fraction_receiving=FRACTION_RECEIVING,
    )
    recovered = pensions.total_to_pia(
        pension_benefit=pb,
        pia=pia_input,
        period=29,
        his=0,
        marginal_tax_rate=mtr,
        imp_intercept=PW_IMP_INTERCEPT,
        imp_pia_coeff=PW_IMP_PIA,
        imp_pia_kink_0_coeff=PW_IMP_PIA_KINK_0_COEFF,
        imp_pia_kink_1_coeff=PW_IMP_PIA_KINK_1_COEFF,
        imp_kink_0=PIA_IMP_KINK_0,
        imp_kink_1=PIA_IMP_KINK_1,
    )
    assert jnp.isclose(recovered, pia_input, atol=ATOL)


def test_convert_total_ben_to_pia_zero_mtr() -> None:
    """Round-trip with zero marginal tax rate."""
    pia_input = jnp.array(8000.0)
    mtr = jnp.array(0.0)

    pb = pensions.benefit(
        pia=pia_input,
        period=29,
        his=0,
        imp_intercept=PW_IMP_INTERCEPT,
        imp_pia_coeff=PW_IMP_PIA,
        imp_pia_kink_0_coeff=PW_IMP_PIA_KINK_0_COEFF,
        imp_pia_kink_1_coeff=PW_IMP_PIA_KINK_1_COEFF,
        imp_kink_0=PIA_IMP_KINK_0,
        imp_kink_1=PIA_IMP_KINK_1,
        imp_fraction_receiving=FRACTION_RECEIVING,
    )
    recovered = pensions.total_to_pia(
        pension_benefit=pb,
        pia=pia_input,
        period=29,
        his=0,
        marginal_tax_rate=mtr,
        imp_intercept=PW_IMP_INTERCEPT,
        imp_pia_coeff=PW_IMP_PIA,
        imp_pia_kink_0_coeff=PW_IMP_PIA_KINK_0_COEFF,
        imp_pia_kink_1_coeff=PW_IMP_PIA_KINK_1_COEFF,
        imp_kink_0=PIA_IMP_KINK_0,
        imp_kink_1=PIA_IMP_KINK_1,
    )
    assert jnp.isclose(recovered, pia_input, atol=ATOL)
