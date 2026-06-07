"""Pension functions: benefit imputation, pension accrual, wealth evolution.

Ported from struct-ret/src/model/baseline/soc_sec_pensions_taxes.py.
"""

import jax.numpy as jnp
from lcm.typing import ContinuousState, FloatND, IntND, Period, ScalarFloat


def full_benefit(
    pia: FloatND,
    period: Period,
    his: IntND,
    imp_intercept: FloatND,
    imp_pia_coeff: FloatND,
    imp_pia_kink_0_coeff: FloatND,
    imp_pia_kink_1_coeff: FloatND,
    imp_kink_0: FloatND,
    imp_kink_1: FloatND,
) -> FloatND:
    """Maximum pension benefit `pbmax_t` (French & Jones 2011, eq. D.2).

    The PIA-imputed benefit for an individual with full pension access, using
    age × HIS-specific coefficients with two kink points. This is *before* the
    fraction-receiving adjustment — `pbmax`, not the received benefit `pb`.
    """
    intercept = imp_intercept[period, his]
    pia_pred = imp_pia_coeff[period, his] * pia

    kink_0_adj = imp_pia_kink_0_coeff[period, his] * jnp.maximum(
        0.0, pia - imp_kink_0[period]
    )
    kink_1_adj = imp_pia_kink_1_coeff[period, his] * jnp.maximum(
        0.0, pia - imp_kink_1[period]
    )

    return jnp.maximum(0.0, intercept + pia_pred + kink_0_adj + kink_1_adj)


def benefit(
    pension_wealth: FloatND,
    imp_fraction_receiving: FloatND,
    epdv_constant_pension: FloatND,
    period: Period,
) -> FloatND:
    """Pension benefit received from pension wealth (French & Jones 2011, eq. D.4).

    `pb_t = pf_t · Γ_t⁻¹ · pw_t`, with `pf_t` the fraction receiving and `Γ_t`
    the annuity factor `epdv_constant_pension`. The fraction enters **once**.

    In solve, `pension_wealth = pbmax · Γ` (the imputation), so this round-trips
    to `pbmax · pf` — the same benefit as the PIA imputation. In simulate,
    `pension_wealth` is the carried true value, so the benefit is the agent's
    actual one.
    """
    return (
        pension_wealth * imp_fraction_receiving[period] / epdv_constant_pension[period]
    )


def total_to_pia(
    pension_benefit: FloatND,
    pia: FloatND,
    period: Period,
    his: IntND,
    marginal_tax_rate: FloatND,
    imp_intercept: FloatND,
    imp_pia_coeff: FloatND,
    imp_pia_kink_0_coeff: FloatND,
    imp_pia_kink_1_coeff: FloatND,
    imp_kink_0: FloatND,
    imp_kink_1: FloatND,
) -> FloatND:
    """Invert pension imputation to recover PIA from total after-tax benefits.

    Piecewise-linear inverse of the pension_benefit mapping, adjusted for
    marginal tax rates.
    """
    after_tax = 1.0 - marginal_tax_rate
    total_ben = after_tax * pension_benefit + pia

    at_intercept = after_tax * imp_intercept[period, his]
    at_pia = after_tax * imp_pia_coeff[period, his]
    at_kink_0 = after_tax * imp_pia_kink_0_coeff[period, his]
    at_kink_1 = after_tax * imp_pia_kink_1_coeff[period, his]

    k0 = imp_kink_0[period]
    k1 = imp_kink_1[period]
    kink_0_tb = at_intercept + k0 * (1.0 + at_pia)
    kink_1_tb = kink_0_tb + (k1 - k0) * (1.0 + at_pia + at_kink_0)

    return jnp.where(
        total_ben < at_intercept,
        0.0,
        jnp.where(
            total_ben < kink_0_tb,
            (total_ben - at_intercept) / (1.0 + at_pia),
            jnp.where(
                total_ben < kink_1_tb,
                k0 + (total_ben - kink_0_tb) / (1.0 + at_pia + at_kink_0),
                k1 + (total_ben - kink_1_tb) / (1.0 + at_pia + at_kink_0 + at_kink_1),
            ),
        ),
    )


def accrual(
    labor_income: FloatND,
    period: Period,
    his: IntND,
    accrual_intercept: FloatND,
    accrual_log_earnings: FloatND,
    accrual_prob_intercept: FloatND,
    accrual_prob_log_earnings: FloatND,
    accrual_prob_log_earnings_sq: FloatND,
) -> FloatND:
    """Compute pension wealth accrual from labor earnings.

    Accrual has two components:
    - Accrual rate among holders (linear in log earnings)
    - Probability of accrual (logistic in log earnings)
    """
    lli = jnp.log(jnp.maximum(1.0, labor_income))

    rate = jnp.maximum(
        -0.1,
        accrual_intercept[period, his] + lli * accrual_log_earnings[period, his],
    )

    logit = (
        accrual_prob_intercept[his]
        + lli * accrual_prob_log_earnings[his]
        + lli**2 * accrual_prob_log_earnings_sq[his]
    )
    prob = jnp.exp(logit) / (1.0 + jnp.exp(logit))

    return jnp.where(labor_income > 0.0, rate * prob * labor_income, 0.0)


def wealth(
    full_benefit: FloatND,
    epdv_constant_pension: FloatND,
    period: Period,
) -> FloatND:
    """Imputed pension wealth (French & Jones 2011, eq. D.3): `pw_t = Γ_t · pbmax_t`.

    The annuity factor `Γ_t` (`epdv_constant_pension`) already carries the
    fraction-receiving stream over future ages, so the level uses the *full*
    benefit `pbmax` (no current-period fraction). This is the solve-phase
    imputation of `pension_wealth`; in simulate the carried true value is used.
    """
    return full_benefit * epdv_constant_pension[period]


def wealth_next_before_adjustment(
    pension_wealth: FloatND,
    pension_benefit: FloatND,
    pension_accrual: FloatND,
    rate_of_return: ScalarFloat,
    unconditional_survival_prob: FloatND,
    period: Period,
) -> FloatND:
    """Exact evolution of pension wealth (before imputation adjustment).

    next = ((1 + r) * wealth + accrual - benefit) / survival_prob
    The (1 + r) factor compounds pension wealth at the market return rate.
    The division by survival prob accounts for the annuity pricing.
    """
    return (
        (1.0 + rate_of_return) * pension_wealth + pension_accrual - pension_benefit
    ) / unconditional_survival_prob[period]


def assets_adjustment(
    pension_wealth_next_before_adjustment: FloatND,
    imputed_pension_wealth_next_period: FloatND,
    marginal_tax_rate: FloatND,
    unconditional_survival_prob: FloatND,
    period: Period,
) -> FloatND:
    """Asset correction for pension imputation error.

    Adjusts next-period assets by the discrepancy between exact pension wealth
    (from accrual tracking) and imputed pension wealth (from next-period PIA),
    scaled by after-tax rate and survival probability.
    """
    return (
        (1.0 - marginal_tax_rate)
        * unconditional_survival_prob[period]
        * (pension_wealth_next_before_adjustment - imputed_pension_wealth_next_period)
    )


def imputed_pension_wealth_next_period(
    next_aime: ContinuousState,
    target_his: IntND,
    period: Period,
    pia_table: FloatND,
    pia_aime_grid: FloatND,
    imp_intercept_next_period: FloatND,
    imp_pia_coeff_next_period: FloatND,
    imp_pia_kink_0_coeff_next_period: FloatND,
    imp_pia_kink_1_coeff_next_period: FloatND,
    imp_kink_0_next_period: FloatND,
    imp_kink_1_next_period: FloatND,
    epdv_constant_pension_next_period: FloatND,
) -> FloatND:
    """Imputed pension wealth at next period using the target regime's HIS.

    `pw_{t+1} = Γ_{t+1} · pbmax_{t+1}` (French & Jones 2011, eq. D.3): the
    *full* next-period benefit times the next-period annuity factor, with no
    current-period fraction (the fraction lives inside `Γ`). Mirrors
    `full_benefit` and `wealth` but indexes 1-period-shifted views so all
    subscripts use bare-name parameters (`period`, `target_his`). Inlining is
    required: pylcm's AST shape inference inspects the registered function's
    body and does not trace through nested calls.
    """
    next_pia = jnp.interp(next_aime, pia_aime_grid, pia_table)

    intercept = imp_intercept_next_period[period, target_his]
    pia_pred = imp_pia_coeff_next_period[period, target_his] * next_pia
    kink_0_adj = imp_pia_kink_0_coeff_next_period[period, target_his] * jnp.maximum(
        0.0, next_pia - imp_kink_0_next_period[period]
    )
    kink_1_adj = imp_pia_kink_1_coeff_next_period[period, target_his] * jnp.maximum(
        0.0, next_pia - imp_kink_1_next_period[period]
    )

    pbmax_next = jnp.maximum(0.0, intercept + pia_pred + kink_0_adj + kink_1_adj)
    return pbmax_next * epdv_constant_pension_next_period[period]
