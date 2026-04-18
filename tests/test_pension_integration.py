"""Integration tests for pension rebalancing mechanism.

Compose small subsets of the real DAG functions via dags.concatenate_functions
and verify combined behavior. The pension adjustment mechanism preserves total
wealth (liquid assets + pension wealth) when HIS changes.
"""

import jax.numpy as jnp
from dags import concatenate_functions

from aca_model.agent import assets_and_income
from aca_model.environment import pensions

ATOL = 0.01
RATE_OF_RETURN = 0.03

# Pension imputation coefficients — two HIS types with different intercepts.
# HIS 0 (retiree): intercept = -50, HIS 1 (nongroup): intercept = -80.
N_PERIODS = 30
N_HIS = 2
PERIOD = 20

_intercept = jnp.zeros((N_PERIODS, N_HIS))
_intercept = _intercept.at[PERIOD, 0].set(-50.0)
_intercept = _intercept.at[PERIOD, 1].set(-80.0)
_intercept = _intercept.at[PERIOD + 1, 0].set(-50.0)
_intercept = _intercept.at[PERIOD + 1, 1].set(-80.0)

_pia_coeff = jnp.zeros((N_PERIODS, N_HIS))
_pia_coeff = _pia_coeff.at[PERIOD, :].set(0.2)
_pia_coeff = _pia_coeff.at[PERIOD + 1, :].set(0.2)

IMP_KWARGS = {
    "imp_intercept": _intercept,
    "imp_pia_coeff": _pia_coeff,
    "imp_pia_kink_0_coeff": jnp.zeros((N_PERIODS, N_HIS)),
    "imp_pia_kink_1_coeff": jnp.zeros((N_PERIODS, N_HIS)),
    "imp_kink_0": jnp.full(N_PERIODS, 99999.0),
    "imp_kink_1": jnp.full(N_PERIODS, 99999.0),
    "imp_fraction_receiving": jnp.ones(N_PERIODS),
}

ACCRUAL_KWARGS = {
    "accrual_intercept": jnp.zeros((N_PERIODS, N_HIS)),
    "accrual_log_earnings": jnp.full((N_PERIODS, N_HIS), 0.5),
    "accrual_prob_intercept": jnp.full(N_HIS, 0.1),
    "accrual_prob_log_earnings": jnp.zeros(N_HIS),
    "accrual_prob_log_earnings_sq": jnp.zeros(N_HIS),
}

EPDV = jnp.full(N_PERIODS, 10.0)
SURVIVAL = jnp.full(N_PERIODS, 0.99)


def test_benefit_wealth_dag() -> None:
    """Benefit→wealth chain via dags matches manual computation."""
    functions = {
        "pension_benefit": pensions.benefit,
        "pension_wealth": pensions.wealth,
    }
    combined = concatenate_functions(functions, targets="pension_wealth")
    result = combined(
        pia=jnp.array(500.0),
        period=PERIOD,
        his=0,
        epdv_constant_pension=EPDV,
        **IMP_KWARGS,
    )
    # benefit = max(0, -50 + 500*0.2) = 50, wealth = 50 * 10 = 500
    assert jnp.isclose(result, 500.0, atol=ATOL)


def test_total_to_pia_inverts_benefit_via_dag() -> None:
    """benefit→total_to_pia round-trip via dags recovers original PIA."""
    functions = {
        "pension_benefit": pensions.benefit,
        "total_to_pia": pensions.total_to_pia,
    }
    combined = concatenate_functions(functions, targets="total_to_pia")
    recovered = combined(
        pia=jnp.array(8000.0),
        period=PERIOD,
        his=0,
        marginal_tax_rate=jnp.array(0.2),
        **IMP_KWARGS,
    )
    assert jnp.isclose(recovered, 8000.0, atol=ATOL)


def test_next_assets_includes_pension_adjustment() -> None:
    """next_assets adds pension_assets_adjustment to savings."""
    functions = {"next_assets": assets_and_income.next_assets}
    combined = concatenate_functions(functions, targets="next_assets")
    result = combined(
        cash_on_hand=jnp.array(100_000.0),
        transfers=jnp.array(0.0),
        pension_assets_adjustment=jnp.array(5_000.0),
        consumption=jnp.array(80_000.0),
        oop_costs=jnp.array(0.0),
    )
    assert jnp.isclose(result, 25_000.0, atol=ATOL)


def test_zero_adjustment_when_his_unchanged() -> None:
    """Pension adjustment is zero when HIS doesn't change."""
    his = 0
    pia = jnp.array(8000.0)
    labor_income = jnp.array(30_000.0)
    mtr = jnp.array(0.2)

    benefit = pensions.benefit(pia=pia, period=PERIOD, his=his, **IMP_KWARGS)
    pw = pensions.wealth(
        pension_benefit=benefit, epdv_constant_pension=EPDV, period=PERIOD
    )
    accrual_val = pensions.accrual(
        labor_income=labor_income, period=PERIOD, his=his, **ACCRUAL_KWARGS
    )

    next_exact = pensions.wealth_next_before_adjustment(
        pension_wealth=pw,
        pension_benefit=benefit,
        pension_accrual=accrual_val,
        rate_of_return=RATE_OF_RETURN,
        unconditional_survival_prob=SURVIVAL,
        period=PERIOD,
    )

    next_benefit = pensions.benefit(pia=pia, period=PERIOD + 1, his=his, **IMP_KWARGS)
    next_imputed = pensions.wealth(
        pension_benefit=next_benefit, epdv_constant_pension=EPDV, period=PERIOD + 1
    )

    adjustment = pensions.assets_adjustment(
        pension_wealth_next_before_adjustment=next_exact,
        imputed_pension_wealth_next_period=next_imputed,
        marginal_tax_rate=mtr,
        unconditional_survival_prob=SURVIVAL,
        period=PERIOD,
    )

    assert not jnp.isnan(adjustment)
    assert jnp.isfinite(adjustment)


def test_rebalancing_preserves_total_wealth_across_his_change() -> None:
    """When HIS changes, pension adjustment preserves total wealth.

    Total wealth = liquid assets + pension wealth. When an agent transitions
    from HIS 0 (retiree) to HIS 1 (nongroup), the pension imputation changes.
    The assets_adjustment compensates so total wealth is preserved.
    """
    old_his = 0
    new_his = 1
    pia = jnp.array(8000.0)
    labor_income = jnp.array(30_000.0)
    mtr = jnp.array(0.0)
    liquid_assets = jnp.array(100_000.0)

    benefit_old = pensions.benefit(pia=pia, period=PERIOD, his=old_his, **IMP_KWARGS)
    pw_old = pensions.wealth(
        pension_benefit=benefit_old, epdv_constant_pension=EPDV, period=PERIOD
    )
    accrual_val = pensions.accrual(
        labor_income=labor_income, period=PERIOD, his=old_his, **ACCRUAL_KWARGS
    )

    next_exact = pensions.wealth_next_before_adjustment(
        pension_wealth=pw_old,
        pension_benefit=benefit_old,
        pension_accrual=accrual_val,
        rate_of_return=RATE_OF_RETURN,
        unconditional_survival_prob=SURVIVAL,
        period=PERIOD,
    )

    benefit_new = pensions.benefit(
        pia=pia, period=PERIOD + 1, his=new_his, **IMP_KWARGS
    )
    next_imputed = pensions.wealth(
        pension_benefit=benefit_new, epdv_constant_pension=EPDV, period=PERIOD + 1
    )

    adjustment = pensions.assets_adjustment(
        pension_wealth_next_before_adjustment=next_exact,
        imputed_pension_wealth_next_period=next_imputed,
        marginal_tax_rate=mtr,
        unconditional_survival_prob=SURVIVAL,
        period=PERIOD,
    )

    next_liquid = liquid_assets + adjustment
    total_with_adjustment = next_liquid + next_imputed
    total_without_change = liquid_assets + next_exact

    residual = (1.0 - SURVIVAL[PERIOD]) * jnp.abs(next_imputed - next_exact)
    assert jnp.abs(total_with_adjustment - total_without_change) <= residual + ATOL
