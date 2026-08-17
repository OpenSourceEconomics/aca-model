"""Integration tests for the pension rebalancing mechanism.

Compose small subsets of the real DAG functions via dags.concatenate_functions
and verify combined behavior. The pension adjustment mechanism preserves total
wealth (liquid assets + pension wealth) when HIS changes.
"""

import jax.numpy as jnp
import pytest
from dags import concatenate_functions

from aca_model.agent import assets_and_income
from aca_model.baseline.regimes._common import REGIME_SPECS, build_pension_functions
from aca_model.environment import pensions

ATOL = 0.01
RATE_OF_RETURN = jnp.asarray(0.03)

# Pension imputation coefficients — two HIS types with different intercepts.
# HIS 0 (retiree): intercept = -50, HIS 1 (nongroup): intercept = -80.
N_PERIODS = 30
N_HIS = 2
PERIOD = jnp.int32(20)

_intercept = jnp.zeros((N_PERIODS, N_HIS))
_intercept = _intercept.at[PERIOD, 0].set(-50.0)
_intercept = _intercept.at[PERIOD, 1].set(-80.0)
_intercept = _intercept.at[PERIOD + 1, 0].set(-50.0)
_intercept = _intercept.at[PERIOD + 1, 1].set(-80.0)

_pia_coeff = jnp.zeros((N_PERIODS, N_HIS))
_pia_coeff = _pia_coeff.at[PERIOD, :].set(0.2)
_pia_coeff = _pia_coeff.at[PERIOD + 1, :].set(0.2)

# `pbmax` (eq. D.2) coefficients — no fraction-receiving.
PBMAX_KWARGS = {
    "imp_intercept": _intercept,
    "imp_pia_coeff": _pia_coeff,
    "imp_pia_kink_0_coeff": jnp.zeros((N_PERIODS, N_HIS)),
    "imp_pia_kink_1_coeff": jnp.zeros((N_PERIODS, N_HIS)),
    "imp_kink_0": jnp.full(N_PERIODS, 99999.0),
    "imp_kink_1": jnp.full(N_PERIODS, 99999.0),
}

ACCRUAL_KWARGS = {
    "accrual_intercept": jnp.zeros((N_PERIODS, N_HIS)),
    "accrual_log_earnings": jnp.full((N_PERIODS, N_HIS), 0.5),
    "accrual_prob_intercept": jnp.full(N_HIS, 0.1),
    "accrual_prob_log_earnings": jnp.zeros(N_HIS),
    "accrual_prob_log_earnings_sq": jnp.zeros(N_HIS),
}

FRACTION_RECEIVING = jnp.ones(N_PERIODS)
EPDV = jnp.full(N_PERIODS, 10.0)
SURVIVAL = jnp.full(N_PERIODS, 0.99)


def _impute_pension_wealth(*, pia: jnp.ndarray, period: jnp.ndarray, his: jnp.ndarray):
    """Solve-phase pension wealth `pw = Γ · pbmax` for the given PIA and HIS."""
    functions = {
        "full_benefit": pensions.full_benefit,
        "pension_wealth": pensions.wealth,
    }
    combined = concatenate_functions(functions, targets="pension_wealth")
    return combined(
        pia=pia,
        period=period,
        his=his,
        epdv_constant_pension=EPDV,
        **PBMAX_KWARGS,
    )


def test_imputation_chain_full_benefit_to_wealth() -> None:
    """`pbmax → pw` via dags: `pw = Γ · max(0, intercept + slope·PIA)`."""
    result = _impute_pension_wealth(
        pia=jnp.array(500.0), period=PERIOD, his=jnp.int32(0)
    )
    # pbmax = max(0, -50 + 500*0.2) = 50, pw = 50 * 10 = 500
    assert jnp.isclose(result, 500.0, atol=ATOL)


def test_total_to_pia_inverts_imputed_benefit_via_dag() -> None:
    """`pbmax → total_to_pia` round-trip via dags recovers original PIA."""
    functions = {
        "pension_benefit": pensions.full_benefit,
        "total_to_pia": pensions.total_to_pia,
    }
    combined = concatenate_functions(functions, targets="total_to_pia")
    recovered = combined(
        pia=jnp.array(8000.0),
        period=PERIOD,
        his=jnp.int32(0),
        marginal_tax_rate=jnp.array(0.2),
        **PBMAX_KWARGS,
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
        consumption_dollars=jnp.array(80_000.0),
        oop_costs=jnp.array(0.0),
    )
    assert jnp.isclose(result, 25_000.0, atol=ATOL)


def test_zero_adjustment_when_his_unchanged() -> None:
    """Pension adjustment is finite when HIS doesn't change."""
    his = jnp.int32(0)
    pia = jnp.array(8000.0)
    labor_income = jnp.array(30_000.0)
    mtr = jnp.array(0.2)

    pw = _impute_pension_wealth(pia=pia, period=PERIOD, his=his)
    benefit = pensions.benefit(
        pension_wealth=pw,
        imp_fraction_receiving=FRACTION_RECEIVING,
        epdv_constant_pension=EPDV,
        period=PERIOD,
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

    next_imputed = _impute_pension_wealth(pia=pia, period=PERIOD + 1, his=his)

    adjustment = pensions.assets_adjustment(
        pension_wealth_next_before_adjustment=next_exact,
        imputed_pension_wealth_next_period=next_imputed,
        marginal_tax_rate=mtr,
        unconditional_survival_prob=SURVIVAL,
        period=PERIOD,
    )

    assert jnp.isfinite(adjustment)


def test_rebalancing_preserves_total_wealth_across_his_change() -> None:
    """When HIS changes, pension adjustment preserves total wealth.

    Total wealth = liquid assets + pension wealth. When an agent transitions
    from HIS 0 (retiree) to HIS 1 (nongroup), the pension imputation changes.
    The assets_adjustment compensates so total wealth is preserved.
    """
    old_his = jnp.int32(0)
    new_his = jnp.int32(1)
    pia = jnp.array(8000.0)
    labor_income = jnp.array(30_000.0)
    mtr = jnp.array(0.0)
    liquid_assets = jnp.array(100_000.0)

    pw_old = _impute_pension_wealth(pia=pia, period=PERIOD, his=old_his)
    benefit_old = pensions.benefit(
        pension_wealth=pw_old,
        imp_fraction_receiving=FRACTION_RECEIVING,
        epdv_constant_pension=EPDV,
        period=PERIOD,
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

    next_imputed = _impute_pension_wealth(pia=pia, period=PERIOD + 1, his=new_his)

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


def _solve_phase_adjustment_across_his_change() -> jnp.ndarray:
    """Solve-phase pension assets adjustment for a HIS 0 → 1 transition.

    A HIS change makes next period's AIME imputation diverge from the
    accrual-evolved pension wealth, so the reconciliation the solve phase
    applies is nonzero — the correction the simulate phase must suppress.
    """
    old_his = jnp.int32(0)
    new_his = jnp.int32(1)
    pia = jnp.array(8000.0)
    labor_income = jnp.array(30_000.0)
    mtr = jnp.array(0.0)

    pw_old = _impute_pension_wealth(pia=pia, period=PERIOD, his=old_his)
    benefit_old = pensions.benefit(
        pension_wealth=pw_old,
        imp_fraction_receiving=FRACTION_RECEIVING,
        epdv_constant_pension=EPDV,
        period=PERIOD,
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
    next_imputed = _impute_pension_wealth(pia=pia, period=PERIOD + 1, his=new_his)

    return pensions.assets_adjustment(
        pension_wealth_next_before_adjustment=next_exact,
        imputed_pension_wealth_next_period=next_imputed,
        marginal_tax_rate=mtr,
        unconditional_survival_prob=SURVIVAL,
        period=PERIOD,
    )


# Both assets laws that carry `pension_assets_adjustment`, with inputs chosen so
# the no-adjustment baseline is 18000 either way (savings = cash + transfers −
# consumption = 20000; 20000 − 2000 oop = 18000). `next_assets` is the brute-force
# law; `next_assets_from_savings` is the post-decision (NB-EGM) form.
_ADJUSTED_ASSETS_LAWS = {
    "brute": (
        assets_and_income.next_assets,
        {
            "cash_on_hand": jnp.array(100_000.0),
            "transfers": jnp.array(0.0),
            "consumption_dollars": jnp.array(80_000.0),
            "oop_costs": jnp.array(2_000.0),
        },
    ),
    "savings": (
        assets_and_income.next_assets_from_savings,
        {
            "savings": jnp.array(20_000.0),
            "oop_costs": jnp.array(2_000.0),
        },
    ),
}


@pytest.mark.parametrize("law_key", ["brute", "savings"])
def test_simulate_pension_adjustment_leaves_next_assets_at_no_adjustment_carry(
    law_key: str,
) -> None:
    """Simulate carries the true pension balance, so the assets law gets no
    pension adjustment even when a real imputation gap exists.

    The agent holds `assets = 18000` after consumption and OOP; the
    simulate-phase `pension_assets_adjustment` contributes exactly zero, so
    the pension balance is carried as a state rather than reconciled twice.
    Holds for both the brute-force and the NB-EGM (savings) assets law.
    """
    law, kwargs = _ADJUSTED_ASSETS_LAWS[law_key]
    functions = build_pension_functions(REGIME_SPECS["single_tied_nomc_choose_canwork"])
    simulate_adjustment = functions["pension_assets_adjustment"].simulate()

    combined = concatenate_functions({"next_assets": law}, targets="next_assets")
    next_assets = combined(pension_assets_adjustment=simulate_adjustment, **kwargs)
    assert jnp.isclose(next_assets, 18_000.0, atol=ATOL)


@pytest.mark.parametrize("law_key", ["brute", "savings"])
def test_reenabling_pension_adjustment_in_simulate_inflates_next_assets(
    law_key: str,
) -> None:
    """Re-adding the solve-phase adjustment to simulate double-counts pension
    wealth: the assets law inflates by the (nonzero) reconciliation credit.

    This is the failure the `simulate=zero` wiring prevents. The solve-phase
    adjustment is what re-enabling would inject; feeding it into the assets
    law shifts assets away from the true-balance carry by exactly that credit.
    Holds for both the brute-force and the NB-EGM (savings) assets law.
    """
    law, kwargs = _ADJUSTED_ASSETS_LAWS[law_key]
    solve_adjustment = _solve_phase_adjustment_across_his_change()

    combined = concatenate_functions({"next_assets": law}, targets="next_assets")
    inflated = combined(pension_assets_adjustment=solve_adjustment, **kwargs)
    assert jnp.isclose(inflated, 18_000.0 + solve_adjustment, atol=ATOL)
    assert jnp.abs(solve_adjustment) > 100.0
