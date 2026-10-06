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
from aca_model.environment import pensions, social_security

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


def test_total_to_pia_keeps_adjusted_total_via_dag() -> None:
    """The carried PIA keeps SS plus after-tax imputed pension at its adjusted total.

    Accrued PIA 8,000 is adjusted to 7,000; the carried PIA `p` satisfies
    `p + (1 - τ) pbmax(p) = 7,000 + (1 - τ) pbmax(8,000)`.
    """
    next_period_kwargs = {f"{k}_next_period": v for k, v in PBMAX_KWARGS.items()}
    functions = {
        "full_benefit_next_period": pensions.full_benefit_next_period,
        "total_to_pia": pensions.total_to_pia,
    }
    combined = concatenate_functions(functions, targets="total_to_pia")
    mtr = jnp.array(0.2)
    carried = combined(
        pia_adjusted_next_period=jnp.array(7000.0),
        pia_unadjusted_next_period=jnp.array(8000.0),
        target_his=jnp.int32(0),
        period=PERIOD,
        marginal_tax_rate=mtr,
        **next_period_kwargs,
    )

    def pbmax(pia: jnp.ndarray) -> jnp.ndarray:
        return pensions.full_benefit_next_period(
            pia_unadjusted_next_period=pia,
            target_his=jnp.int32(0),
            period=PERIOD,
            **next_period_kwargs,
        )

    total_carried = carried + (1.0 - mtr) * pbmax(carried)
    target = 7000.0 + (1.0 - mtr) * pbmax(jnp.array(8000.0))
    assert jnp.isclose(total_carried, target, atol=ATOL)


# `pbmax(p) = max(0, -50 + 0.2 p)` at PERIOD for HIS 0 reaches zero at p = 250.
# With `τ = 0.2` the carried PIA `p` solves `p + 0.8 pbmax(p) = A + 0.8 pbmax(U)`:
# - below the floor: A = 150, U = 200 → total 150, `p = 150`, `pbmax = 0`
# - at the floor: A = 250, U = 200 → total 250, `p = 250`, `pbmax = 0`
# - above the floor: A = 420, U = 1,000 → total 540, `p = 500`, `pbmax = 50`
# The PIA table is linear with slope 0.9 below 900, so `next_aime = p / 0.9`.
_FLOOR_CASES = {
    "below_floor": (150.0, 200.0, 150.0, 0.0),
    "at_floor": (250.0, 200.0, 250.0, 0.0),
    "above_floor": (420.0, 1000.0, 500.0, 50.0),
}
_PIA_TABLE = jnp.array([0.0, 900.0, 2000.0])
_PIA_AIME_GRID = jnp.array([0.0, 1000.0, 5000.0])


def _carried_pia_outcomes(
    *, pia_adjusted: float, pia_unadjusted: float
) -> dict[str, jnp.ndarray]:
    """Carried PIA, its next-period AIME, and the pension benefit it imputes."""
    next_period_kwargs = {f"{k}_next_period": v for k, v in PBMAX_KWARGS.items()}
    functions = {
        "full_benefit_next_period": pensions.full_benefit_next_period,
        "carried_pia": pensions.total_to_pia,
        "next_aime": social_security.next_aime,
    }
    combined = concatenate_functions(functions, targets=["carried_pia", "next_aime"])
    out = combined(
        pia_adjusted_next_period=jnp.array(pia_adjusted),
        pia_unadjusted_next_period=jnp.array(pia_unadjusted),
        target_his=jnp.int32(0),
        period=PERIOD,
        marginal_tax_rate=jnp.array(0.2),
        pia_table=_PIA_TABLE,
        pia_aime_grid=_PIA_AIME_GRID,
        **next_period_kwargs,
    )
    benefit = pensions.full_benefit_next_period(
        pia_unadjusted_next_period=out["carried_pia"],
        target_his=jnp.int32(0),
        period=PERIOD,
        **next_period_kwargs,
    )
    return {**out, "benefit": benefit}


@pytest.mark.parametrize(
    ("pia_adjusted", "pia_unadjusted", "expected_pia", "expected_benefit"),
    list(_FLOOR_CASES.values()),
    ids=list(_FLOOR_CASES),
)
def test_carried_pia_solves_the_floored_total(
    pia_adjusted: float,
    pia_unadjusted: float,
    expected_pia: float,
    expected_benefit: float,
) -> None:
    """The carried PIA keeps SS plus after-tax floored pension at its adjusted total."""
    del expected_benefit
    out = _carried_pia_outcomes(
        pia_adjusted=pia_adjusted, pia_unadjusted=pia_unadjusted
    )
    assert jnp.isclose(out["carried_pia"], expected_pia, atol=ATOL)


@pytest.mark.parametrize(
    ("pia_adjusted", "pia_unadjusted", "expected_pia", "expected_benefit"),
    list(_FLOOR_CASES.values()),
    ids=list(_FLOOR_CASES),
)
def test_carried_pia_imputes_the_floored_pension_benefit(
    pia_adjusted: float,
    pia_unadjusted: float,
    expected_pia: float,
    expected_benefit: float,
) -> None:
    """The pension benefit imputed from the carried PIA is the floored `pbmax`."""
    del expected_pia
    out = _carried_pia_outcomes(
        pia_adjusted=pia_adjusted, pia_unadjusted=pia_unadjusted
    )
    assert jnp.isclose(out["benefit"], expected_benefit, atol=ATOL)


@pytest.mark.parametrize(
    ("pia_adjusted", "pia_unadjusted", "expected_pia", "expected_benefit"),
    list(_FLOOR_CASES.values()),
    ids=list(_FLOOR_CASES),
)
def test_carried_pia_sets_next_aime_on_either_side_of_the_floor(
    pia_adjusted: float,
    pia_unadjusted: float,
    expected_pia: float,
    expected_benefit: float,
) -> None:
    """Next period's AIME is the AIME of the floored-consistent carried PIA."""
    del expected_benefit
    out = _carried_pia_outcomes(
        pia_adjusted=pia_adjusted, pia_unadjusted=pia_unadjusted
    )
    assert jnp.isclose(out["next_aime"], expected_pia / 0.9, atol=ATOL)


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
        labor_income=labor_income,
        age=jnp.int32(60),
        period=PERIOD,
        his=his,
        **ACCRUAL_KWARGS,
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
        labor_income=labor_income,
        age=jnp.int32(60),
        period=PERIOD,
        his=old_his,
        **ACCRUAL_KWARGS,
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
        labor_income=labor_income,
        age=jnp.int32(60),
        period=PERIOD,
        his=old_his,
        **ACCRUAL_KWARGS,
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
# law; `next_assets_from_savings` is the post-decision (DC-EGM / NBEGM) form.
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
    Holds for both the brute-force and the DC-EGM/NBEGM (savings) assets law.
    """
    law, kwargs = _ADJUSTED_ASSETS_LAWS[law_key]
    functions = build_pension_functions(REGIME_SPECS["tied_nomc_choose_canwork"])
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
    Holds for both the brute-force and the DC-EGM/NBEGM (savings) assets law.
    """
    law, kwargs = _ADJUSTED_ASSETS_LAWS[law_key]
    solve_adjustment = _solve_phase_adjustment_across_his_change()

    combined = concatenate_functions({"next_assets": law}, targets="next_assets")
    inflated = combined(pension_assets_adjustment=solve_adjustment, **kwargs)
    assert jnp.isclose(inflated, 18_000.0 + solve_adjustment, atol=ATOL)
    assert jnp.abs(solve_adjustment) > 100.0
