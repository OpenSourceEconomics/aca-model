"""Health insurance types and functions: premiums, OOP costs, SSI/Medicaid.

Ported from struct-ret/src/model/baseline/health_insurance_ssi.py.

In the pylcm port, the HIS dimension (Retiree/Tied/Non-Group) is encoded in the
regime. Medicare eligibility is also encoded in the regime. This eliminates the
sparse HIC representation and most eligibility checks.

What remains:
- Medicaid/SSI eligibility (endogenous, depends on assets + income)
- Premium and OOP cost computation (depends on regime's HIC category)
- SSI benefit computation
"""

import jax.numpy as jnp
from lcm import categorical
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    IntND,
    Period,
)

from aca_model.agent.labor_market import LaborSupply


@categorical(ordered=False)
class BuyPrivate:
    no: int
    yes: int


@categorical(ordered=False)
class HealthInsuranceState:
    retiree: int
    tied: int
    nongroup: int


def countable_income(
    labor_income: FloatND,
    capital_income: FloatND,
    spousal_income: DiscreteState,
    spousal_income_amounts: FloatND,
    ss_benefit: FloatND,
    pension_benefit: FloatND,
    ssi_ignored_overall: float,
    ssi_ignored_earned: float,
) -> FloatND:
    """Compute countable income for SSI eligibility test.

    Earned income counts at 50% rate after disregards.
    """
    earned = jnp.maximum(0.0, 0.5 * (labor_income - ssi_ignored_earned))
    return (
        earned
        + capital_income
        + spousal_income_amounts[spousal_income]
        + ss_benefit
        + pension_benefit
        - ssi_ignored_overall
    )


def is_ssi_eligible(
    assets: ContinuousState,
    countable_income: FloatND,
    spousal_income: DiscreteState,
    gets_medicare: bool,
    ssi_assets_test: FloatND,
    ssi_maximum_benefit: FloatND,
) -> BoolND:
    """Check SSI/Medicaid eligibility.

    Requires: Medicare-eligible AND assets below test AND income below maximum.
    In the regime decomposition, gets_medicare is a known constant (True in mc regimes,
    and needs checking in no_mc regimes based on disability).
    """
    assets_ok = assets < ssi_assets_test[spousal_income]
    income_ok = countable_income < ssi_maximum_benefit[spousal_income]
    return gets_medicare & assets_ok & income_ok


def ssi_benefit(
    countable_income: FloatND,
    spousal_income: DiscreteState,
    is_ssi_eligible: BoolND,
    ssi_maximum_benefit: FloatND,
) -> FloatND:
    """Compute SSI benefit amount.

    SSI = max_benefit - countable_income, if eligible; 0 otherwise.
    """
    benefit = ssi_maximum_benefit[spousal_income] - countable_income
    return jnp.where(is_ssi_eligible, jnp.maximum(0.0, benefit), 0.0)


def premium(
    age: int,
    good_health: IntND,
    is_married: IntND,
    labor_supply: DiscreteAction,
    buy_private: DiscreteAction,
    premium_intercept: float,
    premium_age: int,
    premium_age_sq: float,
    premium_age_cub: float,
    premium_predicted_hcc: float,
    premium_good_health: float,
    premium_married: float,
    premium_works: float,
    premium_married_works: float,
    premium_minimum: float,
    predicted_hcc_insurer: FloatND,
) -> FloatND:
    """Compute health insurance premium for canwork regimes.

    Premium coefficients are regime-specific (different for each HIC type).
    Return 0 when uninsured (`buy_private=no`).
    """
    works = labor_supply != LaborSupply.do_not_work
    raw = (
        premium_intercept
        + premium_age * age
        + premium_age_sq * age**2
        + premium_age_cub * age**3
        + premium_predicted_hcc * predicted_hcc_insurer
        + premium_good_health * good_health
        + premium_married * is_married
        + premium_works * works
        + premium_married_works * is_married * works
    )
    return jnp.where(
        buy_private == BuyPrivate.yes,
        jnp.maximum(premium_minimum, raw),
        0.0,
    )


def premium_insured(
    age: int,
    good_health: IntND,
    is_married: IntND,
    labor_supply: DiscreteAction,
    premium_intercept: float,
    premium_age: int,
    premium_age_sq: float,
    premium_age_cub: float,
    premium_predicted_hcc: float,
    premium_good_health: float,
    premium_married: float,
    premium_works: float,
    premium_married_works: float,
    premium_minimum: float,
    predicted_hcc_insurer: FloatND,
) -> FloatND:
    """Compute health insurance premium for canwork regimes without `buy_private`.

    Used by retiree, tied, and nongroup-with-Medicare regimes where agents
    always have coverage (no uninsured option).
    """
    works = labor_supply != LaborSupply.do_not_work
    raw = (
        premium_intercept
        + premium_age * age
        + premium_age_sq * age**2
        + premium_age_cub * age**3
        + premium_predicted_hcc * predicted_hcc_insurer
        + premium_good_health * good_health
        + premium_married * is_married
        + premium_works * works
        + premium_married_works * is_married * works
    )
    return jnp.maximum(premium_minimum, raw)


def premium_retired(
    age: int,
    good_health: IntND,
    is_married: IntND,
    premium_intercept: float,
    premium_age: int,
    premium_age_sq: float,
    premium_age_cub: float,
    premium_predicted_hcc: float,
    premium_good_health: float,
    premium_married: float,
    premium_minimum: float,
    predicted_hcc_insurer: FloatND,
) -> FloatND:
    """Compute health insurance premium for forcedout regimes.

    No work terms since labor supply is not available.
    """
    premium = (
        premium_intercept
        + premium_age * age
        + premium_age_sq * age**2
        + premium_age_cub * age**3
        + premium_predicted_hcc * predicted_hcc_insurer
        + premium_good_health * good_health
        + premium_married * is_married
    )
    return jnp.maximum(premium_minimum, premium)


def oop_costs(
    total_health_costs: FloatND,
    deductible: float | FloatND,
    coinsurance_rate: float | FloatND,
    oop_max: float | FloatND,
) -> FloatND:
    """Compute out-of-pocket health care costs.

    Standard deductible + coinsurance with OOP maximum.
    """
    oop = jnp.where(
        total_health_costs < deductible,
        total_health_costs,
        deductible + (total_health_costs - deductible) * coinsurance_rate,
    )
    return jnp.minimum(oop, oop_max)


def primary_oop(
    total_health_costs: FloatND,
    buy_private: DiscreteAction,
    deductible: float,
    coinsurance_rate: float,
    oop_max: float,
) -> FloatND:
    """Compute primary OOP costs.

    When uninsured (`buy_private=no`), OOP equals total health costs
    (no coverage).
    """
    insured_oop = oop_costs(total_health_costs, deductible, coinsurance_rate, oop_max)
    return jnp.where(buy_private == BuyPrivate.yes, insured_oop, total_health_costs)


def is_medicaid_eligible(is_ssi_eligible: BoolND) -> BoolND:
    """Baseline: Medicaid eligibility equals SSI eligibility."""
    return is_ssi_eligible


def oop_with_medicaid(
    primary_oop: FloatND,
    is_medicaid_eligible: BoolND,
    deductible_medicaid: float,
    coinsurance_rate_medicaid: float,
    oop_max_medicaid: float,
) -> FloatND:
    """Apply Medicaid cost-sharing on top of primary insurance OOP costs.

    When Medicaid-eligible, Medicaid acts as secondary payer: its
    deductible/coinsurance/OOP-max schedule is applied to the primary OOP.
    """
    medicaid_oop = oop_costs(
        total_health_costs=primary_oop,
        deductible=deductible_medicaid,
        coinsurance_rate=coinsurance_rate_medicaid,
        oop_max=oop_max_medicaid,
    )
    return jnp.where(is_medicaid_eligible, medicaid_oop, primary_oop)


def hcc_insurer_predicted(
    hcc_persistent: ContinuousState,
    predicted_hcc_insurer_table: FloatND,
    hcc_persistent_grid: FloatND,
) -> FloatND:
    """Interpolate pre-computed expected insurer cost for the current HCC state.

    The table contains E[total_costs - oop_costs | hcc_persistent] at each
    persistent grid point. Linear interpolation handles off-grid values
    during simulation (where draw_shock returns continuous AR1 values).
    """
    return jnp.interp(hcc_persistent, hcc_persistent_grid, predicted_hcc_insurer_table)


def compute_hcc_insurer_table(
    hcc_persistent_grid: FloatND,
    hcc_persistent_trans_probs: FloatND,
    hcc_transitory_grid: FloatND,
    hcc_transitory_weights: FloatND,
    log_mean: float,
    log_std: float,
    std_xsect_persistent: float,
    deductible: float,
    coinsurance_rate: float,
    oop_max: float,
) -> FloatND:
    """Compute predicted insurer costs table for all persistent grid points.

    For each source persistent state i, integrate over (target persistent j,
    transitory k) weighted by transition probs and quadrature weights.
    """
    std_trans = jnp.sqrt(1.0 - std_xsect_persistent**2)
    # total_costs[j, k] for all (persistent, transitory) combinations
    total = jnp.exp(
        log_mean
        + log_std
        * (
            hcc_persistent_grid[:, None] * std_xsect_persistent
            + hcc_transitory_grid[None, :] * std_trans
        )
    )
    oop = jnp.where(
        total < deductible,
        total,
        deductible + (total - deductible) * coinsurance_rate,
    )
    oop = jnp.minimum(oop, oop_max)
    insurer_costs = total - oop  # [n_persistent, n_transitory]
    # Weight by transitory quadrature weights -> [n_persistent]
    expected_by_persistent = insurer_costs @ hcc_transitory_weights
    # Weight by persistent transition probs: table[i] = sum_j P[i,j] * expected[j]
    return hcc_persistent_trans_probs @ expected_by_persistent


def total_costs(
    period: Period,
    is_married: IntND,
    good_health: IntND,
    log_mean: FloatND,
    log_std: FloatND,
    hcc_persistent: ContinuousState,
    hcc_transitory: ContinuousState,
    std_xsect_persistent: float,
) -> FloatND:
    """Compute total health care costs from log-normal model.

    ``log_mean`` and ``log_std`` are ``pd.Series`` with ``(age, is_married,
    good_health)`` MultiIndex, resolved by pylcm via ``derived_categoricals``.
    """
    std_trans = jnp.sqrt(1.0 - std_xsect_persistent**2)
    return jnp.exp(
        log_mean[period, is_married, good_health]
        + log_std[period, is_married, good_health]
        * (hcc_persistent * std_xsect_persistent + hcc_transitory * std_trans)
    )
