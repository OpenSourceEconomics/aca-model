"""Health insurance types and functions: premiums, OOP costs, SSI/Medicaid.

Ported from struct-ret/src/model/baseline/health_insurance_ssi.py.

In the pylcm port, the HIS dimension (Retiree/Tied/Non-Group) is encoded in the
regime. Medicare eligibility is also encoded in the regime. This eliminates the
sparse HIC representation and most eligibility checks.

What remains:
- Medicaid/SSI eligibility (endogenous, depends on assets + income)
- Premium and OOP cost computation (depends on regime's HIC category)
- SSI benefit computation

Eligibility is a smooth share, not a boolean: each statutory threshold is
replaced by a quintic-smoothstep ramp over the
+/- `ELIGIBILITY_BAND_HALF_WIDTH` band, and every consumer mixes its
eligible and ineligible branches with that share. This keeps the budget
chain C² in `assets`, which DC-EGM's per-node evaluation of
savings-stage functions requires; outside the band the model is
bit-identical to the boolean rule.
"""

import jax.numpy as jnp
from lcm import categorical
from lcm.typing import (
    Age,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    IntND,
    Period,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)

from aca_model.agent.labor_market import LaborSupply


@categorical(ordered=False)
class BuyPrivate:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=False)
class HealthInsuranceState:
    retiree: ScalarInt
    tied: ScalarInt
    nongroup: ScalarInt


def countable_income(
    labor_income: FloatND,
    capital_income: FloatND,
    spousal_income: DiscreteState,
    spousal_income_amounts: FloatND,
    ss_benefit: FloatND,
    pension_benefit: FloatND,
    ssi_ignored_overall: ScalarFloat,
    ssi_ignored_earned: ScalarFloat,
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


# Half-width (dollars) of the smoothing band around each statutory
# SSI/Medicaid threshold. Pre-registered, never tuned against solver
# output; sensitivity runs vary it explicitly.
ELIGIBILITY_BAND_HALF_WIDTH = 50.0


def share_below_threshold(value: FloatND, threshold: FloatND) -> FloatND:
    """Smooth share with which `value < threshold` holds.

    Quintic smoothstep (C²) with compact support: exactly 1 at or below
    `threshold - ELIGIBILITY_BAND_HALF_WIDTH`, exactly 0 at or above
    `threshold + ELIGIBILITY_BAND_HALF_WIDTH`, 0.5 at the threshold.
    """
    ramp_position = jnp.clip(
        (value - threshold + ELIGIBILITY_BAND_HALF_WIDTH)
        / (2.0 * ELIGIBILITY_BAND_HALF_WIDTH),
        0.0,
        1.0,
    )
    return 1.0 - ramp_position**3 * (
        ramp_position * (6.0 * ramp_position - 15.0) + 10.0
    )


def ssi_eligibility_share(
    assets: ContinuousState,
    countable_income: FloatND,
    spousal_income: DiscreteState,
    gets_medicare: ScalarBool,
    ssi_assets_test: FloatND,
    ssi_maximum_benefit: FloatND,
) -> FloatND:
    """Smooth SSI/Medicaid eligibility share in [0, 1].

    The two statutory tests enter as smoothstep shares whose product is
    the joint share:

    - assets below the household-specific `ssi_assets_test`
    - countable income below the household-specific `ssi_maximum_benefit`

    Medicare stays a hard gate: it is a known constant per regime (True in
    mc regimes, disability-dependent in no_mc regimes), so it cannot
    produce a cliff in a continuous state.
    """
    assets_share = share_below_threshold(assets, ssi_assets_test[spousal_income])
    income_share = share_below_threshold(
        countable_income, ssi_maximum_benefit[spousal_income]
    )
    return gets_medicare * assets_share * income_share


def ssi_benefit(
    countable_income: FloatND,
    spousal_income: DiscreteState,
    ssi_eligibility_share: FloatND,
    ssi_maximum_benefit: FloatND,
) -> FloatND:
    """Compute SSI benefit amount.

    SSI = share * max(0, max_benefit - countable_income): the statutory
    benefit weighted by the smooth eligibility share.
    """
    benefit = ssi_maximum_benefit[spousal_income] - countable_income
    return ssi_eligibility_share * jnp.maximum(0.0, benefit)


def premium(
    age: Age,
    good_health: IntND,
    is_married: IntND,
    labor_supply: DiscreteAction,
    buy_private: DiscreteAction,
    premium_intercept: ScalarFloat,
    premium_age: ScalarFloat,
    premium_age_sq: ScalarFloat,
    premium_age_cub: ScalarFloat,
    premium_predicted_hcc: ScalarFloat,
    premium_good_health: ScalarFloat,
    premium_married: ScalarFloat,
    premium_works: ScalarFloat,
    premium_married_works: ScalarFloat,
    premium_minimum: ScalarFloat,
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
    age: Age,
    good_health: IntND,
    is_married: IntND,
    labor_supply: DiscreteAction,
    premium_intercept: ScalarFloat,
    premium_age: ScalarFloat,
    premium_age_sq: ScalarFloat,
    premium_age_cub: ScalarFloat,
    premium_predicted_hcc: ScalarFloat,
    premium_good_health: ScalarFloat,
    premium_married: ScalarFloat,
    premium_works: ScalarFloat,
    premium_married_works: ScalarFloat,
    premium_minimum: ScalarFloat,
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
    age: Age,
    good_health: IntND,
    is_married: IntND,
    premium_intercept: ScalarFloat,
    premium_age: ScalarFloat,
    premium_age_sq: ScalarFloat,
    premium_age_cub: ScalarFloat,
    premium_predicted_hcc: ScalarFloat,
    premium_good_health: ScalarFloat,
    premium_married: ScalarFloat,
    premium_minimum: ScalarFloat,
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
    deductible: ScalarFloat | FloatND,
    coinsurance_rate: ScalarFloat | FloatND,
    oop_max: ScalarFloat | FloatND,
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
    deductible: ScalarFloat,
    coinsurance_rate: ScalarFloat,
    oop_max: ScalarFloat,
) -> FloatND:
    """Compute primary OOP costs.

    When uninsured (`buy_private=no`), OOP equals total health costs
    (no coverage).
    """
    insured_oop = oop_costs(total_health_costs, deductible, coinsurance_rate, oop_max)
    return jnp.where(buy_private == BuyPrivate.yes, insured_oop, total_health_costs)


def medicaid_eligibility_share(ssi_eligibility_share: FloatND) -> FloatND:
    """Baseline: Medicaid eligibility share equals the SSI share."""
    return ssi_eligibility_share


def target_his(
    his: IntND,
    labor_supply: DiscreteAction,
) -> IntND:
    """Return the HIS class of the deterministic surviving target regime.

    Mirrors the deterministic cross-HIS branch inside
    `_make_transition_canwork`: tied agents who stop working become
    nongroup. The Medicaid path to nongroup is a probability
    (`medicaid_eligibility_share`), not a deterministic override — it
    enters the imputation through the share mixture in
    `imputed_pension_wealth_next_period` instead. Used by
    `imputed_pension_wealth_next_period_no_medicaid` to look up next-period
    imputation coefficients at the target's HIS.
    """
    tied_to_ng = (his == HealthInsuranceState.tied) & (
        labor_supply == LaborSupply.do_not_work
    )
    return jnp.where(
        tied_to_ng,
        HealthInsuranceState.nongroup,
        his,
    ).astype(jnp.int32)


def target_his_forcedout(his: IntND) -> IntND:
    """Return the deterministic target HIS in forced-out regimes.

    Forced-out regimes have no labor-supply choice, and tied agents have
    already moved to nongroup before the forced-out age, so the
    deterministic target keeps the regime's own HIS; the Medicaid path is
    a probability handled by the imputation mixture. Used by
    `imputed_pension_wealth_next_period_no_medicaid` to look up next-period
    imputation coefficients at the target's HIS.
    """
    return jnp.asarray(his).astype(jnp.int32)


def oop_with_medicaid(
    primary_oop: FloatND,
    medicaid_eligibility_share: FloatND,
    deductible_medicaid: ScalarFloat,
    coinsurance_rate_medicaid: ScalarFloat,
    oop_max_medicaid: ScalarFloat,
) -> FloatND:
    """Apply Medicaid cost-sharing on top of primary insurance OOP costs.

    Medicaid acts as secondary payer: its deductible/coinsurance/OOP-max
    schedule is applied to the primary OOP, and the result is mixed with
    the uncovered primary OOP by the smooth eligibility share.
    """
    medicaid_oop = oop_costs(
        total_health_costs=primary_oop,
        deductible=deductible_medicaid,
        coinsurance_rate=coinsurance_rate_medicaid,
        oop_max=oop_max_medicaid,
    )
    return (
        medicaid_eligibility_share * medicaid_oop
        + (1.0 - medicaid_eligibility_share) * primary_oop
    )


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
    std_xsect_persistent: ScalarFloat,
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
