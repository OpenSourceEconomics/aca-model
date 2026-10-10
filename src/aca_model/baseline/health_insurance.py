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

import jax
import jax.numpy as jnp
import lcm
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
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)

from aca_model.agent.health import HealthWithDisability
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
    spousal_income_amount: FloatND,
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
        + spousal_income_amount
        + ss_benefit
        + pension_benefit
        - ssi_ignored_overall
    )


@lcm.piecewise_affine(
    output="is_ssi_eligible",
    variable="assets",
    breakpoints=(
        lcm.affine_breakpoint(
            threshold="ssi_assets_test", kind="jump", indexed_by="spousal_income"
        ),
    ),
)
def is_ssi_eligible(
    assets: ContinuousState,
    countable_income: FloatND,
    spousal_income: DiscreteState,
    crossed_oamc_threshold: ScalarBool,
    is_disabled: BoolND,
    ssi_assets_test: FloatND,
    ssi_maximum_benefit: FloatND,
) -> BoolND:
    """Check SSI/Medicaid eligibility on the categorical track.

    The household qualifies when it is categorically eligible — aged
    (post-65) or disabled — AND passes the SSI asset test AND has SSI
    countable income below the SSI maximum benefit.
    `crossed_oamc_threshold` is a known per-regime constant (post-65
    regimes); `is_disabled` reads the disability health state where the
    regime carries it and is constant False otherwise.

    The asset test is a declared jump breakpoint: where eligibility ends,
    the SSI benefit leaves cash-on-hand discontinuously and Medicaid OOP
    switching moves next-period assets, so NBEGM's partition splits at
    the per-household `ssi_assets_test` instead of extrapolating one
    affine budget across the cliff.
    """
    categorical = crossed_oamc_threshold | is_disabled
    assets_ok = assets < ssi_assets_test[spousal_income]
    income_ok = countable_income < ssi_maximum_benefit[spousal_income]
    return categorical & assets_ok & income_ok


def is_disabled_from_health(health: DiscreteState) -> BoolND:
    """Disability indicator for regimes carrying the disability health state.

    Pre-65 `nomc`/`dimc` regimes use `HealthWithDisability`, whose lowest
    state is `disabled`. The household is disabled exactly in that state.
    """
    return health == HealthWithDisability.disabled


def is_disabled_never() -> BoolND:
    """Disability indicator for regimes without a disability health state.

    Post-65 (`oamc`) regimes use the 2-state `Health` grid with no
    disability category, so no household is disabled there.
    """
    return jnp.asarray(False)


def aca_magi(
    labor_income: FloatND,
    capital_income: FloatND,
    spousal_income_amount: FloatND,
    ss_benefit: FloatND,
    pension_benefit: FloatND,
) -> FloatND:
    """Compute MAGI for ACA Medicaid expansion eligibility.

    Modified adjusted gross income counts every income source in full —
    no SSI disregards and no half-counting of earnings — so it is distinct
    from the SSI `countable_income`.
    """
    return (
        labor_income
        + capital_income
        + spousal_income_amount
        + ss_benefit
        + pension_benefit
    )


@lcm.piecewise_affine(
    output="ssi_benefit",
    variable="countable_income",
    breakpoints=(
        lcm.affine_breakpoint(
            threshold="ssi_maximum_benefit",
            kind="continuous_kink",
            indexed_by="spousal_income",
        ),
    ),
)
def ssi_benefit(
    countable_income: FloatND,
    spousal_income: DiscreteState,
    is_ssi_eligible: BoolND,
    ssi_maximum_benefit: FloatND,
) -> FloatND:
    """Compute SSI benefit amount.

    SSI = max_benefit - countable_income, if eligible; 0 otherwise.

    The income test is a declared continuous-kink breakpoint: the benefit
    reaches zero exactly at `countable_income == ssi_maximum_benefit`, so
    cash-on-hand is continuous there but its asset slope changes (the
    benefit stops offsetting capital income). NBEGM maps the threshold to
    its per-cell asset preimage and splits the partition there.
    """
    benefit = ssi_maximum_benefit[spousal_income] - countable_income
    return jnp.where(is_ssi_eligible, jnp.maximum(0.0, benefit), 0.0)


def premium(
    buy_private: DiscreteAction,
    is_married: IntND,
    predicted_hcc_insurer: FloatND,
    private_premium_intercept: FloatND,
    premium_predicted_hcc: ScalarFloat,
    premium_markup: ScalarFloat,
    premium_minimum: FloatND,
) -> FloatND:
    """Compute the private non-group premium.

    `max(minimum[married], p0 + b (1 + markup) E[insurer cost])` when buying
    private cover, 0 when uninsured (`buy_private=no`). The insurer cannot
    price health or marital status beyond the predicted cost and the
    married minimum.
    """
    raw = (
        private_premium_intercept
        + premium_predicted_hcc * (1.0 + premium_markup) * predicted_hcc_insurer
    )
    return jnp.where(
        buy_private == BuyPrivate.yes,
        jnp.maximum(premium_minimum[is_married], raw),
        0.0,
    )


# Bracket and step count of the intercept bisection. The bracket matches the
# reference implementation; 100 halvings shrink it below float64 resolution.
_INTERCEPT_BRACKET = (-50_000.0, 10_000.0)
_N_INTERCEPT_BISECTIONS = 100


def private_premium_intercept(
    premium_predicted_hcc: ScalarFloat,
    premium_markup: ScalarFloat,
    premium_minimum: FloatND,
    premium_sample_insurer_cost: FloatND,
    premium_sample_is_married: IntND,
) -> FloatND:
    """Solve the private premium intercept from the insurer's zero-profit condition.

    On the sample of private buyers under 65 in the data, the intercept `p0`
    makes premium revenue equal expected payouts plus the markup:

    ```
    mean_i max(minimum[married_i], p0 + b (1 + markup) cost_i)
        = (1 + markup) mean_i cost_i
    ```

    The left side is continuous and non-decreasing in `p0`; bisection on
    `[-50000, 10000]` finds the root. Without a sign change on the bracket
    the result is NaN, so an unpriceable slope `b` fails loudly downstream.
    """
    slope = premium_predicted_hcc * (1.0 + premium_markup)
    minimum = premium_minimum[premium_sample_is_married]
    target = (1.0 + premium_markup) * jnp.mean(premium_sample_insurer_cost)

    def excess_revenue(intercept: FloatND) -> FloatND:
        premiums = jnp.maximum(minimum, intercept + slope * premium_sample_insurer_cost)
        return jnp.mean(premiums) - target

    def halve(
        _: jax.Array, bracket: tuple[FloatND, FloatND]
    ) -> tuple[FloatND, FloatND]:
        low, high = bracket
        mid = 0.5 * (low + high)
        too_high = excess_revenue(mid) > 0.0
        return jnp.where(too_high, low, mid), jnp.where(too_high, mid, high)

    low0 = jnp.asarray(_INTERCEPT_BRACKET[0])
    high0 = jnp.asarray(_INTERCEPT_BRACKET[1])
    low, high = jax.lax.fori_loop(0, _N_INTERCEPT_BISECTIONS, halve, (low0, high0))
    bracketed = (excess_revenue(low0) <= 0.0) & (excess_revenue(high0) >= 0.0)
    return jnp.where(bracketed, 0.5 * (low + high), jnp.nan)


def private_cover_paid_in_full(
    buy_private: DiscreteAction,
    premium_default: FloatND,
) -> BoolND:
    """Private cover is a feasible choice only when its premium is paid in full.

    A household whose resources above the consumption floor fall short of the
    premium cannot buy private cover: the government does not finance private
    insurance through floor transfers, and an unpaid premium buys no private
    cover. Not buying is always feasible.
    """
    return (buy_private == BuyPrivate.no) | (premium_default <= 0.0)


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


def negative_oop_costs(oop_costs: FloatND) -> FloatND:
    """Return the out-of-pocket bill as the shock added to assets."""
    return -oop_costs


def insured_oop(
    total_health_costs: FloatND,
    is_married: IntND,
    deductible: ScalarFloat,
    coinsurance_rate: ScalarFloat,
    oop_max: ScalarFloat,
) -> FloatND:
    """Compute OOP costs under the regime's plan.

    `deductible` and `oop_max` are the single household's; couples'
    deductible and OOP maximum are twice as large.
    """
    couple_scale = 1.0 + is_married
    return oop_costs(
        total_health_costs,
        deductible * couple_scale,
        coinsurance_rate,
        oop_max * couple_scale,
    )


def primary_oop(
    total_health_costs: FloatND,
    buy_private: DiscreteAction,
    is_married: IntND,
    deductible: ScalarFloat,
    coinsurance_rate: ScalarFloat,
    oop_max: ScalarFloat,
) -> FloatND:
    """Compute primary OOP costs for regimes with the private-cover choice.

    Private buyers pay `insured_oop`; when uninsured (`buy_private=no`), OOP
    equals total health costs (no coverage).
    """
    insured = insured_oop(
        total_health_costs, is_married, deductible, coinsurance_rate, oop_max
    )
    return jnp.where(buy_private == BuyPrivate.yes, insured, total_health_costs)


def is_medicaid_eligible(is_ssi_eligible: BoolND) -> BoolND:
    """Baseline: Medicaid eligibility equals SSI eligibility."""
    return is_ssi_eligible


def target_his(
    his: IntND,
    labor_supply: DiscreteAction,
    is_ssi_eligible: BoolND,
) -> IntND:
    """Return the HIS class of the surviving target regime.

    Mirrors the cross-HIS branches inside `_make_transition_canwork` (retiree,
    tied, nongroup): tied agents who stop working become nongroup, and
    categorically (SSI-) Medicaid-eligible agents are overridden to nongroup.
    The ACA Medicaid expansion leaves employer coverage untouched. Used by
    `imputed_pension_wealth_next_period` to look up next-period imputation
    coefficients at the target's HIS.
    """
    tied_to_ng = (his == HealthInsuranceState.tied) & (
        labor_supply == LaborSupply.do_not_work
    )
    return jnp.where(
        tied_to_ng | is_ssi_eligible,
        HealthInsuranceState.nongroup,
        his,
    ).astype(jnp.int32)


def target_his_forcedout(
    his: IntND,
    is_ssi_eligible: BoolND,
) -> IntND:
    """Return the HIS class of the surviving target regime in forced-out regimes.

    Forced-out regimes have no labor-supply choice, and tied agents have
    already moved to nongroup before the forced-out age, so the only HIS
    override is categorical (SSI) Medicaid eligibility → nongroup. Used by
    `imputed_pension_wealth_next_period` to look up next-period imputation
    coefficients at the target's HIS.
    """
    return jnp.where(
        is_ssi_eligible,
        HealthInsuranceState.nongroup,
        his,
    ).astype(jnp.int32)


def oop_with_medicaid(
    primary_oop: FloatND,
    is_medicaid_eligible: BoolND,
    deductible_medicaid: ScalarFloat,
    coinsurance_rate_medicaid: ScalarFloat,
    oop_max_medicaid: ScalarFloat,
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


def oop_with_medicaid_replacing_private(
    total_health_costs: FloatND,
    primary_oop: FloatND,
    is_medicaid_eligible: BoolND,
    deductible_medicaid: ScalarFloat,
    coinsurance_rate_medicaid: ScalarFloat,
    oop_max_medicaid: ScalarFloat,
) -> FloatND:
    """Compute OOP costs in the non-group regimes before Medicare.

    A Medicaid-eligible household is covered by Medicaid alone: Medicaid is
    the payer of last resort and does not cover private premia, so it can
    neither hold private cover nor stay uninsured. Its OOP is Medicaid's
    cost-sharing on total costs, whatever its `buy_private` choice. Other
    households pay the primary (private or uninsured) OOP.
    """
    medicaid_oop = oop_costs(
        total_health_costs=total_health_costs,
        deductible=deductible_medicaid,
        coinsurance_rate=coinsurance_rate_medicaid,
        oop_max=oop_max_medicaid,
    )
    return jnp.where(is_medicaid_eligible, medicaid_oop, primary_oop)


def medicaid_adjusted_premium(
    plan_premium: FloatND,
    is_medicaid_eligible: BoolND,
) -> FloatND:
    """Compute the premium a non-group household pays.

    A Medicaid-eligible household pays no premium: before Medicare its only
    cover is Medicaid, whose premium is zero; with Medicare, Medicaid pays
    the Medicare premium.
    """
    return jnp.where(is_medicaid_eligible, 0.0, plan_premium)


def hcc_insurer_predicted(
    period: Period,
    is_married: IntND,
    good_health: IntND,
    hcc_persistent: ContinuousState,
    predicted_hcc_insurer_table: FloatND,
    hcc_persistent_grid: FloatND,
) -> FloatND:
    """Interpolate the private insurer's expected cost for the household.

    `hcc_persistent` is last period's persistent shock, so the table holds
    this period's E[total_costs - oop_costs | age, married, good health,
    lagged persistent node] on `[period, is_married, good_health, node]`. Linear
    interpolation handles off-grid values during simulation (where
    draw_shock returns continuous AR1 values).
    """
    return jnp.interp(
        hcc_persistent,
        hcc_persistent_grid,
        predicted_hcc_insurer_table[period, is_married, good_health],
    )


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

    For each lagged persistent state i, integrate over this period's
    (persistent j, transitory k) weighted by transition probs and quadrature
    weights.
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
    next_hcc_persistent: ContinuousState,
    next_hcc_transitory: ContinuousState,
    std_xsect_persistent: ScalarFloat,
) -> FloatND:
    """Compute this period's total health care costs from the log-normal model.

    Costs are realised after the period's choices. The persistent and
    transitory shocks that drive them are the draws the transition makes,
    `next_hcc_persistent` and `next_hcc_transitory`: the next period carries
    them as its lagged shocks. Age, marital status and health are this
    period's.

    ``log_mean`` and ``log_std`` are ``pd.Series`` with ``(age, is_married,
    good_health)`` MultiIndex, resolved by pylcm via ``derived_categoricals``.
    """
    std_trans = jnp.sqrt(1.0 - std_xsect_persistent**2)
    return jnp.exp(
        log_mean[period, is_married, good_health]
        + log_std[period, is_married, good_health]
        * (next_hcc_persistent * std_xsect_persistent + next_hcc_transitory * std_trans)
    )
