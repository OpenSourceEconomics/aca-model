"""ACA health insurance functions: mandate, subsidies, cost-sharing, Medicaid.

Also provides ACA-aware versions of `cash_on_hand` and `primary_oop` that
accept ACA policy outputs (premium subsidies, mandate penalty, cost-sharing
scale factor). These replace the simpler baseline versions via function
swapping in the regime DAG.
"""

from collections.abc import Mapping
from enum import Enum, auto
from typing import Any, cast

import jax.numpy as jnp
from lcm.params import MappingLeaf
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarBool,
    ScalarFloat,
)

from aca_model.baseline.health_insurance import (
    BuyPrivate,
    oop_costs,
    share_below_threshold,
)


class PolicyVariant(Enum):
    """ACA policy variant for counterfactual analysis."""

    ACA = auto()
    ACA_NO_MANDATE = auto()
    ACA_NO_MEDICAID_EXPANSION = auto()
    ACA_NO_MEDICAID_EXPANSION_NO_MANDATE = auto()
    ACA_ONLY_MEDICAID_EXPANSION = auto()


def mandate_penalty(
    gross_income: FloatND,
    spousal_income: DiscreteState,
    buy_private: DiscreteAction,
    medicaid_eligibility_share: FloatND,
    mandate_schedule: MappingLeaf,
) -> FloatND:
    """Compute individual mandate penalty for the uninsured.

    Penalty = clip(income * income_fraction, min, max) if uninsured and
    income above exemption threshold; 0 otherwise. Medicaid is
    minimum-essential coverage, so the penalty is scaled down by the
    Medicaid-eligibility share — the Medicaid-eligible owe no penalty.
    """
    sched = cast("Mapping[str, Any]", mandate_schedule.data)
    is_uninsured = buy_private == BuyPrivate.no
    exempt = gross_income < sched["exempt_income"][spousal_income]
    raw = jnp.clip(
        gross_income * sched["income_fraction"],
        sched["min"],
        sched["max"],
    )
    penalty = jnp.where(is_uninsured & ~exempt, raw, 0.0)
    return penalty * (1.0 - medicaid_eligibility_share)


def premium_subsidy(
    hic_premium: FloatND,
    gross_income: FloatND,
    spousal_income: DiscreteState,
    buy_private: DiscreteAction,
    medicaid_eligibility_share: FloatND,
    premium_credit_schedule: MappingLeaf,
) -> FloatND:
    """Compute ACA premium tax credit (advance premium subsidy).

    Piecewise-linear interpolation of applicable income percentage on
    FPL kink points, subsidy = max(0, premium - income * applicable_rate).
    Return 0 when buy_private==no or when income is outside the 100-400% FPL
    range. The subsidy is scaled down by the Medicaid-eligibility share
    (Medicaid is minimum-essential coverage, so no exchange subsidy applies).
    """
    sched = cast("Mapping[str, Any]", premium_credit_schedule.data)
    kinks = sched["kinks"]  # [n_kinks, 3]
    frac_income = sched["frac_income"]  # [n_kinks]

    sp_kinks = kinks[:, spousal_income]
    applicable_rate = jnp.interp(gross_income, sp_kinks, frac_income)
    subsidy = jnp.maximum(0.0, hic_premium - gross_income * applicable_rate)

    in_range = (gross_income >= sp_kinks[0]) & (gross_income < sp_kinks[-1])
    is_insured = buy_private == BuyPrivate.yes
    gated = jnp.where(is_insured & in_range, subsidy, 0.0)
    return gated * (1.0 - medicaid_eligibility_share)


def cost_sharing(
    gross_income: FloatND,
    spousal_income: DiscreteState,
    buy_private: DiscreteAction,
    medicaid_eligibility_share: FloatND,
    cost_sharing_schedule: MappingLeaf,
) -> FloatND:
    """Compute ACA cost-sharing reduction scale factor.

    Bracket lookup on FPL kink points to step-function scale factor.
    Applied to deductible, coinsurance, and OOP max. Returns the neutral
    scale 1.0 when buy_private==no (no reduction for uninsured); for the
    insured the scale is blended toward the neutral 1.0 by the
    Medicaid-eligibility share (Medicaid cost-sharing is handled separately).
    """
    sched = cast("Mapping[str, Any]", cost_sharing_schedule.data)
    kinks = sched["kinks"]  # [n_kinks, 3]
    factors = sched["factors"]  # [n_kinks + 1]
    bracket = jnp.searchsorted(kinks[:, spousal_income], gross_income, side="right")
    scale = factors[bracket]
    is_insured = buy_private == BuyPrivate.yes
    blended = scale * (1.0 - medicaid_eligibility_share) + medicaid_eligibility_share
    return jnp.where(is_insured, blended, 1.0)


def medicaid_eligibility_share(
    ssi_eligibility_share: FloatND,
    aca_magi: FloatND,
    spousal_income: DiscreteState,
    crossed_oamc_threshold: ScalarBool,
    is_disabled: BoolND,
    medicaid_schedule: MappingLeaf,
) -> FloatND:
    """Medicaid eligibility share over two tracks under ACA expansion.

    The household is eligible through either track; the share is the
    probabilistic union of the two track shares:

    - **Categorical** (SSI-linked): `ssi_eligibility_share`, the smooth share
      combining the aged-or-disabled gate with the SSI asset and income tests.
    - **Expansion** (ACA): the under-65, non-disabled population with MAGI
      below the expansion threshold (138% FPL encoded in
      `medicaid_schedule["income_threshold"]`). The `(aged | disabled)`
      categorical gate stays hard — expansion never reaches the aged or the
      disabled, who stay on the categorical track with its asset test — and
      only the MAGI income test is smoothed by the same quintic-smoothstep
      band as the baseline tests (`share_below_threshold`).

    Expansion uses full-count MAGI, not the half-counted SSI
    `countable_income`. Combining the two shares as `s + (1 - s) · e` is the
    probability that at least one track qualifies, keeping the result in
    `[0, 1]` and reducing to the categorical share when expansion is shut off.
    """
    sched = cast("Mapping[str, Any]", medicaid_schedule.data)
    threshold = sched["income_threshold"]
    expansion = ((~crossed_oamc_threshold) & (~is_disabled)) * share_below_threshold(
        aca_magi, threshold[spousal_income]
    )
    return ssi_eligibility_share + (1.0 - ssi_eligibility_share) * expansion


def premium_default(
    assets: ContinuousState,
    after_tax_income: FloatND,
    ssi_benefit: FloatND,
    hic_premium: FloatND,
    hic_premium_subsidy: FloatND,
    consumption_dollars_floor: FloatND,
) -> FloatND:
    """Compute the unpaid (defaulted) part of the ACA net premium.

    The premium subject to default is the household's net premium, after the
    ACA premium tax credit: `net_premium = hic_premium - hic_premium_subsidy`.
    A household pays the net premium only up to what it can afford while
    staying at the consumption floor; it defaults on the rest as
    uncompensated care:

    ```
    affordable_premium = max(0, resources - consumption_dollars_floor)
    premium_default    = max(0, net_premium - affordable_premium)
    ```

    The mandate penalty is a separate non-defaultable tax and is excluded
    from this computation.
    """
    net_premium = hic_premium - hic_premium_subsidy
    resources = assets + after_tax_income + ssi_benefit
    affordable_premium = jnp.maximum(0.0, resources - consumption_dollars_floor)
    return jnp.maximum(0.0, net_premium - affordable_premium)


def cash_on_hand(
    assets: ContinuousState,
    after_tax_income: FloatND,
    ssi_benefit: FloatND,
    hic_premium: FloatND,
    hic_premium_subsidy: FloatND,
    premium_default: FloatND,
    mandate_penalty: FloatND,
) -> FloatND:
    """Compute cash on hand with ACA premium subsidies and mandate penalty.

    Only the affordable part of the net premium
    (`hic_premium - hic_premium_subsidy - premium_default`) leaves
    cash-on-hand; the defaulted part is never paid. The mandate penalty is a
    separate non-defaultable tax that always leaves cash-on-hand.

    OOP health costs are NOT deducted here — they are deducted from
    next-period assets instead, matching the timing where HCC shocks are
    integrated over (agent does not condition consumption on OOP).
    """
    net_premium = hic_premium - hic_premium_subsidy
    effective_premium = net_premium - premium_default
    return assets + after_tax_income + ssi_benefit - effective_premium - mandate_penalty


def primary_oop(
    total_health_costs: FloatND,
    cost_sharing_scale: FloatND,
    buy_private: DiscreteAction,
    deductible: ScalarFloat,
    coinsurance_rate: ScalarFloat,
    oop_max: ScalarFloat,
) -> FloatND:
    """Compute primary OOP costs with ACA cost-sharing reductions.

    Scale deductible, coinsurance rate, and OOP max by the cost-sharing
    factor before applying the standard OOP calculation. When uninsured
    (`buy_private=no`), OOP equals total health costs (no coverage).
    """
    insured_oop = oop_costs(
        total_health_costs,
        deductible * cost_sharing_scale,
        coinsurance_rate * cost_sharing_scale,
        oop_max * cost_sharing_scale,
    )
    return jnp.where(buy_private == BuyPrivate.yes, insured_oop, total_health_costs)
