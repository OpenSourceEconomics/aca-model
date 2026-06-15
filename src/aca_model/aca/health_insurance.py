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

from aca_model.baseline.health_insurance import BuyPrivate, oop_costs


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
    is_medicaid_eligible: BoolND,
    mandate_schedule: MappingLeaf,
) -> FloatND:
    """Compute individual mandate penalty for the uninsured.

    Penalty = clip(income * income_fraction, min, max) if uninsured and
    income above exemption threshold; 0 otherwise. Medicaid is
    minimum-essential coverage, so the Medicaid-eligible owe no penalty.
    """
    sched = cast("Mapping[str, Any]", mandate_schedule.data)
    is_uninsured = buy_private == BuyPrivate.no
    exempt = gross_income < sched["exempt_income"][spousal_income]
    raw = jnp.clip(
        gross_income * sched["income_fraction"],
        sched["min"],
        sched["max"],
    )
    return jnp.where(is_uninsured & ~exempt & ~is_medicaid_eligible, raw, 0.0)


def premium_subsidy(
    hic_premium: FloatND,
    gross_income: FloatND,
    spousal_income: DiscreteState,
    buy_private: DiscreteAction,
    is_medicaid_eligible: BoolND,
    premium_credit_schedule: MappingLeaf,
) -> FloatND:
    """Compute ACA premium tax credit (advance premium subsidy).

    Piecewise-linear interpolation of applicable income percentage on
    FPL kink points, subsidy = max(0, premium - income * applicable_rate).
    Return 0 when buy_private==no, when income is outside the 100-400% FPL
    range, or when Medicaid-eligible (Medicaid is minimum-essential coverage,
    so no exchange subsidy applies).
    """
    sched = cast("Mapping[str, Any]", premium_credit_schedule.data)
    kinks = sched["kinks"]  # [n_kinks, 3]
    frac_income = sched["frac_income"]  # [n_kinks]

    sp_kinks = kinks[:, spousal_income]
    applicable_rate = jnp.interp(gross_income, sp_kinks, frac_income)
    subsidy = jnp.maximum(0.0, hic_premium - gross_income * applicable_rate)

    in_range = (gross_income >= sp_kinks[0]) & (gross_income < sp_kinks[-1])
    is_insured = buy_private == BuyPrivate.yes
    return jnp.where(is_insured & in_range & ~is_medicaid_eligible, subsidy, 0.0)


def cost_sharing(
    gross_income: FloatND,
    spousal_income: DiscreteState,
    buy_private: DiscreteAction,
    is_medicaid_eligible: BoolND,
    cost_sharing_schedule: MappingLeaf,
) -> FloatND:
    """Compute ACA cost-sharing reduction scale factor.

    Bracket lookup on FPL kink points to step-function scale factor.
    Applied to deductible, coinsurance, and OOP max. Returns the neutral
    scale 1.0 when buy_private==no (no reduction for uninsured) or when
    Medicaid-eligible (Medicaid cost-sharing is handled separately).
    """
    sched = cast("Mapping[str, Any]", cost_sharing_schedule.data)
    kinks = sched["kinks"]  # [n_kinks, 3]
    factors = sched["factors"]  # [n_kinks + 1]
    bracket = jnp.searchsorted(kinks[:, spousal_income], gross_income, side="right")
    scale = factors[bracket]
    is_insured = buy_private == BuyPrivate.yes
    return jnp.where(is_insured & ~is_medicaid_eligible, scale, 1.0)


def is_medicaid_eligible(
    is_ssi_eligible: BoolND,
    aca_magi: FloatND,
    spousal_income: DiscreteState,
    is_aged: ScalarBool,
    is_disabled: BoolND,
    medicaid_schedule: MappingLeaf,
) -> BoolND:
    """Determine Medicaid eligibility on two tracks under ACA expansion.

    A household is eligible when it qualifies on either track:

    - **Categorical** (SSI-linked): `is_ssi_eligible`, which applies the
      aged-or-disabled gate plus the SSI asset and income tests.
    - **Expansion** (ACA): the under-65, non-disabled population with MAGI
      below the expansion threshold (138% FPL encoded in
      `medicaid_schedule["income_threshold"]`). Expansion never reaches the
      aged or the disabled; they stay on the categorical track with its
      asset test.

    Expansion uses full-count MAGI, not the half-counted SSI
    `countable_income`.
    """
    sched = cast("Mapping[str, Any]", medicaid_schedule.data)
    threshold = sched["income_threshold"]
    expansion = (~is_aged) & (~is_disabled) & (aca_magi < threshold[spousal_income])
    return is_ssi_eligible | expansion


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
