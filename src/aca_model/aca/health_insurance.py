"""ACA health insurance functions: mandate, subsidies, cost-sharing, Medicaid.

Also provides ACA-aware versions of `cash_on_hand` and `primary_oop` that
accept ACA policy outputs (premium subsidies, mandate penalty, cost-sharing
scale factor). These replace the simpler baseline versions via function
swapping in the regime DAG.
"""

from enum import Enum, auto

import jax.numpy as jnp
from lcm.params import MappingLeaf
from lcm.typing import BoolND, ContinuousState, DiscreteAction, DiscreteState, FloatND

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
    mandate_schedule: MappingLeaf,
) -> FloatND:
    """Compute individual mandate penalty for the uninsured.

    Penalty = clip(income * income_fraction, min, max) if uninsured and
    income above exemption threshold; 0 otherwise.
    """
    sched = mandate_schedule.data
    is_uninsured = buy_private == BuyPrivate.no
    exempt = gross_income < sched["exempt_income"][spousal_income]
    raw = jnp.clip(
        gross_income * sched["income_fraction"],
        sched["min"],
        sched["max"],
    )
    return jnp.where(is_uninsured & ~exempt, raw, 0.0)


def premium_subsidy(
    hic_premium: FloatND,
    gross_income: FloatND,
    spousal_income: DiscreteState,
    buy_private: DiscreteAction,
    premium_credit_schedule: MappingLeaf,
) -> FloatND:
    """Compute ACA premium tax credit (advance premium subsidy).

    Piecewise-linear interpolation of applicable income percentage on
    FPL kink points, subsidy = max(0, premium - income * applicable_rate).
    Return 0 when buy_private==no or income outside 100-400% FPL range.
    """
    sched = premium_credit_schedule.data
    kinks = sched["kinks"]  # [n_kinks, 3]
    frac_income = sched["frac_income"]  # [n_kinks]

    sp_kinks = kinks[:, spousal_income]
    applicable_rate = jnp.interp(gross_income, sp_kinks, frac_income)
    subsidy = jnp.maximum(0.0, hic_premium - gross_income * applicable_rate)

    in_range = (gross_income >= sp_kinks[0]) & (gross_income < sp_kinks[-1])
    is_insured = buy_private == BuyPrivate.yes
    return jnp.where(is_insured & in_range, subsidy, 0.0)


def cost_sharing(
    gross_income: FloatND,
    spousal_income: DiscreteState,
    buy_private: DiscreteAction,
    cost_sharing_schedule: MappingLeaf,
) -> FloatND:
    """Compute ACA cost-sharing reduction scale factor.

    Bracket lookup on FPL kink points to step-function scale factor.
    Applied to deductible, coinsurance, and OOP max.
    Return 1.0 when buy_private==no (no reduction for uninsured).
    """
    sched = cost_sharing_schedule.data
    kinks = sched["kinks"]  # [n_kinks, 3]
    factors = sched["factors"]  # [n_kinks + 1]
    bracket = jnp.searchsorted(kinks[:, spousal_income], gross_income, side="right")
    scale = factors[bracket]
    is_insured = buy_private == BuyPrivate.yes
    return jnp.where(is_insured, scale, 1.0)


def is_medicaid_eligible(
    countable_income: FloatND,
    spousal_income: DiscreteState,
    medicaid_schedule: MappingLeaf,
) -> BoolND:
    """Determine ACA Medicaid expansion eligibility: income below 133% FPL.

    Unlike baseline SSI-based Medicaid, ACA expansion uses only income
    (no assets test, no Medicare requirement).
    """
    threshold = medicaid_schedule.data["income_threshold"]
    return countable_income < threshold[spousal_income]


def cash_on_hand(
    assets: ContinuousState,
    after_tax_income: FloatND,
    ssi_benefit: FloatND,
    hic_premium: FloatND,
    hic_premium_subsidy: FloatND,
    mandate_penalty: FloatND,
) -> FloatND:
    """Compute cash on hand with ACA premium subsidies and mandate penalty.

    OOP health costs are NOT deducted here — they are deducted from
    next-period assets instead, matching the timing where HCC shocks are
    integrated over (agent does not condition consumption on OOP).
    """
    return (
        assets
        + after_tax_income
        + ssi_benefit
        - hic_premium
        + hic_premium_subsidy
        - mandate_penalty
    )


def primary_oop(
    total_health_costs: FloatND,
    cost_sharing_scale: FloatND,
    buy_private: DiscreteAction,
    deductible: float,
    coinsurance_rate: float,
    oop_max: float,
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
