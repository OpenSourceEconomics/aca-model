"""Tax computation functions: income tax, payroll tax, SS benefit taxation.

Ported from struct-ret/src/model/baseline/soc_sec_pensions_taxes.py.

These functions use vectorized bracket lookups via jnp.searchsorted
instead of the original Numba for-loops.
"""

from collections.abc import Mapping
from typing import Any, cast

import jax.numpy as jnp
import lcm
from lcm.params import MappingLeaf
from lcm.typing import DiscreteState, FloatND, IntND

# Federal income tax bracket edges declared for the NBEGM M1 slice. The
# progressive tax is continuous and piecewise-affine in `gross_income`, kinking
# at each finite bracket edge `income_tax_schedule.brackets_upper[spousal_income,
# k]`; the final edge is the +inf top-bracket sentinel and is not a kink. The
# decorator is metadata-only — brute and DC-EGM solve identically.
_N_INCOME_TAX_KINKS = 7


def gross_income(
    capital_income: FloatND,
    labor_income: FloatND,
    spousal_income: DiscreteState,
    spousal_income_amounts: FloatND,
    taxable_ss_benefit: FloatND,
    pension_benefit: FloatND,
) -> FloatND:
    """Compute gross taxable income from all sources."""
    return (
        capital_income
        + labor_income
        + spousal_income_amounts[spousal_income]
        + taxable_ss_benefit
        + pension_benefit
    )


def taxable_ss_benefit(
    capital_income: FloatND,
    labor_income: FloatND,
    spousal_income: DiscreteState,
    spousal_income_amounts: FloatND,
    ss_benefit: FloatND,
    pension_benefit: FloatND,
    ss_tax_schedule: MappingLeaf,
) -> FloatND:
    """Compute the taxable portion of Social Security benefits.

    Two-tier system: fraction of benefits subject to income tax depends
    on provisional income relative to bracket thresholds.
    """
    sched = cast("Mapping[str, Any]", ss_tax_schedule.data)
    prov_income = (
        capital_income
        + labor_income
        + spousal_income_amounts[spousal_income]
        + sched["ben_fraction_prov_income"] * ss_benefit
        + pension_benefit
    )

    # Tier 1
    tier_1_excess = jnp.maximum(
        0.0,
        jnp.minimum(prov_income, sched["brackets_upper"][spousal_income, 0])
        - sched["brackets_lower"][spousal_income, 0],
    )
    tier_1 = sched["fraction_considered"][spousal_income, 0] * jnp.minimum(
        ss_benefit, tier_1_excess
    )

    # Tier 2
    tier_2_excess = jnp.maximum(
        0.0, prov_income - sched["brackets_lower"][spousal_income, 1]
    )
    tier_2_raw = (
        tier_1 + sched["fraction_considered"][spousal_income, 1] * tier_2_excess
    )
    tier_2 = jnp.minimum(
        sched["fraction_considered"][spousal_income, 1] * ss_benefit, tier_2_raw
    )

    return jnp.where(
        prov_income < sched["brackets_lower"][spousal_income, 0],
        0.0,
        jnp.where(
            prov_income < sched["brackets_lower"][spousal_income, 1], tier_1, tier_2
        ),
    )


@lcm.piecewise_affine(
    "after_tax_income",
    variable="gross_income",
    breakpoints=tuple(
        lcm.affine_breakpoint(
            "income_tax_schedule.brackets_upper",
            kind="continuous_kink",
            indexed_by="spousal_income",
            static_index=k,
        )
        for k in range(_N_INCOME_TAX_KINKS)
    ),
)
def after_tax_income(
    gross_income: FloatND,
    ss_benefit: FloatND,
    taxable_ss_benefit: FloatND,
    labor_income: FloatND,
    spousal_income: DiscreteState,
    income_tax_schedule: MappingLeaf,
    payroll_tax_schedule: MappingLeaf,
) -> FloatND:
    """Compute after-tax income from all sources.

    Applies federal income tax brackets, then adds back non-taxable SS portion,
    then subtracts payroll tax on labor income.
    """
    sched = cast("Mapping[str, Any]", income_tax_schedule.data)

    # Find income tax bracket
    bracket_id = _find_bracket(gross_income, sched["brackets_upper"][spousal_income])
    bracket_lower = jnp.maximum(
        0.0, sched["brackets_lower"][spousal_income, bracket_id]
    )
    keep_rate = 1.0 - sched["marginal_rates"][spousal_income, bracket_id]

    ati = (
        sched["after_tax_at_lower"][spousal_income, bracket_id]
        + (gross_income - bracket_lower) * keep_rate
    )

    # Add back non-taxable SS portion
    ati = ati + ss_benefit - taxable_ss_benefit

    # Subtract payroll tax
    psched = cast("Mapping[str, Any]", payroll_tax_schedule.data)
    payroll = _payroll_tax(
        labor_income,
        psched["brackets_lower"],
        psched["brackets_upper"],
        psched["marginal_rates"],
        psched["taxes_at_lower"],
    )
    return ati - payroll


def marginal_rate(
    gross_income: FloatND,
    spousal_income: DiscreteState,
    income_tax_schedule: MappingLeaf,
) -> FloatND:
    """Compute the marginal income tax rate for pension adjustments."""
    sched = cast("Mapping[str, Any]", income_tax_schedule.data)
    bracket_id = _find_bracket(gross_income, sched["brackets_upper"][spousal_income])
    return sched["marginal_rates"][spousal_income, bracket_id]


def _find_bracket(income: FloatND, upper_bounds: FloatND) -> IntND:
    """Find the tax bracket index for a given income level."""
    return jnp.searchsorted(upper_bounds, income, side="right")


def _payroll_tax(
    labor_income: FloatND,
    brackets_lower: FloatND,
    brackets_upper: FloatND,
    marginal_rates: FloatND,
    taxes_at_lower: FloatND,
) -> FloatND:
    """Compute payroll tax on labor income."""
    bracket_id = jnp.searchsorted(brackets_upper, labor_income, side="right")
    bracket_lower = jnp.maximum(0.0, brackets_lower[bracket_id])
    return (
        taxes_at_lower[bracket_id]
        + (labor_income - bracket_lower) * marginal_rates[bracket_id]
    )
