"""Labor market types and functions: hours, wages, earnings, transitions.

Ported from struct-ret/src/model/auxiliaries.py.
"""

import jax.numpy as jnp
from lcm import categorical
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    IntND,
    Period,
)


@categorical(ordered=True)
class LaborSupply:
    do_not_work: int
    h1000: int
    h1500: int
    h2000: int
    h2500: int


@categorical(ordered=False)
class LaggedLaborSupply:
    did_not_work: int
    worked: int


@categorical(ordered=False)
class SpousalIncome:
    single: int
    married_no_inc: int
    married_has_inc: int


HOURS_VALUES = jnp.array([0.0, 1000.0, 1500.0, 2000.0, 2500.0])


def working_hours_value(labor_supply: DiscreteAction) -> FloatND:
    """Map labor supply choice to annual hours worked."""
    return HOURS_VALUES[labor_supply]


def income(
    log_ft_wage_res: ContinuousState,
    labor_supply: DiscreteAction,
    period: Period,
    good_health: IntND,
    log_ft_wage_mean: FloatND,
    log_ft_wage_std: FloatND,
    adj_wage_hours_exp: float,
    adj_wage_hours_int: float,
) -> FloatND:
    """Labor income with wage-hours interaction (French & Jones 2011).

    income = wage * hours^(1 + exp) * int^(-exp)

    ``log_ft_wage_mean`` is a ``pd.Series`` with ``(age, good_health)`` index,
    resolved by pylcm via ``derived_categoricals``.
    """
    wage = jnp.exp(
        log_ft_wage_mean[period, good_health] + log_ft_wage_std * log_ft_wage_res
    )
    hours = HOURS_VALUES[labor_supply]
    return jnp.where(
        hours > 0.0,
        wage
        * hours ** (1.0 + adj_wage_hours_exp)
        * adj_wage_hours_int ** (-adj_wage_hours_exp),
        0.0,
    )


def next_lagged_supply(labor_supply: DiscreteAction) -> DiscreteState:
    """Deterministic transition: did the agent work this period?"""
    return jnp.where(
        labor_supply == LaborSupply.do_not_work,
        LaggedLaborSupply.did_not_work,
        LaggedLaborSupply.worked,
    )


@categorical(ordered=True)
class IsMarried:
    """Derived categorical for is_married DAG output (0=no, 1=yes)."""

    no: int
    yes: int


def is_married(spousal_income: DiscreteState) -> IntND:
    """Derive binary married indicator from spousal income category.

    single → 0, married (with or without income) → 1.
    """
    return jnp.int32(spousal_income > SpousalIncome.single)


def next_spousal_income(
    spousal_income: DiscreteState,
    period: Period,
    spousal_income_trans_probs: FloatND,
) -> FloatND:
    """Stochastic spousal income transition."""
    return spousal_income_trans_probs[period, spousal_income]
