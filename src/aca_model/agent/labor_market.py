"""Labor market types and functions: hours, wages, earnings, transitions.

Ported from struct-ret/src/model/auxiliaries.py.
"""

import jax.numpy as jnp
import numpy as np
from lcm import categorical
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    IntND,
    Period,
    ScalarFloat,
    ScalarInt,
)


@categorical(ordered=True)
class LaborSupply:
    do_not_work: ScalarInt
    h1000: ScalarInt
    h1500: ScalarInt
    h2000: ScalarInt
    h2500: ScalarInt


@categorical(ordered=False)
class LaggedLaborSupply:
    did_not_work: ScalarInt
    worked: ScalarInt


@categorical(ordered=False)
class SpousalIncome:
    single: ScalarInt
    married_no_inc: ScalarInt
    married_has_inc: ScalarInt


# Host array, not a module-level JAX array: a device array here would
# reserve the GPU memory pool at import time in every process that imports
# the model. It is converted to a device array at each indexing site, where
# the value folds into the surrounding compiled function.
HOURS_VALUES = np.array([0.0, 1000.0, 1500.0, 2000.0, 2500.0])


def working_hours_value(labor_supply: DiscreteAction) -> FloatND:
    """Map labor supply choice to annual hours worked."""
    return jnp.asarray(HOURS_VALUES)[labor_supply]


def wage(
    log_ft_wage_res: ContinuousState,
    period: Period,
    good_health: IntND,
    log_ft_wage_mean: FloatND,
    log_ft_wage_std: FloatND,
) -> FloatND:
    """Full-time-equivalent wage rate from the age/health profile and AR(1) residual.

    ``log_ft_wage_mean`` is a ``pd.Series`` with ``(age, good_health)`` index,
    resolved by pylcm via ``derived_categoricals``.
    """
    return jnp.exp(
        log_ft_wage_mean[period, good_health] + log_ft_wage_std * log_ft_wage_res
    )


def income(
    wage: FloatND,
    labor_supply: DiscreteAction,
    adj_wage_hours_exp: ScalarFloat,
    adj_wage_hours_int: ScalarFloat,
) -> FloatND:
    """Labor income with wage-hours interaction (French & Jones 2011).

    income = wage * hours^(1 + exp) * int^(-exp)
    """
    hours = jnp.asarray(HOURS_VALUES)[labor_supply]
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

    no: ScalarInt
    yes: ScalarInt


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
