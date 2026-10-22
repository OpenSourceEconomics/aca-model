"""Health status types and transitions.

HealthWithDisability (3-state: disabled/bad/good) is used in pre-65 regimes.
Health (2-state: bad/good) is used in post-65 regimes (mc=oamc).
"""

import jax.numpy as jnp
from lcm import categorical
from lcm.typing import (
    DiscreteAction,
    DiscreteState,
    FloatND,
    IntND,
    Period,
    ScalarInt,
)

from aca_model.agent.labor_market import LaborSupply


@categorical(ordered=True)
class HealthWithDisability:
    disabled: ScalarInt
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=True)
class Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=True)
class GoodHealth:
    """Derived categorical for good_health DAG output (0=no, 1=yes)."""

    no: ScalarInt
    yes: ScalarInt


def is_good_health_3(health: DiscreteState) -> IntND:
    """Integer indicator for HealthWithDisability: 1 if good, 0 otherwise."""
    return jnp.int32(health == HealthWithDisability.good)


def is_good_health_2(health: DiscreteState) -> IntND:
    """Integer indicator for Health: 1 if good, 0 otherwise."""
    return jnp.int32(health == Health.good)


def next_health(
    health: DiscreteState,
    period: Period,
    health_trans_probs: FloatND,
) -> FloatND:
    """Stochastic health transition (same-grid: 3->3 or 2->2)."""
    return health_trans_probs[period, health]


def next_health_cross(
    health: DiscreteState,
    period: Period,
    health_trans_probs_cross: FloatND,
) -> FloatND:
    """Stochastic health transition across grids (3->2).

    Used when pre-65 regimes (HealthWithDisability) transition to post-65
    regimes (Health) at the age-65 boundary.
    """
    return health_trans_probs_cross[period, health]


def next_health_into_dimc(
    health: DiscreteState,
    period: Period,
    health_trans_probs: FloatND,
) -> FloatND:
    """Next health of a household that lands on disability Medicare: disabled.

    Pre-65 Medicare goes to households disabled next period, so landing in a
    `dimc` regime reveals the next health state.
    """
    probs = health_trans_probs[period, health]
    return jnp.zeros_like(probs).at[HealthWithDisability.disabled].set(1.0)


def next_health_into_nomc(
    health: DiscreteState,
    period: Period,
    labor_supply: DiscreteAction,
    health_trans_probs: FloatND,
) -> FloatND:
    """Next health of a pre-65 household that lands without Medicare.

    A household that does not work this period gets disability Medicare
    whenever it is disabled next period, so landing without Medicare means it
    is not disabled: its next health is the transition row with the disabled
    state removed and renormalised. A worker keeps the unconditional row.
    """
    probs = health_trans_probs[period, health]
    off_disability = probs.at[HealthWithDisability.disabled].set(0.0)
    mass = off_disability.sum()
    conditional = off_disability / jnp.where(mass > 0.0, mass, 1.0)
    return jnp.where(labor_supply == LaborSupply.do_not_work, conditional, probs)
