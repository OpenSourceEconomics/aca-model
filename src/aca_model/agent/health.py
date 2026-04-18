"""Health status types and transitions.

HealthWithDisability (3-state: disabled/bad/good) is used in pre-65 regimes.
Health (2-state: bad/good) is used in post-65 regimes (mc=oamc).
"""

import jax.numpy as jnp
from lcm import categorical
from lcm.typing import DiscreteState, FloatND, IntND, Period


@categorical(ordered=True)
class HealthWithDisability:
    disabled: int
    bad: int
    good: int


@categorical(ordered=True)
class Health:
    bad: int
    good: int


@categorical(ordered=True)
class GoodHealth:
    """Derived categorical for good_health DAG output (0=no, 1=yes)."""

    no: int
    yes: int


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
