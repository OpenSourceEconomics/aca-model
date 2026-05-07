"""Preference functions: utility, leisure, bequests.

Ported from struct-ret/src/model/preferences_utility.py and auxiliaries.py.
"""

import jax.numpy as jnp
from lcm import categorical
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    IntND,
)

from aca_model.agent.labor_market import LaggedLaborSupply


@categorical(ordered=False)
class PrefType:
    """Unobserved preference type for heterogeneity in estimation."""

    type_0: int
    type_1: int
    type_2: int


@categorical(ordered=False)
class BenchmarkPrefType:
    """Compact 2-type variant of `PrefType` used by the ASV benchmark.

    The benchmark model exercises the full 18-regime DAG with compact
    continuous grids; shrinking `pref_type` from 3 to 2 types cuts the
    partition-axis cardinality too, so the benchmark finishes faster
    without changing anything structural about the kernel being
    measured.
    """

    type_0: int
    type_1: int


def positive_leisure(leisure: FloatND) -> BoolND:
    """Return True where leisure is strictly positive."""
    return leisure > 0


def equivalence_scale(is_married: IntND, exponent: float) -> FloatND:
    """Return the equivalence scale for household size adjustment.

    Single (is_married=False) → 1.0, married (is_married=True) → 2^exponent.
    """
    return jnp.where(is_married, 2.0**exponent, 1.0)


def leisure(
    working_hours_value: FloatND,
    age: int,
    good_health: IntND,
    lagged_labor_supply: DiscreteState,
    time_endowment: float,
    leisure_cost_of_bad_health: float,
    fixed_cost_of_work_intercept: float,
    fixed_cost_of_work_age_trend: float,
    labor_force_reentry_cost: float,
    reference_age: int,
) -> FloatND:
    """Compute leisure given hours worked and state variables.

    Fixed cost of work is age-dependent: intercept + trend * (age - reference_age).
    Reentry cost applies when returning to work after not working last period.
    Working status is derived from working_hours_value > 0.
    """
    is_working = working_hours_value > 0.0
    health_loss = jnp.where(good_health, 0.0, leisure_cost_of_bad_health)

    fixed_cost = fixed_cost_of_work_intercept + fixed_cost_of_work_age_trend * (
        age - reference_age
    )
    reentry_cost = jnp.where(
        lagged_labor_supply == LaggedLaborSupply.did_not_work,
        labor_force_reentry_cost,
        0.0,
    )
    work_loss = jnp.where(
        is_working, working_hours_value + fixed_cost + reentry_cost, 0.0
    )

    return time_endowment - health_loss - work_loss


def leisure_tied(
    working_hours_value: FloatND,
    age: int,
    good_health: IntND,
    time_endowment: float,
    leisure_cost_of_bad_health: float,
    fixed_cost_of_work_intercept: float,
    fixed_cost_of_work_age_trend: float,
    reference_age: int,
) -> FloatND:
    """Compute leisure for tied regimes (no reentry cost, no lagged_labor_supply)."""
    health_loss = jnp.where(good_health, 0.0, leisure_cost_of_bad_health)
    fixed_cost = fixed_cost_of_work_intercept + fixed_cost_of_work_age_trend * (
        age - reference_age
    )
    work_loss = jnp.where(
        working_hours_value > 0.0, working_hours_value + fixed_cost, 0.0
    )
    return time_endowment - health_loss - work_loss


def leisure_retired(
    good_health: IntND,
    time_endowment: float,
    leisure_cost_of_bad_health: float,
) -> FloatND:
    """Compute leisure for retired agents (no work)."""
    health_loss = jnp.where(good_health, 0.0, leisure_cost_of_bad_health)
    return time_endowment - health_loss


def utility(
    consumption: ContinuousAction,
    leisure: FloatND,
    pref_type: DiscreteState,
    consumption_weight: FloatND,
    coefficient_rra: FloatND,
    equivalence_scale: FloatND,
    utility_scale_factor: FloatND,
) -> FloatND:
    """Within-period utility: CES aggregator over consumption and leisure.

    u = utility_scale_factor * ((c/eq_scale)^α * l^(1-α))^(1-γ) / (1-γ)
    with log case for γ=1. `consumption_weight` and `coefficient_rra` are
    pref-type-indexed Series sourced directly from params; `utility_scale_factor`
    is a regime-function output (already a per-cell scalar — must NOT be
    re-indexed by pref_type, see `aca_model.agent.preferences.utility_scale_factor`
    for why).
    """
    alpha = consumption_weight[pref_type]
    gamma = coefficient_rra[pref_type]
    equiv_cons = consumption / equivalence_scale
    composite = equiv_cons**alpha * leisure ** (1.0 - alpha)

    one_minus_gamma = jnp.where(jnp.isclose(gamma, 1.0), 1.0, 1.0 - gamma)
    u = jnp.where(
        jnp.isclose(gamma, 1.0),
        jnp.log(composite),
        composite**one_minus_gamma / one_minus_gamma,
    )
    return u * utility_scale_factor


def discount_factor(
    pref_type: DiscreteState,
    discount_factor_by_type: FloatND,
) -> FloatND:
    """Per-period discount factor indexed by preference type.

    Wired as a DAG function so pylcm's default Bellman aggregator can
    consume the scalar it returns (pylcm's `Q_and_F` resolves any H
    argument that is also a `regime.functions` name as a DAG output).
    """
    return discount_factor_by_type[pref_type]


def utility_scale_factor(
    pref_type: DiscreteState,
    average_consumption: float,
    consumption_weight: FloatND,
    coefficient_rra: FloatND,
    time_endowment: float,
    fixed_cost_of_work_intercept: float,
    fixed_cost_of_work_age_trend: float,
    scale_reference_hours: float,
    reference_age: int,
    scale_reference_age: int,
) -> FloatND:
    """Compute the scale factor so utility is approximately 1 at typical values.

    Returns the scalar for the cell's `pref_type`. Mirrors the `discount_factor`
    pattern: take the state as input, return a per-cell scalar. Registering this
    as a regime function and then doing `utility_scale_factor[pref_type]` in a
    downstream consumer is invalid — pylcm broadcasts function outputs to
    per-cell scalars before consumption, and the validator in
    `lcm.regime_building.validation` raises on that clash.
    """
    alpha = consumption_weight[pref_type]
    gamma = coefficient_rra[pref_type]
    age_offset = scale_reference_age - reference_age
    average_leisure = (
        time_endowment
        - scale_reference_hours
        - (fixed_cost_of_work_intercept + fixed_cost_of_work_age_trend * age_offset)
    )
    u_cons = average_consumption**alpha
    u_leisure = average_leisure ** (1.0 - alpha)

    one_minus_gamma = jnp.where(jnp.isclose(gamma, 1.0), 1.0, 1.0 - gamma)
    raw = jnp.where(
        jnp.isclose(gamma, 1.0),
        jnp.log(u_cons * u_leisure),
        (u_cons * u_leisure) ** one_minus_gamma / one_minus_gamma,
    )
    return jnp.abs(1.0 / raw)


def scaled_bequest_weight(
    bequest_weight: float,
    consumption_weight: float,
    coefficient_rra: float,
    time_endowment: float,
    time_discount_factor: float,
    rate_of_return: float,
) -> FloatND:
    """Transform raw bequest weight into the form used in the bequest function.

    result = T^ξ * (bw / (1+r-bw))^ξ₂ / β
    where ξ = (1-α)(1-γ) and ξ₂ = α(1-γ) - 1.
    """
    xi = (1.0 - consumption_weight) * (1.0 - coefficient_rra)
    xi2 = consumption_weight * (1.0 - coefficient_rra) - 1.0
    safe_bw = jnp.where(bequest_weight > 0.0, bequest_weight, 1.0)
    return jnp.where(
        bequest_weight > 0.0,
        time_endowment**xi
        * (safe_bw / (1.0 + rate_of_return - safe_bw)) ** xi2
        / time_discount_factor,
        0.0,
    )


def bequest(
    assets: ContinuousState,
    pref_type: DiscreteState,
    bequest_shifter: float,
    scaled_bequest_weight: float,
    consumption_weight: FloatND,
    coefficient_rra: FloatND,
    utility_scale_factor: FloatND,
) -> FloatND:
    """Bequest function for terminal/dead states.

    bequest = scale * bwt * (max(0,a) + shifter)^(α*(1-γ)) / (1-γ)
    `consumption_weight` and `coefficient_rra` are pref-type-indexed Series
    from params; `utility_scale_factor` is a regime-function output (already a
    per-cell scalar — must NOT be re-indexed by pref_type).
    """
    alpha = consumption_weight[pref_type]
    gamma = coefficient_rra[pref_type]
    assets_shifted = jnp.maximum(0.0, assets) + bequest_shifter

    one_minus_gamma = jnp.where(jnp.isclose(gamma, 1.0), 1.0, 1.0 - gamma)
    val = jnp.where(
        jnp.isclose(gamma, 1.0),
        jnp.log(assets_shifted),
        assets_shifted ** (one_minus_gamma * alpha) / one_minus_gamma,
    )
    return val * scaled_bequest_weight * utility_scale_factor
