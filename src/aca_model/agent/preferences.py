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


def consumption_equiv(
    consumption_unequiv: ContinuousAction,
    equivalence_scale: FloatND,
) -> FloatND:
    """Utility-equivalized consumption."""
    return consumption_unequiv / equivalence_scale


def u_working_life(
    consumption_equiv: FloatND,
    leisure: FloatND,
    consumption_weight: FloatND,
    coefficient_rra: FloatND,
    utility_scale_factor: FloatND,
) -> FloatND:
    """Within-period utility for canwork regimes: CES over consumption and leisure."""
    composite = consumption_equiv**consumption_weight * leisure ** (
        1.0 - consumption_weight
    )

    one_minus_rra = jnp.where(
        jnp.isclose(coefficient_rra, 1.0), 1.0, 1.0 - coefficient_rra
    )
    u = jnp.where(
        jnp.isclose(coefficient_rra, 1.0),
        jnp.log(composite),
        composite**one_minus_rra / one_minus_rra,
    )
    return u * utility_scale_factor


def u_retired(
    consumption_equiv: FloatND,
    good_health: IntND,
    consumption_weight: FloatND,
    coefficient_rra: FloatND,
    utility_scale_factor: FloatND,
    time_endowment: float,
    leisure_cost_of_bad_health: float,
) -> FloatND:
    """Within-period utility for forcedout regimes (no work, retired leisure)."""
    leisure = leisure_retired(
        good_health=good_health,
        time_endowment=time_endowment,
        leisure_cost_of_bad_health=leisure_cost_of_bad_health,
    )
    return u_working_life(
        consumption_equiv=consumption_equiv,
        leisure=leisure,
        consumption_weight=consumption_weight,
        coefficient_rra=coefficient_rra,
        utility_scale_factor=utility_scale_factor,
    )


def u_dead(
    assets: ContinuousState,
    bequest_shifter: float,
    scaled_bequest_weight: float,
    consumption_weight: FloatND,
    coefficient_rra: FloatND,
    utility_scale_factor: FloatND,
) -> FloatND:
    """Terminal bequest utility for the dead regime."""
    return bequest(
        assets=assets,
        bequest_shifter=bequest_shifter,
        scaled_bequest_weight=scaled_bequest_weight,
        consumption_weight=consumption_weight,
        coefficient_rra=coefficient_rra,
        utility_scale_factor=utility_scale_factor,
    )


def consumption_weight(
    consumption_weights: FloatND,
    pref_type: DiscreteState,
) -> FloatND:
    """Per-type consumption weight indexed by preference type.

    Wired as a DAG function so pylcm broadcasts the scalar to every cell;
    mirrors `discount_factor`.
    """
    return consumption_weights[pref_type]


def coefficient_rra(
    coefficients_rra: FloatND,
    pref_type: DiscreteState,
) -> FloatND:
    """Per-type CRRA coefficient indexed by preference type.

    Wired as a DAG function so pylcm broadcasts the scalar to every cell;
    mirrors `discount_factor`.
    """
    return coefficients_rra[pref_type]


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
    average_consumption_unequiv: float,
    consumption_weight: FloatND,
    coefficient_rra: FloatND,
    time_endowment: float,
    fixed_cost_of_work_intercept: float,
    reference_hours: float,
) -> FloatND:
    """Compute the scale factor so utility is approximately 1 at typical values."""
    average_leisure = time_endowment - reference_hours - fixed_cost_of_work_intercept
    u_cons = average_consumption_unequiv**consumption_weight
    u_leisure = average_leisure ** (1.0 - consumption_weight)

    one_minus_rra = jnp.where(
        jnp.isclose(coefficient_rra, 1.0), 1.0, 1.0 - coefficient_rra
    )
    raw = jnp.where(
        jnp.isclose(coefficient_rra, 1.0),
        jnp.log(u_cons * u_leisure),
        (u_cons * u_leisure) ** one_minus_rra / one_minus_rra,
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
    bequest_shifter: float,
    scaled_bequest_weight: float,
    consumption_weight: FloatND,
    coefficient_rra: FloatND,
    utility_scale_factor: FloatND,
) -> FloatND:
    """Bequest function for terminal/dead states.

    bequest = scale * bwt *
        (max(0,a) + shifter)^(consumption_weight*(1 - coefficient_rra))
        / (1 - coefficient_rra)
    """
    assets_shifted = jnp.maximum(0.0, assets) + bequest_shifter

    one_minus_rra = jnp.where(
        jnp.isclose(coefficient_rra, 1.0), 1.0, 1.0 - coefficient_rra
    )
    val = jnp.where(
        jnp.isclose(coefficient_rra, 1.0),
        jnp.log(assets_shifted),
        assets_shifted ** (one_minus_rra * consumption_weight) / one_minus_rra,
    )
    return val * scaled_bequest_weight * utility_scale_factor
