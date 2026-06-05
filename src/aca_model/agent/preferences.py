"""Preference functions: utility, leisure, bequests.

Ported from struct-ret/src/model/preferences_utility.py and auxiliaries.py.
"""

import jax.numpy as jnp
from lcm import categorical
from lcm.typing import (
    Age,
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    IntND,
    ScalarFloat,
    ScalarInt,
)

from aca_model.agent.labor_market import LaggedLaborSupply


@categorical(ordered=False)
class PrefType:
    """Unobserved preference type for heterogeneity in estimation."""

    type_0: ScalarInt
    type_1: ScalarInt
    type_2: ScalarInt


@categorical(ordered=False)
class BenchmarkPrefType:
    """Compact 2-type variant of `PrefType` used by the ASV benchmark.

    The benchmark model exercises the full 18-regime DAG with compact
    continuous grids; shrinking `pref_type` from 3 to 2 types cuts the
    partition-axis cardinality too, so the benchmark finishes faster
    without changing anything structural about the kernel being
    measured.
    """

    type_0: ScalarInt
    type_1: ScalarInt


def positive_leisure(leisure: FloatND) -> BoolND:
    """Return True where leisure is strictly positive."""
    return leisure > 0


def equivalence_scale(is_married: IntND, exponent: ScalarFloat) -> FloatND:
    """Return the equivalence scale for household size adjustment.

    Single (is_married=False) → 1.0, married (is_married=True) → 2^exponent.
    """
    return jnp.where(is_married, 2.0**exponent, 1.0)


def fixed_cost_of_work(
    age: Age,
    fixed_cost_of_work_intercept: ScalarFloat,
    fixed_cost_of_work_age_trend: ScalarFloat,
    reference_age: ScalarInt,
) -> ScalarFloat:
    """Age-dependent fixed cost of working (intercept + trend slope on age)."""
    return fixed_cost_of_work_intercept + fixed_cost_of_work_age_trend * (
        age - reference_age
    )


def leisure_canwork_retiree_or_nongroup(
    working_hours_value: FloatND,
    good_health: IntND,
    lagged_labor_supply: DiscreteState,
    time_endowment: ScalarFloat,
    leisure_cost_of_bad_health: ScalarFloat,
    fixed_cost_of_work: ScalarFloat,
    labor_force_reentry_cost: ScalarFloat,
) -> FloatND:
    """Compute leisure for canwork retiree / nongroup regimes.

    Reentry cost applies when returning to work after not working last period.
    """
    health_loss = jnp.where(good_health, 0.0, leisure_cost_of_bad_health)
    reentry_cost = jnp.where(
        lagged_labor_supply == LaggedLaborSupply.did_not_work,
        labor_force_reentry_cost,
        0.0,
    )
    work_loss = jnp.where(
        working_hours_value > 0.0,
        working_hours_value + fixed_cost_of_work + reentry_cost,
        0.0,
    )

    return time_endowment - health_loss - work_loss


def leisure_canwork_tied(
    working_hours_value: FloatND,
    good_health: IntND,
    time_endowment: ScalarFloat,
    leisure_cost_of_bad_health: ScalarFloat,
    fixed_cost_of_work: ScalarFloat,
) -> FloatND:
    """Compute leisure for canwork tied regimes.

    No need to consider reentry costs.
    """
    health_loss = jnp.where(good_health, 0.0, leisure_cost_of_bad_health)
    work_loss = jnp.where(
        working_hours_value > 0.0, working_hours_value + fixed_cost_of_work, 0.0
    )
    return time_endowment - health_loss - work_loss


def leisure_forcedout(
    good_health: IntND,
    time_endowment: ScalarFloat,
    leisure_cost_of_bad_health: ScalarFloat,
) -> FloatND:
    """Compute leisure for forcedout regimes (no work)."""
    health_loss = jnp.where(good_health, 0.0, leisure_cost_of_bad_health)
    return time_endowment - health_loss


def consumption_equiv(
    consumption_dollars: ContinuousAction,
    equivalence_scale: FloatND,
) -> FloatND:
    """Utility-equivalized consumption."""
    return consumption_dollars / equivalence_scale


def u_alive(
    consumption_equiv: FloatND,
    leisure: FloatND,
    consumption_weight: FloatND,
    coefficient_rra: FloatND,
    utility_scale_factor: FloatND,
) -> FloatND:
    """Within-period utility for every non-dead regime.

    CES over consumption and leisure. `leisure` is a DAG input — supplied
    per-regime by `leisure_canwork_retiree_or_nongroup`,
    `leisure_canwork_tied`, or `leisure_forcedout`.
    """
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
    average_consumption_equiv: ScalarFloat,
    consumption_weight: FloatND,
    coefficient_rra: FloatND,
    time_endowment: ScalarFloat,
    fixed_cost_of_work_intercept: ScalarFloat,
    reference_hours: ScalarFloat,
) -> FloatND:
    """Compute the scale factor so utility is approximately 1 at typical values."""
    average_leisure = time_endowment - reference_hours - fixed_cost_of_work_intercept
    u_cons = average_consumption_equiv**consumption_weight
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
    bequest_weight: ScalarFloat,
    consumption_weight: ScalarFloat,
    coefficient_rra: ScalarFloat,
    time_endowment: ScalarFloat,
    time_discount_factor: ScalarFloat,
    rate_of_return: ScalarFloat,
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
    bequest_shifter: ScalarFloat,
    scaled_bequest_weight: ScalarFloat,
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
