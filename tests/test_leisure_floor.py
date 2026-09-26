"""Float32 felicity at the leisure floor, against a 60-digit reference.

The class covers the canwork cells at production preference values: every pref
type, health, lagged labor supply, hours choice and canwork age, at consumption
points spanning the NB-EGM numeric-inverse bracket (`1e-8` to the top of the
action range). Marginal utility is `jax.grad` of the consumption-dollar
felicity, the route pylcm's NB-EGM uses.
"""

import itertools
from decimal import Decimal, getcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aca_model.agent import preferences

getcontext().prec = 60

CONSUMPTION_WEIGHTS = (0.6776541629520845, 0.8805772328591686, 0.0718086283445225)
COEFFICIENTS_RRA = (3.841252231680976, 0.9990771146810682, 3.8328505891095137)
TIME_ENDOWMENT = 3926.9478390365557
BAD_HEALTH_COST = 408.9190313043897
REENTRY_COST = 119.61095526191421
FIXED_COST_INTERCEPT = 337.5223349543413
FIXED_COST_AGE_TREND = 84.09242960636178
REFERENCE_AGE = 50
REFERENCE_HOURS = 1000.0
AVERAGE_CONSUMPTION = 20000.0
HOURS = (0.0, 1000.0, 1500.0, 2000.0, 2500.0)
CANWORK_AGES = tuple(range(51, 72))
CONSUMPTION = (1e-8, 1597.0921419521899, 20000.0, 300_000.0, 721271853.846792)
FLOAT32_REL_TOL = 1e-4
# Leisure-map widths as fractions of the endowment: smoothing width and floor.
SMOOTHING_FRACTION = 0.01
FLOOR_FRACTION = 1e-5


def _f32(x: float) -> jnp.ndarray:
    return jnp.asarray(x, dtype=jnp.float32)


def _felicity_of_dollars(consumption, leisure, weight, rra, scale):
    return preferences.u_alive(
        consumption_equiv=preferences.consumption_equiv(
            consumption_dollars=consumption,
            equivalence_scale=jnp.ones_like(consumption),
        ),
        leisure=leisure,
        consumption_weight=weight,
        coefficient_rra=rra,
        utility_scale_factor=scale,
    )


def _leisure_available(good: int, lagged: int, hours: float, age: int) -> float:
    fixed_cost = FIXED_COST_INTERCEPT + FIXED_COST_AGE_TREND * (age - REFERENCE_AGE)
    reentry = REENTRY_COST if lagged == 0 else 0.0
    work = hours + fixed_cost + reentry if hours > 0 else 0.0
    return TIME_ENDOWMENT - (0.0 if good else BAD_HEALTH_COST) - work


def _cells() -> list[tuple[float, float]]:
    """All `(leisure_available, consumption)` pairs of the class."""
    return [
        (_leisure_available(good, lagged, hours, age), c)
        for good, lagged, hours, age, c in itertools.product(
            (0, 1), (0, 1), HOURS, CANWORK_AGES, CONSUMPTION
        )
    ]


def _float32_u_and_marginal(
    pref_type: int, cells: list[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    available = jnp.asarray([a for a, _ in cells], dtype=jnp.float32)
    consumption = jnp.asarray([c for _, c in cells], dtype=jnp.float32)
    weight = _f32(CONSUMPTION_WEIGHTS[pref_type])
    rra = _f32(COEFFICIENTS_RRA[pref_type])
    leisure = _floored_leisure(available, _f32(TIME_ENDOWMENT))
    scale = preferences.utility_scale_factor(
        average_consumption_equiv=_f32(AVERAGE_CONSUMPTION),
        consumption_weight=weight,
        coefficient_rra=rra,
        time_endowment=_f32(TIME_ENDOWMENT),
        fixed_cost_of_work_intercept=_f32(FIXED_COST_INTERCEPT),
        reference_hours=_f32(REFERENCE_HOURS),
    )
    args = (weight, rra, scale)
    u = jax.vmap(_felicity_of_dollars, in_axes=(0, 0, None, None, None))(
        consumption, leisure, *args
    )
    marginal = jax.vmap(
        jax.grad(_felicity_of_dollars), in_axes=(0, 0, None, None, None)
    )(consumption, leisure, *args)
    assert u.dtype == marginal.dtype == jnp.float32
    return np.asarray(u, dtype=np.float64), np.asarray(marginal, dtype=np.float64)


def _floored_leisure(available, endowment):
    """Production leisure at a given `leisure_available`, via the tied-regime map.

    Good health and no fixed cost make the work loss the only deduction, so hours of
    `endowment - available` leave exactly `available`.
    """
    return preferences.leisure_canwork_tied(
        working_hours_value=endowment - available,
        good_health=jnp.ones(available.shape, dtype=jnp.int32),
        time_endowment=endowment,
        leisure_cost_of_bad_health=jnp.zeros_like(endowment),
        fixed_cost_of_work=jnp.zeros_like(endowment),
    )


def _d(x: float) -> Decimal:
    return Decimal(repr(float(x)))


def _reference_u_and_marginal(
    pref_type: int, cells: list[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Closed forms in 60-digit log space, independent of the JAX code path."""
    weight, rra = _d(CONSUMPTION_WEIGHTS[pref_type]), _d(COEFFICIENTS_RRA[pref_type])
    endowment = _d(TIME_ENDOWMENT)
    smoothing = _d(SMOOTHING_FRACTION) * endowment
    floor = _d(FLOOR_FRACTION) * endowment
    average_leisure = endowment - _d(REFERENCE_HOURS) - _d(FIXED_COST_INTERCEPT)
    log_average = weight * _d(AVERAGE_CONSUMPTION).ln() + (1 - weight) * (
        average_leisure.ln()
    )
    scale = abs((1 - rra) / ((1 - rra) * log_average).exp())
    u, marginal = [], []
    for available, c in cells:
        leisure = (
            smoothing
            * ((floor / smoothing).exp() + (_d(available) / smoothing).exp()).ln()
        )
        log_composite = weight * _d(c).ln() + (1 - weight) * leisure.ln()
        power = ((1 - rra) * log_composite).exp()
        u.append(float(scale * power / (1 - rra)))
        marginal.append(float(scale * weight / _d(c) * power))
    return np.array(u), np.array(marginal)


OUTPUTS = {"u": 0, "marginal": 1}
WITNESS = [(_leisure_available(0, 0, 2500.0, 71), c) for c in CONSUMPTION]


@pytest.mark.parametrize("output", ["u", "marginal"])
def test_float32_felicity_at_witness_cell_matches_reference(output: str) -> None:
    """Pref type 2, bad health, did not work, 2500 h, age 71: finite and accurate."""
    got = _float32_u_and_marginal(2, WITNESS)[OUTPUTS[output]]
    ref = _reference_u_and_marginal(2, WITNESS)[OUTPUTS[output]]
    np.testing.assert_allclose(got, ref, rtol=FLOAT32_REL_TOL)


@pytest.mark.parametrize("output", ["u", "marginal"])
@pytest.mark.parametrize("pref_type", [0, 1, 2])
def test_float32_felicity_over_canwork_class_matches_reference(
    pref_type: int, output: str
) -> None:
    """Every canwork cell and consumption point: float32 equals the reference."""
    cells = _cells()
    got = _float32_u_and_marginal(pref_type, cells)[OUTPUTS[output]]
    ref = _reference_u_and_marginal(pref_type, cells)[OUTPUTS[output]]
    np.testing.assert_allclose(got, ref, rtol=FLOAT32_REL_TOL)


@pytest.mark.parametrize("pref_type", [0, 1, 2])
def test_leisure_floor_leaves_utility_unchanged_away_from_the_floor(
    pref_type: int,
) -> None:
    """With leisure_available above 10% of the endowment, u moves by at most 1e-7.

    Compared against the unfloored softplus `s * log(1 + e^(x/s))`.
    """
    available = np.array([a for a, _ in _cells() if a > 0.1 * TIME_ENDOWMENT])
    assert available.size > 0
    smoothing = SMOOTHING_FRACTION * TIME_ENDOWMENT
    unfloored = smoothing * np.logaddexp(0.0, available / smoothing)
    floored = np.asarray(
        _floored_leisure(jnp.asarray(available), jnp.asarray(TIME_ENDOWMENT))
    )
    exponent = (1.0 - CONSUMPTION_WEIGHTS[pref_type]) * (
        1.0 - COEFFICIENTS_RRA[pref_type]
    )
    relative_change = np.abs((floored / unfloored) ** exponent - 1.0)
    np.testing.assert_array_less(relative_change, 1e-7)
