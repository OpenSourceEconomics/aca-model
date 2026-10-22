"""Tests for preference functions, ported from struct-ret.

Parameter values from struct-ret PreferenceParameters fixture.
"""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from dags import concatenate_functions

from aca_model.agent import preferences
from aca_model.baseline.regimes._common import build_dead_regime, build_model_functions

# Struct-ret preference parameters. Tests call DAG functions directly, so
# every scalar fixed_param is supplied as a 0-d jax array (the type pylcm
# casts user-provided Python scalars to before passing them into the DAG).
CONSUMPTION_WEIGHT = jnp.asarray(0.6)
TIME_DISCOUNT_FACTOR = jnp.asarray(0.85)
TIME_ENDOWMENT = jnp.asarray(5000.0)
FIXED_COST_INTERCEPT = jnp.asarray(0.0)
AVERAGE_CONSUMPTION = jnp.asarray(10000.0)
RATE_OF_RETURN = jnp.asarray(0.01)
BEQUEST_WEIGHT = jnp.asarray(0.02)
BEQUEST_SHIFTER = jnp.asarray(500_000.0)
REFERENCE_HOURS = jnp.asarray(1000.0)


# --- utility_scale_factor ---


def test_utility_scale_factor_crra() -> None:
    result = preferences.utility_scale_factor(
        average_consumption_equiv=AVERAGE_CONSUMPTION,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(5.0),
        time_endowment=TIME_ENDOWMENT,
        fixed_cost_of_work_intercept=FIXED_COST_INTERCEPT,
        reference_hours=REFERENCE_HOURS,
    )
    assert jnp.isclose(result, 9_233_279_397_806_166.0, rtol=1e-6)


def test_utility_scale_factor_log() -> None:
    result = preferences.utility_scale_factor(
        average_consumption_equiv=AVERAGE_CONSUMPTION,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(1.0),
        time_endowment=TIME_ENDOWMENT,
        fixed_cost_of_work_intercept=FIXED_COST_INTERCEPT,
        reference_hours=REFERENCE_HOURS,
    )
    assert jnp.isclose(result, 0.113_073_257_794_546_72, rtol=1e-6)


# --- scaled_bequest_weight ---


def test_scaled_bequest_weight_positive() -> None:
    result = preferences.scaled_bequest_weight(
        bequest_weight=BEQUEST_WEIGHT,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(5.0),
        time_endowment=TIME_ENDOWMENT,
        discount_factor=TIME_DISCOUNT_FACTOR,
        rate_of_return=RATE_OF_RETURN,
    )
    assert jnp.isclose(result, 0.820_137_639_127_977_3, rtol=1e-6)


def test_scaled_bequest_weight_log() -> None:
    result = preferences.scaled_bequest_weight(
        bequest_weight=BEQUEST_WEIGHT,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(1.0),
        time_endowment=TIME_ENDOWMENT,
        discount_factor=TIME_DISCOUNT_FACTOR,
        rate_of_return=RATE_OF_RETURN,
    )
    assert jnp.isclose(result, 58.235_294_117_647_05, rtol=1e-6)


def test_scaled_bequest_weight_zero() -> None:
    result = preferences.scaled_bequest_weight(
        bequest_weight=jnp.asarray(0.0),
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(5.0),
        time_endowment=TIME_ENDOWMENT,
        discount_factor=TIME_DISCOUNT_FACTOR,
        rate_of_return=RATE_OF_RETURN,
    )
    assert result == 0.0


# --- utility with scale factor (regression tests from struct-ret) ---


def test_utility_log_regression() -> None:
    scale = preferences.utility_scale_factor(
        average_consumption_equiv=AVERAGE_CONSUMPTION,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(1.0),
        time_endowment=TIME_ENDOWMENT,
        fixed_cost_of_work_intercept=FIXED_COST_INTERCEPT,
        reference_hours=REFERENCE_HOURS,
    )
    result = preferences.u_alive(
        consumption_equiv=jnp.array(50000.0),
        leisure=jnp.array(400.0),
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(1.0),
        utility_scale_factor=scale,
    )
    assert jnp.isclose(result, 1.005_046_313_660_588_5, rtol=1e-5)


def test_utility_crra_regression() -> None:
    scale = preferences.utility_scale_factor(
        average_consumption_equiv=AVERAGE_CONSUMPTION,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(5.0),
        time_endowment=TIME_ENDOWMENT,
        fixed_cost_of_work_intercept=FIXED_COST_INTERCEPT,
        reference_hours=REFERENCE_HOURS,
    )
    result = preferences.u_alive(
        consumption_equiv=jnp.array(50000.0),
        leisure=jnp.array(400.0),
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(5.0),
        utility_scale_factor=scale,
    )
    assert jnp.isclose(result, -0.836_511_642_073_019_1, rtol=1e-5)


def test_utility_married_equivalence() -> None:
    """Married with equiv-scaled consumption_dollars should equal single utility."""
    scale = preferences.utility_scale_factor(
        average_consumption_equiv=AVERAGE_CONSUMPTION,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(5.0),
        time_endowment=TIME_ENDOWMENT,
        fixed_cost_of_work_intercept=FIXED_COST_INTERCEPT,
        reference_hours=REFERENCE_HOURS,
    )
    single = preferences.u_alive(
        consumption_equiv=jnp.array(50000.0),
        leisure=jnp.array(400.0),
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(5.0),
        utility_scale_factor=scale,
    )
    married = preferences.u_alive(
        consumption_equiv=jnp.array(50000.0),
        leisure=jnp.array(400.0),
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(5.0),
        utility_scale_factor=scale,
    )
    assert jnp.isclose(single, married, rtol=1e-5)


# --- bequest (regression tests from struct-ret) ---


def test_bequest_log_regression() -> None:
    scale = preferences.utility_scale_factor(
        average_consumption_equiv=AVERAGE_CONSUMPTION,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(1.0),
        time_endowment=TIME_ENDOWMENT,
        fixed_cost_of_work_intercept=FIXED_COST_INTERCEPT,
        reference_hours=REFERENCE_HOURS,
    )
    bwt = preferences.scaled_bequest_weight(
        bequest_weight=BEQUEST_WEIGHT,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(1.0),
        time_endowment=TIME_ENDOWMENT,
        discount_factor=TIME_DISCOUNT_FACTOR,
        rate_of_return=RATE_OF_RETURN,
    )
    result = preferences.bequest(
        assets=jnp.array(10000.0),
        bequest_shifter=BEQUEST_SHIFTER,
        scaled_bequest_weight=bwt,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(1.0),
        utility_scale_factor=scale,
    )
    assert jnp.isclose(result, 86.539_249_963_643_88, rtol=1e-5)


def test_bequest_crra_regression() -> None:
    scale = preferences.utility_scale_factor(
        average_consumption_equiv=AVERAGE_CONSUMPTION,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(5.0),
        time_endowment=TIME_ENDOWMENT,
        fixed_cost_of_work_intercept=FIXED_COST_INTERCEPT,
        reference_hours=REFERENCE_HOURS,
    )
    bwt = preferences.scaled_bequest_weight(
        bequest_weight=BEQUEST_WEIGHT,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(5.0),
        time_endowment=TIME_ENDOWMENT,
        discount_factor=TIME_DISCOUNT_FACTOR,
        rate_of_return=RATE_OF_RETURN,
    )
    result = preferences.bequest(
        assets=jnp.array(10000.0),
        bequest_shifter=BEQUEST_SHIFTER,
        scaled_bequest_weight=bwt,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(5.0),
        utility_scale_factor=scale,
    )
    assert jnp.isclose(result, -37.932_748_117_035_63, rtol=1e-5)


def test_bequest_uses_signed_assets_for_indebted_decedent() -> None:
    """A decedent with negative assets values their estate at `(A + κ)`.

    Negative assets enter the bequest unclamped: the estate argument is
    `assets + bequest_shifter`, so an indebted decedent's bequest is strictly
    below what a zero-asset decedent would receive.
    """
    assets = jnp.array(-50_000.0)
    consumption_weight = jnp.asarray(0.6)
    coefficient_rra = jnp.asarray(5.0)
    scale = preferences.utility_scale_factor(
        average_consumption_equiv=AVERAGE_CONSUMPTION,
        consumption_weight=consumption_weight,
        coefficient_rra=coefficient_rra,
        time_endowment=TIME_ENDOWMENT,
        fixed_cost_of_work_intercept=FIXED_COST_INTERCEPT,
        reference_hours=REFERENCE_HOURS,
    )
    bwt = preferences.scaled_bequest_weight(
        bequest_weight=BEQUEST_WEIGHT,
        consumption_weight=consumption_weight,
        coefficient_rra=coefficient_rra,
        time_endowment=TIME_ENDOWMENT,
        discount_factor=TIME_DISCOUNT_FACTOR,
        rate_of_return=RATE_OF_RETURN,
    )
    result = preferences.bequest(
        assets=assets,
        bequest_shifter=BEQUEST_SHIFTER,
        scaled_bequest_weight=bwt,
        consumption_weight=consumption_weight,
        coefficient_rra=coefficient_rra,
        utility_scale_factor=scale,
    )

    # bequest = scale · bwt · (A + κ)^(0.6·(1−5)) / (1−5), with A + κ = 450_000.
    estate = assets + BEQUEST_SHIFTER
    expected = (
        scale * bwt * estate ** (consumption_weight * (1.0 - coefficient_rra)) / -4.0
    )
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_bequest_floors_estate_base_when_debt_exceeds_the_shifter() -> None:
    """A death-time transfer floors the estate base at a nominal `1.0`.

    When debt runs deeper than the curvature shifter `κ` — so `assets + κ` would
    turn non-positive — the estate base is floored at `1.0` rather than entering a
    fractional power as a negative number (which is undefined). The bequest is then
    finite and equals `scale · bwt · 1^(…) / (1 − rra) = scale · bwt / (1 − rra)`.
    """
    assets = jnp.asarray(-(float(BEQUEST_SHIFTER) + 100_000.0))
    consumption_weight = jnp.asarray(0.6)
    coefficient_rra = jnp.asarray(5.0)
    scale = preferences.utility_scale_factor(
        average_consumption_equiv=AVERAGE_CONSUMPTION,
        consumption_weight=consumption_weight,
        coefficient_rra=coefficient_rra,
        time_endowment=TIME_ENDOWMENT,
        fixed_cost_of_work_intercept=FIXED_COST_INTERCEPT,
        reference_hours=REFERENCE_HOURS,
    )
    bwt = preferences.scaled_bequest_weight(
        bequest_weight=BEQUEST_WEIGHT,
        consumption_weight=consumption_weight,
        coefficient_rra=coefficient_rra,
        time_endowment=TIME_ENDOWMENT,
        discount_factor=TIME_DISCOUNT_FACTOR,
        rate_of_return=RATE_OF_RETURN,
    )
    result = preferences.bequest(
        assets=assets,
        bequest_shifter=BEQUEST_SHIFTER,
        scaled_bequest_weight=bwt,
        consumption_weight=consumption_weight,
        coefficient_rra=coefficient_rra,
        utility_scale_factor=scale,
    )

    expected = scale * bwt * 1.0 / (1.0 - coefficient_rra)
    np.testing.assert_allclose(result, expected, atol=1e-9)


# --- dead-regime bequest utility per preference type ---

# Draft estimates (struct-ret original_data/baseline/final/prefs.json).
_DRAFT_CONSUMPTION_WEIGHTS = jnp.asarray(
    [0.6776541629520845, 0.8805772328591686, 0.0718086283445225]
)
_DRAFT_COEFFICIENTS_RRA = jnp.asarray(
    [3.841252231680976, 0.9990771146810682, 3.8328505891095137]
)
_DRAFT_DISCOUNT_FACTORS = jnp.asarray(
    [0.839255583514979, 0.9121549199895648, 1.0595836603747046]
)


def _dead_regime_utility() -> Callable[..., Any]:
    """Compose the dead regime's utility DAG as pylcm merges it.

    Model-level functions come first; the dead regime's own entries override
    them, and `None` entries mask a model-level function out of the regime.
    """
    functions = dict(build_model_functions())
    for name, func in build_dead_regime().functions.items():
        if func is None:
            functions.pop(name, None)
        else:
            functions[name] = func
    return concatenate_functions(functions, targets="utility")


@pytest.mark.parametrize(
    ("pref_type", "expected"),
    [
        (0, -75.59884789663738),
        (1, 39.12342314306149),
        (2, -12.47026859098475),
    ],
)
def test_dead_regime_bequest_uses_per_type_transformed_weight(
    pref_type: int, expected: float
) -> None:
    """At the draft estimates and r = 0.05, the bequest utility of 100k assets
    equals `θ_B(type) · (A + κ)^((1-ν)γ) / (1-ν) · scale(type)` with the
    per-type `θ_B = T^ξ (bw / (1 + r - bw))^ξ₂ / β(type)`."""
    utility = _dead_regime_utility()
    result = utility(
        assets=jnp.asarray(100_000.0),
        pref_type=jnp.asarray(pref_type, dtype=jnp.int32),
        bequest_shifter=jnp.asarray(334460.20744719007),
        bequest_weight=jnp.asarray(0.02861082652580584),
        consumption_weights=_DRAFT_CONSUMPTION_WEIGHTS,
        coefficients_rra=_DRAFT_COEFFICIENTS_RRA,
        discount_factor_by_type=_DRAFT_DISCOUNT_FACTORS,
        time_endowment=jnp.asarray(3926.9478390365557),
        rate_of_return=jnp.asarray(0.05),
        average_consumption_equiv=jnp.asarray(20_000.0),
        fixed_cost_of_work_intercept=jnp.asarray(337.5223349543413),
        reference_hours=jnp.asarray(1000.0),
    )
    np.testing.assert_allclose(result, expected, rtol=1e-9)
