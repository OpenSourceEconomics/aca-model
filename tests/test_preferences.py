"""Tests for preference functions, ported from struct-ret.

Parameter values from struct-ret PreferenceParameters fixture.
"""

import jax.numpy as jnp

from aca_model.agent import preferences

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
        time_discount_factor=TIME_DISCOUNT_FACTOR,
        rate_of_return=RATE_OF_RETURN,
    )
    assert jnp.isclose(result, 0.820_137_639_127_977_3, rtol=1e-6)


def test_scaled_bequest_weight_log() -> None:
    result = preferences.scaled_bequest_weight(
        bequest_weight=BEQUEST_WEIGHT,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(1.0),
        time_endowment=TIME_ENDOWMENT,
        time_discount_factor=TIME_DISCOUNT_FACTOR,
        rate_of_return=RATE_OF_RETURN,
    )
    assert jnp.isclose(result, 58.235_294_117_647_05, rtol=1e-6)


def test_scaled_bequest_weight_zero() -> None:
    result = preferences.scaled_bequest_weight(
        bequest_weight=jnp.asarray(0.0),
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=jnp.asarray(5.0),
        time_endowment=TIME_ENDOWMENT,
        time_discount_factor=TIME_DISCOUNT_FACTOR,
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
        time_discount_factor=TIME_DISCOUNT_FACTOR,
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
        time_discount_factor=TIME_DISCOUNT_FACTOR,
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
