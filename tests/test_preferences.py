"""Tests for preference functions, ported from struct-ret.

Parameter values from struct-ret PreferenceParameters fixture.
"""

import jax.numpy as jnp

from aca_model.agent import preferences

# Struct-ret preference parameters
CONSUMPTION_WEIGHT = 0.6
TIME_DISCOUNT_FACTOR = 0.85
TIME_ENDOWMENT = 5000.0
FIXED_COST_INTERCEPT = 0.0
AVERAGE_CONSUMPTION = 10000.0
RATE_OF_RETURN = 0.01
BEQUEST_WEIGHT = 0.02
BEQUEST_SHIFTER = 500_000.0
REFERENCE_HOURS = 1000.0


# --- utility_scale_factor ---


def test_utility_scale_factor_crra() -> None:
    result = preferences.utility_scale_factor(
        average_consumption_dollars=AVERAGE_CONSUMPTION,
        consumption_weight=jnp.array(CONSUMPTION_WEIGHT),
        coefficient_rra=jnp.array(5.0),
        time_endowment=TIME_ENDOWMENT,
        fixed_cost_of_work_intercept=FIXED_COST_INTERCEPT,
        reference_hours=REFERENCE_HOURS,
    )
    assert jnp.isclose(result, 9_233_279_397_806_166.0, rtol=1e-6)


def test_utility_scale_factor_log() -> None:
    result = preferences.utility_scale_factor(
        average_consumption_dollars=AVERAGE_CONSUMPTION,
        consumption_weight=jnp.array(CONSUMPTION_WEIGHT),
        coefficient_rra=jnp.array(1.0),
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
        coefficient_rra=5.0,
        time_endowment=TIME_ENDOWMENT,
        time_discount_factor=TIME_DISCOUNT_FACTOR,
        rate_of_return=RATE_OF_RETURN,
    )
    assert jnp.isclose(result, 0.820_137_639_127_977_3, rtol=1e-6)


def test_scaled_bequest_weight_log() -> None:
    result = preferences.scaled_bequest_weight(
        bequest_weight=BEQUEST_WEIGHT,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=1.0,
        time_endowment=TIME_ENDOWMENT,
        time_discount_factor=TIME_DISCOUNT_FACTOR,
        rate_of_return=RATE_OF_RETURN,
    )
    assert jnp.isclose(result, 58.235_294_117_647_05, rtol=1e-6)


def test_scaled_bequest_weight_zero() -> None:
    result = preferences.scaled_bequest_weight(
        bequest_weight=0.0,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=5.0,
        time_endowment=TIME_ENDOWMENT,
        time_discount_factor=TIME_DISCOUNT_FACTOR,
        rate_of_return=RATE_OF_RETURN,
    )
    assert result == 0.0


# --- utility with scale factor (regression tests from struct-ret) ---


def test_utility_log_regression() -> None:
    scale = preferences.utility_scale_factor(
        average_consumption_dollars=AVERAGE_CONSUMPTION,
        consumption_weight=jnp.array(CONSUMPTION_WEIGHT),
        coefficient_rra=jnp.array(1.0),
        time_endowment=TIME_ENDOWMENT,
        fixed_cost_of_work_intercept=FIXED_COST_INTERCEPT,
        reference_hours=REFERENCE_HOURS,
    )
    result = preferences.u_can_work(
        consumption_equiv=jnp.array(50000.0),
        leisure=jnp.array(400.0),
        consumption_weight=jnp.array(CONSUMPTION_WEIGHT),
        coefficient_rra=jnp.array(1.0),
        utility_scale_factor=scale,
    )
    assert jnp.isclose(result, 1.005_046_313_660_588_5, rtol=1e-5)


def test_utility_crra_regression() -> None:
    scale = preferences.utility_scale_factor(
        average_consumption_dollars=AVERAGE_CONSUMPTION,
        consumption_weight=jnp.array(CONSUMPTION_WEIGHT),
        coefficient_rra=jnp.array(5.0),
        time_endowment=TIME_ENDOWMENT,
        fixed_cost_of_work_intercept=FIXED_COST_INTERCEPT,
        reference_hours=REFERENCE_HOURS,
    )
    result = preferences.u_can_work(
        consumption_equiv=jnp.array(50000.0),
        leisure=jnp.array(400.0),
        consumption_weight=jnp.array(CONSUMPTION_WEIGHT),
        coefficient_rra=jnp.array(5.0),
        utility_scale_factor=scale,
    )
    assert jnp.isclose(result, -0.836_511_642_073_019_1, rtol=1e-5)


def test_utility_married_equivalence() -> None:
    """Married with equiv-scaled consumption_dollars should equal single utility."""
    scale = preferences.utility_scale_factor(
        average_consumption_dollars=AVERAGE_CONSUMPTION,
        consumption_weight=jnp.array(CONSUMPTION_WEIGHT),
        coefficient_rra=jnp.array(5.0),
        time_endowment=TIME_ENDOWMENT,
        fixed_cost_of_work_intercept=FIXED_COST_INTERCEPT,
        reference_hours=REFERENCE_HOURS,
    )
    single = preferences.u_can_work(
        consumption_equiv=jnp.array(50000.0),
        leisure=jnp.array(400.0),
        consumption_weight=jnp.array(CONSUMPTION_WEIGHT),
        coefficient_rra=jnp.array(5.0),
        utility_scale_factor=scale,
    )
    married = preferences.u_can_work(
        consumption_equiv=jnp.array(50000.0),
        leisure=jnp.array(400.0),
        consumption_weight=jnp.array(CONSUMPTION_WEIGHT),
        coefficient_rra=jnp.array(5.0),
        utility_scale_factor=scale,
    )
    assert jnp.isclose(single, married, rtol=1e-5)


# --- bequest (regression tests from struct-ret) ---


def test_bequest_log_regression() -> None:
    scale = preferences.utility_scale_factor(
        average_consumption_dollars=AVERAGE_CONSUMPTION,
        consumption_weight=jnp.array(CONSUMPTION_WEIGHT),
        coefficient_rra=jnp.array(1.0),
        time_endowment=TIME_ENDOWMENT,
        fixed_cost_of_work_intercept=FIXED_COST_INTERCEPT,
        reference_hours=REFERENCE_HOURS,
    )
    bwt = preferences.scaled_bequest_weight(
        bequest_weight=BEQUEST_WEIGHT,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=1.0,
        time_endowment=TIME_ENDOWMENT,
        time_discount_factor=TIME_DISCOUNT_FACTOR,
        rate_of_return=RATE_OF_RETURN,
    )
    result = preferences.bequest(
        assets=jnp.array(10000.0),
        bequest_shifter=BEQUEST_SHIFTER,
        scaled_bequest_weight=bwt.item(),
        consumption_weight=jnp.array(CONSUMPTION_WEIGHT),
        coefficient_rra=jnp.array(1.0),
        utility_scale_factor=scale,
    )
    assert jnp.isclose(result, 86.539_249_963_643_88, rtol=1e-5)


def test_bequest_crra_regression() -> None:
    scale = preferences.utility_scale_factor(
        average_consumption_dollars=AVERAGE_CONSUMPTION,
        consumption_weight=jnp.array(CONSUMPTION_WEIGHT),
        coefficient_rra=jnp.array(5.0),
        time_endowment=TIME_ENDOWMENT,
        fixed_cost_of_work_intercept=FIXED_COST_INTERCEPT,
        reference_hours=REFERENCE_HOURS,
    )
    bwt = preferences.scaled_bequest_weight(
        bequest_weight=BEQUEST_WEIGHT,
        consumption_weight=CONSUMPTION_WEIGHT,
        coefficient_rra=5.0,
        time_endowment=TIME_ENDOWMENT,
        time_discount_factor=TIME_DISCOUNT_FACTOR,
        rate_of_return=RATE_OF_RETURN,
    )
    result = preferences.bequest(
        assets=jnp.array(10000.0),
        bequest_shifter=BEQUEST_SHIFTER,
        scaled_bequest_weight=bwt.item(),
        consumption_weight=jnp.array(CONSUMPTION_WEIGHT),
        coefficient_rra=jnp.array(5.0),
        utility_scale_factor=scale,
    )
    assert jnp.isclose(result, -37.932_748_117_035_63, rtol=1e-5)
