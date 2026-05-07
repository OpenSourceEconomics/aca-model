"""Tests for individual model component functions."""

import jax.numpy as jnp

from aca_model.agent import preferences
from aca_model.environment import social_security


def test_equivalence_scale_single() -> None:
    result = preferences.equivalence_scale(jnp.array(False), 0.7)
    assert jnp.isclose(result, 1.0)


def test_equivalence_scale_married() -> None:
    result = preferences.equivalence_scale(jnp.array(True), 0.7)
    assert jnp.isclose(result, 2.0**0.7)


def test_leisure_not_working() -> None:
    result = preferences.leisure(
        working_hours_value=jnp.array(0.0),
        age=60,
        good_health=jnp.array(1.0),
        lagged_labor_supply=jnp.array(0),
        time_endowment=5000.0,
        leisure_cost_of_bad_health=500.0,
        fixed_cost_of_work_intercept=100.0,
        fixed_cost_of_work_age_trend=5,
        labor_force_reentry_cost=200.0,
        reference_age=50,
    )
    assert jnp.isclose(result, 5000.0)


def test_leisure_working_good_health() -> None:
    result = preferences.leisure(
        working_hours_value=jnp.array(2000.0),
        age=60,
        good_health=jnp.array(1.0),
        lagged_labor_supply=jnp.array(1),
        time_endowment=5000.0,
        leisure_cost_of_bad_health=500.0,
        fixed_cost_of_work_intercept=100.0,
        fixed_cost_of_work_age_trend=5,
        labor_force_reentry_cost=200.0,
        reference_age=50,
    )
    # 5000 - 0 (good health) - (2000 + 100 + 5*(60-50) + 0 (lagged=1))
    expected = 5000.0 - 2000.0 - 100.0 - 50.0
    assert jnp.isclose(result, expected)


def test_leisure_reentry_cost() -> None:
    result = preferences.leisure(
        working_hours_value=jnp.array(2000.0),
        age=60,
        good_health=jnp.array(1.0),
        lagged_labor_supply=jnp.array(0),
        time_endowment=5000.0,
        leisure_cost_of_bad_health=500.0,
        fixed_cost_of_work_intercept=100.0,
        fixed_cost_of_work_age_trend=5,
        labor_force_reentry_cost=200.0,
        reference_age=50,
    )
    expected = 5000.0 - 2000.0 - 100.0 - 50.0 - 200.0
    assert jnp.isclose(result, expected)


def test_leisure_bad_health() -> None:
    result = preferences.leisure_retired(
        good_health=jnp.array(0.0),
        time_endowment=5000.0,
        leisure_cost_of_bad_health=500.0,
    )
    assert jnp.isclose(result, 4500.0)


def test_utility_positive_leisure() -> None:
    result = preferences.utility(
        consumption=jnp.array(10000.0),
        leisure=jnp.array(3000.0),
        pref_type=jnp.array(0),
        consumption_weight=jnp.array([0.4, 0.4, 0.4]),
        coefficient_rra=jnp.array([2.0, 2.0, 2.0]),
        equivalence_scale=jnp.array(1.0),
        utility_scale_factor=jnp.array(1.0),
    )
    assert jnp.isfinite(result)


def test_utility_log_case() -> None:
    result = preferences.utility(
        consumption=jnp.array(10000.0),
        leisure=jnp.array(3000.0),
        pref_type=jnp.array(0),
        consumption_weight=jnp.array([0.4, 0.4, 0.4]),
        coefficient_rra=jnp.array([1.0, 1.0, 1.0]),
        equivalence_scale=jnp.array(1.0),
        utility_scale_factor=jnp.array(1.0),
    )
    composite = 10000.0**0.4 * 3000.0**0.6
    expected = jnp.log(composite)
    assert jnp.isclose(result, expected, rtol=1e-5)


def test_bequest_positive_assets() -> None:
    result = preferences.bequest(
        assets=jnp.array(100000.0),
        pref_type=jnp.array(0),
        bequest_shifter=5000.0,
        scaled_bequest_weight=0.5,
        consumption_weight=jnp.array([0.4, 0.4, 0.4]),
        coefficient_rra=jnp.array([2.0, 2.0, 2.0]),
        utility_scale_factor=jnp.array(1.0),
    )
    assert jnp.isfinite(result)


def test_bequest_zero_assets() -> None:
    result = preferences.bequest(
        assets=jnp.array(0.0),
        pref_type=jnp.array(0),
        bequest_shifter=5000.0,
        scaled_bequest_weight=0.5,
        consumption_weight=jnp.array([0.4, 0.4, 0.4]),
        coefficient_rra=jnp.array([2.0, 2.0, 2.0]),
        utility_scale_factor=jnp.array(1.0),
    )
    assert jnp.isfinite(result)
    assert result < 0  # CRRA with γ>1 gives negative values


def test_aime_to_pia_three_brackets() -> None:
    # Below first kink
    pia_low = social_security.aime_to_pia(
        aime=jnp.array(500.0),
        aime_kink_0=791.0,
        aime_kink_1=4768.0,
        pia_conversion_rate_0=0.9,
        pia_conversion_rate_1=0.32,
        pia_conversion_rate_2=0.15,
        pia_kink_0=711.9,
        pia_kink_1=1983.54,
    )
    assert jnp.isclose(pia_low, 0.9 * 500.0)

    # Between kinks
    pia_mid = social_security.aime_to_pia(
        aime=jnp.array(2000.0),
        aime_kink_0=791.0,
        aime_kink_1=4768.0,
        pia_conversion_rate_0=0.9,
        pia_conversion_rate_1=0.32,
        pia_conversion_rate_2=0.15,
        pia_kink_0=711.9,
        pia_kink_1=1983.54,
    )
    expected = 711.9 + 0.32 * (2000.0 - 791.0)
    assert jnp.isclose(pia_mid, expected)


def test_next_aime_accrual() -> None:
    ratio = jnp.zeros(100)
    ratio = ratio.at[55].set(0.5)
    result = social_security.next_aime(
        aime=jnp.array(1000.0),
        labor_income=jnp.array(50000.0),
        period=55,
        age=55,
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=70,
        pia_table=jnp.array([0.0, 711.9, 2115.1, 3015.1]),
        pia_aime_grid=jnp.array([0.0, 791.0, 4768.0, 8000.0]),
        aime_accrual_factor=1 / 35,
        aggregate_wage_growth=0.02,
        aime_last_age_with_indexing=60,
        aime_kink_2=8000.0,
        ratio_lowest_earnings=ratio,
    )
    assert result > 1000.0
    assert result <= 8000.0
