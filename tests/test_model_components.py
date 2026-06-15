"""Tests for individual model component functions."""

import jax.numpy as jnp

from aca_model.agent import preferences
from aca_model.environment import social_security


def test_equivalence_scale_single() -> None:
    result = preferences.equivalence_scale(jnp.int32(0), jnp.asarray(0.7))
    assert jnp.isclose(result, 1.0)


def test_equivalence_scale_married() -> None:
    result = preferences.equivalence_scale(jnp.int32(1), jnp.asarray(0.7))
    assert jnp.isclose(result, 2.0**0.7)


def test_leisure_not_working() -> None:
    result = preferences.leisure_canwork_retiree_or_nongroup(
        working_hours_value=jnp.array(0.0),
        good_health=jnp.int32(1),
        lagged_labor_supply=jnp.int32(0),
        time_endowment=jnp.asarray(5000.0),
        leisure_cost_of_bad_health=jnp.asarray(500.0),
        fixed_cost_of_work=jnp.asarray(150.0),
        labor_force_reentry_cost=jnp.asarray(200.0),
    )
    assert jnp.isclose(result, 5000.0)


def test_leisure_working_good_health() -> None:
    result = preferences.leisure_canwork_retiree_or_nongroup(
        working_hours_value=jnp.array(2000.0),
        good_health=jnp.int32(1),
        lagged_labor_supply=jnp.int32(1),
        time_endowment=jnp.asarray(5000.0),
        leisure_cost_of_bad_health=jnp.asarray(500.0),
        fixed_cost_of_work=jnp.asarray(150.0),
        labor_force_reentry_cost=jnp.asarray(200.0),
    )
    # 5000 - 0 (good health) - (2000 + 150 + 0 (lagged=1))
    expected = 5000.0 - 2000.0 - 150.0
    assert jnp.isclose(result, expected)


def test_leisure_reentry_cost() -> None:
    result = preferences.leisure_canwork_retiree_or_nongroup(
        working_hours_value=jnp.array(2000.0),
        good_health=jnp.int32(1),
        lagged_labor_supply=jnp.int32(0),
        time_endowment=jnp.asarray(5000.0),
        leisure_cost_of_bad_health=jnp.asarray(500.0),
        fixed_cost_of_work=jnp.asarray(150.0),
        labor_force_reentry_cost=jnp.asarray(200.0),
    )
    expected = 5000.0 - 2000.0 - 150.0 - 200.0
    assert jnp.isclose(result, expected)


def test_leisure_bad_health() -> None:
    result = preferences.leisure_forcedout(
        good_health=jnp.int32(0),
        time_endowment=jnp.asarray(5000.0),
        leisure_cost_of_bad_health=jnp.asarray(500.0),
    )
    assert jnp.isclose(result, 4500.0)


def test_utility_positive_leisure() -> None:
    result = preferences.u_alive(
        consumption_equiv=jnp.array(10000.0),
        leisure=jnp.array(3000.0),
        consumption_weight=jnp.array(0.4),
        coefficient_rra=jnp.array(2.0),
        utility_scale_factor=jnp.array(1.0),
    )
    assert jnp.isfinite(result)


def test_utility_log_case() -> None:
    result = preferences.u_alive(
        consumption_equiv=jnp.array(10000.0),
        leisure=jnp.array(3000.0),
        consumption_weight=jnp.array(0.4),
        coefficient_rra=jnp.array(1.0),
        utility_scale_factor=jnp.array(1.0),
    )
    composite = 10000.0**0.4 * 3000.0**0.6
    expected = jnp.log(composite)
    assert jnp.isclose(result, expected, rtol=1e-5)


def test_bequest_positive_assets() -> None:
    result = preferences.bequest(
        assets=jnp.array(100000.0),
        bequest_shifter=jnp.asarray(5000.0),
        scaled_bequest_weight=jnp.asarray(0.5),
        consumption_weight=jnp.array(0.4),
        coefficient_rra=jnp.array(2.0),
        utility_scale_factor=jnp.array(1.0),
    )
    assert jnp.isfinite(result)


def test_bequest_zero_assets() -> None:
    result = preferences.bequest(
        assets=jnp.array(0.0),
        bequest_shifter=jnp.asarray(5000.0),
        scaled_bequest_weight=jnp.asarray(0.5),
        consumption_weight=jnp.array(0.4),
        coefficient_rra=jnp.array(2.0),
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
        period=jnp.int32(55),
        age=jnp.int32(55),
        claim_ss=jnp.array(social_security.ClaimedSS.no),
        claimed_ss=jnp.array(social_security.ClaimedSS.no),
        normal_retirement_age=jnp.int32(66),
        early_ret_adjustment=jnp.ones(100),
        benefit_withheld_fraction=jnp.array(0.0),
        earnings_test_credited_back=jnp.zeros(100),
        earnings_test_repealed_age=jnp.int32(70),
        pia_table=jnp.array([0.0, 711.9, 2115.1, 3015.1]),
        pia_aime_grid=jnp.array([0.0, 791.0, 4768.0, 8000.0]),
        aime_accrual_factor=jnp.asarray(1 / 35),
        aggregate_wage_growth=jnp.asarray(0.02),
        aime_last_age_with_indexing=jnp.int32(60),
        aime_kink_2=jnp.asarray(8000.0),
        ratio_lowest_earnings=ratio,
    )
    assert result > 1000.0
    assert result <= 8000.0
