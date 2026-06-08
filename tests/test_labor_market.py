"""Tests for labor-market wage and income functions."""

import jax.numpy as jnp
import numpy as np

from aca_model.agent import labor_market
from aca_model.agent.labor_market import LaborSupply


def test_wage_combines_age_health_profile_with_residual() -> None:
    """`wage = exp(log_ft_wage_mean[period, good_health] + log_ft_wage_std * res)`."""
    log_ft_wage_mean = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # [period, good_health]
    result = labor_market.wage(
        log_ft_wage_res=jnp.asarray(0.5),
        period=jnp.asarray(1, dtype=jnp.int32),
        good_health=jnp.asarray(0, dtype=jnp.int32),
        log_ft_wage_mean=log_ft_wage_mean,
        log_ft_wage_std=jnp.asarray(2.0),
    )
    expected = float(jnp.exp(3.0 + 2.0 * 0.5))  # mean[1, 0] = 3.0
    np.testing.assert_allclose(float(result), expected, rtol=1e-6)


def test_income_scales_wage_by_hours_interaction() -> None:
    """`income = wage * hours^(1 + exp) * int^(-exp)` for a working agent."""
    result = labor_market.income(
        wage=jnp.asarray(10.0),
        labor_supply=LaborSupply.h2000,
        adj_wage_hours_exp=jnp.asarray(0.4),
        adj_wage_hours_int=jnp.asarray(2000.0),
    )
    expected = 10.0 * 2000.0 ** (1.0 + 0.4) * 2000.0 ** (-0.4)
    np.testing.assert_allclose(float(result), expected, rtol=1e-6)


def test_income_is_zero_when_not_working() -> None:
    """A non-working labor supply yields zero income regardless of the wage."""
    result = labor_market.income(
        wage=jnp.asarray(10.0),
        labor_supply=LaborSupply.do_not_work,
        adj_wage_hours_exp=jnp.asarray(0.4),
        adj_wage_hours_int=jnp.asarray(2000.0),
    )
    assert float(result) == 0.0
