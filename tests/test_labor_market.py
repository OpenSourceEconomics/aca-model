"""Tests for labor-market wage and income functions."""

import jax.numpy as jnp
import numpy as np
import pytest

from aca_model.agent import labor_market
from aca_model.agent.labor_market import LaborSupply


def test_hours_values_is_host_array_so_import_allocates_no_device_memory() -> None:
    """`HOURS_VALUES` is a host (NumPy) array, not a device-pinned JAX array.

    A module-level JAX array materializes on the default device the moment the
    module is imported, reserving the GPU memory pool in every process that
    imports the model — including the estimation orchestrator, which only
    launches GPU worker ranks and must leave the devices free for them.
    """
    assert isinstance(labor_market.HOURS_VALUES, np.ndarray)


@pytest.mark.parametrize(
    ("choice", "expected_hours"),
    [(0, 0.0), (1, 1000.0), (2, 1500.0), (3, 2000.0), (4, 2500.0)],
)
def test_working_hours_value_maps_choice_to_annual_hours(
    choice: int, expected_hours: float
) -> None:
    """Each labor-supply choice maps to its annual hours worked."""
    result = labor_market.working_hours_value(jnp.asarray(choice, dtype=jnp.int32))
    np.testing.assert_allclose(float(result), expected_hours)


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
