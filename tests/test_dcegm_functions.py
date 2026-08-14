"""DC-EGM building blocks: resources, savings, the savings-form assets laws,
and the CES inverse marginal utility.

These functions re-express the existing budget identity in post-decision
(savings) form — algebraically identical to the brute-force spec — and
provide the analytical `(u')⁻¹` the Euler inversion needs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aca_model.agent import assets_and_income, preferences

jax.config.update("jax_enable_x64", True)


def test_resources_equal_cash_on_hand_plus_transfers() -> None:
    """`resources = max(cash_on_hand, floor)` matches the budget identity
    `cash_on_hand + transfers` everywhere."""
    cash_on_hand = jnp.asarray([-50_000.0, 0.0, 8_000.0, 350_000.0])
    floor = jnp.asarray([9_000.0, 9_000.0, 9_000.0, 9_000.0])
    transfers = assets_and_income.transfers(
        cash_on_hand=cash_on_hand, consumption_dollars_floor=floor
    )
    np.testing.assert_allclose(
        assets_and_income.resources(
            cash_on_hand=cash_on_hand,
            consumption_dollars_floor=floor,
        ),
        cash_on_hand + transfers,
        atol=1e-9,
    )


def test_next_assets_from_savings_matches_direct_form() -> None:
    """The savings-form assets law equals the direct
    `cash_on_hand + transfers + adjustment - consumption - oop` law."""
    cash_on_hand = jnp.asarray([-20_000.0, 12_000.0, 90_000.0])
    floor = jnp.full(3, 9_000.0)
    consumption = jnp.asarray([5_000.0, 9_500.0, 40_000.0])
    oop = jnp.asarray([1_000.0, 0.0, 25_000.0])
    adjustment = jnp.asarray([-300.0, 0.0, 700.0])

    transfers = assets_and_income.transfers(
        cash_on_hand=cash_on_hand, consumption_dollars_floor=floor
    )
    expected = assets_and_income.next_assets(
        cash_on_hand=cash_on_hand,
        transfers=transfers,
        pension_assets_adjustment=adjustment,
        consumption_dollars=consumption,
        oop_costs=oop,
    )
    savings = assets_and_income.savings(
        resources=assets_and_income.resources(
            cash_on_hand=cash_on_hand,
            consumption_dollars_floor=floor,
        ),
        consumption_dollars=consumption,
    )
    np.testing.assert_allclose(
        assets_and_income.next_assets_from_savings(
            savings=savings, pension_assets_adjustment=adjustment, oop_costs=oop
        ),
        expected,
        atol=1e-9,
    )


def test_next_assets_when_dead_from_savings_matches_direct_form() -> None:
    """The savings-form dead-target law equals the direct law (no pension
    adjustment term)."""
    cash_on_hand = jnp.asarray([15_000.0, 200_000.0])
    floor = jnp.full(2, 9_000.0)
    consumption = jnp.asarray([10_000.0, 60_000.0])
    oop = jnp.asarray([3_000.0, 80_000.0])

    transfers = assets_and_income.transfers(
        cash_on_hand=cash_on_hand, consumption_dollars_floor=floor
    )
    expected = assets_and_income.next_assets_when_dead(
        cash_on_hand=cash_on_hand,
        transfers=transfers,
        consumption_dollars=consumption,
        oop_costs=oop,
    )
    savings = assets_and_income.savings(
        resources=assets_and_income.resources(
            cash_on_hand=cash_on_hand,
            consumption_dollars_floor=floor,
        ),
        consumption_dollars=consumption,
    )
    np.testing.assert_allclose(
        assets_and_income.next_assets_when_dead_from_savings(
            savings=savings, oop_costs=oop
        ),
        expected,
        atol=1e-9,
    )


@pytest.mark.parametrize("coefficient_rra", [0.7, 1.0, 1.5, 3.2])
def test_inverse_marginal_utility_inverts_du_dc(coefficient_rra: float) -> None:
    """`(u')⁻¹(u'(c)) == c` where `u'` is the autodiff derivative of `u_alive`
    with respect to consumption dollars."""
    consumption = jnp.asarray(27_500.0)
    leisure = jnp.asarray(0.62)
    equivalence_scale = jnp.asarray(1.34)
    consumption_weight = jnp.asarray(0.55)
    rra = jnp.asarray(coefficient_rra)
    scale = jnp.asarray(140.0)

    def utility_of_consumption(consumption_dollars: jnp.ndarray) -> jnp.ndarray:
        return preferences.u_alive(
            consumption_equiv=preferences.consumption_equiv(
                consumption_dollars=consumption_dollars,
                equivalence_scale=equivalence_scale,
            ),
            leisure=leisure,
            consumption_weight=consumption_weight,
            coefficient_rra=rra,
            utility_scale_factor=scale,
        )

    marginal = jax.grad(utility_of_consumption)(consumption)
    recovered = preferences.inverse_marginal_utility(
        marginal_continuation=marginal,
        leisure=leisure,
        equivalence_scale=equivalence_scale,
        consumption_weight=consumption_weight,
        coefficient_rra=rra,
        utility_scale_factor=scale,
    )
    np.testing.assert_allclose(recovered, consumption, rtol=1e-9)
