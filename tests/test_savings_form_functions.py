"""The post-decision budget: resources, savings, and the savings-form laws.

NB-EGM reads the budget in post-decision (savings) form. These functions
re-express the brute-force budget identity in that form and are algebraically
identical to it, which is what makes a solver swap a solver swap rather than a
change of model.
"""

import jax
import jax.numpy as jnp
import numpy as np

from aca_model.agent import assets_and_income

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
            consumption_floor_schedule=floor,
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
            consumption_floor_schedule=floor,
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
            consumption_floor_schedule=floor,
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
