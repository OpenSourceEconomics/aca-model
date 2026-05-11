"""Subjects at extreme negative assets must clear `validate_initial_conditions`.

The transfer system (`agent.assets_and_income.transfers`) tops cash-on-hand
to the household-$ consumption floor at any starting state, so the lowest
consumption_dollars-grid point is always a feasible action regardless of
how negative starting assets are. The model's constraints — and pylcm's
`validate_initial_conditions` pass — must reflect this.
"""

import jax.numpy as jnp
from lcm import DiscreteGrid
from lcm.simulation.initial_conditions import validate_initial_conditions

from aca_model.agent.assets_and_income import borrowing_constraint
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.benchmark import (
    create_benchmark_model,
    get_benchmark_initial_conditions,
    get_benchmark_params,
)


def test_borrowing_constraint_admits_consumption_dollars_at_floor() -> None:
    """`consumption_dollars == consumption_dollars_floor` at the kink is feasible by equality."""
    consumption_dollars_floor = jnp.asarray(5_000.0)
    cash_on_hand = jnp.asarray(-50_000.0)  # below floor — RHS = floor

    admitted = bool(
        borrowing_constraint(
            consumption_dollars=consumption_dollars_floor,
            cash_on_hand=cash_on_hand,
            consumption_dollars_floor=consumption_dollars_floor,
        )
    )
    assert admitted


def test_borrowing_constraint_admits_consumption_dollars_at_married_floor() -> None:
    """At a married household's higher floor, the equivalence-scale-lifted floor is feasible."""
    consumption_equiv_floor = jnp.asarray(5_000.0)
    married_floor = consumption_equiv_floor * jnp.asarray(2.0) ** 0.7
    cash_on_hand = jnp.asarray(-50_000.0)

    admitted = bool(
        borrowing_constraint(
            consumption_dollars=married_floor,
            cash_on_hand=cash_on_hand,
            consumption_dollars_floor=married_floor,
        )
    )
    assert admitted


def test_borrowing_constraint_rejects_consumption_dollars_above_post_transfer_resources() -> (
    None
):
    """`consumption_dollars > max(cash_on_hand, floor)` is rejected."""
    consumption_dollars_floor = jnp.asarray(5_000.0)
    cash_on_hand = jnp.asarray(-50_000.0)
    consumption_dollars = consumption_dollars_floor + 1.0

    admitted = bool(
        borrowing_constraint(
            consumption_dollars=consumption_dollars,
            cash_on_hand=cash_on_hand,
            consumption_dollars_floor=consumption_dollars_floor,
        )
    )
    assert not admitted


def test_borrowing_constraint_admits_floor_at_million_dollar_negative_cash() -> None:
    """The kink-boundary check survives sub-ULP rounding at `|cash_on_hand| ~ 1e6`.

    At large negative `assets`, the algebraically equivalent
    `cash_on_hand + transfers` form rounds to `floor - 5.7e-11` at fp64,
    flipping `consumption_dollars <= ...` for the lowest
    consumption_dollars gridpoint. The `max(cash_on_hand, floor)` form
    returns `floor` exactly.
    """
    consumption_dollars_floor = jnp.asarray(1597.0921419521899)  # production value
    cash_on_hand = jnp.asarray(-1_000_000.0)
    consumption_dollars = consumption_dollars_floor  # lowest grid point

    admitted = bool(
        borrowing_constraint(
            consumption_dollars=consumption_dollars,
            cash_on_hand=cash_on_hand,
            consumption_dollars_floor=consumption_dollars_floor,
        )
    )
    assert admitted


def test_extreme_negative_assets_subject_passes_validation() -> None:
    """A subject placed at `assets = -1_000_000` clears initial-conditions validation.

    A large-but-reasonable negative value (very bad draws for both HCC shocks)
    should remain in the simulated population: the consumption floor /
    transfer system absorbs them, with `c = c_floor` always feasible.
    """
    n_subjects = 1
    model = create_benchmark_model(
        n_subjects=n_subjects,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )
    _, _, params = get_benchmark_params(model=model)

    initial_conditions = get_benchmark_initial_conditions(
        model=model, n_subjects=n_subjects, seed=0
    )
    initial_conditions = {
        **initial_conditions,
        "assets": jnp.asarray([-1_000_000.0]),
        "regime": jnp.asarray(
            [model.regime_names_to_ids["retiree_nomc_inelig_canwork"]],
            dtype=jnp.int32,
        ),
    }

    internal_params = model._process_params(params)  # noqa: SLF001
    validate_initial_conditions(
        initial_conditions=initial_conditions,
        internal_regimes=model.internal_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
        internal_params=internal_params,
        ages=model.ages,
    )
