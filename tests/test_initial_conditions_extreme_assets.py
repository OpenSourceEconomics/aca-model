"""Subjects at extreme negative assets must clear `validate_initial_conditions`.

The transfer system (`agent.assets_and_income.transfers`) tops cash-on-hand
to `consumption_floor * equivalence_scale` at any starting state, so the
lowest consumption-grid point is always a feasible action regardless of
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


def test_borrowing_constraint_admits_consumption_at_floor() -> None:
    """`consumption == consumption_floor` at the kink is feasible by equality."""
    consumption_floor = 5_000.0
    equivalence_scale = jnp.asarray(1.0)
    cash_on_hand = jnp.asarray(-50_000.0)  # below floor — RHS = floor

    admitted = bool(
        borrowing_constraint(
            consumption=jnp.asarray(consumption_floor),
            cash_on_hand=cash_on_hand,
            consumption_floor=consumption_floor,
            equivalence_scale=equivalence_scale,
        )
    )
    assert admitted


def test_borrowing_constraint_rejects_consumption_above_post_transfer_resources() -> (
    None
):
    """`consumption > max(cash_on_hand, floor)` is rejected."""
    consumption_floor = 5_000.0
    equivalence_scale = jnp.asarray(1.0)
    cash_on_hand = jnp.asarray(-50_000.0)
    consumption = jnp.asarray(consumption_floor + 1.0)

    admitted = bool(
        borrowing_constraint(
            consumption=consumption,
            cash_on_hand=cash_on_hand,
            consumption_floor=consumption_floor,
            equivalence_scale=equivalence_scale,
        )
    )
    assert not admitted


def test_borrowing_constraint_admits_floor_at_million_dollar_negative_cash() -> None:
    """The kink-boundary check survives sub-ULP rounding at `|cash_on_hand| ~ 1e6`.

    Reproduces the production failure mode at `assets=-$1{,}000{,}000$` (HRS
    bottom-code): the algebraically equivalent `cash_on_hand + transfers`
    form rounds to `floor - 5.7e-11` at fp64, flipping `consumption <= ...`
    for the lowest consumption gridpoint. The `max(cash_on_hand, floor)`
    form returns `floor` exactly.
    """
    consumption_floor = 1597.0921419521899  # production value
    equivalence_scale = jnp.asarray(1.0)
    cash_on_hand = jnp.asarray(-1_000_000.0)
    consumption = jnp.asarray(consumption_floor)  # lowest grid point

    admitted = bool(
        borrowing_constraint(
            consumption=consumption,
            cash_on_hand=cash_on_hand,
            consumption_floor=consumption_floor,
            equivalence_scale=equivalence_scale,
        )
    )
    assert admitted


def test_extreme_negative_assets_subject_passes_validation() -> None:
    """A subject placed at `assets = -1_000_000` clears initial-conditions validation.

    HRS bottom-codes very-large-negative net wealth at exactly $-1{,}000{,}000$.
    Such subjects should remain in the simulated population: the consumption
    floor / transfer system absorbs them, with `c = c_floor` always feasible.
    """
    n_subjects = 1
    model = create_benchmark_model(
        n_subjects=n_subjects,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )
    _, params = get_benchmark_params(model=model)

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
