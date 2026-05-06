"""Subjects at extreme negative assets must clear `validate_initial_conditions`.

The transfer system (`agent.assets_and_income.transfers`) tops cash-on-hand
to `consumption_floor * equivalence_scale` at any starting state, so the
lowest consumption-grid point is always a feasible action regardless of
how negative starting assets are. The model's constraints — and pylcm's
`validate_initial_conditions` pass — must reflect this.
"""

import jax.numpy as jnp
from lcm.simulation.initial_conditions import validate_initial_conditions

from aca_model.agent.assets_and_income import borrowing_constraint
from aca_model.benchmark import (
    create_benchmark_model,
    get_benchmark_initial_conditions,
    get_benchmark_params,
)


def test_borrowing_constraint_admits_c_floor_at_million_dollar_negative_cash() -> None:
    """At `cash_on_hand = -$1M` (fp32), `c = c_floor` remains a feasible choice.

    Computing `cash_on_hand + transfers` directly suffers float32 catastrophic
    cancellation: `(-1e6) + (c_floor + 1e6)` loses ~0.1 of precision, enough
    to wipe out the `c == c_floor` boundary. The constraint must use the
    algebraically equivalent but numerically stable `max(cash_on_hand, floor)`
    form.
    """
    consumption_floor = 5_000.0
    admitted = bool(
        borrowing_constraint(
            consumption=jnp.float32(consumption_floor),
            cash_on_hand=jnp.float32(-1_000_000.0),
            consumption_floor=consumption_floor,
            equivalence_scale=jnp.float32(1.0),
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
    model = create_benchmark_model(n_subjects=n_subjects)
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
