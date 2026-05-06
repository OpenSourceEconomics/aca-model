"""Subjects at extreme negative assets must clear `validate_initial_conditions`.

The transfer system (`agent.assets_and_income.transfers`) tops cash-on-hand
to `consumption_floor * equivalence_scale` at any starting state, so the
lowest consumption-grid point is always a feasible action regardless of
how negative starting assets are. The model's constraints — and pylcm's
`validate_initial_conditions` pass — must reflect this.
"""

import jax.numpy as jnp
from lcm.simulation.initial_conditions import validate_initial_conditions

from aca_model.benchmark import (
    create_benchmark_model,
    get_benchmark_initial_conditions,
    get_benchmark_params,
)


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
