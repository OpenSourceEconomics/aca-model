"""Consumption-grid invariants required by the borrowing constraint.

The borrowing constraint in `agent.assets_and_income.borrowing_constraint`
compares the lowest consumption action against
`max(cash_on_hand, consumption_floor * equivalence_scale)`. For subjects
with cash below the floor (HRS bottom-coded `assets=-$1{,}000{,}000$`,
moderate-negative-asset retirees etc.) this RHS collapses to exactly
`consumption_floor` for singles. The constraint is feasible iff the
lowest consumption gridpoint is `<= consumption_floor`.

`jnp.geomspace(start, stop, num=n)` returns `start * r^i` with
`r = (stop/start)^(1/(n-1))`; mathematically `r^0 == 1` so the first
point equals `start`, but XLA backends can drift by sub-ULP for some
`(start, stop, n)` combinations (observed: CUDA, n=70, drift +2.27e-13).
A positive drift above `consumption_floor` flips the kink-boundary `<=`
and rejects every action for those subjects.

`_compute_consumption_points` therefore pins the first point back to
`consumption_floor` after `geomspace`. Test that invariant directly.
"""

import jax.numpy as jnp
import pytest

from aca_model.consumption_grid import _compute_consumption_points


@pytest.mark.parametrize("n_points", [5, 16, 64, 70, 100])
def test_compute_consumption_points_first_equals_floor_exactly(n_points: int) -> None:
    """The first gridpoint equals `consumption_floor` exactly under any `n_points`."""
    consumption_floor = 1597.0921419521899  # production value
    pts = _compute_consumption_points(
        consumption_floor=consumption_floor,
        max_consumption=300_000.0,
        n_points=n_points,
    )
    assert float(pts[0]) == consumption_floor


def test_compute_consumption_points_strictly_increasing() -> None:
    """Gridpoints are strictly increasing — no kink-pinning ties."""
    pts = _compute_consumption_points(
        consumption_floor=1597.0921419521899,
        max_consumption=300_000.0,
        n_points=70,
    )
    diffs = jnp.diff(pts)
    assert bool((diffs > 0).all())
