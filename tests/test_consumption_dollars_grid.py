"""Consumption-grid invariants required by the borrowing constraint.

The borrowing constraint in `agent.assets_and_income.borrowing_constraint`
compares the lowest consumption_dollars action against
`max(cash_on_hand, consumption_dollars_floor)`. For subjects with cash
below the floor (large-negative-asset subjects, moderate-negative-asset
retirees, etc.) this RHS collapses to exactly
`consumption_dollars_floor`. The constraint is feasible iff the
relevant household-floor gridpoint is `<= consumption_dollars_floor`.

For singles (`equivalence_scale = 1`) that floor is
`consumption_equiv_floor`; for married households
(`equivalence_scale = 2 ** exponent`) it is
`consumption_equiv_floor * 2 ** exponent`. Both must land **exactly**
on the consumption_dollars grid.

`jnp.geomspace(start, stop, num=n)` returns `start * r^i` with
`r = (stop/start)^(1/(n-1))`; mathematically `r^0 == 1` so the first
point equals `start`, but XLA backends can drift by sub-ULP for some
`(start, stop, n)` combinations (observed: CUDA, n=70, drift +2.27e-13).
A positive drift above the floor flips the kink-boundary `<=` and
rejects every action for the affected subjects.

`_compute_consumption_dollars_points` therefore prepends the singles'
floor as `pts[0]`, runs `geomspace` from the married floor up to the
caller-supplied `max_consumption_dollars` for the rest, and pins the
geomspace start back to the married floor exactly. Test those invariants
directly.
"""

import jax.numpy as jnp
import pytest

from aca_model.consumption_dollars_grid import _compute_consumption_dollars_points

EXPONENT = 0.7  # production value (env_constants["exponent"])
SINGLE_FLOOR = 1597.0921419521899  # production value
MARRIED_SCALE = 2.0**EXPONENT
MAX_CONSUMPTION_DOLLARS = 300_000.0  # production value (env_constants)


@pytest.mark.parametrize("n_points", [5, 16, 64, 70, 100])
def test_compute_consumption_dollars_points_first_equals_singles_floor(
    n_points: int,
) -> None:
    """`pts[0]` equals the singles' floor exactly under any `n_points`."""
    pts = _compute_consumption_dollars_points(
        consumption_equiv_floor=jnp.asarray(SINGLE_FLOOR),
        exponent=jnp.asarray(EXPONENT),
        max_consumption_dollars=jnp.asarray(MAX_CONSUMPTION_DOLLARS),
        n_points=n_points,
    )
    assert float(pts[0]) == SINGLE_FLOOR


@pytest.mark.parametrize("n_points", [5, 16, 64, 70, 100])
def test_compute_consumption_dollars_points_second_equals_married_floor(
    n_points: int,
) -> None:
    """`pts[1]` equals `consumption_equiv_floor * 2 ** exponent` exactly."""
    pts = _compute_consumption_dollars_points(
        consumption_equiv_floor=jnp.asarray(SINGLE_FLOOR),
        exponent=jnp.asarray(EXPONENT),
        max_consumption_dollars=jnp.asarray(MAX_CONSUMPTION_DOLLARS),
        n_points=n_points,
    )
    expected = float(jnp.asarray(SINGLE_FLOOR) * jnp.asarray(2.0) ** EXPONENT)
    assert float(pts[1]) == expected


def test_compute_consumption_dollars_points_strictly_increasing() -> None:
    """Gridpoints are strictly increasing — no kink-pinning ties."""
    pts = _compute_consumption_dollars_points(
        consumption_equiv_floor=jnp.asarray(SINGLE_FLOOR),
        exponent=jnp.asarray(EXPONENT),
        max_consumption_dollars=jnp.asarray(MAX_CONSUMPTION_DOLLARS),
        n_points=70,
    )
    diffs = jnp.diff(pts)
    assert bool((diffs > 0).all())


def test_compute_consumption_dollars_points_last_equals_max() -> None:
    """The final point is the configured upper bound."""
    pts = _compute_consumption_dollars_points(
        consumption_equiv_floor=jnp.asarray(SINGLE_FLOOR),
        exponent=jnp.asarray(EXPONENT),
        max_consumption_dollars=jnp.asarray(MAX_CONSUMPTION_DOLLARS),
        n_points=70,
    )
    assert float(pts[-1]) == pytest.approx(MAX_CONSUMPTION_DOLLARS)
