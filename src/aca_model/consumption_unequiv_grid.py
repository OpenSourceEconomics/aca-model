"""Runtime-supplied gridpoints for the consumption_unequiv action.

Consumption is declared as `IrregSpacedGrid(n_points=N)` in
`baseline.regimes._common.build_grids` so the bounds can track
runtime parameters: the lower bound from the per-iteration
`consumption_equiv_floor` parameter (and its couples-scaled twin),
the upper bound from `MAX_CONSUMPTION_UNEQUIV` in
`baseline.regimes._common`. Callers must inject the actual gridpoints
into `params` via `inject_consumption_unequiv_points` before calling
`model.solve()` / `model.simulate()`.

The grid pins the two regime-relevant transfer-floor levels exactly
on the action grid so the borrowing constraint's
`max(cash_on_hand, floor)` boundary lands on a feasible action for
both single and married households:

- `pts[0] = consumption_equiv_floor` (single household: equiv_scale=1)
- `pts[1] = consumption_equiv_floor * 2 ** exponent` (married)
- `pts[2:] = geomspace(pts[1], MAX_CONSUMPTION_UNEQUIV, n_points - 1)`
"""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from jax import Array
from lcm import IrregSpacedGrid, Model

from aca_model.baseline.regimes._common import MAX_CONSUMPTION_UNEQUIV


def inject_consumption_unequiv_points(
    *,
    params: Mapping[str, Any],
    model: Model,
) -> dict[str, Any]:
    """Inject consumption_unequiv gridpoints into per-regime params.

    Walks every regime, finds the action whose grid is an
    `IrregSpacedGrid` with runtime-supplied points, and writes
    `params[regime_name]["consumption_unequiv"] = {"points": <pts>}`.

    The lower two gridpoints are the single and married unequiv
    transfer floors (`consumption_equiv_floor` and
    `consumption_equiv_floor * 2 ** exponent`); the rest are
    geomspaced from the married floor up to `MAX_CONSUMPTION_UNEQUIV`.

    Args:
        params: Existing params mapping with `consumption_equiv_floor`
            (per-equivalent floor, varies per iteration). Returned as a
            new dict; the input is not mutated.
        model: Model whose regime specs determine which regimes need points
            and whose `fixed_params["exponent"]` sets the married
            equivalence-scale exponent.

    Returns:
        New params dict with consumption_unequiv points injected.
    """
    consumption_equiv_floor = jnp.asarray(params["consumption_equiv_floor"])
    exponent = jnp.asarray(model.fixed_params["exponent"])
    out: dict[str, Any] = dict(params)
    for regime_name, regime in model.regimes.items():
        grid = regime.actions.get("consumption_unequiv")
        if not (isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime):
            continue
        # Runtime-points grids always have `n_points` set (the constructor
        # rejects the (points=None, n_points=None) combo); narrow for ty.
        assert grid.n_points is not None
        points = _compute_consumption_unequiv_points(
            consumption_equiv_floor=consumption_equiv_floor,
            exponent=exponent,
            n_points=grid.n_points,
        )
        regime_entry = dict(out.get(regime_name, {}))
        regime_entry["consumption_unequiv"] = {"points": points}
        out[regime_name] = regime_entry
    return out


def _compute_consumption_unequiv_points(
    *,
    consumption_equiv_floor: Array,
    exponent: Array,
    n_points: int,
) -> Array:
    """Return log-spaced consumption_unequiv gridpoints with both floors pinned.

    Single and married households face different unequiv (in-$) floors
    (`consumption_equiv_floor` and `consumption_equiv_floor *
    2 ** exponent` respectively). Both must land exactly on the action
    grid so the borrowing constraint's `max(cash_on_hand, floor)` kink
    boundary is a feasible action; otherwise sub-ULP drift can flip
    the `<=` comparison for subjects with very negative cash. The
    geomspace tail starts at the married floor and runs to
    `MAX_CONSUMPTION_UNEQUIV` so the two pinned points stay strictly
    increasing.

    All arithmetic stays in jax — multiplying `consumption_equiv_floor`
    by `2 ** exponent` in jnp keeps both pinned floors at the canonical
    float dtype the model uses everywhere else.
    """
    married_unequiv_floor = consumption_equiv_floor * jnp.asarray(2.0) ** exponent
    tail = jnp.geomspace(
        married_unequiv_floor, MAX_CONSUMPTION_UNEQUIV, num=n_points - 1
    )
    pts = jnp.concatenate([consumption_equiv_floor[None], tail])
    # `jnp.geomspace` returns `start * r^0` for the first tail element,
    # which mathematically equals `married_unequiv_floor` but drifts by
    # sub-ULP on some XLA backends. Pin the slot back to the exact
    # arithmetic value so the borrowing-constraint kink boundary at the
    # married floor is exactly representable.
    return pts.at[1].set(married_unequiv_floor)
