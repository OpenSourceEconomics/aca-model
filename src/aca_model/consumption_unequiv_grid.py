"""Runtime-supplied gridpoints for the consumption_unequiv action.

Consumption is declared as `IrregSpacedGrid(n_points=N)` in
`baseline.regimes._common.build_grids` so the bounds can track
runtime parameters: the lower bound from the per-iteration
`consumption_unequiv_floor` parameter, the upper bound from
`MAX_CONSUMPTION_UNEQUIV` in `baseline.regimes._common`, which the
`create_model` factories attach to `model.max_consumption_unequiv`.
Callers must inject the actual gridpoints into `params` via
`inject_consumption_unequiv_points` before calling `model.solve()` /
`model.simulate()`.
"""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from jax import Array
from lcm import IrregSpacedGrid, Model


def inject_consumption_unequiv_points(
    *,
    params: Mapping[str, Any],
    model: Model,
) -> dict[str, Any]:
    """Inject consumption_unequiv gridpoints into per-regime params.

    Walks every regime, finds the action whose grid is an
    `IrregSpacedGrid` with runtime-supplied points, and writes
    `params[regime_name]["consumption_unequiv"] = {"points": <pts>}`.

    Lower bound: `params["consumption_unequiv_floor"]` (varies per iteration).
    Upper bound: `model.max_consumption_unequiv` (set by the `create_model`
    factory from `MAX_CONSUMPTION_UNEQUIV` in `baseline.regimes._common`).

    Args:
        params: Existing params mapping. Returned as a new dict; the input is
            not mutated.
        model: Model whose regime specs determine which regimes need points.

    Returns:
        New params dict with consumption_unequiv points injected.
    """
    consumption_unequiv_floor = float(params["consumption_unequiv_floor"])
    max_consumption_unequiv = float(model.max_consumption_unequiv)
    out: dict[str, Any] = dict(params)
    for regime_name, regime in model.regimes.items():
        grid = regime.actions.get("consumption_unequiv")
        if not (isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime):
            continue
        # Runtime-points grids always have `n_points` set (the constructor
        # rejects the (points=None, n_points=None) combo); narrow for ty.
        assert grid.n_points is not None
        points = _compute_consumption_unequiv_points(
            consumption_unequiv_floor=consumption_unequiv_floor,
            max_consumption_unequiv=max_consumption_unequiv,
            n_points=grid.n_points,
        )
        regime_entry = dict(out.get(regime_name, {}))
        regime_entry["consumption_unequiv"] = {"points": points}
        out[regime_name] = regime_entry
    return out


def _compute_consumption_unequiv_points(
    *,
    consumption_unequiv_floor: float,
    max_consumption_unequiv: float,
    n_points: int,
) -> Array:
    """Return log-spaced consumption_unequiv gridpoints from floor to max.

    `jnp.geomspace` computes intermediate points as `start * r^i` with
    `r = (stop/start)^(1/(n-1))`; the first point is `start * r^0`,
    which is `start` mathematically but can be off by sub-ULP under
    some XLA backends (CUDA + 70 points: `start + 2.27e-13`). The
    borrowing constraint compares the first action against
    `max(cash_on_hand, consumption_unequiv_floor)`, and any positive drift
    above `consumption_unequiv_floor` flips the kink-boundary `<=` for
    subjects with very negative cash. Pin the first element back to
    `consumption_unequiv_floor` exactly.
    """
    pts = jnp.geomspace(consumption_unequiv_floor, max_consumption_unequiv, num=n_points)
    return pts.at[0].set(consumption_unequiv_floor)
