"""Runtime-supplied gridpoints for the consumption action.

Consumption is declared as `IrregSpacedGrid(n_points=N)` in
`baseline.regimes._common.build_grids` so the bounds can track
runtime parameters: the lower bound from the per-iteration
`consumption_floor` parameter, the upper bound from
`MAX_CONSUMPTION` in `baseline.regimes._common`, which the
`create_model` factories attach to `model.max_consumption`.
Callers must inject the actual gridpoints into `params` via
`inject_consumption_points` before calling `model.solve()` /
`model.simulate()`.
"""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from jax import Array
from lcm import IrregSpacedGrid, Model


def inject_consumption_points(
    *,
    params: Mapping[str, Any],
    model: Model,
) -> dict[str, Any]:
    """Inject consumption gridpoints into per-regime params.

    Walks every regime, finds the action whose grid is an
    `IrregSpacedGrid` with runtime-supplied points, and writes
    `params[regime_name]["consumption"] = {"points": <pts>}`.

    Lower bound: `params["consumption_floor"]` (varies per iteration).
    Upper bound: `model.max_consumption` (set by the `create_model`
    factory from `MAX_CONSUMPTION` in `baseline.regimes._common`).

    Args:
        params: Existing params mapping. Returned as a new dict; the input is
            not mutated.
        model: Model whose regime specs determine which regimes need points.

    Returns:
        New params dict with consumption points injected.
    """
    consumption_floor = float(params["consumption_floor"])
    max_consumption = float(model.max_consumption)
    out: dict[str, Any] = dict(params)
    for regime_name, regime in model.regimes.items():
        grid = regime.actions.get("consumption")
        if not (isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime):
            continue
        # Runtime-points grids always have `n_points` set (the constructor
        # rejects the (points=None, n_points=None) combo); narrow for ty.
        assert grid.n_points is not None
        points = _compute_consumption_points(
            consumption_floor=consumption_floor,
            max_consumption=max_consumption,
            n_points=grid.n_points,
        )
        regime_entry = dict(out.get(regime_name, {}))
        regime_entry["consumption"] = {"points": points}
        out[regime_name] = regime_entry
    return out


def _compute_consumption_points(
    *,
    consumption_floor: float,
    max_consumption: float,
    n_points: int,
) -> Array:
    """Return log-spaced consumption gridpoints from floor to max."""
    return jnp.geomspace(consumption_floor, max_consumption, num=n_points)
