"""Runtime-supplied gridpoints for the consumption action.

Consumption is declared as `IrregSpacedGrid(n_points=N)` in
`baseline.regimes._common.build_grids` so the lower bound can track
the per-iteration `consumption_floor` parameter. Callers must inject
the actual gridpoints into `params` via `inject_consumption_points`
before calling `model.solve()` / `model.simulate()`.
"""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from jax import Array
from lcm import IrregSpacedGrid, Model

MAX_CONSUMPTION: float = 300_000.0
"""Upper bound of the consumption grid in $/year. Brackets the unconstrained
CRRA optimum for the highest-asset, highest-income agents in the state space."""


def compute_consumption_points(*, consumption_floor: float, n_points: int) -> Array:
    """Return log-spaced consumption gridpoints from the floor to `MAX_CONSUMPTION`.

    Args:
        consumption_floor: Lowest gridpoint, equal to the `consumption_floor`
            parameter so the agent cannot pick `c < floor` even when saving
            from a transfer top-up.
        n_points: Total number of gridpoints.

    Returns:
        1-D float array of length `n_points`.
    """
    return jnp.geomspace(consumption_floor, MAX_CONSUMPTION, num=n_points)


def inject_consumption_points(
    *,
    params: Mapping[str, Any],
    model: Model,
    consumption_floor: float | None = None,
) -> dict[str, Any]:
    """Inject consumption gridpoints into per-regime params.

    Walks `model.regimes`, finds those with `consumption` declared as
    `IrregSpacedGrid` with runtime-supplied points, and writes
    `params[regime_name]["consumption"] = {"points": <pts>}`.

    Args:
        params: Existing params mapping. Returned as a new dict; the input is
            not mutated.
        model: Model whose regime specs determine which regimes need points.
        consumption_floor: Lowest gridpoint. When `None`, taken from
            `params["consumption_floor"]`.

    Returns:
        New params dict with consumption points injected.
    """
    if consumption_floor is None:
        consumption_floor = float(params["consumption_floor"])
    out: dict[str, Any] = dict(params)
    for regime_name, regime in model.regimes.items():
        grid = regime.actions.get("consumption")
        if not (isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime):
            continue
        points = compute_consumption_points(
            consumption_floor=consumption_floor, n_points=grid.n_points
        )
        regime_entry = dict(out.get(regime_name, {}))
        regime_entry["consumption"] = {"points": points}
        out[regime_name] = regime_entry
    return out
