"""Runtime-supplied gridpoints for the consumption action.

Consumption is declared as `IrregSpacedGrid(n_points=N)` in
`baseline.regimes._common.build_grids` so the bounds can track
runtime parameters: the lower bound from the per-iteration
`consumption_floor` parameter, the upper bound from the per-creation-time
`max_consumption` fixed param. Callers must inject the actual
gridpoints into `params` via `inject_consumption_points` before
calling `model.solve()` / `model.simulate()`.
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

    Walks `model.regimes`, finds those with `consumption` declared as
    `IrregSpacedGrid` with runtime-supplied points, and writes
    `params[regime_name]["consumption"] = {"points": <pts>}`.

    Lower bound: `params["consumption_floor"]` (varies per iteration).
    Upper bound: `max_consumption` from the regime's resolved
    fixed-params (set once at model creation).

    Args:
        params: Existing params mapping. Returned as a new dict; the input is
            not mutated.
        model: Model whose regime specs determine which regimes need points.

    Returns:
        New params dict with consumption points injected.
    """
    consumption_floor = float(params["consumption_floor"])
    out: dict[str, Any] = dict(params)
    for regime_name, regime in model.regimes.items():
        grid = regime.actions.get("consumption")
        if not (isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime):
            continue
        # Runtime-points grids always have `n_points` set (the constructor
        # rejects the (points=None, n_points=None) combo); narrow for ty.
        assert grid.n_points is not None
        max_consumption = float(
            model.internal_regimes[regime_name].resolved_fixed_params["max_consumption"]
        )
        points = _compute_consumption_points(
            consumption_floor=consumption_floor,
            max_consumption=max_consumption,
            n_points=grid.n_points,
        )
        regime_entry = dict(out.get(regime_name, {}))
        regime_entry["consumption"] = {"points": points}
        out[regime_name] = regime_entry
    return out


def consumption_grid_upper_bound(max_consumption: float) -> float:
    """Surface `max_consumption` in the regime params template.

    pylcm builds the params template from each regime function's
    signature. `max_consumption` is read at runtime by
    `inject_consumption_points` from `resolved_fixed_params`; for
    that to work via pylcm's fixed-params machinery, the key must
    appear in some function's signature. This marker function is
    the entry point — its output is intentionally unused, and
    dags.tree pruning drops the call at solve / simulate time.
    """
    return max_consumption


def _compute_consumption_points(
    *,
    consumption_floor: float,
    max_consumption: float,
    n_points: int,
) -> Array:
    """Return log-spaced consumption gridpoints from floor to max."""
    return jnp.geomspace(consumption_floor, max_consumption, num=n_points)
