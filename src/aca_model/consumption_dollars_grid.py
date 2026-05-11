"""Runtime-supplied gridpoints for the consumption_dollars action.

Consumption is declared as `IrregSpacedGrid(n_points=N)` in
`baseline.regimes._common.build_grids` so the bounds can track
runtime parameters: the lower bound from the per-iteration
`consumption_equiv_floor` parameter (and its couples-scaled twin),
the upper bound from `MAX_CONSUMPTION_DOLLARS` in
`baseline.regimes._common`. Callers must inject the actual gridpoints
into `params` via `inject_consumption_dollars_points` before calling
`model.solve()` / `model.simulate()`.

The grid pins the two regime-relevant transfer-floor levels exactly
on the action grid so the borrowing constraint's
`max(cash_on_hand, floor)` boundary lands on a feasible action for
both single and married households:

- `pts[0] = consumption_equiv_floor` (single household: equiv_scale=1)
- `pts[1] = consumption_equiv_floor * 2 ** exponent` (married)
- `pts[2:] = geomspace(pts[1], MAX_CONSUMPTION_DOLLARS, n_points - 1)`
"""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from jax import Array
from lcm import IrregSpacedGrid, Model

from aca_model.baseline.regimes._common import MAX_CONSUMPTION_DOLLARS


def inject_consumption_dollars_points(
    *,
    params: Mapping[str, Any],
    model: Model,
) -> dict[str, Any]:
    """Inject consumption_dollars gridpoints into per-regime params.

    Walks every regime, reads its `consumption_dollars` action grid,
    and writes `params[regime_name]["consumption_dollars"] = {"points": <pts>}`.

    The lower two gridpoints are the single and married Dollar-valued
    transfer floors; the rest are geomspaced from the married floor up
    to `MAX_CONSUMPTION_DOLLARS`.

    Args:
        params: Existing params mapping with `consumption_equiv_floor`
            (per-equivalent floor, varies per iteration). Returned as a
            new dict; the input is not mutated.
        model: Model whose regimes carry the runtime-points grid and
            whose `fixed_params["exponent"]` sets the married
            equivalence-scale exponent.

    Returns:
        New params dict with consumption_dollars points injected.

    Raises:
        ValueError: If a regime is missing the `consumption_dollars`
            action, or its grid is not an `IrregSpacedGrid` with
            `pass_points_at_runtime=True`.
    """
    consumption_equiv_floor = jnp.asarray(params["consumption_equiv_floor"])
    exponent = jnp.asarray(model.fixed_params["exponent"])
    out: dict[str, Any] = dict(params)
    for regime_name, regime in model.regimes.items():
        if regime.terminal:
            continue
        grid = regime.actions.get("consumption_dollars")
        if grid is None:
            msg = (
                f"Regime {regime_name!r} is missing the `consumption_dollars` "
                f"action — the runtime-points grid must be on every regime."
            )
            raise ValueError(msg)
        if not (isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime):
            msg = (
                f"Regime {regime_name!r} has a `consumption_dollars` action "
                f"whose grid is not an `IrregSpacedGrid(pass_points_at_runtime=True)`; "
                f"got {type(grid).__name__}."
            )
            raise ValueError(msg)
        # Runtime-points grids always have `n_points` set (the constructor
        # rejects the (points=None, n_points=None) combo); narrow for ty.
        assert grid.n_points is not None
        points = _compute_consumption_dollars_points(
            consumption_equiv_floor=consumption_equiv_floor,
            exponent=exponent,
            n_points=grid.n_points,
        )
        regime_entry = dict(out.get(regime_name, {}))
        regime_entry["consumption_dollars"] = {"points": points}
        out[regime_name] = regime_entry
    return out


def _compute_consumption_dollars_points(
    *,
    consumption_equiv_floor: Array,
    exponent: Array,
    n_points: int,
) -> Array:
    """Return log-spaced consumption_dollars gridpoints with both floors pinned.

    Single and married households face different Dollar-valued floors
    (`consumption_equiv_floor` and the married-scaled twin
    respectively). Both must land exactly on the action grid so the
    borrowing constraint's `max(cash_on_hand, floor)` kink boundary is
    a feasible action; otherwise sub-ULP drift can flip the `<=`
    comparison for subjects with very negative cash. The geomspace
    tail starts at the married floor and runs to
    `MAX_CONSUMPTION_DOLLARS` so the two pinned points stay strictly
    increasing.
    """
    married_dollar_floor = consumption_equiv_floor * jnp.asarray(2.0) ** exponent
    tail = jnp.geomspace(
        married_dollar_floor, MAX_CONSUMPTION_DOLLARS, num=n_points - 1
    )
    pts = jnp.concatenate([consumption_equiv_floor[None], tail])
    # `jnp.geomspace` returns `start * r^0` for the first tail element,
    # which mathematically equals `married_dollar_floor` but drifts by
    # sub-ULP on some XLA backends. Pin the slot back to the exact
    # arithmetic value so the borrowing-constraint kink boundary at the
    # married floor is exactly representable.
    pts = pts.at[1].set(married_dollar_floor)
    # The runtime params are concrete, not JIT-traced — a Python `if`
    # is fine. Guard against a degenerate grid where the geomspace step
    # is too small for the next point to clear `married_dollar_floor`.
    if not float(married_dollar_floor) < float(pts[2]):
        msg = (
            f"consumption_dollars grid is not strictly increasing at the "
            f"married-floor kink: pts[1]={float(married_dollar_floor):.6g}, "
            f"pts[2]={float(pts[2]):.6g}. Either `MAX_CONSUMPTION_DOLLARS` "
            f"is too close to the married floor or `n_points` is too small."
        )
        raise ValueError(msg)
    return pts
