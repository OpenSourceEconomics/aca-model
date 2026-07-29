"""Runtime-supplied gridpoints for the consumption_dollars action.

Consumption is declared as `IrregSpacedGrid(n_points=N)` in
`baseline.regimes._common.build_grids` so the bounds can track
runtime parameters: the lower bound from the per-iteration
`consumption_equiv_floor` parameter (and its couples-scaled twin),
the upper bound from `max_consumption_dollars` supplied directly
by the caller. Callers must inject the actual gridpoints into
`params` via `inject_consumption_dollars_points` before calling
`model.solve()` / `model.simulate()`.

The grid pins the two regime-relevant transfer-floor levels exactly
on the action grid so the borrowing constraint's
`max(cash_on_hand, floor)` boundary lands on a feasible action for
both single and married households:

- `pts[0] = consumption_equiv_floor` (single household: equiv_scale=1)
- `pts[1] = consumption_equiv_floor * 2 ** exponent` (married)
- `pts[2:] = geomspace(pts[1], max_consumption_dollars, n_points - 1)`
"""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from jax import Array
from lcm import IrregSpacedGrid, Model


def inject_consumption_dollars_points(
    *,
    params: Mapping[str, Any],
    model: Model,
    max_consumption_dollars: float,
) -> dict[str, Any]:
    """Inject consumption_dollars gridpoints into per-regime params.

    Walks every regime, reads its `consumption_dollars` action grid,
    and writes `params[regime_name]["consumption_dollars"] = {"points": <pts>}`.

    The lower two gridpoints are the single and married Dollar-valued
    transfer floors; the rest are geomspaced from the married floor up
    to `max_consumption_dollars`.

    Args:
        params: Existing params mapping with `consumption_equiv_floor`
            (per-equivalent floor, varies per iteration). Returned as a
            new dict; the input is not mutated.
        model: Model whose regimes carry the runtime-points grid and
            whose `fixed_params` supplies `exponent` (married
            equivalence-scale exponent).
        max_consumption_dollars: Grid upper bound. Sourced from the
            caller (e.g. aca-data's `environment_constants.pkl`); not
            routed through pylcm's params machinery because no DAG
            function consumes it.

    Returns:
        New params dict with consumption_dollars points injected.

    Raises:
        ValueError: If a regime is missing the `consumption_dollars`
            action.
        TypeError: If a regime's `consumption_dollars` grid is not an
            `IrregSpacedGrid`.
    """
    consumption_equiv_floor = jnp.asarray(params["consumption_equiv_floor"])
    exponent = jnp.asarray(model.fixed_params["exponent"])
    max_consumption_dollars_arr = jnp.asarray(max_consumption_dollars)
    out: dict[str, Any] = dict(params)
    for regime_name, regime in model.user_regimes.items():
        if regime.terminal:
            continue
        grid = regime.actions.get("consumption_dollars")
        if grid is None:
            msg = (
                f"Regime {regime_name!r} is missing the `consumption_dollars` "
                f"action — the runtime-points grid must be on every regime."
            )
            raise ValueError(msg)
        if isinstance(grid, IrregSpacedGrid) and not grid.pass_points_at_runtime:
            # Construction-time points (the DC-EGM path) — nothing to inject.
            continue
        if not isinstance(grid, IrregSpacedGrid):
            msg = (
                f"Regime {regime_name!r} has a `consumption_dollars` action "
                f"whose grid is not an `IrregSpacedGrid(pass_points_at_runtime=True)`; "
                f"got {type(grid).__name__}."
            )
            raise TypeError(msg)
        # Runtime-points grids always have `n_points` set (the constructor
        # rejects the (points=None, n_points=None) combo); narrow for ty.
        assert grid.n_points is not None
        points = compute_consumption_dollars_points(
            consumption_equiv_floor=consumption_equiv_floor,
            exponent=exponent,
            max_consumption_dollars=max_consumption_dollars_arr,
            n_points=grid.n_points,
        )
        regime_entry = dict(out.get(regime_name, {}))
        regime_entry["consumption_dollars"] = {"points": points}
        out[regime_name] = regime_entry
    return out


def inject_consumption_floor_schedule(
    *,
    params: Mapping[str, Any],
    model: Model,
) -> dict[str, Any]:
    """Inject the spousal-indexed floor table into per-regime params.

    Every regime carrying the savings-form `resources` budget reads the
    declared floor-kink threshold from `consumption_floor_schedule`. The
    table is derived from the per-iteration `consumption_equiv_floor` and
    the married equivalence-scale `exponent`, so it is injected alongside
    the consumption gridpoints rather than frozen into fixed params.

    Args:
        params: Existing params mapping with `consumption_equiv_floor`.
            Returned as a new dict; the input is not mutated.
        model: Model whose `fixed_params` supplies `exponent` and whose
            regimes determine where the table is required.

    Returns:
        New params dict with the floor schedule injected.

    """
    schedule = compute_consumption_floor_schedule(
        consumption_equiv_floor=jnp.asarray(params["consumption_equiv_floor"]),
        exponent=jnp.asarray(model.fixed_params["exponent"]),
    )
    out: dict[str, Any] = dict(params)
    for regime_name, regime in model.user_regimes.items():
        if regime.terminal or "resources" not in regime.functions:
            continue
        regime_entry = dict(out.get(regime_name, {}))
        resources_entry = dict(regime_entry.get("resources", {}))
        resources_entry["consumption_floor_schedule"] = schedule
        regime_entry["resources"] = resources_entry
        out[regime_name] = regime_entry
    return out


def compute_consumption_floor_schedule(
    *,
    consumption_equiv_floor: Array,
    exponent: Array,
) -> Array:
    """Return the per-`spousal_income` $-floor table.

    One row per `SpousalIncome` code — `[single, married_no_inc,
    married_has_inc]` — with the married codes sharing the scaled floor
    `consumption_equiv_floor * 2 ** exponent`. Equal by construction to
    `consumption_dollars_floor` evaluated at each code.
    """
    married_dollar_floor = consumption_equiv_floor * jnp.asarray(2.0) ** exponent
    return jnp.stack(
        [consumption_equiv_floor, married_dollar_floor, married_dollar_floor]
    )


def compute_consumption_dollars_points(
    *,
    consumption_equiv_floor: Array,
    exponent: Array,
    max_consumption_dollars: Array,
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
    `max_consumption_dollars` so the two pinned points stay strictly
    increasing.
    """
    married_dollar_floor = consumption_equiv_floor * jnp.asarray(2.0) ** exponent
    tail = jnp.geomspace(
        married_dollar_floor, max_consumption_dollars, num=n_points - 1
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
            f"pts[2]={float(pts[2]):.6g}. Either `max_consumption_dollars` "
            f"is too close to the married floor or `n_points` is too small."
        )
        raise ValueError(msg)
    return pts
