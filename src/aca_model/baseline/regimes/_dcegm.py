"""DC-EGM solver configuration for the ACA regimes.

DC-EGM changes the model spec, not just a flag: consumption is recovered from
post-transfer resources (`resources = max(cash_on_hand, floor)`), the assets
laws are rewritten in post-decision (savings) form, the borrowing constraint
is dropped (the savings grid's lower bound enforces it), and
`inverse_marginal_utility` joins the regime functions. The function-level
rewiring lives in `_common` (`build_dcegm_functions`, the savings-form assets
laws) and is algebraically identical to the brute-force spec (locked by
`tests/test_dcegm_functions.py`); this module holds only the solver config,
the one piece that needs `lcm.solvers`.
"""

from lcm import IrregSpacedGrid
from lcm.solvers import DCEGM

from aca_model.baseline.regimes._common import Grids


def build_dcegm_solver(grids: Grids) -> DCEGM:
    """Build the per-regime DC-EGM configuration.

    The savings grid's lower bound is 0 — `consumption <= resources` is the
    borrowing constraint in post-decision form. Its upper bound is the assets
    ceiling plus the labor-income headroom already encoded in the assets
    grid's (negative) floor. Nodes are cubically clustered toward the
    constraint, where the value function curves hardest.
    """
    n_points = grids.grid_config.n_savings_gridpoints
    _fail_if_too_few_savings_gridpoints(n_points)
    assets_points = grids.assets.to_jax()
    savings_stop = float(assets_points[-1]) - float(assets_points[0])
    savings_grid = IrregSpacedGrid(
        points=tuple(savings_stop * (i / (n_points - 1)) ** 3 for i in range(n_points)),
        batch_size=grids.grid_config.n_savings_batch_size,
    )
    return DCEGM(
        continuous_state="assets",
        continuous_action="consumption_dollars",
        resources="resources",
        post_decision_function="savings",
        savings_grid=savings_grid,
        stochastic_node_batch_size=grids.grid_config.n_stochastic_node_batch_size,
    )


def _fail_if_too_few_savings_gridpoints(n_savings_gridpoints: int) -> None:
    if n_savings_gridpoints < 2:
        msg = (
            f"n_savings_gridpoints must be >= 2 to form the cubically clustered "
            f"DC-EGM savings grid, got {n_savings_gridpoints}."
        )
        raise ValueError(msg)
