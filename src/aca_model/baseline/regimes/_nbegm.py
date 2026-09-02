"""NBEGM solver configuration for the ACA M1 vertical-slice regime.

NBEGM is the case-piece endogenous-grid solver for a single 1-D
consumption/savings regime whose budget is split by institutional breakpoints
on a derived monotone income quantity. It shares DC-EGM's post-decision
(savings) spec — consumption is recovered from `resources = max(cash_on_hand,
floor)`, the assets laws are in savings form, the borrowing constraint is the
savings grid's lower bound — but solves only one regime with at most one
discrete action, so it attaches per regime rather than globally. The
function-level rewiring is shared with DC-EGM (`build_dcegm_functions`, the
savings-form assets laws in `_common`); this module holds only the solver
config.
"""

import dataclasses

from lcm import IrregSpacedGrid
from lcm.solvers import NBEGM

from aca_model.baseline.regimes._common import Grids


def build_nbegm_solver(grids: Grids) -> NBEGM:
    """Build the per-regime NBEGM configuration.

    The savings grid mirrors DC-EGM's: lower bound 0 (the borrowing constraint
    in post-decision form), upper bound the assets span, cubically clustered
    toward the constraint. Which DAG nodes play the liquid roles is the
    regime's declaration (`ACA_LIQUID_MARGIN`), not the solver's; this config
    carries numerical settings only.
    """
    n_points = grids.grid_config.n_savings_gridpoints
    _fail_if_too_few_savings_gridpoints(n_points)
    assets_points = grids.assets.to_jax()
    savings_stop = float(assets_points[-1]) - float(assets_points[0])
    savings_grid = IrregSpacedGrid(
        points=tuple(savings_stop * (i / (n_points - 1)) ** 3 for i in range(n_points)),
        batch_size=grids.grid_config.n_savings_batch_size,
    )
    solver = NBEGM(
        savings_grid=savings_grid,
        # Splay the child stochastic-node expectation per the grid config: `0` (the
        # default) reads the whole node mesh in one pass on a memory-rich device; a
        # positive value loops it in blocks to fit a tighter budget (a CPU run).
        stochastic_node_batch_size=grids.grid_config.n_nbegm_stochastic_node_batch_size,
        # Stream the per-interval upper envelope over candidate-segment blocks per
        # the grid config; `0` keeps the one-shot dense envelope.
        envelope_segment_block_size=(
            grids.grid_config.n_nbegm_envelope_segment_block_size
        ),
        # Which arithmetic decides envelope ownership per the grid config;
        # "certified" is exact and can abstain, "ordinary" reads in the working
        # format at a fraction of the cost.
        envelope_arithmetic=grids.grid_config.nbegm_envelope_arithmetic,
        # Stream continuation intervals, or let the active byte planner choose
        # when the configured width is zero.
        interval_batch_size=grids.grid_config.n_nbegm_interval_batch_size,
        # Stream both ride-along cores over ride-cell blocks per the grid config;
        # `0` vmaps the whole flattened mesh at once.
        cell_block_size=grids.grid_config.n_nbegm_cell_block_size,
        # Stream the discrete-action branch axis in blocks per the grid config;
        # `0` runs the whole axis in one vectorized pass.
        branch_batch_size=grids.grid_config.n_nbegm_branch_batch_size,
        # Cliff-read mode: exact one-sided limits (default) or the fast bridged
        # read for inner estimation loops (see `GridConfig.nbegm_jump_read`).
        jump_read=grids.grid_config.nbegm_jump_read,
    )
    budget = grids.grid_config.n_nbegm_max_device_workspace_bytes
    if budget is None:
        return solver

    fields = {field.name for field in dataclasses.fields(NBEGM)}
    if "max_device_workspace_bytes" not in fields:
        msg = (
            "GridConfig.n_nbegm_max_device_workspace_bytes requires a pylcm "
            "build with the experimental NB-EGM workspace planner."
        )
        raise RuntimeError(msg)
    return dataclasses.replace(solver, max_device_workspace_bytes=budget)


def _fail_if_too_few_savings_gridpoints(n_savings_gridpoints: int) -> None:
    if n_savings_gridpoints < 2:
        msg = (
            f"n_savings_gridpoints must be >= 2 to form the cubically clustered "
            f"NBEGM savings grid, got {n_savings_gridpoints}."
        )
        raise ValueError(msg)
