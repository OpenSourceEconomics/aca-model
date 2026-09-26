"""NB-EGM numerical configuration for the structural retirement model.

Savings gridpoints, cliff reads, and envelope arithmetic belong to the solver.
Device placement and compiled program widths belong to the model execution policy.
"""

from lcm import IrregSpacedGrid
from lcm.solvers import NBEGM

from aca_model.baseline.regimes._common import Grids


def build_nbegm_solver(grids: Grids) -> NBEGM:
    """Build NB-EGM with the model's savings grid and numerical read rules."""
    n_points = grids.grid_config.n_savings_gridpoints
    _fail_if_too_few_savings_gridpoints(n_points)
    assets_points = grids.assets.to_jax()
    savings_stop = float(assets_points[-1]) - float(assets_points[0])
    savings_grid = IrregSpacedGrid(
        points=tuple(savings_stop * (i / (n_points - 1)) ** 3 for i in range(n_points)),
    )
    return NBEGM(
        savings_grid=savings_grid,
        envelope_arithmetic=grids.grid_config.nbegm_envelope_arithmetic,
        jump_read=grids.grid_config.nbegm_jump_read,
    )


def _fail_if_too_few_savings_gridpoints(n_savings_gridpoints: int) -> None:
    if n_savings_gridpoints < 2:
        msg = (
            f"n_savings_gridpoints must be >= 2 to form the cubically clustered "
            f"NBEGM savings grid, got {n_savings_gridpoints}."
        )
        raise ValueError(msg)
