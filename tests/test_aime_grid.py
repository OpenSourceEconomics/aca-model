"""Tests for the AIME piecewise grid builder."""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np

from aca_model.baseline.regimes._common import (
    _AIME_PIECE_N_POINTS,
    _build_aime_grid,
)
from aca_model.config import BENCHMARK_GRID_CONFIG

# Production SSA bend points plus the delayed-retirement-credit extension:
# 0, kink_0, kink_1, taxable-max, and the extension point that carries the
# largest delayed credit (1.32 * max_pia round-tripped to AIME).
_PIA_AIME_GRID = jnp.asarray([0.0, 9792.0, 59004.0, 117000.0, 187954.752])
_FIXED_PARAMS = MappingProxyType({"pia_aime_grid": _PIA_AIME_GRID})


def test_build_aime_grid_has_four_segments() -> None:
    """The AIME grid spans all four PIA intervals, including the extension."""
    grid = _build_aime_grid(
        grid_config=BENCHMARK_GRID_CONFIG, fixed_params=_FIXED_PARAMS
    )
    assert len(grid.segments) == 4


def test_build_aime_grid_top_point_is_extension_aime() -> None:
    """The grid reaches the delayed-credit extension AIME at its top."""
    grid = _build_aime_grid(
        grid_config=BENCHMARK_GRID_CONFIG, fixed_params=_FIXED_PARAMS
    )
    np.testing.assert_allclose(float(grid.to_jax().max()), 187954.752, rtol=1e-5)


def test_aime_piece_n_points_has_four_entries() -> None:
    """One point-count per segment, including the sparse extension region."""
    assert len(_AIME_PIECE_N_POINTS) == 4
