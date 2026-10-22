"""Tests for the asset state grid."""

import numpy as np
import pytest

from aca_model.baseline.regimes._common import build_assets_grid

FLOOR = -221_270.85


def _points(n_points: int = 24) -> np.ndarray:
    return np.asarray(build_assets_grid(floor=FLOOR, n_points=n_points).to_jax())


def test_assets_grid_top_covers_largest_initial_assets() -> None:
    """The top node is 12 million, the largest initial asset holding."""
    assert _points()[-1] == 12_000_000.0


def test_assets_grid_bottom_is_the_borrowing_limit() -> None:
    """The bottom node is the borrowing limit passed in."""
    assert _points()[0] == FLOOR


def test_assets_grid_has_a_node_at_zero() -> None:
    """Zero assets is a node: the asset tests and the floor bite next to it."""
    assert np.any(_points() == 0.0)


def test_assets_grid_is_dense_near_zero() -> None:
    """At least five nodes lie in [0, 30k], as in struct-ret's grid."""
    points = _points()
    assert int(((points >= 0.0) & (points <= 30_000.0)).sum()) >= 5


@pytest.mark.parametrize("n_points", [3, 8, 24])
def test_assets_grid_has_requested_number_of_points(n_points: int) -> None:
    """The node count is the configured `n_assets_gridpoints`."""
    assert len(_points(n_points)) == n_points


@pytest.mark.parametrize(
    ("index", "expected"),
    [(5, 0.0), (6, 4_460.02), (9, 27_305.37), (15, 377_388.77), (19, 2_128_241.24)],
)
def test_assets_grid_nodes_are_sinh_spaced(index: int, expected: float) -> None:
    """Positive nodes are `10k * sinh(u)` with `u` evenly spaced from 0 to
    `asinh(12M / 10k)` over 19 nodes."""
    np.testing.assert_allclose(_points()[index], expected, rtol=1e-6)
