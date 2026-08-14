"""NBEGM solves the M1 regime with a live labor-supply choice, matching brute.

The M1 regime `nongroup_nomc_inelig_canwork` carries `labor_supply` (5 levels) as a
genuine discrete action while `buy_private` is fixed. Labor supply feeds the `aime`
co-state (earnings accrual), the `lagged_labor_supply` co-state, and the leisure term
in period utility — every branch-dependent channel at once. NBEGM's ride-along
discrete envelope solves each labor branch against its own continuation and utility;
the value function must match a brute-force solve on the same state-action space.

Both solvers run the full 18-regime model at the benchmark grid with `buy_private`
fixed and `labor_supply` live, in NBEGM's `"bridged"` cliff-read mode so the
comparison isolates the solver machinery from the asset-grid read convention.
"""

import dataclasses

import numpy as np
import pytest
from helpers.model import _DERIVED_CATEGORICALS  # ty: ignore[unresolved-import]
from lcm import DiscreteGrid

import aca_model.baseline.regimes._nongroup as nongroup_mod
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline.model import create_model
from aca_model.baseline.regimes import SolverName
from aca_model.baseline.regimes._common import Grids, RegimeSpec
from aca_model.benchmark import get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG

_M1_REGIME = "single_nongroup_nomc_inelig_canwork"


def _is_m1(spec: RegimeSpec) -> bool:
    return spec["ss"] == "inelig" and spec["mc"] == "nomc"


@pytest.fixture
def m1_labor_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `labor_supply` live and fix only `buy_private` on M1, under every solver.

    The NBEGM wiring fixes both discrete actions on its own; this override keeps
    labor supply as a genuine action (so the branch compiler solves it) and fixes
    `buy_private`, applying the same choice to the brute build so both solvers share
    one state-action space.
    """
    original_build_functions = nongroup_mod._build_functions  # noqa: SLF001
    original_build_actions = nongroup_mod.build_actions

    def build_functions_labor_live(spec: RegimeSpec, **kwargs: bool) -> dict:
        if _is_m1(spec):
            return original_build_functions(
                spec, fix_buy_private=True, fix_labor_supply=False
            )
        return original_build_functions(spec, **kwargs)

    def build_actions_labor_live(
        spec: RegimeSpec, grids: Grids, **kwargs: bool
    ) -> dict:
        if _is_m1(spec):
            return original_build_actions(
                spec, grids, drop_buy_private=True, drop_labor_supply=False
            )
        return original_build_actions(spec, grids, **kwargs)

    monkeypatch.setattr(nongroup_mod, "_build_functions", build_functions_labor_live)
    monkeypatch.setattr(nongroup_mod, "build_actions", build_actions_labor_live)


def _solve_m1(solver: SolverName) -> tuple[dict[int, np.ndarray], int]:
    # The CPU XLA backend does not fuse the ride-cell fan-out and materialises the
    # whole flattened ride mesh at once, so a full-model solve needs hundreds of GiB on
    # host even at a tiny asset grid. `n_nbegm_cell_block_size` streams the mesh in
    # blocks (identical result) to bound the peak to the GPU's few-GiB footprint; the
    # live-labor branch axis makes this essential on CPU. A coarser savings grid keeps
    # the check quick. On GPU the whole-mesh vmap stays small, so production sets 0.
    grid_config = dataclasses.replace(
        BENCHMARK_GRID_CONFIG,
        nbegm_jump_read="bridged",
        n_nbegm_cell_block_size=32,
        n_savings_gridpoints=50,
    )
    fixed_params, wage_params, _ = get_benchmark_params(model=None)
    model = create_model(
        n_subjects=1,
        fixed_params=fixed_params,
        wage_params=wage_params,
        derived_categoricals=_DERIVED_CATEGORICALS,
        grid_config=grid_config,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver=solver,
    )
    _, _, params = get_benchmark_params(model=model)
    solution = model.solve(params=params, log_level="off")
    assets_axis = tuple(
        model._regimes[_M1_REGIME].solution.state_names  # noqa: SLF001
    ).index("assets")
    return {
        period: np.asarray(regimes[_M1_REGIME])
        for period, regimes in solution.items()
        if _M1_REGIME in regimes
    }, assets_axis


def _cliff_band_mask(reference: np.ndarray, assets_axis: int) -> np.ndarray:
    """Cells adjacent to a value discontinuity in the reference solve, along assets.

    A cliff is an increment between neighboring asset cells that dwarfs the
    row's typical smooth increment: a gap counts when it exceeds ten times the
    row's median finite gap (plus an absolute floor against all-flat rows).
    Both neighbors of every such gap belong to the band. Scaling by the row's
    own increments keeps the detector scale-free — value levels near zero do
    not inflate it the way a relative-jump criterion would. NaN/inf cells never
    enter the comparison, so they are excluded here.
    """
    moved = np.moveaxis(reference, assets_axis, -1)
    finite = np.isfinite(moved)
    gaps = np.abs(np.diff(moved, axis=-1))
    typical = np.nanmedian(
        np.where(np.isfinite(gaps), gaps, np.nan), axis=-1, keepdims=True
    )
    threshold = 10.0 * np.nan_to_num(typical, nan=np.inf) + 1e-8
    is_cliff_gap = np.nan_to_num(gaps, nan=0.0, posinf=np.inf) > threshold
    band = np.zeros(moved.shape, dtype=bool)
    band[..., :-1] |= is_cliff_gap
    band[..., 1:] |= is_cliff_gap
    return np.moveaxis(band & finite, -1, assets_axis)


@pytest.mark.long_running
@pytest.mark.usefixtures("m1_labor_live")
def test_nbegm_m1_labor_live_agrees_with_brute_split_by_cliff_band() -> None:
    """The M1 value functions agree cell-wise with labor live, gated per region.

    Finite masks must be identical. Outside the cliff band — the asset cells
    adjacent to a large relative jump in the brute reference — disagreement must
    sit at interpolation-error order in the p99 *and* the maximum, because
    nothing there is allowed to bridge a discontinuity. Inside the band the two
    solvers read the cliff under different conventions, so the gate only bounds
    the band's share of cells and its maximum against the value scale; the
    band's signed error is a measured quantity, not a directional guarantee.
    """
    bq, assets_axis = _solve_m1("nbegm")
    brute, _ = _solve_m1("brute_force")

    assert bq.keys() == brute.keys()
    bulk_diffs = []
    band_diffs = []
    band_cells = 0
    finite_cells = 0
    for period in bq:
        finite_bq = np.isfinite(bq[period])
        finite_brute = np.isfinite(brute[period])
        np.testing.assert_array_equal(finite_bq, finite_brute)
        band = _cliff_band_mask(brute[period], assets_axis)
        rel = np.abs(bq[period] - brute[period]) / np.maximum(
            np.abs(brute[period]), 1.0
        )
        bulk_diffs.append(rel[finite_brute & ~band])
        band_diffs.append(rel[finite_brute & band])
        band_cells += int(np.sum(band))
        finite_cells += int(np.sum(finite_brute))
    bulk = np.concatenate(bulk_diffs)
    band = np.concatenate(band_diffs) if band_cells else np.zeros(1)

    stats = {
        "bulk_median": float(np.quantile(bulk, 0.5)),
        "bulk_p99": float(np.quantile(bulk, 0.99)),
        "bulk_max": float(np.max(bulk)),
        "band_share": band_cells / finite_cells,
        "band_median": float(np.quantile(band, 0.5)),
        "band_p99": float(np.quantile(band, 0.99)),
        "band_max": float(np.max(band)),
    }
    print(f"split-gate stats: {stats}")  # noqa: T201  (calibration record in -s runs)

    assert stats["bulk_median"] < 1e-3, stats
    assert stats["bulk_p99"] < 5e-2, stats
    assert stats["bulk_max"] < 0.5, stats
    # The band is a thin set of cliff-adjacent cells, not a broad region.
    assert stats["band_share"] < 0.2, stats
    assert stats["band_max"] < 2.0, stats
