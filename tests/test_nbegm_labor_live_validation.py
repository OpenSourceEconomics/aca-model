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

_M1_REGIME = "nongroup_nomc_inelig_canwork"


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


def _solve_m1(solver: SolverName) -> dict[int, np.ndarray]:
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
    return {
        period: np.asarray(regimes[_M1_REGIME])
        for period, regimes in solution.items()
        if _M1_REGIME in regimes
    }


@pytest.mark.long_running
@pytest.mark.usefixtures("m1_labor_live")
def test_nbegm_m1_labor_live_agrees_with_brute_in_the_bulk() -> None:
    """The M1 value functions agree cell-wise away from the cliff tail with labor live.

    Finite masks must be identical and the median relative difference must sit at
    interpolation-error order; a branch-compiler defect (wrong per-branch continuation,
    utility, or envelope over the 5 labor levels) moves the bulk by orders of magnitude.
    The tail gets only a sanity ceiling at this coarse grid.
    """
    bq = _solve_m1("nbegm")
    brute = _solve_m1("brute_force")

    assert bq.keys() == brute.keys()
    rel_diffs = []
    for period in bq:
        finite_bq = np.isfinite(bq[period])
        finite_brute = np.isfinite(brute[period])
        np.testing.assert_array_equal(finite_bq, finite_brute)
        denominator = np.maximum(np.abs(brute[period][finite_brute]), 1.0)
        rel_diffs.append(
            np.abs(bq[period][finite_bq] - brute[period][finite_brute]) / denominator
        )
    pooled = np.concatenate(rel_diffs)
    quantiles = np.quantile(pooled, [0.5, 0.9, 0.99])
    assert quantiles[0] < 1e-3, quantiles
    assert quantiles[2] < 2.0, quantiles
