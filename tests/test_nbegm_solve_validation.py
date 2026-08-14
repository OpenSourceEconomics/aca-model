"""NBEGM vs brute-force agreement on the M1 slice value function.

Both solvers run the full 18-regime model at the benchmark grid on the same
state-action space — the M1 regime declares its `labor_supply` and `buy_private`
choices under either solver — so the value functions differ only through the
continuous-consumption solver.

NBEGM runs its `"bridged"` cliff-read mode so both solvers share the same
read convention (a finite brute reads child values by linear interpolation
across value cliffs on its asset grid): the comparison then isolates the
solver machinery — EGM inversion, candidate set, envelope — rather than the
convention difference, which grows with asset-grid coarseness and is
adjudicated separately at production grids. The comparison is quantile-based;
the tail carries the residual discretization mismatch between a coarse
consumption grid and a continuous solver.
"""

import dataclasses

import numpy as np
import pytest
from helpers.model import _DERIVED_CATEGORICALS  # ty: ignore[unresolved-import]
from lcm import DiscreteGrid

from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline.model import create_model
from aca_model.baseline.regimes import SolverName
from aca_model.benchmark import get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG

_M1_REGIME = "nongroup_nomc_inelig_canwork"

_MULTIPLE_DISCRETE_ACTIONS = (
    "pylcm's ride-along discrete envelope is written over one action's grid and "
    "refuses a regime declaring several; the M1 regime declares both "
    "`labor_supply` and `buy_private`."
)


def _solve_m1(solver: SolverName) -> dict[int, np.ndarray]:
    grid_config = dataclasses.replace(BENCHMARK_GRID_CONFIG, nbegm_jump_read="bridged")
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
@pytest.mark.xfail(strict=True, reason=_MULTIPLE_DISCRETE_ACTIONS)
def test_nbegm_m1_value_function_agrees_with_brute_in_the_bulk() -> None:
    """The M1 value functions agree cell-wise away from the cliff tail.

    Finite masks must be identical and the median relative difference must sit
    at interpolation-error order — any solver-machinery defect (wrong Euler
    inversion, envelope, case masking, feasibility) moves the bulk by orders
    of magnitude. The tail only gets a sanity ceiling: at this grid's three
    asset points, cliff-adjacent cells differ at order one relative under any
    read convention, so tight tail agreement is certified at production grids
    on GPU, not here.
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
