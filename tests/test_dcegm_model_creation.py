"""DC-EGM solver wiring: `create_model(..., solver="dcegm")` builds the
post-decision spec — `DCEGM` config on every living regime, savings-form
assets laws, the solver-contract functions broadcast, no borrowing
constraint — and the model-level acceptance status against pylcm's DC-EGM
contract is pinned explicitly.
"""

from collections.abc import Mapping
from typing import cast

import pytest
from helpers.model import _DERIVED_CATEGORICALS  # ty: ignore[unresolved-import]
from lcm import DiscreteGrid, IrregSpacedGrid, Model, Regime
from lcm.solvers import DCEGM

from aca_model.agent import assets_and_income
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline.model import create_model
from aca_model.baseline.regimes import (
    REGIME_SPECS,
    SolverName,
    build_all_regimes,
    build_model_slots,
)
from aca_model.benchmark import get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG

_FIXED_PARAMS, _WAGE_PARAMS, _ = get_benchmark_params(model=None)


def _build_regimes(solver: SolverName) -> dict[str, Regime]:
    return build_all_regimes(
        grid_config=BENCHMARK_GRID_CONFIG,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver=solver,
    )


def _build_model(solver: SolverName) -> Model:
    return create_model(
        n_subjects=1,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        derived_categoricals=_DERIVED_CATEGORICALS,
        grid_config=BENCHMARK_GRID_CONFIG,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver=solver,
    )


def test_every_living_regime_gets_the_dcegm_solver() -> None:
    """`solver="dcegm"` attaches a `DCEGM` config with assets as the Euler
    state to every living regime; the terminal regime keeps the default."""
    regimes = _build_regimes("dcegm")
    for name in REGIME_SPECS:
        solver = regimes[name].solver
        assert isinstance(solver, DCEGM), name
        assert solver.continuous_state == "assets"
        assert solver.continuous_action == "consumption_dollars"
    assert not isinstance(regimes["dead"].solver, DCEGM)


def test_dcegm_assets_laws_take_the_savings_form() -> None:
    """Under DC-EGM every per-target assets law consumes the post-decision
    state instead of cash-on-hand and consumption directly."""
    regimes = _build_regimes("dcegm")
    for name in REGIME_SPECS:
        assets_laws = cast(
            "Mapping[str, object]", regimes[name].state_transitions["assets"]
        )
        for target_name, law in assets_laws.items():
            expected = (
                assets_and_income.next_assets_when_dead_from_savings
                if target_name == "dead"
                else assets_and_income.next_assets_from_savings
            )
            assert law is expected, (name, target_name)


def test_dcegm_model_slots_swap_constraint_for_contract_functions() -> None:
    """The DC-EGM slots broadcast `resources`/`savings`/
    `inverse_marginal_utility` and declare no borrowing constraint; the
    savings grid's lower bound is the zero borrowing limit."""
    slots = build_model_slots(
        grid_config=BENCHMARK_GRID_CONFIG,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver="dcegm",
    )
    assert slots["constraints"] == {}
    for name in ("resources", "savings", "inverse_marginal_utility"):
        assert name in slots["functions"]
    solver = _build_regimes("dcegm")["retiree_nomc_inelig_canwork"].solver
    assert isinstance(solver, DCEGM)
    assert isinstance(solver.savings_grid, IrregSpacedGrid)
    savings_points = solver.savings_grid.points
    assert savings_points is not None
    assert savings_points[0] == 0.0


def test_dead_masks_cover_the_dcegm_contract_functions() -> None:
    """`dead` masks the broadcast solver-contract functions so their
    unresolvable inputs don't surface as params in the dead template."""
    regimes = _build_regimes("dcegm")
    for name in ("resources", "savings", "inverse_marginal_utility"):
        assert regimes["dead"].functions[name] is None


@pytest.mark.xfail(
    strict=False,
    reason=(
        "pylcm's DC-EGM contract does not yet admit the ACA budget: the "
        "assets law reaches `assets` outside the post-decision function — "
        "through `oop_costs` (Medicaid eligibility → `countable_income` → "
        "`capital_income`) and `pension_assets_adjustment` "
        "(`marginal_tax_rate` → `gross_income` → `capital_income`). "
        "Fixes land upstream in pylcm, not here."
    ),
)
def test_dcegm_benchmark_model_builds() -> None:
    """The benchmark model accepts `solver="dcegm"` end to end.

    The acceptance criterion for the upstream DC-EGM stack: once pylcm's
    contract admits the ACA budget chains, this builds without error.
    """
    model = _build_model("dcegm")
    assert isinstance(model.user_regimes["retiree_nomc_inelig_canwork"].solver, DCEGM)
