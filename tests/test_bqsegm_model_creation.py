"""BQSEGM solver wiring: `solver="bqsegm"` is a per-regime option.

Unlike DC-EGM (a global Euler solver on every living regime), BQSEGM solves a
single 1-D consumption/savings regime with at most one discrete action, so it
attaches only to the M1 vertical-slice regime `nongroup_nomc_inelig_canwork`;
every other living regime keeps brute force. The savings-form spec is shared
with DC-EGM (BQSEGM's budget node is `resources`, the post-decision function is
`savings`).
"""

import dataclasses
from collections.abc import Mapping
from typing import cast

from helpers.model import _DERIVED_CATEGORICALS  # ty: ignore[unresolved-import]
from lcm import DiscreteGrid, Model, Regime
from lcm.solvers import BQSEGM, GridSearch

from aca_model.agent import assets_and_income
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline import health_insurance
from aca_model.baseline.model import create_model
from aca_model.baseline.regimes import (
    REGIME_SPECS,
    SolverName,
    build_all_regimes,
)
from aca_model.baseline.regimes._bqsegm import build_bqsegm_solver
from aca_model.baseline.regimes._common import Grids, build_grids
from aca_model.benchmark import get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG

_FIXED_PARAMS, _WAGE_PARAMS, _ = get_benchmark_params(model=None)

_M1_REGIME = "nongroup_nomc_inelig_canwork"
_BRUTE_REGIME = "retiree_nomc_inelig_canwork"


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


def _grids() -> Grids:
    return build_grids(
        grid_config=BENCHMARK_GRID_CONFIG,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )


def test_bqsegm_attaches_only_to_the_m1_regime() -> None:
    """`solver="bqsegm"` gives the M1 slice regime a `BQSEGM` config and leaves
    every other living regime on brute force."""
    regimes = _build_regimes("bqsegm")
    assert isinstance(regimes[_M1_REGIME].solver, BQSEGM)
    for name in REGIME_SPECS:
        if name == _M1_REGIME:
            continue
        assert isinstance(regimes[name].solver, GridSearch), name


def test_build_bqsegm_solver_uses_the_savings_form_resources_budget() -> None:
    """The BQSEGM config inverts against `resources` in post-decision savings
    form, matching the DC-EGM contract the regime is rewired into."""
    solver = build_bqsegm_solver(_grids())
    assert isinstance(solver, BQSEGM)
    assert solver.budget_target == "resources"
    assert solver.post_decision_function == "savings"


def test_build_bqsegm_solver_names_assets_as_the_euler_axis() -> None:
    """`assets` is the liquid (Euler) axis; `aime` and the stochastic shock grids
    ride along, so the solver names the Euler axis explicitly."""
    solver = build_bqsegm_solver(_grids())
    assert solver.continuous_state == "assets"


def test_build_bqsegm_solver_forwards_the_jump_read_mode() -> None:
    """`GridConfig.bqsegm_jump_read` selects the solver's cliff-read mode.

    `"bridged"` is the fast estimation setting (plain carry rows, fold kept);
    `"one_sided"` (the default) publishes exact one-sided cliff limits.
    """
    grid_config = dataclasses.replace(BENCHMARK_GRID_CONFIG, bqsegm_jump_read="bridged")
    grids = build_grids(
        grid_config=grid_config,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )
    solver = build_bqsegm_solver(grids)
    assert solver.jump_read == "bridged"


def test_bqsegm_m1_regime_fixes_buy_private() -> None:
    """The BQSEGM M1 slice drops `buy_private` as an action (fixed to purchase),
    so the only choice is continuous consumption; the brute M1 regime keeps it."""
    bqsegm_m1 = _build_regimes("bqsegm")[_M1_REGIME]
    brute_m1 = _build_regimes("brute_force")[_M1_REGIME]
    assert "buy_private" not in bqsegm_m1.actions
    assert "buy_private" in brute_m1.actions


def test_bqsegm_m1_regime_fixes_labor_supply() -> None:
    """The BQSEGM M1 slice drops `labor_supply` as an action (fixed to full-time
    work), so no discrete action remains and the only choice is continuous
    consumption; the brute M1 regime keeps `labor_supply`."""
    bqsegm_m1 = _build_regimes("bqsegm")[_M1_REGIME]
    brute_m1 = _build_regimes("brute_force")[_M1_REGIME]
    assert "labor_supply" not in bqsegm_m1.actions
    assert "labor_supply" in brute_m1.actions


def test_bqsegm_m1_regime_has_no_discrete_action() -> None:
    """With both discrete actions fixed, the BQSEGM M1 slice leaves only the
    continuous consumption choice — no `DiscreteGrid` action remains."""
    bqsegm_m1 = _build_regimes("bqsegm")[_M1_REGIME]
    assert not any(
        isinstance(grid, DiscreteGrid) for grid in bqsegm_m1.actions.values()
    )


def test_bqsegm_m1_regime_takes_the_savings_form_assets_laws() -> None:
    """The M1 regime under BQSEGM consumes the post-decision assets laws, like
    DC-EGM; the other (brute) regimes keep the cash-on-hand form."""
    regimes = _build_regimes("bqsegm")
    assets_laws = cast(
        "Mapping[str, object]", regimes[_M1_REGIME].state_transitions["assets"]
    )
    for target_name, law in assets_laws.items():
        expected = (
            assets_and_income.next_assets_when_dead_from_savings
            if target_name == "dead"
            else assets_and_income.next_assets_from_savings
        )
        assert law is expected, target_name


def test_bqsegm_savings_form_functions_are_scoped_to_the_m1_regime() -> None:
    """Under BQSEGM only the M1 regime carries the savings-form budget functions
    (`resources`, `savings`); brute regimes keep the cash-on-hand form and carry
    neither."""
    model = _build_model("bqsegm")
    m1_functions = model.user_regimes[_M1_REGIME].functions
    assert "resources" in m1_functions
    assert "savings" in m1_functions
    assert "resources" not in model.user_regimes[_BRUTE_REGIME].functions


def test_bqsegm_m1_regime_does_not_carry_inverse_marginal_utility() -> None:
    """BQSEGM inverts the Euler equation internally, so the M1 regime never
    carries the DC-EGM `inverse_marginal_utility` function (whose
    solver-supplied `marginal_continuation` would otherwise be a required
    parameter)."""
    model = _build_model("bqsegm")
    assert "inverse_marginal_utility" not in model.user_regimes[_M1_REGIME].functions


def test_bqsegm_brute_regimes_keep_the_borrowing_constraint() -> None:
    """BQSEGM enforces the borrowing limit through its savings grid's lower bound,
    so the M1 regime drops the explicit constraint; every brute regime keeps it."""
    model = _build_model("bqsegm")
    assert "borrowing_constraint" not in model.user_regimes[_M1_REGIME].constraints
    assert "borrowing_constraint" in model.user_regimes[_BRUTE_REGIME].constraints


def test_resources_declares_the_consumption_floor_kink() -> None:
    """`resources = max(cash_on_hand, floor)` is a declared piecewise-affine
    schedule on `cash_on_hand`, so BQSEGM's partition splits each cell at the
    household's floor and the flat sub-interval is solved as such rather than
    extrapolated."""
    meta = assets_and_income.resources.__lcm_piecewise_affine__  # ty: ignore[unresolved-attribute]
    assert meta.output == "resources"
    assert meta.variable == "cash_on_hand"
    (floor_breakpoint,) = meta.breakpoints
    assert floor_breakpoint.threshold == "consumption_floor_schedule"
    assert floor_breakpoint.kind == "continuous_kink"
    assert floor_breakpoint.indexed_by == "spousal_income"


def test_is_ssi_eligible_declares_the_asset_test_jump() -> None:
    """The SSI asset test is a declared jump on `assets`: cash-on-hand drops by
    the SSI benefit where eligibility ends, so BQSEGM's partition must split
    there instead of extrapolating one affine budget across the cliff."""
    meta = health_insurance.is_ssi_eligible.__lcm_piecewise_affine__  # ty: ignore[unresolved-attribute]
    assert meta.variable == "assets"
    (asset_test,) = meta.breakpoints
    assert asset_test.threshold == "ssi_assets_test"
    assert asset_test.kind == "jump"
    assert asset_test.indexed_by == "spousal_income"


def test_ssi_benefit_declares_the_income_test_kink() -> None:
    """The SSI income test is a declared continuous kink on `countable_income`:
    the benefit reaches zero exactly at the test, so the budget's slope changes
    (capital income stops being offset) without a jump."""
    meta = health_insurance.ssi_benefit.__lcm_piecewise_affine__  # ty: ignore[unresolved-attribute]
    assert meta.variable == "countable_income"
    (income_test,) = meta.breakpoints
    assert income_test.threshold == "ssi_maximum_benefit"
    assert income_test.kind == "continuous_kink"
    assert income_test.indexed_by == "spousal_income"
