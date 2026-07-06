"""NBEGM solver wiring: `solver="nbegm"` is a per-regime option.

Unlike DC-EGM (a global Euler solver on every living regime), NBEGM solves a
single 1-D consumption/savings regime with at most one discrete action, so it
attaches only to the M1 vertical-slice regime `nongroup_nomc_inelig_canwork`;
every other living regime keeps brute force. The savings-form spec is shared
with DC-EGM (NBEGM's budget node is `resources`, the post-decision function is
`savings`).
"""

import dataclasses
from collections.abc import Mapping
from typing import cast

import pytest
from helpers.model import _DERIVED_CATEGORICALS  # ty: ignore[unresolved-import]
from lcm import DiscreteGrid, Model, Regime
from lcm.solvers import NBEGM, GridSearch

from aca_model.aca import PolicyVariant
from aca_model.aca.model import create_model as create_aca_model
from aca_model.aca.regimes import build_all_regimes as build_all_aca_regimes
from aca_model.agent import assets_and_income
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline import health_insurance
from aca_model.baseline.model import create_model
from aca_model.baseline.regimes import (
    REGIME_SPECS,
    SolverName,
    build_all_regimes,
)
from aca_model.baseline.regimes._common import Grids, build_grids
from aca_model.baseline.regimes._nbegm import build_nbegm_solver
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


def test_nbegm_attaches_only_to_the_m1_regime() -> None:
    """`solver="nbegm"` gives the M1 slice regime a `NBEGM` config and leaves
    every other living regime on brute force."""
    regimes = _build_regimes("nbegm")
    assert isinstance(regimes[_M1_REGIME].solver, NBEGM)
    for name in REGIME_SPECS:
        if name == _M1_REGIME:
            continue
        assert isinstance(regimes[name].solver, GridSearch), name


def test_build_nbegm_solver_uses_the_savings_form_resources_budget() -> None:
    """The NBEGM config inverts against `resources` in post-decision savings
    form, matching the DC-EGM contract the regime is rewired into."""
    solver = build_nbegm_solver(_grids())
    assert isinstance(solver, NBEGM)
    assert solver.budget_target == "resources"
    assert solver.post_decision_function == "savings"


def test_build_nbegm_solver_names_assets_as_the_euler_axis() -> None:
    """`assets` is the liquid (Euler) axis; `aime` and the stochastic shock grids
    ride along, so the solver names the Euler axis explicitly."""
    solver = build_nbegm_solver(_grids())
    assert solver.continuous_state == "assets"


def test_build_nbegm_solver_forwards_the_jump_read_mode() -> None:
    """`GridConfig.nbegm_jump_read` selects the solver's cliff-read mode.

    `"bridged"` is the fast estimation setting (plain carry rows, fold kept);
    `"one_sided"` (the default) publishes exact one-sided cliff limits.
    """
    grid_config = dataclasses.replace(BENCHMARK_GRID_CONFIG, nbegm_jump_read="bridged")
    grids = build_grids(
        grid_config=grid_config,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )
    solver = build_nbegm_solver(grids)
    assert solver.jump_read == "bridged"


def test_nbegm_m1_regime_fixes_buy_private() -> None:
    """The NBEGM M1 slice drops `buy_private` as an action (fixed to purchase),
    so the only choice is continuous consumption; the brute M1 regime keeps it."""
    nbegm_m1 = _build_regimes("nbegm")[_M1_REGIME]
    brute_m1 = _build_regimes("brute_force")[_M1_REGIME]
    assert "buy_private" not in nbegm_m1.actions
    assert "buy_private" in brute_m1.actions


def test_nbegm_m1_regime_fixes_labor_supply() -> None:
    """The NBEGM M1 slice drops `labor_supply` as an action (fixed to full-time
    work), so no discrete action remains and the only choice is continuous
    consumption; the brute M1 regime keeps `labor_supply`."""
    nbegm_m1 = _build_regimes("nbegm")[_M1_REGIME]
    brute_m1 = _build_regimes("brute_force")[_M1_REGIME]
    assert "labor_supply" not in nbegm_m1.actions
    assert "labor_supply" in brute_m1.actions


def test_nbegm_m1_regime_has_no_discrete_action() -> None:
    """With both discrete actions fixed, the NBEGM M1 slice leaves only the
    continuous consumption choice — no `DiscreteGrid` action remains."""
    nbegm_m1 = _build_regimes("nbegm")[_M1_REGIME]
    assert not any(isinstance(grid, DiscreteGrid) for grid in nbegm_m1.actions.values())


def test_nbegm_m1_regime_takes_the_savings_form_assets_laws() -> None:
    """The M1 regime under NBEGM consumes the post-decision assets laws, like
    DC-EGM; the other (brute) regimes keep the cash-on-hand form."""
    regimes = _build_regimes("nbegm")
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


def test_nbegm_savings_form_functions_are_scoped_to_the_m1_regime() -> None:
    """Under NBEGM only the M1 regime carries the savings-form budget functions
    (`resources`, `savings`); brute regimes keep the cash-on-hand form and carry
    neither."""
    model = _build_model("nbegm")
    m1_functions = model.user_regimes[_M1_REGIME].functions
    assert "resources" in m1_functions
    assert "savings" in m1_functions
    assert "resources" not in model.user_regimes[_BRUTE_REGIME].functions


def test_nbegm_m1_regime_does_not_carry_inverse_marginal_utility() -> None:
    """NBEGM inverts the Euler equation internally, so the M1 regime never
    carries the DC-EGM `inverse_marginal_utility` function (whose
    solver-supplied `marginal_continuation` would otherwise be a required
    parameter)."""
    model = _build_model("nbegm")
    assert "inverse_marginal_utility" not in model.user_regimes[_M1_REGIME].functions


def test_nbegm_m1_regime_keeps_the_borrowing_constraint() -> None:
    """The M1 regime declares the borrowing constraint like every brute regime.

    The EGM solve enforces the borrowing limit through the savings grid's lower
    bound, but forward simulation re-decides consumption by an argmax over the
    consumption grid — without the explicit `consumption <= resources`
    feasibility mask, floor-region subjects (negative cash-on-hand rescued by
    the consumption floor) would pick the consumption grid's top value.
    """
    model = _build_model("nbegm")
    assert "borrowing_constraint" in model.user_regimes[_M1_REGIME].constraints
    assert "borrowing_constraint" in model.user_regimes[_BRUTE_REGIME].constraints


def test_resources_declares_the_consumption_floor_kink() -> None:
    """`resources = max(cash_on_hand, floor)` is a declared piecewise-affine
    schedule on `cash_on_hand`, so NBEGM's partition splits each cell at the
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
    the SSI benefit where eligibility ends, so NBEGM's partition must split
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


def test_nbegm_keeps_labor_supply_live_when_configured() -> None:
    """With `nbegm_live_labor_supply=True`, the M1 regime carries `labor_supply`
    as a genuine discrete action under NBEGM while `buy_private` stays fixed, so
    the branch compiler solves the labor choice against the cliffed budget."""
    grid_config = dataclasses.replace(
        BENCHMARK_GRID_CONFIG, nbegm_live_labor_supply=True
    )
    regimes = build_all_regimes(
        grid_config=grid_config,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver="nbegm",
    )
    actions = regimes[_M1_REGIME].actions
    assert "labor_supply" in actions
    assert "buy_private" not in actions


def test_nbegm_fixes_labor_supply_by_default() -> None:
    """By default NBEGM fixes both discrete actions on the M1 regime, so the
    only remaining choice is continuous consumption against the cliffed budget."""
    regimes = _build_regimes("nbegm")
    actions = regimes[_M1_REGIME].actions
    assert "labor_supply" not in actions
    assert "buy_private" not in actions


@pytest.mark.parametrize("policy", list(PolicyVariant))
def test_nbegm_builds_every_aca_policy_variant(policy: PolicyVariant) -> None:
    """Every ACA policy variant builds a model under NBEGM with the M1 regime on
    the solver and labor live — the overlay's function swaps compose with the
    branch compiler's per-regime wiring."""
    grid_config = dataclasses.replace(
        BENCHMARK_GRID_CONFIG, nbegm_live_labor_supply=True
    )
    model = create_aca_model(
        n_subjects=1,
        policy=policy,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        derived_categoricals=_DERIVED_CATEGORICALS,
        grid_config=grid_config,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver="nbegm",
    )
    assert isinstance(model, Model)
    regimes = build_all_aca_regimes(
        policy=policy,
        grid_config=grid_config,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver="nbegm",
    )
    assert isinstance(regimes[_M1_REGIME].solver, NBEGM)
    assert "labor_supply" in regimes[_M1_REGIME].actions


@pytest.mark.parametrize("policy", list(PolicyVariant))
def test_nbegm_aca_variants_leave_no_free_buy_private_params(
    policy: PolicyVariant,
) -> None:
    """With `buy_private` fixed under the NBEGM M1 slice, no ACA-swapped
    function may leave `buy_private` as a free parameter — the params template
    holds no `buy_private` leaves, so solve/simulate never demand a
    `*__buy_private` entry the pipeline cannot supply."""
    grid_config = dataclasses.replace(
        BENCHMARK_GRID_CONFIG, nbegm_live_labor_supply=True
    )
    model = create_aca_model(
        n_subjects=1,
        policy=policy,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        derived_categoricals=_DERIVED_CATEGORICALS,
        grid_config=grid_config,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver="nbegm",
    )
    template = model.get_params_template()
    offenders = [
        (regime_name, function_name)
        for regime_name, functions in template.items()
        if isinstance(functions, dict)
        for function_name, params in functions.items()
        if isinstance(params, dict) and "buy_private" in params
    ]
    assert offenders == [], offenders
