"""NBEGM solver wiring.

`solver="nbegm"` solves every living regime with NB-EGM, so each of them carries
the savings-form budget the solver reads: the budget node is `resources` and the
post-decision function is `savings`, the same spec DC-EGM uses. The `dead` regime
is terminal and keeps its own solver.

The regime declares every choice its structure affords — whether to buy
non-group coverage, and how many hours to work — and the discrete envelope
branches over the Cartesian product of those grids. The regime is never narrowed
to fit the solver: a solver that cannot carry a choice refuses the regime rather
than being handed a model that omits it.
"""

import dataclasses
from collections.abc import Mapping
from typing import cast

import pytest
from helpers.model import _DERIVED_CATEGORICALS  # ty: ignore[unresolved-import]
from lcm import DiscreteGrid, Model, Regime
from lcm.exceptions import RegimeInitializationError
from lcm.solvers import NBEGM

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
from aca_model.config import BENCHMARK_GRID_CONFIG, GridConfig
from aca_model.environment import taxes

_FIXED_PARAMS, _WAGE_PARAMS, _ = get_benchmark_params(model=None)

_M1_REGIME = "single_nongroup_nomc_inelig_canwork"
_BRUTE_REGIME = "single_retiree_nomc_inelig_canwork"

# `labor_supply` enters `countable_income`, which carries the SSI income test, so each
# labor level puts that breakpoint at a different liquid level. The one-sided read
# publishes its cliff limits on a single query grid shared across branches, which the
# two cannot both satisfy, so a regime carrying `labor_supply` needs the bridged read.
_BRIDGED_GRID_CONFIG = dataclasses.replace(
    BENCHMARK_GRID_CONFIG, nbegm_jump_read="bridged"
)


def _build_regimes(solver: SolverName) -> dict[str, Regime]:
    return build_all_regimes(
        grid_config=BENCHMARK_GRID_CONFIG,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver=solver,
    )


def _build_model_with(solver: SolverName, grid_config: GridConfig) -> Model:
    return create_model(
        n_subjects=1,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        derived_categoricals=_DERIVED_CATEGORICALS,
        grid_config=grid_config,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver=solver,
    )


def _build_model(solver: SolverName) -> Model:
    return _build_model_with(solver, _BRIDGED_GRID_CONFIG)


def _grids() -> Grids:
    return build_grids(
        grid_config=BENCHMARK_GRID_CONFIG,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )


def test_nbegm_attaches_to_every_living_regime() -> None:
    """`solver="nbegm"` solves every living regime with NB-EGM.

    A solver that reached only some regimes would leave the rest on brute
    force while still reporting itself as the model's solver, so the choice of
    solver would not be visible in the result it produced.
    """
    regimes = _build_regimes("nbegm")
    on_brute_force = [
        name for name in REGIME_SPECS if not isinstance(regimes[name].solver, NBEGM)
    ]
    assert on_brute_force == []


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


def test_nbegm_m1_regime_declares_the_same_actions_as_brute_force() -> None:
    """Which solver runs a regime does not change the choices the household has.

    The M1 regime affords a coverage choice and an hours choice, so it declares
    both under either solver.
    """
    nbegm_m1 = _build_regimes("nbegm")[_M1_REGIME]
    brute_m1 = _build_regimes("brute_force")[_M1_REGIME]
    assert set(nbegm_m1.actions) == set(brute_m1.actions)


def test_nbegm_m1_regime_declares_both_discrete_actions() -> None:
    """The M1 regime's discrete choices are whether to buy non-group coverage and
    how many hours to work."""
    nbegm_m1 = _build_regimes("nbegm")[_M1_REGIME]
    discrete = {
        name
        for name, grid in nbegm_m1.actions.items()
        if isinstance(grid, DiscreteGrid)
    }
    assert discrete == {"buy_private", "labor_supply"}


def test_nbegm_model_builds_a_regime_declaring_several_discrete_actions() -> None:
    """The M1 regime builds under NBEGM with both of its discrete actions live.

    The discrete envelope branches over the Cartesian product of the regime's
    discrete action grids, so a regime is never narrowed to fit the solver.
    """
    model = _build_model("nbegm")
    nbegm_m1 = model.user_regimes[_M1_REGIME]
    discrete = {
        name
        for name, grid in nbegm_m1.actions.items()
        if isinstance(grid, DiscreteGrid)
    }
    assert discrete == {"buy_private", "labor_supply"}


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


def test_nbegm_gives_every_living_regime_the_savings_form_budget() -> None:
    """Under NBEGM every living regime carries `resources` and `savings`.

    They are the solver's budget contract, so a regime NB-EGM solves without
    them cannot be built at all; a regime it does not solve has no use for
    them, which is why the brute-force build omits them.
    """
    model = _build_model("nbegm")
    missing = [
        name
        for name in REGIME_SPECS
        if not {"resources", "savings"} <= set(model.user_regimes[name].functions)
    ]
    assert missing == []


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
    assert floor_breakpoint.threshold == "consumption_dollars_floor"
    assert floor_breakpoint.kind == "continuous_kink"


def test_is_ssi_eligible_declares_the_asset_test_jump() -> None:
    """The SSI asset test is a declared jump on `assets`: cash-on-hand drops by
    the SSI benefit where eligibility ends, so NBEGM's partition must split
    there instead of extrapolating one affine budget across the cliff."""
    meta = health_insurance.is_ssi_eligible.__lcm_piecewise_affine__  # ty: ignore[unresolved-attribute]
    assert meta.variable == "assets"
    (asset_test,) = meta.breakpoints
    assert asset_test.threshold == "ssi_assets_test"
    assert asset_test.kind == "jump"


def test_ssi_benefit_declares_the_income_test_kink() -> None:
    """The SSI income test is a declared continuous kink on `countable_income`:
    the benefit reaches zero exactly at the test, so the budget's slope changes
    (capital income stops being offset) without a jump."""
    meta = health_insurance.ssi_benefit.__lcm_piecewise_affine__  # ty: ignore[unresolved-attribute]
    assert meta.variable == "countable_income"
    (income_test,) = meta.breakpoints
    assert income_test.threshold == "ssi_maximum_benefit"
    assert income_test.kind == "continuous_kink"


@pytest.mark.parametrize(
    "declared",
    [
        assets_and_income.resources,
        health_insurance.is_ssi_eligible,
        health_insurance.ssi_benefit,
        taxes.after_tax_income,
    ],
)
def test_no_breakpoint_threshold_is_indexed_by_a_state(declared: object) -> None:
    """Every breakpoint threshold is a plain name, not a table indexed by a
    ride-along state.

    Marital status is a regime axis, so a regime's schedules already carry only
    its own row. An `indexed_by` declaration would name a state the regime does
    not have, and NBEGM would read the wrong cell of a table that no longer has
    a marital dimension to index.
    """
    meta = declared.__lcm_piecewise_affine__  # ty: ignore[unresolved-attribute]
    for breakpoint_ in meta.breakpoints:
        assert breakpoint_.indexed_by is None, breakpoint_.threshold


def test_nbegm_labor_supply_requires_the_bridged_cliff_read() -> None:
    """The M1 regime builds under NBEGM only with `nbegm_jump_read="bridged"`.

    `labor_supply` enters `countable_income`, which carries the SSI income test,
    so each labor level puts that breakpoint at a different liquid level. The
    one-sided read publishes its cliff limits on one query grid shared across
    branches, so the two cannot both hold and the build is refused.
    """
    with pytest.raises(RegimeInitializationError, match="must not enter any schedule"):
        _build_model_with("nbegm", BENCHMARK_GRID_CONFIG)

    assert isinstance(_build_model_with("nbegm", _BRIDGED_GRID_CONFIG), Model)


@pytest.mark.parametrize("policy", list(PolicyVariant))
def test_nbegm_builds_every_aca_policy_variant(policy: PolicyVariant) -> None:
    """Every ACA policy variant builds a model under NBEGM with the M1 regime on
    the solver and both discrete choices live — the overlay's function swaps
    compose with the branch compiler's per-regime wiring."""
    grid_config = _BRIDGED_GRID_CONFIG
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
    """`buy_private` is a choice, never a parameter.

    No ACA-swapped function may leave it as a free parameter — the params
    template holds no `buy_private` leaves, so solve/simulate never demand a
    `*__buy_private` entry the pipeline cannot supply.
    """
    grid_config = _BRIDGED_GRID_CONFIG
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
