"""DC-EGM solver wiring: `create_model(..., solver="dcegm")` builds the
post-decision spec — `DCEGM` config on every living regime, savings-form
assets laws, the solver-contract functions broadcast, no borrowing
constraint — and the model-level acceptance status against pylcm's DC-EGM
contract is pinned explicitly.
"""

import dataclasses
from collections.abc import Mapping
from typing import cast

import numpy as np
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
from aca_model.benchmark import (
    get_benchmark_consumption_dollars_points,
    get_benchmark_params,
)
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


def test_dcegm_model_slots_broadcast_contract_functions_and_constraint() -> None:
    """The DC-EGM slots broadcast `resources`/`savings`/
    `inverse_marginal_utility` and keep the borrowing constraint: the EGM
    solve enforces the limit through the savings grid's lower bound, but
    forward simulation re-decides consumption by an argmax over the
    consumption grid and needs the explicit feasibility mask."""
    slots = build_model_slots(
        grid_config=BENCHMARK_GRID_CONFIG,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver="dcegm",
    )
    assert "borrowing_constraint" in slots["constraints"]
    for name in ("resources", "savings", "inverse_marginal_utility"):
        assert name in slots["functions"]
    solver = _build_regimes("dcegm")["single_retiree_nomc_inelig_canwork"].solver
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


def test_dcegm_requires_construction_time_consumption_points() -> None:
    """`solver="dcegm"` without `consumption_dollars_points` fails at the
    aca-model boundary: the DC-EGM kernel needs the continuous-action grid
    at model construction, so the runtime-injection path cannot be used."""
    with pytest.raises(ValueError, match="consumption_dollars_points"):
        create_model(
            n_subjects=1,
            fixed_params=_FIXED_PARAMS,
            wage_params=_WAGE_PARAMS,
            derived_categoricals=_DERIVED_CATEGORICALS,
            grid_config=BENCHMARK_GRID_CONFIG,
            pref_type_grid=DiscreteGrid(BenchmarkPrefType),
            solver="dcegm",
        )


def test_benchmark_consumption_points_pin_both_floors() -> None:
    """The construction-time benchmark consumption grid uses the same
    formula as the runtime injection: the single floor first, the married
    floor second, a geomspace tail up to `max_consumption_dollars`."""
    _, _, params = get_benchmark_params(model=None)
    points = get_benchmark_consumption_dollars_points(n_points=5)
    floor = float(params["consumption_equiv_floor"])
    exponent = float(_FIXED_PARAMS["exponent"])
    np.testing.assert_allclose(points[:2], [floor, floor * 2.0**exponent], rtol=1e-12)


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
    contract admits the ACA budget chains, this builds without error. The
    construction-time consumption points are supplied so the build reaches
    the upstream limitation rather than the missing-points guard.
    """
    model = create_model(
        n_subjects=1,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        derived_categoricals=_DERIVED_CATEGORICALS,
        grid_config=BENCHMARK_GRID_CONFIG,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver="dcegm",
        consumption_dollars_points=get_benchmark_consumption_dollars_points(
            n_points=BENCHMARK_GRID_CONFIG.n_consumption_dollars_gridpoints
        ),
    )
    assert isinstance(
        model.user_regimes["single_retiree_nomc_inelig_canwork"].solver, DCEGM
    )


def test_savings_grid_batch_size_follows_grid_config() -> None:
    """`GridConfig.n_savings_batch_size` sets the `batch_size` on every
    living regime's DC-EGM savings grid, so the post-decision continuation
    splays into `lax.map` blocks of that width."""
    grid_config = dataclasses.replace(BENCHMARK_GRID_CONFIG, n_savings_batch_size=50)
    regimes = build_all_regimes(
        grid_config=grid_config,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver="dcegm",
    )
    for name in REGIME_SPECS:
        solver = cast("DCEGM", regimes[name].solver)
        assert solver.savings_grid.batch_size == 50, name


def test_stochastic_node_batch_size_follows_grid_config() -> None:
    """`GridConfig.n_stochastic_node_batch_size` sets `stochastic_node_batch_size`
    on every living regime's DC-EGM solver, so the child stochastic-node
    expectation splays into `lax.map` blocks of that width."""
    grid_config = dataclasses.replace(
        BENCHMARK_GRID_CONFIG, n_stochastic_node_batch_size=7
    )
    regimes = build_all_regimes(
        grid_config=grid_config,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver="dcegm",
    )
    for name in REGIME_SPECS:
        solver = cast("DCEGM", regimes[name].solver)
        assert solver.stochastic_node_batch_size == 7, name


def test_savings_grid_length_follows_grid_config() -> None:
    """`GridConfig.n_savings_gridpoints` sets the number of nodes on every
    living regime's DC-EGM savings grid."""
    grid_config = dataclasses.replace(BENCHMARK_GRID_CONFIG, n_savings_gridpoints=70)
    regimes = build_all_regimes(
        grid_config=grid_config,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver="dcegm",
    )
    for name in REGIME_SPECS:
        solver = cast("DCEGM", regimes[name].solver)
        assert len(solver.savings_grid.to_jax()) == 70, name


def test_dcegm_savings_grid_rejects_too_few_points() -> None:
    """`n_savings_gridpoints < 2` cannot form the cubically clustered DC-EGM
    savings grid, so building the dcegm regimes raises a clear `ValueError`."""
    grid_config = dataclasses.replace(BENCHMARK_GRID_CONFIG, n_savings_gridpoints=1)
    with pytest.raises(ValueError, match="n_savings_gridpoints"):
        build_all_regimes(
            grid_config=grid_config,
            fixed_params=_FIXED_PARAMS,
            wage_params=_WAGE_PARAMS,
            pref_type_grid=DiscreteGrid(BenchmarkPrefType),
            solver="dcegm",
        )
