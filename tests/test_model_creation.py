"""Tests for baseline model creation and regime structure."""

import inspect
from collections.abc import Mapping

import pytest
from helpers.model import (  # ty: ignore[unresolved-import]
    make_aca_model,
    make_baseline_model,
)
from lcm import DiscreteGrid, Phased

from aca_model.aca import health_insurance as aca_hi
from aca_model.aca.health_insurance import PolicyVariant
from aca_model.aca.regimes import build_all_regimes as _build_aca_regimes
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline.regimes import REGIME_SPECS, RegimeId
from aca_model.baseline.regimes import build_regime as _build_regime
from aca_model.baseline.regimes._common import (
    build_grids,
    build_model_state_transitions,
    build_model_states,
)
from aca_model.benchmark import get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG
from aca_model.environment import pensions

_FIXED_PARAMS, _WAGE_PARAMS, _ = get_benchmark_params(model=None)


def build_aca_regimes(policy: PolicyVariant) -> dict:
    return _build_aca_regimes(
        policy=policy,
        grid_config=BENCHMARK_GRID_CONFIG,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )


_GRIDS = build_grids(
    grid_config=BENCHMARK_GRID_CONFIG,
    fixed_params=_FIXED_PARAMS,
    wage_params=_WAGE_PARAMS,
    pref_type_grid=DiscreteGrid(BenchmarkPrefType),
)


def build_regime(name: str):
    return _build_regime(name, _GRIDS)


def test_model_creates_successfully() -> None:
    model = make_baseline_model(n_subjects=1)
    assert len(model.user_regimes) == 37
    assert model.n_periods == 45


def test_model_age_range() -> None:
    model = make_baseline_model(n_subjects=1)
    assert model.ages.values[0] == 51.0
    assert model.ages.values[-1] == 95.0


def test_dead_regime_is_terminal() -> None:
    model = make_baseline_model(n_subjects=1)
    assert model.user_regimes["dead"].terminal


def test_non_terminal_regimes_not_terminal() -> None:
    model = make_baseline_model(n_subjects=1)
    for name in REGIME_SPECS:
        assert not model.user_regimes[name].terminal


def test_regime_id_dead_is_last() -> None:
    assert RegimeId.dead == 36


@pytest.mark.parametrize(
    "name",
    [n for n, s in REGIME_SPECS.items() if s["canwork"] == "forcedout"],
)
def test_forcedout_regimes_no_labor_supply(name: str) -> None:
    regime = build_regime(name)
    assert "labor_supply" not in regime.actions
    assert "log_ft_wage_res" not in regime.states
    assert "consumption_dollars" in regime.actions


@pytest.mark.parametrize(
    "name",
    [n for n, s in REGIME_SPECS.items() if s["canwork"] == "canwork"],
)
def test_choose_regimes_have_labor_supply(name: str) -> None:
    regime = build_regime(name)
    assert "labor_supply" in regime.actions
    assert "log_ft_wage_res" in regime.states


@pytest.mark.parametrize(
    "name",
    [n for n, s in REGIME_SPECS.items() if s["his"] == "tied"],
)
def test_tied_regimes_have_no_lagged_labor_supply(name: str) -> None:
    regime = build_regime(name)
    assert "lagged_labor_supply" not in regime.states


@pytest.mark.parametrize(
    "name",
    [n for n, s in REGIME_SPECS.items() if s["ss"] == "choose"],
)
def test_ss_choose_regimes_have_claim_ss(name: str) -> None:
    regime = build_regime(name)
    assert "claim_ss" in regime.actions
    assert "claimed_ss" in regime.states


@pytest.mark.parametrize(
    "name",
    [
        n
        for n, s in REGIME_SPECS.items()
        if s["his"] == "nongroup" and s["mc"] == "nomc"
    ],
)
def test_nongroup_inelig_have_buy_private(name: str) -> None:
    regime = build_regime(name)
    assert "buy_private" in regime.actions


@pytest.mark.parametrize(
    "name",
    [
        n
        for n, s in REGIME_SPECS.items()
        if s["his"] == "nongroup" and s["mc"] != "nomc"
    ],
)
def test_nongroup_with_mc_no_buy_private(name: str) -> None:
    regime = build_regime(name)
    assert "buy_private" not in regime.actions


def test_all_non_terminal_regimes_have_core_states() -> None:
    """`health` varies per regime; the other core states are broadcast from
    the model level into every regime."""
    for name in REGIME_SPECS:
        regime = build_regime(name)
        assert "health" in regime.states
    model_states = build_model_states(_GRIDS)
    for state_name in ("assets", "aime", "hcc_persistent", "hcc_transitory"):
        assert state_name in model_states


def test_pre65_regimes_use_health_with_disability() -> None:
    for name, spec in REGIME_SPECS.items():
        if spec["mc"] in ("nomc", "dimc"):
            regime = build_regime(name)
            grid = regime.states["health"]
            assert len(grid.categories) == 3, f"{name} should use HealthWithDisability"


def test_post65_regimes_use_health() -> None:
    for name, spec in REGIME_SPECS.items():
        if spec["mc"] == "oamc":
            regime = build_regime(name)
            grid = regime.states["health"]
            assert len(grid.categories) == 2, f"{name} should use Health"


def test_all_regimes_have_aime() -> None:
    """`aime` is broadcast from the model level; its spec-dependent law of
    motion stays regime-level."""
    assert "aime" in build_model_states(_GRIDS)
    for name in REGIME_SPECS:
        regime = build_regime(name)
        assert "aime" in regime.state_transitions, f"{name} should have an aime law"


@pytest.mark.parametrize(
    "name",
    [n for n, s in REGIME_SPECS.items() if s["ss"] in ("inelig", "forced")],
)
def test_inelig_and_forced_regimes_aime_law_takes_no_claim_inputs(name: str) -> None:
    """Regimes that cannot choose when to claim use a claim-free AIME law.

    `inelig` agents cannot claim and `forced` agents claim by rule, so neither
    carries the `claim_ss` action or the `claimed_ss` state. Their AIME law of
    motion must not require those as inputs, or model validation rejects the
    regime for a missing `next_aime__claim_ss` / `next_aime__claimed_ss` param.
    """
    regime = build_regime(name)
    aime_law = regime.state_transitions["aime"]
    params = set(inspect.signature(aime_law).parameters)
    assert "claim_ss" not in params
    assert "claimed_ss" not in params


@pytest.mark.parametrize(
    "name",
    [n for n, s in REGIME_SPECS.items() if s["ss"] == "choose"],
)
def test_choose_regimes_aime_law_takes_claim_inputs(name: str) -> None:
    """`ss=choose` regimes bake the claim-age adjustment, so their AIME law
    reads the `claim_ss` action and `claimed_ss` state."""
    regime = build_regime(name)
    aime_law = regime.state_transitions["aime"]
    params = set(inspect.signature(aime_law).parameters)
    assert "claim_ss" in params
    assert "claimed_ss" in params


def test_regime_specs_keys_match_regime_id() -> None:
    """Every REGIME_SPECS key has a matching RegimeId field."""
    for name in REGIME_SPECS:
        assert hasattr(RegimeId, name), f"RegimeId missing field for {name}"


def test_all_non_terminal_regimes_carry_pension_wealth_as_carried_state() -> None:
    """`pension_wealth` is a carried state broadcast into every living regime.

    Imputed from AIME during solve (never a solve grid axis) yet seeded and
    evolved as the agent's actual pension wealth during simulate, so the true
    value survives every regime transition rather than being reset to the
    AIME imputation on entering retirement / forced-out regimes.
    """
    carried = build_model_states(_GRIDS)["pension_wealth"]
    assert isinstance(carried, Phased)
    assert carried.solve is pensions.wealth
    assert (
        build_model_state_transitions()["pension_wealth"]
        is pensions.wealth_next_before_adjustment
    )
    model = make_baseline_model(n_subjects=1)
    for name in REGIME_SPECS:
        assert isinstance(model.user_regimes[name].states["pension_wealth"], Phased), (
            name
        )


def test_pension_wealth_is_not_a_solve_function() -> None:
    """The carried pension-wealth state lives in `states`, not `functions`.

    Its solve variant supplies the imputed value, so a separate
    `functions["pension_wealth"]` entry would double-define it.
    """
    regime = build_regime("single_retiree_nomc_inelig_canwork")
    assert "pension_wealth" not in regime.functions


def test_pension_assets_adjustment_is_phase_variant() -> None:
    """The pension assets adjustment corrects the imputation gap in solve only.

    In simulate the true pension wealth is carried directly, so the adjustment
    is zero — a `Phased` function with a zero simulate variant.
    """
    regime = build_regime("single_retiree_nomc_inelig_canwork")
    adjustment = regime.functions["pension_assets_adjustment"]
    assert isinstance(adjustment, Phased)
    assert adjustment.solve is pensions.assets_adjustment


def test_per_target_health_transitions() -> None:
    """All regimes use per-target health transition dicts."""
    for name in REGIME_SPECS:
        regime = build_regime(name)
        health_trans = regime.state_transitions["health"]
        assert isinstance(health_trans, Mapping), (
            f"{name} should have per-target health transitions"
        )


def test_hcc_persistent_and_transitory_are_shock_grids() -> None:
    """hcc_persistent and hcc_transitory are ShockGrids with intrinsic transitions."""
    for name in REGIME_SPECS:
        regime = build_regime(name)
        assert "hcc_persistent" not in regime.state_transitions
        assert "hcc_transitory" not in regime.state_transitions


def test_aca_model_creates_successfully() -> None:
    model = make_aca_model(n_subjects=1, policy=PolicyVariant.ACA)
    assert len(model.user_regimes) == 37
    assert model.n_periods == 45


def test_aca_nongroup_inelig_has_real_functions() -> None:
    """Nongroup+nomc regimes get real ACA functions under ACA policy."""
    regimes = build_aca_regimes(PolicyVariant.ACA)
    regime = regimes["single_nongroup_nomc_inelig_canwork"]
    assert regime.functions["mandate_penalty"] is aca_hi.mandate_penalty
    assert regime.functions["hic_premium_subsidy"] is aca_hi.premium_subsidy
    assert regime.functions["cost_sharing_scale"] is aca_hi.cost_sharing
    assert regime.functions["cash_on_hand"] is aca_hi.cash_on_hand
    assert regime.functions["primary_oop"] is aca_hi.primary_oop
    assert regime.functions["is_medicaid_eligible"] is aca_hi.is_medicaid_eligible


def test_aca_no_mandate_has_no_mandate_function() -> None:
    """ACA_NO_MANDATE: mandate_penalty is a fixed param, not a DAG function."""
    regimes = build_aca_regimes(PolicyVariant.ACA_NO_MANDATE)
    regime = regimes["single_nongroup_nomc_inelig_canwork"]
    assert "mandate_penalty" not in regime.functions
    assert regime.functions["hic_premium_subsidy"] is aca_hi.premium_subsidy
    assert regime.functions["cost_sharing_scale"] is aca_hi.cost_sharing
    assert regime.functions["cash_on_hand"] is aca_hi.cash_on_hand


def test_aca_other_regimes_have_no_aca_policy_keys() -> None:
    """Non-nongroup regimes have no mandate/subsidy/cost-sharing keys."""
    regimes = build_aca_regimes(PolicyVariant.ACA)
    regime = regimes["single_retiree_nomc_inelig_canwork"]
    assert "mandate_penalty" not in regime.functions
    assert "hic_premium_subsidy" not in regime.functions
    assert "cost_sharing_scale" not in regime.functions
    # Medicaid expansion applies to ALL regimes
    assert regime.functions["is_medicaid_eligible"] is aca_hi.is_medicaid_eligible


@pytest.mark.parametrize("policy", list(PolicyVariant))
def test_all_policy_variants_create(policy: PolicyVariant) -> None:
    """All policy variants create valid models."""
    model = make_aca_model(n_subjects=1, policy=policy)
    assert len(model.user_regimes) == 37


def test_aca_no_medicaid_expansion_keeps_baseline_medicaid() -> None:
    """ACA_NO_MEDICAID_EXPANSION: baseline Medicaid, but has subsidies + mandate."""
    regimes = build_aca_regimes(PolicyVariant.ACA_NO_MEDICAID_EXPANSION)
    retiree = regimes["single_retiree_nomc_inelig_canwork"]
    assert retiree.functions["is_medicaid_eligible"] is not aca_hi.is_medicaid_eligible

    nongroup = regimes["single_nongroup_nomc_inelig_canwork"]
    assert nongroup.functions["mandate_penalty"] is aca_hi.mandate_penalty
    assert nongroup.functions["hic_premium_subsidy"] is aca_hi.premium_subsidy


def test_aca_no_medicaid_expansion_no_mandate() -> None:
    """ACA_NO_MEDICAID_EXPANSION_NO_MANDATE: baseline Medicaid, subsidies, no mandate."""
    regimes = build_aca_regimes(PolicyVariant.ACA_NO_MEDICAID_EXPANSION_NO_MANDATE)
    retiree = regimes["single_retiree_nomc_inelig_canwork"]
    assert retiree.functions["is_medicaid_eligible"] is not aca_hi.is_medicaid_eligible

    nongroup = regimes["single_nongroup_nomc_inelig_canwork"]
    assert "mandate_penalty" not in nongroup.functions
    assert nongroup.functions["hic_premium_subsidy"] is aca_hi.premium_subsidy


def test_aca_only_medicaid_expansion() -> None:
    """ACA_ONLY_MEDICAID_EXPANSION: Medicaid expansion only, no subsidies/mandate."""
    regimes = build_aca_regimes(PolicyVariant.ACA_ONLY_MEDICAID_EXPANSION)
    retiree = regimes["single_retiree_nomc_inelig_canwork"]
    assert retiree.functions["is_medicaid_eligible"] is aca_hi.is_medicaid_eligible

    nongroup = regimes["single_nongroup_nomc_inelig_canwork"]
    assert "mandate_penalty" not in nongroup.functions
    assert "hic_premium_subsidy" not in nongroup.functions
    assert "cost_sharing_scale" not in nongroup.functions


def test_baseline_model_creates() -> None:
    """Baseline model creates successfully without PolicyVariant."""
    model = make_baseline_model(n_subjects=1)
    assert len(model.user_regimes) == 37


def test_a_sharded_pref_type_grid_reaches_the_model_level_states() -> None:
    """A `distributed` `pref_type` grid arrives at the model level intact.

    `pref_type` is the only shardable axis — sharding is legal only on
    model-level states, and it is the one that remains — so the caller builds
    the grid with the flag and the model must carry it through unchanged.
    """
    grids = build_grids(
        grid_config=BENCHMARK_GRID_CONFIG,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType, distributed=True),
    )
    model_states = build_model_states(grids)
    assert model_states["pref_type"].distributed is True


@pytest.mark.parametrize(
    "state_name",
    ["lagged_labor_supply", "claimed_ss", "spousal_income"],
)
def test_discrete_state_distributed_flag_defaults_to_false(state_name: str) -> None:
    """`distributed` on inline-built discrete states defaults to `False` so
    configurations that do not opt in see no behaviour change."""
    grid = build_regime("married_retiree_dimc_choose_canwork").states[state_name]
    assert grid.distributed is False


def test_dead_regime_prunes_unused_broadcast_states() -> None:
    """States every living regime shares are declared once at the model level;
    `dead` keeps only what the bequest DAG reads (`assets`, `pref_type`).
    `pension_wealth` is masked (carried states are illegal in terminal
    regimes); the other unused broadcast states are pruned by reachability."""
    model = make_baseline_model(n_subjects=1)
    assert model.pruned_variables["dead"] == frozenset(
        {"aime", "hcc_persistent", "hcc_transitory"}
    )
    assert set(model.user_regimes["dead"].states) == {"assets", "pref_type"}


def test_living_regimes_keep_every_broadcast_state() -> None:
    """Every model-level state is read by each living regime's DAG, so
    pruning removes nothing outside `dead`."""
    model = make_baseline_model(n_subjects=1)
    for name in REGIME_SPECS:
        assert model.pruned_variables[name] == frozenset()
