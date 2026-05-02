"""Tests for baseline model creation and regime structure."""

from collections.abc import Mapping

import pytest

from aca_model.aca import health_insurance as aca_hi
from aca_model.aca.health_insurance import PolicyVariant
from aca_model.aca.model import create_model as create_aca_model
from aca_model.aca.regimes import build_all_regimes as build_aca_regimes
from aca_model.baseline.model import create_model
from aca_model.baseline.regimes import REGIME_SPECS, RegimeId
from aca_model.baseline.regimes import build_regime as _build_regime
from aca_model.baseline.regimes._common import build_grids

_GRIDS = build_grids()


def build_regime(name: str):
    return _build_regime(name, _GRIDS)


def test_model_creates_successfully() -> None:
    model = create_model(n_subjects=1)
    assert len(model.regimes) == 19
    assert model.n_periods == 45


def test_model_age_range() -> None:
    model = create_model(n_subjects=1)
    assert model.ages.values[0] == 51.0
    assert model.ages.values[-1] == 95.0


def test_dead_regime_is_terminal() -> None:
    model = create_model(n_subjects=1)
    assert model.regimes["dead"].terminal


def test_non_terminal_regimes_not_terminal() -> None:
    model = create_model(n_subjects=1)
    for name in REGIME_SPECS:
        assert not model.regimes[name].terminal


def test_regime_id_dead_is_last() -> None:
    assert RegimeId.dead == 18


@pytest.mark.parametrize(
    "name",
    [n for n, s in REGIME_SPECS.items() if s["canwork"] == "forcedout"],
)
def test_forcedout_regimes_no_labor_supply(name: str) -> None:
    regime = build_regime(name)
    assert "labor_supply" not in regime.actions
    assert "log_ft_wage_res" not in regime.states
    assert "consumption" in regime.actions


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
    for name in REGIME_SPECS:
        regime = build_regime(name)
        assert "health" in regime.states
        assert "spousal_income" in regime.states
        assert "assets" in regime.states
        assert "hcc_persistent" in regime.states
        assert "hcc_transitory" in regime.states


def test_pre65_regimes_use_health_with_disability() -> None:
    for name, spec in REGIME_SPECS.items():
        if spec["mc"] in ("nomc", "dimc"):
            regime = build_regime(name)
            grid = regime.states["health"]
            assert len(grid.categories) == 3, f"{name} should use HealthWithDisability"  # ty: ignore[unresolved-attribute]


def test_post65_regimes_use_health() -> None:
    for name, spec in REGIME_SPECS.items():
        if spec["mc"] == "oamc":
            regime = build_regime(name)
            grid = regime.states["health"]
            assert len(grid.categories) == 2, f"{name} should use Health"  # ty: ignore[unresolved-attribute]


def test_all_regimes_have_aime() -> None:
    for name in REGIME_SPECS:
        regime = build_regime(name)
        assert "aime" in regime.states, f"{name} should have aime"


def test_regime_specs_keys_match_regime_id() -> None:
    """Every REGIME_SPECS key has a matching RegimeId field."""
    for name in REGIME_SPECS:
        assert hasattr(RegimeId, name), f"RegimeId missing field for {name}"


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
    model = create_aca_model(n_subjects=1)
    assert len(model.regimes) == 19
    assert model.n_periods == 45


def test_aca_nongroup_inelig_has_real_functions() -> None:
    """Nongroup+nomc regimes get real ACA functions under ACA policy."""
    regimes = build_aca_regimes(PolicyVariant.ACA)
    regime = regimes["nongroup_nomc_inelig_canwork"]
    assert regime.functions["mandate_penalty"] is aca_hi.mandate_penalty
    assert regime.functions["hic_premium_subsidy"] is aca_hi.premium_subsidy
    assert regime.functions["cost_sharing_scale"] is aca_hi.cost_sharing
    assert regime.functions["cash_on_hand"] is aca_hi.cash_on_hand
    assert regime.functions["primary_oop"] is aca_hi.primary_oop
    assert regime.functions["is_medicaid_eligible"] is aca_hi.is_medicaid_eligible


def test_aca_no_mandate_has_no_mandate_function() -> None:
    """ACA_NO_MANDATE: mandate_penalty is a fixed param, not a DAG function."""
    regimes = build_aca_regimes(PolicyVariant.ACA_NO_MANDATE)
    regime = regimes["nongroup_nomc_inelig_canwork"]
    assert "mandate_penalty" not in regime.functions
    assert regime.functions["hic_premium_subsidy"] is aca_hi.premium_subsidy
    assert regime.functions["cost_sharing_scale"] is aca_hi.cost_sharing
    assert regime.functions["cash_on_hand"] is aca_hi.cash_on_hand


def test_aca_other_regimes_have_no_aca_policy_keys() -> None:
    """Non-nongroup regimes have no mandate/subsidy/cost-sharing keys."""
    regimes = build_aca_regimes(PolicyVariant.ACA)
    regime = regimes["retiree_nomc_inelig_canwork"]
    assert "mandate_penalty" not in regime.functions
    assert "hic_premium_subsidy" not in regime.functions
    assert "cost_sharing_scale" not in regime.functions
    # Medicaid expansion applies to ALL regimes
    assert regime.functions["is_medicaid_eligible"] is aca_hi.is_medicaid_eligible


@pytest.mark.parametrize("policy", list(PolicyVariant))
def test_all_policy_variants_create(policy: PolicyVariant) -> None:
    """All policy variants create valid models."""
    model = create_aca_model(n_subjects=1, policy=policy)
    assert len(model.regimes) == 19


def test_aca_no_medicaid_expansion_keeps_baseline_medicaid() -> None:
    """ACA_NO_MEDICAID_EXPANSION: baseline Medicaid, but has subsidies + mandate."""
    regimes = build_aca_regimes(PolicyVariant.ACA_NO_MEDICAID_EXPANSION)
    retiree = regimes["retiree_nomc_inelig_canwork"]
    assert retiree.functions["is_medicaid_eligible"] is not aca_hi.is_medicaid_eligible

    nongroup = regimes["nongroup_nomc_inelig_canwork"]
    assert nongroup.functions["mandate_penalty"] is aca_hi.mandate_penalty
    assert nongroup.functions["hic_premium_subsidy"] is aca_hi.premium_subsidy


def test_aca_no_medicaid_expansion_no_mandate() -> None:
    """ACA_NO_MEDICAID_EXPANSION_NO_MANDATE: baseline Medicaid, subsidies, no mandate."""
    regimes = build_aca_regimes(PolicyVariant.ACA_NO_MEDICAID_EXPANSION_NO_MANDATE)
    retiree = regimes["retiree_nomc_inelig_canwork"]
    assert retiree.functions["is_medicaid_eligible"] is not aca_hi.is_medicaid_eligible

    nongroup = regimes["nongroup_nomc_inelig_canwork"]
    assert "mandate_penalty" not in nongroup.functions
    assert nongroup.functions["hic_premium_subsidy"] is aca_hi.premium_subsidy


def test_aca_only_medicaid_expansion() -> None:
    """ACA_ONLY_MEDICAID_EXPANSION: Medicaid expansion only, no subsidies/mandate."""
    regimes = build_aca_regimes(PolicyVariant.ACA_ONLY_MEDICAID_EXPANSION)
    retiree = regimes["retiree_nomc_inelig_canwork"]
    assert retiree.functions["is_medicaid_eligible"] is aca_hi.is_medicaid_eligible

    nongroup = regimes["nongroup_nomc_inelig_canwork"]
    assert "mandate_penalty" not in nongroup.functions
    assert "hic_premium_subsidy" not in nongroup.functions
    assert "cost_sharing_scale" not in nongroup.functions


def test_baseline_model_creates() -> None:
    """Baseline model creates successfully without PolicyVariant."""
    model = create_model(n_subjects=1)
    assert len(model.regimes) == 19
