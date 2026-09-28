"""Real-package acceptance tests; requires the new upstream law-coverage API."""

import pytest
from helpers.model import make_baseline_model


@pytest.fixture(scope="module")
def model():
    return make_baseline_model()


def test_full_coverage_without_a_duplicate_entry_table(model):
    assert len(model.reachability.nodes) == 182
    assert model.initial_nodes == model.reachability.nodes
    assert (51, "dead") in model.reachability.nodes
    assert (64, "retiree_nomc_choose_canwork") in model.initial_nodes
    assert (65, "retiree_nomc_choose_canwork") not in model.initial_nodes
    assert not any(hasattr(r, "active") for r in model.user_regimes.values())


def test_boundary_targets_change_but_numeric_cells_are_shared(model):
    schedule = model.user_regimes[
        "retiree_nomc_inelig_canwork"
    ].regime_transitions.resolve(model.ages)
    assert set(schedule.at(60)) == {
        "retiree_nomc_inelig_canwork",
        "nongroup_nomc_inelig_canwork",
        "dead",
    }
    assert set(schedule.at(61)) == {
        "retiree_nomc_choose_canwork",
        "nongroup_nomc_choose_canwork",
        "dead",
    }
    assert schedule.at(60)["dead"] is schedule.at(61)["dead"]


def test_bequest_continuation_is_not_removed(model):
    regime = model.user_regimes["retiree_oamc_forced_forcedout"]
    assert set(regime.regime_transitions.resolve(model.ages).at(94)) == {"dead"}
    assert "consumption_dollars" in regime.actions
    assert (94, "retiree_oamc_forced_forcedout") in model.reachability.nodes
    assert (95, "dead") in model.reachability.nodes


def test_terminal_regime_keeps_the_original_none_declaration(model):
    assert model.user_regimes["dead"].regime_transitions is None
    assert {
        (age, "dead") for age in model.ages.exact_values
    } <= model.reachability.nodes
