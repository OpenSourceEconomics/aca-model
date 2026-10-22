"""The baseline's solved domain derives from its declared starting pairs."""

import jax.numpy as jnp
import pandas as pd
import pytest
from helpers.model import make_baseline_model
from lcm.exceptions import InvalidInitialConditionsError

from aca_model.baseline.regimes import ENTRY_REGIMES, REGIME_SPECS
from aca_model.baseline.regimes._common import build_model_edges
from aca_model.benchmark import get_benchmark_initial_conditions, get_benchmark_params
from aca_model.config import MODEL_CONFIG
from aca_model.simulation import select_admissible_starts


@pytest.fixture(scope="module")
def model():
    return make_baseline_model()


def test_initial_nodes_are_the_age_51_to_60_entry_regimes(model):
    expected = {
        (age, regime)
        for age in range(MODEL_CONFIG.start_age, MODEL_CONFIG.last_start_age + 1)
        for regime in ENTRY_REGIMES
    }
    assert {(float(a), r) for a, r in model.graph.initial_nodes} == {
        (float(a), r) for a, r in expected
    }


def test_solved_domain_has_184_nodes(model):
    assert len(model.graph.nodes) == 184


def test_dead_at_the_first_age_is_not_solved(model):
    assert (51, "dead") not in model.graph.nodes


def test_later_living_nodes_are_solved_but_not_admissible(model):
    node = (64, "retiree_nomc_choose_canwork")
    assert (node in model.graph.nodes, node in model.graph.initial_nodes) == (
        True,
        False,
    )


def test_regimes_carry_no_active_attribute(model):
    assert not any(hasattr(r, "active") for r in model.user_regimes.values())


@pytest.mark.parametrize(
    ("age", "regime"),
    [
        (61.0, "retiree_nomc_inelig_canwork"),
        (62.0, "retiree_nomc_choose_canwork"),
        (69.0, "retiree_oamc_choose_canwork"),
        (51.0, "dead"),
        (60.0, "dead"),
    ],
)
def test_simulate_rejects_starts_outside_the_entry_contract(model, age, regime):
    _, _, params = get_benchmark_params(model=model)
    ic = get_benchmark_initial_conditions(model=model, n_subjects=2, seed=0)
    ic = {
        **ic,
        "age": jnp.full(2, age),
        "regime_id": jnp.full(2, model.regime_names_to_ids[regime], dtype=jnp.int32),
    }
    with pytest.raises(InvalidInitialConditionsError):
        model.simulate(params=params, initial_conditions=ic, log_level="off")


def test_boundary_targets_change_with_source_age(model):
    assert set(
        model.graph.solution.targets(period=9, source="retiree_nomc_inelig_canwork")
    ) == {
        "retiree_nomc_inelig_canwork",
        "retiree_dimc_inelig_canwork",
        "nongroup_nomc_inelig_canwork",
        "nongroup_dimc_inelig_canwork",
        "dead",
    }
    assert set(
        model.graph.solution.targets(period=10, source="retiree_nomc_inelig_canwork")
    ) == {
        "retiree_nomc_choose_canwork",
        "retiree_dimc_choose_canwork",
        "nongroup_nomc_choose_canwork",
        "nongroup_dimc_choose_canwork",
        "dead",
    }
    edges = model.graph.edges.solve["retiree_nomc_inelig_canwork"]
    assert (
        60 in edges["retiree_nomc_inelig_canwork"],
        61 in edges["retiree_nomc_inelig_canwork"],
    ) == (True, False)


@pytest.mark.parametrize("regime", list(REGIME_SPECS))
def test_each_regime_law_reaches_exactly_its_declared_targets(model, regime):
    """The law's age schedule names each target at the ages that can reach it."""
    assert dict(model.graph.edges.solve[regime]) == {
        target: frozenset(ages) for target, ages in build_model_edges()[regime].items()
    }


def test_bequest_continuation_is_not_removed(model):
    regime = model.user_regimes["retiree_oamc_forced_forcedout"]
    assert model.graph.solution.targets(
        period=44, source="retiree_oamc_forced_forcedout"
    ) == ("dead",)
    assert "consumption_dollars" in regime.actions
    assert (95, "retiree_oamc_forced_forcedout") in model.graph.nodes
    assert (96, "dead") in model.graph.nodes


def test_dead_regime_is_terminal_at_every_age(model):
    assert model.graph.laws["dead"].terminal
    assert {(age, "dead") for age in model.ages.exact_values[1:]} <= model.graph.nodes


def test_select_admissible_starts_drops_starts_at_61_and_above(model):
    ic = pd.DataFrame(
        {
            "age": [60.0, 61.0, 62.0],
            "regime_name": ["retiree_nomc_inelig_canwork"] * 2
            + ["retiree_nomc_choose_canwork"],
        },
        index=pd.Index([1, 2, 3], name="id"),
    )
    kept = select_admissible_starts(model=model, initial_conditions=ic)
    assert list(kept.index) == [1]


def test_medical_cost_draws_keep_living_states_without_choice_constraints(model):
    """Living regimes carry medical shocks; bequests depend only on assets and type."""
    for name, regime in model.user_regimes.items():
        assert "medical_cost_shocks_carried" not in regime.constraints
        if name == "dead":
            assert set(regime.states) == {"assets", "pref_type"}
        else:
            assert {"hcc_persistent", "hcc_transitory"} <= set(regime.states)
