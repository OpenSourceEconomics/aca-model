"""Integration tests for regime transition functions.

Test that transition functions produce correct target regime IDs based on
labor supply, Medicaid eligibility, and age brackets.
"""

import jax.numpy as jnp
import pytest

from aca_model.agent.labor_market import LaborSupply
from aca_model.baseline.regimes._common import RegimeId, make_targets
from aca_model.baseline.regimes._nongroup import (
    _make_transition_canwork as nongroup_canwork,
)
from aca_model.baseline.regimes._nongroup import (
    _make_transition_forcedout as nongroup_forcedout,
)
from aca_model.baseline.regimes._retiree import (
    _make_transition_canwork as retiree_canwork,
)
from aca_model.baseline.regimes._retiree import (
    _make_transition_forcedout as retiree_forcedout,
)
from aca_model.baseline.regimes._tied import _make_transition_canwork as tied_canwork
from aca_model.config import MODEL_CONFIG

N_REGIMES = 19
N_PERIODS = MODEL_CONFIG.end_age - MODEL_CONFIG.start_age
SURVIVAL = jnp.ones(N_PERIODS) * 0.99


def _target_from_probs(probs: jnp.ndarray) -> int:
    """Extract the non-dead target regime from a probability vector."""
    # Zero out the dead regime, find the one with weight
    live = probs.at[RegimeId.dead].set(0.0)
    return int(jnp.argmax(live))


# --- Tied: stop working → nongroup ---


def test_tied_stop_working_becomes_nongroup() -> None:
    """Tied agent who stops working loses employer coverage → nongroup."""
    own, ng = make_targets("tied_nomc_inelig_canwork")
    transition = tied_canwork(gets_medicare=False, own=own, ng=ng)

    probs = transition(
        age=55,
        period=4,
        labor_supply=jnp.array(LaborSupply.do_not_work),
        is_medicaid_eligible=jnp.array(False),
        survival_probs=SURVIVAL,
    )
    target = _target_from_probs(probs)
    # Should be nongroup (not tied) since agent stopped working
    assert target == RegimeId.nongroup_nomc_inelig_canwork


def test_tied_keeps_working_stays_tied() -> None:
    """Tied agent who keeps working retains employer coverage."""
    own, ng = make_targets("tied_nomc_inelig_canwork")
    transition = tied_canwork(gets_medicare=False, own=own, ng=ng)

    probs = transition(
        age=55,
        period=4,
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(False),
        survival_probs=SURVIVAL,
    )
    target = _target_from_probs(probs)
    assert target == RegimeId.tied_nomc_inelig_canwork


# --- Medicaid override → nongroup ---


def test_retiree_medicaid_override_to_nongroup() -> None:
    """Medicaid-eligible retiree is overridden to nongroup."""
    own, ng = make_targets("retiree_nomc_inelig_canwork")
    transition = retiree_canwork(gets_medicare=False, own=own, ng=ng)

    probs = transition(
        age=55,
        period=4,
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(True),
        survival_probs=SURVIVAL,
    )
    target = _target_from_probs(probs)
    assert target == RegimeId.nongroup_nomc_inelig_canwork


def test_retiree_not_medicaid_stays_retiree() -> None:
    """Non-Medicaid retiree stays retiree."""
    own, ng = make_targets("retiree_nomc_inelig_canwork")
    transition = retiree_canwork(gets_medicare=False, own=own, ng=ng)

    probs = transition(
        age=55,
        period=4,
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(False),
        survival_probs=SURVIVAL,
    )
    target = _target_from_probs(probs)
    assert target == RegimeId.retiree_nomc_inelig_canwork


def test_retiree_forcedout_medicaid_override() -> None:
    """Forcedout retiree with Medicaid → nongroup."""
    own, ng = make_targets("retiree_oamc_forced_forcedout")
    transition = retiree_forcedout(gets_medicare=True, own=own, ng=ng)

    probs = transition(
        age=80,
        period=29,
        is_medicaid_eligible=jnp.array(True),
        survival_probs=SURVIVAL,
    )
    target = _target_from_probs(probs)
    assert target == RegimeId.nongroup_oamc_forced_forcedout


# --- Age bracket transitions ---


@pytest.mark.parametrize(
    ("age", "expected_target"),
    [
        # age 55 → next_age 56: still inelig (< 62)
        (55.0, RegimeId.retiree_nomc_inelig_canwork),
        # age 61 → next_age 62: ss becomes choose
        (61.0, RegimeId.retiree_nomc_choose_canwork),
        # age 64 → next_age 65: mc becomes oamc
        (64.0, RegimeId.retiree_oamc_choose_canwork),
        # age 69 → next_age 70: ss becomes forced
        (69.0, RegimeId.retiree_oamc_forced_canwork),
        # age 71 → next_age 72: work becomes forcedout
        (71.0, RegimeId.retiree_oamc_forced_forcedout),
    ],
)
def test_retiree_age_bracket_transitions(
    age: float,
    expected_target: int,
) -> None:
    """Retiree transitions to correct regime at age boundaries."""
    # Use nomc+inelig as starting point — the transition function resolves
    # the target based on next_age, not current spec.
    own, ng = make_targets("retiree_nomc_inelig_canwork")
    transition = retiree_canwork(gets_medicare=False, own=own, ng=ng)

    period = int(age - MODEL_CONFIG.start_age)
    probs = transition(
        age=age,
        period=period,
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(False),
        survival_probs=SURVIVAL,
    )
    target = _target_from_probs(probs)
    assert target == expected_target


# --- Nongroup: valid probabilities, no Medicaid param ---


def test_nongroup_canwork_valid_probs() -> None:
    """Nongroup canwork produces valid probability vector."""
    own, _ng = make_targets("nongroup_nomc_inelig_canwork")
    transition = nongroup_canwork(gets_medicare=False, own=own)

    probs = transition(
        age=55,
        period=4,
        labor_supply=jnp.array(LaborSupply.h2000),
        survival_probs=SURVIVAL,
    )
    assert jnp.isclose(jnp.sum(probs), 1.0, atol=1e-6)
    assert jnp.all(probs >= 0.0)


def test_nongroup_forcedout_valid_probs() -> None:
    """Nongroup forcedout produces valid probability vector."""
    own, _ng = make_targets("nongroup_oamc_forced_forcedout")
    transition = nongroup_forcedout(gets_medicare=True, own=own)

    probs = transition(
        age=80,
        period=29,
        survival_probs=SURVIVAL,
    )
    assert jnp.isclose(jnp.sum(probs), 1.0, atol=1e-6)
    assert jnp.all(probs >= 0.0)


# --- Survival probability → dead weight ---


def test_tied_medicaid_override_to_nongroup() -> None:
    """Tied + Medicaid-eligible → nongroup (SSI override)."""
    own, ng = make_targets("tied_nomc_inelig_canwork")
    transition = tied_canwork(gets_medicare=False, own=own, ng=ng)

    probs = transition(
        age=55,
        period=4,
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(True),
        survival_probs=SURVIVAL,
    )
    target = _target_from_probs(probs)
    assert target == RegimeId.nongroup_nomc_inelig_canwork


def test_tied_at_medicare_age_with_medicaid() -> None:
    """Tied at age 64→65 (Medicare onset) + Medicaid → nongroup+oamc."""
    own, ng = make_targets("tied_nomc_choose_canwork")
    transition = tied_canwork(gets_medicare=False, own=own, ng=ng)

    period = int(64 - MODEL_CONFIG.start_age)
    probs = transition(
        age=64,
        period=period,
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(True),
        survival_probs=SURVIVAL,
    )
    target = _target_from_probs(probs)
    # At 65: mc→oamc, Medicaid override → nongroup
    assert target == RegimeId.nongroup_oamc_choose_canwork


def test_survival_prob_determines_death_weight() -> None:
    """Dead regime gets (1 - survival) probability weight."""
    own, ng = make_targets("retiree_nomc_inelig_canwork")
    transition = retiree_canwork(gets_medicare=False, own=own, ng=ng)

    survival = jnp.ones(N_PERIODS) * 0.85
    probs = transition(
        age=55,
        period=4,
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(False),
        survival_probs=survival,
    )
    assert jnp.isclose(probs[RegimeId.dead], 0.15, atol=1e-6)
    # Living target gets 0.85
    live_probs = probs.at[RegimeId.dead].set(0.0)
    assert jnp.isclose(jnp.sum(live_probs), 0.85, atol=1e-6)
