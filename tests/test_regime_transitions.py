"""Integration tests for regime transition functions.

Test that transition functions produce correct target regime IDs based on
labor supply, Medicaid eligibility, age brackets, and the marital lottery.
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

N_REGIMES = 37
N_PERIODS = MODEL_CONFIG.end_age - MODEL_CONFIG.start_age
SURVIVAL = jnp.ones(N_PERIODS) * 0.99

# Degenerate marital lotteries, so a test that is about the HIS / Medicare / SS
# axes leaves exactly one living target with weight.
STAYS_SINGLE = jnp.array([1.0, 0.0])
STAYS_MARRIED = jnp.array([0.0, 1.0])


def _target_from_probs(probs: jnp.ndarray) -> int:
    """Extract the single living target of a degenerate marital lottery."""
    live = probs.at[RegimeId.dead].set(0.0)
    return int(jnp.argmax(live))


# --- Tied: stop working → nongroup ---


def test_tied_stop_working_becomes_nongroup() -> None:
    """Tied agent who stops working loses employer coverage → nongroup."""
    own, ng = make_targets("single_tied_nomc_inelig_canwork")
    transition = tied_canwork(gets_medicare=False, own=own, ng=ng)

    probs = transition(
        age=jnp.int32(55),
        period=jnp.int32(4),
        labor_supply=jnp.array(LaborSupply.do_not_work),
        is_medicaid_eligible=jnp.array(False),
        survival_probs=SURVIVAL,
        marital_probs=STAYS_SINGLE,
    )
    target = _target_from_probs(probs)
    # Should be nongroup (not tied) since agent stopped working
    assert target == RegimeId.single_nongroup_nomc_inelig_canwork


def test_tied_keeps_working_stays_tied() -> None:
    """Tied agent who keeps working retains employer coverage."""
    own, ng = make_targets("single_tied_nomc_inelig_canwork")
    transition = tied_canwork(gets_medicare=False, own=own, ng=ng)

    probs = transition(
        age=jnp.int32(55),
        period=jnp.int32(4),
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(False),
        survival_probs=SURVIVAL,
        marital_probs=STAYS_SINGLE,
    )
    target = _target_from_probs(probs)
    assert target == RegimeId.single_tied_nomc_inelig_canwork


# --- Medicaid override → nongroup ---


def test_retiree_medicaid_override_to_nongroup() -> None:
    """Medicaid-eligible retiree is overridden to nongroup."""
    own, ng = make_targets("single_retiree_nomc_inelig_canwork")
    transition = retiree_canwork(gets_medicare=False, own=own, ng=ng)

    probs = transition(
        age=jnp.int32(55),
        period=jnp.int32(4),
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(True),
        survival_probs=SURVIVAL,
        marital_probs=STAYS_SINGLE,
    )
    target = _target_from_probs(probs)
    assert target == RegimeId.single_nongroup_nomc_inelig_canwork


def test_retiree_not_medicaid_stays_retiree() -> None:
    """Non-Medicaid retiree stays retiree."""
    own, ng = make_targets("single_retiree_nomc_inelig_canwork")
    transition = retiree_canwork(gets_medicare=False, own=own, ng=ng)

    probs = transition(
        age=jnp.int32(55),
        period=jnp.int32(4),
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(False),
        survival_probs=SURVIVAL,
        marital_probs=STAYS_SINGLE,
    )
    target = _target_from_probs(probs)
    assert target == RegimeId.single_retiree_nomc_inelig_canwork


def test_retiree_forcedout_medicaid_override() -> None:
    """Forcedout retiree with Medicaid → nongroup."""
    own, ng = make_targets("single_retiree_oamc_forced_forcedout")
    transition = retiree_forcedout(gets_medicare=True, own=own, ng=ng)

    probs = transition(
        age=jnp.int32(80),
        period=jnp.int32(29),
        is_medicaid_eligible=jnp.array(True),
        survival_probs=SURVIVAL,
        marital_probs=STAYS_SINGLE,
    )
    target = _target_from_probs(probs)
    assert target == RegimeId.single_nongroup_oamc_forced_forcedout


# --- Marital status is an axis of the target regime ---


def test_marriage_moves_a_single_agent_to_the_married_copy() -> None:
    """A single agent who marries lands in the married copy of its target."""
    own, ng = make_targets("single_retiree_nomc_inelig_canwork")
    transition = retiree_canwork(gets_medicare=False, own=own, ng=ng)

    probs = transition(
        age=jnp.int32(55),
        period=jnp.int32(4),
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(False),
        survival_probs=SURVIVAL,
        marital_probs=STAYS_MARRIED,
    )
    target = _target_from_probs(probs)
    assert target == RegimeId.married_retiree_nomc_inelig_canwork


def test_marital_lottery_splits_survival_mass_across_both_copies() -> None:
    """The marital lottery scales the surviving mass across both target copies."""
    own, ng = make_targets("married_retiree_nomc_inelig_canwork")
    transition = retiree_canwork(gets_medicare=False, own=own, ng=ng)

    probs = transition(
        age=jnp.int32(55),
        period=jnp.int32(4),
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(False),
        survival_probs=jnp.ones(N_PERIODS) * 0.8,
        marital_probs=jnp.array([0.25, 0.75]),
    )
    expected = jnp.zeros(N_REGIMES)
    expected = expected.at[RegimeId.dead].set(0.2)
    expected = expected.at[RegimeId.single_retiree_nomc_inelig_canwork].set(0.2)
    expected = expected.at[RegimeId.married_retiree_nomc_inelig_canwork].set(0.6)
    assert jnp.allclose(probs, expected, atol=1e-6)


# --- Age bracket transitions ---


@pytest.mark.parametrize(
    ("age", "expected_target"),
    [
        # age 55 → next_age 56: still inelig (< 62)
        (55.0, RegimeId.single_retiree_nomc_inelig_canwork),
        # age 61 → next_age 62: ss becomes choose
        (61.0, RegimeId.single_retiree_nomc_choose_canwork),
        # age 64 → next_age 65: mc becomes oamc
        (64.0, RegimeId.single_retiree_oamc_choose_canwork),
        # age 69 → next_age 70: ss becomes forced
        (69.0, RegimeId.single_retiree_oamc_forced_canwork),
        # age 71 → next_age 72: work becomes forcedout
        (71.0, RegimeId.single_retiree_oamc_forced_forcedout),
    ],
)
def test_retiree_age_bracket_transitions(
    age: float,
    expected_target: int,
) -> None:
    """Retiree transitions to correct regime at age boundaries."""
    # Use nomc+inelig as starting point — the transition function resolves
    # the target based on next_age, not current spec.
    own, ng = make_targets("single_retiree_nomc_inelig_canwork")
    transition = retiree_canwork(gets_medicare=False, own=own, ng=ng)

    period = jnp.int32(age - MODEL_CONFIG.start_age)
    probs = transition(
        age=jnp.asarray(age),
        period=period,
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(False),
        survival_probs=SURVIVAL,
        marital_probs=STAYS_SINGLE,
    )
    target = _target_from_probs(probs)
    assert target == expected_target


# --- Nongroup: valid probabilities, no Medicaid param ---


def test_nongroup_canwork_valid_probs() -> None:
    """Nongroup canwork produces valid probability vector."""
    own, _ng = make_targets("single_nongroup_nomc_inelig_canwork")
    transition = nongroup_canwork(gets_medicare=False, own=own)

    probs = transition(
        age=jnp.int32(55),
        period=jnp.int32(4),
        labor_supply=jnp.array(LaborSupply.h2000),
        survival_probs=SURVIVAL,
        marital_probs=jnp.array([0.4, 0.6]),
    )
    assert jnp.isclose(jnp.sum(probs), 1.0, atol=1e-6)
    assert jnp.all(probs >= 0.0)


def test_nongroup_forcedout_valid_probs() -> None:
    """Nongroup forcedout produces valid probability vector."""
    own, _ng = make_targets("single_nongroup_oamc_forced_forcedout")
    transition = nongroup_forcedout(gets_medicare=True, own=own)

    probs = transition(
        age=jnp.int32(80),
        period=jnp.int32(29),
        survival_probs=SURVIVAL,
        marital_probs=jnp.array([0.4, 0.6]),
    )
    assert jnp.isclose(jnp.sum(probs), 1.0, atol=1e-6)
    assert jnp.all(probs >= 0.0)


# --- Survival probability → dead weight ---


def test_tied_medicaid_override_to_nongroup() -> None:
    """Tied + Medicaid-eligible → nongroup (SSI override)."""
    own, ng = make_targets("single_tied_nomc_inelig_canwork")
    transition = tied_canwork(gets_medicare=False, own=own, ng=ng)

    probs = transition(
        age=jnp.int32(55),
        period=jnp.int32(4),
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(True),
        survival_probs=SURVIVAL,
        marital_probs=STAYS_SINGLE,
    )
    target = _target_from_probs(probs)
    assert target == RegimeId.single_nongroup_nomc_inelig_canwork


def test_tied_at_medicare_age_with_medicaid() -> None:
    """Tied at age 64→65 (Medicare onset) + Medicaid → nongroup+oamc."""
    own, ng = make_targets("single_tied_nomc_choose_canwork")
    transition = tied_canwork(gets_medicare=False, own=own, ng=ng)

    period = jnp.int32(64 - MODEL_CONFIG.start_age)
    probs = transition(
        age=jnp.int32(64),
        period=period,
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(True),
        survival_probs=SURVIVAL,
        marital_probs=STAYS_SINGLE,
    )
    target = _target_from_probs(probs)
    # At 65: mc→oamc, Medicaid override → nongroup
    assert target == RegimeId.single_nongroup_oamc_choose_canwork


def test_survival_prob_determines_death_weight() -> None:
    """Dead regime gets (1 - survival) probability weight."""
    own, ng = make_targets("single_retiree_nomc_inelig_canwork")
    transition = retiree_canwork(gets_medicare=False, own=own, ng=ng)

    survival = jnp.ones(N_PERIODS) * 0.85
    probs = transition(
        age=jnp.int32(55),
        period=jnp.int32(4),
        labor_supply=jnp.array(LaborSupply.h2000),
        is_medicaid_eligible=jnp.array(False),
        survival_probs=survival,
        marital_probs=STAYS_SINGLE,
    )
    assert jnp.isclose(probs[RegimeId.dead], 0.15, atol=1e-6)
    # Living target gets 0.85
    live_probs = probs.at[RegimeId.dead].set(0.0)
    assert jnp.isclose(jnp.sum(live_probs), 0.85, atol=1e-6)
