"""Tests for health-dependent mortality and the terminal age."""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from aca_model.agent.labor_market import LaborSupply
from aca_model.baseline.regimes._common import (
    REGIME_SPECS,
    RegimeId,
    make_active_func,
    make_targets,
    select_target_for_age,
)
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

N_PERIODS = MODEL_CONFIG.end_age - MODEL_CONFIG.start_age
# Survival by (period, health): distinct per health state.
SURVIVAL = jnp.tile(jnp.array([0.90, 0.95, 0.99]), (N_PERIODS, 1))
# P(disabled next period | health); inert for the working households here.
PROB_DISABLED = jnp.zeros((N_PERIODS, 3))


def _tied(health: int) -> jnp.ndarray:
    own, ng = make_targets("tied_nomc_inelig_canwork")
    return tied_canwork(own=own, ng=ng)(
        age=jnp.int32(55),
        period=jnp.int32(4),
        health=jnp.int32(health),
        labor_supply=jnp.array(LaborSupply.h2000),
        prob_disabled_next=PROB_DISABLED,
        is_ssi_eligible=jnp.array(False),
        survival_probs=SURVIVAL,
    )


def _retiree(health: int) -> jnp.ndarray:
    own, ng = make_targets("retiree_nomc_inelig_canwork")
    return retiree_canwork(own=own, ng=ng)(
        age=jnp.int32(55),
        period=jnp.int32(4),
        health=jnp.int32(health),
        labor_supply=jnp.array(LaborSupply.h2000),
        prob_disabled_next=PROB_DISABLED,
        is_ssi_eligible=jnp.array(False),
        survival_probs=SURVIVAL,
    )


def _retiree_forcedout(health: int) -> jnp.ndarray:
    own, ng = make_targets("retiree_oamc_forced_forcedout")
    return retiree_forcedout(gets_medicare=True, own=own, ng=ng)(
        age=jnp.int32(80),
        period=jnp.int32(29),
        health=jnp.int32(health),
        is_ssi_eligible=jnp.array(False),
        survival_probs=SURVIVAL,
    )


def _nongroup(health: int) -> jnp.ndarray:
    own, _ng = make_targets("nongroup_nomc_inelig_canwork")
    return nongroup_canwork(own=own)(
        age=jnp.int32(55),
        period=jnp.int32(4),
        health=jnp.int32(health),
        labor_supply=jnp.array(LaborSupply.h2000),
        prob_disabled_next=PROB_DISABLED,
        survival_probs=SURVIVAL,
    )


def _nongroup_forcedout(health: int) -> jnp.ndarray:
    own, _ng = make_targets("nongroup_oamc_forced_forcedout")
    return nongroup_forcedout(gets_medicare=True, own=own)(
        age=jnp.int32(80),
        period=jnp.int32(29),
        health=jnp.int32(health),
        survival_probs=SURVIVAL,
    )


@pytest.mark.parametrize(
    "transition",
    [_tied, _retiree, _retiree_forcedout, _nongroup, _nongroup_forcedout],
)
@pytest.mark.parametrize(("health", "expected_death"), [(0, 0.10), (1, 0.05)])
def test_death_probability_is_one_minus_survival_at_current_health(
    transition: Callable[[int], Any], health: int, expected_death: float
) -> None:
    """The dead regime gets `1 - survival_probs[period, health]`."""
    probs = transition(health)
    np.testing.assert_allclose(probs[RegimeId.dead], expected_death, atol=1e-12)


def test_survivor_at_94_moves_to_forcedout_regime_at_95() -> None:
    """95 is an age at which people can be alive."""
    own, _ng = make_targets("nongroup_oamc_forced_forcedout")
    target = select_target_for_age(95, True, own)
    assert int(target) == int(RegimeId.nongroup_oamc_forced_forcedout)


def test_everybody_is_dead_at_96() -> None:
    """Nobody is alive at `end_age` (96)."""
    own, _ng = make_targets("nongroup_oamc_forced_forcedout")
    target = select_target_for_age(MODEL_CONFIG.end_age, True, own)
    assert int(target) == int(RegimeId.dead)


@pytest.mark.parametrize(
    "regime", ["nongroup_oamc_forced_forcedout", "retiree_oamc_forced_forcedout"]
)
def test_forcedout_regimes_are_active_at_95(regime: str) -> None:
    """The work-forced-out regimes are active through the last alive age, 95."""
    assert bool(make_active_func(REGIME_SPECS[regime])(95))
