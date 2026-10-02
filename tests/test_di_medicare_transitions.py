"""Disability Medicare before 65: disabled and not working last year.

A pre-65 household is on disability Medicare next period exactly when it does
not work this period and is disabled next period. The regime transition splits
the surviving mass by next-period disability; each target's health law is the
health distribution conditional on landing there.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from aca_model.agent import health
from aca_model.agent.health import HealthWithDisability
from aca_model.agent.labor_market import LaborSupply
from aca_model.baseline.regimes._common import (
    REGIME_SPECS,
    RegimeId,
    _build_per_target_regime_health,
    build_scheduled_regime_transition,
    make_targets,
)
from aca_model.baseline.regimes._nongroup import (
    _make_transition_canwork as nongroup_canwork,
)
from aca_model.baseline.regimes._retiree import (
    _make_transition_canwork as retiree_canwork,
)
from aca_model.baseline.regimes._tied import _make_transition_canwork as tied_canwork
from aca_model.config import MODEL_CONFIG

N_PERIODS = MODEL_CONFIG.end_age - MODEL_CONFIG.start_age
SURVIVAL = jnp.full((N_PERIODS, 3), 0.99)
# Rows: current health (disabled, bad, good); columns: next health.
_HEALTH_ROWS = jnp.array([[0.98, 0.01, 0.01], [0.05, 0.75, 0.20], [0.002, 0.058, 0.94]])
HEALTH_TRANS = jnp.broadcast_to(_HEALTH_ROWS, (N_PERIODS, 3, 3))
# P(disabled next period | health), the first column of the rows above.
PROB_DISABLED = HEALTH_TRANS[:, :, 0]


def _nongroup_probs(regime: str, health_state: int, labor: int) -> jnp.ndarray:
    own, _ = make_targets(regime)
    return nongroup_canwork(own=own)(
        age=jnp.int32(55),
        period=jnp.int32(4),
        health=jnp.int32(health_state),
        labor_supply=jnp.array(labor),
        survival_probs=SURVIVAL,
        prob_disabled_next=PROB_DISABLED,
    )


@pytest.mark.parametrize(
    ("regime", "health_state", "labor", "target", "expected"),
    [
        # Bad health, not working: disabled next year with prob 0.05.
        (
            "nongroup_nomc_inelig_canwork",
            HealthWithDisability.bad,
            LaborSupply.do_not_work,
            "nongroup_dimc_inelig_canwork",
            0.99 * 0.05,
        ),
        (
            "nongroup_nomc_inelig_canwork",
            HealthWithDisability.bad,
            LaborSupply.do_not_work,
            "nongroup_nomc_inelig_canwork",
            0.99 * 0.95,
        ),
        # Working: no disability Medicare next year.
        (
            "nongroup_nomc_inelig_canwork",
            HealthWithDisability.bad,
            LaborSupply.h2000,
            "nongroup_dimc_inelig_canwork",
            0.0,
        ),
        # On disability Medicare and recovering: leaves it.
        (
            "nongroup_dimc_inelig_canwork",
            HealthWithDisability.disabled,
            LaborSupply.do_not_work,
            "nongroup_nomc_inelig_canwork",
            0.99 * 0.02,
        ),
        (
            "nongroup_dimc_inelig_canwork",
            HealthWithDisability.disabled,
            LaborSupply.do_not_work,
            "nongroup_dimc_inelig_canwork",
            0.99 * 0.98,
        ),
        # On disability Medicare and working: leaves it.
        (
            "nongroup_dimc_inelig_canwork",
            HealthWithDisability.disabled,
            LaborSupply.h2000,
            "nongroup_nomc_inelig_canwork",
            0.99,
        ),
    ],
)
def test_nongroup_transition_di_medicare_entry_and_exit(
    regime: str, health_state: int, labor: int, target: str, expected: float
) -> None:
    """Probability of each target splits survival by next-period disability."""
    probs = _nongroup_probs(regime, health_state, labor)
    np.testing.assert_allclose(probs[getattr(RegimeId, target)], expected, atol=1e-12)


def test_retiree_transition_newly_disabled_non_worker_keeps_retiree_cover() -> None:
    """A retiree who stops working and becomes disabled holds EPHI-Medicare."""
    own, ng = make_targets("retiree_nomc_inelig_canwork")
    probs = retiree_canwork(own=own, ng=ng)(
        age=jnp.int32(55),
        period=jnp.int32(4),
        health=jnp.int32(HealthWithDisability.good),
        labor_supply=jnp.array(LaborSupply.do_not_work),
        is_ssi_eligible=jnp.array(False),
        survival_probs=SURVIVAL,
        prob_disabled_next=PROB_DISABLED,
    )
    np.testing.assert_allclose(
        probs[RegimeId.retiree_dimc_inelig_canwork], 0.99 * 0.002, atol=1e-12
    )


def test_tied_transition_newly_disabled_non_worker_moves_to_nongroup_medicare() -> None:
    """A tied worker who stops and becomes disabled moves to non-group Medicare."""
    own, ng = make_targets("tied_nomc_inelig_canwork")
    probs = tied_canwork(own=own, ng=ng)(
        age=jnp.int32(55),
        period=jnp.int32(4),
        health=jnp.int32(HealthWithDisability.bad),
        labor_supply=jnp.array(LaborSupply.do_not_work),
        is_ssi_eligible=jnp.array(False),
        survival_probs=SURVIVAL,
        prob_disabled_next=PROB_DISABLED,
    )
    np.testing.assert_allclose(
        probs[RegimeId.nongroup_dimc_inelig_canwork], 0.99 * 0.05, atol=1e-12
    )


def test_next_health_into_dimc_is_disabled() -> None:
    """Landing on disability Medicare means being disabled."""
    result = health.next_health_into_dimc(
        health=jnp.int32(HealthWithDisability.bad),
        period=jnp.int32(4),
        health_trans_probs=HEALTH_TRANS,
    )
    np.testing.assert_allclose(result, [1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("labor", "expected"),
    [
        (LaborSupply.do_not_work, [0.0, 0.75 / 0.95, 0.20 / 0.95]),
        (LaborSupply.h2000, [0.05, 0.75, 0.20]),
    ],
)
def test_next_health_into_nomc_excludes_disability_for_non_workers(
    labor: int, expected: list[float]
) -> None:
    """A non-worker without disability Medicare next year is not disabled then."""
    result = health.next_health_into_nomc(
        health=jnp.int32(HealthWithDisability.bad),
        period=jnp.int32(4),
        labor_supply=jnp.array(labor),
        health_trans_probs=HEALTH_TRANS,
    )
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize(
    ("source", "target", "law"),
    [
        (
            "nongroup_nomc_inelig_canwork",
            "nongroup_dimc_inelig_canwork",
            "next_health_into_dimc",
        ),
        (
            "nongroup_nomc_inelig_canwork",
            "nongroup_nomc_inelig_canwork",
            "next_health_into_nomc",
        ),
        (
            "retiree_dimc_choose_canwork",
            "retiree_nomc_choose_canwork",
            "next_health_into_nomc",
        ),
        (
            "retiree_dimc_choose_canwork",
            "retiree_oamc_choose_canwork",
            "next_health_cross",
        ),
        (
            "nongroup_oamc_choose_canwork",
            "nongroup_oamc_forced_canwork",
            "next_health",
        ),
    ],
)
def test_per_target_health_law(source: str, target: str, law: str) -> None:
    """Pre-65 targets condition next health on the Medicare outcome."""
    laws = _build_per_target_regime_health(REGIME_SPECS[source])
    assert laws[target].func is getattr(health, law)


@pytest.mark.parametrize(
    ("regime", "make_transition", "dimc_target"),
    [
        (
            "nongroup_nomc_inelig_canwork",
            nongroup_canwork,
            "nongroup_dimc_inelig_canwork",
        ),
        ("retiree_nomc_inelig_canwork", retiree_canwork, "retiree_dimc_inelig_canwork"),
        ("tied_nomc_inelig_canwork", tied_canwork, "nongroup_dimc_inelig_canwork"),
    ],
)
def test_scheduled_targets_include_disability_medicare_entry(
    regime: str, make_transition: object, dimc_target: str
) -> None:
    """A pre-65 regime without Medicare declares its disability-Medicare target."""
    own, ng = make_targets(regime)
    transition_func = (
        make_transition(own=own)
        if make_transition is nongroup_canwork
        else make_transition(own=own, ng=ng)
    )
    schedule = build_scheduled_regime_transition(
        spec=REGIME_SPECS[regime],
        transition_func=transition_func,
        target_groups=(own,) if make_transition is nongroup_canwork else (own, ng),
    )
    targets_at_55 = next(
        targets
        for ages, targets in schedule._cases  # noqa: SLF001
        if 55 in ages
    )
    assert dimc_target in targets_at_55
