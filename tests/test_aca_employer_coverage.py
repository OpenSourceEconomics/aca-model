"""The ACA Medicaid expansion leaves employer-provided coverage untouched.

A retiree or tied household whose only Medicaid route is the ACA MAGI
expansion keeps its employer coverage next period. Only categorical (SSI)
eligibility moves an employer-covered household to non-group.
"""

from collections.abc import Callable

import jax.numpy as jnp
import pytest
from dags import concatenate_functions
from lcm.params import MappingLeaf

from aca_model.aca import health_insurance as aca_hi
from aca_model.agent.labor_market import LaborSupply
from aca_model.baseline import health_insurance
from aca_model.baseline.health_insurance import HealthInsuranceState
from aca_model.baseline.regimes._common import RegimeId, make_targets
from aca_model.baseline.regimes._retiree import (
    _make_transition_canwork as retiree_canwork,
)
from aca_model.baseline.regimes._tied import _make_transition_canwork as tied_canwork
from aca_model.config import MODEL_CONFIG

N_PERIODS = MODEL_CONFIG.end_age - MODEL_CONFIG.start_age
SURVIVAL = jnp.ones((N_PERIODS, 3)) * 0.99
MEDICAID_SCHEDULE = MappingLeaf(
    {"income_threshold": jnp.array([16104.6, 21707.4, 21707.4])}
)

# Under 65, not categorically eligible, MAGI below 138% FPL: eligible only
# through the expansion.
_EXPANSION_ONLY_INPUTS = {
    "is_ssi_eligible": jnp.array(False),
    "aca_magi": jnp.array(10000.0),
    "spousal_income": jnp.int32(0),
    "crossed_oamc_threshold": jnp.asarray(False),
    "medicaid_schedule": MEDICAID_SCHEDULE,
}


def _live_target(probs: jnp.ndarray) -> int:
    return int(jnp.argmax(probs.at[RegimeId.dead].set(0.0)))


def _compose_with_aca_eligibility(
    target_name: str, target_func: Callable, inputs: dict
) -> dict:
    """Evaluate `target_func` in a DAG that also carries ACA Medicaid eligibility."""
    combined = concatenate_functions(
        {"is_medicaid_eligible": aca_hi.is_medicaid_eligible, target_name: target_func},
        targets=[target_name, "is_medicaid_eligible"],
        return_type="dict",
    )
    return combined(**inputs)


@pytest.mark.parametrize(
    ("regime_name", "make_transition", "extra_inputs", "expected"),
    [
        (
            "retiree_nomc_inelig_canwork",
            retiree_canwork,
            {"labor_supply": jnp.array(LaborSupply.h2000)},
            RegimeId.retiree_nomc_inelig_canwork,
        ),
        (
            "tied_nomc_inelig_canwork",
            tied_canwork,
            {"labor_supply": jnp.array(LaborSupply.h2000)},
            RegimeId.tied_nomc_inelig_canwork,
        ),
    ],
)
def test_expansion_only_medicaid_keeps_employer_coverage(
    regime_name: str,
    make_transition: Callable,
    extra_inputs: dict,
    expected: int,
) -> None:
    """An employer-covered household eligible only via the expansion keeps it."""
    own, ng = make_targets(regime_name)
    transition = make_transition(own=own, ng=ng)
    result = _compose_with_aca_eligibility(
        "next_regime",
        transition,
        {
            **_EXPANSION_ONLY_INPUTS,
            **extra_inputs,
            "age": jnp.int32(55),
            "period": jnp.int32(55 - MODEL_CONFIG.start_age),
            "health": jnp.int32(2),
            "survival_probs": SURVIVAL,
            "prob_disabled_next": jnp.zeros((N_PERIODS, 3)),
        },
    )
    assert _live_target(result["next_regime"]) == expected


def test_expansion_only_medicaid_keeps_retiree_his_for_pension_imputation() -> None:
    """The pension imputation's target HIS stays retiree for an expansion-only case."""
    result = _compose_with_aca_eligibility(
        "target_his",
        health_insurance.target_his,
        {
            **_EXPANSION_ONLY_INPUTS,
            "his": jnp.int32(HealthInsuranceState.retiree),
            "labor_supply": jnp.array(LaborSupply.h2000),
        },
    )
    assert int(result["target_his"]) == HealthInsuranceState.retiree
