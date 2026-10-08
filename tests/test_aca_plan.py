"""The ACA private non-group plan is community-rated.

Under every ACA variant with the reformed non-group market, the premium of
a private non-group policy before premium credits no longer depends on the
household's expected medical costs: it is a flat amount per household
size. The tests evaluate the real non-group regime DAG after the ACA
overlay, with the baseline risk-rated premium pinned to a constant, so they
check which premium the overlay wires in.
"""

import inspect

import jax.numpy as jnp
import pytest
from dags import concatenate_functions

from aca_model.aca.health_insurance import PolicyVariant
from aca_model.aca.regimes._overrides import apply_aca_overrides
from aca_model.agent.labor_market import LaborSupply
from aca_model.baseline import health_insurance
from aca_model.baseline.health_insurance import BuyPrivate
from aca_model.baseline.regimes._common import REGIME_SPECS, build_model_functions
from aca_model.baseline.regimes._nongroup import _build_functions

RISK_RATED_PREMIUM = 4620.0
ACA_PREMIUM_SINGLE = 6000.0
ACA_PREMIUM_MARRIED_EXTRA = 3000.0
REGIME = "nongroup_nomc_inelig_canwork"


def _is_base_premium(func: object) -> bool:
    """Whether `func` is one of the baseline premium formulas."""
    return getattr(func, "__module__", "") == health_insurance.__name__ and getattr(
        func, "__name__", ""
    ).startswith("premium")


def _aca_premium(policy: PolicyVariant, spousal_income: int) -> float:
    """Evaluate `hic_premium` of a non-Medicaid household buying private cover."""
    spec = REGIME_SPECS[REGIME]
    functions = {**build_model_functions(), **_build_functions(spec)}
    functions = {
        name: (lambda: jnp.asarray(RISK_RATED_PREMIUM))
        if _is_base_premium(func)
        else func
        for name, func in functions.items()
        if callable(func)
    }
    apply_aca_overrides(functions, spec, policy)
    functions["is_medicaid_eligible"] = lambda: jnp.asarray(False)
    combined = concatenate_functions(
        functions, targets=["hic_premium"], return_type="tuple"
    )
    inputs = {
        "buy_private": jnp.asarray(BuyPrivate.yes),
        "labor_supply": jnp.asarray(LaborSupply.do_not_work),
        "spousal_income": jnp.int32(spousal_income),
        "aca_premium_single": jnp.asarray(ACA_PREMIUM_SINGLE),
        "aca_premium_married_extra": jnp.asarray(ACA_PREMIUM_MARRIED_EXTRA),
    }
    needed = inspect.signature(combined).parameters
    return float(combined(**{k: v for k, v in inputs.items() if k in needed})[0])


@pytest.mark.parametrize(
    ("policy", "spousal_income", "expected"),
    [
        (PolicyVariant.ACA, 0, ACA_PREMIUM_SINGLE),
        (PolicyVariant.ACA, 2, ACA_PREMIUM_SINGLE + ACA_PREMIUM_MARRIED_EXTRA),
        (PolicyVariant.ACA_NO_MANDATE, 1, 9000.0),
        (PolicyVariant.ACA_NO_MEDICAID_EXPANSION, 0, 6000.0),
        (PolicyVariant.ACA_NO_MEDICAID_EXPANSION_NO_MANDATE, 2, 9000.0),
        (PolicyVariant.ACA_ONLY_MEDICAID_EXPANSION, 0, RISK_RATED_PREMIUM),
    ],
)
def test_aca_non_group_premium_is_community_rated(
    policy: PolicyVariant, spousal_income: int, expected: float
) -> None:
    """The ACA plan costs 6000 for a single and 9000 for a married household.

    The variant without the non-group reform keeps the risk-rated premium.
    """
    assert _aca_premium(policy, spousal_income) == pytest.approx(expected)


RISK_RATED_PREMIUM_PARAMS = frozenset(
    {
        "premium_markup",
        "premium_minimum",
        "premium_predicted_hcc",
        "premium_sample_insurer_cost",
        "premium_sample_is_married",
    }
)


def _free_arguments(policy: PolicyVariant, regime: str) -> frozenset[str]:
    """Arguments of a non-group regime's DAG that no function of it produces."""
    spec = REGIME_SPECS[regime]
    functions = {**build_model_functions(), **_build_functions(spec)}
    functions = {name: func for name, func in functions.items() if callable(func)}
    apply_aca_overrides(functions, spec, policy)
    arguments = {
        name
        for func in functions.values()
        for name in inspect.signature(func).parameters
    }
    return frozenset(arguments - functions.keys())


@pytest.mark.parametrize(
    "regime", ["nongroup_nomc_inelig_canwork", "nongroup_nomc_choose_canwork"]
)
@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (PolicyVariant.ACA, frozenset()),
        (PolicyVariant.ACA_NO_MANDATE, frozenset()),
        (PolicyVariant.ACA_NO_MEDICAID_EXPANSION, frozenset()),
        (PolicyVariant.ACA_NO_MEDICAID_EXPANSION_NO_MANDATE, frozenset()),
        (PolicyVariant.ACA_ONLY_MEDICAID_EXPANSION, RISK_RATED_PREMIUM_PARAMS),
    ],
)
def test_aca_non_group_regime_reads_risk_rated_premium_params_only_without_reform(
    policy: PolicyVariant, regime: str, expected: frozenset[str]
) -> None:
    """A community-rated non-group market reads none of the risk-rated inputs.

    The variant without the non-group reform still prices private cover by
    risk and reads all of them.
    """
    assert _free_arguments(policy, regime) & RISK_RATED_PREMIUM_PARAMS == expected
