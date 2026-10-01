"""Medicaid in the non-group regimes: last-resort payer and Medicare premia.

The tests evaluate the real non-group regime DAG with the upstream cost and
premium nodes pinned to constants, so they check how Medicaid eligibility
routes those quantities rather than how the quantities themselves are built.

- A Medicaid-eligible non-group household before Medicare is covered by
  Medicaid alone: it can neither hold private cover nor stay uninsured, so
  its OOP is Medicaid's cost-sharing on total costs and it pays no premium
  whatever its `buy_private` choice.
- Medicaid pays the Medicare premium of a Medicaid-eligible household.
"""

import inspect

import jax.numpy as jnp
import pytest
from dags import concatenate_functions

from aca_model.agent.labor_market import LaborSupply
from aca_model.baseline import health_insurance
from aca_model.baseline.health_insurance import BuyPrivate
from aca_model.baseline.regimes._common import REGIME_SPECS, build_model_functions
from aca_model.baseline.regimes._nongroup import _build_functions

TOTAL_COSTS = 20000.0
PRIVATE_OOP = 5285.0
PREMIUM_BEFORE_MEDICAID = 4620.0
MEDICAID_COINSURANCE = 0.0352

_MEDICAID_INPUTS = {
    "deductible_medicaid": jnp.asarray(0.0),
    "coinsurance_rate_medicaid": jnp.asarray(MEDICAID_COINSURANCE),
    "oop_max_medicaid": jnp.asarray(jnp.inf),
}


def _is_base_premium(func: object) -> bool:
    """Whether `func` is one of the baseline premium formulas."""
    return getattr(func, "__module__", "") == health_insurance.__name__ and getattr(
        func, "__name__", ""
    ).startswith("premium")


def _evaluate_nongroup(regime_name: str, buy_private: int) -> dict:
    """Evaluate `oop_costs` and `hic_premium` of a Medicaid-eligible household."""
    functions = {
        **build_model_functions(),
        **_build_functions(REGIME_SPECS[regime_name]),
    }
    functions = {
        name: (lambda: jnp.asarray(PREMIUM_BEFORE_MEDICAID))
        if _is_base_premium(func)
        else func
        for name, func in functions.items()
        if callable(func)
    }
    functions["total_health_costs"] = lambda: jnp.asarray(TOTAL_COSTS)
    functions["primary_oop"] = lambda: jnp.asarray(PRIVATE_OOP)
    functions["is_medicaid_eligible"] = lambda: jnp.asarray(True)
    combined = concatenate_functions(
        functions,
        targets=["oop_costs", "hic_premium"],
        return_type="dict",
    )
    inputs = {
        **_MEDICAID_INPUTS,
        "buy_private": jnp.asarray(buy_private),
        "labor_supply": jnp.asarray(LaborSupply.do_not_work),
    }
    needed = inspect.signature(combined).parameters
    return combined(**{k: v for k, v in inputs.items() if k in needed})


def test_medicaid_eligible_buyer_pays_medicaid_cost_sharing_on_total_costs() -> None:
    """With `buy_private=yes`, OOP is Medicaid's 3.52% of total costs: 704."""
    result = _evaluate_nongroup("nongroup_nomc_inelig_canwork", BuyPrivate.yes)
    assert jnp.isclose(result["oop_costs"], MEDICAID_COINSURANCE * TOTAL_COSTS)


def test_medicaid_eligible_buyer_pays_no_private_premium() -> None:
    """With `buy_private=yes`, a Medicaid-eligible household pays no premium."""
    result = _evaluate_nongroup("nongroup_nomc_inelig_canwork", BuyPrivate.yes)
    assert jnp.isclose(result["hic_premium"], 0.0)


@pytest.mark.parametrize(
    "regime_name",
    [
        "nongroup_dimc_inelig_canwork",
        "nongroup_oamc_choose_canwork",
        "nongroup_oamc_forced_forcedout",
    ],
)
def test_medicaid_pays_the_medicare_premium(regime_name: str) -> None:
    """A Medicaid-eligible household with Medicare pays no Medicare premium."""
    result = _evaluate_nongroup(regime_name, BuyPrivate.no)
    assert jnp.isclose(result["hic_premium"], 0.0)
