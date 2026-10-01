"""The ACA overlay charges the ACA surtaxes in every regime's after-tax income.

- Every ACA variant charges the 3.8% surtax on unearned income above 200,000.
- The full ACA also charges the 0.9% surtax on earnings above 200,000.

The tests evaluate the real regime DAG after the ACA overlay, with income
before the surtaxes pinned to a constant, so they check the wiring rather
than the tax schedule.
"""

import inspect

import jax.numpy as jnp
import pytest
from dags import concatenate_functions

from aca_model.aca.health_insurance import PolicyVariant
from aca_model.aca.regimes._overrides import apply_aca_overrides
from aca_model.baseline.regimes import REGIME_SPECS
from aca_model.baseline.regimes._common import build_model_functions
from aca_model.baseline.regimes._nongroup import _build_functions as nongroup_functions
from aca_model.baseline.regimes._retiree import _build_functions as retiree_functions
from aca_model.baseline.regimes._tied import _build_functions as tied_functions
from aca_model.environment import taxes

INCOME_BEFORE_SURTAXES = 100_000.0


def _after_tax_income(
    *,
    regime_name: str,
    policy: PolicyVariant,
    labor_income: float,
    unearned_income: float,
) -> float:
    functions = {
        **build_model_functions(),
        **_regime_functions(regime_name),
    }
    apply_aca_overrides(functions, REGIME_SPECS[regime_name], policy)
    functions = {
        name: (lambda: jnp.asarray(INCOME_BEFORE_SURTAXES))
        if func is taxes.after_tax_income
        else func
        for name, func in functions.items()
        if callable(func)
    }
    functions["labor_income"] = lambda: jnp.asarray(labor_income)
    functions["gross_income"] = lambda: jnp.asarray(labor_income + unearned_income)
    combined = concatenate_functions(
        functions, targets=["after_tax_income"], return_type="tuple"
    )
    assert not inspect.signature(combined).parameters
    return float(combined()[0])


def _regime_functions(regime_name: str) -> dict:
    """Regime-level functions of the baseline regime, as its builder sets them."""
    build = {
        "retiree": retiree_functions,
        "tied": tied_functions,
        "nongroup": nongroup_functions,
    }[REGIME_SPECS[regime_name]["his"]]
    return build(REGIME_SPECS[regime_name])


@pytest.mark.parametrize(
    ("regime_name", "policy", "labor_income", "unearned_income", "expected"),
    [
        # 3.8% of 50,000 unearned income above 200,000.
        ("retiree_nomc_inelig_canwork", PolicyVariant.ACA, 0.0, 250_000.0, 98_100.0),
        # 0.9% of 50,000 earnings above 200,000.
        ("nongroup_nomc_inelig_canwork", PolicyVariant.ACA, 250_000.0, 0.0, 99_550.0),
        # No earnings surtax outside the full ACA.
        (
            "tied_nomc_inelig_canwork",
            PolicyVariant.ACA_NO_MANDATE,
            250_000.0,
            0.0,
            100_000.0,
        ),
        (
            "nongroup_oamc_forced_forcedout",
            PolicyVariant.ACA_ONLY_MEDICAID_EXPANSION,
            0.0,
            250_000.0,
            98_100.0,
        ),
    ],
)
def test_aca_after_tax_income_charges_the_aca_surtaxes(
    regime_name: str,
    policy: PolicyVariant,
    labor_income: float,
    unearned_income: float,
    expected: float,
) -> None:
    """After-tax income under the ACA is income before the surtaxes less them."""
    result = _after_tax_income(
        regime_name=regime_name,
        policy=policy,
        labor_income=labor_income,
        unearned_income=unearned_income,
    )
    assert result == pytest.approx(expected)
