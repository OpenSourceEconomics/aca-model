"""Integration tests for SSI → Medicaid → OOP chain.

Compose via dags: countable_income → is_ssi_eligible → ssi_benefit,
and is_medicaid_eligible → oop_with_medicaid.
"""

import jax.numpy as jnp
from dags import concatenate_functions

from aca_model.baseline import health_insurance
from aca_model.baseline.health_insurance import BuyPrivate

ATOL = 0.01

SSI_ASSETS_TEST = jnp.array([2000.0, 3000.0, 3000.0])
SSI_MAX_BENEFIT = jnp.array([8000.0, 12000.0, 12000.0])


def test_low_income_qualifies_for_ssi_and_medicaid() -> None:
    """Low-income agent with Medicare → SSI eligible → Medicaid → reduced OOP."""
    functions = {
        "countable_income": health_insurance.countable_income,
        "is_ssi_eligible": health_insurance.is_ssi_eligible,
        "ssi_benefit": health_insurance.ssi_benefit,
        "is_medicaid_eligible": health_insurance.is_medicaid_eligible,
    }
    combined = concatenate_functions(
        functions,
        targets=["is_ssi_eligible", "ssi_benefit", "is_medicaid_eligible"],
        return_type="dict",
    )
    result = combined(
        labor_income=jnp.array(0.0),
        capital_income=jnp.array(0.0),
        spousal_income_amounts=jnp.array([0.0, 0.0, 20000.0]),
        ss_benefit=jnp.array(500.0),
        pension_benefit=jnp.array(0.0),
        ssi_ignored_overall=20.0,
        ssi_ignored_earned=65.0,
        assets=jnp.array(1000.0),
        spousal_income=jnp.array(0),
        gets_medicare=True,
        ssi_assets_test=SSI_ASSETS_TEST,
        ssi_maximum_benefit=SSI_MAX_BENEFIT,
    )
    assert result["is_ssi_eligible"]
    assert result["is_medicaid_eligible"]
    assert result["ssi_benefit"] > 0.0


def test_high_income_ineligible_for_ssi() -> None:
    """High-income agent → SSI ineligible → Medicaid ineligible."""
    functions = {
        "countable_income": health_insurance.countable_income,
        "is_ssi_eligible": health_insurance.is_ssi_eligible,
        "is_medicaid_eligible": health_insurance.is_medicaid_eligible,
    }
    combined = concatenate_functions(
        functions,
        targets=["is_ssi_eligible", "is_medicaid_eligible"],
        return_type="dict",
    )
    result = combined(
        labor_income=jnp.array(50000.0),
        capital_income=jnp.array(5000.0),
        spousal_income_amounts=jnp.array([0.0, 0.0, 20000.0]),
        ss_benefit=jnp.array(2000.0),
        pension_benefit=jnp.array(0.0),
        ssi_ignored_overall=20.0,
        ssi_ignored_earned=65.0,
        assets=jnp.array(1000.0),
        spousal_income=jnp.array(0),
        gets_medicare=True,
        ssi_assets_test=SSI_ASSETS_TEST,
        ssi_maximum_benefit=SSI_MAX_BENEFIT,
    )
    assert not result["is_ssi_eligible"]
    assert not result["is_medicaid_eligible"]


def test_no_medicare_blocks_ssi_under_baseline() -> None:
    """Baseline SSI requires gets_medicare; without it, even low income fails."""
    functions = {
        "countable_income": health_insurance.countable_income,
        "is_ssi_eligible": health_insurance.is_ssi_eligible,
        "is_medicaid_eligible": health_insurance.is_medicaid_eligible,
    }
    combined = concatenate_functions(
        functions,
        targets="is_ssi_eligible",
    )
    result = combined(
        labor_income=jnp.array(0.0),
        capital_income=jnp.array(0.0),
        spousal_income_amounts=jnp.array([0.0, 0.0, 20000.0]),
        ss_benefit=jnp.array(0.0),
        pension_benefit=jnp.array(0.0),
        ssi_ignored_overall=20.0,
        ssi_ignored_earned=65.0,
        assets=jnp.array(100.0),
        spousal_income=jnp.array(0),
        gets_medicare=False,
        ssi_assets_test=SSI_ASSETS_TEST,
        ssi_maximum_benefit=SSI_MAX_BENEFIT,
    )
    assert not result


def test_medicaid_reduces_oop() -> None:
    """Medicaid as secondary payer reduces OOP below primary insurance OOP."""
    functions = {
        "primary_oop": health_insurance.primary_oop,
        "is_medicaid_eligible": health_insurance.is_medicaid_eligible,
        "oop_costs": health_insurance.oop_with_medicaid,
    }
    combined = concatenate_functions(functions, targets="oop_costs")

    # Medicaid-eligible: OOP should be lower
    oop_medicaid = combined(
        total_health_costs=jnp.array(10000.0),
        buy_private=jnp.array(BuyPrivate.yes),
        deductible=500.0,
        coinsurance_rate=0.2,
        oop_max=5000.0,
        is_ssi_eligible=jnp.array(True),
        deductible_medicaid=100.0,
        coinsurance_rate_medicaid=0.05,
        oop_max_medicaid=1000.0,
    )

    # Not Medicaid-eligible: primary OOP only
    oop_no_medicaid = combined(
        total_health_costs=jnp.array(10000.0),
        buy_private=jnp.array(BuyPrivate.yes),
        deductible=500.0,
        coinsurance_rate=0.2,
        oop_max=5000.0,
        is_ssi_eligible=jnp.array(False),
        deductible_medicaid=100.0,
        coinsurance_rate_medicaid=0.05,
        oop_max_medicaid=1000.0,
    )

    assert oop_medicaid < oop_no_medicaid
