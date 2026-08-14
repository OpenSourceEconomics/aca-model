"""Integration tests for SSI → Medicaid → OOP chain.

Compose via dags: countable_income → is_ssi_eligible → ssi_benefit,
and is_medicaid_eligible → oop_with_medicaid.
"""

import jax.numpy as jnp
from dags import concatenate_functions
from lcm.params import MappingLeaf

from aca_model.aca import health_insurance as aca_hi
from aca_model.baseline import health_insurance
from aca_model.baseline.health_insurance import BuyPrivate

ATOL = 0.01

SSI_ASSETS_TEST = jnp.asarray(2000.0)
SSI_MAX_BENEFIT = jnp.asarray(8000.0)
MEDICAID_SCHEDULE = MappingLeaf({"income_threshold": jnp.asarray(15000.0)})


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
        spousal_income_amount=jnp.asarray(0.0),
        ss_benefit=jnp.array(500.0),
        pension_benefit=jnp.array(0.0),
        ssi_ignored_overall=jnp.asarray(20.0),
        ssi_ignored_earned=jnp.asarray(65.0),
        assets=jnp.array(1000.0),
        crossed_oamc_threshold=jnp.asarray(True),
        is_disabled=jnp.asarray(False),
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
        spousal_income_amount=jnp.asarray(0.0),
        ss_benefit=jnp.array(2000.0),
        pension_benefit=jnp.array(0.0),
        ssi_ignored_overall=jnp.asarray(20.0),
        ssi_ignored_earned=jnp.asarray(65.0),
        assets=jnp.array(1000.0),
        crossed_oamc_threshold=jnp.asarray(True),
        is_disabled=jnp.asarray(False),
        ssi_assets_test=SSI_ASSETS_TEST,
        ssi_maximum_benefit=SSI_MAX_BENEFIT,
    )
    assert not result["is_ssi_eligible"]
    assert not result["is_medicaid_eligible"]


def test_disabled_under_65_qualifies_for_ssi_and_medicaid() -> None:
    """A disabled, under-65, no-Medicare household qualifies on the categorical track.

    Disability — not Medicare status — opens the categorical SSI/Medicaid
    track, so an asset- and income-eligible disabled household qualifies even
    before reaching the Medicare age.
    """
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
        labor_income=jnp.array(0.0),
        capital_income=jnp.array(0.0),
        spousal_income_amount=jnp.asarray(0.0),
        ss_benefit=jnp.array(0.0),
        pension_benefit=jnp.array(0.0),
        ssi_ignored_overall=jnp.asarray(20.0),
        ssi_ignored_earned=jnp.asarray(65.0),
        assets=jnp.array(100.0),
        crossed_oamc_threshold=jnp.asarray(False),
        is_disabled=jnp.asarray(True),
        ssi_assets_test=SSI_ASSETS_TEST,
        ssi_maximum_benefit=SSI_MAX_BENEFIT,
    )
    assert result["is_ssi_eligible"]
    assert result["is_medicaid_eligible"]


def test_aca_expansion_uses_magi_not_countable_income() -> None:
    """A worker below threshold on half-counted SSI income but above it on MAGI fails.

    The ACA expansion test reads full-count MAGI, so a worker whose SSI
    `countable_income` (earnings half-counted) lands below the threshold but
    whose `aca_magi` (full earnings) exceeds it is not expansion-eligible.
    """
    functions = {
        "countable_income": health_insurance.countable_income,
        "aca_magi": health_insurance.aca_magi,
        "is_ssi_eligible": health_insurance.is_ssi_eligible,
        "is_medicaid_eligible": aca_hi.is_medicaid_eligible,
    }
    combined = concatenate_functions(
        functions,
        targets=["countable_income", "aca_magi", "is_medicaid_eligible"],
        return_type="dict",
    )
    result = combined(
        # Earned 28000: half-counted countable income ≈ 13967 < 15000 threshold,
        # but full MAGI 28000 > 15000 threshold.
        labor_income=jnp.array(28000.0),
        capital_income=jnp.array(0.0),
        spousal_income_amount=jnp.asarray(0.0),
        ss_benefit=jnp.array(0.0),
        pension_benefit=jnp.array(0.0),
        ssi_ignored_overall=jnp.asarray(20.0),
        ssi_ignored_earned=jnp.asarray(65.0),
        assets=jnp.array(100.0),
        crossed_oamc_threshold=jnp.asarray(False),
        is_disabled=jnp.asarray(False),
        ssi_assets_test=SSI_ASSETS_TEST,
        ssi_maximum_benefit=SSI_MAX_BENEFIT,
        medicaid_schedule=MEDICAID_SCHEDULE,
    )
    assert result["countable_income"] < 15000.0
    assert result["aca_magi"] > 15000.0
    assert not result["is_medicaid_eligible"]


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
        deductible=jnp.asarray(500.0),
        coinsurance_rate=jnp.asarray(0.2),
        oop_max=jnp.asarray(5000.0),
        is_ssi_eligible=jnp.array(True),
        deductible_medicaid=jnp.asarray(100.0),
        coinsurance_rate_medicaid=jnp.asarray(0.05),
        oop_max_medicaid=jnp.asarray(1000.0),
    )

    # Not Medicaid-eligible: primary OOP only
    oop_no_medicaid = combined(
        total_health_costs=jnp.array(10000.0),
        buy_private=jnp.array(BuyPrivate.yes),
        deductible=jnp.asarray(500.0),
        coinsurance_rate=jnp.asarray(0.2),
        oop_max=jnp.asarray(5000.0),
        is_ssi_eligible=jnp.array(False),
        deductible_medicaid=jnp.asarray(100.0),
        coinsurance_rate_medicaid=jnp.asarray(0.05),
        oop_max_medicaid=jnp.asarray(1000.0),
    )

    assert oop_medicaid < oop_no_medicaid
