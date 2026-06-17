"""Integration tests for the budget constraint chain.

Compose via dags: capital_income → taxable_ss_benefit → gross_income →
after_tax_income → cash_on_hand → transfers → next_assets.
"""

import inspect
from collections.abc import Callable

import jax.numpy as jnp
from dags import concatenate_functions
from lcm.params import MappingLeaf

from aca_model.aca import health_insurance as aca_hi
from aca_model.agent import assets_and_income
from aca_model.baseline import health_insurance
from aca_model.environment import taxes

ATOL = 1.0

# Simplified tax schedules (2 brackets: 0% below standard deduction, 20% above)
INCOME_TAX_SCHEDULE = MappingLeaf(
    {
        "brackets_lower": jnp.array([[0.0, 12000.0]]),
        "brackets_upper": jnp.array([[12000.0, jnp.inf]]),
        "marginal_rates": jnp.array([[0.0, 0.2]]),
        "after_tax_at_lower": jnp.array([[0.0, 12000.0]]),
    }
)
PAYROLL_TAX_SCHEDULE = MappingLeaf(
    {
        "brackets_lower": jnp.array([0.0]),
        "brackets_upper": jnp.array([jnp.inf]),
        "marginal_rates": jnp.array([0.0765]),
        "taxes_at_lower": jnp.array([0.0]),
    }
)
SS_TAX_SCHEDULE = MappingLeaf(
    {
        "brackets_lower": jnp.array([[25000.0, 34000.0]]),
        "brackets_upper": jnp.array([[34000.0, jnp.inf]]),
        "fraction_considered": jnp.array([[0.5, 0.85]]),
        "ben_fraction_prov_income": 0.5,
    }
)


def test_working_agent_cash_on_hand() -> None:
    """Working agent: labor income → taxes → cash_on_hand is positive."""
    functions = {
        "capital_income": assets_and_income.capital_income,
        "taxable_ss_benefit": taxes.taxable_ss_benefit,
        "gross_income": taxes.gross_income,
        "after_tax_income": taxes.after_tax_income,
        "premium_default": assets_and_income.premium_default,
        "cash_on_hand": assets_and_income.cash_on_hand,
    }
    combined = concatenate_functions(functions, targets="cash_on_hand")

    result = combined(
        assets=jnp.array(50000.0),
        rate_of_return=jnp.asarray(0.03),
        labor_income=jnp.array(40000.0),
        spousal_income_amounts=jnp.array([0.0, 0.0, 20000.0]),
        ss_benefit=jnp.array(0.0),
        pension_benefit=jnp.array(0.0),
        spousal_income=jnp.int32(0),
        income_tax_schedule=INCOME_TAX_SCHEDULE,
        payroll_tax_schedule=PAYROLL_TAX_SCHEDULE,
        ss_tax_schedule=SS_TAX_SCHEDULE,
        ssi_benefit=jnp.array(0.0),
        hic_premium=jnp.array(2000.0),
        consumption_dollars_floor=jnp.array(3000.0),
    )

    assert result > 0.0
    assert result < 50000.0 + 41500.0  # Less than assets + gross


def test_retired_agent_with_pension() -> None:
    """Retired agent: zero labor income, pension flows through gross income."""
    functions = {
        "capital_income": assets_and_income.capital_income,
        "taxable_ss_benefit": taxes.taxable_ss_benefit,
        "gross_income": taxes.gross_income,
        "after_tax_income": taxes.after_tax_income,
        "premium_default": assets_and_income.premium_default,
        "cash_on_hand": assets_and_income.cash_on_hand,
    }
    combined = concatenate_functions(
        functions,
        targets=["gross_income", "after_tax_income", "cash_on_hand"],
        return_type="dict",
    )

    result = combined(
        assets=jnp.array(200000.0),
        rate_of_return=jnp.asarray(0.03),
        labor_income=jnp.array(0.0),
        spousal_income_amounts=jnp.array([0.0, 0.0, 20000.0]),
        ss_benefit=jnp.array(15000.0),
        pension_benefit=jnp.array(10000.0),
        spousal_income=jnp.int32(0),
        income_tax_schedule=INCOME_TAX_SCHEDULE,
        payroll_tax_schedule=PAYROLL_TAX_SCHEDULE,
        ss_tax_schedule=SS_TAX_SCHEDULE,
        ssi_benefit=jnp.array(0.0),
        hic_premium=jnp.array(3000.0),
        consumption_dollars_floor=jnp.array(3000.0),
    )

    # gross_income includes only the taxable SS portion
    assert result["gross_income"] > 0.0
    # after_tax_income adds back the non-taxable SS portion, so can exceed gross
    assert result["after_tax_income"] > 0.0
    assert result["cash_on_hand"] > 0.0


def test_transfers_kick_in_below_floor() -> None:
    """When cash_on_hand < consumption_dollars_floor, transfers fill the gap."""
    functions = {
        "premium_default": assets_and_income.premium_default,
        "cash_on_hand": assets_and_income.cash_on_hand,
        "transfers": assets_and_income.transfers,
        "next_assets": assets_and_income.next_assets,
    }
    combined = concatenate_functions(
        functions,
        targets=["transfers", "next_assets"],
        return_type="dict",
    )

    result = combined(
        assets=jnp.array(500.0),
        after_tax_income=jnp.array(200.0),
        ssi_benefit=jnp.array(0.0),
        hic_premium=jnp.array(0.0),
        oop_costs=jnp.array(0.0),
        consumption_dollars_floor=jnp.array(5000.0),
        pension_assets_adjustment=jnp.array(0.0),
        consumption_dollars=jnp.array(4000.0),
    )

    # cash_on_hand = 500 + 200 = 700
    # transfers = max(0, 5000 - 700) = 4300
    assert jnp.isclose(result["transfers"], 4300.0, atol=ATOL)
    # next_assets = 700 + 4300 + 0 - 4000 = 1000
    assert jnp.isclose(result["next_assets"], 1000.0, atol=ATOL)


def _premium_default_dag() -> Callable[..., dict]:
    """DAG composing premium_default → cash_on_hand → transfers."""
    functions = {
        "premium_default": assets_and_income.premium_default,
        "cash_on_hand": assets_and_income.cash_on_hand,
        "transfers": assets_and_income.transfers,
    }
    return concatenate_functions(
        functions,
        targets=["premium_default", "cash_on_hand", "transfers"],
        return_type="dict",
    )


def test_premium_default_partially_unaffordable_premium() -> None:
    """A premium exceeding affordable resources defaults the excess.

    With resources 7000, premium 5000, floor 3000, the household can afford
    7000 - 3000 = 4000 of the premium and defaults the remaining 1000. The
    affordable part leaves cash-on-hand exactly at the floor and no income
    transfer is needed.
    """
    combined = _premium_default_dag()
    result = combined(
        assets=jnp.array(4000.0),
        after_tax_income=jnp.array(3000.0),
        ssi_benefit=jnp.array(0.0),
        hic_premium=jnp.array(5000.0),
        consumption_dollars_floor=jnp.array(3000.0),
    )
    assert jnp.isclose(result["premium_default"], 1000.0, atol=ATOL)
    assert jnp.isclose(result["cash_on_hand"], 3000.0, atol=ATOL)
    assert jnp.isclose(result["transfers"], 0.0, atol=ATOL)


def test_premium_default_genuinely_poor_household() -> None:
    """Default and income transfer are separate channels.

    With resources 2000, premium 5000, floor 3000, the whole 5000 premium is
    defaulted (nothing is affordable above the floor), cash-on-hand equals the
    untouched 2000 resources, and the transfer system tops the 1000 income
    shortfall to the floor.
    """
    combined = _premium_default_dag()
    result = combined(
        assets=jnp.array(1500.0),
        after_tax_income=jnp.array(500.0),
        ssi_benefit=jnp.array(0.0),
        hic_premium=jnp.array(5000.0),
        consumption_dollars_floor=jnp.array(3000.0),
    )
    assert jnp.isclose(result["premium_default"], 5000.0, atol=ATOL)
    assert jnp.isclose(result["cash_on_hand"], 2000.0, atol=ATOL)
    assert jnp.isclose(result["transfers"], 1000.0, atol=ATOL)


def test_premium_default_affluent_household_pays_full_premium() -> None:
    """When resources comfortably exceed floor + premium, nothing is defaulted."""
    combined = _premium_default_dag()
    result = combined(
        assets=jnp.array(50000.0),
        after_tax_income=jnp.array(20000.0),
        ssi_benefit=jnp.array(0.0),
        hic_premium=jnp.array(5000.0),
        consumption_dollars_floor=jnp.array(3000.0),
    )
    assert jnp.isclose(result["premium_default"], 0.0, atol=ATOL)
    # Full premium leaves cash-on-hand: 70000 - 5000 = 65000.
    assert jnp.isclose(result["cash_on_hand"], 65000.0, atol=ATOL)


def _aca_premium_default_dag() -> Callable[..., dict]:
    """DAG composing the ACA premium_default → cash_on_hand."""
    functions = {
        "premium_default": aca_hi.premium_default,
        "cash_on_hand": aca_hi.cash_on_hand,
    }
    return concatenate_functions(
        functions,
        targets=["premium_default", "cash_on_hand"],
        return_type="dict",
    )


def test_aca_premium_default_uses_net_premium_and_keeps_mandate() -> None:
    """The ACA default applies to the subsidy-net premium; the mandate stays.

    With resources 7000, gross premium 5000, subsidy 1000 (net premium 4000),
    floor 3000, the affordable amount is 7000 - 3000 = 4000, so the net premium
    is fully paid and nothing defaults. The 500 mandate penalty is a separate
    non-defaultable tax that still leaves cash-on-hand:
    `7000 - 4000 - 500 = 2500`.
    """
    combined = _aca_premium_default_dag()
    result = combined(
        assets=jnp.array(4000.0),
        after_tax_income=jnp.array(3000.0),
        ssi_benefit=jnp.array(0.0),
        hic_premium=jnp.array(5000.0),
        hic_premium_subsidy=jnp.array(1000.0),
        mandate_penalty=jnp.array(500.0),
        consumption_dollars_floor=jnp.array(3000.0),
    )
    assert jnp.isclose(result["premium_default"], 0.0, atol=ATOL)
    assert jnp.isclose(result["cash_on_hand"], 2500.0, atol=ATOL)


def test_aca_premium_default_defaults_unaffordable_net_premium() -> None:
    """When the subsidy-net premium exceeds affordability, the excess defaults.

    Resources 5000, gross premium 6000, subsidy 1000 (net 5000), floor 3000:
    affordable is 5000 - 3000 = 2000, so 5000 - 2000 = 3000 of the net premium
    defaults. The mandate penalty is zero here, and cash-on-hand sits at the
    floor: `5000 - (5000 - 3000) = 3000`.
    """
    combined = _aca_premium_default_dag()
    result = combined(
        assets=jnp.array(2000.0),
        after_tax_income=jnp.array(3000.0),
        ssi_benefit=jnp.array(0.0),
        hic_premium=jnp.array(6000.0),
        hic_premium_subsidy=jnp.array(1000.0),
        mandate_penalty=jnp.array(0.0),
        consumption_dollars_floor=jnp.array(3000.0),
    )
    assert jnp.isclose(result["premium_default"], 3000.0, atol=ATOL)
    assert jnp.isclose(result["cash_on_hand"], 3000.0, atol=ATOL)


def test_premium_default_does_not_change_oop_coverage() -> None:
    """Defaulting on a premium leaves OOP coverage untouched.

    The uncompensated-care channel only stops part of the premium being paid;
    the household keeps the same out-of-pocket protection that period, so OOP
    costs do not depend on `premium_default`.
    """
    sig = inspect.signature(health_insurance.oop_with_medicaid)
    assert "premium_default" not in sig.parameters
