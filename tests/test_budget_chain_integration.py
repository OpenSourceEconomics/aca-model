"""Integration tests for the budget constraint chain.

Compose via dags: capital_income → taxable_ss_benefit → gross_income →
after_tax_income → cash_on_hand → transfers → next_assets.
"""

import jax.numpy as jnp
from dags import concatenate_functions
from lcm.params import MappingLeaf

from aca_model.agent import assets_and_income
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
        "cash_on_hand": assets_and_income.cash_on_hand,
    }
    combined = concatenate_functions(functions, targets="cash_on_hand")

    result = combined(
        assets=jnp.array(50000.0),
        rate_of_return=0.03,
        labor_income=jnp.array(40000.0),
        spousal_income_amounts=jnp.array([0.0, 0.0, 20000.0]),
        ss_benefit=jnp.array(0.0),
        pension_benefit=jnp.array(0.0),
        spousal_income=jnp.array(0),
        income_tax_schedule=INCOME_TAX_SCHEDULE,
        payroll_tax_schedule=PAYROLL_TAX_SCHEDULE,
        ss_tax_schedule=SS_TAX_SCHEDULE,
        ssi_benefit=jnp.array(0.0),
        hic_premium=jnp.array(2000.0),
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
        "cash_on_hand": assets_and_income.cash_on_hand,
    }
    combined = concatenate_functions(
        functions,
        targets=["gross_income", "after_tax_income", "cash_on_hand"],
        return_type="dict",
    )

    result = combined(
        assets=jnp.array(200000.0),
        rate_of_return=0.03,
        labor_income=jnp.array(0.0),
        spousal_income_amounts=jnp.array([0.0, 0.0, 20000.0]),
        ss_benefit=jnp.array(15000.0),
        pension_benefit=jnp.array(10000.0),
        spousal_income=jnp.array(0),
        income_tax_schedule=INCOME_TAX_SCHEDULE,
        payroll_tax_schedule=PAYROLL_TAX_SCHEDULE,
        ss_tax_schedule=SS_TAX_SCHEDULE,
        ssi_benefit=jnp.array(0.0),
        hic_premium=jnp.array(3000.0),
    )

    # gross_income includes only the taxable SS portion
    assert result["gross_income"] > 0.0
    # after_tax_income adds back the non-taxable SS portion, so can exceed gross
    assert result["after_tax_income"] > 0.0
    assert result["cash_on_hand"] > 0.0


def test_transfers_kick_in_below_floor() -> None:
    """When cash_on_hand < consumption_floor, transfers fill the gap."""
    functions = {
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
        consumption_floor=5000.0,
        equivalence_scale=jnp.array(1.0),
        pension_assets_adjustment=jnp.array(0.0),
        consumption=jnp.array(4000.0),
    )

    # cash_on_hand = 500 + 200 = 700
    # floor = 5000 * 1.0 = 5000
    # transfers = max(0, 5000 - 700) = 4300
    assert jnp.isclose(result["transfers"], 4300.0, atol=ATOL)
    # next_assets = 700 + 4300 + 0 - 4000 = 1000
    assert jnp.isclose(result["next_assets"], 1000.0, atol=ATOL)
