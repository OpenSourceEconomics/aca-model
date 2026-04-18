"""Tests for tax functions, ported from struct-ret."""

import jax.numpy as jnp
from lcm.params import MappingLeaf

from aca_model.environment import taxes
from aca_model.environment.taxes import _payroll_tax

ATOL = 0.01

# Tax brackets from struct-ret fixture (sp_inc=0 is single, 1/2 are married)
BRACKETS_LOWER = jnp.array(
    [
        [
            -jnp.inf,
            0.0,
            6200.0,
            15275.0,
            43100.0,
            95550.0,
            11700.0,
            192550.0,
            411300.0,
            412950.0,
        ],
        [
            -jnp.inf,
            0.0,
            12400.0,
            30550.0,
            86200.0,
            117000.0,
            161250.0,
            239250.0,
            417500.0,
            470000.0,
        ],
        [
            -jnp.inf,
            0.0,
            12400.0,
            30550.0,
            86200.0,
            117000.0,
            161250.0,
            239250.0,
            417500.0,
            470000.0,
        ],
    ]
)
BRACKETS_UPPER = jnp.array(
    [
        [
            0.0,
            6200.0,
            15275.0,
            43100.0,
            95550.0,
            11700.0,
            192550.0,
            411300.0,
            412950.0,
            jnp.inf,
        ],
        [
            0.0,
            12400.0,
            30550.0,
            86200.0,
            117000.0,
            161250.0,
            239250.0,
            417500.0,
            470000.0,
            jnp.inf,
        ],
        [
            0.0,
            12400.0,
            30550.0,
            86200.0,
            117000.0,
            161250.0,
            239250.0,
            417500.0,
            470000.0,
            jnp.inf,
        ],
    ]
)
MARGINAL_RATES = jnp.array(
    [
        [0.0, 0.0765, 0.199, 0.2584, 0.3734, 0.4069, 0.3449, 0.3998, 0.4214, 0.4703],
        [0.0, 0.0765, 0.199, 0.2584, 0.3734, 0.3114, 0.3449, 0.3998, 0.4214, 0.4703],
        [0.0, 0.0765, 0.199, 0.2584, 0.3734, 0.3114, 0.3449, 0.3998, 0.4214, 0.4703],
    ]
)
AFTER_TAX_AT_LOWER = jnp.array(
    [
        [
            0.0,
            0.0,
            5725.7,
            12994.775,
            33629.795,
            66494.965,
            16763.53,
            135238.365,
            266532.115,
            267486.805,
        ],
        [
            0.0,
            0.0,
            11451.4,
            25989.55,
            67259.59,
            86558.87,
            117029.42,
            168127.22,
            275112.87,
            305489.37,
        ],
        [
            0.0,
            0.0,
            11451.4,
            25989.55,
            67259.59,
            86558.87,
            117029.42,
            168127.22,
            275112.87,
            305489.37,
        ],
    ]
)

# Payroll brackets
PAYROLL_LOWER = jnp.array([-jnp.inf, 0.0, 68400.0])
PAYROLL_UPPER = jnp.array([0.0, 68400.0, jnp.inf])
PAYROLL_RATES = jnp.array([0.0, 0.0765, 0.0145])
PAYROLL_AT_LOWER = jnp.array([0.0, 0.0, 5232.6])

# SS benefit taxation brackets
SS_BRACKETS_LOWER = jnp.array(
    [
        [25000.0, 34000.0],
        [32000.0, 44000.0],
        [32000.0, 44000.0],
    ]
)
SS_BRACKETS_UPPER = jnp.array(
    [
        [34000.0, jnp.inf],
        [44000.0, jnp.inf],
        [44000.0, jnp.inf],
    ]
)
SS_FRACTION_CONSIDERED = jnp.array(
    [
        [0.5, 0.85],
        [0.5, 0.85],
        [0.5, 0.85],
    ]
)
SS_BEN_FRACTION_PROV_INCOME = 0.5

# MappingLeaf schedule objects
INCOME_TAX_SCHEDULE = MappingLeaf(
    {
        "brackets_lower": BRACKETS_LOWER,
        "brackets_upper": BRACKETS_UPPER,
        "marginal_rates": MARGINAL_RATES,
        "after_tax_at_lower": AFTER_TAX_AT_LOWER,
    }
)
PAYROLL_TAX_SCHEDULE = MappingLeaf(
    {
        "brackets_lower": PAYROLL_LOWER,
        "brackets_upper": PAYROLL_UPPER,
        "marginal_rates": PAYROLL_RATES,
        "taxes_at_lower": PAYROLL_AT_LOWER,
    }
)
SS_TAX_SCHEDULE = MappingLeaf(
    {
        "brackets_lower": SS_BRACKETS_LOWER,
        "brackets_upper": SS_BRACKETS_UPPER,
        "fraction_considered": SS_FRACTION_CONSIDERED,
        "ben_fraction_prov_income": SS_BEN_FRACTION_PROV_INCOME,
    }
)


def test_payroll_tax_below_cap() -> None:
    result = _payroll_tax(
        jnp.array(50000.0),
        PAYROLL_LOWER,
        PAYROLL_UPPER,
        PAYROLL_RATES,
        PAYROLL_AT_LOWER,
    )
    assert jnp.isclose(result, 50000.0 * 0.0765, atol=ATOL)


def test_payroll_tax_above_cap() -> None:
    result = _payroll_tax(
        jnp.array(100000.0),
        PAYROLL_LOWER,
        PAYROLL_UPPER,
        PAYROLL_RATES,
        PAYROLL_AT_LOWER,
    )
    expected = 5232.6 + (100000.0 - 68400.0) * 0.0145
    assert jnp.isclose(result, expected, atol=ATOL)


def test_taxable_ss_benefit_below_threshold() -> None:
    result = taxes.taxable_ss_benefit(
        capital_income=jnp.array(0.0),
        labor_income=jnp.array(10000.0),
        spousal_income_amounts=jnp.array([0.0, 0.0, 20000.0]),
        ss_benefit=jnp.array(5000.0),
        pension_benefit=jnp.array(0.0),
        spousal_income=jnp.array(0),
        ss_tax_schedule=SS_TAX_SCHEDULE,
    )
    # Provisional income = 10000 + 0.5*5000 = 12500, below 25000 threshold
    assert jnp.isclose(result, 0.0, atol=ATOL)


def test_gross_income_basic() -> None:
    result = taxes.gross_income(
        capital_income=jnp.array(1000.0),
        labor_income=jnp.array(5000.0),
        spousal_income=jnp.array(1),
        spousal_income_amounts=jnp.array([0.0, 2000.0, 20000.0]),
        taxable_ss_benefit=jnp.array(500.0),
        pension_benefit=jnp.array(300.0),
    )
    assert jnp.isclose(result, 8800.0, atol=ATOL)


def test_after_tax_income_zero() -> None:
    gi = taxes.gross_income(
        capital_income=jnp.array(0.0),
        labor_income=jnp.array(0.0),
        spousal_income=jnp.array(0),
        spousal_income_amounts=jnp.array([0.0, 0.0, 20000.0]),
        taxable_ss_benefit=jnp.array(0.0),
        pension_benefit=jnp.array(0.0),
    )
    result = taxes.after_tax_income(
        gross_income=gi,
        ss_benefit=jnp.array(0.0),
        taxable_ss_benefit=jnp.array(0.0),
        labor_income=jnp.array(0.0),
        spousal_income=jnp.array(0),
        income_tax_schedule=INCOME_TAX_SCHEDULE,
        payroll_tax_schedule=PAYROLL_TAX_SCHEDULE,
    )
    assert jnp.isclose(result, 0.0, atol=ATOL)


def test_after_tax_income_low_bracket() -> None:
    gross = 5000.0
    gi = taxes.gross_income(
        capital_income=jnp.array(0.0),
        labor_income=jnp.array(gross),
        spousal_income=jnp.array(0),
        spousal_income_amounts=jnp.array([0.0, 0.0, 20000.0]),
        taxable_ss_benefit=jnp.array(0.0),
        pension_benefit=jnp.array(0.0),
    )
    result = taxes.after_tax_income(
        gross_income=gi,
        ss_benefit=jnp.array(0.0),
        taxable_ss_benefit=jnp.array(0.0),
        labor_income=jnp.array(gross),
        spousal_income=jnp.array(0),
        income_tax_schedule=INCOME_TAX_SCHEDULE,
        payroll_tax_schedule=PAYROLL_TAX_SCHEDULE,
    )
    # Bracket 1: 0-6200 at 7.65%
    income_tax_part = gross * (1 - 0.0765)
    payroll = gross * 0.0765
    expected = income_tax_part - payroll
    assert jnp.isclose(result, expected, atol=ATOL)


def test_marginal_tax_rate_low_bracket() -> None:
    result = taxes.marginal_rate(
        gross_income=jnp.array(5000.0),
        spousal_income=jnp.array(0),
        income_tax_schedule=INCOME_TAX_SCHEDULE,
    )
    # 5000 is in bracket 1 (0-6200), rate = 0.0765
    assert jnp.isclose(result, 0.0765, atol=ATOL)


def test_marginal_tax_rate_mid_bracket() -> None:
    result = taxes.marginal_rate(
        gross_income=jnp.array(10000.0),
        spousal_income=jnp.array(0),
        income_tax_schedule=INCOME_TAX_SCHEDULE,
    )
    # 10000 is in bracket 2 (6200-15275), rate = 0.199
    assert jnp.isclose(result, 0.199, atol=ATOL)
