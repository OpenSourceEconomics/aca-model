"""Tests for ACA policy functions: mandate, premium subsidy, cost-sharing, Medicaid."""

import inspect

import jax.numpy as jnp
import pytest
from lcm.params import MappingLeaf

from aca_model.aca import health_insurance as aca_hi
from aca_model.baseline.health_insurance import BuyPrivate

FPL_SINGLE = 11770.0

MANDATE_SCHEDULE = MappingLeaf(
    {
        "min": 695.0,
        "income_fraction": 0.025,
        "max": 2085.0,
        "exempt_income": jnp.array([10350.0, 20700.0, 20700.0]),
    }
)

PREMIUM_CREDIT_SCHEDULE = MappingLeaf(
    {
        "kinks": jnp.array(
            [
                [1.0 * FPL_SINGLE, 0.0, 0.0],
                [1.33 * FPL_SINGLE, 0.0, 0.0],
                [1.5 * FPL_SINGLE, 0.0, 0.0],
                [2.0 * FPL_SINGLE, 0.0, 0.0],
                [2.5 * FPL_SINGLE, 0.0, 0.0],
                [3.0 * FPL_SINGLE, 0.0, 0.0],
                [4.0 * FPL_SINGLE, 0.0, 0.0],
            ]
        ),
        "frac_income": jnp.array([0.02, 0.02, 0.03, 0.04, 0.063, 0.0805, 0.095]),
    }
)

COST_SHARING_SCHEDULE = MappingLeaf(
    {
        "kinks": jnp.array(
            [
                [1.0 * FPL_SINGLE, 0.0, 0.0],
                [1.5 * FPL_SINGLE, 0.0, 0.0],
                [2.0 * FPL_SINGLE, 0.0, 0.0],
                [2.5 * FPL_SINGLE, 0.0, 0.0],
            ]
        ),
        "factors": jnp.array([1.0, 0.1721, 0.3887, 0.8819, 1.0]),
    }
)

MEDICAID_SCHEDULE = MappingLeaf(
    {
        "income_threshold": jnp.array([15580.0, 20902.0, 20902.0]),
    }
)


def test_mandate_penalty_uninsured_above_exempt() -> None:
    """Penalty = clip(income * 2.5%, 695, 2085) for uninsured above exempt."""
    income = jnp.array(40000.0)  # 40000 * 0.025 = 1000, within [695, 2085]
    result = aca_hi.mandate_penalty(
        gross_income=income,
        spousal_income=jnp.array(0),
        buy_private=jnp.array(BuyPrivate.no),
        mandate_schedule=MANDATE_SCHEDULE,
    )
    assert jnp.isclose(result, 1000.0)


def test_mandate_penalty_insured_zero() -> None:
    """buy_private=yes produces no penalty."""
    result = aca_hi.mandate_penalty(
        gross_income=jnp.array(40000.0),
        spousal_income=jnp.array(0),
        buy_private=jnp.array(BuyPrivate.yes),
        mandate_schedule=MANDATE_SCHEDULE,
    )
    assert jnp.isclose(result, 0.0)


def test_mandate_penalty_below_exempt_zero() -> None:
    """Income below exemption produces no penalty."""
    result = aca_hi.mandate_penalty(
        gross_income=jnp.array(5000.0),  # below 10350
        spousal_income=jnp.array(0),
        buy_private=jnp.array(BuyPrivate.no),
        mandate_schedule=MANDATE_SCHEDULE,
    )
    assert jnp.isclose(result, 0.0)


def test_mandate_penalty_clips_to_min() -> None:
    """Low income above exempt clips penalty to the minimum."""
    # 12000 * 0.025 = 300, below min of 695
    result = aca_hi.mandate_penalty(
        gross_income=jnp.array(12000.0),
        spousal_income=jnp.array(0),
        buy_private=jnp.array(BuyPrivate.no),
        mandate_schedule=MANDATE_SCHEDULE,
    )
    assert jnp.isclose(result, 695.0)


def test_mandate_penalty_clips_to_max() -> None:
    """High income clips penalty to the maximum."""
    # 200000 * 0.025 = 5000, above max of 2085
    result = aca_hi.mandate_penalty(
        gross_income=jnp.array(200000.0),
        spousal_income=jnp.array(0),
        buy_private=jnp.array(BuyPrivate.no),
        mandate_schedule=MANDATE_SCHEDULE,
    )
    assert jnp.isclose(result, 2085.0)


def test_hic_premium_subsidy_below_fpl_zero() -> None:
    """Below 100% FPL produces no subsidy (not eligible for marketplace)."""
    result = aca_hi.premium_subsidy(
        hic_premium=jnp.array(5000.0),
        gross_income=jnp.array(10000.0),  # below FPL_SINGLE
        spousal_income=jnp.array(0),
        buy_private=jnp.array(BuyPrivate.yes),
        premium_credit_schedule=PREMIUM_CREDIT_SCHEDULE,
    )
    assert jnp.isclose(result, 0.0)


def test_hic_premium_subsidy_above_400_fpl_zero() -> None:
    """Above 400% FPL produces no subsidy."""
    result = aca_hi.premium_subsidy(
        hic_premium=jnp.array(5000.0),
        gross_income=jnp.array(50000.0),  # above 4 * FPL_SINGLE = 47080
        spousal_income=jnp.array(0),
        buy_private=jnp.array(BuyPrivate.yes),
        premium_credit_schedule=PREMIUM_CREDIT_SCHEDULE,
    )
    assert jnp.isclose(result, 0.0)


def test_hic_premium_subsidy_at_200_fpl() -> None:
    """At 200% FPL, applicable rate is interpolated between kink 3 and 4."""
    income = 2.0 * FPL_SINGLE  # exactly at 200% FPL kink, rate = 0.04
    premium = 5000.0
    expected = max(0.0, premium - income * 0.04)
    result = aca_hi.premium_subsidy(
        hic_premium=jnp.array(premium),
        gross_income=jnp.array(income),
        spousal_income=jnp.array(0),
        buy_private=jnp.array(BuyPrivate.yes),
        premium_credit_schedule=PREMIUM_CREDIT_SCHEDULE,
    )
    assert jnp.isclose(result, expected, atol=1.0)


def test_hic_premium_subsidy_uninsured_zero() -> None:
    """buy_private=no produces no subsidy (not buying insurance)."""
    result = aca_hi.premium_subsidy(
        hic_premium=jnp.array(5000.0),
        gross_income=jnp.array(2.0 * FPL_SINGLE),
        spousal_income=jnp.array(0),
        buy_private=jnp.array(BuyPrivate.no),
        premium_credit_schedule=PREMIUM_CREDIT_SCHEDULE,
    )
    assert jnp.isclose(result, 0.0)


@pytest.mark.parametrize(
    ("income_fpl_frac", "expected_factor"),
    [
        (0.5, 1.0),  # below 100% FPL
        (1.2, 0.1721),  # 100-150% FPL
        (1.7, 0.3887),  # 150-200% FPL
        (2.3, 0.8819),  # 200-250% FPL
        (3.0, 1.0),  # above 250% FPL
    ],
)
def test_cost_sharing_scale_brackets(
    income_fpl_frac: float, expected_factor: float
) -> None:
    """Verify each cost-sharing bracket produces the correct factor."""
    result = aca_hi.cost_sharing(
        gross_income=jnp.array(income_fpl_frac * FPL_SINGLE),
        spousal_income=jnp.array(0),
        buy_private=jnp.array(BuyPrivate.yes),
        cost_sharing_schedule=COST_SHARING_SCHEDULE,
    )
    assert jnp.isclose(result, expected_factor, atol=0.001)


def test_cost_sharing_scale_uninsured_one() -> None:
    """buy_private=no produces scale=1.0 (no reduction)."""
    result = aca_hi.cost_sharing(
        gross_income=jnp.array(1.2 * FPL_SINGLE),  # would be 0.1721 if insured
        spousal_income=jnp.array(0),
        buy_private=jnp.array(BuyPrivate.no),
        cost_sharing_schedule=COST_SHARING_SCHEDULE,
    )
    assert jnp.isclose(result, 1.0)


def test_medicaid_eligible_aca_below_threshold() -> None:
    """Income below 133% FPL produces eligible."""
    result = aca_hi.is_medicaid_eligible(
        countable_income=jnp.array(10000.0),  # below 15580
        spousal_income=jnp.array(0),
        medicaid_schedule=MEDICAID_SCHEDULE,
    )
    assert result


def test_medicaid_eligible_aca_above_threshold() -> None:
    """Income above 133% FPL produces not eligible."""
    result = aca_hi.is_medicaid_eligible(
        countable_income=jnp.array(20000.0),  # above 15580
        spousal_income=jnp.array(0),
        medicaid_schedule=MEDICAID_SCHEDULE,
    )
    assert not result


def test_medicaid_eligible_aca_ignores_assets() -> None:
    """ACA Medicaid has no asset test; function signature has no assets param."""
    sig = inspect.signature(aca_hi.is_medicaid_eligible)
    assert "assets" not in sig.parameters


# --- Premium subsidy at exact FPL boundaries ---


def test_premium_subsidy_exactly_at_100_fpl() -> None:
    """At exactly 100% FPL, subsidy should apply (lower bound inclusive)."""
    result = aca_hi.premium_subsidy(
        hic_premium=jnp.array(5000.0),
        gross_income=jnp.array(1.0 * FPL_SINGLE),
        spousal_income=jnp.array(0),
        buy_private=jnp.array(BuyPrivate.yes),
        premium_credit_schedule=PREMIUM_CREDIT_SCHEDULE,
    )
    assert result > 0.0


def test_premium_subsidy_exactly_at_400_fpl() -> None:
    """At exactly 400% FPL, subsidy should be zero (upper bound exclusive)."""
    result = aca_hi.premium_subsidy(
        hic_premium=jnp.array(5000.0),
        gross_income=jnp.array(4.0 * FPL_SINGLE),
        spousal_income=jnp.array(0),
        buy_private=jnp.array(BuyPrivate.yes),
        premium_credit_schedule=PREMIUM_CREDIT_SCHEDULE,
    )
    assert jnp.isclose(result, 0.0)


def test_premium_subsidy_just_below_400_fpl() -> None:
    """Just below 400% FPL should still get a subsidy."""
    result = aca_hi.premium_subsidy(
        hic_premium=jnp.array(5000.0),
        gross_income=jnp.array(4.0 * FPL_SINGLE - 1.0),
        spousal_income=jnp.array(0),
        buy_private=jnp.array(BuyPrivate.yes),
        premium_credit_schedule=PREMIUM_CREDIT_SCHEDULE,
    )
    assert result > 0.0
