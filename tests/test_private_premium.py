"""Private non-group premium: zero-profit intercept, pricing rule, and floor rule."""

import jax.numpy as jnp
import numpy as np
import pytest

from aca_model.aca import health_insurance as aca_health_insurance
from aca_model.baseline import health_insurance
from aca_model.baseline.health_insurance import BuyPrivate
from aca_model.baseline.regimes import _nongroup
from aca_model.baseline.regimes._common import REGIME_SPECS

_MARKUP = jnp.asarray(0.1696)
_MINIMUM = jnp.array([1000.0, 2000.0])
_SAMPLE_COST = jnp.array([1000.0, 3000.0, 8000.0])
_SAMPLE_MARRIED = jnp.array([0, 1, 0], dtype=jnp.int32)


@pytest.mark.parametrize(
    ("coeff", "expected"),
    [
        # 3 p0 + 0.5 * 1.1696 * 12000 = 1.1696 * 12000; no minimum binds.
        (0.5, 2339.2),
        # The single 1000-cost buyer sits at the 1000 minimum:
        # 1000 + 2 p0 + 1.5 * 1.1696 * 11000 = 1.1696 * 12000.
        (1.5, -3131.6),
    ],
)
def test_private_premium_intercept_concrete_values(
    coeff: float, expected: float
) -> None:
    """The intercept solves mean premium = (1 + markup) * mean insurer cost."""
    intercept = health_insurance.private_premium_intercept(
        premium_predicted_hcc=jnp.asarray(coeff),
        premium_markup=_MARKUP,
        premium_minimum=_MINIMUM,
        premium_sample_insurer_cost=_SAMPLE_COST,
        premium_sample_is_married=_SAMPLE_MARRIED,
    )
    np.testing.assert_allclose(intercept, expected, atol=1e-6)


def test_private_premium_intercept_zero_profit_on_sample() -> None:
    """At the solved intercept, premium revenue covers insurer cost plus markup.

    The sample mean of `max(minimum[married], p0 + b (1 + markup) cost)`
    equals `(1 + markup)` times the sample mean of the insurer cost.
    """
    rng = np.random.default_rng(seed=3)
    cost = jnp.asarray(np.exp(rng.normal(loc=7.5, scale=1.2, size=2000)))
    married = jnp.asarray(rng.integers(0, 2, size=2000), dtype=jnp.int32)
    coeff = jnp.asarray(1.161245437600243)
    intercept = health_insurance.private_premium_intercept(
        premium_predicted_hcc=coeff,
        premium_markup=_MARKUP,
        premium_minimum=_MINIMUM,
        premium_sample_insurer_cost=cost,
        premium_sample_is_married=married,
    )
    premiums = jnp.maximum(_MINIMUM[married], intercept + coeff * (1 + _MARKUP) * cost)
    assert bool((premiums == _MINIMUM[married]).any())
    np.testing.assert_allclose(premiums.mean(), (1 + _MARKUP) * cost.mean(), rtol=1e-10)


def test_private_premium_intercept_without_root_in_bracket_is_nan() -> None:
    """No intercept in [-50000, 10000] balances the books: the result is NaN."""
    intercept = health_insurance.private_premium_intercept(
        premium_predicted_hcc=jnp.asarray(100.0),
        premium_markup=_MARKUP,
        premium_minimum=_MINIMUM,
        premium_sample_insurer_cost=_SAMPLE_COST,
        premium_sample_is_married=_SAMPLE_MARRIED,
    )
    assert bool(jnp.isnan(intercept))


@pytest.mark.parametrize(
    ("is_married", "insurer_cost", "expected"),
    [
        (0, 1913.49, 1529.488080),
        (1, 1913.49, 2000.0),
        (0, 371.13, 1000.0),
        (1, 8319.48, 10230.056703),
    ],
)
def test_premium_private_buyer(
    is_married: int, insurer_cost: float, expected: float
) -> None:
    """Premium is `max(minimum[married], p0 + b (1 + markup) E[insurer cost])`.

    No marital-status shifter: couples differ only through the minimum and
    their insurer cost.
    """
    result = health_insurance.premium(
        buy_private=jnp.array(BuyPrivate.yes),
        is_married=jnp.int32(is_married),
        predicted_hcc_insurer=jnp.asarray(insurer_cost),
        private_premium_intercept=jnp.asarray(-1069.4),
        premium_predicted_hcc=jnp.asarray(1.161245437600243),
        premium_markup=_MARKUP,
        premium_minimum=_MINIMUM,
    )
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_premium_uninsured_pays_nothing() -> None:
    """A household that does not buy private cover pays no premium."""
    result = health_insurance.premium(
        buy_private=jnp.array(BuyPrivate.no),
        is_married=jnp.int32(0),
        predicted_hcc_insurer=jnp.asarray(1913.49),
        private_premium_intercept=jnp.asarray(-1069.4),
        premium_predicted_hcc=jnp.asarray(1.161245437600243),
        premium_markup=_MARKUP,
        premium_minimum=_MINIMUM,
    )
    assert float(result) == 0.0


@pytest.mark.parametrize(
    ("buy_private", "premium_default", "expected"),
    [
        (BuyPrivate.yes, 0.0, True),
        (BuyPrivate.yes, 0.01, False),
        (BuyPrivate.no, 500.0, True),
        (BuyPrivate.no, 0.0, True),
    ],
)
def test_private_cover_paid_in_full(
    buy_private: int, premium_default: float, expected: bool
) -> None:
    """Private cover is a valid choice only if the household pays its premium.

    A household at the consumption floor cannot hold private cover financed by
    transfers or by not paying the premium.
    """
    result = health_insurance.private_cover_paid_in_full(
        buy_private=jnp.array(buy_private),
        premium_default=jnp.asarray(premium_default),
    )
    assert bool(result) is expected


def test_predicted_hcc_insurer_reads_age_marriage_health_cell() -> None:
    """Predicted insurer cost interpolates the (period, married, health) row."""
    table = jnp.zeros((2, 2, 2, 3)).at[1, 1, 0].set(jnp.array([100.0, 200.0, 400.0]))
    result = health_insurance.hcc_insurer_predicted(
        period=jnp.int32(1),
        is_married=jnp.int32(1),
        good_health=jnp.int32(0),
        hcc_persistent=jnp.asarray(0.5),
        predicted_hcc_insurer_table=table,
        hcc_persistent_grid=jnp.array([-1.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(result, 300.0, atol=1e-9)


@pytest.mark.parametrize(
    ("is_married", "total", "expected"),
    [
        (0, 60000.0, 7472.3),
        (1, 60000.0, 2652.2 + (60000.0 - 2652.2) * 0.2120),
        (1, 2000.0, 2000.0),
        (1, 200000.0, 14944.6),
    ],
)
def test_insured_oop_doubles_couples_deductible_and_oop_max(
    is_married: int, total: float, expected: float
) -> None:
    """Couples' deductible and OOP maximum are twice the single ones."""
    result = health_insurance.insured_oop(
        total_health_costs=jnp.asarray(total),
        is_married=jnp.int32(is_married),
        deductible=jnp.asarray(1326.1),
        coinsurance_rate=jnp.asarray(0.2120),
        oop_max=jnp.asarray(7472.3),
    )
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    ("buy_private", "expected"),
    [(BuyPrivate.yes, 2652.2 + (20000.0 - 2652.2) * 0.2120), (BuyPrivate.no, 20000.0)],
)
def test_primary_oop_married_private_buyer(buy_private: int, expected: float) -> None:
    """A married private buyer faces the doubled deductible; uninsured pay all."""
    result = health_insurance.primary_oop(
        total_health_costs=jnp.asarray(20000.0),
        buy_private=jnp.array(buy_private),
        is_married=jnp.int32(1),
        deductible=jnp.asarray(1326.1),
        coinsurance_rate=jnp.asarray(0.2120),
        oop_max=jnp.asarray(7472.3),
    )
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    ("regime", "expected"),
    [
        ("nongroup_nomc_choose_canwork", ("private_cover_paid_in_full",)),
        ("nongroup_dimc_choose_canwork", ()),
        ("nongroup_oamc_choose_canwork", ()),
    ],
)
def test_nongroup_regimes_constrain_private_cover_to_paid_premiums(
    regime: str, expected: tuple[str, ...]
) -> None:
    """Regimes offering private cover require its premium to be paid in full."""
    constraints = _nongroup.build_constraints(REGIME_SPECS[regime])
    assert {name: func.__name__ for name, func in constraints.items()} == {
        name: name for name in expected
    }


@pytest.mark.parametrize(
    ("is_married", "expected"),
    [
        (0, 1109.843 + (10000.0 - 1109.843) * 0.1774133),
        (1, 2219.686 + (10000.0 - 2219.686) * 0.1774133),
    ],
)
def test_aca_primary_oop_doubles_couples_deductible_and_oop_max(
    is_married: int, expected: float
) -> None:
    """Under the ACA plan too, couples' deductible and OOP maximum double."""
    result = aca_health_insurance.primary_oop(
        total_health_costs=jnp.asarray(10000.0),
        cost_sharing_scale=jnp.asarray(1.0),
        buy_private=jnp.array(BuyPrivate.yes),
        is_married=jnp.int32(is_married),
        deductible=jnp.asarray(1109.843),
        coinsurance_rate=jnp.asarray(0.1774133),
        oop_max=jnp.asarray(6500.0),
    )
    np.testing.assert_allclose(result, expected, atol=1e-6)
