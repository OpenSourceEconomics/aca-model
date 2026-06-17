"""SSI/Medicaid eligibility is a smooth share, not a boolean cliff.

Eligibility enters the model as a C² share in [0, 1]: exactly 1 below the
statutory threshold band, exactly 0 above it, and a quintic smoothstep ramp
inside the +/- `ELIGIBILITY_BAND_HALF_WIDTH` band around the threshold.
Every consumer of eligibility is a share-weighted mixture of the eligible
and ineligible branches, so the budget chain stays differentiable in
`assets` — the contract DC-EGM's per-node evaluation of savings-stage
functions requires. Both household thresholds (`ssi_assets_test` indexed by
`spousal_income`) get the same treatment, including dedicated assets-grid
nodes across each band.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from lcm import DiscreteGrid
from lcm.params import MappingLeaf

from aca_model.aca import health_insurance as aca_hi
from aca_model.agent.labor_market import LaborSupply
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline import health_insurance
from aca_model.baseline.health_insurance import HealthInsuranceState
from aca_model.baseline.regimes._common import (
    build_grids,
    build_regime_probs,
    make_targets,
    select_target_for_age,
)
from aca_model.baseline.regimes._retiree import _make_transition_canwork
from aca_model.benchmark import get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG, GridConfig
from aca_model.environment import pensions

ATOL = 1e-6

SSI_ASSETS_TEST = jnp.array([2000.0, 3000.0, 3000.0])
SSI_MAX_BENEFIT = jnp.array([8000.0, 12000.0, 12000.0])


def _ssi_share(
    *,
    assets: float,
    countable_income: float = 0.0,
    spousal_income: int = 0,
    crossed_oamc_threshold: bool = True,
    is_disabled: bool = False,
) -> jnp.ndarray:
    return health_insurance.ssi_eligibility_share(
        assets=jnp.array(assets),
        countable_income=jnp.array(countable_income),
        spousal_income=jnp.int32(spousal_income),
        crossed_oamc_threshold=jnp.asarray(crossed_oamc_threshold),
        is_disabled=jnp.asarray(is_disabled),
        ssi_assets_test=SSI_ASSETS_TEST,
        ssi_maximum_benefit=SSI_MAX_BENEFIT,
    )


def test_band_half_width_is_preregistered() -> None:
    """The smoothing band is +/- $50 around the statutory threshold.

    Pre-registered, never adjusted after seeing solver results; sensitivity
    runs vary it explicitly.
    """
    assert health_insurance.ELIGIBILITY_BAND_HALF_WIDTH == 50.0


@pytest.mark.parametrize(
    ("assets", "spousal_income", "expected"),
    [
        (1950.0, 0, 1.0),
        (2000.0, 0, 0.5),
        (2050.0, 0, 0.0),
        (1000.0, 0, 1.0),
        (5000.0, 0, 0.0),
        (2950.0, 1, 1.0),
        (3000.0, 1, 0.5),
        (3050.0, 1, 0.0),
        (2500.0, 1, 1.0),
    ],
)
def test_ssi_share_assets_leg(
    assets: float, spousal_income: int, expected: float
) -> None:
    """Share is 1 below the band, 0.5 at the threshold, 0 above the band.

    The household-specific threshold (`ssi_assets_test[spousal_income]`)
    centers the band: $2,000 for singles, $3,000 for couples.
    """
    share = _ssi_share(assets=assets, spousal_income=spousal_income)
    np.testing.assert_allclose(share, expected, atol=ATOL)


def test_ssi_share_assets_ramp_is_quintic() -> None:
    """The ramp takes off with zero slope and curvature at the band edge.

    $1 into the band the quintic smoothstep has moved less than 1e-4 (a
    linear ramp would move 0.01, a cubic smoothstep 3e-4).
    """
    share = _ssi_share(assets=1951.0)
    assert share > 1.0 - 1e-4


@pytest.mark.parametrize(
    ("countable_income", "expected"),
    [(7950.0, 1.0), (8000.0, 0.5), (8050.0, 0.0)],
)
def test_ssi_share_income_leg(countable_income: float, expected: float) -> None:
    """The income test smooths the same way around `ssi_maximum_benefit`."""
    share = _ssi_share(assets=0.0, countable_income=countable_income)
    np.testing.assert_allclose(share, expected, atol=ATOL)


def test_ssi_share_zero_without_categorical_gate() -> None:
    """Failing the aged-or-disabled gate gives share exactly 0 inside the band."""
    share = _ssi_share(assets=2000.0, crossed_oamc_threshold=False, is_disabled=False)
    np.testing.assert_allclose(share, 0.0, atol=ATOL)


def test_ssi_share_monotone_nonincreasing_in_assets() -> None:
    """Sweeping assets through both bands never increases the share."""
    assets = jnp.linspace(0.0, 5000.0, 2001)
    shares = health_insurance.ssi_eligibility_share(
        assets=assets,
        countable_income=jnp.zeros_like(assets),
        spousal_income=jnp.int32(0),
        crossed_oamc_threshold=jnp.asarray(True),
        is_disabled=jnp.asarray(False),
        ssi_assets_test=SSI_ASSETS_TEST,
        ssi_maximum_benefit=SSI_MAX_BENEFIT,
    )
    assert bool(jnp.all(jnp.diff(shares) <= ATOL))


def test_medicaid_share_equals_ssi_share_in_baseline() -> None:
    """Baseline Medicaid eligibility is the SSI share, unchanged."""
    share = health_insurance.medicaid_eligibility_share(
        ssi_eligibility_share=jnp.array(0.37)
    )
    np.testing.assert_allclose(share, 0.37, atol=ATOL)


@pytest.mark.parametrize(
    ("share", "expected"),
    [(1.0, 5000.0), (0.0, 0.0), (0.5, 2500.0)],
)
def test_ssi_benefit_is_share_weighted(share: float, expected: float) -> None:
    """`ssi_benefit = share * max(0, max_benefit - countable_income)`."""
    benefit = health_insurance.ssi_benefit(
        countable_income=jnp.array(3000.0),
        spousal_income=jnp.int32(0),
        ssi_eligibility_share=jnp.array(share),
        ssi_maximum_benefit=SSI_MAX_BENEFIT,
    )
    np.testing.assert_allclose(benefit, expected, atol=ATOL)


@pytest.mark.parametrize(
    ("share", "expected"),
    [
        # primary_oop 2400 -> medicaid schedule (100 ded, 5% coins, 1000 max)
        # gives min(100 + 2300 * 0.05, 1000) = 215.
        (1.0, 215.0),
        (0.0, 2400.0),
        (0.5, (215.0 + 2400.0) / 2),
    ],
)
def test_oop_with_medicaid_is_share_weighted(share: float, expected: float) -> None:
    """OOP is the share-weighted mix of Medicaid-covered and primary OOP."""
    oop = health_insurance.oop_with_medicaid(
        primary_oop=jnp.array(2400.0),
        medicaid_eligibility_share=jnp.array(share),
        deductible_medicaid=jnp.asarray(100.0),
        coinsurance_rate_medicaid=jnp.asarray(0.05),
        oop_max_medicaid=jnp.asarray(1000.0),
    )
    np.testing.assert_allclose(oop, expected, atol=ATOL)


@pytest.mark.parametrize(
    ("his", "labor_supply", "expected"),
    [
        (
            HealthInsuranceState.tied,
            LaborSupply.do_not_work,
            HealthInsuranceState.nongroup,
        ),
        (HealthInsuranceState.tied, LaborSupply.h2000, HealthInsuranceState.tied),
        (
            HealthInsuranceState.retiree,
            LaborSupply.do_not_work,
            HealthInsuranceState.retiree,
        ),
    ],
)
def test_target_his_reflects_only_the_tied_to_nongroup_move(
    his: int, labor_supply: int, expected: int
) -> None:
    """`target_his` is the deterministic target HIS: tied agents who stop
    working become nongroup; the Medicaid path is a probability, not a
    deterministic override, and enters via the imputation mixture instead."""
    result = health_insurance.target_his(
        his=jnp.int32(his),
        labor_supply=jnp.array(labor_supply),
    )
    assert int(result) == int(expected)


def test_target_his_forcedout_is_own_his() -> None:
    """Forced-out regimes keep their own HIS as the deterministic target."""
    result = health_insurance.target_his_forcedout(
        his=jnp.int32(HealthInsuranceState.retiree)
    )
    assert int(result) == int(HealthInsuranceState.retiree)


_IMP_TABLES_HIS = {
    "imp_intercept_next_period": jnp.array([[0.0, 0.0, 0.0], [5.0, 10.0, 7.0]]),
    "imp_pia_coeff_next_period": jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 1.5]]),
    "imp_pia_kink_0_coeff_next_period": jnp.zeros((2, 3)),
    "imp_pia_kink_1_coeff_next_period": jnp.zeros((2, 3)),
    "imp_kink_0_next_period": jnp.array([0.0, 0.0]),
    "imp_kink_1_next_period": jnp.array([0.0, 0.0]),
    "epdv_constant_pension_next_period": jnp.array([0.0, 3.0]),
}


def test_imputed_pension_wealth_no_medicaid_looks_up_target_his() -> None:
    """The deterministic-target leg imputes at `[period, target_his]`.

    pia_unadjusted = 50, intercept = 10, coeff = 2 -> pbmax = 110, epdv = 3.
    """
    result = pensions.imputed_pension_wealth_next_period_no_medicaid(
        pia_unadjusted_next_period=jnp.array(50.0),
        target_his=jnp.int32(HealthInsuranceState.tied),
        period=jnp.int32(1),
        **_IMP_TABLES_HIS,
    )
    np.testing.assert_allclose(result, 330.0, atol=ATOL)


def test_imputed_pension_wealth_medicaid_uses_nongroup_slices() -> None:
    """The Medicaid leg imputes from age-indexed nongroup table slices.

    pia_unadjusted = 50, intercept_ng = 20, coeff_ng = 1 -> pbmax = 70, epdv = 3.
    """
    result = pensions.imputed_pension_wealth_next_period_medicaid(
        pia_unadjusted_next_period=jnp.array(50.0),
        period=jnp.int32(1),
        imp_intercept_next_period_ng=jnp.array([0.0, 20.0]),
        imp_pia_coeff_next_period_ng=jnp.array([0.0, 1.0]),
        imp_pia_kink_0_coeff_next_period_ng=jnp.zeros(2),
        imp_pia_kink_1_coeff_next_period_ng=jnp.zeros(2),
        imp_kink_0_next_period=jnp.array([0.0, 0.0]),
        imp_kink_1_next_period=jnp.array([0.0, 0.0]),
        epdv_constant_pension_next_period=jnp.array([0.0, 3.0]),
    )
    np.testing.assert_allclose(result, 210.0, atol=ATOL)


def test_imputed_pension_wealth_next_period_is_share_mix() -> None:
    """The imputation feeding `assets_adjustment` mixes the two legs:
    `share * medicaid + (1 - share) * no_medicaid`."""
    result = pensions.imputed_pension_wealth_next_period(
        imputed_pension_wealth_next_period_no_medicaid=jnp.array(330.0),
        imputed_pension_wealth_next_period_medicaid=jnp.array(210.0),
        medicaid_eligibility_share=jnp.array(0.25),
    )
    np.testing.assert_allclose(result, 0.75 * 330.0 + 0.25 * 210.0, atol=ATOL)


def test_with_nongroup_imputation_slices_extracts_age_indexed_tables() -> None:
    """Each his-indexed imputation table gains an age-indexed `_ng` sibling
    holding its `target_his="nongroup"` slice; original keys are kept."""
    index = pd.MultiIndex.from_product(
        [[51, 52], ["retiree", "tied", "nongroup"]], names=["age", "target_his"]
    )
    sr = pd.Series(np.arange(6, dtype=float), index=index)
    fixed_params = {
        "imp_intercept_next_period": sr,
        "imp_pia_coeff_next_period": sr * 2,
        "imp_pia_kink_0_coeff_next_period": sr * 3,
        "imp_pia_kink_1_coeff_next_period": sr * 4,
        "unrelated": 1.0,
    }
    result = pensions.with_nongroup_imputation_slices(fixed_params)
    expected_ng = sr.xs("nongroup", level="target_his")
    pd.testing.assert_series_equal(result["imp_intercept_next_period_ng"], expected_ng)
    assert set(fixed_params) < set(result)


def test_regime_transition_mixes_probability_mass_by_share() -> None:
    """With share s, survival mass splits s to the nongroup-SSI target and
    1 - s to the own target; the dead mass is unaffected."""
    own, ng = make_targets("retiree_dimc_inelig_canwork")
    transition = _make_transition_canwork(True, own, ng)
    survival_probs = jnp.array([0.9])
    probs = transition(
        age=jnp.int32(60),
        period=jnp.int32(0),
        labor_supply=jnp.array(LaborSupply.do_not_work),
        medicaid_eligibility_share=jnp.array(0.5),
        survival_probs=survival_probs,
    )
    own_target = select_target_for_age(61, True, own)
    ng_target = select_target_for_age(61, True, ng)
    survival = jnp.array(0.9)
    expected = 0.5 * build_regime_probs(ng_target, survival) + 0.5 * build_regime_probs(
        own_target, survival
    )
    np.testing.assert_allclose(probs, expected, atol=ATOL)


def test_regime_transition_at_share_zero_matches_discrete_target() -> None:
    """At share 0 the mixture collapses to the deterministic own target."""
    own, ng = make_targets("retiree_dimc_inelig_canwork")
    transition = _make_transition_canwork(True, own, ng)
    probs = transition(
        age=jnp.int32(60),
        period=jnp.int32(0),
        labor_supply=jnp.array(LaborSupply.h2000),
        medicaid_eligibility_share=jnp.array(0.0),
        survival_probs=jnp.array([0.9]),
    )
    own_target = select_target_for_age(61, False, own)
    np.testing.assert_allclose(
        probs, build_regime_probs(own_target, jnp.array(0.9)), atol=ATOL
    )


def test_aca_medicaid_share_smooths_the_income_threshold() -> None:
    """The ACA expansion share ramps over the same +/- $50 band around the
    133%-FPL MAGI threshold (income-only expansion track, no categorical share)."""
    schedule = MappingLeaf(
        data={"income_threshold": jnp.array([15000.0, 20000.0, 20000.0])}
    )
    shares = [
        aca_hi.medicaid_eligibility_share(
            ssi_eligibility_share=jnp.array(0.0),
            aca_magi=jnp.array(income),
            spousal_income=jnp.int32(0),
            crossed_oamc_threshold=jnp.asarray(False),
            is_disabled=jnp.asarray(False),
            medicaid_schedule=schedule,
        )
        for income in (14950.0, 15000.0, 15050.0)
    ]
    np.testing.assert_allclose(np.array(shares), [1.0, 0.5, 0.0], atol=ATOL)


def test_assets_grid_has_dedicated_nodes_across_both_bands() -> None:
    """The assets grid carries nodes at threshold and both band edges for
    every distinct `ssi_assets_test` value, on top of the configured base
    resolution — a band without nodes is a cliff at node resolution."""
    fixed_params, wage_params, _ = get_benchmark_params(model=None)
    grids = build_grids(
        grid_config=GridConfig(),
        fixed_params=fixed_params,
        wage_params=wage_params,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )
    points = np.asarray(grids.assets.to_jax())
    band_nodes = {1950.0, 2000.0, 2050.0, 2950.0, 3000.0, 3050.0}
    missing = {n for n in band_nodes if not np.isclose(points, n).any()}
    assert not missing, f"assets grid lacks band nodes: {sorted(missing)}"


def test_assets_grid_band_nodes_skipped_below_minimum_budget() -> None:
    """Tiny smoke-test grids keep the plain linspaced assets grid: with
    fewer than `MIN_ASSETS_GRIDPOINTS_FOR_BAND_NODES` points there is no
    base resolution worth refining."""
    fixed_params, wage_params, _ = get_benchmark_params(model=None)
    grids = build_grids(
        grid_config=BENCHMARK_GRID_CONFIG,
        fixed_params=fixed_params,
        wage_params=wage_params,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )
    assert int(grids.assets.n_points) == BENCHMARK_GRID_CONFIG.n_assets_gridpoints
