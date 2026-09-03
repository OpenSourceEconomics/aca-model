"""Brute-force vs DC-EGM parity on the benchmark-shaped baseline model.

Tolerances are pre-registered from the brute-force leg's own precision: the
panel distance between solving at the parity grid and at a doubled
consumption grid is the resolution-noise floor, and any DC-EGM deviation is
judged against it. `test_brute_noise_floor_holds` re-derives the floor and
pins the recorded constants; the DC-EGM legs ride the same non-strict xfail
as `test_dcegm_benchmark_model_builds` until pylcm's DC-EGM contract admits
the ACA budget chains.

All tests here solve the 19-regime model twice and are `long_running`; run
them via `pytest -m long_running` (compile cost dominates on CPU — they are
sized for the GPU leg of the acceptance run).
"""

import numpy as np
import pandas as pd
import pytest
from lcm import DiscreteGrid, Model

from aca_model.agent.health import GoodHealth
from aca_model.agent.labor_market import IsMarried
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline.health_insurance import HealthInsuranceState
from aca_model.baseline.model import SolverName, create_model
from aca_model.benchmark import (
    get_benchmark_consumption_dollars_points,
    get_benchmark_initial_conditions,
    get_benchmark_params,
)
from aca_model.config import GridConfig

# The grid every parity leg solves on.
PARITY_GRID_CONFIG = GridConfig(
    n_assets_gridpoints=8,
    n_aime_gridpoints=3,
    n_consumption_dollars_gridpoints=16,
    n_wage_res_gridpoints=3,
    n_hcc_persistent_gridpoints=3,
    n_hcc_transitory_gridpoints=3,
)

# The doubled-consumption brute reference defining the noise floor.
REFERENCE_GRID_CONFIG = GridConfig(
    n_assets_gridpoints=8,
    n_aime_gridpoints=3,
    n_consumption_dollars_gridpoints=32,
    n_wage_res_gridpoints=3,
    n_hcc_persistent_gridpoints=3,
    n_hcc_transitory_gridpoints=3,
)

N_SUBJECTS = 200
SEED = 0

# Pre-registered tolerances, recorded from the noise-floor derivation
# (brute at PARITY_GRID_CONFIG vs brute at REFERENCE_GRID_CONFIG, seed 0,
# 200 subjects). A DC-EGM deviation within these bounds is indistinguishable
# from brute-force resolution noise; bounds are not adjusted after seeing
# DC-EGM results. `None` means not yet recorded — the floor tests skip.
NOISE_FLOOR_ACTION_AGREEMENT: float | None = None  # RECORDED-FROM-DERIVATION
NOISE_FLOOR_VALUE_RDIFF: float | None = None  # RECORDED-FROM-DERIVATION
NOISE_FLOOR_CONSUMPTION_RDIFF: float | None = None  # RECORDED-FROM-DERIVATION


def _recorded_noise_floor() -> tuple[float, float, float]:
    """Return the recorded tolerances, skipping while still unrecorded."""
    if (
        NOISE_FLOOR_ACTION_AGREEMENT is None
        or NOISE_FLOOR_VALUE_RDIFF is None
        or NOISE_FLOOR_CONSUMPTION_RDIFF is None
    ):
        pytest.skip("noise-floor constants not yet recorded from the derivation")
    return (
        NOISE_FLOOR_ACTION_AGREEMENT,
        NOISE_FLOOR_VALUE_RDIFF,
        NOISE_FLOOR_CONSUMPTION_RDIFF,
    )


_DERIVED_CATEGORICALS = {
    "good_health": DiscreteGrid(GoodHealth),
    "is_married": DiscreteGrid(IsMarried),
    "his": DiscreteGrid(HealthInsuranceState),
    "target_his": DiscreteGrid(HealthInsuranceState),
    "pref_type": DiscreteGrid(BenchmarkPrefType),
}

_DISCRETE_COLUMNS = ("regime_name", "claim_ss", "labor_supply", "buy_private")

_XFAIL_UNTIL_DCEGM_ADMITS_THE_ACA_BUDGET = pytest.mark.xfail(
    strict=False,
    reason=(
        "pylcm's DC-EGM contract does not admit the ACA budget: the assets "
        "law reaches `assets` outside the post-decision function — through "
        "`oop_costs` (Medicaid eligibility → `countable_income` → "
        "`capital_income`) and `pension_assets_adjustment` "
        "(`marginal_tax_rate` → `gross_income` → `capital_income`). Flips "
        "with `test_dcegm_benchmark_model_builds`; the fixes land upstream "
        "in pylcm, not here."
    ),
)


def _make_model(*, solver: SolverName, grid_config: GridConfig) -> Model:
    fixed_params, wage_params, _ = get_benchmark_params(model=None)
    # DC-EGM needs the consumption action grid fixed at construction; the
    # brute leg supplies its points at runtime, so only pass them for dcegm.
    consumption_dollars_points = (
        get_benchmark_consumption_dollars_points(
            n_points=grid_config.n_consumption_dollars_gridpoints
        )
        if solver == "dcegm"
        else None
    )
    return create_model(
        n_subjects=N_SUBJECTS,
        fixed_params=fixed_params,
        wage_params=wage_params,
        derived_categoricals=_DERIVED_CATEGORICALS,
        grid_config=grid_config,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver=solver,
        consumption_dollars_points=consumption_dollars_points,
    )


def _seeded_panel(model: Model) -> pd.DataFrame:
    """Solve + simulate the fixed seeded panel and return it in subject order."""
    _, _, params = get_benchmark_params(model=model)
    initial_conditions = get_benchmark_initial_conditions(
        model=model, n_subjects=N_SUBJECTS, seed=SEED
    )
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        log_level="off",
    )
    df = result.to_dataframe()
    return df.sort_values(["subject_id", "period"]).reset_index(drop=True)


def panel_distance(panel_a: pd.DataFrame, panel_b: pd.DataFrame) -> dict[str, float]:
    """Distance between two seeded panels of the same subjects.

    Returns:

    - `agreement_rate`: share of rows whose discrete decisions (regime plus
      every discrete action) coincide. A discrete action that is masked
      (NaN) in the regimes where it does not apply counts as agreeing when
      both panels mask it — identical masking is identical behavior, not a
      disagreement.
    - `prefix_share`: share of rows inside each subject's agreeing prefix —
      after a subject's first discrete disagreement, the paths legitimately
      diverge, so continuous comparisons stop there.
    - `value_rdiff` / `consumption_rdiff`: max relative deviation of the
      recorded value and consumption inside the agreeing prefixes (`0.0`
      when no row shares an agreeing prefix).

    """
    discrete_cols = [c for c in _DISCRETE_COLUMNS if c in panel_a.columns]
    agree_row = np.ones(len(panel_a), dtype=bool)
    for col in discrete_cols:
        col_a, col_b = panel_a[col], panel_b[col]
        both_masked = (col_a.isna() & col_b.isna()).to_numpy()
        equal = col_a.to_numpy(dtype=object) == col_b.to_numpy(dtype=object)
        agree_row &= equal | both_masked

    frame = pd.DataFrame(
        {"subject_id": panel_a["subject_id"].to_numpy(), "agree": agree_row}
    )
    prefix = frame.groupby("subject_id")["agree"].cummin().to_numpy()

    def _max_rdiff(series_a: pd.Series, series_b: pd.Series) -> float:
        left = series_a.to_numpy()[prefix]
        right = series_b.to_numpy()[prefix]
        if left.size == 0:
            return 0.0
        return float(np.nanmax(np.abs(left - right) / np.maximum(np.abs(right), 1e-8)))

    return {
        "agreement_rate": float(agree_row.mean()),
        "prefix_share": float(prefix.mean()),
        "value_rdiff": _max_rdiff(panel_a["value"], panel_b["value"]),
        "consumption_rdiff": _max_rdiff(
            panel_a["consumption_dollars"], panel_b["consumption_dollars"]
        ),
    }


@pytest.mark.long_running
def test_brute_noise_floor_holds() -> None:
    """The recorded noise-floor constants reproduce within a small margin.

    Guards the tolerance derivation itself: brute at the parity grid vs
    brute at the doubled-consumption reference must stay at least as close
    as when the constants were recorded.
    """
    agreement_floor, value_floor, consumption_floor = _recorded_noise_floor()
    panel_lo = _seeded_panel(
        _make_model(solver="brute_force", grid_config=PARITY_GRID_CONFIG)
    )
    panel_hi = _seeded_panel(
        _make_model(solver="brute_force", grid_config=REFERENCE_GRID_CONFIG)
    )
    distance = panel_distance(panel_lo, panel_hi)
    assert distance["agreement_rate"] >= agreement_floor
    assert distance["value_rdiff"] <= value_floor
    assert distance["consumption_rdiff"] <= consumption_floor


@pytest.mark.long_running
@_XFAIL_UNTIL_DCEGM_ADMITS_THE_ACA_BUDGET
def test_dcegm_panel_within_brute_noise_floor() -> None:
    """The DC-EGM seeded panel deviates from brute force by no more than
    brute force deviates from its own doubled-consumption reference."""
    agreement_floor, value_floor, consumption_floor = _recorded_noise_floor()
    panel_brute = _seeded_panel(
        _make_model(solver="brute_force", grid_config=PARITY_GRID_CONFIG)
    )
    panel_dcegm = _seeded_panel(
        _make_model(solver="dcegm", grid_config=PARITY_GRID_CONFIG)
    )
    distance = panel_distance(panel_brute, panel_dcegm)
    assert distance["agreement_rate"] >= agreement_floor
    assert distance["value_rdiff"] <= value_floor
    assert distance["consumption_rdiff"] <= consumption_floor


@pytest.mark.long_running
@_XFAIL_UNTIL_DCEGM_ADMITS_THE_ACA_BUDGET
def test_dcegm_solves_the_parity_model() -> None:
    """`solver="dcegm"` solves the parity-grid model: every active
    (regime, period) cell carries a finite value-function array."""
    model = _make_model(solver="dcegm", grid_config=PARITY_GRID_CONFIG)
    _, _, params = get_benchmark_params(model=model)
    solution = model.solve(params=params, log_level="off")
    for regime_to_v in solution.values.values():
        for v_arr in regime_to_v.values():
            assert bool(np.isfinite(np.asarray(v_arr)).any())
