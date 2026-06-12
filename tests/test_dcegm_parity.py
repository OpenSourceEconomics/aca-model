"""Brute-force vs DC-EGM parity on the benchmark-shaped baseline model.

Tolerances are pre-registered from the brute-force leg's own precision: the
panel distance between solving at the parity grid and at a doubled
consumption grid is the resolution-noise floor, and any DC-EGM deviation is
judged against it. `test_brute_noise_floor_holds` re-derives the floor and
pins the recorded constants; the DC-EGM legs ride the same non-strict xfail
as `test_dcegm_benchmark_model_builds` until pylcm's savings-stage contract
admits smooth Euler-state-dependent regime transitions per node.

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

_XFAIL_UNTIL_SAVINGS_STAGE_CONTRACT = pytest.mark.xfail(
    strict=False,
    reason=(
        "pylcm's savings-stage rule does not yet admit Euler-state-dependent "
        "regime transitions: the build dies in "
        "`_fail_if_savings_stage_function_depends_on_decision` because the "
        "regime transition probabilities read `assets` through the smoothed "
        "`medicaid_eligibility_share`. Flips with "
        "`test_dcegm_benchmark_model_builds` once the per-node "
        "smooth-savings-stage extension lands upstream; the parity grid "
        "resolves the SSI bands with dedicated assets nodes."
    ),
)


def _make_model(*, solver: SolverName, grid_config: GridConfig) -> Model:
    fixed_params, wage_params, _ = get_benchmark_params(model=None)
    return create_model(
        n_subjects=N_SUBJECTS,
        fixed_params=fixed_params,
        wage_params=wage_params,
        derived_categoricals=_DERIVED_CATEGORICALS,
        grid_config=grid_config,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver=solver,
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
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    df = result.to_dataframe()
    return df.sort_values(["subject_id", "period"]).reset_index(drop=True)


def panel_distance(panel_a: pd.DataFrame, panel_b: pd.DataFrame) -> dict[str, float]:
    """Distance between two seeded panels of the same subjects.

    Returns:

    - `agreement_rate`: share of rows whose discrete decisions (regime plus
      every discrete action) coincide.
    - `prefix_share`: share of rows inside each subject's agreeing prefix —
      after a subject's first discrete disagreement, the paths legitimately
      diverge, so continuous comparisons stop there.
    - `value_rdiff` / `consumption_rdiff`: max relative deviation of the
      recorded value and consumption inside the agreeing prefixes.

    """
    discrete_cols = [c for c in _DISCRETE_COLUMNS if c in panel_a.columns]
    agree_row = np.ones(len(panel_a), dtype=bool)
    for col in discrete_cols:
        agree_row &= panel_a[col].to_numpy(dtype=object) == panel_b[col].to_numpy(
            dtype=object
        )

    frame = pd.DataFrame(
        {"subject_id": panel_a["subject_id"].to_numpy(), "agree": agree_row}
    )
    prefix = frame.groupby("subject_id")["agree"].cummin().to_numpy()

    value_a = panel_a["value"].to_numpy()[prefix]
    value_b = panel_b["value"].to_numpy()[prefix]
    cons_a = panel_a["consumption_dollars"].to_numpy()[prefix]
    cons_b = panel_b["consumption_dollars"].to_numpy()[prefix]
    return {
        "agreement_rate": float(agree_row.mean()),
        "prefix_share": float(prefix.mean()),
        "value_rdiff": float(
            np.nanmax(np.abs(value_a - value_b) / np.maximum(np.abs(value_b), 1e-8))
        ),
        "consumption_rdiff": float(
            np.nanmax(np.abs(cons_a - cons_b) / np.maximum(np.abs(cons_b), 1e-8))
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
@_XFAIL_UNTIL_SAVINGS_STAGE_CONTRACT
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
@_XFAIL_UNTIL_SAVINGS_STAGE_CONTRACT
def test_dcegm_solves_the_parity_model() -> None:
    """`solver="dcegm"` solves the parity-grid model: every active
    (regime, period) cell carries a finite value-function array."""
    model = _make_model(solver="dcegm", grid_config=PARITY_GRID_CONFIG)
    _, _, params = get_benchmark_params(model=model)
    period_to_regime_to_v = model.solve(params=params, log_level="off")
    for regime_to_v in period_to_regime_to_v.values():
        for v_arr in regime_to_v.values():
            assert bool(np.isfinite(np.asarray(v_arr)).any())
