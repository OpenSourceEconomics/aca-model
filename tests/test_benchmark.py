"""Integration test: the benchmark-sized baseline solves + simulates end-to-end."""

import numpy as np
import pytest
from lcm import DiscreteGrid

from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.benchmark import (
    create_benchmark_model,
    get_benchmark_initial_conditions,
    get_benchmark_params,
)


@pytest.mark.long_running
def test_benchmark_model_simulates_end_to_end() -> None:
    n_subjects = 20
    model = create_benchmark_model(
        n_subjects=n_subjects,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )
    _, _, params = get_benchmark_params(model=model)
    initial_conditions = get_benchmark_initial_conditions(
        model=model, n_subjects=n_subjects, seed=0
    )

    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
    )

    df = result.to_dataframe()
    assert len(df) == n_subjects * model.n_periods
    # Period 0 rows reflect initial conditions — no NaN in continuous states.
    period_0 = df.loc[df["period"] == 0]
    assert not period_0[["assets", "aime", "value"]].isna().any().any()


@pytest.mark.long_running
def test_benchmark_panel_exposes_hic_premium_and_wage_targets() -> None:
    """`hic_premium` and `wage` are computable simulate targets.

    The MSM estimation matches `hic_private_premium_*` and `wage_growth_*`
    moments, so the simulated panel must carry the premium each agent pays
    (defined in every living regime) and the wage rate of workers (defined
    in can-work regimes). Both are DAG functions exposed through
    `to_dataframe(additional_targets=...)`; dead/forced-out rows that lack a
    target get NaN, alive/worker rows carry a finite value.
    """
    n_subjects = 20
    model = create_benchmark_model(
        n_subjects=n_subjects,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )
    _, _, params = get_benchmark_params(model=model)
    initial_conditions = get_benchmark_initial_conditions(
        model=model, n_subjects=n_subjects, seed=0
    )

    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
    )

    df = result.to_dataframe(additional_targets=["hic_premium", "wage"])
    assert {"hic_premium", "wage"}.issubset(df.columns)

    # Every alive agent pays a finite, non-negative premium.
    alive = df.loc[df["regime_name"] != "dead"]
    assert alive["hic_premium"].notna().all()
    assert (alive["hic_premium"].to_numpy() >= 0).all()

    # The wage rate is exp(...) > 0 wherever it is defined (can-work rows),
    # and it is defined for at least some of the panel.
    defined_wage = df.loc[df["wage"].notna(), "wage"].to_numpy()
    assert len(defined_wage) > 0
    assert (defined_wage > 0).all()


@pytest.mark.long_running
def test_benchmark_simulate_obeys_borrowing_constraint() -> None:
    """`consumption_dollars <= max(cash_on_hand, floor)` holds for every alive row.

    The simulator only ever picks feasible actions — the borrowing
    constraint must hold post-hoc on the simulated panel. A regression
    that drops the constraint from a regime, replaces the floor with
    something looser, or lets an action grid skip the floor would
    surface as a row with `consumption_dollars > max(cash_on_hand, floor)`.

    The constraint's RHS is `max(cash_on_hand, floor)` rather than
    `cash_on_hand + transfers`: the additive form rounds short by
    sub-ULP at extreme `|cash_on_hand|`, so the post-hoc check would
    also flip on the same kink.

    Scope — this is the CPU/exact-arithmetic spec, and holds exactly
    there. On GPU at coarse asset grids, a handful of mid-wealth rows can
    show `consumption_dollars > max(cash_on_hand, floor)`: the consumption
    policy interpolated across sparse asset nodes overshoots the exact
    simulated budget by one action-grid cell. That is brute-force
    coarse-grid interpolation noise (the same low-resolution unreliability
    the DC-EGM oracle comparisons exclude), and it disappears at
    production asset resolution — it is not a feasibility regression.

    Such overshoot rows are NOT the consumption-floor segment, and
    must not be mistaken for it: `consumption_dollars_floor` pins the
    single/married floor onto the lowest action-grid points, whereas the
    overshooting rows sit at cash-on-hand far above the floor (so
    `max(cash_on_hand, floor) = cash_on_hand`) and pick an interior
    consumption point, not the floor.
    """
    n_subjects = 4
    model = create_benchmark_model(
        n_subjects=n_subjects,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )
    _, _, params = get_benchmark_params(model=model)
    initial_conditions = get_benchmark_initial_conditions(
        model=model, n_subjects=n_subjects, seed=0
    )

    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
    )

    df = result.to_dataframe(additional_targets=["cash_on_hand", "equivalence_scale"])
    alive = df.loc[df["regime_name"] != "dead"].copy()
    consumption_equiv_floor = float(params["consumption_equiv_floor"])
    floor = consumption_equiv_floor * alive["equivalence_scale"].to_numpy()
    rhs = np.maximum(alive["cash_on_hand"].to_numpy(), floor)
    slack = rhs - alive["consumption_dollars"].to_numpy()
    assert (slack >= 0).all(), (
        f"borrowing_constraint violated on {int((slack < 0).sum())} row(s); "
        f"min slack = {slack.min():.6g}"
    )
