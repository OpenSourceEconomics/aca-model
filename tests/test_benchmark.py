"""Integration test: the benchmark-sized baseline solves + simulates end-to-end."""

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
    _, params = get_benchmark_params(model=model)
    initial_conditions = get_benchmark_initial_conditions(
        model=model, n_subjects=n_subjects, seed=0
    )

    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
        check_initial_conditions=False,
    )

    df = result.to_dataframe()
    assert len(df) == n_subjects * model.n_periods
    # Period 0 rows reflect initial conditions — no NaN in continuous states.
    period_0 = df.loc[df["period"] == 0]
    assert not period_0[["assets", "aime", "value"]].isna().any().any()


@pytest.mark.long_running
def test_benchmark_simulate_obeys_borrowing_constraint() -> None:
    """`consumption <= cash_on_hand + transfers` holds for every alive row.

    The simulator only ever picks feasible actions — the borrowing
    constraint must hold post-hoc on the simulated panel. A regression
    that drops the constraint from a regime, replaces transfers with
    something looser, or lets an action grid skip the floor would
    surface as a row with `consumption > cash_on_hand + transfers`.
    """
    n_subjects = 4
    model = create_benchmark_model(
        n_subjects=n_subjects,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )
    _, params = get_benchmark_params(model=model)
    initial_conditions = get_benchmark_initial_conditions(
        model=model, n_subjects=n_subjects, seed=0
    )

    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
        check_initial_conditions=False,
    )

    df = result.to_dataframe(additional_targets=["cash_on_hand", "transfers"])
    alive = df.loc[df["regime"] != "dead"].copy()
    slack = (alive["cash_on_hand"] + alive["transfers"]) - alive["consumption"]
    # Non-negative within fp64 tolerance; allow 1e-6 of the magnitude scale
    # to absorb the float64 rounding budget.
    assert (slack >= -1e-6).all(), (
        f"borrowing_constraint violated on "
        f"{int((slack < -1e-6).sum())} row(s); "
        f"min slack = {slack.min():.6g}"
    )
