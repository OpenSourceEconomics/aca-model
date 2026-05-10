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
    """`consumption_unequiv <= max(cash_on_hand, floor)` holds for every alive row.

    The simulator only ever picks feasible actions — the borrowing
    constraint must hold post-hoc on the simulated panel. A regression
    that drops the constraint from a regime, replaces the floor with
    something looser, or lets an action grid skip the floor would
    surface as a row with `consumption_unequiv > max(cash_on_hand, floor)`.

    The constraint's RHS is `max(cash_on_hand, floor)` rather than
    `cash_on_hand + transfers`: the additive form rounds short by
    sub-ULP at extreme `|cash_on_hand|`, so the post-hoc check would
    also flip on the same kink.
    """
    import numpy as np

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

    df = result.to_dataframe(
        additional_targets=["cash_on_hand", "equivalence_scale"]
    )
    alive = df.loc[df["regime"] != "dead"].copy()
    consumption_unequiv_floor = float(params["consumption_unequiv_floor"])
    floor = consumption_unequiv_floor * alive["equivalence_scale"].to_numpy()
    rhs = np.maximum(alive["cash_on_hand"].to_numpy(), floor)
    slack = rhs - alive["consumption_unequiv"].to_numpy()
    assert (slack >= 0).all(), (
        f"borrowing_constraint violated on {int((slack < 0).sum())} row(s); "
        f"min slack = {slack.min():.6g}"
    )
