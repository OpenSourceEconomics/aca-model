"""Integration test: the benchmark-sized baseline solves + simulates end-to-end."""

import pytest

from aca_model.benchmark import (
    create_benchmark_model,
    get_benchmark_initial_conditions,
    get_benchmark_params,
)


@pytest.mark.long_running
def test_benchmark_model_simulates_end_to_end() -> None:
    n_subjects = 20
    model = create_benchmark_model(n_subjects=n_subjects, max_consumption=300_000.0)
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
