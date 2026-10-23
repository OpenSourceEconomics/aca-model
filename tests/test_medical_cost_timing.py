"""Medical costs are realised after the period's choices.

The state a household carries into period t holds last year's persistent
shock and last year's transitory shock. This year's shocks are drawn after
consumption, work, claiming and insurance are chosen; the resulting
out-of-pocket bill is taken off end-of-period assets. So the value and every
choice in period t depend on the lagged persistent shock, which predicts this
year's costs, and not at all on the lagged transitory shock.
"""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from lcm import DiscreteGrid

from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.benchmark import (
    create_benchmark_model,
    get_benchmark_initial_conditions,
    get_benchmark_params,
)
from aca_model.config import BENCHMARK_GRID_CONFIG

_SHOCKS = ("hcc_persistent", "hcc_transitory")
# Period 0's choices look one year ahead, so ages 51-53 (two living periods and
# the age at which everybody is dead) carry the whole timing question.
_FINAL_AGE = 53
# The twins differ in one lagged shock at its grid ends, so only the two shock
# grids keep the benchmark resolution and the transitory shock stays a state.
# Every other grid is at the smallest size the model accepts, except that
# consumption keeps an interior point above the two floors and below the cap,
# so the consumption comparison still allows a choice that cash would move:
# - assets: borrowing limit, zero and the top holding;
# - AIME: two points per PIA segment;
# - wage residual: two points.
_GRID_CONFIG = replace(
    BENCHMARK_GRID_CONFIG,
    fold_hcc_transitory=False,
    subtract_hcc_bill=False,
    n_assets_gridpoints=3,
    aime_points_per_segment=(2, 2, 2, 2),
    n_consumption_dollars_gridpoints=4,
    n_wage_res_gridpoints=2,
)


@pytest.fixture(scope="module")
def period_0_by_shock() -> dict[tuple[str, str], pd.Series]:
    """Period-0 rows of twins that differ in one lagged shock, at its grid ends."""
    grid_config = _GRID_CONFIG
    model = create_benchmark_model(
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        grid_config=grid_config,
        final_age=_FINAL_AGE,
    )
    _, _, params = get_benchmark_params(model=model)
    base = get_benchmark_initial_conditions(
        model=model, n_subjects=1, seed=0, grid_config=grid_config
    )
    grids = model.user_regimes["retiree_nomc_inelig_canwork"].states
    initial_conditions = {name: jnp.tile(values, 4) for name, values in base.items()}
    for row, shock in enumerate(_SHOCKS):
        points = np.asarray(grids[shock].to_jax())  # ty: ignore[unresolved-attribute]
        values = np.asarray(initial_conditions[shock]).copy()
        values[2 * row : 2 * row + 2] = (points.min(), points.max())
        initial_conditions[shock] = jnp.asarray(values)
    result = model.simulate(
        params=params, initial_conditions=initial_conditions, log_level="off"
    )
    df = result.to_dataframe(terminal_rows="all")
    period_0 = df.loc[df["period"] == 0].sort_values("subject_id")
    return {
        (shock, end): period_0.iloc[2 * row + i]
        for row, shock in enumerate(_SHOCKS)
        for i, end in enumerate(("low", "high"))
    }


def test_value_does_not_depend_on_lagged_transitory_shock(
    period_0_by_shock: dict[tuple[str, str], pd.Series],
) -> None:
    """Twins differing only in last year's transitory shock have equal value."""
    low = period_0_by_shock["hcc_transitory", "low"]["value"]
    high = period_0_by_shock["hcc_transitory", "high"]["value"]
    assert low == high


def test_consumption_does_not_depend_on_lagged_transitory_shock(
    period_0_by_shock: dict[tuple[str, str], pd.Series],
) -> None:
    """Twins differing only in last year's transitory shock consume the same."""
    low = period_0_by_shock["hcc_transitory", "low"]["consumption_dollars"]
    high = period_0_by_shock["hcc_transitory", "high"]["consumption_dollars"]
    assert low == high


def test_value_falls_with_lagged_persistent_shock(
    period_0_by_shock: dict[tuple[str, str], pd.Series],
) -> None:
    """A higher lagged persistent shock predicts higher costs and lowers value."""
    low = period_0_by_shock["hcc_persistent", "low"]["value"]
    high = period_0_by_shock["hcc_persistent", "high"]["value"]
    assert high < low
