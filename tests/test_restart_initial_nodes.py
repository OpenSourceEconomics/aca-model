"""Starts mid-life, at the regimes a simulation can have reached by that age."""

import pandas as pd
import pytest
from lcm import DiscreteGrid

from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline.regimes import build_restart_initial_nodes
from aca_model.benchmark import create_benchmark_model
from aca_model.simulation import select_admissible_starts


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (
            61,
            (
                "retiree_nomc_inelig_canwork",
                "tied_nomc_inelig_canwork",
                "nongroup_nomc_inelig_canwork",
                "retiree_dimc_inelig_canwork",
                "nongroup_dimc_inelig_canwork",
            ),
        ),
        (
            66,
            (
                "retiree_oamc_choose_canwork",
                "tied_oamc_choose_canwork",
                "nongroup_oamc_choose_canwork",
            ),
        ),
        (
            73,
            ("retiree_oamc_forced_forcedout", "nongroup_oamc_forced_forcedout"),
        ),
    ],
)
def test_build_restart_initial_nodes_admits_every_living_regime_of_the_age(
    age: int, expected: tuple[str, ...]
) -> None:
    """At each age, the starts are exactly the living regimes active then."""
    nodes = build_restart_initial_nodes(ages=(age,))
    assert set(nodes.by_age[age]) == set(expected)


def test_model_with_restart_nodes_admits_a_start_on_old_age_medicare() -> None:
    """A model built with restart nodes accepts a start at 66 on Medicare."""
    model = create_benchmark_model(
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        initial_nodes=build_restart_initial_nodes(ages=range(51, 70)),
    )
    starts = pd.DataFrame({"age": [66.0], "regime_name": ["tied_oamc_choose_canwork"]})
    admitted = select_admissible_starts(model=model, initial_conditions=starts)
    assert admitted["regime_name"].tolist() == ["tied_oamc_choose_canwork"]
