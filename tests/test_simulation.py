"""Tests for the pylcm simulate-interaction helpers."""

import numpy as np
import pandas as pd

from aca_model.simulation import restore_subject_ids, simulate_with_dense_index


class _StubModel:
    """Records the initial-conditions index it is simulated with."""

    def __init__(self) -> None:
        self.seen_index: list[int] = []

    def simulate(self, *, initial_conditions: pd.DataFrame, **_kwargs: object) -> str:
        self.seen_index = list(initial_conditions.index)
        return "RESULT"


def test_simulate_with_dense_index_hands_pylcm_a_dense_range() -> None:
    """The simulator receives a dense 0..n-1 index regardless of the caller's ids.

    pylcm indexes the initial conditions by their index labels and assigns
    `subject_id` positionally, so a sparse id index would index out of bounds.
    """
    ic = pd.DataFrame(
        {"regime_name": ["a", "b", "c"]},
        index=pd.Index([3010, 500_000_000, 959_738_010], name="id"),
    )
    model = _StubModel()
    simulate_with_dense_index(model=model, initial_conditions=ic, params={})
    assert model.seen_index == [0, 1, 2]


def test_simulate_with_dense_index_returns_ids_aligned_to_subject() -> None:
    """Returned ids align to subject_id: position i is the id of subject i."""
    ic = pd.DataFrame(
        {"regime_name": ["a", "b"]},
        index=pd.Index([3010, 959_738_010], name="id"),
    )
    _result, ids = simulate_with_dense_index(
        model=_StubModel(), initial_conditions=ic, params={}
    )
    np.testing.assert_array_equal(ids, [3010, 959_738_010])


def test_restore_subject_ids_maps_positional_subject_back_to_original() -> None:
    """`restore_subject_ids` recovers the caller's id from the positional subject_id."""
    original_ids = np.array([3010, 959_738_010])
    panel = pd.DataFrame({"subject_id": [0, 1, 0, 1], "period": [0, 0, 1, 1]})
    out = restore_subject_ids(panel, original_ids)
    np.testing.assert_array_equal(
        out["id"].to_numpy(), [3010, 959_738_010, 3010, 959_738_010]
    )
