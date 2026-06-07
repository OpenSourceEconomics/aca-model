"""Helpers for driving pylcm's simulate across a non-positional id index.

pylcm assigns each simulated subject a positional `subject_id` (`0..n-1`) and,
when given a DataFrame of initial conditions, scatters each regime group's
state values into the result arrays *by the group's index labels*. A seed
indexed by anything other than a dense `[0, n)` range — e.g. HRS person ids —
therefore indexes out of bounds, and the original id never reaches the output.

These helpers wrap that boundary: reset the seed to a dense range just before
simulating, and map the positional `subject_id` back to the caller's ids just
after.
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def simulate_with_dense_index(
    *,
    model: Any,
    initial_conditions: pd.DataFrame,
    **simulate_kwargs: Any,
) -> tuple[Any, NDArray[Any]]:
    """Simulate `model` with a dense subject index, preserving the caller's ids.

    Args:
        model: A pylcm `Model`.
        initial_conditions: Seed DataFrame indexed by the caller's subject ids
            (any index — typically the HRS person id).
        **simulate_kwargs: Forwarded to `model.simulate` (e.g. `params`,
            `period_to_regime_to_V_arr`, `log_level`).

    Returns:
        Tuple of the `SimulationResult` and the original id array aligned to
        `subject_id` — position `i` holds the id of the subject pylcm labels
        `subject_id == i`. Pass it to `restore_subject_ids` to recover the ids
        on the result panel.

    """
    original_ids = np.asarray(initial_conditions.index)
    dense = initial_conditions.reset_index(drop=True)
    result = model.simulate(initial_conditions=dense, **simulate_kwargs)
    return result, original_ids


def restore_subject_ids(
    panel: pd.DataFrame,
    original_ids: NDArray[Any],
    *,
    id_col: str = "id",
) -> pd.DataFrame:
    """Map a result panel's positional `subject_id` back to the caller's ids.

    Args:
        panel: Simulation result DataFrame carrying a positional `subject_id`
            column (`0..n-1`).
        original_ids: Original ids aligned to `subject_id`, as returned by
            `simulate_with_dense_index`.
        id_col: Name of the column to add with the restored ids.

    Returns:
        A copy of `panel` with the restored id column added.

    """
    return panel.assign(**{id_col: original_ids[panel["subject_id"].to_numpy()]})
