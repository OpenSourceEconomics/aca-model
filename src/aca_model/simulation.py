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
    admissible = select_admissible_starts(
        model=model, initial_conditions=initial_conditions
    )
    original_ids = np.asarray(admissible.index)
    dense = admissible.reset_index(drop=True)
    result = model.simulate(initial_conditions=dense, **simulate_kwargs)
    return result, original_ids


def select_admissible_starts(
    *, model: Any, initial_conditions: pd.DataFrame
) -> pd.DataFrame:
    """Keep the rows that start at or before the model's last admissible start age.

    Rows older than every age in `model.graph.initial_nodes` are outside the sample
    the model is meant to simulate and are dropped. Every kept row must start at
    an admissible `(age, regime_name)` pair; any other kept row raises.

    Args:
        model: A pylcm `Model` (anything exposing `graph.initial_nodes`, the
            admissible `(age, regime_name)` pairs).
        initial_conditions: Seed DataFrame with `age` and `regime_name` columns.

    Returns:
        The rows of `initial_conditions` inside the model's entry ages.

    Raises:
        ValueError: If a kept row starts at a pair outside `model.graph.initial_nodes`.

    """
    admitted = {(float(age), regime) for age, regime in model.graph.initial_nodes}
    last_entry_age = max(age for age, _ in admitted)
    kept = initial_conditions.loc[initial_conditions["age"] <= last_entry_age]
    pairs = set(zip(kept["age"].astype(float), kept["regime_name"], strict=True))
    _fail_if_starts_not_admitted(refused=pairs - admitted)
    return kept


def _fail_if_starts_not_admitted(*, refused: set[tuple[float, str]]) -> None:
    if refused:
        msg = f"Initial conditions start at inadmissible pairs: {sorted(refused)}"
        raise ValueError(msg)


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
