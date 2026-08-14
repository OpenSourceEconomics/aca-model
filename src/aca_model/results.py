"""Post-processing helpers for aca-model simulation result panels."""

import pandas as pd

from aca_model.baseline.regimes._common import REGIME_SPECS

# Health-insurance source of each regime, `dead` included. Read from
# `REGIME_SPECS` rather than parsed out of the name: the name's leading token
# is marital status, and every further axis sits at a position that another
# axis gaining a value would shift.
_HIS_BY_REGIME_NAME: dict[str, str] = {
    name: spec["his"] for name, spec in REGIME_SPECS.items()
} | {"dead": "dead"}

# Marital status of each regime. The terminal `dead` regime has none; it maps
# to `False` so the column stays a plain boolean, and no moment selects on it.
_IS_MARRIED_BY_REGIME_NAME: dict[str, bool] = {
    name: spec["marital"] == "married" for name, spec in REGIME_SPECS.items()
} | {"dead": False}


def add_regime_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Return `panel` with `his` and `is_married` derived from `regime_name`.

    Neither is a per-subject model output — both are axes of the regime a
    subject occupies. The terminal `dead` regime has no health-insurance
    component and maps to `"dead"`, which no HIS moment selects.

    Args:
        panel: Simulation result DataFrame carrying a `regime_name` column.

    Returns:
        A copy of `panel` with an added string `his` column and a boolean
        `is_married` column.

    Raises:
        KeyError: If `regime_name` carries a name that is not a model regime.

    """
    names = panel["regime_name"].astype("str")
    unknown = set(names.unique()) - _HIS_BY_REGIME_NAME.keys()
    if unknown:
        msg = f"regime_name carries unknown regimes: {sorted(unknown)}"
        raise KeyError(msg)
    return panel.assign(
        his=names.map(_HIS_BY_REGIME_NAME).astype("str"),
        is_married=names.map(_IS_MARRIED_BY_REGIME_NAME).astype("bool"),
    )
