"""Post-processing helpers for aca-model simulation result panels."""

import pandas as pd


def add_his_from_regime_name(panel: pd.DataFrame) -> pd.DataFrame:
    """Return `panel` with a `his` column derived from `regime_name`.

    The current health-insurance state is not a per-subject model output — it is
    the regime a subject occupies. Regime names follow the
    `{his}_{mc}_{ss}_{canwork}` convention (see
    `aca_model.baseline.regimes.REGIME_SPECS`), so `his` is the leading
    underscore-delimited token of `regime_name`. The terminal `dead` regime has
    no HIS component and maps to `"dead"`, which no HIS moment selects.

    Args:
        panel: Simulation result DataFrame carrying a `regime_name` column.

    Returns:
        A copy of `panel` with an added string `his` column.

    """
    his = panel["regime_name"].astype("str").str.split("_", n=1).str[0].astype("str")
    return panel.assign(his=his)
