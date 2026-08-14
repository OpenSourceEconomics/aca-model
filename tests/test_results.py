import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from aca_model.results import add_regime_columns

_PANEL = pd.DataFrame(
    {
        "regime_name": pd.Categorical(
            [
                "single_retiree_nomc_inelig_canwork",
                "married_tied_nomc_choose_canwork",
                "single_nongroup_oamc_forced_forcedout",
                "dead",
            ]
        ),
        "value": [1.0, 2.0, 3.0, 4.0],
    }
)


def test_add_regime_columns_reads_the_health_insurance_source():
    """`his` is the regime's health-insurance axis; `dead` maps to `"dead"`."""
    result = add_regime_columns(_PANEL)

    assert_series_equal(
        result["his"],
        pd.Series(["retiree", "tied", "nongroup", "dead"], name="his"),
    )


def test_add_regime_columns_reads_the_marital_axis():
    """`is_married` is the regime's marital axis; `dead` carries none."""
    result = add_regime_columns(_PANEL)

    assert_series_equal(
        result["is_married"],
        pd.Series([False, True, False, False], name="is_married"),
    )


def test_add_regime_columns_rejects_a_name_that_is_not_a_regime():
    """A panel carrying an unknown regime name is an error, not a silent NA."""
    panel = _PANEL.assign(
        regime_name=pd.Categorical(
            ["retiree_nomc_inelig_canwork", "dead", "dead", "dead"]
        )
    )

    with pytest.raises(KeyError, match="retiree_nomc_inelig_canwork"):
        add_regime_columns(panel)
