import pandas as pd
from pandas.testing import assert_series_equal

from aca_model.results import add_his_from_regime_name


def test_add_his_from_regime_name_extracts_leading_token():
    """`his` is the leading token of `regime_name` (`{his}_{mc}_{ss}_{canwork}`).

    The current health-insurance state is encoded as the first underscore-delimited
    component of the regime name; the terminal `dead` regime maps to `"dead"`.
    """
    panel = pd.DataFrame(
        {
            "regime_name": pd.Categorical(
                [
                    "retiree_nomc_inelig_canwork",
                    "tied_dimc_choose_canwork",
                    "nongroup_oamc_forced_forcedout",
                    "dead",
                ]
            ),
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = add_his_from_regime_name(panel)

    assert_series_equal(
        result["his"],
        pd.Series(["retiree", "tied", "nongroup", "dead"], name="his"),
    )
