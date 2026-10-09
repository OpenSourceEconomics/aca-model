"""Declaring the asset law as resources minus the medical bill is exact.

With `subtract_hcc_bill`, the asset law toward every target is
`lcm.SubtractedBill`: end-of-period resources before the out-of-pocket bill,
minus the bill. pylcm then averages each target's value over the transitory
medical-cost shock once per period, on the merged points `{a_j + bill_k}`,
rather than interpolating it at every shock node of every source point. With
linear interpolation in assets that average is exact, so the values agree with
the plain asset law to rounding.
"""

from dataclasses import replace

import lcm
import numpy as np
import pytest
from helpers.model import make_baseline_model

from aca_model.benchmark import get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG

_FOLDED = replace(BENCHMARK_GRID_CONFIG, fold_hcc_transitory=True)
_SUBTRACTED = replace(_FOLDED, subtract_hcc_bill=True)
_REGIME = "retiree_nomc_inelig_canwork"


@pytest.fixture(scope="module")
def subtracted_model():
    return make_baseline_model(grid_config=_SUBTRACTED)


def test_every_target_takes_the_subtracted_bill(subtracted_model):
    laws = [
        law
        for name, regime in subtracted_model.user_regimes.items()
        if name != "dead"
        for law in regime.state_transitions["assets"].values()
    ]

    assert laws
    assert all(isinstance(law, lcm.SubtractedBill) for law in laws)


@pytest.mark.long_running
def test_subtracted_bill_value_equals_the_plain_law_value(subtracted_model):
    """The two routes sum the same terms in a different order."""
    folded_model = make_baseline_model(grid_config=_FOLDED)
    _, _, params = get_benchmark_params(model=folded_model)
    folded = np.asarray(
        folded_model.solve(params=params, log_level="off").values[0][_REGIME]
    )
    subtracted = np.asarray(
        subtracted_model.solve(params=params, log_level="off").values[0][_REGIME]
    )

    np.testing.assert_allclose(subtracted, folded, rtol=1e-9, atol=0)
