"""Social Security lookup-table helpers used by tests.

Inlined from the aca-data pipeline so the aca-model test suite has no
runtime dependency on the data-prep package.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def compute_pia_table(
    aime_kink_0: float,
    aime_kink_1: float,
    pia_conversion_rate_0: float,
    pia_conversion_rate_1: float,
    pia_conversion_rate_2: float,
    max_aime: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pre-compute PIA lookup table on the minimal 4-point grid.

    PIA is piecewise linear with 2 kinks, so 4 grid points suffice for
    exact interpolation via `jnp.interp` at any AIME value.
    """
    pia_kink_0 = pia_conversion_rate_0 * aime_kink_0
    pia_kink_1 = pia_kink_0 + pia_conversion_rate_1 * (aime_kink_1 - aime_kink_0)

    aime_grid = np.array([0.0, aime_kink_0, aime_kink_1, max_aime])
    pia_table = np.array(
        [
            0.0,
            pia_kink_0,
            pia_kink_1,
            pia_kink_1 + pia_conversion_rate_2 * (max_aime - aime_kink_1),
        ]
    )
    return aime_grid, pia_table


def compute_di_dropout_scale(
    ratio_lowest_earnings: pd.Series,
    aime_accrual_factor: float,
    start_age: int,
    n_periods: int,
) -> np.ndarray:
    """Pre-compute DI dropout-year scale factors for all periods.

    For each period, compute the multiplicative factor by which AIME is
    scaled up when disregarding years with lowest earnings
    (CRS Report R43370).
    """
    result = np.ones(n_periods)

    for period in range(n_periods):
        age = start_age + period
        elapsed = min(age - 22, 35)

        if elapsed <= 0:
            continue

        years_to_drop = int(min(elapsed // 5, 5))

        if years_to_drop == 0:
            continue

        product = 1.0
        for adjust_age in range(age - 1 - years_to_drop, age - 1):
            product *= 1.0 - ratio_lowest_earnings[adjust_age] * aime_accrual_factor

        result[period] = product * elapsed / (elapsed - years_to_drop)

    return result
