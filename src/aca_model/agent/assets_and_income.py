"""Asset transitions and budget constraint functions.

Ported from struct-ret/src/model/compute_within_period_quantities.py.
"""

import jax.numpy as jnp
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
)


def capital_income(
    assets: ContinuousState,
    rate_of_return: float,
) -> FloatND:
    """Compute capital income from assets."""
    return assets * rate_of_return


def cash_on_hand(
    assets: ContinuousState,
    after_tax_income: FloatND,
    ssi_benefit: FloatND,
    hic_premium: FloatND,
) -> FloatND:
    """Compute cash on hand available for consumption and saving.

    OOP health costs are NOT deducted here — they are deducted from
    next-period assets instead, matching the timing where HCC shocks are
    integrated over (agent does not condition consumption on OOP).
    """
    return assets + after_tax_income + ssi_benefit - hic_premium


def transfers(
    cash_on_hand: FloatND,
    consumption_floor: float,
    equivalence_scale: FloatND,
) -> FloatND:
    """Government transfers to enforce consumption floor.

    tr = max{0, C_min * equivalence_scale - cash_on_hand}
    """
    floor = consumption_floor * equivalence_scale
    return jnp.maximum(0.0, floor - cash_on_hand)


def next_assets(
    cash_on_hand: FloatND,
    transfers: FloatND,
    pension_assets_adjustment: FloatND,
    consumption: ContinuousAction,
    oop_costs: FloatND,
) -> ContinuousState:
    """Compute beginning-of-next-period assets.

    OOP health costs are deducted here (not from cash_on_hand) so that the
    consumption choice does not condition on the HCC shock realization.
    """
    return (
        cash_on_hand + transfers + pension_assets_adjustment - consumption - oop_costs
    )


def borrowing_constraint(
    consumption: ContinuousAction,
    cash_on_hand: FloatND,
    transfers: FloatND,
    pension_assets_adjustment: FloatND,
) -> BoolND:
    """Consumption cannot exceed available resources (no borrowing)."""
    return consumption <= cash_on_hand + transfers + pension_assets_adjustment
