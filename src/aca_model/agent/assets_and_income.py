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
    """Compute beginning-of-next-period assets for non-terminal targets.

    OOP health costs are deducted here (not from cash_on_hand) so that the
    consumption choice does not condition on the HCC shock realization.
    """
    return (
        cash_on_hand + transfers + pension_assets_adjustment - consumption - oop_costs
    )


def next_assets_terminal(
    cash_on_hand: FloatND,
    transfers: FloatND,
    consumption: ContinuousAction,
    oop_costs: FloatND,
) -> ContinuousState:
    """Compute beginning-of-next-period assets for the dead/terminal target.

    No `pension_assets_adjustment` term: with no future, there is no
    next-period pension wealth to impute against. Avoiding the dependency
    also keeps the `dead` per-target transition's DAG free of `next_aime`
    (which would otherwise need to come from a transition `dead` does not
    have, since `aime` is not a state in the terminal regime).
    """
    return cash_on_hand + transfers - consumption - oop_costs


def borrowing_constraint(
    consumption: ContinuousAction,
    cash_on_hand: FloatND,
    consumption_floor: float,
    equivalence_scale: FloatND,
) -> BoolND:
    """Consumption cannot exceed available resources after transfers.

    Post-transfer resources are `max(cash_on_hand, consumption_floor *
    equivalence_scale)`: the transfer system tops `cash_on_hand` to the
    floor when below, otherwise resources are unchanged. The algebraic
    identity is `cash_on_hand + transfers == max(cash_on_hand, floor)`,
    but writing it as `cash_on_hand + transfers` triggers float32
    catastrophic cancellation when `|cash_on_hand|` dwarfs
    `consumption_floor` — e.g. a subject at $-1{,}000{,}000$ in starting
    assets gives `(-1e6) + (c_floor + 1e6)` with ~0.1 of rounding error,
    which can wipe out the `c == c_floor` boundary and reject every
    feasible action. The `max` form has no cancellation.

    `pension_assets_adjustment` is excluded: it can be negative (e.g.
    when the imputation overstates next-period pension wealth at a
    cross-HIS transition), and including it here can leave no feasible
    action at low-asset / mid-AIME corners. The correction enters
    `next_assets` instead — a post-decision shift that does not gate
    the current consumption choice.
    """
    floor = consumption_floor * equivalence_scale
    return consumption <= jnp.maximum(cash_on_hand, floor)
