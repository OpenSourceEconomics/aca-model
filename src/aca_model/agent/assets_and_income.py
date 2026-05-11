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


def consumption_dollars_floor(
    consumption_equiv_floor: float,
    equivalence_scale: FloatND,
) -> FloatND:
    """Per-household $-floor on consumption."""
    return consumption_equiv_floor * equivalence_scale


def transfers(
    cash_on_hand: FloatND,
    consumption_dollars_floor: FloatND,
) -> FloatND:
    """Government transfers to enforce the consumption floor."""
    return jnp.maximum(0.0, consumption_dollars_floor - cash_on_hand)


def next_assets(
    cash_on_hand: FloatND,
    transfers: FloatND,
    pension_assets_adjustment: FloatND,
    consumption_dollars: ContinuousAction,
    oop_costs: FloatND,
) -> ContinuousState:
    """Compute beginning-of-next-period assets for non-terminal targets.

    OOP health costs are deducted here (not from cash_on_hand) so that the
    consumption choice does not condition on the HCC shock realization.
    """
    return (
        cash_on_hand
        + transfers
        + pension_assets_adjustment
        - consumption_dollars
        - oop_costs
    )


def next_assets_when_dead(
    cash_on_hand: FloatND,
    transfers: FloatND,
    consumption_dollars: ContinuousAction,
    oop_costs: FloatND,
) -> ContinuousState:
    """Compute beginning-of-next-period assets for the dead/terminal target.

    No `pension_assets_adjustment` term: with no future, there is no
    next-period pension wealth to impute against. Avoiding the dependency
    also keeps the `dead` per-target transition's DAG free of `next_aime`
    (which would otherwise need to come from a transition `dead` does not
    have, since `aime` is not a state in the terminal regime).
    """
    return cash_on_hand + transfers - consumption_dollars - oop_costs


def borrowing_constraint(
    consumption_dollars: ContinuousAction,
    cash_on_hand: FloatND,
    consumption_dollars_floor: FloatND,
) -> BoolND:
    """Consumption cannot exceed post-transfer resources.

    Post-transfer resources are `max(cash_on_hand, consumption_dollars_floor)`:
    the transfer system tops `cash_on_hand` to the floor when below,
    otherwise resources are unchanged. The algebraic identity is
    `cash_on_hand + transfers == max(cash_on_hand, floor)`; the `max`
    form is preferred because the additive form rounds to `floor + ε`
    (with `|ε| ~ ULP(|cash_on_hand|)`) at extreme cash, which flips
    the kink-boundary comparison at large negative values of `assets`.
    The `max` form returns `floor` exactly.
    """
    return consumption_dollars <= jnp.maximum(cash_on_hand, consumption_dollars_floor)
