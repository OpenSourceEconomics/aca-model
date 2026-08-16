"""Asset transitions and budget constraint functions.

Ported from struct-ret/src/model/compute_within_period_quantities.py.
"""

import jax.numpy as jnp
import lcm
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarFloat,
)


def capital_income(
    assets: ContinuousState,
    rate_of_return: ScalarFloat,
) -> FloatND:
    """Compute capital income from assets."""
    return assets * rate_of_return


def premium_default(
    assets: ContinuousState,
    after_tax_income: FloatND,
    ssi_benefit: FloatND,
    hic_premium: FloatND,
    consumption_dollars_floor: FloatND,
) -> FloatND:
    """Compute the unpaid (defaulted) part of the insurance premium.

    A household pays its premium only up to what it can afford while staying
    at the consumption floor; it defaults on the rest as uncompensated care
    (non-payment of medical bills). Affordability is measured against the
    consumption floor, not the chosen consumption level, so the default is a
    clean function of resources and the premium:

    ```
    affordable_premium = max(0, resources - consumption_dollars_floor)
    premium_default    = max(0, hic_premium - affordable_premium)
    ```

    where `resources = assets + after_tax_income + ssi_benefit`.
    """
    resources = assets + after_tax_income + ssi_benefit
    affordable_premium = jnp.maximum(0.0, resources - consumption_dollars_floor)
    return jnp.maximum(0.0, hic_premium - affordable_premium)


def cash_on_hand(
    assets: ContinuousState,
    after_tax_income: FloatND,
    ssi_benefit: FloatND,
    hic_premium: FloatND,
    premium_default: FloatND,
) -> FloatND:
    """Compute cash on hand available for consumption and saving.

    Only the affordable part of the premium leaves cash-on-hand; the defaulted
    part (`premium_default`) is never paid, so the effective premium is
    `hic_premium - premium_default`.

    OOP health costs are NOT deducted here — they are deducted from
    next-period assets instead, matching the timing where HCC shocks are
    integrated over (agent does not condition consumption on OOP).
    """
    effective_premium = hic_premium - premium_default
    return assets + after_tax_income + ssi_benefit - effective_premium


def consumption_dollars_floor(
    consumption_equiv_floor: ScalarFloat,
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


@lcm.piecewise_affine(
    output="resources",
    variable="cash_on_hand",
    breakpoints=(
        lcm.affine_breakpoint(
            threshold="consumption_floor_schedule",
            kind="continuous_kink",
        ),
    ),
)
def resources(
    cash_on_hand: FloatND,
    consumption_floor_schedule: FloatND,
) -> FloatND:
    """Post-transfer resources out of which consumption is paid (DC-EGM `R`).

    Algebraically `cash_on_hand + transfers`; the `max` form is preferred
    because the additive form rounds to `floor + ε` at extreme cash (see
    `borrowing_constraint`). Non-decreasing in `assets` (flat where the
    floor binds), as the DC-EGM contract requires.

    The floor arrives as `consumption_floor_schedule`, a parameter rather
    than a DAG node: NBEGM reads a declared threshold from the solve's params
    before any DAG runs, so a value the DAG produces would be unreachable.
    With marital status on the regime axis the floor no longer varies within
    a regime, so the parameter carries one scalar per regime — equal by
    construction to that regime's `consumption_dollars_floor`. The kink where
    the floor stops binding is then a declared breakpoint, and NBEGM's
    partition splits each cell there instead of extrapolating one affine
    budget across it.
    """
    return jnp.maximum(cash_on_hand, consumption_floor_schedule)


def savings(
    resources: FloatND,
    consumption_dollars: ContinuousAction,
) -> FloatND:
    """End-of-period savings (the DC-EGM post-decision state).

    `savings >= 0` encodes the borrowing constraint
    `consumption_dollars <= max(cash_on_hand, floor)` via the savings
    grid's lower bound, so no explicit constraint is declared under DC-EGM.
    """
    return resources - consumption_dollars


def next_assets_from_savings(
    savings: FloatND,
    pension_assets_adjustment: FloatND,
    oop_costs: FloatND,
) -> ContinuousState:
    """Assets law for non-terminal targets in post-decision form.

    Algebraically identical to `next_assets`:
    `savings = cash_on_hand + transfers - consumption_dollars`.
    """
    return savings + pension_assets_adjustment - oop_costs


def next_assets_when_dead_from_savings(
    savings: FloatND,
    oop_costs: FloatND,
) -> ContinuousState:
    """Assets law for the dead/terminal target in post-decision form.

    No `pension_assets_adjustment` term, mirroring `next_assets_when_dead`.
    """
    return savings - oop_costs


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
