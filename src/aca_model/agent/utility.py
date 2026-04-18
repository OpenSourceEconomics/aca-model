"""Utility function variants for different regime types.

- retired: forcedout regimes (no work, computes leisure_retired internally)
- dead: terminal bequest

Canwork regimes use `preferences.utility` directly, with `leisure` computed
as a separate DAG function (`preferences.leisure` / `preferences.leisure_tied`).
"""

from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    IntND,
)

from aca_model.agent import preferences


def retired(
    consumption: ContinuousAction,
    good_health: IntND,
    equivalence_scale: FloatND,
    pref_type: DiscreteState,
    consumption_weight: FloatND,
    coefficient_rra: FloatND,
    utility_scale_factor: FloatND,
    time_endowment: float,
    leisure_cost_of_bad_health: float,
) -> FloatND:
    """Utility for forcedout regimes (no work)."""
    lei = preferences.leisure_retired(
        good_health=good_health,
        time_endowment=time_endowment,
        leisure_cost_of_bad_health=leisure_cost_of_bad_health,
    )
    return preferences.utility(
        consumption=consumption,
        leisure=lei,
        pref_type=pref_type,
        consumption_weight=consumption_weight,
        coefficient_rra=coefficient_rra,
        equivalence_scale=equivalence_scale,
        utility_scale_factor=utility_scale_factor,
    )


def dead(
    assets: ContinuousState,
    pref_type: DiscreteState,
    bequest_shifter: float,
    scaled_bequest_weight: float,
    consumption_weight: FloatND,
    coefficient_rra: FloatND,
    utility_scale_factor: FloatND,
) -> FloatND:
    """Terminal bequest utility for dead regime."""
    return preferences.bequest(
        assets=assets,
        pref_type=pref_type,
        bequest_shifter=bequest_shifter,
        scaled_bequest_weight=scaled_bequest_weight,
        consumption_weight=consumption_weight,
        coefficient_rra=coefficient_rra,
        utility_scale_factor=utility_scale_factor,
    )
