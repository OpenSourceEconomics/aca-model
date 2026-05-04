"""Regime transitions and builder for nongroup HIS regimes.

Nongroup regimes: agents purchasing individual-market health insurance.
Already nongroup, so no SSI/Medicaid override needed for HIS transitions.
"""

from collections.abc import Callable

from lcm import MarkovTransition, Regime
from lcm.typing import DiscreteAction, FloatND, Period

from aca_model.agent import assets_and_income, preferences
from aca_model.agent.labor_market import LaborSupply
from aca_model.baseline import health_insurance
from aca_model.baseline.regimes._common import (
    REGIME_SPECS,
    Grids,
    build_actions,
    build_common_functions,
    build_regime_probs,
    build_state_transitions,
    build_states,
    make_active_func,
    make_targets,
    select_ss_benefit,
    select_target_for_age,
    select_utility,
)
from aca_model.environment import pensions


def _make_transition_canwork(
    gets_medicare: bool,
    own: dict[str, int],
) -> Callable[..., FloatND]:
    """Create transition for canwork nongroup regimes.

    Already nongroup — no SSI override needed. Gets Medicare if stops working
    (when gets_medicare is True).
    """

    def transition(
        age: int,
        period: Period,
        labor_supply: DiscreteAction,
        survival_probs: FloatND,
    ) -> FloatND:
        sp = survival_probs[period]
        mc_next = gets_medicare & (labor_supply == LaborSupply.do_not_work)
        target = select_target_for_age(age + 1, mc_next, own)
        return build_regime_probs(target, sp)

    return transition


def _make_transition_forcedout(
    gets_medicare: bool,
    own: dict[str, int],
) -> Callable[..., FloatND]:
    """Create transition for forcedout nongroup regimes.

    Simplest transition: no labor supply, no SSI override. Gets Medicare
    based on regime constant.
    """

    def transition(
        age: int,
        period: Period,
        survival_probs: FloatND,
    ) -> FloatND:
        target = select_target_for_age(age + 1, gets_medicare, own)
        return build_regime_probs(target, survival_probs[period])

    return transition


def _build_functions(spec: dict[str, str]) -> dict:
    """Build functions dict for a nongroup regime."""
    can_work = spec["canwork"] == "canwork"
    functions = build_common_functions(spec)

    functions["utility"] = select_utility(spec)
    functions["ss_benefit"] = select_ss_benefit(spec)

    # his and gets_medicare are fixed params (constants per regime),
    # not DAG functions. pylcm resolves them from the params dict.

    has_buy_private = spec["his"] == "nongroup" and spec["mc"] == "nomc"
    if has_buy_private:
        functions["hic_premium"] = health_insurance.premium
    elif can_work:
        functions["hic_premium"] = health_insurance.premium_insured
    else:
        functions["hic_premium"] = health_insurance.premium_retired
    functions["pension_benefit"] = pensions.benefit
    functions["pension_wealth"] = pensions.wealth
    if can_work and spec["ss"] != "forced":
        functions["pension_accrual"] = pensions.accrual
        functions["pension_wealth_next_before_adjustment"] = (
            pensions.wealth_next_before_adjustment
        )
        functions["target_his"] = health_insurance.target_his
        functions["imputed_pension_wealth_next_period"] = (
            pensions.imputed_pension_wealth_next_period
        )
        functions["pension_assets_adjustment"] = pensions.assets_adjustment
        functions["total_to_pia"] = pensions.total_to_pia

    return functions


def build_regime(name: str, grids: Grids) -> Regime:
    """Build a nongroup regime."""
    spec = REGIME_SPECS[name]
    gets_mc = spec["mc"] != "nomc"
    own, _ng = make_targets(name)

    if spec["canwork"] == "canwork":
        transition_func = _make_transition_canwork(gets_mc, own)
    else:
        transition_func = _make_transition_forcedout(gets_mc, own)

    states = build_states(spec, grids)
    constraints: dict = {
        "borrowing_constraint": assets_and_income.borrowing_constraint,
    }
    if spec["canwork"] == "canwork":
        constraints["positive_leisure"] = preferences.positive_leisure

    return Regime(
        transition=MarkovTransition(transition_func),
        active=make_active_func(spec),
        states=states,
        state_transitions=build_state_transitions(spec),
        actions=build_actions(spec, grids),
        functions=_build_functions(spec),
        constraints=constraints,
    )
