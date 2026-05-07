"""Regime transitions and builder for tied HIS regimes.

Tied regimes: agents with employer-tied health insurance.
Tied agents who stop working become nongroup.
Medicaid-eligible agents are also overridden to nongroup.
"""

from collections.abc import Callable

import jax.numpy as jnp
from lcm import MarkovTransition, Regime
from lcm.typing import BoolND, DiscreteAction, FloatND, Period

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
    ng: dict[str, int],
) -> Callable[..., FloatND]:
    """Create transition for canwork tied regimes.

    Tied agents who stop working become nongroup (lose employer coverage).
    Medicaid-eligible agents are also overridden to nongroup targets.
    """

    def transition(
        age: int,
        period: Period,
        labor_supply: DiscreteAction,
        is_medicaid_eligible: BoolND,
        survival_probs: FloatND,
    ) -> FloatND:
        sp = survival_probs[period]
        next_age = age + 1
        mc_next = gets_medicare & (labor_supply == LaborSupply.do_not_work)
        target = select_target_for_age(next_age, mc_next, own)
        # Tied agents who stop working become nongroup
        stopped = labor_supply == LaborSupply.do_not_work
        ng_target = select_target_for_age(next_age, mc_next, ng)
        target = jnp.where(stopped, ng_target, target)
        # Medicaid eligibility overrides to nongroup
        ng_ssi = select_target_for_age(next_age, mc_next, ng)
        target = jnp.where(is_medicaid_eligible, ng_ssi, target)
        return build_regime_probs(target, sp)

    return transition


def _build_functions(spec: dict[str, str]) -> dict:
    """Build functions dict for a tied regime."""
    functions = build_common_functions(spec)

    functions["utility"] = select_utility(spec)
    functions["ss_benefit"] = select_ss_benefit(spec)

    # his and gets_medicare are fixed params (constants per regime),
    # not DAG functions. pylcm resolves them from the params dict.

    functions["hic_premium"] = health_insurance.premium_insured
    functions["pension_benefit"] = pensions.benefit
    functions["pension_wealth"] = pensions.wealth
    if spec["canwork"] == "canwork" and spec["ss"] != "forced":
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
    """Build a tied regime (all tied regimes are canwork)."""
    spec = REGIME_SPECS[name]
    gets_mc = spec["mc"] != "nomc"
    own, ng = make_targets(name)

    transition_func = _make_transition_canwork(gets_mc, own, ng)

    states = build_states(spec, grids)
    return Regime(
        transition=MarkovTransition(transition_func),
        active=make_active_func(spec),
        states=states,
        state_transitions=build_state_transitions(spec),
        actions=build_actions(spec, grids),
        functions=_build_functions(spec),
        constraints={
            "borrowing_constraint": assets_and_income.borrowing_constraint,
            "positive_leisure": preferences.positive_leisure,
        },
    )
