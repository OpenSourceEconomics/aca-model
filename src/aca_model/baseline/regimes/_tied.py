"""Regime transitions and builder for tied HIS regimes.

Tied regimes: agents with employer-tied health insurance.
Tied agents who stop working become nongroup.
Medicaid-eligible agents are also overridden to nongroup.
"""

from collections.abc import Callable

import jax.numpy as jnp
from lcm import Regime
from lcm.solvers import BQSEGM, DCEGM
from lcm.typing import Age, BoolND, DiscreteAction, FloatND, Period

from aca_model.agent.labor_market import LaborSupply
from aca_model.baseline import health_insurance
from aca_model.baseline.regimes._common import (
    REGIME_SPECS,
    Grids,
    RegimeSpec,
    build_actions,
    build_common_functions,
    build_granular_regime_transition,
    build_pension_functions,
    build_regime_probs,
    build_state_transitions,
    build_states,
    make_active_func,
    make_targets,
    select_ss_benefit,
    select_target_for_age,
)


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
        age: Age,
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


def _build_functions(spec: RegimeSpec) -> dict:
    """Build functions dict for a tied regime."""
    functions = build_common_functions(spec)

    functions["ss_benefit"] = select_ss_benefit(spec)

    # his and crossed_oamc_threshold are fixed params (constants per regime),
    # not DAG functions. pylcm resolves them from the params dict.

    functions["hic_premium"] = health_insurance.premium_insured
    functions.update(build_pension_functions(spec))

    return functions


def build_regime(
    name: str,
    grids: Grids,
    *,
    dcegm_solver: DCEGM | None = None,
    bqsegm_solver: BQSEGM | None = None,
) -> Regime:
    """Build a tied regime (all tied regimes are canwork)."""
    spec = REGIME_SPECS[name]
    gets_mc = spec["mc"] != "nomc"
    own, ng = make_targets(name)

    transition_func = _make_transition_canwork(gets_mc, own, ng)

    states = build_states(spec, grids)
    egm_solver = dcegm_solver if dcegm_solver is not None else bqsegm_solver
    solver_kwargs: dict = {} if egm_solver is None else {"solver": egm_solver}
    state_solver = (
        "brute_force"
        if egm_solver is None
        else ("bqsegm" if bqsegm_solver is not None else "dcegm")
    )
    return Regime(
        transition=build_granular_regime_transition(
            transition_func=transition_func, target_ids=(*own.values(), *ng.values())
        ),
        active=make_active_func(spec),
        states=states,
        state_transitions=build_state_transitions(spec, solver=state_solver),
        actions=build_actions(spec, grids),
        functions=_build_functions(spec),
        **solver_kwargs,
    )
