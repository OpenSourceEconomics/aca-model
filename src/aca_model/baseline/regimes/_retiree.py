"""Regime transitions and builder for retiree HIS regimes.

Retiree regimes: agents with employer-sponsored retiree health insurance.
Medicaid-eligible agents are overridden to nongroup.
"""

from collections.abc import Callable

import jax.numpy as jnp
from lcm import Regime
from lcm.solvers import NBEGM
from lcm.typing import Age, BoolND, DiscreteAction, FloatND, Period

from aca_model.agent.labor_market import LaborSupply
from aca_model.baseline import health_insurance
from aca_model.baseline.regimes._common import (
    MARITAL_STATUSES,
    REGIME_SPECS,
    Grids,
    RegimeSpec,
    build_actions,
    build_common_functions,
    build_granular_regime_transition,
    build_nbegm_functions,
    build_pension_functions,
    build_regime_probs,
    build_state_transitions,
    build_states,
    flatten_targets,
    make_active_func,
    make_targets,
    select_ss_benefit,
    select_target_for_age,
)


def _make_transition_canwork(
    gets_medicare: bool,
    own: dict[str, dict[str, int]],
    ng: dict[str, dict[str, int]],
) -> Callable[..., FloatND]:
    """Create transition for canwork retiree regimes.

    Retirees who stop working get Medicare (if gets_medicare).
    Medicaid-eligible agents are overridden to nongroup targets.
    """

    def transition(
        age: Age,
        period: Period,
        labor_supply: DiscreteAction,
        is_medicaid_eligible: BoolND,
        survival_probs: FloatND,
        marital_probs: FloatND,
    ) -> FloatND:
        sp = survival_probs[period]
        next_age = age + 1
        mc_next = gets_medicare & (labor_supply == LaborSupply.do_not_work)
        targets = {}
        for marital in MARITAL_STATUSES:
            target = select_target_for_age(next_age, mc_next, own[marital])
            # Medicaid eligibility overrides to nongroup
            ng_ssi = select_target_for_age(next_age, mc_next, ng[marital])
            targets[marital] = jnp.where(is_medicaid_eligible, ng_ssi, target)
        return build_regime_probs(
            target_single=targets["single"],
            target_married=targets["married"],
            survival=sp,
            marital_probs=marital_probs,
        )

    return transition


def _make_transition_forcedout(
    gets_medicare: bool,
    own: dict[str, dict[str, int]],
    ng: dict[str, dict[str, int]],
) -> Callable[..., FloatND]:
    """Create transition for forcedout retiree regimes.

    No labor supply action. Medicaid-eligible agents are overridden to nongroup.
    """

    def transition(
        age: Age,
        period: Period,
        is_medicaid_eligible: BoolND,
        survival_probs: FloatND,
        marital_probs: FloatND,
    ) -> FloatND:
        sp = survival_probs[period]
        next_age = age + 1
        targets = {}
        for marital in MARITAL_STATUSES:
            target = select_target_for_age(next_age, gets_medicare, own[marital])
            ng_ssi = select_target_for_age(next_age, gets_medicare, ng[marital])
            targets[marital] = jnp.where(is_medicaid_eligible, ng_ssi, target)
        return build_regime_probs(
            target_single=targets["single"],
            target_married=targets["married"],
            survival=sp,
            marital_probs=marital_probs,
        )

    return transition


def _build_functions(spec: RegimeSpec) -> dict:
    """Build functions dict for a retiree regime."""
    can_work = spec["canwork"] == "canwork"
    functions = build_common_functions(spec)

    functions["ss_benefit"] = select_ss_benefit(spec)

    # his and crossed_oamc_threshold are fixed params (constants per regime),
    # not DAG functions. pylcm resolves them from the params dict.

    functions["hic_premium"] = (
        health_insurance.premium_insured
        if can_work
        else health_insurance.premium_retired
    )
    functions.update(build_pension_functions(spec))

    return functions


def build_regime(
    name: str,
    grids: Grids,
    *,
    nbegm_solver: NBEGM | None = None,
) -> Regime:
    """Build a retiree regime."""
    spec = REGIME_SPECS[name]
    gets_mc = spec["mc"] != "nomc"
    own, ng = make_targets(name)

    if spec["canwork"] == "canwork":
        transition_func = _make_transition_canwork(gets_mc, own, ng)
    else:
        transition_func = _make_transition_forcedout(gets_mc, own, ng)

    states = build_states(spec, grids)

    solver_kwargs: dict = {} if nbegm_solver is None else {"solver": nbegm_solver}
    state_solver = "brute_force" if nbegm_solver is None else "nbegm"
    functions = _build_functions(spec)
    if nbegm_solver is not None:
        # NBEGM's solver contract is stated per regime: it reads the budget in
        # savings form off `resources` and the post-decision node off
        # `savings`, neither of which the brute-force build needs.
        functions = {**functions, **build_nbegm_functions()}
    return Regime(
        transition=build_granular_regime_transition(
            transition_func=transition_func, target_ids=flatten_targets(own, ng)
        ),
        active=make_active_func(spec),
        states=states,
        state_transitions=build_state_transitions(spec, solver=state_solver),
        actions=build_actions(spec, grids),
        functions=functions,
        **solver_kwargs,
    )
