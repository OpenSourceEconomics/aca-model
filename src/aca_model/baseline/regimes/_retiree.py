"""Regime transitions and builder for retiree HIS regimes.

Retiree regimes: agents with employer-sponsored retiree health insurance.
Categorically (SSI-) Medicaid-eligible agents are overridden to nongroup; the
ACA Medicaid expansion leaves employer coverage untouched.
"""

from collections.abc import Callable

import jax.numpy as jnp
from lcm import Regime
from lcm.solvers import DCEGM, NBEGM
from lcm.typing import (
    Age,
    BoolND,
    DiscreteAction,
    DiscreteState,
    FloatND,
    IntND,
    Period,
)

from aca_model.baseline import health_insurance
from aca_model.baseline.regimes._common import (
    REGIME_SPECS,
    Grids,
    RegimeSpec,
    build_actions,
    build_alive_regime,
    build_common_functions,
    build_nbegm_functions,
    build_pension_functions,
    build_regime_probs,
    build_regime_probs_with_di_medicare,
    build_regime_transition,
    build_state_transitions,
    build_states,
    make_targets,
    next_model_age,
    prob_di_medicare_next,
    select_ss_benefit,
    select_target_for_age,
)


def _make_transition_canwork(
    own: dict[str, int],
    ng: dict[str, int],
) -> Callable[..., FloatND]:
    """Create transition for canwork retiree regimes.

    Before 65, a household that does not work and is disabled next period
    holds disability Medicare. Categorically (SSI-) Medicaid-eligible
    agents are overridden to
    nongroup targets.
    """

    def transition(
        age: Age,
        period: Period,
        health: DiscreteState,
        labor_supply: DiscreteAction,
        is_ssi_eligible: BoolND,
        survival_probs: FloatND,
        prob_disabled_next: FloatND,
    ) -> FloatND:
        del age  # Keep the legacy signature; period owns the clock lookup.
        next_age = next_model_age(period)

        def target(mc_next: bool) -> IntND:
            own_target = select_target_for_age(next_age, mc_next, own)
            ng_target = select_target_for_age(next_age, mc_next, ng)
            return jnp.where(is_ssi_eligible, ng_target, own_target)

        return build_regime_probs_with_di_medicare(
            target_dimc=target(True),
            target_nomc=target(False),
            prob_dimc=prob_di_medicare_next(
                labor_supply, prob_disabled_next[period, health]
            ),
            survival=survival_probs[period, health],
        )

    return transition


def _make_transition_forcedout(
    gets_medicare: bool,
    own: dict[str, int],
    ng: dict[str, int],
) -> Callable[..., FloatND]:
    """Create transition for forcedout retiree regimes.

    No labor supply action. Categorically (SSI-) Medicaid-eligible agents are
    overridden to nongroup.
    """

    def transition(
        age: Age,
        period: Period,
        health: DiscreteState,
        is_ssi_eligible: BoolND,
        survival_probs: FloatND,
    ) -> FloatND:
        del age  # Keep the legacy signature; period owns the clock lookup.
        sp = survival_probs[period, health]
        next_age = next_model_age(period)
        target = select_target_for_age(next_age, gets_medicare, own)
        ng_ssi = select_target_for_age(next_age, gets_medicare, ng)
        target = jnp.where(is_ssi_eligible, ng_ssi, target)
        return build_regime_probs(target, sp)

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
    dcegm_solver: DCEGM | None = None,
    nbegm_solver: NBEGM | None = None,
) -> Regime:
    """Build a retiree regime."""
    spec = REGIME_SPECS[name]
    gets_mc = spec["mc"] != "nomc"
    own, ng = make_targets(name)

    if spec["canwork"] == "canwork":
        transition_func = _make_transition_canwork(own, ng)
    else:
        transition_func = _make_transition_forcedout(gets_mc, own, ng)

    states = build_states(spec, grids)

    egm_solver = dcegm_solver if dcegm_solver is not None else nbegm_solver
    state_solver = (
        "brute_force"
        if egm_solver is None
        else ("nbegm" if nbegm_solver is not None else "dcegm")
    )
    functions = _build_functions(spec)
    if nbegm_solver is not None:
        # NBEGM's solver contract is stated per regime: it reads the budget in
        # savings form off `resources` and the post-decision node off
        # `savings`, neither of which the brute-force build needs.
        functions = {**functions, **build_nbegm_functions()}
    return build_alive_regime(
        egm_solver=egm_solver,
        regime_transitions=build_regime_transition(
            spec=spec, transition_func=transition_func, target_groups=(own, ng)
        ),
        states=states,
        state_transitions=build_state_transitions(spec, solver=state_solver),
        actions=build_actions(spec, grids),
        functions=functions,
    )
