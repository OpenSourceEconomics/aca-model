"""Regime transitions and builder for tied HIS regimes.

Tied regimes: agents with employer-tied health insurance.
Tied agents who stop working become nongroup.
Categorically (SSI-) Medicaid-eligible agents are also overridden to nongroup;
the ACA Medicaid expansion leaves employer coverage untouched.
"""

from collections.abc import Callable

import jax.numpy as jnp
from lcm import ByAge, Regime
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

from aca_model.agent.labor_market import LaborSupply
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
    """Create transition for canwork tied regimes.

    Tied agents who stop working become nongroup (lose employer coverage);
    before 65, those disabled next period hold disability Medicare there.
    Categorically (SSI-) Medicaid-eligible agents are also overridden to
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
        to_nongroup = (labor_supply == LaborSupply.do_not_work) | is_ssi_eligible

        def target(mc_next: bool) -> IntND:
            own_target = select_target_for_age(next_age, mc_next, own)
            ng_target = select_target_for_age(next_age, mc_next, ng)
            return jnp.where(to_nongroup, ng_target, own_target)

        return build_regime_probs_with_di_medicare(
            target_dimc=target(True),
            target_nomc=target(False),
            prob_dimc=prob_di_medicare_next(
                labor_supply, prob_disabled_next[period, health]
            ),
            survival=survival_probs[period, health],
        )

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


def build_law(name: str) -> ByAge:
    """Build the tied regime's per-target transition probabilities by source age."""
    spec = REGIME_SPECS[name]
    own, ng = make_targets(name)
    transition_func = _make_transition_canwork(own, ng)
    return build_regime_transition(
        spec=spec, transition_func=transition_func, target_groups=(own, ng)
    )


def build_regime(
    name: str,
    grids: Grids,
    *,
    dcegm_solver: DCEGM | None = None,
    nbegm_solver: NBEGM | None = None,
) -> Regime:
    """Build a tied regime (all tied regimes are canwork)."""
    spec = REGIME_SPECS[name]
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
        states=states,
        state_transitions=build_state_transitions(spec, solver=state_solver),
        actions=build_actions(spec, grids),
        functions=functions,
    )
