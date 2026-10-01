"""Regime transitions and builder for nongroup HIS regimes.

Nongroup regimes: agents purchasing individual-market health insurance.
Already nongroup, so no SSI/Medicaid override needed for HIS transitions.
"""

from collections.abc import Callable

from lcm import Regime
from lcm.solvers import DCEGM, NBEGM
from lcm.typing import Age, DiscreteAction, DiscreteState, FloatND, Period

from aca_model.baseline import health_insurance
from aca_model.baseline.regimes._common import (
    REGIME_SPECS,
    Grids,
    RegimeSpec,
    build_actions,
    build_alive_regime,
    build_common_functions,
    build_granular_regime_transition,
    build_nbegm_functions,
    build_pension_functions,
    build_regime_probs,
    build_regime_probs_with_di_medicare,
    build_state_transitions,
    build_states,
    make_active_func,
    make_targets,
    prob_di_medicare_next,
    select_ss_benefit,
    select_target_for_age,
)


def _make_transition_canwork(
    own: dict[str, int],
) -> Callable[..., FloatND]:
    """Create transition for canwork nongroup regimes.

    Already nongroup — no SSI override needed. Before 65, a household that
    does not work and is disabled next period holds disability Medicare.
    """

    def transition(
        age: Age,
        period: Period,
        health: DiscreteState,
        labor_supply: DiscreteAction,
        survival_probs: FloatND,
        prob_disabled_next: FloatND,
    ) -> FloatND:
        return build_regime_probs_with_di_medicare(
            target_dimc=select_target_for_age(age + 1, True, own),
            target_nomc=select_target_for_age(age + 1, False, own),
            prob_dimc=prob_di_medicare_next(
                labor_supply, prob_disabled_next[period, health]
            ),
            survival=survival_probs[period, health],
        )

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
        age: Age,
        period: Period,
        health: DiscreteState,
        survival_probs: FloatND,
    ) -> FloatND:
        target = select_target_for_age(age + 1, gets_medicare, own)
        return build_regime_probs(target, survival_probs[period, health])

    return transition


def _build_functions(spec: RegimeSpec) -> dict:
    """Build functions dict for a nongroup regime."""
    can_work = spec["canwork"] == "canwork"
    functions = build_common_functions(spec)

    functions["ss_benefit"] = select_ss_benefit(spec)

    # his and crossed_oamc_threshold are fixed params (constants per regime),
    # not DAG functions. pylcm resolves them from the params dict.

    if _has_buy_private(spec):
        functions["plan_premium"] = health_insurance.premium
        functions["private_premium_intercept"] = (
            health_insurance.private_premium_intercept
        )
    elif can_work:
        functions["plan_premium"] = health_insurance.premium_insured
    else:
        functions["plan_premium"] = health_insurance.premium_retired
    # Medicaid-eligible households pay no premium: Medicaid replaces private
    # cover and pays the Medicare premium.
    functions["hic_premium"] = health_insurance.medicaid_adjusted_premium

    functions.update(build_pension_functions(spec))

    return functions


def _has_buy_private(spec: RegimeSpec) -> bool:
    return spec["his"] == "nongroup" and spec["mc"] == "nomc"


def build_constraints(spec: RegimeSpec) -> dict:
    """Build the regime's choice constraints.

    Where private cover is a choice, it is feasible only if its premium is
    paid in full (`private_cover_paid_in_full`).
    """
    if _has_buy_private(spec):
        return {
            "private_cover_paid_in_full": health_insurance.private_cover_paid_in_full
        }
    return {}


def build_regime(
    name: str,
    grids: Grids,
    *,
    dcegm_solver: DCEGM | None = None,
    nbegm_solver: NBEGM | None = None,
) -> Regime:
    """Build a nongroup regime."""
    spec = REGIME_SPECS[name]
    gets_mc = spec["mc"] != "nomc"
    own, _ng = make_targets(name)

    if spec["canwork"] == "canwork":
        transition_func = _make_transition_canwork(own)
    else:
        transition_func = _make_transition_forcedout(gets_mc, own)

    states = build_states(spec, grids)

    egm_solver = dcegm_solver if dcegm_solver is not None else nbegm_solver
    state_solver = (
        "brute_force"
        if egm_solver is None
        else ("nbegm" if nbegm_solver is not None else "dcegm")
    )
    functions = _build_functions(spec)
    # The EGM solvers reject constraints that read the continuous state, which
    # the private-cover constraint does through `premium_default`.
    constraints = build_constraints(spec) if egm_solver is None else {}
    if nbegm_solver is not None:
        # NBEGM solves only this regime, so its solver-contract functions are
        # regime-level here rather than broadcast model-wide. The broadcast
        # borrowing constraint stays: the EGM solve enforces the limit through
        # the savings grid's lower bound, but forward simulation re-decides
        # consumption by an argmax over the consumption grid and needs the
        # explicit feasibility mask.
        functions = {**functions, **build_nbegm_functions()}
    return build_alive_regime(
        egm_solver=egm_solver,
        transition=build_granular_regime_transition(
            transition_func=transition_func, target_ids=own.values()
        ),
        active=make_active_func(spec),
        states=states,
        state_transitions=build_state_transitions(spec, solver=state_solver),
        actions=build_actions(spec, grids),
        functions=functions,
        constraints=constraints,
    )
