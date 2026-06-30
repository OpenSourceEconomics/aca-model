"""Regime transitions and builder for nongroup HIS regimes.

Nongroup regimes: agents purchasing individual-market health insurance.
Already nongroup, so no SSI/Medicaid override needed for HIS transitions.
"""

import functools
from collections.abc import Callable

from lcm import Regime
from lcm.solvers import BQSEGM, DCEGM
from lcm.typing import Age, DiscreteAction, FloatND, Period

from aca_model.agent.labor_market import LaborSupply
from aca_model.baseline import health_insurance
from aca_model.baseline.health_insurance import BuyPrivate
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
) -> Callable[..., FloatND]:
    """Create transition for canwork nongroup regimes.

    Already nongroup — no SSI override needed. Gets Medicare if stops working
    (when gets_medicare is True).
    """

    def transition(
        age: Age,
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
        age: Age,
        period: Period,
        survival_probs: FloatND,
    ) -> FloatND:
        target = select_target_for_age(age + 1, gets_medicare, own)
        return build_regime_probs(target, survival_probs[period])

    return transition


def _build_functions(spec: RegimeSpec, *, fix_buy_private: bool = False) -> dict:
    """Build functions dict for a nongroup regime.

    With `fix_buy_private`, the BQSEGM M1 slice binds `buy_private` to
    `BuyPrivate.yes` in its consumers (premium, OOP), so the discrete action is
    a single fixed level. The binding selects the purchase branch of each
    consumer — algebraically the `buy_private == BuyPrivate.yes` arm — and
    leaves the remaining budget structure untouched.
    """
    can_work = spec["canwork"] == "canwork"
    functions = build_common_functions(spec)

    functions["ss_benefit"] = select_ss_benefit(spec)

    # his and crossed_oamc_threshold are fixed params (constants per regime),
    # not DAG functions. pylcm resolves them from the params dict.

    has_buy_private = spec["his"] == "nongroup" and spec["mc"] == "nomc"
    if has_buy_private:
        functions["hic_premium"] = health_insurance.premium
    elif can_work:
        functions["hic_premium"] = health_insurance.premium_insured
    else:
        functions["hic_premium"] = health_insurance.premium_retired

    if has_buy_private and fix_buy_private:
        functions["hic_premium"] = functools.partial(
            health_insurance.premium, buy_private=BuyPrivate.yes
        )
        functions["primary_oop"] = functools.partial(
            health_insurance.primary_oop, buy_private=BuyPrivate.yes
        )

    functions.update(build_pension_functions(spec))

    return functions


def build_regime(
    name: str,
    grids: Grids,
    *,
    dcegm_solver: DCEGM | None = None,
    bqsegm_solver: BQSEGM | None = None,
) -> Regime:
    """Build a nongroup regime."""
    spec = REGIME_SPECS[name]
    gets_mc = spec["mc"] != "nomc"
    own, _ng = make_targets(name)

    if spec["canwork"] == "canwork":
        transition_func = _make_transition_canwork(gets_mc, own)
    else:
        transition_func = _make_transition_forcedout(gets_mc, own)

    states = build_states(spec, grids)

    egm_solver = dcegm_solver if dcegm_solver is not None else bqsegm_solver
    solver_kwargs: dict = {} if egm_solver is None else {"solver": egm_solver}
    state_solver = (
        "brute_force"
        if egm_solver is None
        else ("bqsegm" if bqsegm_solver is not None else "dcegm")
    )
    # The BQSEGM M1 slice fixes the discrete actions to a single level so the
    # only choice is continuous consumption against the cliffed budget.
    fix_for_bqsegm = bqsegm_solver is not None
    return Regime(
        transition=build_granular_regime_transition(
            transition_func=transition_func, target_ids=own.values()
        ),
        active=make_active_func(spec),
        states=states,
        state_transitions=build_state_transitions(spec, solver=state_solver),
        actions=build_actions(spec, grids, drop_buy_private=fix_for_bqsegm),
        functions=_build_functions(spec, fix_buy_private=fix_for_bqsegm),
        **solver_kwargs,
    )
