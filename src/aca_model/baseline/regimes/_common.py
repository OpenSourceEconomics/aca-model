"""Shared definitions and spec-driven builders for all regime types.

Contains RegimeId, REGIME_SPECS, grid constants, state/action builders, and
build_common_functions. No policy logic, no HIS-specific conditionals.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import lcm.shocks.ar1
import lcm.shocks.iid
from lcm import (
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Regime,
    categorical,
)
from lcm.typing import BoolND, FloatND

from aca_model.agent import (
    assets_and_income,
    health,
    labor_market,
    preferences,
    utility,
)
from aca_model.agent.health import Health, HealthWithDisability
from aca_model.agent.labor_market import LaborSupply, LaggedLaborSupply, SpousalIncome
from aca_model.agent.preferences import PrefType
from aca_model.baseline import health_insurance
from aca_model.baseline.health_insurance import BuyPrivate
from aca_model.config import GRID_CONFIG, MODEL_CONFIG, GridConfig
from aca_model.environment import social_security, taxes
from aca_model.environment.social_security import ClaimedSS


@categorical(ordered=False)
class RegimeId:
    retiree_nomc_inelig_canwork: int
    tied_nomc_inelig_canwork: int
    nongroup_nomc_inelig_canwork: int
    retiree_dimc_inelig_canwork: int
    nongroup_dimc_inelig_canwork: int
    retiree_nomc_choose_canwork: int
    tied_nomc_choose_canwork: int
    nongroup_nomc_choose_canwork: int
    retiree_dimc_choose_canwork: int
    nongroup_dimc_choose_canwork: int
    retiree_oamc_choose_canwork: int
    tied_oamc_choose_canwork: int
    nongroup_oamc_choose_canwork: int
    retiree_oamc_forced_canwork: int
    tied_oamc_forced_canwork: int
    nongroup_oamc_forced_canwork: int
    retiree_oamc_forced_forcedout: int
    nongroup_oamc_forced_forcedout: int
    dead: int


# {his}_{mc}_{ss}_{canwork}
REGIME_SPECS: dict[str, dict[str, str]] = {
    "retiree_nomc_inelig_canwork": {
        "his": "retiree",
        "mc": "nomc",
        "ss": "inelig",
        "canwork": "canwork",
    },
    "tied_nomc_inelig_canwork": {
        "his": "tied",
        "mc": "nomc",
        "ss": "inelig",
        "canwork": "canwork",
    },
    "nongroup_nomc_inelig_canwork": {
        "his": "nongroup",
        "mc": "nomc",
        "ss": "inelig",
        "canwork": "canwork",
    },
    "retiree_dimc_inelig_canwork": {
        "his": "retiree",
        "mc": "dimc",
        "ss": "inelig",
        "canwork": "canwork",
    },
    "nongroup_dimc_inelig_canwork": {
        "his": "nongroup",
        "mc": "dimc",
        "ss": "inelig",
        "canwork": "canwork",
    },
    "retiree_nomc_choose_canwork": {
        "his": "retiree",
        "mc": "nomc",
        "ss": "choose",
        "canwork": "canwork",
    },
    "tied_nomc_choose_canwork": {
        "his": "tied",
        "mc": "nomc",
        "ss": "choose",
        "canwork": "canwork",
    },
    "nongroup_nomc_choose_canwork": {
        "his": "nongroup",
        "mc": "nomc",
        "ss": "choose",
        "canwork": "canwork",
    },
    "retiree_dimc_choose_canwork": {
        "his": "retiree",
        "mc": "dimc",
        "ss": "choose",
        "canwork": "canwork",
    },
    "nongroup_dimc_choose_canwork": {
        "his": "nongroup",
        "mc": "dimc",
        "ss": "choose",
        "canwork": "canwork",
    },
    "retiree_oamc_choose_canwork": {
        "his": "retiree",
        "mc": "oamc",
        "ss": "choose",
        "canwork": "canwork",
    },
    "tied_oamc_choose_canwork": {
        "his": "tied",
        "mc": "oamc",
        "ss": "choose",
        "canwork": "canwork",
    },
    "nongroup_oamc_choose_canwork": {
        "his": "nongroup",
        "mc": "oamc",
        "ss": "choose",
        "canwork": "canwork",
    },
    "retiree_oamc_forced_canwork": {
        "his": "retiree",
        "mc": "oamc",
        "ss": "forced",
        "canwork": "canwork",
    },
    "tied_oamc_forced_canwork": {
        "his": "tied",
        "mc": "oamc",
        "ss": "forced",
        "canwork": "canwork",
    },
    "nongroup_oamc_forced_canwork": {
        "his": "nongroup",
        "mc": "oamc",
        "ss": "forced",
        "canwork": "canwork",
    },
    "retiree_oamc_forced_forcedout": {
        "his": "retiree",
        "mc": "oamc",
        "ss": "forced",
        "canwork": "forcedout",
    },
    "nongroup_oamc_forced_forcedout": {
        "his": "nongroup",
        "mc": "oamc",
        "ss": "forced",
        "canwork": "forcedout",
    },
}

config = MODEL_CONFIG


@dataclass(frozen=True)
class Grids:
    assets: LinSpacedGrid
    aime: LinSpacedGrid
    consumption: LinSpacedGrid
    wage_res: Any
    hcc_persistent: Any
    hcc_transitory: Any


def build_grids(grid_config: GridConfig = GRID_CONFIG) -> Grids:
    """Build continuous-state/action grids from a GridConfig."""
    return Grids(
        assets=LinSpacedGrid(
            start=0.0,
            stop=500_000.0,
            n_points=grid_config.n_assets_gridpoints,
            batch_size=grid_config.n_assets_batch_size,
        ),
        aime=LinSpacedGrid(
            start=0.0,
            stop=8_000.0,
            n_points=grid_config.n_aime_gridpoints,
        ),
        consumption=LinSpacedGrid(
            start=100.0,
            stop=100_000.0,
            n_points=grid_config.n_consumption_gridpoints,
        ),
        wage_res=lcm.shocks.ar1.Rouwenhorst(
            n_points=grid_config.n_wage_res_gridpoints,
            rho=0.977,
            sigma=0.5627,
            mu=0.0,
        ),
        hcc_persistent=lcm.shocks.ar1.Rouwenhorst(
            n_points=grid_config.n_hcc_persistent_gridpoints,
            rho=0.925,
            sigma=0.5772347875864725,
            mu=0.0,
        ),
        hcc_transitory=lcm.shocks.iid.Normal(
            n_points=grid_config.n_hcc_transitory_gridpoints,
            gauss_hermite=True,
            mu=0.0,
            sigma=1.0,
        ),
    )


_ACTIVE_PREDICATES: dict[tuple[str, str, str], Callable[..., Any]] = {
    ("nomc", "inelig", "canwork"): lambda age: age < config.ss_early_age,
    ("dimc", "inelig", "canwork"): lambda age: age < config.ss_early_age,
    ("nomc", "choose", "canwork"): lambda age: (
        (age >= config.ss_early_age) & (age < config.medicare_age)
    ),
    ("dimc", "choose", "canwork"): lambda age: (
        (age >= config.ss_early_age) & (age < config.medicare_age)
    ),
    ("oamc", "choose", "canwork"): lambda age: (
        (age >= config.medicare_age) & (age < config.ss_forced_age)
    ),
    ("oamc", "forced", "canwork"): lambda age: (
        (age >= config.ss_forced_age) & (age < config.work_forced_out_age)
    ),
    ("oamc", "forced", "forcedout"): lambda age: (
        (age >= config.work_forced_out_age) & (age < config.end_age - 1)
    ),
}


def make_active_func(spec: dict[str, str]) -> Callable[..., Any]:
    """Return the age predicate for a regime spec."""
    key = (spec["mc"], spec["ss"], spec["canwork"])
    predicate = _ACTIVE_PREDICATES.get(key)
    if predicate is None:
        msg = f"Unknown regime spec: {spec}"
        raise ValueError(msg)
    return predicate


def build_states(spec: dict[str, str], grids: Grids) -> dict:
    """Build the state dict for a non-dead regime."""
    can_work = spec["canwork"] == "canwork"

    states: dict = {}
    states["assets"] = grids.assets
    states["aime"] = grids.aime
    states["health"] = DiscreteGrid(
        Health if spec["mc"] == "oamc" else HealthWithDisability
    )
    states["hcc_persistent"] = grids.hcc_persistent
    states["hcc_transitory"] = grids.hcc_transitory
    states["spousal_income"] = DiscreteGrid(SpousalIncome)
    states["pref_type"] = DiscreteGrid(PrefType)
    if can_work:
        states["log_ft_wage_res"] = grids.wage_res
    if can_work and spec["his"] != "tied":
        states["lagged_labor_supply"] = DiscreteGrid(LaggedLaborSupply)
    if spec["ss"] == "choose":
        states["claimed_ss"] = DiscreteGrid(ClaimedSS)
    return states


def build_actions(spec: dict[str, str], grids: Grids) -> dict:
    """Build the action dict for a non-dead regime."""
    actions: dict = {}
    if spec["ss"] == "choose":
        actions["claim_ss"] = DiscreteGrid(ClaimedSS)
    if spec["canwork"] == "canwork":
        actions["labor_supply"] = DiscreteGrid(LaborSupply)
    if spec["his"] == "nongroup" and spec["mc"] == "nomc":
        actions["buy_private"] = DiscreteGrid(BuyPrivate)
    actions["consumption"] = grids.consumption
    return actions


def build_regime_probs(target: FloatND, survival: FloatND) -> FloatND:
    """Build regime transition probability vector."""
    probs = jnp.zeros(19)
    probs = probs.at[RegimeId.dead].set(1.0 - survival)
    return probs.at[target].add(survival)


def build_dead_regime(grids: Grids) -> Regime:
    """Build the terminal dead regime.

    `pref_type` is retained as a state so type-indexed preference params
    (`consumption_weight`, `coefficient_rra`, `utility_scale_factor`) can
    be indexed by it in the bequest utility.
    """
    return Regime(
        transition=None,
        functions={
            "utility": utility.dead,
            "utility_scale_factor": preferences.utility_scale_factor,
        },
        states={
            "assets": grids.assets,
            "pref_type": DiscreteGrid(PrefType),
        },
        active=lambda _age: True,
    )


def select_ss_benefit(spec: dict[str, str]) -> Callable[..., Any]:
    """Select the appropriate SS benefit function for a regime."""
    ss = spec["ss"]

    if ss == "forced":
        return social_security.benefit_forced
    if ss == "choose" and spec["mc"] == "oamc":
        return social_security.benefit_choose_post65
    if ss == "choose":
        return social_security.benefit_choose_pre65
    return social_security.benefit_inelig_pre65


def select_utility(spec: dict[str, str]) -> Callable[..., Any]:
    """Select the utility function for a regime."""
    if spec["canwork"] != "canwork":
        return utility.retired
    return preferences.utility


def _select_leisure(spec: dict[str, str]) -> Callable[..., Any]:
    """Select the leisure function for a canwork regime."""
    if spec["his"] == "tied":
        return preferences.leisure_tied
    return preferences.leisure


def build_common_functions(spec: dict[str, str]) -> dict:
    """Build the shared functions dict for a non-dead regime.

    Contains all functions common to every HIS type. Per-HIS modules add
    utility, ss_benefit, his, gets_medicare, hic_premium, and pension entries.
    """
    can_work = spec["canwork"] == "canwork"

    functions: dict = {}
    functions["good_health"] = (
        health.is_good_health_2 if spec["mc"] == "oamc" else health.is_good_health_3
    )
    functions["total_health_costs"] = health_insurance.total_costs
    has_buy_private = spec["his"] == "nongroup" and spec["mc"] == "nomc"
    functions["primary_oop"] = (
        health_insurance.primary_oop if has_buy_private else health_insurance.oop_costs
    )
    functions["oop_costs"] = health_insurance.oop_with_medicaid

    if can_work:
        functions["working_hours_value"] = labor_market.working_hours_value
        functions["leisure"] = _select_leisure(spec)
        functions["labor_income"] = labor_market.income

    functions["capital_income"] = assets_and_income.capital_income
    # spousal_income_amounts is a lookup table param, not a DAG function
    functions["is_married"] = labor_market.is_married
    functions["equivalence_scale"] = preferences.equivalence_scale
    functions["utility_scale_factor"] = preferences.utility_scale_factor
    # `discount_factor` is a DAG function that indexes the per-type
    # Series by the pref_type state and returns a scalar. pylcm's
    # default H picks the scalar up as a DAG-output H input.
    functions["discount_factor"] = preferences.discount_factor

    # PIA from pre-computed lookup table
    functions["pia"] = social_security.pia
    if spec["mc"] != "oamc":  # pre-65: SSDI needs dropout-adjusted PIA
        functions["ssdi_pia"] = social_security.ssdi_pia

    # SSI/Medicaid
    functions["countable_income"] = health_insurance.countable_income
    functions["is_ssi_eligible"] = health_insurance.is_ssi_eligible
    functions["is_medicaid_eligible"] = health_insurance.is_medicaid_eligible
    functions["ssi_benefit"] = health_insurance.ssi_benefit

    # Taxes
    functions["taxable_ss_benefit"] = taxes.taxable_ss_benefit
    functions["gross_income"] = taxes.gross_income
    functions["after_tax_income"] = taxes.after_tax_income
    if spec["ss"] != "forced" and can_work:
        functions["marginal_tax_rate"] = taxes.marginal_rate

    # HIC premium
    functions["predicted_hcc_insurer"] = health_insurance.hcc_insurer_predicted

    # Earnings test credit-back (only choose+canwork: has claim_ss + claimed_ss)
    if spec["ss"] == "choose" and can_work:
        functions["benefit_withheld_fraction"] = (
            social_security.benefit_withheld_fraction
        )

    # Cash on hand and transfers
    functions["cash_on_hand"] = assets_and_income.cash_on_hand
    functions["transfers"] = assets_and_income.transfers

    return functions


def precompute_targets(spec: dict[str, str]) -> dict[str, int]:
    """Pre-compute target regime IDs for each next-age bracket."""

    def _resolve(his_val: str, mc_val: str, ss_val: str, canwork_val: str) -> int:
        for name, s in REGIME_SPECS.items():
            if (
                s["his"] == his_val
                and s["mc"] == mc_val
                and s["ss"] == ss_val
                and s["canwork"] == canwork_val
            ):
                return getattr(RegimeId, name)
        return RegimeId.dead

    ng_his = "nongroup" if spec["his"] == "tied" else spec["his"]

    return {
        "forcedout": _resolve(ng_his, "oamc", "forced", "forcedout"),
        "forcedout_ng": _resolve("nongroup", "oamc", "forced", "forcedout"),
        "forced_forced": _resolve(spec["his"], "oamc", "forced", "canwork"),
        "forced_forced_ng": _resolve("nongroup", "oamc", "forced", "canwork"),
        "forced_choose": _resolve(spec["his"], "oamc", "choose", "canwork"),
        "forced_choose_ng": _resolve("nongroup", "oamc", "choose", "canwork"),
        "dimc_choose": _resolve(spec["his"], "dimc", "choose", "canwork"),
        "dimc_choose_ng": _resolve("nongroup", "dimc", "choose", "canwork"),
        "nomc_choose": _resolve(spec["his"], "nomc", "choose", "canwork"),
        "nomc_choose_ng": _resolve("nongroup", "nomc", "choose", "canwork"),
        "dimc_inelig": _resolve(spec["his"], "dimc", "inelig", "canwork"),
        "dimc_inelig_ng": _resolve("nongroup", "dimc", "inelig", "canwork"),
        "nomc_inelig": _resolve(spec["his"], "nomc", "inelig", "canwork"),
        "nomc_inelig_ng": _resolve("nongroup", "nomc", "inelig", "canwork"),
    }


_TARGET_KEYS = (
    "forcedout",
    "forced_forced",
    "forced_choose",
    "dimc_choose",
    "nomc_choose",
    "dimc_inelig",
    "nomc_inelig",
)


def make_targets(name: str) -> tuple[dict[str, int], dict[str, int]]:
    """Build own and nongroup target subsets for a regime name."""
    tgts = precompute_targets(REGIME_SPECS[name])
    own = {k: tgts[k] for k in _TARGET_KEYS}
    ng = {k: tgts[k + "_ng"] for k in _TARGET_KEYS}
    return own, ng


def select_target_for_age(
    next_age: int | FloatND,
    mc_next: bool | BoolND,
    tgts: dict[str, int],
) -> FloatND:
    """Select target regime ID based on next-period age bracket."""
    ss_choose = jnp.where(
        jnp.array(mc_next),
        tgts["dimc_choose"],
        tgts["nomc_choose"],
    )
    ss_inelig = jnp.where(
        jnp.array(mc_next),
        tgts["dimc_inelig"],
        tgts["nomc_inelig"],
    )
    return jnp.where(
        next_age >= config.end_age - 1,
        RegimeId.dead,
        jnp.where(
            next_age >= config.work_forced_out_age,
            tgts["forcedout"],
            jnp.where(
                next_age >= config.ss_forced_age,
                tgts["forced_forced"],
                jnp.where(
                    next_age >= config.medicare_age,
                    tgts["forced_choose"],
                    jnp.where(next_age >= config.ss_early_age, ss_choose, ss_inelig),
                ),
            ),
        ),
    )


def build_state_transitions(spec: dict[str, str]) -> dict:
    """Build the state transitions dict for a non-dead regime."""
    transitions: dict = {}
    transitions["health"] = _build_per_target_health(spec)
    transitions["assets"] = assets_and_income.next_assets
    transitions["pref_type"] = None
    transitions["aime"] = (
        social_security.next_aime
        if spec["mc"] == "oamc"
        else social_security.next_aime_disabled
    )
    transitions["spousal_income"] = MarkovTransition(labor_market.next_spousal_income)
    lagged_supply_transition = _build_per_target_lagged_labor_supply(spec)
    if lagged_supply_transition:
        transitions["lagged_labor_supply"] = lagged_supply_transition
    claimed_ss_transition = _build_per_target_claimed_ss(spec)
    if claimed_ss_transition:
        transitions["claimed_ss"] = claimed_ss_transition
    return transitions


def _build_per_target_health(spec: dict[str, str]) -> dict:
    """Build per-target health transitions.

    Pre-65 regimes use HealthWithDisability (3-state), post-65 use Health (2-state).
    Cross-grid transitions (3->2) happen at the age-65 boundary.
    """
    targets = precompute_targets(spec)
    id_to_name = {getattr(RegimeId, name): name for name in REGIME_SPECS}

    result: dict[str, MarkovTransition] = {}
    seen_ids: set[int] = set()

    for target_id in targets.values():
        if target_id == RegimeId.dead or target_id in seen_ids:
            continue
        seen_ids.add(target_id)
        target_name = id_to_name.get(target_id)
        if target_name is None:
            continue
        target_spec = REGIME_SPECS[target_name]
        target_is_post65 = target_spec["mc"] == "oamc"

        if spec["mc"] != "oamc" and target_is_post65:
            result[target_name] = MarkovTransition(health.next_health_cross)
        else:
            result[target_name] = MarkovTransition(health.next_health)

    return result


def _build_per_target_claimed_ss(spec: dict[str, str]) -> dict:
    """Build per-target claimed_ss transitions.

    - `choose` regimes (source has `claimed_ss`): absorbing transition.
    - `inelig` regimes (source lacks `claimed_ss`): enter with `ClaimedSS.no`.
    - `forced`/`forcedout` regimes: no targets have `claimed_ss` → empty.
    """
    if spec["ss"] in ("forced", "forcedout"):
        return {}

    targets = precompute_targets(spec)
    id_to_name = {getattr(RegimeId, name): name for name in REGIME_SPECS}

    result: dict = {}
    seen_ids: set[int] = set()

    for target_id in targets.values():
        if target_id == RegimeId.dead or target_id in seen_ids:
            continue
        seen_ids.add(target_id)
        target_name = id_to_name.get(target_id)
        if target_name is None:
            continue
        target_spec = REGIME_SPECS[target_name]
        if target_spec["ss"] != "choose":
            continue

        if spec["ss"] == "choose":
            result[target_name] = social_security.next_claimed_ss
        elif spec["ss"] == "inelig":
            result[target_name] = social_security.enter_claimed_ss

    return result


def _build_per_target_lagged_labor_supply(spec: dict[str, str]) -> dict:
    """Build per-target lagged_labor_supply transitions.

    `lagged_labor_supply` exists in canwork non-tied regimes. Tied regimes
    don't have it as a state but can transition to nongroup regimes that do.
    The transition function is the same (`next_lagged_supply`) since tied
    regimes have `labor_supply` as an action.

    Forcedout regimes have no `labor_supply` and their targets don't have
    `lagged_labor_supply`.
    """
    if spec["canwork"] != "canwork":
        return {}

    targets = precompute_targets(spec)
    id_to_name = {getattr(RegimeId, name): name for name in REGIME_SPECS}

    result: dict = {}
    seen_ids: set[int] = set()

    for target_id in targets.values():
        if target_id == RegimeId.dead or target_id in seen_ids:
            continue
        seen_ids.add(target_id)
        target_name = id_to_name.get(target_id)
        if target_name is None:
            continue
        target_spec = REGIME_SPECS[target_name]
        target_has_lagged = (
            target_spec["canwork"] == "canwork" and target_spec["his"] != "tied"
        )
        if target_has_lagged:
            result[target_name] = labor_market.next_lagged_supply

    return result
