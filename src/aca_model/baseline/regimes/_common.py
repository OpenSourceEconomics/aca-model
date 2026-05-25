"""Shared definitions and spec-driven builders for all regime types.

Contains RegimeId, REGIME_SPECS, grid constants, state/action builders, and
build_common_functions. No policy logic, no HIS-specific conditionals.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, TypedDict

import jax.numpy as jnp
import numpy as np
from lcm import (
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    NormalIIDProcess,
    PiecewiseGridSegment,
    PiecewiseLinSpacedGrid,
    Regime,
    RouwenhorstAR1Process,
    categorical,
)
from lcm.typing import BoolND, FloatND, IntND, RegimeName, ScalarInt, UserParams

from aca_model.agent import (
    assets_and_income,
    health,
    labor_market,
    preferences,
)
from aca_model.agent.health import Health, HealthWithDisability
from aca_model.agent.labor_market import LaborSupply, LaggedLaborSupply, SpousalIncome
from aca_model.baseline import health_insurance
from aca_model.baseline.health_insurance import BuyPrivate
from aca_model.config import MODEL_CONFIG, GridConfig
from aca_model.environment import social_security, taxes
from aca_model.environment.social_security import ClaimedSS


@categorical(ordered=False)
class RegimeId:
    retiree_nomc_inelig_canwork: ScalarInt
    tied_nomc_inelig_canwork: ScalarInt
    nongroup_nomc_inelig_canwork: ScalarInt
    retiree_dimc_inelig_canwork: ScalarInt
    nongroup_dimc_inelig_canwork: ScalarInt
    retiree_nomc_choose_canwork: ScalarInt
    tied_nomc_choose_canwork: ScalarInt
    nongroup_nomc_choose_canwork: ScalarInt
    retiree_dimc_choose_canwork: ScalarInt
    nongroup_dimc_choose_canwork: ScalarInt
    retiree_oamc_choose_canwork: ScalarInt
    tied_oamc_choose_canwork: ScalarInt
    nongroup_oamc_choose_canwork: ScalarInt
    retiree_oamc_forced_canwork: ScalarInt
    tied_oamc_forced_canwork: ScalarInt
    nongroup_oamc_forced_canwork: ScalarInt
    retiree_oamc_forced_forcedout: ScalarInt
    nongroup_oamc_forced_forcedout: ScalarInt
    dead: ScalarInt


class RegimeSpec(TypedDict):
    """Structural decomposition of a regime: (HIS, Medicare, SS, work) axes."""

    his: Literal["retiree", "tied", "nongroup"]
    mc: Literal["nomc", "dimc", "oamc"]
    ss: Literal["inelig", "choose", "forced"]
    canwork: Literal["canwork", "forcedout"]


# {his}_{mc}_{ss}_{canwork}
REGIME_SPECS: dict[str, RegimeSpec] = {
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
    aime: PiecewiseLinSpacedGrid
    consumption_dollars: IrregSpacedGrid
    wage_res: Any
    hcc_persistent: Any
    hcc_transitory: Any
    pref_type: DiscreteGrid


# AIME piecewise grid: number of points per segment between the PIA
# bend points (0 → kink_0 → kink_1 → kink_2). Total = 32.
_AIME_PIECE_N_POINTS: tuple[int, int, int] = (10, 11, 11)


# AR(1) persistence of the Rouwenhorst shocks. Calibrated once; not
# routed through fixed_params because they shape the grid topology
# rather than feed any DAG function. The Rouwenhorst innovation std is
# `sqrt(1 - rho**2)` so the grid carries unit unconditional variance.
_HCC_RHO = 0.925
_WAGE_RHO = 0.977


def build_grids(
    *,
    grid_config: GridConfig,
    fixed_params: UserParams,
    wage_params: Mapping[str, Any],
    pref_type_grid: DiscreteGrid,
) -> Grids:
    """Build continuous-state/action grids from a `GridConfig`.

    The AIME grid is `PiecewiseLinSpacedGrid` breakpointed at the PIA
    bends from `fixed_params["pia_aime_grid"]` (total 32 points). The
    assets grid's lower bound is `-max_annual_labor_income` computed
    from `wage_params` (`log_ft_wage_mean`, `log_ft_wage_std`,
    `adj_wage_hours_*`).

    `wage_params` is passed separately rather than via `fixed_params`
    because `log_ft_wage_mean` is a per-iteration param at estimation
    time (reconstructed from `wage_bias_coeffs_*`), not a fixed one;
    the grid floor must still be known at build time.
    """
    # Unit-variance standardised shocks: the total_costs / wage
    # formulas rescale these by fixed_params-level std parameters
    # (std_xsect_persistent for hcc, log_ft_wage_std for wage). For the
    # grid to have unconditional variance 1, the Rouwenhorst innovation
    # std must be √(1 − ρ²). Passing the σ_y itself (≈0.577 for hcc,
    # 0.5627 for wage) would mis-scale the grid.
    wage_res = RouwenhorstAR1Process(
        n_points=grid_config.n_wage_res_gridpoints,
        rho=_WAGE_RHO,
        sigma=(1.0 - _WAGE_RHO**2) ** 0.5,
        mu=0.0,
        batch_size=grid_config.n_wage_res_batch_size,
    )
    hcc_persistent = get_hcc_persistent_shock(grid_config=grid_config)
    hcc_transitory = NormalIIDProcess(
        n_points=grid_config.n_hcc_transitory_gridpoints,
        gauss_hermite=True,
        mu=0.0,
        sigma=1.0,
    )

    assets_start = -_compute_max_annual_labor_income(
        wage_params=wage_params, wage_res_grid=wage_res
    )

    return Grids(
        assets=LinSpacedGrid(
            start=assets_start,
            stop=500_000.0,
            n_points=grid_config.n_assets_gridpoints,
            batch_size=grid_config.n_assets_batch_size,
            distributed=True,
        ),
        aime=_build_aime_grid(grid_config=grid_config, fixed_params=fixed_params),
        consumption_dollars=IrregSpacedGrid(
            n_points=grid_config.n_consumption_dollars_gridpoints,
        ),
        wage_res=wage_res,
        hcc_persistent=hcc_persistent,
        hcc_transitory=hcc_transitory,
        pref_type=pref_type_grid,
    )


def get_hcc_persistent_shock(*, grid_config: GridConfig) -> RouwenhorstAR1Process:
    """Return the persistent-HCC AR(1) shock grid for a given `grid_config`.

    Exposed so callers that need the shock's gridpoints / transition
    probs (e.g. `assemble_fixed_params`, the HCC insurer predictor)
    can derive them from `grid_config` alone without instantiating a
    full `Model`.
    """
    return RouwenhorstAR1Process(
        n_points=grid_config.n_hcc_persistent_gridpoints,
        rho=_HCC_RHO,
        sigma=(1.0 - _HCC_RHO**2) ** 0.5,
        mu=0.0,
    )


def get_hcc_persistent_grid_points(*, grid_config: GridConfig) -> FloatND:
    """Materialise the persistent-HCC shock gridpoints for `grid_config`."""
    return get_hcc_persistent_shock(grid_config=grid_config).to_jax()


def _build_aime_grid(
    *, grid_config: GridConfig, fixed_params: UserParams
) -> PiecewiseLinSpacedGrid:
    """Return the AIME grid.

    The grid is piecewise-linspaced with breakpoints at the PIA bends
    in `fixed_params["pia_aime_grid"]` and `_AIME_PIECE_N_POINTS` in
    each segment. `n_aime_gridpoints` from `grid_config` is ignored on
    this path; the total is fixed by the PIA structure (32 points).
    """
    kinks = [float(k) for k in np.asarray(fixed_params["pia_aime_grid"])]
    segments = (
        PiecewiseGridSegment(
            interval=f"[{kinks[0]}, {kinks[1]})", n_points=_AIME_PIECE_N_POINTS[0]
        ),
        PiecewiseGridSegment(
            interval=f"[{kinks[1]}, {kinks[2]})", n_points=_AIME_PIECE_N_POINTS[1]
        ),
        PiecewiseGridSegment(
            interval=f"[{kinks[2]}, {kinks[3]}]", n_points=_AIME_PIECE_N_POINTS[2]
        ),
    )
    return PiecewiseLinSpacedGrid(
        segments=segments, batch_size=grid_config.n_aime_batch_size
    )


def _compute_max_annual_labor_income(
    *,
    wage_params: Mapping[str, Any],
    wage_res_grid: RouwenhorstAR1Process,
) -> float:
    """Return the annual labor income at the top of the wage grid.

    Used to set the assets-floor so that someone at the floor cannot
    close the gap in a single year even working full-time at the
    wage-grid upper bound — the "tough case" the model should be able
    to represent without extrapolating outside the asset grid.

    Formula matches `labor_market.labor_income` at the max-wage,
    max-hours corner:
        max_wage = exp(max(log_ft_wage_mean) + log_ft_wage_std * max(wage_res))
        income   = max_wage * max_hours**(1 + exp) * int**(-exp)
    """
    log_ft_wage_mean = wage_params["log_ft_wage_mean"]
    log_ft_wage_std = float(wage_params["log_ft_wage_std"])
    adj_wage_hours_exp = float(wage_params["adj_wage_hours_exp"])
    adj_wage_hours_int = float(wage_params["adj_wage_hours_int"])

    max_wage_res = float(wage_res_grid.get_gridpoints().max())
    max_wage = float(
        np.exp(float(log_ft_wage_mean.max()) + log_ft_wage_std * max_wage_res)
    )
    max_hours = float(labor_market.HOURS_VALUES.max())

    return (
        max_wage
        * max_hours ** (1.0 + adj_wage_hours_exp)
        * adj_wage_hours_int ** (-adj_wage_hours_exp)
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


def make_active_func(spec: RegimeSpec) -> Callable[..., Any]:
    """Return the age predicate for a regime spec."""
    key = (spec["mc"], spec["ss"], spec["canwork"])
    predicate = _ACTIVE_PREDICATES.get(key)
    if predicate is None:
        msg = f"Unknown regime spec: {spec}"
        raise ValueError(msg)
    return predicate


def build_states(spec: RegimeSpec, grids: Grids) -> dict:
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
    states["pref_type"] = grids.pref_type
    if can_work:
        states["log_ft_wage_res"] = grids.wage_res
    if can_work and spec["his"] != "tied":
        states["lagged_labor_supply"] = DiscreteGrid(LaggedLaborSupply)
    if spec["ss"] == "choose":
        states["claimed_ss"] = DiscreteGrid(ClaimedSS)
    return states


def build_actions(spec: RegimeSpec, grids: Grids) -> dict:
    """Build the action dict for a non-dead regime."""
    actions: dict = {}
    if spec["ss"] == "choose":
        actions["claim_ss"] = DiscreteGrid(ClaimedSS)
    if spec["canwork"] == "canwork":
        actions["labor_supply"] = DiscreteGrid(LaborSupply)
    if spec["his"] == "nongroup" and spec["mc"] == "nomc":
        actions["buy_private"] = DiscreteGrid(BuyPrivate)
    actions["consumption_dollars"] = grids.consumption_dollars
    return actions


def build_regime_probs(target: IntND, survival: FloatND) -> FloatND:
    """Build regime transition probability vector."""
    probs = jnp.zeros(19)
    probs = probs.at[RegimeId.dead].set(1.0 - survival)
    return probs.at[target].add(survival)


def build_dead_regime(grids: Grids) -> Regime:
    """Build the terminal dead regime.

    `pref_type` is retained as a state so the pref-type-indexed DAG
    functions (`consumption_weight`, `coefficient_rra`,
    `utility_scale_factor`) can resolve their per-cell scalar in the
    bequest utility.
    """
    return Regime(
        transition=None,
        functions={
            "utility": preferences.bequest,
            "consumption_weight": preferences.consumption_weight,
            "coefficient_rra": preferences.coefficient_rra,
            "utility_scale_factor": preferences.utility_scale_factor,
        },
        states={
            "assets": grids.assets,
            "pref_type": grids.pref_type,
        },
        active=lambda _age: True,
    )


def select_ss_benefit(spec: RegimeSpec) -> Callable[..., Any]:
    """Select the appropriate SS benefit function for a regime."""
    ss = spec["ss"]

    if ss == "forced":
        return social_security.benefit_forced
    if ss == "choose" and spec["mc"] == "oamc":
        return social_security.benefit_choose_post65
    if ss == "choose":
        return social_security.benefit_choose_pre65
    return social_security.benefit_inelig_pre65


def _select_leisure(spec: RegimeSpec) -> Callable[..., Any]:
    """Select the leisure function for a non-dead regime."""
    if spec["canwork"] == "forcedout":
        return preferences.leisure_forcedout
    if spec["his"] == "tied":
        return preferences.leisure_canwork_tied
    return preferences.leisure_canwork_retiree_or_nongroup


def build_common_functions(spec: RegimeSpec) -> dict:
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
        functions["labor_income"] = labor_market.income
        functions["fixed_cost_of_work"] = preferences.fixed_cost_of_work

    functions["leisure"] = _select_leisure(spec)
    functions["utility"] = preferences.u_alive
    functions["capital_income"] = assets_and_income.capital_income
    # spousal_income_amounts is a lookup table param, not a DAG function
    functions["is_married"] = labor_market.is_married
    functions["equivalence_scale"] = preferences.equivalence_scale
    functions["utility_scale_factor"] = preferences.utility_scale_factor
    functions["consumption_weight"] = preferences.consumption_weight
    functions["coefficient_rra"] = preferences.coefficient_rra
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
    functions["consumption_dollars_floor"] = assets_and_income.consumption_dollars_floor
    functions["transfers"] = assets_and_income.transfers
    functions["consumption_equiv"] = preferences.consumption_equiv

    return functions


def precompute_target_regimes(spec: RegimeSpec) -> MappingProxyType[str, int]:
    """Pre-compute target regime IDs for each next-age bracket.

    Coerces each `RegimeId.<name>` (`ScalarInt`, post-pylcm#349) to a
    Python `int` so the returned mapping's values can serve as dict
    keys and `in`-set members downstream.
    """

    def _resolve(his_val: str, mc_val: str, ss_val: str, canwork_val: str) -> int:
        for name, s in REGIME_SPECS.items():
            if (
                s["his"] == his_val
                and s["mc"] == mc_val
                and s["ss"] == ss_val
                and s["canwork"] == canwork_val
            ):
                return int(getattr(RegimeId, name))
        return int(RegimeId.dead)

    ng_his = "nongroup" if spec["his"] == "tied" else spec["his"]

    return MappingProxyType(
        {
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
    )


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
    target_regimes = precompute_target_regimes(REGIME_SPECS[name])
    own = {k: target_regimes[k] for k in _TARGET_KEYS}
    ng = {k: target_regimes[k + "_ng"] for k in _TARGET_KEYS}
    return own, ng


def select_target_for_age(
    next_age: int | IntND | FloatND,
    mc_next: bool | BoolND,
    tgts: dict[str, int],
) -> IntND:
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


def build_state_transitions(spec: RegimeSpec) -> dict:
    """Build the state transitions dict for a non-dead regime."""
    transitions: dict = {}
    transitions["assets"] = _build_per_target_regime_assets(spec)
    transitions["health"] = _build_per_target_regime_health(spec)
    claimed_ss_transition = _build_per_target_regime_claimed_ss(spec)
    if claimed_ss_transition:
        transitions["claimed_ss"] = claimed_ss_transition
    lagged_labor_supply_transition = _build_per_target_regime_lagged_labor_supply(spec)
    if lagged_labor_supply_transition:
        transitions["lagged_labor_supply"] = lagged_labor_supply_transition
    transitions["pref_type"] = None
    transitions["aime"] = (
        social_security.next_aime
        if spec["mc"] == "oamc"
        else social_security.next_aime_disabled
    )
    transitions["spousal_income"] = MarkovTransition(labor_market.next_spousal_income)
    return transitions


def _build_per_target_regime_assets(
    spec: RegimeSpec,
) -> dict[RegimeName, Callable[..., FloatND]]:
    """Build per-target assets transitions.

    The `dead` target uses `next_assets_when_dead` (no
    `pension_assets_adjustment`), so the dead per-target DAG does not
    pull in the `next_aime`-dependent imputation chain — `dead` has no
    `aime` state and pylcm cannot resolve `next_aime` there. Non-dead
    targets use the full `next_assets` with the pension correction.
    """
    target_regimes = precompute_target_regimes(spec)
    id_to_name = {int(getattr(RegimeId, name)): name for name in REGIME_SPECS}

    result: dict[RegimeName, Callable[..., FloatND]] = {}
    seen_ids: set[int] = set()

    for target_id in target_regimes.values():
        if target_id in seen_ids:
            continue
        seen_ids.add(target_id)
        target_name = id_to_name.get(target_id)
        if target_name is None:
            continue
        result[target_name] = assets_and_income.next_assets

    result["dead"] = assets_and_income.next_assets_when_dead
    return result


def _build_per_target_regime_health(
    spec: RegimeSpec,
) -> dict[RegimeName, MarkovTransition]:
    """Build per-target health transitions.

    Pre-65 regimes use HealthWithDisability (3-state), post-65 use Health (2-state).
    Cross-grid transitions (3->2) happen at the age-65 boundary.
    """
    target_regimes = precompute_target_regimes(spec)
    id_to_name = {int(getattr(RegimeId, name)): name for name in REGIME_SPECS}

    result: dict[RegimeName, MarkovTransition] = {}
    seen_ids: set[int] = set()

    for target_id in target_regimes.values():
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


def _build_per_target_regime_claimed_ss(
    spec: RegimeSpec,
) -> dict[RegimeName, Callable[..., BoolND]]:
    """Build per-target claimed_ss transitions.

    - `choose` regimes (source has `claimed_ss`): absorbing transition.
    - `inelig` regimes (source lacks `claimed_ss`): enter with `ClaimedSS.no`.
    - `forced`/`forcedout` regimes: no targets have `claimed_ss` → empty.
    """
    if spec["ss"] in ("forced", "forcedout"):
        return {}

    target_regimes = precompute_target_regimes(spec)
    id_to_name = {int(getattr(RegimeId, name)): name for name in REGIME_SPECS}

    result: dict[RegimeName, Callable[..., BoolND]] = {}
    seen_ids: set[int] = set()

    for target_id in target_regimes.values():
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


def _build_per_target_regime_lagged_labor_supply(
    spec: RegimeSpec,
) -> dict[RegimeName, Callable[..., BoolND]]:
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

    target_regimes = precompute_target_regimes(spec)
    id_to_name = {int(getattr(RegimeId, name)): name for name in REGIME_SPECS}

    result: dict[RegimeName, Callable[..., BoolND]] = {}
    seen_ids: set[int] = set()

    for target_id in target_regimes.values():
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
