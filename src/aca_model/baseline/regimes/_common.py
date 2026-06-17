"""Shared definitions and spec-driven builders for all regime types.

Contains RegimeId, REGIME_SPECS, grid constants, state/action builders, and
build_common_functions. No policy logic, no HIS-specific conditionals.
"""

import functools
import itertools
from collections.abc import Callable, Iterable, Mapping
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
    Phased,
    PiecewiseGridSegment,
    PiecewiseLinSpacedGrid,
    Regime,
    RouwenhorstAR1Process,
    categorical,
    fixed_transition,
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
from aca_model.environment import pensions, social_security, taxes
from aca_model.environment.social_security import ClaimedSS

SolverName = Literal["brute_force", "dcegm"]


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
    assets: LinSpacedGrid | PiecewiseLinSpacedGrid
    aime: PiecewiseLinSpacedGrid
    pension_wealth: LinSpacedGrid
    consumption_dollars: IrregSpacedGrid
    wage_res: Any
    hcc_persistent: Any
    hcc_transitory: Any
    pref_type: DiscreteGrid
    grid_config: GridConfig
    """The originating `GridConfig`. Exposed on `Grids` so `build_states`
    can read per-axis `batch_size` settings for the discrete states it
    constructs inline (health, spousal_income, lagged_labor_supply,
    claimed_ss) without changing the `build_states`/`build_regime` API."""


# AIME piecewise grid: number of points per segment between the PIA
# bend points (0 → kink_0 → kink_1 → kink_2 → extension). The fourth
# segment covers the sparse delayed-retirement-credit region above the
# taxable max, so it carries fewer points than the dense lower segments.
_AIME_PIECE_N_POINTS: tuple[int, int, int, int] = (10, 11, 11, 6)


# `pension_wealth` is a carried state (`Phased(solve=..., simulate=Grid)`):
# imputed via a derived function in solve (never a solve grid axis) and
# carried per subject in simulate. The grid is simulate-side metadata only —
# it fixes the state's continuous kind and float dtype. Seeds are taken
# verbatim from the initial conditions without bound-clamping, and the state
# is never an interpolation axis, so neither the bounds nor the point count
# affect solve cost or the simulated values.
_PENSION_WEALTH_GRID = LinSpacedGrid(start=0.0, stop=2_000_000.0, n_points=2)


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
    consumption_dollars_points: tuple[float, ...] | None = None,
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

    `consumption_dollars_points` fixes the consumption action grid at
    construction (the DC-EGM kernel needs it then); `None` keeps the
    runtime-points grid completed per iteration via params injection.
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
        assets=_build_assets_grid(
            grid_config=grid_config,
            fixed_params=fixed_params,
            assets_start=assets_start,
        ),
        aime=_build_aime_grid(grid_config=grid_config, fixed_params=fixed_params),
        pension_wealth=_PENSION_WEALTH_GRID,
        consumption_dollars=(
            IrregSpacedGrid(n_points=grid_config.n_consumption_dollars_gridpoints)
            if consumption_dollars_points is None
            else IrregSpacedGrid(points=consumption_dollars_points)
        ),
        wage_res=wage_res,
        hcc_persistent=hcc_persistent,
        hcc_transitory=hcc_transitory,
        pref_type=pref_type_grid,
        grid_config=grid_config,
    )


# Smallest assets-grid budget that gets dedicated SSI-band nodes. Below
# this (smoke-test sizes) there is no base resolution worth refining, so
# the grid stays plain linspaced.
MIN_ASSETS_GRIDPOINTS_FOR_BAND_NODES = 8

_ASSETS_GRID_STOP = 500_000.0


def _build_assets_grid(
    *,
    grid_config: GridConfig,
    fixed_params: UserParams,
    assets_start: float,
) -> LinSpacedGrid | PiecewiseLinSpacedGrid:
    """Return the assets grid, with dedicated nodes across each SSI band.

    Each distinct `ssi_assets_test` value `t` (one per household type)
    gets nodes at `{t - h, t, t + h}` with
    `h = ELIGIBILITY_BAND_HALF_WIDTH`, so the smoothed eligibility ramp
    is resolved at node resolution — a band narrower than a grid cell is
    a cliff to any solver that reasons per node. The band nodes sit on
    top of the configured base resolution, which spreads over the
    segments between bands proportionally to their span (at least two
    points per segment, the piecewise-grid minimum). Budgets below
    `MIN_ASSETS_GRIDPOINTS_FOR_BAND_NODES` keep the plain linspaced grid.
    """
    n_base = grid_config.n_assets_gridpoints
    half_width = health_insurance.ELIGIBILITY_BAND_HALF_WIDTH
    thresholds = sorted(
        {
            float(v)
            for v in np.asarray(fixed_params["ssi_assets_test"])
            if assets_start < float(v) - half_width
            and float(v) + half_width < _ASSETS_GRID_STOP
        }
    )
    if n_base < MIN_ASSETS_GRIDPOINTS_FOR_BAND_NODES or not thresholds:
        return LinSpacedGrid(
            start=assets_start,
            stop=_ASSETS_GRID_STOP,
            n_points=n_base,
            batch_size=grid_config.n_assets_batch_size,
        )
    _fail_if_bands_overlap(thresholds=thresholds, half_width=half_width)

    outer_bounds = [
        (assets_start, thresholds[0] - half_width),
        *[
            (lo + half_width, hi - half_width)
            for lo, hi in itertools.pairwise(thresholds)
        ],
        (thresholds[-1] + half_width, _ASSETS_GRID_STOP),
    ]
    outer_n_points = _allocate_points_by_span(bounds=outer_bounds, total=n_base)

    segments: list[PiecewiseGridSegment] = []
    for (lo, hi), n_outer, threshold in zip(
        outer_bounds[:-1], outer_n_points[:-1], thresholds, strict=True
    ):
        segments.append(
            PiecewiseGridSegment(interval=f"[{lo}, {hi})", n_points=n_outer)
        )
        segments.append(
            PiecewiseGridSegment(interval=f"[{hi}, {threshold})", n_points=2)
        )
        segments.append(
            PiecewiseGridSegment(
                interval=f"[{threshold}, {threshold + half_width})", n_points=2
            )
        )
    final_lo, final_hi = outer_bounds[-1]
    segments.append(
        PiecewiseGridSegment(
            interval=f"[{final_lo}, {final_hi}]", n_points=outer_n_points[-1]
        )
    )
    return PiecewiseLinSpacedGrid(
        segments=tuple(segments), batch_size=grid_config.n_assets_batch_size
    )


def _allocate_points_by_span(
    *, bounds: list[tuple[float, float]], total: int
) -> list[int]:
    """Split `total` points over segments proportionally to their span.

    Every segment gets at least the two points `PiecewiseLinSpacedGrid`
    requires; the remainder goes one-by-one to the segment with the
    largest gap between its proportional entitlement and its current
    allocation.
    """
    spans = [hi - lo for lo, hi in bounds]
    full_span = sum(spans)
    entitlements = [total * span / full_span for span in spans]
    alloc = [2] * len(bounds)
    for _ in range(max(0, total - 2 * len(bounds))):
        shortfalls = [e - a for e, a in zip(entitlements, alloc, strict=True)]
        alloc[shortfalls.index(max(shortfalls))] += 1
    return alloc


def _fail_if_bands_overlap(*, thresholds: list[float], half_width: float) -> None:
    for lo, hi in itertools.pairwise(thresholds):
        if hi - lo <= 2.0 * half_width:
            msg = (
                f"SSI assets-test thresholds {lo} and {hi} are closer than "
                f"the {2 * half_width}-wide smoothing band; bands must not "
                "overlap."
            )
            raise ValueError(msg)


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
    each segment. The fifth bend point is the delayed-retirement-credit
    extension above the taxable max, so the grid carries four segments.
    `n_aime_gridpoints` from `grid_config` is ignored on this path; the
    total is fixed by the PIA structure (`sum(_AIME_PIECE_N_POINTS)`).
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
            interval=f"[{kinks[2]}, {kinks[3]})", n_points=_AIME_PIECE_N_POINTS[2]
        ),
        PiecewiseGridSegment(
            interval=f"[{kinks[3]}, {kinks[4]}]", n_points=_AIME_PIECE_N_POINTS[3]
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
    """Build the regime-level state dict for a non-dead regime.

    Contains only the spec-dependent states; the states shared by every
    living regime are broadcast from the model level (`build_model_states`).
    """
    can_work = spec["canwork"] == "canwork"
    gc = grids.grid_config

    states: dict = {}
    states["health"] = DiscreteGrid(
        Health if spec["mc"] == "oamc" else HealthWithDisability,
        batch_size=gc.n_health_batch_size,
    )
    if can_work:
        states["log_ft_wage_res"] = grids.wage_res
    if can_work and spec["his"] != "tied":
        states["lagged_labor_supply"] = DiscreteGrid(
            LaggedLaborSupply,
            batch_size=gc.n_lagged_labor_supply_batch_size,
        )
    if spec["ss"] == "choose":
        states["claimed_ss"] = DiscreteGrid(
            ClaimedSS,
            batch_size=gc.n_claimed_ss_batch_size,
        )
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


def build_granular_regime_transition(
    *,
    transition_func: Callable[..., FloatND],
    target_ids: Iterable[int],
) -> dict[RegimeName, MarkovTransition]:
    """Declare the regime's reachable targets via per-target probability cells.

    Each cell evaluates the regime's probability vector and selects its
    target's entry — identical arithmetic to the coarse vector form, with the
    key set making every other regime structurally unreachable.
    """
    id_to_name = {
        int(getattr(RegimeId, name)): name for name in (*REGIME_SPECS, "dead")
    }
    declared = sorted({*(int(i) for i in target_ids), int(RegimeId.dead)})
    return {
        id_to_name[target_id]: MarkovTransition(
            _prob_of_target(transition_func=transition_func, target_id=target_id)
        )
        for target_id in declared
    }


def _prob_of_target(
    *, transition_func: Callable[..., FloatND], target_id: int
) -> Callable[..., FloatND]:
    """Select one target's probability from the regime's probability vector."""

    @functools.wraps(transition_func)
    def cell(*args: Any, **kwargs: Any) -> FloatND:
        return transition_func(*args, **kwargs)[target_id]

    return cell


# Broadcast functions the bequest DAG reads: the pref-type-indexed scalars
# resolve their per-cell value from the broadcast `pref_type` state.
_DEAD_KEEPS = frozenset(
    {"consumption_weight", "coefficient_rra", "utility_scale_factor"}
)


def build_dead_regime(*, solver: SolverName = "brute_force") -> Regime:
    """Build the terminal dead regime.

    Everything `dead` carries arrives via the model-level broadcast:

    - states: `assets` and `pref_type` survive DAG pruning (the bequest
      utility reads them); the remaining broadcast states are pruned.
    - functions: the pref-type-indexed scalars in `_DEAD_KEEPS` stay; every
      other broadcast function is masked with `None` so its unresolvable
      inputs (e.g. `pension_benefit`) don't surface as params in the dead
      template.
    - constraints: the borrowing constraint is masked — `dead` has no
      consumption action. (Under DC-EGM no constraint is broadcast, so
      there is nothing to mask.)
    - `pension_wealth` is masked explicitly: a carried state is rejected in
      terminal regimes before pruning could drop it.
    """
    function_masks = {
        name: None
        for name in build_model_functions(solver=solver)
        if name not in _DEAD_KEEPS
    }
    constraint_masks = dict.fromkeys(build_model_constraints(solver=solver))
    return Regime(
        transition=None,
        functions={"utility": preferences.bequest, **function_masks},
        constraints=constraint_masks,
        states={"pension_wealth": None},
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


def build_model_functions(*, solver: SolverName = "brute_force") -> dict:
    """Build the model-level functions broadcast into every regime.

    Contains exactly the functions that are identical across all 18 living
    regimes AND are never swapped by the ACA policy overlay. Spec-dependent
    selections (`good_health`, `leisure`, …) and overlay-swapped names
    (`medicaid_eligibility_share`, `cash_on_hand`, `primary_oop`) stay
    regime-level in `build_common_functions`. The `dead` regime masks every entry the
    bequest DAG does not read (see `build_dead_regime`). Under DC-EGM the
    solver-contract functions join the broadcast set.
    """
    functions: dict = {}
    if solver == "dcegm":
        functions |= build_dcegm_functions()
    functions["total_health_costs"] = health_insurance.total_costs
    functions["oop_costs"] = health_insurance.oop_with_medicaid
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

    # SSI/Medicaid (eligibility itself is overlay-swapped, hence regime-level)
    functions["countable_income"] = health_insurance.countable_income
    functions["ssi_eligibility_share"] = health_insurance.ssi_eligibility_share
    functions["ssi_benefit"] = health_insurance.ssi_benefit

    # Taxes
    functions["taxable_ss_benefit"] = taxes.taxable_ss_benefit
    functions["gross_income"] = taxes.gross_income
    functions["after_tax_income"] = taxes.after_tax_income
    # Every living regime carries pension wealth and the solve-phase pension
    # assets adjustment, both of which scale by the marginal income tax rate.
    functions["marginal_tax_rate"] = taxes.marginal_rate

    # HIC premium
    functions["predicted_hcc_insurer"] = health_insurance.hcc_insurer_predicted

    # Transfers
    functions["consumption_dollars_floor"] = assets_and_income.consumption_dollars_floor
    functions["transfers"] = assets_and_income.transfers
    functions["consumption_equiv"] = preferences.consumption_equiv

    return functions


def build_dcegm_functions() -> dict:
    """Build the regime functions the DC-EGM contract requires.

    Invariant across all living regimes, so they join the model-level
    broadcast; `dead` masks them like the other broadcast functions.
    """
    return {
        "resources": assets_and_income.resources,
        "savings": assets_and_income.savings,
        "inverse_marginal_utility": preferences.inverse_marginal_utility,
    }


def build_model_constraints(*, solver: SolverName = "brute_force") -> dict:
    """Build the model-level constraints broadcast into every regime.

    `dead` masks the borrowing constraint — it has no consumption action.
    Under DC-EGM there is no explicit borrowing constraint: the savings
    grid's lower bound enforces it.
    """
    if solver == "dcegm":
        return {}
    return {"borrowing_constraint": assets_and_income.borrowing_constraint}


def build_model_states(grids: Grids) -> dict:
    """Build the model-level states broadcast into every regime.

    These are the states every living regime carries with an identical grid.
    pylcm prunes them per regime by DAG reachability, so `dead` keeps only
    `assets` and `pref_type` (the bequest DAG). `spousal_income` carries the
    `distributed` flag — sharding is legal only on model-level states.
    """
    gc = grids.grid_config
    return {
        "assets": grids.assets,
        "aime": grids.aime,
        "pension_wealth": Phased(
            solve=pensions.wealth,
            simulate=grids.pension_wealth,
        ),
        "hcc_persistent": grids.hcc_persistent,
        "hcc_transitory": grids.hcc_transitory,
        "spousal_income": DiscreteGrid(
            SpousalIncome,
            batch_size=gc.n_spousal_income_batch_size,
            distributed=gc.spousal_income_distributed,
        ),
        "pref_type": grids.pref_type,
    }


def build_model_state_transitions() -> dict:
    """Build the model-level laws of motion for the broadcast states.

    Only the laws that are identical across all living regimes live here;
    `assets` and `aime` evolve spec-dependently and keep their laws in
    `build_state_transitions`. The hcc shocks are stochastic processes with
    intrinsic transitions.
    """
    return {
        "pref_type": fixed_transition("pref_type"),
        "spousal_income": MarkovTransition(labor_market.next_spousal_income),
        # Carried state: evolved only in simulate (in solve, `pension_wealth`
        # is re-imputed from AIME each period and has no transition).
        "pension_wealth": pensions.wealth_next_before_adjustment,
    }


def build_common_functions(spec: RegimeSpec) -> dict:
    """Build the regime-level functions dict for a non-dead regime.

    Contains the spec-dependent selections and the overlay-swapped names;
    everything identical across living regimes is broadcast from the model
    level (`build_model_functions`). Per-HIS modules add utility, ss_benefit,
    his, gets_medicare, hic_premium, and pension entries.
    """
    can_work = spec["canwork"] == "canwork"

    functions: dict = {}
    functions["good_health"] = (
        health.is_good_health_2 if spec["mc"] == "oamc" else health.is_good_health_3
    )
    has_buy_private = spec["his"] == "nongroup" and spec["mc"] == "nomc"
    functions["primary_oop"] = (
        health_insurance.primary_oop if has_buy_private else health_insurance.oop_costs
    )

    if can_work:
        functions["working_hours_value"] = labor_market.working_hours_value
        functions["wage"] = labor_market.wage
        functions["labor_income"] = labor_market.income
        functions["fixed_cost_of_work"] = preferences.fixed_cost_of_work

    functions["leisure"] = _select_leisure(spec)
    functions["utility"] = preferences.u_alive

    if spec["mc"] != "oamc":  # pre-65: SSDI needs dropout-adjusted PIA
        functions["ssdi_pia"] = social_security.ssdi_pia

    # SSI categorical track: `crossed_oamc_threshold` is a per-regime constant
    # fixed param;
    # `is_disabled` reads the disability health state where the regime carries
    # it (pre-65 `nomc`/`dimc`) and is constant False post-65 (`oamc`).
    functions["is_disabled"] = (
        health_insurance.is_disabled_never
        if spec["mc"] == "oamc"
        else health_insurance.is_disabled_from_health
    )
    # MAGI for the ACA Medicaid expansion track; pruned in the baseline DAG.
    functions["aca_magi"] = health_insurance.aca_magi

    # Swapped per policy variant by the ACA overlay, hence regime-level
    functions["medicaid_eligibility_share"] = (
        health_insurance.medicaid_eligibility_share
    )
    functions["premium_default"] = assets_and_income.premium_default
    functions["cash_on_hand"] = assets_and_income.cash_on_hand

    # Earnings test credit-back (only choose+canwork: has claim_ss + claimed_ss)
    if spec["ss"] == "choose" and can_work:
        functions["benefit_withheld_fraction"] = (
            social_security.benefit_withheld_fraction
        )

    return functions


def _zero_pension_accrual() -> FloatND:
    """Pension accrual for regimes without labor earnings (French & Jones 2011).

    Forced-out regimes have no labor supply, so no earnings accrue to pension
    wealth; only the annuity decumulation in
    `pensions.wealth_next_before_adjustment` moves it.
    """
    return jnp.asarray(0.0)


def _zero_pension_assets_adjustment() -> FloatND:
    """Pension assets adjustment during simulate: identically zero.

    Solve corrects next-period assets for the gap between the AIME-imputed
    pension wealth and its accrual-evolved value. Simulate carries the true
    pension wealth as a state, so there is no imputation gap to reconcile.
    """
    return jnp.asarray(0.0)


def build_pension_functions(spec: RegimeSpec) -> dict:
    """Build the pension DAG functions shared by every living regime.

    `pension_wealth` itself is a carried state declared in `build_states`
    (its law of motion lives in `build_state_transitions`): imputed from AIME
    in solve, carried as the agent's true wealth in simulate. These functions
    complete the French & Jones (2011) pension block around it:

    - `full_benefit` (eq. D.2) and the carried state's solve variant
      `pensions.wealth` (eq. D.3) impute pension wealth from PIA.
    - `pension_benefit` (eq. D.4) draws the received benefit from whichever
      pension wealth the phase supplies — imputed in solve, true in simulate.
    - `pension_accrual` is the labour-earnings accrual where the agent can
      work and zero otherwise.
    - `pension_wealth_next_before_adjustment`, `target_his`, and
      `imputed_pension_wealth_next_period` feed the solve-phase
      `pension_assets_adjustment`, which reconciles the accrual-evolved
      pension with next period's AIME imputation; in simulate the adjustment
      is zero because the true wealth is carried directly. The imputation is
      the `medicaid_eligibility_share`-weighted mixture of two legs — the
      deterministic `target_his` lookup and the nongroup (Medicaid-target)
      lookup — matching the probability mixture in the regime transition.
    """
    can_work = spec["canwork"] == "canwork"

    functions: dict = {}
    functions["full_benefit"] = pensions.full_benefit
    functions["pension_benefit"] = pensions.benefit
    functions["pension_accrual"] = (
        pensions.accrual if can_work else _zero_pension_accrual
    )
    functions["pension_wealth_next_before_adjustment"] = (
        pensions.wealth_next_before_adjustment
    )
    functions["target_his"] = (
        health_insurance.target_his
        if can_work
        else health_insurance.target_his_forcedout
    )
    functions["pia_unadjusted_next_period"] = social_security.pia_unadjusted_next_period
    functions["imputed_pension_wealth_next_period_no_medicaid"] = (
        pensions.imputed_pension_wealth_next_period_no_medicaid
    )
    functions["imputed_pension_wealth_next_period_medicaid"] = (
        pensions.imputed_pension_wealth_next_period_medicaid
    )
    functions["imputed_pension_wealth_next_period"] = (
        pensions.imputed_pension_wealth_next_period
    )
    functions["pension_assets_adjustment"] = Phased(
        solve=pensions.assets_adjustment,
        simulate=_zero_pension_assets_adjustment,
    )
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


def build_state_transitions(
    spec: RegimeSpec, *, solver: SolverName = "brute_force"
) -> dict:
    """Build the regime-level state transitions dict for a non-dead regime.

    Contains only the spec-dependent laws; uniform laws for the broadcast
    states live at the model level (`build_model_state_transitions`).
    `assets` and `aime` are broadcast states whose laws differ per spec,
    so the laws stay here. Under DC-EGM the assets laws take their
    post-decision (savings) form.
    """
    transitions: dict = {}
    transitions["assets"] = _build_per_target_regime_assets(spec, solver=solver)
    transitions["health"] = _build_per_target_regime_health(spec)
    claimed_ss_transition = _build_per_target_regime_claimed_ss(spec)
    if claimed_ss_transition:
        transitions["claimed_ss"] = claimed_ss_transition
    lagged_labor_supply_transition = _build_per_target_regime_lagged_labor_supply(spec)
    if lagged_labor_supply_transition:
        transitions["lagged_labor_supply"] = lagged_labor_supply_transition
    transitions["aime"] = _select_aime_law(spec)
    return transitions


def _select_aime_law(spec: RegimeSpec) -> Callable[..., FloatND]:
    """Select the AIME law of motion for a non-dead regime.

    The claim-age actuarial bake applies only where the agent chooses when to
    claim (`ss=choose`), so only those regimes carry the `claim_ss`/`claimed_ss`
    inputs. `ss=inelig` (cannot claim) and `ss=forced` (claims by rule) use the
    plain-accrual variant: no claim adjustment, no claim inputs. A forced
    claimant who claimed early carries the reduction in from the choose regime;
    plain accrual preserves it.

    - post-65 (`oamc`), `choose` → `next_aime` (claim-adjusted)
    - post-65 (`oamc`), `forced` → `next_aime_plain`
    - pre-65 (`nomc`/`dimc`), `choose` → `next_aime_disabled` (claim-adjusted)
    - pre-65 (`nomc`/`dimc`), `inelig` → `next_aime_disabled_plain`
    """
    is_choose = spec["ss"] == "choose"
    if spec["mc"] == "oamc":
        return (
            social_security.next_aime if is_choose else social_security.next_aime_plain
        )
    return (
        social_security.next_aime_disabled
        if is_choose
        else social_security.next_aime_disabled_plain
    )


def _build_per_target_regime_assets(
    spec: RegimeSpec, *, solver: SolverName = "brute_force"
) -> dict[RegimeName, Callable[..., FloatND]]:
    """Build per-target assets transitions.

    The `dead` target uses `next_assets_when_dead` (no
    `pension_assets_adjustment`), so the dead per-target DAG does not
    pull in the `next_aime`-dependent imputation chain — `dead` has no
    `aime` state and pylcm cannot resolve `next_aime` there. Non-dead
    targets use the full `next_assets` with the pension correction.
    Under DC-EGM both laws take their post-decision (savings) form.
    """
    if solver == "dcegm":
        living_law = assets_and_income.next_assets_from_savings
        dead_law = assets_and_income.next_assets_when_dead_from_savings
    else:
        living_law = assets_and_income.next_assets
        dead_law = assets_and_income.next_assets_when_dead

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
        result[target_name] = living_law

    result["dead"] = dead_law
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
