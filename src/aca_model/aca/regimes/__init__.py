"""ACA regime construction: applies ACA overrides to baseline regimes."""

import dataclasses
import functools
import inspect
from collections.abc import Mapping
from typing import Any

from lcm import DiscreteGrid, Regime
from lcm.typing import UserParams

from aca_model.aca.health_insurance import PolicyVariant
from aca_model.aca.regimes._overrides import apply_aca_overrides
from aca_model.baseline.health_insurance import BuyPrivate
from aca_model.baseline.regimes import SolverName
from aca_model.baseline.regimes import build_all_regimes as baseline_build_all_regimes
from aca_model.baseline.regimes._common import REGIME_SPECS
from aca_model.config import GridConfig


def build_all_regimes(
    *,
    policy: PolicyVariant,
    grid_config: GridConfig,
    fixed_params: UserParams,
    wage_params: Mapping[str, Any],
    pref_type_grid: DiscreteGrid,
    solver: SolverName = "brute_force",
) -> dict[str, Regime]:
    """Build all 37 regimes with ACA policy overrides."""
    regimes = baseline_build_all_regimes(
        grid_config=grid_config,
        fixed_params=fixed_params,
        wage_params=wage_params,
        pref_type_grid=pref_type_grid,
        solver=solver,
    )
    result = {}
    for name, regime in regimes.items():
        if name == "dead":
            result[name] = regime
            continue
        spec = REGIME_SPECS[name]
        functions = dict(regime.functions)
        apply_aca_overrides(functions, spec, policy)
        if "buy_private" not in regime.actions:
            _bind_fixed_buy_private(functions)
        result[name] = dataclasses.replace(regime, functions=functions)
    return result


def _bind_fixed_buy_private(functions: dict) -> None:
    """Bind `buy_private` to its fixed level in every function reading it.

    A regime that fixes `buy_private` (the NBEGM slice) carries no
    `buy_private` action, so an ACA-swapped function reading it would surface
    the argument as a free parameter in the params template. Binding at the
    swap site keeps the fixed-level semantics of the baseline builder for
    every consumer, present and future.
    """
    for function_name, func in functions.items():
        if callable(func) and "buy_private" in inspect.signature(func).parameters:
            functions[function_name] = functools.partial(
                func, buy_private=BuyPrivate.yes
            )
