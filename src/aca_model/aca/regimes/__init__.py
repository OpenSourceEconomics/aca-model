"""ACA regime construction: applies ACA overrides to baseline regimes."""

import dataclasses

from lcm import Regime

from aca_model.aca.health_insurance import PolicyVariant
from aca_model.aca.regimes._overrides import apply_aca_overrides
from aca_model.baseline.regimes import build_all_regimes as baseline_build_all_regimes
from aca_model.baseline.regimes._common import REGIME_SPECS
from aca_model.config import GRID_CONFIG, GridConfig


def build_all_regimes(
    policy: PolicyVariant, grid_config: GridConfig = GRID_CONFIG
) -> dict[str, Regime]:
    """Build all 19 regimes with ACA policy overrides."""
    regimes = baseline_build_all_regimes(grid_config)
    result = {}
    for name, regime in regimes.items():
        if name == "dead":
            result[name] = regime
            continue
        spec = REGIME_SPECS[name]
        functions = dict(regime.functions)
        apply_aca_overrides(functions, spec, policy)
        result[name] = dataclasses.replace(regime, functions=functions)
    return result
