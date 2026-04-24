"""Baseline regime construction package.

Builds the 18-regime model decomposition programmatically from four
structural dimensions: HIS x Medicare x SS x Work.

Pre-65 regimes (mc=nomc/dimc) use HealthWithDisability (3-state).
Post-65 regimes (mc=oamc) use Health (2-state).

Each HIS type (retiree, tied, nongroup) has its own transition logic in a
dedicated submodule. Shared definitions and builders live in _common.
"""

from collections.abc import Mapping
from typing import Any

from lcm import DiscreteGrid, Regime

from aca_model.baseline.regimes import _nongroup as nongroup
from aca_model.baseline.regimes import _retiree as retiree
from aca_model.baseline.regimes import _tied as tied
from aca_model.baseline.regimes._common import (
    REGIME_SPECS,
    Grids,
    RegimeId,
    build_dead_regime,
    build_grids,
)
from aca_model.config import GRID_CONFIG, GridConfig

__all__ = [
    "REGIME_SPECS",
    "RegimeId",
    "build_all_regimes",
    "build_regime",
    "nongroup",
    "retiree",
    "tied",
]

_HIS_BUILDERS = {
    "retiree": retiree.build_regime,
    "tied": tied.build_regime,
    "nongroup": nongroup.build_regime,
}


def build_regime(name: str, grids: Grids) -> Regime:
    """Build a single baseline Regime object for the given regime name."""
    if name == "dead":
        return build_dead_regime(grids)

    spec = REGIME_SPECS[name]
    builder = _HIS_BUILDERS.get(spec["his"])
    if builder is None:
        msg = f"Unknown HIS type: {spec['his']}"
        raise ValueError(msg)
    return builder(name, grids)


def build_all_regimes(
    grid_config: GridConfig = GRID_CONFIG,
    *,
    fixed_params: Mapping[str, Any] | None = None,
    wage_params: Mapping[str, Any] | None = None,
    pref_type_grid: DiscreteGrid | None = None,
) -> dict[str, Regime]:
    """Build all 19 baseline regimes (18 non-terminal + dead).

    `fixed_params` is forwarded to `build_grids` for data-driven AIME
    breakpoints; `wage_params` for the data-driven assets floor;
    either being `None` keeps the corresponding static fallback.
    `pref_type_grid` lets callers inject a compact or partition-lifted
    `DiscreteGrid(...)` (e.g. the benchmark uses a 2-type
    `BenchmarkPrefType` with `DispatchStrategy.PARTITION_SCAN`).
    """
    grids = build_grids(
        grid_config,
        fixed_params=fixed_params,
        wage_params=wage_params,
        pref_type_grid=pref_type_grid,
    )
    regimes = {}
    for name in REGIME_SPECS:
        regimes[name] = build_regime(name, grids)
    regimes["dead"] = build_dead_regime(grids)
    return regimes
