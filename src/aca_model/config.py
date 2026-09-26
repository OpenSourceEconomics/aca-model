"""Economic grids and numerical solver choices for the aca_model package."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SRC = Path(__file__).parent.resolve()
ROOT = SRC.parents[1]
BLD = ROOT / "bld"


@dataclass(frozen=True)
class ModelConfig:
    start_age: int = 51
    end_age: int = 96
    ss_early_age: int = 62
    ss_forced_age: int = 70
    work_forced_out_age: int = 72
    medicare_age: int = 65


@dataclass(frozen=True)
class GridConfig:
    """Economic resolution and numerical choices, independent of device execution."""

    n_assets_gridpoints: int = 24
    n_aime_gridpoints: int = 12
    n_consumption_dollars_gridpoints: int = 70
    n_wage_res_gridpoints: int = 5
    n_hcc_persistent_gridpoints: int = 3
    n_hcc_transitory_gridpoints: int = 5
    # The post-decision savings grid is cubically clustered toward the borrowing
    # constraint. Its node count controls resolution under DC-EGM and NB-EGM.
    n_savings_gridpoints: int = 200
    # Arithmetic used to compare upper-envelope candidates:
    # - "certified": double-double comparisons publish only separated winners.
    # - "ordinary": comparisons use the working floating-point format.
    nbegm_envelope_arithmetic: Literal["certified", "ordinary"] = "certified"
    # How NB-EGM parents read institutional cliffs in the child value:
    # - "one_sided": duplicated abscissae preserve one-sided limits.
    # - "bridged": interpolation may bridge finite-grid discontinuities.
    # Live labor choices require "bridged" because branch-dependent income
    # thresholds cannot share one query grid of one-sided cliff limits.
    nbegm_jump_read: Literal["one_sided", "bridged"] = "one_sided"


MODEL_CONFIG = ModelConfig()
GRID_CONFIG = GridConfig()

BENCHMARK_GRID_CONFIG = GridConfig(
    n_assets_gridpoints=3,
    n_aime_gridpoints=3,
    n_consumption_dollars_gridpoints=5,
    n_wage_res_gridpoints=3,
    n_hcc_persistent_gridpoints=3,
    n_hcc_transitory_gridpoints=3,
)
