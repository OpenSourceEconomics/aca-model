"""Configuration for the aca_model package."""

from dataclasses import dataclass
from pathlib import Path

import plotly.io as pio
from pytask import DataCatalog

SRC = Path(__file__).parent.resolve()
ROOT = SRC.parents[1]
BLD = ROOT / "bld"

data_catalog = DataCatalog()

pio.templates.default = "plotly_dark+presentation"


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
    n_assets_gridpoints: int = 24
    n_aime_gridpoints: int = 12
    n_consumption_gridpoints: int = 70
    n_wage_res_gridpoints: int = 5
    n_hcc_persistent_gridpoints: int = 3
    n_hcc_transitory_gridpoints: int = 5
    # `batch_size` on the assets / AIME grids: chunked vmap stride for the
    # outer state loop. Both partition the per-period Q intermediate so it
    # fits in V100 16 GB once we splay across `pref_type`. Set to 0 in
    # BENCHMARK_GRID_CONFIG to skip the Python-loop overhead.
    n_assets_batch_size: int = 1
    n_aime_batch_size: int = 1


MODEL_CONFIG = ModelConfig()
GRID_CONFIG = GridConfig()

BENCHMARK_GRID_CONFIG = GridConfig(
    n_assets_gridpoints=3,
    n_aime_gridpoints=3,
    n_consumption_gridpoints=5,
    n_wage_res_gridpoints=3,
    n_hcc_persistent_gridpoints=3,
    n_hcc_transitory_gridpoints=3,
    n_assets_batch_size=0,
    n_aime_batch_size=0,
)
