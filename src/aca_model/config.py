"""Configuration for the aca_model package."""

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

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
    n_assets_gridpoints: int = 24
    n_aime_gridpoints: int = 12
    n_consumption_dollars_gridpoints: int = 70
    n_wage_res_gridpoints: int = 5
    n_hcc_persistent_gridpoints: int = 3
    n_hcc_transitory_gridpoints: int = 5
    # `batch_size` on the assets / AIME grids: chunked vmap stride for the
    # outer state loop. The assets axis is hardcoded `distributed=True`
    # in regimes, so `n_assets_batch_size` must stay `0` — `>0 + distributed`
    # is rejected by pylcm's grid-init guard. `n_aime_batch_size` is free
    # to splay; `1` shrinks the per-period Q intermediate by 12x on hosts
    # where the unsplayed kernel doesn't fit.
    n_assets_batch_size: int = 0
    n_aime_batch_size: int = 1
    # `batch_size` on the `pref_type` discrete grid: chunked vmap stride
    # for the pref-type axis during solve. `1` (one pref-type per Python
    # dispatch) shrinks the per-period Q intermediate by `n_pref_types`
    # at the cost of an outer Python loop; `0` lets a single kernel span
    # all pref-types. Defaults to `0` — the production overrides set it
    # to `1` on hardware where the unsplayed kernel doesn't fit.
    n_pref_type_batch_size: int = 0
    # Per-device chunk size for the simulate-side per-subject dispatch,
    # keyed by `log_level`. Empty → 0 (no chunking) for every level.
    # `log_level="off"` skips `validate_V` and its forced host-sync, which
    # lets XLA pipeline across periods and reuse scratch — affordable
    # chunk size grows. Use `get_subjects_batch_size(log_level)`.
    subjects_batch_size_by_log_level: MappingProxyType[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def get_subjects_batch_size(self, log_level: str) -> int:
        """Return the per-device simulate chunk size for `log_level`.

        Returns 0 (no chunking) when this `GridConfig` defines no entry for
        the given log level.
        """
        return self.subjects_batch_size_by_log_level.get(log_level, 0)


MODEL_CONFIG = ModelConfig()
GRID_CONFIG = GridConfig()

BENCHMARK_GRID_CONFIG = GridConfig(
    n_assets_gridpoints=3,
    n_aime_gridpoints=3,
    n_consumption_dollars_gridpoints=5,
    n_wage_res_gridpoints=3,
    n_hcc_persistent_gridpoints=3,
    n_hcc_transitory_gridpoints=3,
    n_assets_batch_size=0,
    n_aime_batch_size=0,
)
