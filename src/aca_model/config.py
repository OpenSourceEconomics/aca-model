"""Configuration for the aca_model package."""

from dataclasses import dataclass
from pathlib import Path

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
    # outer state loop. `1` shrinks the per-period Q intermediate by that
    # axis's cardinality on hosts where the unsplayed kernel doesn't fit;
    # `0` lets a single kernel span the axis.
    n_assets_batch_size: int = 0
    n_aime_batch_size: int = 0
    # Sharding flags for discrete state grids. pylcm distributes the
    # grid across available devices when the flag is `True`. Sharding
    # is only supported on discrete state grids; continuous axes
    # (`assets`, `aime`, `wage_res`, `hcc_*`) compile to an all-gather
    # of the full V-array per device and are rejected at grid
    # construction. Mutually exclusive with `batch_size>0` on the same
    # axis (pylcm rejects the combination). `spousal_income_distributed`
    # routes through `baseline/regimes/_common.py:build_states` to its
    # inline-built `DiscreteGrid(...)` call.
    pref_type_distributed: bool = False
    spousal_income_distributed: bool = False
    # `batch_size` on the inline-constructed discrete state grids —
    # health, spousal_income, lagged_labor_supply, claimed_ss. These
    # are read in `build_states` via `grids.grid_config`. Setting any
    # of them to `1` puts that axis in a Python-level outer loop within
    # the discrete-states block of the productmap
    # (`_ordered_state_action_names`), shrinking the per-call Q
    # intermediate by that axis's cardinality at the cost of one extra
    # lax.scan layer. Defaults to `0`; production overrides set to `1`
    # to compound the splay across the unsharded discretes.
    n_health_batch_size: int = 0
    n_spousal_income_batch_size: int = 0
    n_lagged_labor_supply_batch_size: int = 0
    n_claimed_ss_batch_size: int = 0
    # `batch_size` on the `pref_type` discrete grid: chunked vmap stride
    # for the pref-type axis during solve. `1` (one pref-type per Python
    # dispatch) shrinks the per-period Q intermediate by `n_pref_types`
    # at the cost of an outer Python loop; `0` lets a single kernel span
    # all pref-types. Defaults to `0` — the production overrides set it
    # to `1` on hardware where the unsplayed kernel doesn't fit.
    n_pref_type_batch_size: int = 0
    # `batch_size` on the `wage_res` stochastic shock process: chunked
    # productmap stride along the wage-residual stoch axis inside Q_and_F.
    # `1` shrinks the per-target Q intermediate by `n_wage_res_gridpoints`
    # at the cost of an inner Python loop; `0` lets the productmap span
    # the full axis. Defaults to `0` — production overrides set it to `1`
    # on hardware where the ACA-overlay per-cell DAG blows the kernel's
    # compile-time working set past device HBM.
    n_wage_res_batch_size: int = 0


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
