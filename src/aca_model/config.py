"""Configuration for the aca_model package."""

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
    # axis (pylcm rejects the combination). `spousal_income` is not among
    # them: it lives on the `married` regimes, and sharding is legal only
    # on model-level states.
    pref_type_distributed: bool = False
    # `batch_size` on the inline-constructed discrete state grids —
    # health, lagged_labor_supply, claimed_ss. These
    # are read in `build_states` via `grids.grid_config`. Setting any
    # of them to `1` puts that axis in a Python-level outer loop within
    # the discrete-states block of the productmap
    # (`_ordered_state_action_names`), shrinking the per-call Q
    # intermediate by that axis's cardinality at the cost of one extra
    # lax.scan layer. Defaults to `0`; production overrides set to `1`
    # to compound the splay across the unsharded discretes.
    n_health_batch_size: int = 0
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
    # Number of nodes on the NBEGM savings grid (the post-decision endogenous
    # grid), cubically clustered toward the borrowing constraint. Drives the
    # padded-grid dimension the egm_step kernel carries, so it scales both the
    # rolling carry and the gather mesh roughly linearly — the dominant lever on
    # EGM device memory. Only consulted under `solver="nbegm"`.
    n_savings_gridpoints: int = 200
    # `batch_size` on the NBEGM savings grid (the post-decision endogenous
    # grid). pylcm splays the per-savings-node continuation into `lax.map`
    # blocks of this size, shrinking the binding `egm_step` working buffer by
    # roughly the block factor while the upper envelope still runs on the full
    # gathered grid (value function unchanged). `0` keeps the whole grid in one
    # kernel. Only consulted under `solver="nbegm"`.
    n_savings_batch_size: int = 0
    # `batch_size` on the EGM child stochastic-node expectation (the process-state
    # mesh the egm_step kernel sums over). pylcm splays that expectation into
    # `lax.map` blocks of this size, collapsing the gather buffer's node axis by
    # roughly the block factor (value function unchanged) — the grid-independent
    # lever for the residual egm_step transient that savings-batching leaves
    # behind. `0` evaluates the whole mesh in one kernel. Only consulted under
    # `solver="nbegm"`.
    n_stochastic_node_batch_size: int = 0
    # Block size for splaying the NBEGM continuation's child stochastic-node
    # expectation (health, health-cost shocks, the wage residual). `0` reads the
    # whole joint node mesh in one pass — fast, but its peak intermediate scales
    # with the full ride-along × node × child-grid product. A positive value loops
    # the mesh in blocks of that size, trading runtime for a much smaller peak; `1`
    # (one node at a time) is the memory-minimal setting for a CPU validation grid.
    # Only consulted under `solver="nbegm"`.
    n_nbegm_stochastic_node_batch_size: int = 0
    # Streams the per-interval upper envelope over candidate-segment blocks of this
    # size instead of materialising the full (query x candidate) bracket matrix per
    # ride cell. `0` keeps the one-shot dense envelope; the result is identical
    # either way — the knob trades peak device memory against a sequential scan.
    # Only consulted under `solver="nbegm"`.
    n_nbegm_envelope_segment_block_size: int = 0
    # Which arithmetic decides ownership in the merged upper envelope:
    # - "certified" (default) — candidates are compared in double-double precision
    #   and no winner is published where none is separated, so the reported owner is
    #   one the arithmetic could prove. Ordering survives the cancellation a nearly
    #   tied crossing produces, at roughly an order of magnitude more arithmetic per
    #   read — and the envelope read is the dominant per-cell cost of a case-piece
    #   solve, so this is the setting that decides the solver's runtime.
    # - "ordinary" — each candidate is read in the working format and the largest
    #   owns the query. Adequate wherever candidate values are separated by much
    #   more than the format's resolution at their own magnitude.
    # Incompatible with a positive `n_nbegm_envelope_segment_block_size`, which
    # selects a blocked scan that carries the certified arithmetic only.
    # Only consulted under `solver="nbegm"`.
    nbegm_envelope_arithmetic: Literal["certified", "ordinary"] = "certified"
    # Streams both NBEGM ride-along cores (continuation fan-out and envelope
    # solve) over ride-cell blocks of this size instead of vmapping the whole
    # flattened ride mesh at once — the dominant peak-memory term at production
    # mesh sizes. `0` keeps the whole-mesh vmap; the result is identical either
    # way. Only consulted under `solver="nbegm"`.
    #
    # Backend-dependent tuning at production ride-mesh sizes (under both cliff-read
    # modes). On GPU the whole-mesh vmap stays within a few GiB, so `0` is fine. The
    # CPU XLA backend does not fuse the fan-out and materialises the whole flattened
    # ride mesh at once — a production-grid solve then needs hundreds of GiB even at
    # a small asset grid (the blow-up rides the aime/shock/health mesh, not assets).
    # Set this to a positive block (e.g. 64) for a CPU solve; it bounds the peak to
    # the GPU's few-GiB footprint at the cost of serialising the mesh into a
    # `lax.map` scan.
    n_nbegm_cell_block_size: int = 0
    # How NBEGM parents read the child value's institutional cliffs:
    # - "one_sided" (default) — carry rows hold each cliff preimage as a duplicated
    #   abscissa with exact one-sided limits; reads never average across a cliff,
    #   but publishing the topology gates the stochastic-dim fold off (slower).
    # - "bridged" — plain carry rows; interpolation may bridge a cliff like any
    #   finite-grid solver, and the fold stays available. The fast setting for
    #   inner estimation loops, polished afterwards under "one_sided".
    # Only consulted under `solver="nbegm"`.
    nbegm_jump_read: Literal["one_sided", "bridged"] = "one_sided"
    # Block size for streaming the discrete-action branch axis in both NBEGM
    # ride-along cores (one continuation row / one continuous subproblem per live
    # labor level). `0` runs the whole branch axis in one vectorized pass; a positive
    # value scans it in blocks of that many branches (identical result, per-branch
    # intermediates bounded by one block). Only consulted under `solver="nbegm"`.
    n_nbegm_branch_batch_size: int = 0
    # `labor_supply` enters `countable_income`, which carries the SSI income test, so
    # each labor level puts that breakpoint at a different liquid level. The one-sided
    # read publishes its cliff limits on a single query grid shared across branches,
    # which the two cannot both satisfy, so a regime carrying `labor_supply` builds
    # only under `nbegm_jump_read="bridged"`.


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
