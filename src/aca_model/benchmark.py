"""Benchmark-sized baseline model with frozen parameters.

Exports a compact, fully self-contained setup that exercises the full
18-regime DAG with `BENCHMARK_GRID_CONFIG` grids (tiny n_points per
continuous state). Use for ASV benchmarks and fast end-to-end
integration tests without requiring the aca-data pipeline.

The benchmark substitutes a 2-type `BenchmarkPrefType` for the
production 3-type `PrefType`, which saves ~33% of the compile +
execution volume over all 18 regimes. By default the pref_type axis
is handled via pylcm's fused-vmap dispatch (no `DispatchStrategy`
imported — this module stays compatible with pylcm versions that
pre-date the enum). Callers that want partition-lifted dispatch
(`PARTITION_SCAN` / `PARTITION_VMAP`) construct the grid themselves
and pass it via `pref_type_grid`.

Parameters (`fixed_params` + `params`) are a committed stub fixture
packaged alongside the module at
`src/aca_model/_benchmark_data/benchmark_params.pkl` — aggregate-level
values (policy schedules, transition probabilities, fitted
coefficients) with no runtime dependency on aca-data or any data-prep
package. The pref-type-indexed entries in `params` are truncated to
two rows on load to match `BenchmarkPrefType`.

Initial conditions are drawn randomly per call — assets/aime/wage_res
from their grid ranges, discrete states from their categories, regimes
sampled from the five regimes active at `start_age=51`.
"""

from dataclasses import fields
from pathlib import Path
from typing import Any

import cloudpickle
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array
from lcm import DiscreteGrid, Model

from aca_model.agent.health import GoodHealth
from aca_model.agent.labor_market import IsMarried
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline.health_insurance import HealthInsuranceState
from aca_model.baseline.model import create_model
from aca_model.config import BENCHMARK_GRID_CONFIG
from aca_model.consumption_grid import inject_consumption_points

_PARAMS_FILE = (
    Path(__file__).resolve().parent / "_benchmark_data" / "benchmark_params.pkl"
)

_N_BENCHMARK_PREF_TYPES = len(fields(BenchmarkPrefType))

_DERIVED_CATEGORICALS = {
    "good_health": DiscreteGrid(GoodHealth),
    "is_married": DiscreteGrid(IsMarried),
    "his": DiscreteGrid(HealthInsuranceState),
    "pref_type": DiscreteGrid(BenchmarkPrefType),
}

# Five regimes active at start_age=51 (inelig + canwork). All have
# HealthWithDisability (3-state); four have lagged_labor_supply (tied
# does not — it's implied by the regime).
_INITIAL_REGIMES = (
    "retiree_nomc_inelig_canwork",
    "tied_nomc_inelig_canwork",
    "nongroup_nomc_inelig_canwork",
    "retiree_dimc_inelig_canwork",
    "nongroup_dimc_inelig_canwork",
)


def create_benchmark_model(
    *,
    n_subjects: int,
    pref_type_grid: DiscreteGrid | None = None,
) -> Model:
    """Create the aca baseline with `BENCHMARK_GRID_CONFIG` and frozen fixed_params.

    The benchmark uses a 2-type `BenchmarkPrefType`. No `batch_size != 0`
    on any grid (continuous grids inherit
    `BENCHMARK_GRID_CONFIG.n_assets_batch_size = 0` and
    `n_aime_batch_size = 0`).

    Args:
        pref_type_grid: Override for the pref_type grid. Default is a plain
            `DiscreteGrid(BenchmarkPrefType)` (fused vmap). Pass
            `DiscreteGrid(BenchmarkPrefType, dispatch=DispatchStrategy.PARTITION_SCAN)`
            (or `PARTITION_VMAP`) to get the partition-lifted kernel — the
            recommended production setting for aca-model at scale, but only
            supported on pylcm versions that expose `DispatchStrategy`.
        n_subjects: Forwarded to `lcm.Model(n_subjects=...)`. When set, the
            first matching `simulate(...)` call AOT-compiles all simulate
            functions for that batch shape.
    """
    if pref_type_grid is None:
        pref_type_grid = DiscreteGrid(BenchmarkPrefType)
    fixed_params, _ = get_benchmark_params()
    return create_model(
        grid_config=BENCHMARK_GRID_CONFIG,
        fixed_params=fixed_params,
        derived_categoricals=_DERIVED_CATEGORICALS,
        pref_type_grid=pref_type_grid,
        n_subjects=n_subjects,
    )


def get_benchmark_params(
    *, model: Model | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the frozen `(fixed_params, params)` snapshot.

    Pref-type-indexed `pd.Series` in `params` are truncated to
    `_N_BENCHMARK_PREF_TYPES` rows so they line up with
    `BenchmarkPrefType`'s categories.

    When `model` is provided, consumption gridpoints are injected into
    `params` for each regime that declares `consumption` as an
    `IrregSpacedGrid` with runtime-supplied points. The lower bound is
    read from `params["consumption_floor"]`.
    """
    with _PARAMS_FILE.open("rb") as fh:
        data = cloudpickle.load(fh)
    fixed_params = data["fixed_params"]
    params = _truncate_pref_type_indexed(data["params"])
    if model is not None:
        params = inject_consumption_points(params=params, model=model)
    return fixed_params, params


def _truncate_pref_type_indexed(params: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `params` with pref_type-indexed Series cut to 2 rows.

    A Series is pref_type-indexed when its index labels start with
    `"type_"`. The first `_N_BENCHMARK_PREF_TYPES` rows are kept so the
    Series aligns with `BenchmarkPrefType.type_0`, `type_1`, ...
    """
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, pd.Series) and all(
            str(label).startswith("type_") for label in value.index
        ):
            out[key] = value.iloc[:_N_BENCHMARK_PREF_TYPES]
        else:
            out[key] = value
    return out


def get_benchmark_initial_conditions(
    *, model: Model, n_subjects: int = 100, seed: int = 42
) -> dict[str, Array]:
    """Draw random feasible initial conditions across five age-51 regimes.

    Every subject gets a random regime from `_INITIAL_REGIMES`; continuous
    states are drawn uniformly over the regime's grid range, discrete states
    uniformly over categories. States absent from a subject's regime are
    filled with 0 (pylcm ignores them for that regime).
    """
    rng = np.random.default_rng(seed)
    regime_ids = tuple(model.regime_names_to_ids[n] for n in _INITIAL_REGIMES)
    regime = rng.choice(regime_ids, size=n_subjects).astype(np.int32)

    # Grid ranges come from any of the five regimes (shared structure).
    # Use to_jax() so the helper handles both LinSpacedGrid and
    # PiecewiseLinSpacedGrid (the latter has no `.start` / `.stop`).
    ref_regime = model.regimes[_INITIAL_REGIMES[0]]
    grids = ref_regime.states
    assets_pts = np.asarray(grids["assets"].to_jax())
    aime_pts = np.asarray(grids["aime"].to_jax())
    assets_lo, assets_hi = float(assets_pts.min()), float(assets_pts.max())
    aime_lo, aime_hi = float(aime_pts.min()), float(aime_pts.max())
    hcc_p_pts = np.asarray(grids["hcc_persistent"].to_jax())
    hcc_t_pts = np.asarray(grids["hcc_transitory"].to_jax())
    wage_res_pts = np.asarray(grids["log_ft_wage_res"].to_jax())

    # For lagged_labor_supply: 0 for tied rows (regime lacks the state),
    # random 0/1 for the others.
    tied_id = model.regime_names_to_ids["tied_nomc_inelig_canwork"]
    lls = np.where(regime == tied_id, 0, rng.integers(0, 2, size=n_subjects)).astype(
        np.int32
    )

    return {
        "regime": jnp.asarray(regime),
        "age": jnp.full(n_subjects, 51.0),
        "assets": jnp.asarray(rng.uniform(assets_lo, assets_hi, n_subjects)),
        "aime": jnp.asarray(rng.uniform(aime_lo, aime_hi, n_subjects)),
        "health": jnp.asarray(rng.integers(0, 3, n_subjects).astype(np.int32)),
        "hcc_persistent": jnp.asarray(rng.choice(hcc_p_pts, n_subjects)),
        "hcc_transitory": jnp.asarray(rng.choice(hcc_t_pts, n_subjects)),
        "spousal_income": jnp.asarray(rng.integers(0, 3, n_subjects).astype(np.int32)),
        "pref_type": jnp.asarray(
            rng.integers(0, _N_BENCHMARK_PREF_TYPES, n_subjects).astype(np.int32)
        ),
        "log_ft_wage_res": jnp.asarray(rng.choice(wage_res_pts, n_subjects)),
        "lagged_labor_supply": jnp.asarray(lls),
        # claimed_ss is required by non-initial regimes (ss="choose");
        # everyone starts as not-yet-claimed.
        "claimed_ss": jnp.zeros(n_subjects, dtype=jnp.int32),
    }
