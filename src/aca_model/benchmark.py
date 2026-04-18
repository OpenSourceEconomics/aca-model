"""Benchmark-sized baseline model with frozen parameters.

Exports a compact, fully self-contained setup that exercises the full
18-regime DAG with `BENCHMARK_GRID_CONFIG` grids (tiny n_points per
continuous state). Use for ASV benchmarks and fast end-to-end
integration tests without requiring the aca-data pipeline.

Parameters (`fixed_params` + `params`) are a frozen snapshot of one
production run, shipped as a pickle in
`aca-model/tests/data/benchmark_params.pkl`. Regenerate via
`scripts/generate_benchmark_params.py` after changes to parameter
assembly.

Initial conditions are drawn randomly per call — assets/aime/wage_res
from their grid ranges, discrete states from their categories, regimes
sampled from the five regimes active at `start_age=51`.
"""

from pathlib import Path
from typing import Any

import cloudpickle
import jax.numpy as jnp
import numpy as np
from jax import Array
from lcm import DiscreteGrid, Model

from aca_model.agent.health import GoodHealth
from aca_model.agent.labor_market import IsMarried
from aca_model.agent.preferences import PrefType
from aca_model.baseline.health_insurance import HealthInsuranceState
from aca_model.baseline.model import create_model
from aca_model.config import BENCHMARK_GRID_CONFIG

_PARAMS_FILE = (
    Path(__file__).resolve().parents[2] / "tests" / "data" / "benchmark_params.pkl"
)

_DERIVED_CATEGORICALS = {
    "good_health": DiscreteGrid(GoodHealth),
    "is_married": DiscreteGrid(IsMarried),
    "his": DiscreteGrid(HealthInsuranceState),
    "pref_type": DiscreteGrid(PrefType),
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


def create_benchmark_model() -> Model:
    """Create the aca baseline with `BENCHMARK_GRID_CONFIG` and frozen fixed_params."""
    fixed_params, _ = get_benchmark_params()
    return create_model(
        grid_config=BENCHMARK_GRID_CONFIG,
        fixed_params=fixed_params,
        derived_categoricals=_DERIVED_CATEGORICALS,
    )


def get_benchmark_params() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the frozen `(fixed_params, params)` snapshot."""
    with _PARAMS_FILE.open("rb") as fh:
        data = cloudpickle.load(fh)
    return data["fixed_params"], data["params"]


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
    ref_regime = model.regimes[_INITIAL_REGIMES[0]]
    grids = ref_regime.states
    assets_lo, assets_hi = grids["assets"].start, grids["assets"].stop
    aime_lo, aime_hi = grids["aime"].start, grids["aime"].stop
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
        "pref_type": jnp.asarray(rng.integers(0, 3, n_subjects).astype(np.int32)),
        "log_ft_wage_res": jnp.asarray(rng.choice(wage_res_pts, n_subjects)),
        "lagged_labor_supply": jnp.asarray(lls),
        # claimed_ss is required by non-initial regimes (ss="choose");
        # everyone starts as not-yet-claimed.
        "claimed_ss": jnp.zeros(n_subjects, dtype=jnp.int32),
    }
