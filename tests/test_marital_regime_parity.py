"""Parity of the marital-status-in-regimes encoding.

Marital status is an axis of the regime rather than a state carried inside it,
so a household that used to sit in regime `R` with the three-code
`spousal_income = z` now sits in `f"{marital(z)}_{R}"` carrying only the
two-code within-marriage state. The two encodings describe the same dynamic
program, so the value function and the policy at mapped states must agree
exactly; the simulated *paths* need not, because replacing one three-category
draw by a marital draw plus a conditional draw consumes a different random-key
sequence. The gate is split accordingly:

- **Deterministic parity** — period-zero values and discrete decisions at
  mapped seeds, compared directly. Period zero is before any shock is drawn,
  so the value is `V(state)` and the decision is its argmax; the model carries
  no taste shocks, so both are deterministic. Needs the reference artifact
  from the pre-split encoding (see `scripts/dump_marital_parity_reference.py`)
  and skips without it.
- **Stochastic parity** — the three-by-three marital/spousal transition
  recovered from simulated frequencies against the one the model's own factors
  imply. Compares distributions, not paths.

Both simulate the full model and are `long_running`. The retained-target mass
check needs no solve and runs with the fast suite: regime probabilities that
fail to sum to one over the retained targets give silent `NaN` value
functions rather than an error.
"""

import dataclasses
import inspect
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from helpers.model import _DERIVED_CATEGORICALS  # ty: ignore[unresolved-import]
from lcm import DiscreteGrid, MarkovTransition, Model

from aca_model.agent.labor_market import LaborSupply
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline.model import create_model
from aca_model.baseline.regimes import REGIME_SPECS, build_regime
from aca_model.baseline.regimes._common import RegimeId, build_grids
from aca_model.benchmark import (
    get_benchmark_initial_conditions,
    get_benchmark_params,
)
from aca_model.config import BENCHMARK_GRID_CONFIG, MODEL_CONFIG

PARITY_GRID_CONFIG = dataclasses.replace(
    BENCHMARK_GRID_CONFIG,
    n_assets_gridpoints=8,
    n_aime_gridpoints=3,
    n_consumption_dollars_gridpoints=16,
    n_wage_res_gridpoints=3,
    n_hcc_persistent_gridpoints=3,
    n_hcc_transitory_gridpoints=3,
)

N_SUBJECTS = 200
SEED = 0
N_PERIODS = MODEL_CONFIG.end_age - MODEL_CONFIG.start_age

# The pre-split panel and the seed it was drawn from, keyed by the old
# `(regime_name, spousal_income)` identity. Regenerate with
# `scripts/dump_marital_parity_reference.py` on a pre-split checkout.
REFERENCE_FILE = Path(__file__).parent / "data" / "marital_parity_reference.pkl"

# What each pre-split `spousal_income` code — `(single, married_no_inc,
# married_has_inc)`, in declaration order — maps to under the split.
_MARITAL_OF_CODE = ("single", "married", "married")
_WITHIN_MARRIAGE_OF_CODE = (0, 0, 1)

_DISCRETE_COLUMNS = ("regime_name", "claim_ss", "labor_supply", "buy_private")

# Mass may leak only at the level a float32 probability vector can carry.
_MASS_ATOL = 1e-6
# fp32 value functions agree to a few ulps of a value on the order of
# hundreds; the tolerance is absolute so it does not silently widen with the
# magnitude of the value.
_VALUE_ATOL = 1e-6


def _grids(grid_config=PARITY_GRID_CONFIG):
    fixed_params, wage_params, _ = get_benchmark_params(model=None)
    return build_grids(
        grid_config=grid_config,
        fixed_params=fixed_params,
        wage_params=wage_params,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )


def _make_model() -> Model:
    fixed_params, wage_params, _ = get_benchmark_params(model=None)
    return create_model(
        n_subjects=N_SUBJECTS,
        fixed_params=fixed_params,
        wage_params=wage_params,
        derived_categoricals=_DERIVED_CATEGORICALS,
        grid_config=PARITY_GRID_CONFIG,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )


def _seeded_panel(model: Model, initial_conditions: dict) -> pd.DataFrame:
    """Solve + simulate the seeded panel and return it in subject order."""
    _, _, params = get_benchmark_params(model=model)
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    df = result.to_dataframe()
    return df.sort_values(["subject_id", "period"]).reset_index(drop=True)


@pytest.mark.parametrize("name", list(REGIME_SPECS))
def test_regime_probabilities_sum_to_one_over_retained_targets(name: str) -> None:
    """Every regime's declared targets carry the whole unit of probability.

    A regime declares a narrow per-target support; any mass the transition
    function places outside it is dropped silently and shows up only as a
    `NaN` value function, so the retained mass is asserted directly.
    """
    regime = build_regime(name, _grids())
    cells = cast("Mapping[str, MarkovTransition]", regime.transition)
    inputs: dict = {
        "survival_probs": jnp.linspace(0.99, 0.5, N_PERIODS),
        "marital_probs": jnp.array([0.3, 0.7]),
        "labor_supply": jnp.array(LaborSupply.h2000),
        "is_medicaid_eligible": jnp.array(False),
    }
    for age in range(MODEL_CONFIG.start_age, MODEL_CONFIG.end_age - 1):
        if not bool(regime.active(jnp.int32(age))):
            continue
        retained = 0.0
        for cell in cells.values():
            declared = set(inspect.signature(cell.func).parameters)
            retained += float(
                cell(
                    age=jnp.int32(age),
                    period=jnp.int32(age - MODEL_CONFIG.start_age),
                    **{k: v for k, v in inputs.items() if k in declared},
                )
            )
        assert abs(retained - 1.0) <= _MASS_ATOL, f"age {age}: {retained!r}"


def _implied_chain_from_params(params: dict) -> np.ndarray:
    """Return the `[n_ages, 3, 3]` chain the model's own factors compose to.

    Row `z` is `(pS(z), pM(z) q0(z), pM(z) q1(z))` read off the regime that
    carries source code `z`: the single regimes carry `z = single`, the
    married ones both married codes.
    """
    single = params["single_retiree_nomc_inelig_canwork"]
    married = params["married_retiree_nomc_inelig_canwork"]
    single_marital = np.asarray(single["marital_probs"])
    single_income = np.asarray(single["spousal_income_trans_probs"])
    married_marital = np.asarray(married["marital_probs"])
    married_income = np.asarray(married["spousal_income_trans_probs"])

    n_ages = single_marital.shape[0]
    chain = np.empty((n_ages, 3, 3))
    chain[:, 0, 0] = single_marital[:, 0]
    chain[:, 0, 1:] = single_marital[:, 1, None] * single_income
    chain[:, 1:, 0] = married_marital[:, :, 0]
    chain[:, 1:, 1:] = married_marital[:, :, 1, None] * married_income
    return chain


def _source_codes(panel: pd.DataFrame) -> np.ndarray:
    """Return each row's three-code spousal state, `-1` where the row is dead."""
    names = panel["regime_name"].astype("str")
    within = np.nan_to_num(
        pd.to_numeric(panel["spousal_income"], errors="coerce").to_numpy()
    ).astype(int)
    codes = np.where(names.str.startswith("married_").to_numpy(), 1 + within, 0)
    return np.where((names == "dead").to_numpy(), -1, codes)


@pytest.mark.long_running
def test_simulated_transitions_reproduce_the_implied_three_by_three() -> None:
    """Simulated `z -> z'` frequencies match the chain the model composes.

    The split replaces one three-category draw by a marital draw and a
    conditional one; the composed kernel must be the original, which the
    realised frequencies confirm up to sampling error.
    """
    model = _make_model()
    _, _, params = get_benchmark_params(model=model)
    panel = _seeded_panel(
        model,
        get_benchmark_initial_conditions(model=model, n_subjects=N_SUBJECTS, seed=SEED),
    )
    expected = _implied_chain_from_params(params)

    codes = _source_codes(panel)
    subject = panel["subject_id"].to_numpy()
    period = panel["period"].to_numpy()
    same_subject = subject[:-1] == subject[1:]
    alive = (codes[:-1] >= 0) & (codes[1:] >= 0) & same_subject

    counts = np.zeros((3, 3))
    weighted = np.zeros((3, 3))
    for source, target, per in zip(
        codes[:-1][alive], codes[1:][alive], period[:-1][alive], strict=True
    ):
        counts[source, target] += 1.0
        weighted[source] += expected[per, source]

    realised_rows = counts.sum(axis=1)
    covered = realised_rows > 0
    realised = counts[covered] / realised_rows[covered, None]
    predicted = weighted[covered] / realised_rows[covered, None]
    # Binomial standard error at the smallest covered row, times four.
    tolerance = 4.0 / np.sqrt(realised_rows[covered].min())
    max_error = float(np.abs(realised - predicted).max())
    assert max_error <= tolerance, (
        f"max frequency deviation {max_error:.4f} exceeds {tolerance:.4f}"
    )


def _reference() -> dict:
    if not REFERENCE_FILE.exists():
        pytest.skip(
            f"no pre-split reference at {REFERENCE_FILE}; regenerate with "
            "scripts/dump_marital_parity_reference.py on a pre-split checkout"
        )
    with REFERENCE_FILE.open("rb") as fh:
        return pickle.load(fh)


def _mapped_initial_conditions(model: Model, reference: dict) -> dict:
    """Map the pre-split seed onto the split encoding's regimes and states."""
    seed = dict(reference["initial_conditions"])
    codes = np.asarray(seed.pop("spousal_income")).astype(int)
    old_names = np.asarray(reference["initial_regime_names"], dtype=object)
    new_names = [
        f"{_MARITAL_OF_CODE[code]}_{name}"
        for code, name in zip(codes, old_names, strict=True)
    ]
    return {
        **{key: jnp.asarray(np.asarray(value)) for key, value in seed.items()},
        "regime_id": jnp.asarray(
            np.array([model.regime_names_to_ids[n] for n in new_names], dtype=np.int32)
        ),
        "spousal_income": jnp.asarray(
            np.array([_WITHIN_MARRIAGE_OF_CODE[c] for c in codes], dtype=np.int32)
        ),
    }


def _period_zero(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.loc[panel["period"] == 0].sort_values("subject_id")


@pytest.mark.long_running
def test_period_zero_value_matches_the_pre_split_encoding() -> None:
    """`V` at a mapped seed is the same number under both encodings.

    Period zero precedes every shock draw, so the recorded value is the value
    function evaluated at the seed — the quantity the split must preserve.
    """
    reference = _reference()
    model = _make_model()
    panel = _seeded_panel(model, _mapped_initial_conditions(model, reference))

    new = _period_zero(panel)["value"].to_numpy()
    old = _period_zero(reference["panel"])["value"].to_numpy()
    max_error = float(np.abs(new - old).max())
    assert max_error <= _VALUE_ATOL, f"max absolute value deviation {max_error:.3e}"


@pytest.mark.long_running
def test_period_zero_policy_matches_the_pre_split_encoding() -> None:
    """Every discrete decision at a mapped seed is identical under both
    encodings, once the regime name is stripped of its marital axis."""
    reference = _reference()
    model = _make_model()
    panel = _seeded_panel(model, _mapped_initial_conditions(model, reference))

    new = _period_zero(panel)
    old = _period_zero(reference["panel"])
    for column in _DISCRETE_COLUMNS:
        if column not in old.columns:
            continue
        left = new[column].astype("str").to_numpy()
        if column == "regime_name":
            left = np.array([name.split("_", 1)[1] for name in left], dtype=object)
        right = old[column].astype("str").to_numpy()
        assert np.array_equal(left, right), column


@pytest.mark.long_running
def test_seed_maps_onto_the_regime_carrying_its_spousal_code() -> None:
    """The mapping sends each pre-split code to the regime that can hold it.

    Guards the mapping itself: `single` must land in a single regime, and both
    married codes in the married copy of the same regime.
    """
    reference = _reference()
    model = _make_model()
    mapped = _mapped_initial_conditions(model, reference)

    ids_to_names = {int(v): k for k, v in model.regime_names_to_ids.items()}
    codes = np.asarray(reference["initial_conditions"]["spousal_income"]).astype(int)
    names = [ids_to_names[int(i)] for i in np.asarray(mapped["regime_id"])]
    for code, name, old_name in zip(
        codes, names, reference["initial_regime_names"], strict=True
    ):
        assert name == f"{_MARITAL_OF_CODE[code]}_{old_name}", (code, name)
        assert REGIME_SPECS[name]["marital"] == _MARITAL_OF_CODE[code]


def test_regime_id_orders_single_before_married() -> None:
    """`RegimeId` assigns every single regime an id below every married one,
    with `dead` last, so the id encodes the marital axis contiguously."""
    ids = {name: int(getattr(RegimeId, name)) for name in REGIME_SPECS}
    single = [i for name, i in ids.items() if REGIME_SPECS[name]["marital"] == "single"]
    married = [
        i for name, i in ids.items() if REGIME_SPECS[name]["marital"] == "married"
    ]
    assert max(single) < min(married)
    assert int(RegimeId.dead) > max(married)
