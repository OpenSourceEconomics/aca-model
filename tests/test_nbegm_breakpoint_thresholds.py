"""Every declared breakpoint threshold names a parameter the model supplies.

`lcm.affine_breakpoint(threshold=...)` names a *parameter* of the decorated
schedule function: NBEGM reads it from the solve's params as
`f"{output}__{threshold}"` when it splits a cell at the threshold. A threshold
that names a DAG function instead has no such key, and the solve dies with a
`KeyError` deep in the interval partition — but only on the code path that
actually computes cell breakpoints, so a model whose thresholds are wrong can
still build and can still solve wherever that path is skipped.

These tests check the declaration against the params template directly, so the
mismatch is caught at model-creation cost rather than only by whichever solve
happens to reach the interval partition.
"""

import dataclasses

import pytest
from helpers.model import _DERIVED_CATEGORICALS  # ty: ignore[unresolved-import]
from lcm import DiscreteGrid, Model

from aca_model.agent import assets_and_income
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline.model import create_model
from aca_model.benchmark import get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG

_FIXED_PARAMS, _WAGE_PARAMS, _ = get_benchmark_params(model=None)
_GRID_CONFIG = dataclasses.replace(BENCHMARK_GRID_CONFIG, nbegm_jump_read="bridged")

# Schedule functions carrying `@lcm.piecewise_affine`, by the name the regimes
# wire them under. NBEGM forms its params key from that name, not from the
# Python function name, so the pairing is what the check needs.
_SCHEDULE_FUNCTIONS = {"resources": assets_and_income.resources}


def _nbegm_model() -> Model:
    return create_model(
        n_subjects=1,
        fixed_params=_FIXED_PARAMS,
        wage_params=_WAGE_PARAMS,
        derived_categoricals=_DERIVED_CATEGORICALS,
        grid_config=_GRID_CONFIG,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        solver="nbegm",
    )


def _declared_threshold_keys() -> set[str]:
    """Return the params keys NBEGM will look up for declared breakpoints."""
    keys = set()
    for output, func in _SCHEDULE_FUNCTIONS.items():
        meta = getattr(func, "__lcm_piecewise_affine__", None)
        if meta is None:
            continue
        keys |= {f"{output}__{bp.threshold}" for bp in meta.breakpoints}
    return keys


# `resources` names its kink threshold `consumption_dollars_floor`, which every
# living regime supplies as a DAG function rather than as a parameter, so the
# key NBEGM looks up does not exist. A solve only notices where it reaches the
# interval partition — the production grid does, the benchmark grid does not.
_KNOWN_UNSUPPLIED = pytest.mark.xfail(
    strict=True,
    reason=(
        "`resources` declares threshold='consumption_dollars_floor', which is a "
        "DAG function in every living regime, not a parameter. NBEGM looks up "
        "`resources__consumption_dollars_floor` in the solve's params and finds "
        "nothing."
    ),
)


def _threshold_params() -> list:
    return [
        pytest.param(
            key,
            marks=[_KNOWN_UNSUPPLIED]
            if key == "resources__consumption_dollars_floor"
            else [],
            id=key,
        )
        for key in sorted(_declared_threshold_keys())
    ]


def test_a_schedule_declares_at_least_one_breakpoint() -> None:
    """The check is only meaningful if the metadata is actually found."""
    assert _declared_threshold_keys()


@pytest.mark.parametrize("key", _threshold_params())
def test_every_declared_threshold_is_a_supplied_parameter(key: str) -> None:
    """Each `<output>__<threshold>` key is present in some regime's params.

    NBEGM reads the threshold from the solve's params. A threshold naming a
    DAG function resolves to no key at all, which surfaces as a `KeyError`
    only once a solve reaches the interval partition.
    """
    _, _, params = get_benchmark_params(model=_nbegm_model())
    supplied = set()
    for value in params.values():
        if isinstance(value, dict):
            supplied |= {str(name) for name in value}
    supplied |= {str(name) for name in params}
    assert key in supplied
