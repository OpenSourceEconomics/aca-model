"""Session-scoped models shared by every test that builds one with the same arguments.

Under pytest-xdist each worker builds each model at most once. Tests read these
models and may solve or simulate them; none reassigns their attributes.
"""

import functools
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
from helpers.model import make_aca_model, make_baseline_model
from lcm import DiscreteGrid, Model

from aca_model.aca.health_insurance import PolicyVariant
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.benchmark import create_benchmark_model


@pytest.fixture(scope="session")
def benchmark_model() -> Model:
    """`create_benchmark_model` with the 2-type preference grid and its defaults.

    The same model as `make_baseline_model(grid_config=BENCHMARK_GRID_CONFIG)`.
    """
    return create_benchmark_model(pref_type_grid=DiscreteGrid(BenchmarkPrefType))


@pytest.fixture(scope="session")
def baseline_model() -> Model:
    """`make_baseline_model()`: the benchmark snapshot on the plain asset law."""
    return make_baseline_model()


@pytest.fixture(scope="session")
def aca_model_for() -> Callable[[PolicyVariant], Model]:
    """Return `make_aca_model` for a policy, built once per policy."""

    @functools.cache
    def build(policy: PolicyVariant) -> Model:
        return make_aca_model(policy=policy)

    return build
