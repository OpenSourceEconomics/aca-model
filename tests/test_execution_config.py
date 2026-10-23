"""Model factories forward pylcm's generic options without changing economic grids."""

from functools import partial

import jax
import pytest
from helpers.model import aca_fixed_params
from lcm import DiscreteGrid, ExecutionConfig, InitialNodes
from lcm.exceptions import ExecutionPlanningError

from aca_model.aca.health_insurance import PolicyVariant
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline import model as baseline_model_module
from aca_model.baseline.model import build_model_spec, create_model
from aca_model.benchmark import create_benchmark_model, get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG

_DOMAIN_KEYS = frozenset(
    {
        "regimes",
        "ages",
        "regime_id_class",
        "description",
        "derived_categoricals",
        "initial_nodes",
        "edges",
        "functions",
        "constraints",
        "states",
        "state_transitions",
    }
)


def _spec(*, policy: PolicyVariant | None = None) -> dict:
    fixed_params, wage_params, _ = get_benchmark_params(model=None)
    return build_model_spec(
        policy=policy,
        grid_config=BENCHMARK_GRID_CONFIG,
        pia_aime_grid=fixed_params["pia_aime_grid"],
        wage_params=wage_params,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )


def _factory(kind):
    if kind == "benchmark":
        return partial(
            create_benchmark_model,
            pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        )
    fixed_params, _, _ = get_benchmark_params(model=None)
    policy = PolicyVariant.ACA if kind == "aca" else None
    if policy is not None:
        fixed_params = aca_fixed_params(fixed_params=fixed_params, policy=policy)
    return partial(create_model, spec=_spec(policy=policy), fixed_params=fixed_params)


@pytest.mark.parametrize("policy", [None, PolicyVariant.ACA])
def test_build_model_spec_holds_exactly_the_domain_arguments(
    policy: PolicyVariant | None,
) -> None:
    """The spec carries the model's domain and no generic `Model` option."""
    assert set(_spec(policy=policy)) == _DOMAIN_KEYS


@pytest.mark.parametrize("kind", ["baseline", "aca", "benchmark"])
def test_factory_preserves_explicit_device_selection(kind):
    """A factory uses exactly the selected device and preserves the model grids."""
    devices = (jax.devices()[-1].id,)
    model = _factory(kind)(execution_config=ExecutionConfig(devices=devices))

    assert model.execution_devices == devices
    assert len(model.user_regimes) == 19
    regime = model.user_regimes["retiree_nomc_inelig_canwork"]
    assert regime.states["assets"].n_points == 3
    assert regime.states["aime"].n_points == 38
    assert regime.actions["consumption_dollars"].n_points == 5


@pytest.mark.parametrize("kind", ["baseline", "aca", "benchmark"])
def test_factory_preserves_axis_width_validation(kind):
    """A width for an undeclared program axis is refused by the model."""
    config = ExecutionConfig(axis_widths={"aca_undeclared_axis": 1})
    with pytest.raises(ExecutionPlanningError, match="aca_undeclared_axis"):
        _factory(kind)(execution_config=config)


class _ConstructionObservedError(Exception):
    def __init__(self, kwargs: dict) -> None:
        super().__init__()
        self.kwargs = kwargs


def _observe_model_kwargs(monkeypatch, factory, **options) -> dict:
    def observe_pylcm_constructor(**kwargs):
        raise _ConstructionObservedError(kwargs)

    monkeypatch.setattr(baseline_model_module, "Model", observe_pylcm_constructor)
    with pytest.raises(_ConstructionObservedError) as observed:
        factory(**options)
    return observed.value.kwargs


@pytest.mark.parametrize("kind", ["baseline", "aca", "benchmark"])
def test_factory_without_execution_config_leaves_pylcm_default(monkeypatch, kind):
    """No execution policy reaches `Model`, so pylcm's device default applies."""
    kwargs = _observe_model_kwargs(monkeypatch, _factory(kind))
    assert "execution_config" not in kwargs


@pytest.mark.parametrize("kind", ["baseline", "aca", "benchmark"])
def test_factory_forwards_an_explicit_execution_config_unchanged(monkeypatch, kind):
    """An explicit policy reaches `Model` as the same object."""
    config = ExecutionConfig(devices=(0,), device_memory_bytes=400)
    kwargs = _observe_model_kwargs(monkeypatch, _factory(kind), execution_config=config)
    assert kwargs["execution_config"] is config


@pytest.mark.parametrize("kind", ["baseline", "benchmark"])
def test_factory_forwards_any_model_option(monkeypatch, kind):
    """A generic `Model` option the spec does not hold is passed through."""
    kwargs = _observe_model_kwargs(monkeypatch, _factory(kind), durable_identity=False)
    assert kwargs["durable_identity"] is False


@pytest.mark.parametrize("kind", ["baseline", "benchmark"])
def test_factory_initial_nodes_override_the_entry_default(monkeypatch, kind):
    """`initial_nodes` passed as an option replaces the spec's entry starts."""
    nodes = InitialNodes(by_age={52: ("retiree_nomc_inelig_canwork",)})
    kwargs = _observe_model_kwargs(monkeypatch, _factory(kind), initial_nodes=nodes)
    assert kwargs["initial_nodes"] is nodes


@pytest.mark.parametrize("kind", ["baseline", "aca", "benchmark"])
def test_factory_rejects_construction_subject_count(kind):
    """Population size is supplied by initial conditions, not model construction."""
    with pytest.raises(TypeError, match="n_subjects"):
        _factory(kind)(n_subjects=1)
