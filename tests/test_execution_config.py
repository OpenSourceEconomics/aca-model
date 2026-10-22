"""Model factories expose pylcm's execution policy without changing economic grids."""

from functools import partial
from types import SimpleNamespace

import jax
import pytest
from helpers.model import aca_fixed_params
from lcm import DiscreteGrid, ExecutionConfig
from lcm.exceptions import ExecutionPlanningError

from aca_model.aca import model as aca_model_module
from aca_model.aca.health_insurance import PolicyVariant
from aca_model.aca.model import create_model as create_aca_model
from aca_model.agent.health import GoodHealth
from aca_model.agent.labor_market import IsMarried
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline import model as baseline_model_module
from aca_model.baseline.health_insurance import HealthInsuranceState
from aca_model.baseline.model import create_model
from aca_model.benchmark import create_benchmark_model, get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG


def _factory(kind):
    if kind == "benchmark":
        return partial(
            create_benchmark_model,
            pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        )
    fixed_params, wage_params, _ = get_benchmark_params(model=None)
    factory = create_model
    if kind == "aca":
        factory = partial(create_aca_model, policy=PolicyVariant.ACA)
        fixed_params = aca_fixed_params(
            fixed_params=fixed_params, policy=PolicyVariant.ACA
        )
    return partial(
        factory,
        fixed_params=fixed_params,
        wage_params=wage_params,
        derived_categoricals={
            "good_health": DiscreteGrid(GoodHealth),
            "is_married": DiscreteGrid(IsMarried),
            "his": DiscreteGrid(HealthInsuranceState),
            "target_his": DiscreteGrid(HealthInsuranceState),
            "pref_type": DiscreteGrid(BenchmarkPrefType),
        },
        grid_config=BENCHMARK_GRID_CONFIG,
        pref_type_grid=DiscreteGrid(BenchmarkPrefType),
    )


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


@pytest.mark.parametrize("kind", ["baseline", "aca", "benchmark"])
@pytest.mark.parametrize(
    ("requested_policy", "expected_budget"),
    [
        (None, 800),
        (
            ExecutionConfig(
                devices=(0,),
                device_memory_bytes=400,
                sharded_states=("pref_type",),
                axis_widths={"cell": 32},
            ),
            400,
        ),
        (ExecutionConfig(devices=(0,)), ExecutionConfig().device_memory_bytes),
    ],
)
def test_factory_supplies_the_requested_or_measured_budget(
    monkeypatch, kind, requested_policy, expected_budget
):
    """Defaults use measured limits; explicit budgets and unbudgeted controls survive."""
    factory = _factory(kind)
    device = SimpleNamespace(
        id=0, platform="gpu", memory_stats=lambda: {"bytes_limit": 800}
    )
    monkeypatch.setattr(jax, "devices", lambda: [device])

    class ConstructionObservedError(Exception):
        pass

    def observe_pylcm_constructor(**kwargs):
        policy = kwargs["execution_config"]
        assert policy.device_memory_bytes == expected_budget
        assert policy.devices == (0,)
        if requested_policy is not None:
            assert policy == requested_policy
        raise ConstructionObservedError

    monkeypatch.setattr(baseline_model_module, "Model", observe_pylcm_constructor)
    monkeypatch.setattr(aca_model_module, "Model", observe_pylcm_constructor)
    with pytest.raises(ConstructionObservedError):
        factory(execution_config=requested_policy)


@pytest.mark.parametrize("kind", ["baseline", "aca", "benchmark"])
def test_factory_rejects_construction_subject_count(kind):
    """Population size is supplied by initial conditions, not model construction."""
    with pytest.raises(TypeError, match="n_subjects"):
        _factory(kind)(n_subjects=1)
