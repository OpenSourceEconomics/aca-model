"""Tiny factories that wrap `create_model` with the benchmark snapshot.

Used by tests that need a structurally faithful model without spelling
out fixed_params, wage_params, and a pref-type grid at every call site.
Production callers (aca-slurm, scripts) assemble these explicitly.
"""

from dataclasses import replace

from lcm import DiscreteGrid, Model

from aca_model.aca.health_insurance import PolicyVariant
from aca_model.agent.preferences import BenchmarkPrefType
from aca_model.baseline.model import build_model_spec, create_model
from aca_model.baseline.regimes import REGIME_SPECS
from aca_model.benchmark import get_benchmark_params
from aca_model.config import BENCHMARK_GRID_CONFIG, GridConfig

_NONGROUP_BEFORE_MEDICARE = frozenset(
    name
    for name, spec in REGIME_SPECS.items()
    if spec["his"] == "nongroup" and spec["mc"] == "nomc"
)

_UNFOLDED_GRID_CONFIG = replace(
    BENCHMARK_GRID_CONFIG, fold_hcc_transitory=False, subtract_hcc_bill=False
)


def make_baseline_model(*, grid_config: GridConfig = _UNFOLDED_GRID_CONFIG) -> Model:
    """Baseline model on `grid_config` with the benchmark snapshot params."""
    fixed_params, wage_params, _ = get_benchmark_params(model=None)
    return create_model(
        spec=build_model_spec(
            pia_aime_grid=fixed_params["pia_aime_grid"],
            wage_params=wage_params,
            grid_config=grid_config,
            pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        ),
        fixed_params=fixed_params,
    )


def aca_fixed_params(*, fixed_params: dict, policy: PolicyVariant) -> dict:
    """Return baseline `fixed_params` fit for the ACA `policy` model.

    Under a variant with the reformed non-group market, the non-group regimes
    before Medicare price cover at the community-rated ACA premium, so the
    risk-rated premium parameters (prefix `premium_`) are dropped from those
    regimes, as aca-slurm does.
    """
    if policy == PolicyVariant.ACA_ONLY_MEDICAID_EXPANSION:
        return fixed_params
    return {
        name: (
            {k: v for k, v in value.items() if not k.startswith("premium_")}
            if name in _NONGROUP_BEFORE_MEDICARE
            else value
        )
        for name, value in fixed_params.items()
    }


def make_aca_model(
    *, policy: PolicyVariant, grid_config: GridConfig = _UNFOLDED_GRID_CONFIG
) -> Model:
    """ACA model on `grid_config` with the benchmark snapshot params."""
    fixed_params, wage_params, _ = get_benchmark_params(model=None)
    return create_model(
        spec=build_model_spec(
            pia_aime_grid=fixed_params["pia_aime_grid"],
            policy=policy,
            wage_params=wage_params,
            grid_config=grid_config,
            pref_type_grid=DiscreteGrid(BenchmarkPrefType),
        ),
        fixed_params=aca_fixed_params(fixed_params=fixed_params, policy=policy),
    )
