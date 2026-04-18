"""Generate the frozen benchmark params pickle from aca-data outputs.

Run this once after a full aca-data pipeline run. Produces
`aca-model/tests/data/benchmark_params.pkl` containing a
`{"fixed_params": ..., "params": ...}` dict suitable for
`aca_model.benchmark.get_benchmark_params()`.

Usage:
    pixi run python aca-model/scripts/generate_benchmark_params.py
"""

import sys
from pathlib import Path

import cloudpickle
from aca_estimation._assemble_params import (
    _NON_MODEL_KEYS,
    assemble_fixed_params,
    assemble_params,
    load,
)

from aca_model.baseline.model import create_model
from aca_model.config import BENCHMARK_GRID_CONFIG

_ROOT = Path(__file__).resolve().parents[2]
_ACA_DATA_BLD = _ROOT / "aca-data" / "bld"
_OUT = _ROOT / "aca-model" / "tests" / "data" / "benchmark_params.pkl"


def main() -> None:
    ss = load(_ACA_DATA_BLD / "social_security_params.pkl")
    tax = load(_ACA_DATA_BLD / "tax_params.pkl")
    ssi = load(_ACA_DATA_BLD / "ssi_medicaid_params.pkl")
    hi = load(_ACA_DATA_BLD / "health_insurance_params.pkl")
    pension = load(_ACA_DATA_BLD / "pension_params.pkl")
    wage = load(_ACA_DATA_BLD / "wage_params.pkl")
    transition = load(_ACA_DATA_BLD / "transition_probs.pkl")
    pref = load(_ACA_DATA_BLD / "preference_start_values.pkl")
    env = load(_ACA_DATA_BLD / "environment_constants.pkl")
    hcc_insurer = load(_ACA_DATA_BLD / "hcc_insurer_params.pkl")

    bare_model = create_model(grid_config=BENCHMARK_GRID_CONFIG)
    fixed_params = assemble_fixed_params(
        bare_model=bare_model,
        ss_params=ss,
        tax_params=tax,
        ssi_params=ssi,
        hi_params=hi,
        pension_params=pension,
        wage_params=wage,
        transition_params=transition,
        env_params=env,
        hcc_insurer_params=hcc_insurer,
        pref_params=pref,
    )
    params = assemble_params(pref_params=pref)
    params = {k: v for k, v in params.items() if k not in _NON_MODEL_KEYS}

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("wb") as fh:
        cloudpickle.dump({"fixed_params": fixed_params, "params": params}, fh)
    sys.stdout.write(
        f"Wrote {_OUT}: {len(fixed_params)} fixed_params keys, "
        f"{len(params)} params keys\n"
    )


if __name__ == "__main__":
    main()
