"""Print cash_on_hand for the failing subjects at every labor_supply choice.

If `cash_on_hand` evaluates to NaN for any subject, that explains why my
new `borrowing_constraint = c <= max(cash_on_hand, floor)` rejects every
action: `max(NaN, floor) == NaN` and `c <= NaN == False`.

Usage on gpu-01:
    cd ~/aca-dev
    pixi run -e cuda12 python aca-model/debug_cash_on_hand.py
"""

import pickle

import jax.numpy as jnp
import numpy as np
import pandas as pd
from dags import concatenate_functions
from lcm.params.processing import broadcast_to_template

from aca_estimation._assemble_params import (
    _NON_MODEL_KEYS,
    assemble_fixed_params,
    assemble_params,
)
from aca_estimation.config import ACA_DATA_BLD
from aca_estimation._type_prediction import triple_initdist_by_pref_type
from aca_model.aca import PolicyVariant
from aca_model.aca.model import create_model as create_aca_model
from aca_model.config import GRID_CONFIG_FOR_RUN
from aca_model.consumption_grid import inject_consumption_points

# Subjects whose `borrowing_constraint=False` in the gpu-01 production
# diagnostic. (subject_id, regime_name) tuples. Subject 1299 is included
# as a positive control: production showed `borrowing_constraint=True`
# for it, so its cash_on_hand should be finite.
_TARGETS: tuple[tuple[int, str], ...] = (
    (1131, "nongroup_nomc_inelig_canwork"),
    (1299, "nongroup_nomc_inelig_canwork"),  # positive control
    (9013, "retiree_nomc_inelig_canwork"),
    (10108, "nongroup_dimc_inelig_canwork"),
)


def _load(name: str):
    with open(ACA_DATA_BLD / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


def main() -> None:
    ss = _load("social_security_params")
    tax = _load("tax_params")
    ssi = _load("ssi_medicaid_params")
    hi = _load("health_insurance_params")
    pension = _load("pension_params")
    wage = _load("wage_params")
    transition = _load("transition_probs")
    env = _load("environment_constants")
    hcc_insurer = _load("hcc_insurer_params")
    pref = _load("preference_start_values")
    initdist_df = pd.read_pickle(ACA_DATA_BLD / "initial_conditions.pkl")

    n_subjects = 3 * len(initdist_df)
    bare_model = create_aca_model(
        policy=PolicyVariant.ACA, grid_config=GRID_CONFIG_FOR_RUN, n_subjects=1
    )
    template = bare_model.get_params_template()
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
    broadcast_to_template(params=fixed_params, template=template, required=False)
    params = assemble_params(
        pref_params=pref, base_wage_profile=wage["log_ft_wage_base"]
    )

    model = create_aca_model(
        n_subjects=n_subjects,
        policy=PolicyVariant.ACA,
        fixed_params=fixed_params,
        wage_params=wage,
        grid_config=GRID_CONFIG_FOR_RUN,
    )
    model_params = {k: v for k, v in params.items() if k not in _NON_MODEL_KEYS}
    model_params = inject_consumption_points(params=model_params, model=model)
    initial = triple_initdist_by_pref_type(initdist_df)

    internal_params = model._process_params(model_params)  # noqa: SLF001

    # Evaluate cash_on_hand and borrowing_constraint for each target subject
    # at each labor_supply choice with c = consumption_floor.
    consumption_floor = float(model_params["consumption_floor"])
    for subject_id, regime_name in _TARGETS:
        regime = model.regimes[regime_name]
        internal_regime = model.internal_regimes[regime_name]
        functions = internal_regime.simulate_functions.functions
        constraints = internal_regime.simulate_functions.constraints
        regime_params = {
            **internal_regime.resolved_fixed_params,
            **dict(internal_params.get(regime_name, {})),
        }

        # Build a function returning (cash_on_hand, borrowing_constraint).
        targets = ["cash_on_hand"]
        if "borrowing_constraint" in constraints:
            targets.append("borrowing_constraint")
        all_funcs = dict(functions)
        all_funcs.update(dict(constraints))
        evaluator = concatenate_functions(
            functions=all_funcs,
            targets=targets,
            return_type="dict",
            enforce_signature=False,
            set_annotations=True,
        )

        # Per-subject states (single subject; pull idx subject_id from the
        # already-tripled initial conditions).
        subject_state = {
            k: v[subject_id : subject_id + 1]
            for k, v in initial.items()
            if k != "regime"
        }

        labor_supply_grid = np.asarray(regime.actions["labor_supply"].to_jax())
        print(f"\n=== subject {subject_id} ({regime_name}) ===")
        print(
            f"  state: assets={float(subject_state['assets'][0]):.2f}, "
            f"aime={float(subject_state['aime'][0]):.2f}, "
            f"spousal_income={int(subject_state['spousal_income'][0])}, "
            f"health={int(subject_state['health'][0])}, "
            f"hcc_persistent={float(subject_state['hcc_persistent'][0]):.4f}, "
            f"hcc_transitory={float(subject_state['hcc_transitory'][0]):.4f}"
        )
        for ls in labor_supply_grid:
            kwargs = {
                **{k: v[0] for k, v in subject_state.items()},
                "consumption": jnp.float32(consumption_floor),
                "labor_supply": jnp.int32(int(ls)),
                "age": jnp.float32(51.0),
                "period": jnp.int32(0),
                **{k: v for k, v in regime_params.items()},
            }
            try:
                out = evaluator(
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k in evaluator.__signature__.parameters
                    }
                )
                coh = float(out["cash_on_hand"])
                bc = (
                    bool(out.get("borrowing_constraint", True))
                    if "borrowing_constraint" in out
                    else "n/a"
                )
                nan_flag = " <-- NaN!" if not np.isfinite(coh) else ""
                print(
                    f"  ls={int(ls):d}: cash_on_hand={coh:14.2f}  "
                    f"borrowing_constraint(c=c_floor)={bc}{nan_flag}"
                )
            except (KeyError, TypeError) as exc:
                print(f"  ls={int(ls):d}: eval failed: {exc!r}")


if __name__ == "__main__":
    main()
