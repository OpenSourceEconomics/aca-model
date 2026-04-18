@../.ai-instructions/profiles/tier-a.md @../.ai-instructions/modules/jax.md
@../.ai-instructions/modules/ml-econometrics.md
@../.ai-instructions/modules/optimagic.md @../.ai-instructions/modules/pandas.md
@../.ai-instructions/modules/plotting.md
@../.ai-instructions/modules/project-structure.md @../.ai-instructions/modules/pytask.md

# aca-model

## Build & Test

```bash
pixi run -e tests-cpu tests           # Run all tests (CPU; -e tests-cuda13 for GPU)
pytest tests/test_pensions.py         # Run a single test file
pytest tests/test_pensions.py -k "test_name"  # Run a single test
pytest -m long_running                # Run long-running tests only
pixi run -e type-checking ty          # Type checking with ty
prek run --all-files                  # Run all pre-commit hooks (from aca-model/)
```

Tests exclude `long_running` by default (configured in pyproject.toml).

## Architecture

### Package Structure (`src/aca_model/`)

Four subpackages, cleanly separated by concern:

- **`agent/`** — Individual behavior: health states & transitions, labor supply & wages,
  preferences (CES utility, leisure, bequests), asset transitions & cash-on-hand
- **`environment/`** — External rules: Social Security (AIME→PIA, earnings test, SSDI),
  private pensions, federal income & payroll taxes
- **`baseline/`** — Pre-ACA model specification: 18 active regimes + dead state, health
  insurance premiums/OOP, SSI/Medicaid eligibility
- **`aca/`** — ACA policy overlay: mandate penalty, premium subsidies, cost-sharing
  reductions, Medicaid expansion. Applied via function swapping on baseline regimes.

### The Regime System

The model is built on **pylcm** (`lcm.Model`, `lcm.Regime`). Each regime is a
self-contained dynamic program with states, actions, functions, state transitions, and
regime transitions.

**18 regimes** are factored along 4 dimensions:

```
{HIS} × {Medicare} × {SS} × {Labor}
HIS ∈ {retiree, tied, nongroup}
Medicare ∈ {nomc, dimc, oamc}
SS ∈ {inelig, choose, forced}
Labor ∈ {canwork, forcedout}
```

Only 18 of the 54 possible combinations are active. Regime names encode the spec:
`"retiree_nomc_inelig_canwork"`. The `REGIME_SPECS` dict in
`baseline/regimes/_common.py` drives all regime construction programmatically.

### Regime Construction

`baseline/regimes/__init__.py` dispatches `build_regime(name)` to HIS-specific builders
in `_retiree.py`, `_tied.py`, `_nongroup.py`. Each builder:

1. Calls shared `build_states(spec)` / `build_actions(spec)` from `_common.py`
1. Adds HIS-specific functions (utility, premiums, pensions, regime transitions)
1. Returns a `Regime` object

### ACA Policy Overlay

ACA variants don't create new regimes — they swap functions on baseline regimes via
`dataclasses.replace()` in `aca/regimes/_overrides.py`:

- Baseline uses stub functions (mandate_penalty → 0.0, subsidies → 0.0, etc.)
- ACA replaces stubs with real policy functions from `aca/policies.py`
- `PolicyVariant` enum controls which policies are active (full ACA, no mandate,
  Medicaid only)

### Key State Variables

- `assets`: Savings grid [0, 500k], 24 points
- `aime`: Average Indexed Monthly Earnings [0, 8k], 12 points
- `health`: `HealthWithDisability` (disabled/bad/good) pre-65, `Health` (bad/good)
  post-65
- `log_ft_wage_res`: AR(1) wage residual shock (5-point Rouwenhorst)
- `hcc_persistent` / `hcc_transitory`: Health cost shocks (`_ShockGrid` — integrated
  over, policy does not condition on them)
- Regime transitions determined by `select_target_for_age()` based on age and actions

### Key Design Decisions

- **AgeGrid with int ages**: Model uses `AgeGrid` with integer start/stop — all `age`
  parameters are `int`, not `float`
- **OOP timing**: OOP health costs are in `next_assets` (post-consumption), not
  `cash_on_hand`. Matches struct-ret: agent doesn't condition consumption on OOP.
- **DAG key ≠ function name**: `functions["marginal_tax_rate"] = taxes.marginal_rate` —
  the dict key must match consuming functions' parameter names, not the definition name.
  Don't stutter module name in function name (`taxes.marginal_rate` not
  `taxes.marginal_tax_rate`).
- **Stub pattern**: Non-applicable policy functions use stubs returning neutral values
  (0.0 for subsidies/penalties, 1.0 for scale factors). See `aca/policies.py`.
- **Constants as fixed params**: When a state transition function needs a parameter not
  computed by any DAG function (e.g., `benefit_withheld_fraction` in inelig regimes),
  pylcm treats it as an external parameter supplied via the params dict. This is
  intentional — don't add stub functions to "fix" missing DAG entries.
- **`buy_private` action**: Only nongroup-nomc regimes (2 of 18) have `buy_private` as
  an action. Those use `premium()` and `primary_oop()` which condition on it. All other
  regimes use `premium_insured()` / `premium_retired()` and `oop_costs` directly — no
  `buy_private` parameter at all.
- **`reference_age` parameter**: Fixed cost of work uses `age - reference_age` (not a
  hardcoded constant). Same parameter appears in `leisure()`, `tied()`, `with_hours()`,
  and `utility_scale_factor()`.
