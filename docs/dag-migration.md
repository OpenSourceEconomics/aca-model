# ACA: one structural stage table, existing numerical programs

Requires the proposed pylcm v3 schedule/support API, not unchanged PR #474.
Both factories forward optional `initial_regimes` unchanged. Default coverage
requires no separate population/activity table; an explicit entry restriction
changes admission, never solved values. Coverage remains exactly **182** nodes.

`_AGE_STAGES` and the host-side clock supply both numerical next-age routing and
source-age support schedules. Each source's existing numerical transition body and
per-target probability cell are shared across age cases. There is no closure
specialization per boundary. Code order/global regime IDs and existing parameter
paths remain; different target schemas can still need different continuation
programs, so no backend compile-time improvement is claimed.

The dead regime remains `regime_transitions=None` at every age, as in the supplied base.
Age 94 retains its living consumption/saving problem and handoff into the separate
age-95 bequest. The three-to-two health grid, claim/lagged-work entry and exit,
carried pension state and phase semantics are not simplified away.

Use `model.reachability.nodes` and `model.initial_nodes`; keep the existing phase
target queries and period-keyed solutions. Normalize schedules before existing
solver validation. A wrapper alone does not make a supported solver case invalid;
genuine unsupported variation must raise via current errors, without silently
changing actions or solver. GridSearch is the first numerical acceptance target.

The original transition test module is retained byte-for-byte; new coverage tests
check nodes, boundary support, shared probability cells and terminal declarations.
Offline JAX row equality is not real model construction, Bellman/panel equivalence,
compiled validation/handoff/replay acceptance or GPU evidence. Run those gates and
matched build/compile/warm timings in the supported package environment. Annual
ACA calibration is not automatically valid on a differently spaced clock.
