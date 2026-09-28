# ACA: one structural stage table, existing numerical programs

Both factories declare the same admissible starts, `INITIAL_REGIMES`: ages 51–61
(`AgeRange(start=start_age, stop=ss_early_age)`) in the five `*_inelig_canwork`
regimes, 55 pairs. Every later age, and `dead` at any age, is reached only through
transitions; the engine derives the solved domain from these roots, **181** nodes.
`(51, "dead")` is not solved. Starts at age ≥ 62 or in `dead` are rejected at
admission.

`simulate_with_dense_index` drops initial-condition rows older than the last
admissible start age and raises on any kept row outside `model.initial_nodes`.

`_AGE_STAGES` and the host-side clock supply both numerical next-age routing and
source-age support schedules. Each source's existing numerical transition body and
per-target probability cell are shared across age cases. There is no closure
specialization per boundary. Code order/global regime IDs and existing parameter
paths remain; different target schemas can still need different continuation
programs, so no backend compile-time improvement is claimed.

The dead regime remains `regime_transitions=None` at every age. Age 94 retains its
living consumption/saving problem and handoff into the separate age-95 bequest. The
three-to-two health grid, claim/lagged-work entry and exit, carried pension state
and phase semantics are not simplified away.

Use `model.reachability.nodes` and `model.initial_nodes`; keep the existing phase
target queries and period-keyed solutions. GridSearch is the first numerical
acceptance target. Annual ACA calibration is not automatically valid on a
differently spaced clock.
