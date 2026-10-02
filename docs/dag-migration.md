# ACA: one structural stage table, existing numerical programs

Both factories declare the same admissible starts, `INITIAL_NODES`: ages 51–60
as explicit `(age, regime)` pairs in the five `*_inelig_canwork`
regimes, 50 pairs. Every later age, and `dead` at any age, is reached only through
transitions; the engine derives the solved domain from these roots, **184** nodes.
`(51, "dead")` is not solved. Starts at age ≥ 61 or in `dead` are rejected at
admission.

`simulate_with_dense_index` drops initial-condition rows older than the last
admissible start age and raises on any kept row outside `model.initial_nodes`.

`_AGE_STAGES` and the host-side clock supply both numerical next-age routing and
`Model.edges` source-age selectors, grouped by destination. Each source has one
mapping of `StochasticTransition` probability cells; the model graph selects the
cells allowed at each source age. Code order/global regime IDs and parameter
paths remain; different target schemas can still need different continuation
programs, so no backend compile-time improvement is claimed.

The dead regime remains `regime_transitions=None` at every age. Age 95 retains its
living consumption/saving problem and handoff into the separate age-96 bequest. The
three-to-two health grid, claim/lagged-work entry and exit, carried pension state
and phase semantics are not simplified away.

Use `model.graph.nodes`, `model.graph.initial_nodes`, and `model.graph.edges`;
`model.graph.solution` and `model.graph.simulation` expose the effective phase
graphs. The declared edge selectors are identical across policy variants and
phases; carried pension wealth retains its distinct solve/simulate laws. Keep the phase
target queries and period-keyed solutions. GridSearch is the first numerical
acceptance target. Annual ACA calibration is not automatically valid on a
differently spaced clock.
