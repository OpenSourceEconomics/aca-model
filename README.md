# aca-model

Core lifecycle model for the ACA structural retirement project.

The IID transitory medical-cost shock is read only as its draw `next_hcc_transitory`, so
pylcm integrates it out and no regime carries it as a state. The benchmark declares the
asset transition as resources plus the negated medical bill; to measure the plain asset
law instead, pass `grid_config=replace(BENCHMARK_GRID_CONFIG, subtract_hcc_bill=False)`
to `create_benchmark_model`, and the same grid configuration to
`get_benchmark_initial_conditions`.

`build_model_spec(...)` returns the model's domain: regimes, ages, edges, broadcast
slots and the default entry `initial_nodes`. `policy=None` builds the baseline; a
`PolicyVariant` overlays its ACA schedules.
`create_model(spec=..., fixed_params=..., **model_options)` forwards every other
`lcm.Model` option unchanged, and an option the spec also holds, such as
`initial_nodes`, overrides it. `create_benchmark_model` takes the same options.

Without `execution_config`, pylcm's default applies: the budget is the smallest selected
device pool limit less headroom, which requires a preallocated allocator pool
(`XLA_PYTHON_CLIENT_PREALLOCATE` not `false`); CPU runs unbudgeted.

To combine a measured budget with device selection and sharding:

```python
from dataclasses import replace

from aca_model.execution import execution_config_for_devices

execution_config = replace(
    execution_config_for_devices(devices=(0, 1, 2)),
    sharded_states=("pref_type",),
)
```

Pass this policy as `execution_config` to `create_model` or `create_benchmark_model`.
Record its exact `device_memory_bytes`, the selected devices' allocator observations,
executed precision and selected program widths with performance evidence. The budget is
an allocator ceiling; it does not establish that all workload phases fit.

Explicit policies pass through unchanged. In particular,
`ExecutionConfig(device_memory_bytes=None)` is an unbudgeted control and retains pylcm's
bootstrap width selection. Economic grids, numerical solver settings and result
retention are separate choices.

`GridConfig` accepts economic grid sizes and numerical solver choices only. Execution
controls belong to `ExecutionConfig`: sharded state names go in `sharded_states`,
declared program widths in `axis_widths`, and the per-device byte ceiling in
`device_memory_bytes`. Obsolete grid execution keywords are rejected. A state-specific
batch size has no automatic conversion to a flattened program width; callers must choose
and record an explicit execution policy.

For comparisons with an older pylcm version, retain its model/configuration pin and use
a separately reviewed API adapter. Preserve grids, transition declarations, numerical
solver choices, retention and seeds across the comparison. An unspecified width in the
new policy means planner selection, not a translation of an old zero-valued batch size.
