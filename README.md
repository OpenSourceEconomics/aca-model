# aca-model

Core lifecycle model for the ACA structural retirement project.

Model factories accept `execution_config=ExecutionConfig(...)`. An omitted policy uses
the smallest JAX allocator `bytes_limit` among the selected accelerators as an explicit
per-device budget. This enables budget-aware width selection; CPU construction remains
unbudgeted. Missing accelerator limits require an explicit policy instead of silently
falling back to unbudgeted execution.

To combine a measured budget with device selection and sharding:

```python
from dataclasses import replace

from aca_model.execution import execution_config_for_devices

execution_config = replace(
    execution_config_for_devices(devices=(0, 1, 2)),
    sharded_states=("pref_type",),
)
```

Pass this policy to the baseline, ACA or benchmark factory. Record its exact
`device_memory_bytes`, the selected devices' allocator observations, executed precision
and selected program widths with performance evidence. The budget is an allocator
ceiling; it does not establish that all workload phases fit.

Explicit policies pass through unchanged. In particular, `ExecutionConfig()` is an
unbudgeted control and retains pylcm's bootstrap width selection. Economic grids,
numerical solver settings and result retention are separate choices.

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
