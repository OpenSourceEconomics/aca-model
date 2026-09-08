"""Hardware-local execution policies derived from allocator observations."""

import jax
from lcm import ExecutionConfig


def execution_config_for_devices(
    *, devices: tuple[int, ...] | None = None
) -> ExecutionConfig:
    """Create an execution policy bounded by the selected accelerator pools.

    The smallest allocator limit is a total per-device ceiling. Pylcm accounts
    for its resident arrays within that ceiling, so current allocator usage is
    not subtracted here. CPU devices do not supply an accelerator-pool budget.

    Args:
        devices: Visible device IDs to select, in order; None selects all devices
            on JAX's default backend.

    Returns:
        ExecutionConfig with explicit device IDs and the measured accelerator
        budget. Axis widths and sharded states remain unspecified.

    Raises:
        ValueError: A selected device is not visible or an accelerator has no
            positive integer allocator limit. Missing limits require an explicit
            policy.
    """
    policy = ExecutionConfig(devices=devices)
    visible = {device.id: device for device in jax.devices()}
    selected = tuple(visible) if policy.devices is None else policy.devices
    missing = tuple(device_id for device_id in selected if device_id not in visible)
    if missing:
        msg = f"Selected devices are not visible: {missing}."
        raise ValueError(msg)
    limits = []
    for device_id in selected:
        device = visible[device_id]
        if device.platform == "cpu":
            continue
        stats = device.memory_stats() or {}
        limit = stats.get("bytes_limit")
        if type(limit) is not int or limit <= 0:
            msg = (
                "No positive integer allocator byte limit reported for "
                f"device {device_id}."
            )
            raise ValueError(msg)
        limits.append(limit)
    return ExecutionConfig(
        devices=selected,
        device_memory_bytes=min(limits) if limits else None,
    )
