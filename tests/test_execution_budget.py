"""Allocator observations determine the execution ceiling on selected devices."""

from dataclasses import dataclass

import jax
import pytest

from aca_model import execution


@dataclass
class _Device:
    id: int
    platform: str
    stats: dict[str, int] | None

    def memory_stats(self):
        return self.stats


def test_budget_uses_smallest_selected_allocator_limit(monkeypatch):
    """Only selected GPU pools bound execution; current usage is not subtracted."""
    devices = [
        _Device(0, "gpu", {"bytes_limit": 1000, "bytes_in_use": 200}),
        _Device(1, "gpu", {"bytes_limit": 800, "bytes_in_use": 100}),
        _Device(2, "gpu", None),
    ]
    monkeypatch.setattr(jax, "devices", lambda: devices)

    config = execution.execution_config_for_devices(devices=(0, 1))

    assert config.devices == (0, 1)
    assert config.device_memory_bytes == 800
    assert config.axis_widths == {}
    assert config.sharded_states == ()


@pytest.mark.parametrize("stats", [None, {}, {"bytes_limit": 0}])
def test_missing_gpu_limit_is_refused(monkeypatch, stats):
    """An unknown selected GPU ceiling must not silently select bootstrap widths."""
    monkeypatch.setattr(jax, "devices", lambda: [_Device(0, "gpu", stats)])

    with pytest.raises(ValueError, match=r"allocator.*limit.*device 0"):
        execution.execution_config_for_devices()


def test_cpu_construction_does_not_invent_an_allocator_budget(monkeypatch):
    """CPU devices retain an unspecified byte budget when no pool limit exists."""
    monkeypatch.setattr(jax, "devices", lambda: [_Device(0, "cpu", None)])

    config = execution.execution_config_for_devices()

    assert config.devices == (0,)
    assert config.device_memory_bytes is None


def test_selected_device_must_be_visible(monkeypatch):
    """An unavailable selected device is rejected before reading allocator stats."""
    monkeypatch.setattr(jax, "devices", lambda: [_Device(0, "gpu", None)])

    with pytest.raises(ValueError, match="Selected devices are not visible"):
        execution.execution_config_for_devices(devices=(1,))
