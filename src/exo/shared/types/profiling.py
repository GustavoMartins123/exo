import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol, Self, cast

import psutil

from exo.shared.types.memory import Memory
from exo.shared.types.thunderbolt import ThunderboltIdentifier
from exo.utils.pydantic_ext import FrozenModel

MemoryDeviceKind = Literal["cuda_vram", "apple_unified", "system_ram"]


class _NvmlMemoryInfo(Protocol):
    total: int
    free: int


class MemoryDeviceUsage(FrozenModel):
    name: str
    kind: MemoryDeviceKind
    total: Memory
    available: Memory
    used: Memory
    used_percent: float
    index: int | None = None

    @classmethod
    def from_bytes(
        cls,
        *,
        name: str,
        kind: MemoryDeviceKind,
        total: int,
        available: int,
        index: int | None = None,
    ) -> Self:
        used = max(total - available, 0)
        used_percent = (used / total * 100.0) if total > 0 else 0.0
        return cls(
            name=name,
            kind=kind,
            total=Memory.from_bytes(total),
            available=Memory.from_bytes(available),
            used=Memory.from_bytes(used),
            used_percent=used_percent,
            index=index,
        )


def _decode_device_name(name: object) -> str:
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="replace")
    if isinstance(name, str):
        return name
    return str(name)


def _gather_cuda_memory_devices() -> Sequence[MemoryDeviceUsage]:
    try:
        import pynvml as nvml  # pyright: ignore[reportMissingModuleSource]
    except ImportError:
        return ()

    try:
        nvml.nvmlInit()
        try:
            devices: list[MemoryDeviceUsage] = []
            for index in range(nvml.nvmlDeviceGetCount()):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                info = cast(
                    _NvmlMemoryInfo,
                    nvml.nvmlDeviceGetMemoryInfo(  # pyright: ignore[reportUnknownMemberType]
                        handle
                    ),
                )
                device_name = cast(
                    object,
                    nvml.nvmlDeviceGetName(  # pyright: ignore[reportUnknownMemberType]
                        handle
                    ),
                )
                devices.append(
                    MemoryDeviceUsage.from_bytes(
                        name=_decode_device_name(device_name),
                        kind="cuda_vram",
                        total=int(info.total),
                        available=int(info.free),
                        index=index,
                    )
                )
            return tuple(devices)
        finally:
            nvml.nvmlShutdown()
    except Exception:
        return ()


def _gather_platform_memory_devices(
    *, ram_total: int, ram_available: int
) -> Sequence[MemoryDeviceUsage]:
    cuda_devices = _gather_cuda_memory_devices()
    if cuda_devices:
        return cuda_devices

    if sys.platform == "darwin":
        return (
            MemoryDeviceUsage.from_bytes(
                name="Apple Unified Memory",
                kind="apple_unified",
                total=ram_total,
                available=ram_available,
            ),
        )

    return ()


class MemoryUsage(FrozenModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory
    accelerators: Sequence[MemoryDeviceUsage] = ()

    @property
    def accelerator_available(self) -> Memory:
        return sum((device.available for device in self.accelerators), start=Memory())

    @property
    def inference_available(self) -> Memory:
        accelerator_available = self.accelerator_available
        if accelerator_available.in_bytes > 0:
            return accelerator_available
        return self.ram_available

    @classmethod
    def from_bytes(
        cls,
        *,
        ram_total: int,
        ram_available: int,
        swap_total: int,
        swap_available: int,
        accelerators: Sequence[MemoryDeviceUsage] = (),
    ) -> Self:
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
            accelerators=accelerators,
        )

    @classmethod
    def from_psutil(cls, *, override_memory: int | None) -> Self:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        return cls.from_bytes(
            ram_total=vm.total,
            ram_available=vm.available if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
            accelerators=_gather_platform_memory_devices(
                ram_total=vm.total,
                ram_available=vm.available if override_memory is None else override_memory,
            ),
        )


class DiskUsage(FrozenModel):
    """Disk space usage for the models directory."""

    total: Memory
    available: Memory

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Get disk usage stats for the partition containing path."""
        total, _used, free = shutil.disk_usage(path)
        return cls(
            total=Memory.from_bytes(total),
            available=Memory.from_bytes(free),
        )


class SystemPerformanceProfile(FrozenModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0


InterfaceType = Literal["wifi", "ethernet", "maybe_ethernet", "thunderbolt", "unknown"]


class NetworkInterfaceInfo(FrozenModel):
    name: str
    ip_address: str
    interface_type: InterfaceType = "unknown"


class NodeIdentity(FrozenModel):
    """Static and slow-changing node identification data."""

    model_id: str = "Unknown"
    chip_id: str = "Unknown"
    friendly_name: str = "Unknown"
    os_version: str = "Unknown"
    os_build_version: str = "Unknown"


class NodeNetworkInfo(FrozenModel):
    """Network interface information for a node."""

    interfaces: Sequence[NetworkInterfaceInfo] = []


class NodeThunderboltInfo(FrozenModel):
    """Thunderbolt interface identifiers for a node."""

    interfaces: Sequence[ThunderboltIdentifier] = []


class NodeRdmaCtlStatus(FrozenModel):
    """Whether RDMA is enabled on this node (via rdma_ctl)."""

    enabled: bool


class ThunderboltBridgeStatus(FrozenModel):
    """Whether the Thunderbolt Bridge network service is enabled on this node."""

    enabled: bool
    exists: bool
    service_name: str | None = None
