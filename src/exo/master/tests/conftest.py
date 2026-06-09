from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    MemoryDeviceKind,
    MemoryDeviceUsage,
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
)
from exo.shared.types.topology import RDMAConnection, SocketConnection


def create_node_memory(memory: int) -> MemoryUsage:
    return MemoryUsage.from_bytes(
        ram_total=1000,
        ram_available=memory,
        swap_total=1000,
        swap_available=1000,
    )


def create_node_accelerator_memory(
    memory: int, kind: MemoryDeviceKind = "cuda_vram"
) -> MemoryUsage:
    return MemoryUsage.from_bytes(
        ram_total=1000,
        ram_available=1000,
        swap_total=1000,
        swap_available=1000,
        accelerators=(
            MemoryDeviceUsage.from_bytes(
                name="test accelerator",
                kind=kind,
                total=memory,
                available=memory,
            ),
        ),
    )


def create_node_network() -> NodeNetworkInfo:
    return NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address=f"169.254.0.{i}")
            for i in range(10)
        ]
    )


def create_socket_connection(ip: int, sink_port: int = 1234) -> SocketConnection:
    return SocketConnection(
        sink_multiaddr=Multiaddr(address=f"/ip4/169.254.0.{ip}/tcp/{sink_port}"),
    )


def create_rdma_connection(iface: int) -> RDMAConnection:
    return RDMAConnection(
        source_rdma_iface=f"rdma_en{iface}", sink_rdma_iface=f"rdma_en{iface}"
    )
