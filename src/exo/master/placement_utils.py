import os
from collections.abc import Generator, Mapping

from loguru import logger

from exo.shared.models.model_cards import ModelCard
from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage, NodeNetworkInfo
from exo.shared.types.topology import Cycle, RDMAConnection, SocketConnection
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    Sharding,
    ShardMetadata,
    TensorShardMetadata,
)

_PIPELINE_RESERVE_MIN_MODEL_SIZE = Memory.from_gb(
    float(os.environ.get("EXO_PIPELINE_RESERVE_MIN_MODEL_GB", "2"))
)
_PIPELINE_MIN_STATIC_RESERVE = Memory.from_gb(
    float(os.environ.get("EXO_PIPELINE_MIN_STATIC_RESERVE_GB", "2"))
)
_PIPELINE_STATIC_RESERVE_RATIO = float(
    os.environ.get("EXO_PIPELINE_STATIC_RESERVE_RATIO", "0.45")
)
_PIPELINE_SMALL_DEVICE_THRESHOLD = Memory.from_gb(
    float(os.environ.get("EXO_PIPELINE_SMALL_DEVICE_THRESHOLD_GB", "16"))
)
_PIPELINE_MID_DEVICE_THRESHOLD = Memory.from_gb(
    float(os.environ.get("EXO_PIPELINE_MID_DEVICE_THRESHOLD_GB", "32"))
)
_PIPELINE_SMALL_DEVICE_RESERVE_CAP_RATIO = float(
    os.environ.get("EXO_PIPELINE_SMALL_DEVICE_RESERVE_CAP_RATIO", "0.90")
)
_PIPELINE_MID_DEVICE_RESERVE_CAP_RATIO = float(
    os.environ.get("EXO_PIPELINE_MID_DEVICE_RESERVE_CAP_RATIO", "0.55")
)
_PIPELINE_LARGE_DEVICE_RESERVE_CAP_RATIO = float(
    os.environ.get("EXO_PIPELINE_LARGE_DEVICE_RESERVE_CAP_RATIO", "0.40")
)


def filter_cycles_by_memory(
    cycles: list[Cycle],
    node_memory: Mapping[NodeId, MemoryUsage],
    required_memory: Memory,
) -> list[Cycle]:
    filtered_cycles: list[Cycle] = []
    for cycle in cycles:
        if not all(node in node_memory for node in cycle):
            continue

        total_mem = sum(
            (node_memory[node_id].inference_available for node_id in cycle.node_ids),
            start=Memory(),
        )
        if total_mem >= required_memory:
            filtered_cycles.append(cycle)
    return filtered_cycles


def _pipeline_device_size(memory_usage: MemoryUsage) -> Memory:
    accelerator_total = sum(
        (device.total for device in memory_usage.accelerators),
        start=Memory(),
    )
    if accelerator_total.in_bytes > 0:
        return accelerator_total
    return memory_usage.ram_total


def _pipeline_reserve_cap_ratio(memory_usage: MemoryUsage) -> float:
    device_size = _pipeline_device_size(memory_usage)
    if device_size <= _PIPELINE_SMALL_DEVICE_THRESHOLD:
        return _PIPELINE_SMALL_DEVICE_RESERVE_CAP_RATIO
    if device_size <= _PIPELINE_MID_DEVICE_THRESHOLD:
        return _PIPELINE_MID_DEVICE_RESERVE_CAP_RATIO
    return _PIPELINE_LARGE_DEVICE_RESERVE_CAP_RATIO


def get_pipeline_static_reserve(
    model_card: ModelCard, memory_usage: MemoryUsage
) -> Memory:
    if model_card.storage_size < _PIPELINE_RESERVE_MIN_MODEL_SIZE:
        return Memory()
    model_scaled_reserve = model_card.storage_size * _PIPELINE_STATIC_RESERVE_RATIO
    desired_reserve = max(model_scaled_reserve, _PIPELINE_MIN_STATIC_RESERVE)
    available = memory_usage.inference_available
    reserve_cap = available * _pipeline_reserve_cap_ratio(memory_usage)
    return min(desired_reserve, reserve_cap)


def get_pipeline_layer_memory_budget(
    node_id: NodeId,
    node_memory: Mapping[NodeId, MemoryUsage],
    model_card: ModelCard,
) -> Memory:
    """Memory used as the *proportional allocation hint* for a node.

    The static reserve nudges layers off small GPUs toward big ones. It is
    capped by device class so very large BF16 models do not zero out mid-size
    GPUs such as a 24 GB A5000 while still keeping 12 GB cards conservative.

    The budget must not become 0 — `allocate_layers_proportionally` floors every
    node at one layer, and a 0-byte budget would later trip the downstream
    validator, fail the whole placement and leave the master in a retry loop.
    Always keep at least one layer's worth, capped by what the node actually
    has."""
    memory_usage = node_memory[node_id]
    available = memory_usage.inference_available
    reserve = get_pipeline_static_reserve(model_card, memory_usage)
    after_reserve = Memory.from_bytes(max(0, available.in_bytes - reserve.in_bytes))
    one_layer = model_card.storage_size // model_card.n_layers
    floor = min(available, one_layer)
    return max(after_reserve, floor)


def get_smallest_cycles(
    cycles: list[Cycle],
) -> list[Cycle]:
    min_nodes = min(len(cycle) for cycle in cycles)
    return [cycle for cycle in cycles if len(cycle) == min_nodes]


def allocate_layers_proportionally(
    total_layers: int,
    memory_fractions: list[float],
) -> list[int]:
    n = len(memory_fractions)
    if n == 0:
        raise ValueError("Cannot allocate layers to an empty node list")
    if total_layers < n:
        raise ValueError(
            f"Cannot distribute {total_layers} layers across {n} nodes "
            "(need at least 1 layer per node)"
        )

    # Largest remainder: floor each, then distribute remainder by fractional part
    raw = [f * total_layers for f in memory_fractions]
    result = [int(r) for r in raw]
    by_remainder = sorted(range(n), key=lambda i: raw[i] - result[i], reverse=True)
    for i in range(total_layers - sum(result)):
        result[by_remainder[i]] += 1

    # Ensure minimum 1 per node by taking from the largest
    for i in range(n):
        if result[i] == 0:
            max_idx = max(range(n), key=lambda j: result[j])
            assert result[max_idx] > 1
            result[max_idx] -= 1
            result[i] = 1

    return result


def _validate_cycle(cycle: Cycle) -> None:
    if not cycle.node_ids:
        raise ValueError("Cannot create shard assignments for empty node cycle")


def _compute_total_memory(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    model_card: ModelCard,
) -> Memory:
    total_memory = sum(
        (
            get_pipeline_layer_memory_budget(node_id, node_memory, model_card)
            for node_id in node_ids
        ),
        start=Memory(),
    )
    if total_memory.in_bytes == 0:
        raise ValueError("Cannot create shard assignments: total available memory is 0")
    return total_memory


def _cycle_orientations(cycle: Cycle) -> Generator[Cycle, None, None]:
    node_ids = cycle.node_ids
    for ids in (node_ids, list(reversed(node_ids))):
        for offset in range(len(ids)):
            yield Cycle(node_ids=ids[offset:] + ids[:offset])


def orient_cycle_for_pipeline_memory(
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> Cycle:
    """Rotate/reverse a cycle so pipeline endpoint ranks land on larger devices.

    Pipeline rank 0 and the last rank tend to carry extra dynamic memory pressure
    from embeddings, output projection, prefill buffers, and KV/cache plumbing. A
    cycle can be rotated or reversed without changing its ring adjacency, so use
    that freedom to avoid putting both endpoint ranks on small GPUs.
    """
    _validate_cycle(cycle)
    if len(cycle) <= 2:
        return cycle

    def score(candidate: Cycle) -> tuple[int, int, int]:
        first = node_memory[candidate.node_ids[0]].inference_available.in_bytes
        last = node_memory[candidate.node_ids[-1]].inference_available.in_bytes
        return (first + last, min(first, last), first)

    return max(_cycle_orientations(cycle), key=score)


def _allocate_and_validate_layers(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    total_memory: Memory,
    model_card: ModelCard,
) -> list[int]:
    layer_allocations = allocate_layers_proportionally(
        total_layers=model_card.n_layers,
        memory_fractions=[
            get_pipeline_layer_memory_budget(node_id, node_memory, model_card)
            / total_memory
            for node_id in node_ids
        ],
    )

    total_storage = model_card.storage_size
    total_layers = model_card.n_layers
    for i, node_id in enumerate(node_ids):
        node_layers = layer_allocations[i]
        required_memory = (total_storage * node_layers) // total_layers
        # Validate against actual available memory, not the placement budget.
        # The static reserve is a heuristic to push layers off small GPUs; if
        # we gated validation on it too, any node whose free VRAM happens to
        # drop below the reserve would crash placement entirely — even when
        # the layer would have fit fine in real memory.
        available_memory = node_memory[node_id].inference_available
        if required_memory > available_memory:
            raise ValueError(
                f"Node {i} ({node_id}) has insufficient memory: "
                f"requires {required_memory.in_gb:.2f} GB for {node_layers} layers, "
                f"but only has {available_memory.in_gb:.2f} GB available"
            )

    return layer_allocations


def get_shard_assignments_for_pipeline_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for pipeline parallel execution."""
    world_size = len(cycle)
    use_cfg_parallel = model_card.uses_cfg and world_size >= 2 and world_size % 2 == 0

    if use_cfg_parallel:
        return _get_shard_assignments_for_cfg_parallel(model_card, cycle, node_memory)
    else:
        return _get_shard_assignments_for_pure_pipeline(model_card, cycle, node_memory)


def _get_shard_assignments_for_cfg_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for CFG parallel execution.

    CFG parallel runs two independent pipelines. Group 0 processes the positive
    prompt, group 1 processes the negative prompt. The ring topology places
    group 1's ranks in reverse order so both "last stages" are neighbors for
    efficient CFG exchange.
    """
    _validate_cycle(cycle)

    world_size = len(cycle)
    cfg_world_size = 2
    pipeline_world_size = world_size // cfg_world_size

    # Allocate layers for one pipeline group (both groups run the same layers)
    pipeline_node_ids = cycle.node_ids[:pipeline_world_size]
    pipeline_memory = _compute_total_memory(pipeline_node_ids, node_memory, model_card)
    layer_allocations = _allocate_and_validate_layers(
        pipeline_node_ids, node_memory, pipeline_memory, model_card
    )

    # Ring topology: group 0 ascending [0,1,2,...], group 1 descending [...,2,1,0]
    # This places both last stages as neighbors for CFG exchange.
    position_to_cfg_pipeline = [(0, r) for r in range(pipeline_world_size)] + [
        (1, r) for r in reversed(range(pipeline_world_size))
    ]

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for device_rank, node_id in enumerate(cycle.node_ids):
        cfg_rank, pipeline_rank = position_to_cfg_pipeline[device_rank]
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = CfgShardMetadata(
            model_card=model_card,
            device_rank=device_rank,
            world_size=world_size,
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
            cfg_rank=cfg_rank,
            cfg_world_size=cfg_world_size,
            pipeline_rank=pipeline_rank,
            pipeline_world_size=pipeline_world_size,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def _get_shard_assignments_for_pure_pipeline(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for pure pipeline execution."""
    _validate_cycle(cycle)
    total_memory = _compute_total_memory(cycle.node_ids, node_memory, model_card)

    layer_allocations = _allocate_and_validate_layers(
        cycle.node_ids, node_memory, total_memory, model_card
    )

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for pipeline_rank, node_id in enumerate(cycle.node_ids):
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = PipelineShardMetadata(
            model_card=model_card,
            device_rank=pipeline_rank,
            world_size=len(cycle),
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def get_shard_assignments_for_tensor_parallel(
    model_card: ModelCard,
    cycle: Cycle,
):
    total_layers = model_card.n_layers
    world_size = len(cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for i, node_id in enumerate(cycle):
        shard = TensorShardMetadata(
            model_card=model_card,
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=total_layers,
            n_layers=total_layers,
        )

        runner_id = RunnerId()

        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    shard_assignments = ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments(
    model_card: ModelCard,
    cycle: Cycle,
    sharding: Sharding,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_card=model_card,
                cycle=cycle,
                node_memory=node_memory,
            )
        case Sharding.Tensor:
            return get_shard_assignments_for_tensor_parallel(
                model_card=model_card,
                cycle=cycle,
            )


def get_mlx_jaccl_devices_matrix(
    selected_cycle: list[NodeId],
    cycle_digraph: Topology,
) -> list[list[str | None]]:
    """Build connectivity matrix mapping device i to device j via RDMA interface names.

    The matrix element [i][j] contains the interface name on device i that connects
    to device j, or None if no connection exists or no interface name is found.
    Diagonal elements are always None.
    """
    num_nodes = len(selected_cycle)
    matrix: list[list[str | None]] = [
        [None for _ in range(num_nodes)] for _ in range(num_nodes)
    ]

    for i, node_i in enumerate(selected_cycle):
        for j, node_j in enumerate(selected_cycle):
            if i == j:
                continue

            for conn in cycle_digraph.get_all_connections_between(node_i, node_j):
                if isinstance(conn, RDMAConnection):
                    matrix[i][j] = conn.source_rdma_iface
                    break
            else:
                raise ValueError(
                    "Current jaccl backend requires all-to-all RDMA connections"
                )

    return matrix


def _find_connection_ip(
    node_i: NodeId,
    node_j: NodeId,
    cycle_digraph: Topology,
) -> Generator[str, None, None]:
    """Find all IP addresses that connect node i to node j."""
    for connection in cycle_digraph.get_all_connections_between(node_i, node_j):
        if isinstance(connection, SocketConnection):
            yield connection.sink_multiaddr.ip_address


def find_ip_prioritised(
    node_id: NodeId,
    other_node_id: NodeId,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    ring: bool,
) -> str | None:
    """Find an IP address between nodes with prioritization.

    Priority: ethernet > wifi > unknown > thunderbolt
    """
    ips = list(_find_connection_ip(node_id, other_node_id, cycle_digraph))
    if not ips:
        return None
    other_network = node_network.get(other_node_id, NodeNetworkInfo())
    ip_to_type = {
        iface.ip_address: iface.interface_type for iface in other_network.interfaces
    }

    # Ring should prioritise fastest connection. As a best-effort, we prioritise TB.
    # TODO: Profile and get actual connection speeds.
    if ring:
        priority = {
            "thunderbolt": 0,
            "maybe_ethernet": 1,
            "ethernet": 2,
            "wifi": 3,
            "unknown": 4,
        }

    # RDMA prefers ethernet coordinator
    else:
        priority = {
            "ethernet": 0,
            "wifi": 1,
            "unknown": 2,
            "maybe_ethernet": 3,
            "thunderbolt": 4,
        }
    return min(ips, key=lambda ip: priority.get(ip_to_type.get(ip, "unknown"), 2))


def get_mlx_ring_hosts_by_node(
    selected_cycle: Cycle,
    cycle_digraph: Topology,
    ephemeral_port: int,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, list[Host]]:
    """Generate per-node host lists for MLX ring backend.

    Each node gets a list where:
    - Self position: Host(ip="0.0.0.0", port=ephemeral_port)
    - Left/right neighbors: actual connection IPs
    - Non-neighbors: Host(ip="198.51.100.1", port=0) placeholder (RFC 5737 TEST-NET-2)
    """
    world_size = len(selected_cycle)
    if world_size == 0:
        return {}

    hosts_by_node: dict[NodeId, list[Host]] = {}

    for rank, node_id in enumerate(selected_cycle):
        left_rank = (rank - 1) % world_size
        right_rank = (rank + 1) % world_size

        hosts_for_node: list[Host] = []

        for idx, other_node_id in enumerate(selected_cycle):
            if idx == rank:
                hosts_for_node.append(Host(ip="0.0.0.0", port=ephemeral_port))
                continue

            if idx not in {left_rank, right_rank}:
                # Placeholder IP from RFC 5737 TEST-NET-2
                hosts_for_node.append(Host(ip="198.51.100.1", port=0))
                continue

            connection_ip = find_ip_prioritised(
                node_id, other_node_id, cycle_digraph, node_network, ring=True
            )
            if connection_ip is None:
                raise ValueError(
                    "MLX ring backend requires connectivity between neighbouring nodes"
                )

            hosts_for_node.append(Host(ip=connection_ip, port=ephemeral_port))

        hosts_by_node[node_id] = hosts_for_node

    return hosts_by_node


def get_mlx_jaccl_coordinators(
    coordinator: NodeId,
    coordinator_port: int,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, str]:
    """Get the coordinator addresses for MLX JACCL (rank 0 device).

    Select an IP address that each node can reach for the rank 0 node. Returns
    address in format "X.X.X.X:PORT" per node.
    """
    logger.debug(f"Selecting coordinator: {coordinator}")

    def get_ip_for_node(n: NodeId) -> str:
        if n == coordinator:
            return "0.0.0.0"

        ip = find_ip_prioritised(
            n, coordinator, cycle_digraph, node_network, ring=False
        )
        if ip is not None:
            return ip

        raise ValueError(
            "Current jaccl backend requires all participating devices to be able to communicate"
        )

    return {
        n: f"{get_ip_for_node(n)}:{coordinator_port}"
        for n in cycle_digraph.list_nodes()
    }
