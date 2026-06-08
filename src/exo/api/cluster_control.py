from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

import httpx
import psutil
from loguru import logger
from pydantic import BaseModel, Field

ClusterAction = Literal["start", "stop", "restart", "pull", "status"]

DEFAULT_AGENT_PORT = 8765
DEFAULT_SCAN_TIMEOUT_SECONDS = 0.35
DEFAULT_COMMAND_TIMEOUT_SECONDS = 8.0
DEFAULT_MAX_DISCOVERY_HOSTS = 512


class ClusterConfig(BaseModel):
    is_master: bool
    master_host: str | None = None
    agent_port: int = DEFAULT_AGENT_PORT
    cidrs: list[str] = Field(default_factory=list)
    static_hosts: list[str] = Field(default_factory=list)
    token_configured: bool = False


class ClusterAgentStatus(BaseModel):
    node: str
    host: str
    port: int
    online: bool
    url: str
    is_self: bool = False
    error: str | None = None
    host_status: dict[str, Any] = Field(default_factory=dict)


class ClusterCommandResult(BaseModel):
    node: str
    host: str
    url: str
    ok: bool
    response: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ClusterChildrenStartResponse(BaseModel):
    is_master: bool
    discovered: list[ClusterAgentStatus]
    results: list[ClusterCommandResult]


def load_cluster_env() -> None:
    env_file = os.environ.get("EXO_CLUSTER_ENV_FILE", ".env")
    path = Path(env_file)
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning(f"failed to read cluster env file {path}: {exc}")
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on", "master"}


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _hostname_candidates() -> set[str]:
    candidates = {socket.gethostname()}
    with suppress(OSError):
        candidates.add(socket.getfqdn())
    for address in _local_ipv4_addresses():
        candidates.add(address)
    return {candidate.lower() for candidate in candidates if candidate}


def _local_ipv4_addresses() -> list[str]:
    addresses: list[str] = []
    for interface_addresses in psutil.net_if_addrs().values():
        for address in interface_addresses:
            if address.family != socket.AF_INET:
                continue
            ip = address.address
            if not ip or ip.startswith("127."):
                continue
            addresses.append(ip)
    return addresses


def _local_ipv4_networks() -> list[ipaddress.IPv4Network]:
    networks: list[ipaddress.IPv4Network] = []
    for interface_addresses in psutil.net_if_addrs().values():
        for address in interface_addresses:
            if address.family != socket.AF_INET:
                continue
            if not address.address or address.address.startswith("127."):
                continue
            if not address.netmask:
                continue
            try:
                network = ipaddress.ip_network(
                    f"{address.address}/{address.netmask}",
                    strict=False,
                )
            except ValueError:
                continue
            if network.version == 4:
                networks.append(network)
    return networks


def cluster_config() -> ClusterConfig:
    load_cluster_env()
    master_host = os.environ.get("EXO_CLUSTER_MASTER_HOST") or None
    explicit_master = _truthy(os.environ.get("EXO_CLUSTER_MASTER")) or (
        os.environ.get("EXO_CLUSTER_ROLE", "").lower() == "master"
    )
    is_master = explicit_master
    if master_host:
        is_master = master_host.lower() in _hostname_candidates()

    port = int(os.environ.get("EXO_CLUSTER_AGENT_PORT", str(DEFAULT_AGENT_PORT)))
    return ClusterConfig(
        is_master=is_master,
        master_host=master_host,
        agent_port=port,
        cidrs=_split_csv(os.environ.get("EXO_CLUSTER_AGENT_CIDRS")),
        static_hosts=_split_csv(os.environ.get("EXO_CLUSTER_AGENT_HOSTS")),
        token_configured=bool(os.environ.get("EXO_AGENT_TOKEN")),
    )


def _configured_scan_networks(config: ClusterConfig) -> list[ipaddress.IPv4Network]:
    networks: list[ipaddress.IPv4Network] = []
    for raw_cidr in config.cidrs:
        try:
            network = ipaddress.ip_network(raw_cidr, strict=False)
        except ValueError:
            logger.warning(f"ignoring invalid EXO_CLUSTER_AGENT_CIDRS entry: {raw_cidr}")
            continue
        if network.version == 4:
            networks.append(network)
    if networks:
        return networks
    return _local_ipv4_networks()


def _candidate_hosts(config: ClusterConfig) -> list[str]:
    hosts = list(config.static_hosts)
    max_hosts = int(
        os.environ.get("EXO_CLUSTER_DISCOVERY_MAX_HOSTS", str(DEFAULT_MAX_DISCOVERY_HOSTS))
    )
    for network in _configured_scan_networks(config):
        for scan_network in _bounded_scan_networks(network):
            hosts.extend(str(ip) for ip in scan_network.hosts())
            if len(hosts) >= max_hosts:
                break
        if len(hosts) >= max_hosts:
            break
    deduped: list[str] = []
    seen: set[str] = set()
    for host in hosts[:max_hosts]:
        if host in seen:
            continue
        seen.add(host)
        deduped.append(host)
    return deduped


def _bounded_scan_networks(
    network: ipaddress.IPv4Network,
) -> list[ipaddress.IPv4Network]:
    if network.num_addresses <= 256:
        return [network]

    local_subnets: list[ipaddress.IPv4Network] = []
    for address in _local_ipv4_addresses():
        ip = ipaddress.ip_address(address)
        if ip not in network:
            continue
        local_subnets.append(ipaddress.ip_network(f"{ip}/24", strict=False))

    if local_subnets:
        return sorted(set(local_subnets), key=str)
    return [ipaddress.ip_network(f"{network.network_address}/24", strict=False)]


async def _get_agent_status(
    client: httpx.AsyncClient,
    host: str,
    port: int,
    token: str | None,
) -> ClusterAgentStatus | None:
    url = f"http://{host}:{port}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = await client.get(f"{url}/status", headers=headers)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    node = str(payload.get("node") or host)
    advertised_host = str(payload.get("host") or host)
    advertised_port = int(payload.get("port") or port)
    return ClusterAgentStatus(
        node=node,
        host=advertised_host,
        port=advertised_port,
        online=True,
        url=f"http://{advertised_host}:{advertised_port}",
        is_self=advertised_host in _local_ipv4_addresses()
        or node.lower() in _hostname_candidates(),
        host_status=payload.get("host_status") if isinstance(payload.get("host_status"), dict) else {},
    )


async def discover_cluster_agents() -> list[ClusterAgentStatus]:
    config = cluster_config()
    token = os.environ.get("EXO_AGENT_TOKEN") or None
    hosts = _candidate_hosts(config)
    timeout = float(
        os.environ.get("EXO_CLUSTER_DISCOVERY_TIMEOUT", str(DEFAULT_SCAN_TIMEOUT_SECONDS))
    )
    limits = httpx.Limits(max_connections=128)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = [
            _get_agent_status(client, host, config.agent_port, token)
            for host in hosts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    agents = [
        result
        for result in results
        if isinstance(result, ClusterAgentStatus) and result.online
    ]
    return sorted(agents, key=lambda agent: (agent.is_self, agent.node, agent.host))


async def dispatch_cluster_action(
    agents: Iterable[ClusterAgentStatus],
    action: ClusterAction,
) -> list[ClusterCommandResult]:
    token = os.environ.get("EXO_AGENT_TOKEN") or None
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    timeout = float(
        os.environ.get("EXO_CLUSTER_COMMAND_TIMEOUT", str(DEFAULT_COMMAND_TIMEOUT_SECONDS))
    )

    async def send(agent: ClusterAgentStatus) -> ClusterCommandResult:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{agent.url}/exo/{action}",
                    headers=headers,
                    json={},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            return ClusterCommandResult(
                node=agent.node,
                host=agent.host,
                url=agent.url,
                ok=False,
                error=str(exc),
            )
        return ClusterCommandResult(
            node=agent.node,
            host=agent.host,
            url=agent.url,
            ok=True,
            response=payload if isinstance(payload, dict) else {},
        )

    return list(await asyncio.gather(*(send(agent) for agent in agents)))


async def start_cluster_children() -> ClusterChildrenStartResponse:
    config = cluster_config()
    agents = await discover_cluster_agents()
    children = [agent for agent in agents if not agent.is_self]
    if not config.is_master:
        return ClusterChildrenStartResponse(
            is_master=False,
            discovered=agents,
            results=[],
        )
    results = await dispatch_cluster_action(children, "start")
    return ClusterChildrenStartResponse(
        is_master=True,
        discovered=agents,
        results=results,
    )
