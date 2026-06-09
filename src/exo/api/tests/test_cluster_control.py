import os
from pathlib import Path

import pytest

from exo.api import cluster_control
from exo.api.cluster_control import ClusterAgentStatus


def _clear_cluster_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("EXO_CLUSTER_") or key == "EXO_AGENT_TOKEN":
            monkeypatch.delenv(key, raising=False)


def test_cluster_config_reads_master_from_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cluster_env(monkeypatch)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "EXO_CLUSTER_MASTER=true",
                "EXO_CLUSTER_AGENT_CIDRS=10.10.10.0/24",
                "EXO_CLUSTER_AGENT_PORT=8765",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EXO_CLUSTER_ENV_FILE", str(env_path))

    config = cluster_control.cluster_config()

    assert config.is_master is True
    assert config.cidrs == ["10.10.10.0/24"]
    assert config.agent_port == 8765


def test_cluster_config_uses_master_host_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cluster_env(monkeypatch)
    monkeypatch.setenv("EXO_CLUSTER_MASTER_HOST", "10.10.10.1")
    monkeypatch.setattr(cluster_control, "_hostname_candidates", lambda: {"10.10.10.1"})

    assert cluster_control.cluster_config().is_master is True


def test_candidate_hosts_scans_local_24_inside_large_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cluster_env(monkeypatch)
    monkeypatch.setenv("EXO_CLUSTER_AGENT_CIDRS", "10.10.0.0/16")
    monkeypatch.setattr(
        cluster_control, "_local_ipv4_addresses", lambda: ["10.10.10.1"]
    )
    monkeypatch.setenv("EXO_CLUSTER_DISCOVERY_MAX_HOSTS", "300")

    hosts = cluster_control._candidate_hosts(cluster_control.cluster_config())

    assert "10.10.10.2" in hosts
    assert "10.10.0.2" not in hosts


@pytest.mark.anyio
async def test_start_children_does_not_dispatch_when_not_master(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cluster_env(monkeypatch)
    monkeypatch.setenv("EXO_CLUSTER_MASTER", "false")

    async def fake_discover() -> list[ClusterAgentStatus]:
        return [
            ClusterAgentStatus(
                node="child",
                host="10.10.10.2",
                port=8765,
                online=True,
                url="http://10.10.10.2:8765",
            )
        ]

    async def fail_dispatch(*_args: object, **_kwargs: object) -> list[object]:
        raise AssertionError("dispatch should not run for non-master")

    monkeypatch.setattr(cluster_control, "discover_cluster_agents", fake_discover)
    monkeypatch.setattr(cluster_control, "dispatch_cluster_action", fail_dispatch)

    result = await cluster_control.start_cluster_children()

    assert result.is_master is False
    assert len(result.discovered) == 1
    assert result.results == []
