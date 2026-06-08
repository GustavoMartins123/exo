#!/usr/bin/env python3
"""Small HTTP signal agent for Exo host control.

The containerized agent never starts Exo directly. It only writes command files
to a shared host directory. A host-side systemd path/service consumes those
files and starts/stops Exo outside Docker.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from typing import Any

COMMANDS = {"start", "stop", "restart", "status", "pull"}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(value, dict):
        return value
    return {}


def _local_ip(probe_host: str | None) -> str:
    host = os.environ.get("EXO_AGENT_ADVERTISE_HOST")
    if host:
        return host
    target_host = probe_host or "127.0.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((target_host, 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


class NodeAgent:
    def __init__(
        self,
        shared_dir: Path,
        token: str | None,
        listen_port: int,
        probe_host: str | None,
    ) -> None:
        self.shared_dir = shared_dir
        self.commands_dir = shared_dir / "commands"
        self.status_path = shared_dir / "status.json"
        self.token = token
        self.listen_port = listen_port
        self.probe_host = probe_host
        self.node_name = os.environ.get("EXO_NODE_NAME") or socket.gethostname()
        self.commands_dir.mkdir(parents=True, exist_ok=True)

    def authorize(self, handler: BaseHTTPRequestHandler) -> bool:
        if not self.token:
            return True
        auth = handler.headers.get("Authorization", "")
        expected = f"Bearer {self.token}"
        if auth == expected:
            return True
        _json_response(handler, HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
        return False

    def status(self) -> dict[str, Any]:
        host_status = _read_json(self.status_path)
        return {
            "node": self.node_name,
            "agent": "online",
            "host": _local_ip(self.probe_host),
            "port": self.listen_port,
            "updated_at": int(time.time()),
            "host_status": host_status,
        }

    def enqueue(self, action: str) -> dict[str, Any]:
        if action not in COMMANDS:
            raise ValueError(f"unsupported command: {action}")
        command_id = f"{_now_ms()}-{action}"
        command_path = self.commands_dir / f"{command_id}.cmd"
        payload = {
            "id": command_id,
            "action": action,
            "created_at": int(time.time()),
            "source": "node-agent",
        }
        command_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return {"queued": True, "command": payload}


def _handler(agent: NodeAgent) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            print(f"{self.client_address[0]} - {fmt % args}", flush=True)

        def do_GET(self) -> None:
            if not agent.authorize(self):
                return
            if self.path in {"/health", "/"}:
                _json_response(self, HTTPStatus.OK, {"ok": True, "node": agent.node_name})
                return
            if self.path == "/status":
                _json_response(self, HTTPStatus.OK, agent.status())
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if not agent.authorize(self):
                return
            parts = [part for part in self.path.split("/") if part]
            if len(parts) == 2 and parts[0] == "exo" and parts[1] in COMMANDS:
                try:
                    payload = agent.enqueue(parts[1])
                except ValueError as exc:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                _json_response(self, HTTPStatus.ACCEPTED, payload)
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

    return Handler


def _post_json(url: str, payload: dict[str, Any], token: str | None) -> None:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=5) as response:
        response.read()


def _heartbeat(agent: NodeAgent, controller_url: str, stop_event: Event) -> None:
    register_url = controller_url.rstrip("/") + "/register"
    while not stop_event.wait(5):
        try:
            _post_json(register_url, agent.status(), agent.token)
        except (OSError, urllib.error.URLError):
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description="Exo node signal agent")
    parser.add_argument("--host", default=os.environ.get("EXO_AGENT_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("EXO_AGENT_PORT", "8765")))
    parser.add_argument(
        "--shared-dir",
        default=os.environ.get("EXO_AGENT_SHARED_DIR", "/var/lib/exo-agent"),
    )
    args = parser.parse_args()

    token = os.environ.get("EXO_AGENT_TOKEN") or None
    controller_url = os.environ.get("EXO_CONTROLLER_URL")
    probe_host = (
        urllib.parse.urlparse(controller_url).hostname if controller_url else None
    )
    agent = NodeAgent(Path(args.shared_dir), token, args.port, probe_host)
    stop_event = Event()
    heartbeat_thread: Thread | None = None
    if controller_url:
        heartbeat_thread = Thread(
            target=_heartbeat,
            args=(agent, controller_url, stop_event),
            daemon=True,
        )
        heartbeat_thread.start()

    server = ThreadingHTTPServer((args.host, args.port), _handler(agent))
    print(f"exo node agent listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
