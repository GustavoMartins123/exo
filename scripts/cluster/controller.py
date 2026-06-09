#!/usr/bin/env python3
"""Minimal Exo cluster controller.

Node agents register via POST /register. The controller can then fan out
commands to all registered nodes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

COMMANDS = {"start", "stop", "restart", "status", "pull"}


def _json_response(
    handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]
) -> None:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _read_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    raw_length = handler.headers.get("Content-Length", "0")
    try:
        length = int(raw_length)
    except ValueError:
        length = 0
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        value = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(value, dict):
        return value
    return {}


def _post_json(url: str, token: str | None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=10) as response:
        raw = response.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


class Controller:
    def __init__(self, token: str | None) -> None:
        self.token = token
        self.nodes: dict[str, dict[str, Any]] = {}

    def authorize(self, handler: BaseHTTPRequestHandler) -> bool:
        if not self.token:
            return True
        auth = handler.headers.get("Authorization", "")
        expected = f"Bearer {self.token}"
        if auth == expected:
            return True
        _json_response(handler, HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
        return False

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        node = str(payload.get("node") or payload.get("host") or "unknown")
        host = str(payload.get("host") or "")
        port = int(payload.get("port") or 8765)
        if not host:
            raise ValueError("node registration missing host")
        payload["registered_at"] = int(time.time())
        payload["agent_url"] = f"http://{host}:{port}"
        self.nodes[node] = payload
        return payload

    def dispatch(self, node_name: str, action: str) -> dict[str, Any]:
        if action not in COMMANDS:
            raise ValueError(f"unsupported command: {action}")
        node = self.nodes.get(node_name)
        if node is None:
            raise KeyError(node_name)
        url = str(node["agent_url"]).rstrip("/") + f"/exo/{action}"
        try:
            response = _post_json(url, self.token)
        except (OSError, urllib.error.URLError) as exc:
            return {"node": node_name, "ok": False, "error": str(exc)}
        return {"node": node_name, "ok": True, "response": response}

    def dispatch_all(self, action: str) -> list[dict[str, Any]]:
        return [self.dispatch(node_name, action) for node_name in sorted(self.nodes)]


def _handler(controller: Controller) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            print(f"{self.client_address[0]} - {fmt % args}", flush=True)

        def do_GET(self) -> None:
            if not controller.authorize(self):
                return
            if self.path in {"/health", "/"}:
                _json_response(self, HTTPStatus.OK, {"ok": True})
                return
            if self.path == "/nodes":
                _json_response(self, HTTPStatus.OK, {"nodes": controller.nodes})
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if not controller.authorize(self):
                return
            if self.path == "/register":
                try:
                    node = controller.register(_read_body(self))
                except ValueError as exc:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                _json_response(self, HTTPStatus.OK, {"registered": node})
                return

            parts = [part for part in self.path.split("/") if part]
            if len(parts) == 3 and parts[0] == "nodes" and parts[2] in COMMANDS:
                node_name = parts[1]
                if node_name == "all":
                    result = controller.dispatch_all(parts[2])
                    _json_response(self, HTTPStatus.ACCEPTED, {"results": result})
                    return
                try:
                    result = controller.dispatch(node_name, parts[2])
                except KeyError:
                    _json_response(
                        self, HTTPStatus.NOT_FOUND, {"error": "node not found"}
                    )
                    return
                except ValueError as exc:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                _json_response(self, HTTPStatus.ACCEPTED, result)
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Exo cluster controller")
    parser.add_argument(
        "--host", default=os.environ.get("EXO_CONTROLLER_HOST", "0.0.0.0")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("EXO_CONTROLLER_PORT", "8766"))
    )
    args = parser.parse_args()

    controller = Controller(os.environ.get("EXO_AGENT_TOKEN") or None)
    server = ThreadingHTTPServer((args.host, args.port), _handler(controller))
    print(f"exo cluster controller listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
