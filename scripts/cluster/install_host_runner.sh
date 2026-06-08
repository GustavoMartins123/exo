#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "run with sudo: sudo $0" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_DIR="${EXO_AGENT_SHARED_DIR:-/var/lib/exo-agent}"

install -d -m 0755 "$SHARED_DIR/commands"
install -m 0755 "$SCRIPT_DIR/exo-agent-runner.sh" /usr/local/bin/exo-agent-runner.sh
install -m 0644 "$SCRIPT_DIR/systemd/exo-agent-runner.service" /etc/systemd/system/exo-agent-runner.service
install -m 0644 "$SCRIPT_DIR/systemd/exo-agent-runner.path" /etc/systemd/system/exo-agent-runner.path

systemctl daemon-reload
systemctl enable --now exo-agent-runner.path

echo "installed exo-agent host runner"
echo "shared dir: $SHARED_DIR"
echo "status: systemctl status exo-agent-runner.path"
