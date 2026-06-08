#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SHARED_DIR="${EXO_AGENT_SHARED_DIR:-$HOME/.local/share/exo-agent}"
PLIST_NAME="com.exo.agent-runner.plist"
PLIST_SOURCE="$SCRIPT_DIR/launchd/$PLIST_NAME"
PLIST_TARGET="$HOME/Library/LaunchAgents/$PLIST_NAME"

install -d -m 0755 "$SHARED_DIR/commands"
install -m 0755 "$SCRIPT_DIR/exo-agent-runner.sh" /usr/local/bin/exo-agent-runner.sh
install -d -m 0755 "$HOME/Library/LaunchAgents"

sed -e "s#/var/lib/exo-agent#$SHARED_DIR#g" "$PLIST_SOURCE" >"$PLIST_TARGET"

launchctl unload "$PLIST_TARGET" >/dev/null 2>&1 || true
launchctl load "$PLIST_TARGET"

echo "installed exo-agent launchd runner"
echo "shared dir: $SHARED_DIR"
echo "plist: $PLIST_TARGET"
echo "status: launchctl list | grep com.exo.agent-runner"
