#!/usr/bin/env bash
set -euo pipefail

SHARED_DIR="${EXO_AGENT_SHARED_DIR:-/var/lib/exo-agent}"
COMMAND_DIR="$SHARED_DIR/commands"
STATUS_FILE="$SHARED_DIR/status.json"
EXO_DIR="${EXO_DIR:-$HOME/exo}"
SESSION_NAME="${EXO_SESSION_NAME:-exo}"
START_SCRIPT="${EXO_START_SCRIPT:-$EXO_DIR/scripts/start_exo_detached.sh}"
LOG_FILE="${EXO_LOG_CAPTURE_DIR:-$HOME/.cache/exo}/exo.detached.log"

mkdir -p "$COMMAND_DIR"

json_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  printf '%s' "$value"
}

exo_pid() {
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux list-panes -t "$SESSION_NAME" -F '#{pane_pid}' 2>/dev/null | head -n1
    return 0
  fi

  local pid_file="${EXO_LOG_CAPTURE_DIR:-$HOME/.cache/exo}/exo.detached.pid"
  if [ -f "$pid_file" ]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      printf '%s\n' "$pid"
      return 0
    fi
  fi

  pgrep -f "uv run --extra .* exo" | head -n1 || true
}

exo_running() {
  local pid
  pid="$(exo_pid)"
  [ -n "$pid" ]
}

write_status() {
  local running="false"
  local pid=""
  local git_commit=""
  local ip_addresses=""
  local gpu_summary=""
  local last_log_line=""

  pid="$(exo_pid)"
  if [ -n "$pid" ]; then
    running="true"
  fi
  if [ -d "$EXO_DIR/.git" ]; then
    git_commit="$(git -C "$EXO_DIR" rev-parse --short HEAD 2>/dev/null || true)"
  fi
  ip_addresses="$(hostname -I 2>/dev/null | xargs || true)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_output
    if gpu_output="$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null)"; then
      gpu_summary="$(printf '%s\n' "$gpu_output" | paste -sd ';' -)"
    fi
  fi
  if [ -f "$LOG_FILE" ]; then
    last_log_line="$(tail -n 1 "$LOG_FILE" 2>/dev/null || true)"
  fi

  cat >"$STATUS_FILE" <<EOF
{"exo_running":$running,"pid":"$(json_escape "$pid")","git_commit":"$(json_escape "$git_commit")","exo_dir":"$(json_escape "$EXO_DIR")","ip_addresses":"$(json_escape "$ip_addresses")","gpu_summary":"$(json_escape "$gpu_summary")","last_log_line":"$(json_escape "$last_log_line")","updated_at":$(date +%s)}
EOF
}

start_exo() {
  if [ ! -x "$START_SCRIPT" ]; then
    echo "start script not executable: $START_SCRIPT" >&2
    return 1
  fi
  "$START_SCRIPT"
}

stop_exo() {
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$SESSION_NAME"
  fi

  local pid
  pid="$(exo_pid)"
  if [ -n "$pid" ]; then
    kill "$pid" 2>/dev/null || true
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  fi
}

pull_repo() {
  git -C "$EXO_DIR" pull --ff-only
}

run_action() {
  local action="$1"
  case "$action" in
    start)
      start_exo
      ;;
    stop)
      stop_exo
      ;;
    restart)
      stop_exo
      start_exo
      ;;
    status)
      :
      ;;
    pull)
      pull_repo
      ;;
    *)
      echo "unsupported action: $action" >&2
      return 1
      ;;
  esac
  write_status
}

command_action() {
  local command_file="$1"
  python3 - "$command_file" <<'PY'
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
action = payload.get("action", "")
print(action if isinstance(action, str) else "")
PY
}

drain_commands() {
  shopt -s nullglob
  local command_file
  for command_file in "$COMMAND_DIR"/*.cmd; do
    local action
    action="$(command_action "$command_file")"
    if [ -z "$action" ]; then
      mv "$command_file" "$command_file.bad"
      continue
    fi
    if run_action "$action"; then
      rm -f "$command_file"
    else
      mv "$command_file" "$command_file.failed"
    fi
  done
  write_status
}

main() {
  local action="${1:-drain}"
  case "$action" in
    drain)
      drain_commands
      ;;
    start|stop|restart|status|pull)
      run_action "$action"
      ;;
    *)
      echo "usage: $0 [drain|start|stop|restart|status|pull]" >&2
      return 2
      ;;
  esac
}

main "$@"
