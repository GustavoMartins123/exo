#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${EXO_SESSION_NAME:-exo}"
CLUSTER_NAMESPACE="${EXO_LIBP2P_NAMESPACE:-my-cluster}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_EXO_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
EXO_DIR="${EXO_DIR:-$DEFAULT_EXO_DIR}"
if [ -z "${EXO_EXTRA:-}" ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
    EXO_EXTRA="mlx"
  else
    EXO_EXTRA="mlx-cuda13"
  fi
fi
EXO_ARGS="${EXO_ARGS:--v}"
LOG_DIR="${EXO_LOG_CAPTURE_DIR:-$HOME/.cache/exo}"
LOG_FILE="$LOG_DIR/exo.detached.log"

mkdir -p "$LOG_DIR"

if [ ! -d "$EXO_DIR" ]; then
  echo "exo directory not found: $EXO_DIR" >&2
  exit 1
fi

SHELL_INIT="source '$HOME/.bashrc' >/dev/null 2>&1 || true"
if [ "$(uname -s)" = "Darwin" ]; then
  SHELL_INIT="source '$HOME/.zprofile' >/dev/null 2>&1 || true; source '$HOME/.zshrc' >/dev/null 2>&1 || true; source '$HOME/.bash_profile' >/dev/null 2>&1 || true; source '$HOME/.bashrc' >/dev/null 2>&1 || true"
fi

COMMAND="cd '$EXO_DIR' && \
$SHELL_INIT && \
PY_NVIDIA_LIBS=\$(find '$EXO_DIR/.venv/lib' -path '*/site-packages/nvidia/*/lib' -type d 2>/dev/null | paste -sd: -) && \
if [ -n \"\$PY_NVIDIA_LIBS\" ]; then export LD_LIBRARY_PATH=\"\$PY_NVIDIA_LIBS:\${LD_LIBRARY_PATH:-}\"; fi && \
export EXO_LIBP2P_NAMESPACE='$CLUSTER_NAMESPACE' && \
uv run --extra '$EXO_EXTRA' exo $EXO_ARGS"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already running: $SESSION_NAME"
    echo "attach with: tmux attach -t $SESSION_NAME"
    echo "stop with: tmux kill-session -t $SESSION_NAME"
    echo "docs: docs/start-stop-exo.md"
    exit 0
  fi

  tmux new-session -d -s "$SESSION_NAME" "$COMMAND"
  echo "started exo in tmux session: $SESSION_NAME"
  echo "attach with: tmux attach -t $SESSION_NAME"
  echo "stop with: tmux kill-session -t $SESSION_NAME"
  echo "docs: docs/start-stop-exo.md"
  exit 0
fi

if pgrep -f "uv run --extra $EXO_EXTRA exo" >/dev/null 2>&1; then
  echo "exo appears to already be running"
  echo "stop with pid file: kill \"\$(cat $LOG_DIR/exo.detached.pid)\""
  echo "docs: docs/start-stop-exo.md"
  exit 0
fi

nohup bash -lc "$COMMAND" >>"$LOG_FILE" 2>&1 &
echo "$!" >"$LOG_DIR/exo.detached.pid"
echo "started exo with nohup, pid: $(cat "$LOG_DIR/exo.detached.pid")"
echo "log: $LOG_FILE"
echo "stop with pid file: kill \"\$(cat $LOG_DIR/exo.detached.pid)\""
echo "force stop with pid file: kill -9 \"\$(cat $LOG_DIR/exo.detached.pid)\""
echo "docs: docs/start-stop-exo.md"
