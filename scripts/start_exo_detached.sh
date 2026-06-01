#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${EXO_SESSION_NAME:-exo}"
CLUSTER_NAMESPACE="${EXO_LIBP2P_NAMESPACE:-my-cluster}"
EXO_DIR="${EXO_DIR:-$HOME/exo}"
EXO_EXTRA="${EXO_EXTRA:-mlx-cuda13}"
EXO_ARGS="${EXO_ARGS:--v --no-batch}"
LOG_DIR="${EXO_LOG_CAPTURE_DIR:-$HOME/.cache/exo}"
LOG_FILE="$LOG_DIR/exo.detached.log"

mkdir -p "$LOG_DIR"

if [ ! -d "$EXO_DIR" ]; then
  echo "exo directory not found: $EXO_DIR" >&2
  exit 1
fi

COMMAND="cd '$EXO_DIR' && \
source '$HOME/.bashrc' >/dev/null 2>&1 || true && \
PY_NVIDIA_LIBS=\$(find '$EXO_DIR/.venv/lib' -path '*/site-packages/nvidia/*/lib' -type d 2>/dev/null | paste -sd: -) && \
if [ -n \"\$PY_NVIDIA_LIBS\" ]; then export LD_LIBRARY_PATH=\"\$PY_NVIDIA_LIBS:\${LD_LIBRARY_PATH:-}\"; fi && \
export EXO_LIBP2P_NAMESPACE='$CLUSTER_NAMESPACE' && \
uv run --extra '$EXO_EXTRA' exo $EXO_ARGS"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already running: $SESSION_NAME"
    echo "attach with: tmux attach -t $SESSION_NAME"
    exit 0
  fi

  tmux new-session -d -s "$SESSION_NAME" "$COMMAND"
  echo "started exo in tmux session: $SESSION_NAME"
  echo "attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

if pgrep -f "uv run --extra $EXO_EXTRA exo" >/dev/null 2>&1; then
  echo "exo appears to already be running"
  exit 0
fi

nohup bash -lc "$COMMAND" >>"$LOG_FILE" 2>&1 &
echo "$!" >"$LOG_DIR/exo.detached.pid"
echo "started exo with nohup, pid: $(cat "$LOG_DIR/exo.detached.pid")"
echo "log: $LOG_FILE"
