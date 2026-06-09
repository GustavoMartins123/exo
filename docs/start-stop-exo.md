# Start and stop Exo

Quick commands for machines started with `scripts/start_exo_detached.sh`.

## Start

From the repository root:

```bash
./scripts/start_exo_detached.sh
```

If `tmux` is installed, the script starts Exo in a session named `exo`.
If `tmux` is not available, it falls back to `nohup` and writes a pid file.

## Attach to tmux

```bash
tmux attach -t exo
```

Detach without stopping Exo:

```text
Ctrl-b d
```

## Stop when running in tmux

Preferred:

```bash
tmux kill-session -t exo
```

Graceful interrupt first, then close session:

```bash
tmux send-keys -t exo C-c
tmux kill-session -t exo
```

## Stop when running with nohup / pid file

```bash
kill "$(cat ~/.cache/exo/exo.detached.pid)"
```

If it does not stop:

```bash
kill -9 "$(cat ~/.cache/exo/exo.detached.pid)"
```

## Find Exo processes manually

```bash
ps -ef | grep -E "uv run|exo" | grep -v grep
```

Then stop a specific process:

```bash
kill <PID>
```

Use `kill -9 <PID>` only when the normal `kill <PID>` does not stop it.

## Logs

```bash
tail -f ~/.cache/exo/exo.detached.log
tail -f ~/.cache/exo/exo_log/exo.log
```
