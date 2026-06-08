# Exo Cluster Signal Agent

This cluster helper starts and stops Exo on LAN machines without running Exo
inside Docker.

Architecture with the Exo dashboard button:

```text
Exo master dashboard
  Start Children button
    -> Exo API scans EXO_CLUSTER_AGENT_CIDRS
       -> node agent container on each GPU host
          -> writes command file in /var/lib/exo-agent/commands
             -> host systemd path/service runs exo-agent-runner.sh
                -> starts/stops ~/exo/scripts/start_exo_detached.sh on the host
```

The Docker container is only a signaling layer. GPU, CUDA, MLX, tmux and the
Python venv stay on the host.

## Configure the master

On the machine that has Wi-Fi and access to the dedicated switch network:

```bash
cp .env.example .env
```

Set the dedicated switch CIDR:

```bash
EXO_CLUSTER_MASTER=true
EXO_CLUSTER_AGENT_CIDRS=10.10.10.0/24
EXO_CLUSTER_AGENT_PORT=8765
```

After Exo starts on this machine, the dashboard header shows `Start Children`.
Clicking it discovers node agents on the switch network and sends `start` to
all child nodes. No manual `curl` is needed.

## Install on each GPU host

From the Exo repo:

```bash
sudo scripts/cluster/install_host_runner.sh
```

Optional host settings before installing:

```bash
export EXO_AGENT_SHARED_DIR=/var/lib/exo-agent
export EXO_DIR=/home/iapar/exo
```

Start the node agent container:

```bash
cd scripts/cluster
EXO_NODE_NAME="$(hostname)" \
docker compose -f docker-compose.node.yml up -d --build
```

On unusual routing setups, force the IP that the Exo master should use:

```bash
EXO_AGENT_ADVERTISE_HOST="10.10.10.2" docker compose -f docker-compose.node.yml up -d
```

If you want a shared token:

```bash
export EXO_AGENT_TOKEN="change-me"
```

Use the same token in the master's `.env` and in node containers.

## Optional standalone controller

The dashboard no longer needs this for normal operation, but the standalone
controller is still useful for debugging the node agents without running Exo:

```bash
cd scripts/cluster
docker compose -f docker-compose.controller.yml up -d --build
```

List registered nodes:

```bash
curl http://127.0.0.1:8766/nodes
```

Start Exo on all registered nodes:

```bash
curl -X POST http://127.0.0.1:8766/nodes/all/start
```

Other commands:

```bash
curl -X POST http://127.0.0.1:8766/nodes/all/stop
curl -X POST http://127.0.0.1:8766/nodes/all/restart
curl -X POST http://127.0.0.1:8766/nodes/all/pull
curl -X POST http://127.0.0.1:8766/nodes/all/status
```

Per-node command:

```bash
curl -X POST http://127.0.0.1:8766/nodes/iapar2/start
```

## Direct node command

You can bypass the controller:

```bash
curl -X POST http://10.10.10.2:8765/exo/start
curl http://10.10.10.2:8765/status
```

## Network note

Run this on the dedicated switch interface. Do not connect the switch to the
internet router if you want Exo traffic isolated from the main LAN.
