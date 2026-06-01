# Linux NVIDIA Cluster Setup

Runbook for fresh Ubuntu/Debian machines running exo with NVIDIA GPUs and
`mlx-cuda13`.

Do not commit Hugging Face tokens. Use `hf auth login` on each machine and
revoke temporary tokens after setup.

## Target Tested Setup

- Ubuntu 24.04 with CUDA 13.2
- Debian 13 with CUDA 13.3
- NVIDIA RTX 3060 12 GB nodes
- exo from source
- Model tested: `mlx-community/Qwen3.6-35B-A3B-4bit`

## 1. NVIDIA Driver

Install and validate the NVIDIA driver before installing CUDA toolkit packages
or exo dependencies.

### Ubuntu 24.04

Install helper tools:

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
```

List devices and recommended drivers:

```bash
ubuntu-drivers devices
```

Look for the line marked `recommended`, for example:

```text
driver   : nvidia-driver-610 - distro non-free recommended
```

Install the recommended package explicitly. Do not use automatic driver
installation on cluster machines; inspect the recommendation and choose the
exact package:

```bash
sudo apt install -y nvidia-driver-610
sudo reboot
```

Replace `nvidia-driver-610` with the package marked `recommended` on that
machine.

### Debian 13

Enable Debian non-free firmware/components if they are not already enabled.
Check:

```bash
grep -R "^deb " /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null
```

Your Debian entries should include `contrib non-free non-free-firmware`.
Then install driver detection tools:

```bash
sudo apt update
sudo apt install -y nvidia-detect
```

List the recommended driver:

```bash
nvidia-detect
```

Install the package recommended by `nvidia-detect` and reboot. On typical
Debian NVIDIA systems this is:

```bash
sudo apt install -y nvidia-driver firmware-misc-nonfree
sudo reboot
```

If `nvidia-detect` recommends a different package, install that explicit
package instead.

### Validate Driver

After reboot, run on every machine:

```bash
nvidia-smi
```

Expected:

- GPU name appears, e.g. `NVIDIA GeForce RTX 3060`.
- Driver version appears.
- CUDA version appears.
- No `NVIDIA-SMI has failed` error.

Important: the CUDA version shown by `nvidia-smi` is the maximum CUDA API
supported by the driver. It does not mean cuBLAS, NVRTC, or CUDA headers are
installed. Those are installed later.

## 2. Base Packages

Run on every machine:

```bash
sudo apt update
sudo apt install -y \
  curl wget git build-essential pkg-config libssl-dev \
  nodejs npm jq tmux
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv --version
```

Install Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup toolchain install nightly
```

## 3. Clone And Build Dashboard

Run on every machine:

```bash
git clone https://github.com/exo-explore/exo
cd ~/exo

cd dashboard
npm install
npm run build
cd ..
```

## 4. NVIDIA CUDA Packages

First check the distro:

```bash
source /etc/os-release
echo "$ID $VERSION_ID"
```

### Ubuntu 24.04

Add the NVIDIA CUDA repo:

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

Install CUDA 13.2 packages:

```bash
sudo apt install -y \
  cublas-cuda-13 \
  cuda-nvrtc-13-2 \
  cuda-toolkit-13-2
sudo ldconfig
```

Persist CUDA environment:

```bash
cat >> ~/.bashrc <<'EOF'
export CUDA_HOME=/usr/local/cuda-13.2
export CUDA_PATH=/usr/local/cuda-13.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/libcublas/13:$LD_LIBRARY_PATH
EOF
source ~/.bashrc
```

### Debian 13

Add the NVIDIA CUDA repo:

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

Install CUDA 13.3 packages:

```bash
sudo apt install -y \
  cublas-cuda-13 \
  cuda-nvrtc-13-3 \
  cuda-toolkit-13-3
sudo /sbin/ldconfig
```

Persist CUDA environment:

```bash
cat >> ~/.bashrc <<'EOF'
export CUDA_HOME=/usr/local/cuda-13.3
export CUDA_PATH=/usr/local/cuda-13.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/libcublas/13:$LD_LIBRARY_PATH
EOF
source ~/.bashrc
```

## 5. Install Python Dependencies

Run on every machine:

```bash
cd ~/exo
uv sync --extra mlx-cuda13
```

If `torch` is missing later, reinstall the CUDA 13.0 wheels explicitly from the
PyTorch index using the same PyTorch family pinned by this project:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu130 \
  --force-reinstall \
  torch==2.12.0 torchvision==0.27.0 torchaudio==2.12.0
```

## 6. Validate CUDA/MLX/PyTorch

Run on every machine:

```bash
ldconfig -p 2>/dev/null | grep libcublasLt.so.13 || /sbin/ldconfig -p | grep libcublasLt.so.13
find /usr/local /usr/lib -name 'libnvrtc.so.13*' 2>/dev/null
ls "$CUDA_HOME/include/cuda.h"
ls "$CUDA_HOME/include/cuda_runtime.h"
```

Test imports and a small MLX operation:

```bash
cd ~/exo
uv run --extra mlx-cuda13 python -c "import torch; import mlx.core as mx; a=mx.ones((2,2)); mx.eval(a); print(torch.__version__); print(mx.default_device()); print(a)"
```

Common missing library errors:

- `libcublasLt.so.13`: install `cublas-cuda-13` and run `ldconfig`.
- `libnvrtc.so.13`: install `cuda-nvrtc-13-*`, set `LD_LIBRARY_PATH`.
- `libtorch_cuda.so: undefined symbol: ncclCommResume`: PyTorch is loading an
  incompatible system NCCL before the NCCL packaged in the Python environment.
  Start exo with `scripts/start_exo_detached.sh`, or manually prepend the
  virtualenv NVIDIA libraries before launching:
  ```bash
  cd ~/exo
  PY_NVIDIA_LIBS=$(find .venv/lib -path '*/site-packages/nvidia/*/lib' -type d 2>/dev/null | paste -sd: -)
  export LD_LIBRARY_PATH="$PY_NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
  uv run --extra mlx-cuda13 python -c "import torch; print(torch.__version__)"
  ```
- `Can not find locations of CUDA headers`: set `CUDA_HOME`/`CUDA_PATH` and
  install `cuda-toolkit-13-*`.
- `No module named 'torch'`: run `uv sync --extra mlx-cuda13` or install torch
  with `uv pip install`.

## 7. Hugging Face Login

Create a read token at:

```text
https://huggingface.co/settings/tokens
```

Login on every machine:

```bash
cd ~/exo
uv run --extra mlx-cuda13 hf auth login
uv run --extra mlx-cuda13 hf auth whoami
```

Non-interactive form:

```bash
uv run --extra mlx-cuda13 hf auth login --token 'hf_xxx'
```

Do not save tokens in the repo.

## 8. Download Model Manually

The exo dashboard may stay at `Preparing download...` while the backend only
shows `DownloadPending`. The reliable path is to download directly with the
Hugging Face CLI into exo's normalized model directory.

Run on every machine:

```bash
cd ~/exo
uv run --extra mlx-cuda13 hf download mlx-community/Qwen3.6-35B-A3B-4bit \
  --local-dir ~/.local/share/exo/models/mlx-community--Qwen3.6-35B-A3B-4bit
```

If interrupted, run the same command again. It resumes.

Monitor size:

```bash
watch -n 5 'du -sh ~/.local/share/exo/models/mlx-community--Qwen3.6-35B-A3B-4bit'
```

## 9. Start exo Cluster

Install `tmux` if it was not installed with the base packages:

```bash
sudo apt install -y tmux
```

Start exo inside `tmux` so it keeps running after SSH disconnects.

Run on every machine:

```bash
tmux new -s exo
cd ~/exo
EXO_LIBP2P_NAMESPACE=my-cluster uv run --extra mlx-cuda13 exo -v
```

Detach without stopping exo:

```text
Ctrl+B, then D
```

Reattach later:

```bash
tmux attach -t exo
```

Stop exo when it was started in `tmux`:

```bash
tmux send-keys -t exo C-c
tmux kill-session -t exo
```

If it was started with `scripts/start_exo_detached.sh` and fell back to `nohup`
because `tmux` was unavailable:

```bash
kill "$(cat ~/.cache/exo/exo.detached.pid)"
```

Open the dashboard:

```text
http://NODE_IP:52415
```

Find the node IP:

```bash
hostname -I
```

If auto-discovery fails, use a fixed libp2p port.

Machine 1:

```bash
EXO_LIBP2P_NAMESPACE=my-cluster uv run --extra mlx-cuda13 exo -v --libp2p-port 30000
```

Machine 2:

```bash
EXO_LIBP2P_NAMESPACE=my-cluster uv run --extra mlx-cuda13 exo -v \
  --bootstrap-peers /ip4/MACHINE_1_IP/tcp/30000
```

## 10. Create Instance From API

Preview valid placements:

```bash
curl -s "http://localhost:52415/instance/previews?model_id=mlx-community/Qwen3.6-35B-A3B-4bit" \
  | jq '.previews[] | {error, instance_meta, sharding, memory_delta_by_node}'
```

For NVIDIA/Linux, use `MlxRing`. `MlxJaccl` may report `MlxMetal` backend errors
and can be ignored for this setup.

Create first valid `MlxRing` pipeline instance:

```bash
INSTANCE=$(curl -s "http://localhost:52415/instance/previews?model_id=mlx-community/Qwen3.6-35B-A3B-4bit" \
  | jq -c '.previews[] | select(.error == null and .instance_meta == "MlxRing" and .sharding == "Pipeline") | .instance' \
  | head -n1)

curl -X POST http://localhost:52415/instance \
  -H 'Content-Type: application/json' \
  -d "{\"instance\":$INSTANCE}"
```

## 11. Useful Checks

Downloaded status for one model:

```bash
curl -s http://localhost:52415/state/downloads | jq -r '
to_entries[] as $node |
$node.value[] |
to_entries[0] as $d |
(
  $d.value.shardMetadata.TensorShardMetadata.modelCard.modelId //
  $d.value.shardMetadata.PipelineShardMetadata.modelCard.modelId //
  $d.value.shard_metadata.TensorShardMetadata.model_card.model_id //
  $d.value.shard_metadata.PipelineShardMetadata.model_card.model_id //
  ""
) as $model |
select($model == "mlx-community/Qwen3.6-35B-A3B-4bit") |
[$node.key, $d.key] | @tsv'
```

Expected after manual download:

```text
DownloadCompleted
```

Check instances:

```bash
curl -s http://localhost:52415/state/instances | jq .
```

Search logs:

```bash
grep -R -iE "Download failed|Downloading model_id|hugging|http|exception|traceback|timeout|401|403|429|CUDA|libcublas|libnvrtc|torch" \
  ~/.cache/exo/exo_log | tail -120
```

GPU status:

```bash
nvidia-smi
```

## 12. Notes

- `nvidia-smi` showing CUDA 13.x only means the driver supports CUDA 13.x. It
  does not mean cuBLAS, NVRTC, or CUDA headers are installed.
- The dashboard download screen can stay on `Preparing download...`. Check
  `/state/downloads`; real active download appears as `DownloadOngoing`.
- For the tested `Qwen3.6-35B-A3B-4bit` placement on 2x RTX 3060 12 GB, exo
  estimated about 10.2 GB per node. This is tight but valid on paper.
- Always run the same model download on every node selected by the placement.
