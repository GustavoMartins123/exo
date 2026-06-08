# macOS / Mac Studio setup

Runbook for running Exo from source on Apple Silicon machines, including Mac
Studio nodes.

## Prerequisites

Install Homebrew, then the basic tools:

```bash
brew install uv node
```

Install Rust nightly for the Rust bindings:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
```

Install Xcode from the App Store. Command Line Tools alone are not enough when
MLX has to compile Metal kernels from source.

After installing Xcode:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license accept
```

## Verify the Metal toolchain

Before starting Exo, verify that the Metal compiler exists:

```bash
xcrun -sdk macosx metal --version
```

Expected output looks like:

```text
Apple metal version ...
Target: air64-apple-darwin...
```

If you get this error:

```text
xcrun: error: unable to find utility "metal", not a developer tool or in PATH
```

or:

```text
cannot execute tool 'metal' due to missing Metal Toolchain
```

download the Metal Toolchain component:

```bash
sudo xcodebuild -runFirstLaunch
sudo xcodebuild -downloadComponent MetalToolchain
xcrun -sdk macosx metal --version
```

If `xcodebuild -downloadComponent MetalToolchain` fails because Xcode cannot
load `IDESimulatorFoundation` or `CoreSimulator`, open Xcode once from the GUI
and let it install additional components. Then run:

```bash
sudo xcodebuild -runFirstLaunch
sudo xcodebuild -downloadComponent MetalToolchain
xcrun -sdk macosx metal --version
```

## Manual Metal Toolchain recovery

On some macOS/Xcode installs, the Metal Toolchain asset is already downloaded
but not registered. If the commands above still fail and you can see a Metal
Toolchain asset under `/System/Library/AssetsV2`, mount and copy it manually:

```bash
hdiutil attach /System/Library/AssetsV2/com_apple_MobileAsset_MetalToolchain/*/AssetData/Restore/*.dmg
ls /Volumes/MetalToolchainCryptex/
sudo mkdir -p /Library/Developer/Toolchains
sudo cp -r /Volumes/MetalToolchainCryptex/Metal.xctoolchain /Library/Developer/Toolchains/
xcrun -sdk macosx metal --version
hdiutil detach /Volumes/MetalToolchainCryptex
```

The mounted volume should contain:

```text
Metal.xctoolchain
RestoreVersion.plist
```

If `xcrun -sdk macosx metal --version` works after this, MLX can compile Metal
kernels.

## Build dashboard

From the repository root:

```bash
cd dashboard
npm install
npm run build
cd ..
```

Do not run `cd /dashboard`; that points to a non-existent root-level directory.

## Start Exo manually

From the repository root:

```bash
./scripts/start_exo_detached.sh
```

On macOS, this script:

- detects the repository directory from the script path;
- uses `uv run --extra mlx`;
- does not pass `--no-batch` by default;
- writes logs to `~/.cache/exo/exo.detached.log`.

Watch startup:

```bash
tail -f ~/.cache/exo/exo.detached.log
```

If dependency installation failed halfway, remove the partial virtualenv and
start again:

```bash
rm -rf .venv
./scripts/start_exo_detached.sh
```

## Validate MLX inside Exo's environment

Do not validate with the system `python3`; use the Exo environment:

```bash
uv run --extra mlx python -c "import mlx.core as mx; print(mx.default_device())"
```

`import mlx; print(mlx.__version__)` is not a reliable check because the top
level `mlx` module may not expose `__version__`.

## Optional node-agent runner on macOS

For automatic start/stop from the Exo dashboard, install the launchd runner:

```bash
scripts/cluster/install_host_runner_macos.sh
```

If using the Docker node agent on Mac Studio, use the same shared directory as
the launchd runner:

```bash
cd scripts/cluster
export EXO_AGENT_SHARED_DIR="$HOME/.local/share/exo-agent"
EXO_NODE_NAME="$(hostname)" \
docker compose -f docker-compose.node.yml up -d --build
```

For now, manual startup is enough:

```bash
./scripts/start_exo_detached.sh
```
