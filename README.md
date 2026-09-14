# PyTorch GPU Environment Test & ResNet/CIFAR-10 Training

> **Learning project:** I'm a beginner, currently learning to program, and I'm using this repo to practice PyTorch, GPU/CUDA programming, and Docker on a real (if small) project. Feedback, corrections, and suggestions are very welcome!

A diagnostic script that validates a local PyTorch + NVIDIA GPU setup (typically run inside a Docker container), plus a from-scratch ResNet-20 training pipeline on CIFAR-10. [main.py](main.py) supports two modes via a positional argument:

- `benchmark` (default) — GPU/CUDA/cuDNN sanity checks and micro-benchmarks
- `train` — trains a ResNet-20 (He et al., 2015 CIFAR variant) on CIFAR-10

## Benchmark mode: what it does

Running [main.py](main.py) performs the following checks and benchmarks, in order:

1. **Environment info** — PyTorch version, CUDA availability, cuDNN version, GPU name, total VRAM, compute capability, and Tensor Core support.
2. **Matrix multiplication (cuBLAS)** — runs `torch.mm` at increasing matrix sizes (1000 to 6000) and reports time and estimated GFLOPS. Sizes are tuned to avoid OOM on GPUs with limited VRAM (e.g. a 4GB RTX 3050), and each size fails gracefully if it doesn't fit.
3. **2D convolution (cuDNN)** — runs a small conv stack over a batch of 16 224x224 images and reports average latency and throughput (images/sec).
4. **Mixed precision (FP16 / autocast)** — compares a matmul in FP32 vs. FP16 (`torch.autocast`) to show the Tensor Core speedup.
5. **Simulated training loop** — runs 30 forward/backward/optimizer steps on a small CNN and reports step latency, throughput, and final loss.

If CUDA isn't available, the script still runs on CPU (skipping the FP16 test) so you can see what's missing.

## Train mode: what it does

`python main.py train` trains a **ResNet-20** (the 6n+2 layer CIFAR architecture from the original ResNet paper, built from scratch with basic residual blocks — not `torchvision.models`) on **CIFAR-10**:

- Downloads CIFAR-10 into `./data` (via `torchvision.datasets.CIFAR10`) on first run
- Standard augmentation: random crop (32, padding 4) + random horizontal flip, with per-channel normalization
- SGD (momentum 0.9, nesterov, weight decay 5e-4) with a step LR schedule (decays at 50% and 75% of training)
- Prints per-epoch train/test loss and accuracy, and saves the best checkpoint (by test accuracy) to `resnet20_cifar10.pth`

Options:

```bash
python main.py train --epochs 30 --batch-size 128 --lr 0.1 --data-dir ./data
```

## Local test results

Two real local training runs on this setup, logged in [`TESTE_DE_AMBIENTE_LOCAL_1.txt`](testes-de-ambiente/TESTE_DE_AMBIENTE_LOCAL_1.txt) and [`TESTE_DE_AMBIENTE_LOCAL_2.txt`](testes-de-ambiente/TESTE_DE_AMBIENTE_LOCAL_2.txt).

**Environment tested**

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU — ~4 GB VRAM, Compute Capability 8.6, 16 SMs, Tensor Cores supported
- PyTorch 2.2.1, built with CUDA 12.1, cuDNN 8.9.2

**Runs**

| Run | Epochs | Batch size | Initial LR | Best test accuracy | ~Training time* |
|---|---|---|---|---|---|
| Test 1 | 30 | 128 | 0.1 | 89.40% | ~5.4 min |
| Test 2 | 50 | 64 | 0.05 | 90.81% | ~9.5 min |

\* Excludes the CIFAR-10 download, which only happens once — the first run downloaded and extracted the ~170 MB dataset in ~19 minutes; later runs reused the cached copy in `./data` ("Files already downloaded and verified").

**Observations**

- In both runs, test accuracy jumps sharply right after each learning-rate decay step (50% and 75% of training) — e.g. in Test 1 it jumps from ~79% to ~87% between epoch 15 and 16, and in Test 2 from ~82% to ~89% between epoch 25 and 26.
- Test 2 (more epochs, smaller batch size, lower initial LR) reached a slightly higher final accuracy than Test 1 (90.81% vs 89.40%), at the cost of roughly double the training time.
- Both configurations fit comfortably within the RTX 3050's 4 GB of VRAM.

## Running in CI on a real GPU (Kubernetes + Actions Runner Controller)

Training/benchmarking also runs in GitHub Actions, on a **self-hosted runner backed by my own RTX 3050** — not a GitHub-hosted (virtual) runner, since GitHub doesn't offer GPU runners on the free tier and the whole point of this repo is exercising the real card.

There are two workflow variants, kept side by side under different names so both can live in the repo at once — **only one can actually succeed at a time**, though, since they need different config applied to the single `arc-runner-set-gpu` deployment (see below):

- [.github/workflows/pytorch-gpu-python.yaml](.github/workflows/pytorch-gpu-python.yaml) **(current/default)** — runs `python3 main.py ...` **natively inside the runner pod**, no Docker involved. Needs [k8s/gpu-runner-values.yaml](k8s/gpu-runner-values.yaml) applied to the cluster: the runner pod's own image bundles Python/PyTorch (built from [k8s/runner-image/Dockerfile](k8s/runner-image/Dockerfile)), and it's that same pod's `nvidia.com/gpu: 1` reservation that's actually used to train — no Docker, no socket, one process, one real allocated GPU.
- [.github/workflows/pytorch-gpu-docker.yaml](.github/workflows/pytorch-gpu-docker.yaml) — the earlier approach: builds the [Dockerfile](Dockerfile) image on the runner and calls `docker run --gpus all`, pulling the checkpoint/dataset back out with `docker cp` (its Docker daemon doesn't share the pod's filesystem, so bind mounts don't work). Needs [k8s/gpu-runner-values.docker.yaml](k8s/gpu-runner-values.docker.yaml) applied instead: it bind-mounts the host's Docker socket into the pod. Kept as a reference/fallback — its actual training container runs against the host's Docker daemon directly, outside the pod's own GPU reservation, so the `nvidia.com/gpu: 1` limit there doesn't really correspond to what's using the GPU.
- [.github/workflows/teste-gpu.yaml](.github/workflows/teste-gpu.yaml) — a minimal sanity check (`nvidia-smi`, called directly, no Docker) to confirm the runner pod can see the GPU at all. Works under either of the two configs above, since both request `nvidia.com/gpu: 1` on the runner pod.

Both `k8s/gpu-runner-values*.yaml` files request the same `nvidia.com/gpu: 1` from Kubernetes — what differs is what the pod's container actually does with it. To switch which workflow variant is live:

```bash
helm upgrade --install arc-runner-set-gpu \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  -n arc-systems -f k8s/gpu-runner-values.yaml           # python (default)
  # -f k8s/gpu-runner-values.docker.yaml                 # or: docker
```

- [k8s/](k8s/) — the rest of the cluster config behind `arc-runner-set-gpu`: [kind-gpu-config.yaml](k8s/kind-gpu-config.yaml) passes the RTX 3050 from the host into a [kind](https://kind.sigs.k8s.io/) node (NVIDIA Container Toolkit binaries + `default_runtime_name = "nvidia"` on containerd), and [nvidia-device-plugin.yaml](k8s/nvidia-device-plugin.yaml) is a snapshot of the official [NVIDIA k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) DaemonSet (applied via plain `kubectl create -f <url>`, no Helm/GPU Operator) that turns that into a schedulable `nvidia.com/gpu` node resource — one real GPU, no time-slicing/virtual splitting.

These `k8s/*.yaml` files describe the intended cluster state — they need to be applied by hand (`helm upgrade`/`kubectl apply`) to take effect; they're kept in sync with the host manually, not applied automatically from the repo. `k8s/gpu-runner-values.yaml` also still points `image:` at a placeholder (`<seu-registry>/pytorch-gpu-sandbox-runner:latest`) — build and push `k8s/runner-image/Dockerfile` and update that field before relying on it; until then, `pytorch-gpu-python.yaml` falls back to `pip install --user -r requirements.txt` at the start of the job.

## Requirements

- Docker with NVIDIA Container Toolkit installed (for GPU access)
- An NVIDIA GPU with recent drivers

No local Python setup is required if you use Docker; the image already bundles PyTorch, torchvision, and torchaudio.

### Host environment this was built and tested on

Exact versions of everything installed on the host machine (not inside the containers — those pin nothing beyond `pytorch/pytorch:latest`, see [Local test results](#local-test-results) for the PyTorch/CUDA/cuDNN versions that image resolved to). Useful if something behaves differently for you and you want to diff versions first.

| Component | Version | Installed via |
|---|---|---|
| OS | Ubuntu 26.04.1 LTS (kernel `7.0.0-31-generic`) | — |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GB VRAM | — |
| NVIDIA driver | 615.71.09 (CUDA 13.4 max supported) | distro driver package |
| NVIDIA Container Toolkit | 1.20.0 (`nvidia-ctk`, `nvidia-container-cli`) | apt (`nvidia-container-toolkit` / `-base`) |
| Docker Engine | 29.8.0 | official Docker `apt` repo (`docker-ce`) |
| Docker Compose | v5.5.1 (plugin) | bundled with the above |
| kind | v0.33.0 | binary from [kubernetes-sigs/kind](https://github.com/kubernetes-sigs/kind) releases, `/usr/local/bin` |
| kubectl | v1.37.0 | binary from `dl.k8s.io`, `/usr/local/bin` |
| Helm | v3.22.0 | official install script (`get-helm-3`, `curl \| bash`) — **not** `snap`/`apt` |

Only relevant if you're touching the [K8s + ARC CI setup](#running-in-ci-on-a-real-gpu-kubernetes--actions-runner-controller); the plain Docker Compose path in this README only needs Docker + NVIDIA Container Toolkit + a driver.

## Running with Docker Compose

```bash
docker compose up --build
```

This builds the image from [Dockerfile](Dockerfile), mounts the project directory into the container, and runs `python3 main.py` (benchmark mode by default) with GPU access (`compose.yaml` requests all NVIDIA devices).

To run training instead, override the command:

```bash
docker compose run --rm app python3 main.py train --epochs 30
```

## Debugging

[compose.debug.yaml](compose.debug.yaml) starts the app under `debugpy`, listening on port `5678`, so you can attach a remote debugger before execution continues:

```bash
docker compose -f compose.debug.yaml up --build
```

Unlike `compose.yaml`, this variant requests no GPU device and mounts no volume, so it runs CPU-only and doesn't persist `./data`/checkpoints back to the host — see the comments in [compose.debug.yaml](compose.debug.yaml) if you need to debug with real CUDA.

**In VS Code**, this is wired up end to end: press F5 and pick "Python: Anexar ao container (Docker debug)" — its `preLaunchTask` ([.vscode/tasks.json](.vscode/tasks.json)) runs the command above for you and waits for a `[debugpy] pronto para anexar` marker before attaching, so you don't have to time it by hand. [.vscode/tasks.json](.vscode/tasks.json) also has tasks for the plain benchmark/train Docker Compose commands above, and for the cluster lifecycle scripts (see [Running in CI on a real GPU](#running-in-ci-on-a-real-gpu-kubernetes--actions-runner-controller)) — "K8s: criar cluster GPU", "K8s: configurar ARC", "K8s: recriar cluster do zero" (both in sequence, prompting once for `GITHUB_TOKEN`), and "K8s: derrubar cluster".

## Running without Docker

If you already have a CUDA-enabled PyTorch installed locally:

```bash
pip install -r requirements.txt
python main.py
```

`requirements.txt` lists `torch`, `torchvision`, `torchaudio` unpinned — on Linux, `pip install torch` pulls a CUDA-enabled build automatically. Pin exact versions there if you want a reproducible local setup.

## Project layout

- [main.py](main.py) — the environment/GPU test script (comments and output are in Portuguese)
- [Dockerfile](Dockerfile) — builds on `pytorch/pytorch:latest` and installs torchvision/torchaudio
- [compose.yaml](compose.yaml) — runs the container with GPU access
- [compose.debug.yaml](compose.debug.yaml) — runs the container with `debugpy` for remote debugging
- [.vscode/launch.json](.vscode/launch.json) / [.vscode/tasks.json](.vscode/tasks.json) — VS Code debug config and tasks tying together Docker Compose and the cluster lifecycle scripts (see [Debugging](#debugging))
- [testes-de-ambiente/TESTE_DE_AMBIENTE_LOCAL_1.txt](testes-de-ambiente/TESTE_DE_AMBIENTE_LOCAL_1.txt) / [testes-de-ambiente/TESTE_DE_AMBIENTE_LOCAL_2.txt](testes-de-ambiente/TESTE_DE_AMBIENTE_LOCAL_2.txt) — real output logs from two local training runs (see [Local test results](#local-test-results))
- [.github/workflows/pytorch-gpu-python.yaml](.github/workflows/pytorch-gpu-python.yaml) — runs benchmark/train in CI natively on the real RTX 3050, no Docker (current default; see [Running in CI on a real GPU](#running-in-ci-on-a-real-gpu-kubernetes--actions-runner-controller))
- [.github/workflows/pytorch-gpu-docker.yaml](.github/workflows/pytorch-gpu-docker.yaml) — same, via `docker build`/`docker run --gpus all` (reference/fallback variant)
- [.github/workflows/teste-gpu.yaml](.github/workflows/teste-gpu.yaml) — minimal GPU-visibility sanity check for the self-hosted runner
- [k8s/kind-gpu-config.yaml](k8s/kind-gpu-config.yaml) — kind cluster config that passes the RTX 3050 through from the host
- [k8s/nvidia-device-plugin.yaml](k8s/nvidia-device-plugin.yaml) — snapshot of the NVIDIA k8s-device-plugin DaemonSet applied to the cluster
- [k8s/gpu-runner-values.yaml](k8s/gpu-runner-values.yaml) — Helm values for `arc-runner-set-gpu` matching the Python workflow (current default)
- [k8s/gpu-runner-values.docker.yaml](k8s/gpu-runner-values.docker.yaml) — Helm values matching the Docker workflow instead
- [k8s/runner-image/Dockerfile](k8s/runner-image/Dockerfile) — custom runner image (Python + PyTorch/CUDA) so CI runs `main.py` natively, no Docker-in-CI
- [scripts/1-create-gpu-cluster.sh](scripts/1-create-gpu-cluster.sh) / [scripts/2-setup-arc.sh](scripts/2-setup-arc.sh) / [scripts/3-teardown-cluster.sh](scripts/3-teardown-cluster.sh) — create/configure/tear down the local `kind` + ARC cluster (wired up as the VS Code tasks in [Debugging](#debugging)). `2-setup-arc.sh` takes `REPO [python|docker]` and installs the NVIDIA device plugin and the ARC controller/runner-sets as two parallel tracks (the device plugin doesn't depend on ARC, and the two runner sets only depend on the controller, not on each other), instead of one long serial chain. [scripts/kubernetes-gpu-arc-referencia.sh](scripts/kubernetes-gpu-arc-referencia.sh) is the commented study reference behind them
- [SAGA-DA-RTX3050.md](SAGA-DA-RTX3050.md) — long-form, chapter-by-chapter narrative of how the K8s/ARC CI setup in this README was actually built, bugs and all (Portuguese)
- [no-AI.md](no-AI.md) — a from-scratch roadmap for reproducing this whole project by hand, from official docs alone, without an AI assistant (Portuguese)
- [LICENSE](LICENSE) — MIT

## License

MIT — see [LICENSE](LICENSE).