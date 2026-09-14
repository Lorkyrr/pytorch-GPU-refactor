# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning project (author is a beginner) to practice PyTorch, GPU/CUDA programming, and Docker/Kubernetes. It's a single-script repo: [main.py](main.py) has two modes:

- `benchmark` (default) — GPU/CUDA/cuDNN sanity checks and micro-benchmarks (matmul, conv2d, mixed precision, a simulated training loop)
- `train` — trains a from-scratch ResNet-20 (He et al., 2015 CIFAR variant, not `torchvision.models`) on CIFAR-10

Comments and console output in `main.py` are in Portuguese; keep that convention when editing it. License is MIT (see [LICENSE](LICENSE)).

The companion doc [SAGA-DA-RTX3050.md](SAGA-DA-RTX3050.md) is a long-form, narrative "journal" of how the CI-on-real-GPU setup was built, chapter by chapter, including every bug hit along the way and how it was diagnosed — read it when you need the *story* behind a decision. This file (CLAUDE.md) is the dense, structured reference for working in the repo day to day; the gotchas table near the end of the K8s section below is a condensed extract of the SAGA's hard-won lessons, kept here so you don't have to read 900+ lines to avoid repeating a mistake that's already been solved once.

## Repository map

```
main.py                        # the whole application (benchmark + train modes)
requirements.txt                # torch, torchvision, torchaudio — unpinned
Dockerfile                      # pytorch/pytorch:latest + torchvision/torchaudio
compose.yaml                    # local dev: GPU access + bind mount
compose.debug.yaml              # debugpy variant: CPU-only, no bind mount
.dockerignore                   # Python-specific; excludes data/, *.pth, compose/Dockerfile files
.gitignore                      # .venv/, data/, *.pth, __pycache__
README.md                       # user-facing docs, host tool versions, project layout
SAGA-DA-RTX3050.md              # narrative build log of the K8s/ARC CI setup (18 chapters)
LICENSE                         # MIT
testes-de-ambiente/             # raw console output from 2 real local training runs
  TESTE_DE_AMBIENTE_LOCAL_1.txt
  TESTE_DE_AMBIENTE_LOCAL_2.txt
.github/workflows/
  pytorch-gpu-python.yaml       # current/default CI workflow — native python3, no Docker
  pytorch-gpu-docker.yaml       # reference/fallback CI workflow — docker build/run
  teste-gpu.yaml                # minimal nvidia-smi sanity check
k8s/
  kind-gpu-config.yaml          # kind cluster config: passes host RTX 3050 into the node
  nvidia-device-plugin.yaml     # pinned snapshot of NVIDIA's k8s-device-plugin DaemonSet
  gpu-runner-values.yaml        # Helm values for arc-runner-set-gpu, PYTHON variant (default)
  gpu-runner-values.docker.yaml # Helm values for arc-runner-set-gpu, DOCKER variant (fallback)
  runner-image/Dockerfile       # custom ARC runner image (Python+PyTorch/CUDA baked in) — not yet built/pushed
scripts/
  1-create-gpu-cluster.sh       # (re)creates the local kind cluster with GPU passthrough — DESTRUCTIVE
  2-setup-arc.sh                # installs device plugin + ARC controller + both runner sets
  3-teardown-cluster.sh         # deletes the kind cluster, frees GPU/RAM/CPU
  kubernetes-gpu-arc-referencia.sh  # commented study reference — not meant to be run end to end
.vscode/
  tasks.json                    # wraps Docker Compose + the 3 scripts above as VS Code tasks
  launch.json                   # debug configs: Docker attach (remote) + local file launch
```

## Commands

Docker Compose (default path — no local Python/CUDA setup needed):

```bash
docker compose up --build                              # benchmark mode
docker compose run --rm app python3 main.py train --epochs 30   # train mode
docker compose -f compose.debug.yaml up --build         # runs under debugpy on port 5678 for remote attach
```

Without Docker (requires a local CUDA-enabled PyTorch):

```bash
pip install -r requirements.txt
python main.py                 # benchmark
python main.py train --epochs 30 --batch-size 128 --lr 0.1 --data-dir ./data
```

There is no test suite, linter, or formatter configured in this repo.

### Host tool versions this was built/tested against

Full table with "installed via" notes lives in [README.md](README.md#host-environment-this-was-built-and-tested-on); summary: Ubuntu 26.04.1 LTS, kernel `7.0.0-31-generic`, NVIDIA driver 615.71.09, NVIDIA Container Toolkit 1.20.0, Docker 29.8.0 (official `docker-ce` apt repo), Docker Compose plugin v5.5.1, kind v0.33.0, kubectl v1.37.0, Helm v3.22.0 (official `get-helm-3` script — **not** `snap`/`apt`). GPU is an RTX 3050 Laptop with 4 GB VRAM — several design choices in `main.py` (matmul sizes, batch sizes) exist specifically to fit that VRAM budget without OOM.

## Architecture

### main.py layout

Single-file app, dispatch via `main()` on the positional `modo` argument (`benchmark`|`train`, default `benchmark`). CLI flags `--epochs` (30), `--batch-size` (128), `--lr` (0.1), `--data-dir` (`./data`) only apply to `train` mode.

**Benchmark mode** (`executar_benchmark`, runs these in sequence):
1. `info_ambiente()` — prints PyTorch version, `torch.version.cuda`, CUDA availability, cuDNN enabled/version, GPU name, total VRAM, compute capability, SM count, and whether Tensor Cores are supported (compute capability >= 7.0). Falls back to CPU with a warning if CUDA isn't available (the FP16 test is then skipped, everything else still runs).
2. `teste_matmul(device)` — cuBLAS test: `torch.mm` at N x N for N in `[1000, 2000, 4000, 6000]`, timed with `torch.cuda.synchronize()` around it, reports elapsed time and estimated GFLOPS (`2*N^3/time`). Each size is wrapped in `try/except RuntimeError` so an OOM on a larger size doesn't kill the run — it just prints "[FALHOU]" and calls `torch.cuda.empty_cache()`.
3. `teste_convolucao(device)` — cuDNN test: a 2-layer `Conv2d(3→64)→ReLU→Conv2d(64→128)→ReLU→MaxPool2d` stack over a batch of 16 224x224 images, one untimed warm-up pass (cuDNN's algorithm-selection overhead on the first call would otherwise skew the timing), then 20 timed repetitions; reports avg latency and images/sec throughput.
4. `teste_precisao_mista(device)` — compares a 4000x4000 `torch.mm` in FP32 vs. inside `torch.autocast(device_type="cuda", dtype=torch.float16)`, reports the speedup ratio. Skipped entirely on CPU (`device.type != "cuda"`).
5. `teste_treino_simulado(device)` — a tiny CNN (`Conv2d→ReLU→Conv2d→ReLU→MaxPool2d→Flatten→LazyLinear(256)→ReLU→Linear(256,10)`) trained for 30 steps with Adam (lr=1e-3) on random 64x64 batches of 32, one warm-up step, reports step latency, throughput, and final loss.
6. Wraps with GPU memory accounting: allocated/reserved before and after (`torch.cuda.memory_allocated/reserved`), plus `torch.cuda.max_memory_allocated()` as the peak.

**Train mode** (`treinar_resnet_cifar10`):
- **Model** — `resnet20_cifar()` builds `ResNetCIFAR(blocos_por_estagio=3)`, the 6n+2-layer CIFAR variant from the original ResNet paper (He et al., 2015), **not** `torchvision.models`:
  - Stem: `Conv2d(3→16, 3x3, pad=1, bias=False) → BatchNorm2d → ReLU`.
  - 3 stages of `n=3` `BlocoResidual` blocks each (9 blocks total, plus the stem = 20 weight layers): stage 1 stays at 16 channels/stride 1, stage 2 goes to 32 channels with stride 2 on its first block, stage 3 goes to 64 channels with stride 2 on its first block. Each `BlocoResidual` is `Conv3x3→BN→ReLU→Conv3x3→BN`, added to a shortcut that's an identity *unless* the stride != 1 or the channel count changes, in which case the shortcut is a `1x1 Conv (stride-matched) → BN` projection — standard ResNet "option B" shortcut.
  - Head: `AdaptiveAvgPool2d(1) → Flatten → Linear(64→10)`.
- **Data** (`cifar10_dataloaders`) — `torchvision.datasets.CIFAR10`, auto-downloads into `--data-dir` (default `./data`) on first run (~170–341 MB depending on source, taking up to ~19 minutes on a slow connection per real runs logged in `testes-de-ambiente/`). Per-channel normalization with fixed CIFAR-10 stats: mean `(0.4914, 0.4822, 0.4465)`, std `(0.2470, 0.2435, 0.2616)`. Train transform adds `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()`; test transform is normalization only. `DataLoader` uses `pin_memory=True`, `num_workers=2`.
- **Optimizer/schedule** — SGD, momentum 0.9, nesterov, weight_decay 5e-4; `MultiStepLR` with milestones at 50% and 75% of total epochs, gamma 0.1 (two 10x LR drops).
- **Checkpointing** — after every epoch, if test accuracy improved, saves `model.state_dict()` (not the full model) to `--checkpoint-path`-equivalent, hardcoded default `resnet20_cifar10.pth` at the repo root (gitignored).
- Two real local runs are logged verbatim in `testes-de-ambiente/` — 30 epochs/bs128/lr0.1 reached 89.40% test accuracy in ~5.4 min; 50 epochs/bs64/lr0.05 reached 90.81% in ~9.5 min. Both fit comfortably in the RTX 3050's 4 GB VRAM.

### Docker / Compose

- [Dockerfile](Dockerfile) — `FROM pytorch/pytorch:latest`, `pip install --upgrade pip && pip install torchvision torchaudio`, `COPY . /app`, `CMD ["python3", "main.py"]`. Not pinned to a specific PyTorch/CUDA/cuDNN version — whatever `pytorch/pytorch:latest` resolves to at build time (a real local build resolved to PyTorch 2.2.1 / CUDA 12.1 / cuDNN 8.9.2, logged in the `testes-de-ambiente/` files, but that's a snapshot, not a pin).
- [compose.yaml](compose.yaml) — bind-mounts the whole project (`.:/app`) so `./data` and `resnet20_cifar10.pth` persist back to the host, sets `PYTHONUNBUFFERED=1` (otherwise stdout buffers and progress logs lag), and requests `deploy.resources.reservations.devices` with `driver: nvidia, count: all, capabilities: [gpu]` — this **requires the NVIDIA Container Toolkit** installed on the host; without it the GPU reservation fails at `docker compose up`.
- [compose.debug.yaml](compose.debug.yaml) — deliberately **no** GPU reservation and **no** bind mount (runs CPU-only, isolated). Installs `debugpy` on the fly and runs `python /tmp/debugpy --wait-for-client --listen 0.0.0.0:5678 main.py`, printing a `[debugpy] pronto para anexar` marker right before it starts listening — [.vscode/tasks.json](.vscode/tasks.json)'s background problem matcher watches for that exact string so the VS Code debugger doesn't try to attach to a port that isn't listening yet.
- [.dockerignore](.dockerignore) — Python-specific (rewritten from a stale .NET/Java/Node template): excludes `__pycache__`, `.venv`, `.git*`, `.vscode`, `.env`, `data/` and `*.pth` (the downloaded dataset and checkpoints — large, never needed in the image), and the compose/Dockerfile files themselves.

## CI on a real GPU (Kubernetes + Actions Runner Controller)

CI doesn't use GitHub-hosted runners (no free GPU tier) — it runs on a **self-hosted runner backed by the author's own RTX 3050**, via Actions Runner Controller (ARC) on a local `kind` cluster. This is the part of the repo that requires reading multiple files together to understand.

### The two workflow variants

Only **one** can be live at a time, because both target the same `arc-runner-set-gpu` Helm deployment with different values:

| | [pytorch-gpu-python.yaml](.github/workflows/pytorch-gpu-python.yaml) — **current/default** | [pytorch-gpu-docker.yaml](.github/workflows/pytorch-gpu-docker.yaml) — reference/fallback |
|---|---|---|
| How it runs `main.py` | `python3 main.py ...` **natively inside the runner pod**, no Docker | Builds [Dockerfile](Dockerfile) on the runner, then `docker run --gpus all` |
| Requires | [k8s/gpu-runner-values.yaml](k8s/gpu-runner-values.yaml) applied | [k8s/gpu-runner-values.docker.yaml](k8s/gpu-runner-values.docker.yaml) applied |
| Where the GPU reservation actually lands | On the pod itself — the process using CUDA *is* the process the k8s scheduler reserved `nvidia.com/gpu: 1` for | On the **host's Docker daemon**, reached through a bind-mounted socket — the training container runs entirely outside Kubernetes' own accounting (see "GPU reservation caveat" below) |
| Extracting results | N/A — same filesystem as the job | `docker cp` (the Docker daemon doesn't share the pod's filesystem, so bind mounts don't work here) |

[teste-gpu.yaml](.github/workflows/teste-gpu.yaml) is a minimal `nvidia-smi` sanity check that works under either config. Both GPU-runner workflows share `concurrency: {group: gpu-runner, cancel-in-progress: false}` — there's only one physical GPU, so only one of these jobs is ever allowed to run at a time.

To switch which variant is live:
```bash
helm upgrade --install arc-runner-set-gpu \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  -n arc-systems -f k8s/gpu-runner-values.yaml           # python (default)
  # -f k8s/gpu-runner-values.docker.yaml                 # or: docker
```
(`scripts/2-setup-arc.sh REPO [python|docker]` does this for you — see below.)

**GPU reservation caveat (important nuance, not a bug):** `resources.limits.nvidia.com/gpu: 1` in the values file only guarantees the *runner pod* has a GPU reserved — it does **not**, by itself, guarantee that whatever the pod does with that GPU is what Kubernetes thinks is happening. In the docker variant, `docker run --gpus all` talks directly to the **host's Docker daemon** through the mounted socket, completely bypassing the node's `containerd` (which is what the k8s scheduler's GPU accounting is actually built on) — the training container's lifecycle is invisible to Kubernetes. This currently works safely only because the `concurrency` group above ensures just one workflow (hence one `docker run --gpus all`) is ever active. The python variant closes this gap entirely: no Docker, no socket, one process, and that process *is* the one the pod's `nvidia.com/gpu: 1` was reserved for.

### How the GPU physically reaches a pod

1. [k8s/kind-gpu-config.yaml](k8s/kind-gpu-config.yaml) — the `kind` cluster config. `extraMounts` bring in: `/dev/null` → `/var/run/nvidia-container-devices/all` (the "magic" path the NVIDIA Container Runtime watches to auto-inject driver libraries), the NVIDIA Container Toolkit binaries (`nvidia-container-runtime`, `nvidia-container-cli`, `nvidia-ctk`), and the host's `/var/run/docker.sock`. `containerdConfigPatches` sets `default_runtime_name = "nvidia"` for the node's **internal** containerd (distinct from the host's Docker daemon!) — without this, pods *inside* the cluster (like the device plugin) can't see the GPU even though `docker exec kind-control-plane nvidia-smi` already works.
2. [k8s/nvidia-device-plugin.yaml](k8s/nvidia-device-plugin.yaml) — a pinned snapshot of NVIDIA's official k8s-device-plugin DaemonSet (captured resolving to `k8s-device-plugin:v0.20.0`; the upstream `main` branch is a moving target, hence pinning a copy here instead of always `kubectl create -f <url>`). This is what makes `nvidia.com/gpu` show up as an allocatable resource on the node — it needs to be **reinstalled every time the cluster is recreated** (`kind delete cluster` wipes the original `kube-system` namespace, taking the device plugin with it).
3. ARC's `arc-runner-set-gpu` requests `resources.limits.nvidia.com/gpu: 1` in its pod template, which the scheduler can now satisfy.

### ARC components (namespace `arc-systems`)

- **Controller** (`arc`, chart `gha-runner-scale-set-controller`) — watches GitHub and creates/destroys runner pods on demand.
- **`arc-runner-set`** — the non-GPU runner set, no custom values needed.
- **`arc-runner-set-gpu`** — the GPU runner set, needs one of the two values files above.
- Auth: a `Secret` named `pre-defined-secret` in `arc-systems`, holding `github_token` (a fine-grained PAT with **Repository permissions → Administration: Read and write** on the target repo) — created by `scripts/2-setup-arc.sh`, never committed anywhere.
- `k8s/gpu-runner-values.yaml`'s `image:` still points at a placeholder (`<seu-registry>/pytorch-gpu-sandbox-runner:latest`) — build+push [k8s/runner-image/Dockerfile](k8s/runner-image/Dockerfile) and update that field before relying on the python variant; until then, `pytorch-gpu-python.yaml` bootstraps pip from scratch every run via `get-pip.py` (see pip-install gotchas table below) — slow on a constrained connection, since `torch`'s CUDA wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-cufft-cu12`, `nvidia-nccl-cu12`, ...) easily total 3–5 GB and get re-downloaded every run until the pip cache (`actions/cache` on `~/.cache/pip`) is warm.

### `scripts/` — cluster lifecycle automation

All three resolve their own directory (`$BASH_SOURCE`), so they work from any cwd once the repo is cloned, and read `k8s/*.yaml` directly as the single source of truth (no more regenerating drifting copies under `~/`, which is what an earlier version of these scripts did).

- **`1-create-gpu-cluster.sh`** — `kind delete cluster --name kind` (ignored if none exists) then `kind create cluster --config k8s/kind-gpu-config.yaml`. **Destructive**: wipes any existing `kind` cluster and everything running on it. Prints `docker exec kind-control-plane nvidia-smi` at the end to confirm GPU passthrough worked.
- **`2-setup-arc.sh REPO [python|docker]`** — installs the device plugin + ARC (controller + both runner sets). `REPO` is `owner/repo` (e.g. `Lorkyrr/pytorch-gpu-sandbox`); requires `GITHUB_TOKEN` exported first. Variant arg picks which `k8s/gpu-runner-values*.yaml` to template (via `sed`, substituting `githubConfigUrl`) — **default is `docker`** (works immediately, no image to build first), even though the *checked-in default* for the repo's own CI is `python` (see table above) — switch to `python` explicitly once `k8s/runner-image/Dockerfile` is built and pushed. Runs two things in parallel for speed: Track A (device plugin install + wait + node-capacity poll) and Track B (namespace/secret → controller install/wait → the two runner-set `helm upgrade`s, which are themselves backgrounded against each other since neither depends on the other). Each track buffers its own log and is only printed/checked for failure after both finish, so output doesn't interleave mid-command.
- **`3-teardown-cluster.sh`** — `kind delete cluster --name kind`, then prints `nvidia-smi --query-compute-apps=...` to confirm the GPU is free. Nothing needs regenerating afterward since the config lives in `k8s/`.
- **`kubernetes-gpu-arc-referencia.sh`** — a heavily commented, section-by-section walkthrough of the *entire* manual process (host prerequisites through debugging commands), meant to be read and understood, not executed top-to-bottom. Ends with the gotchas table reproduced below.

[.vscode/tasks.json](.vscode/tasks.json) wraps the first three as VS Code tasks: "K8s: criar cluster GPU", "K8s: configurar ARC (device plugin + runners)", "K8s: recriar cluster do zero" (chains the previous two, prompting for `GITHUB_TOKEN` once via a `promptString` input — never hardcoded), "K8s: derrubar cluster", and "K8s: ver pods do ARC" (`kubectl get pods -n arc-systems`).

### Known gotchas already solved (don't rediscover these)

From building the `kind` + GPU + ARC stack — condensed from `scripts/kubernetes-gpu-arc-referencia.sh`'s own error log:

| Symptom | Root cause | Fix |
|---|---|---|
| `field gpus not found in type v1alpha4.Node` | A `gpus: true` field in the `kind` config that doesn't actually exist | GPU access comes from the `nvidia` runtime being the host Docker's default (Container Toolkit config), not a special `kind` YAML field |
| `exec: nvidia-smi: executable file not found in $PATH` inside the node | Missing `accept-nvidia-visible-devices-as-volume-mounts = true` in `/etc/nvidia-container-runtime/config.toml`, and/or missing the `/var/run/nvidia-container-devices/all` extraMount | Enable that config line + the extraMount in `k8s/kind-gpu-config.yaml` |
| Device plugin pod `Error`/`CrashLoopBackOff`, `ERROR_LIBRARY_NOT_FOUND` | The node's **internal** containerd (not the host Docker) wasn't configured with the `nvidia` runtime | `containerdConfigPatches` in `k8s/kind-gpu-config.yaml` (`default_runtime_name = "nvidia"`) |
| `0/1 nodes are available: 1 Insufficient nvidia.com/gpu` | Either another pod already holds the only GPU, or the device plugin died/vanished (common right after recreating the cluster — it doesn't survive `kind delete cluster`) | Reinstall `k8s/nvidia-device-plugin.yaml`; `2-setup-arc.sh` does this automatically |
| `MountVolume.SetUp failed for volume docker-sock: ... does not exist` | `/var/run/docker.sock` wasn't brought into the `kind` node via `extraMounts` | Add that extraMount in `k8s/kind-gpu-config.yaml` — the runner pod's "host" is the node container, not your actual machine |
| `permission denied while trying to connect to the docker API` | The runner image's default non-root user can't touch the mounted Docker socket | `securityContext.runAsUser: 0` in the values file (docker variant only) |
| GPU runner pod crash-loops fast (`Running` ~1s → `Error`), no clear log | The runner image refuses to start as root even with `runAsUser: 0` set | Also set env `RUNNER_ALLOW_RUNASROOT: "1"` |
| `ModuleNotFoundError: No module named 'torch'` in the python-variant workflow | The stock runner image has no PyTorch (the custom `k8s/runner-image/Dockerfile` hasn't been built/pushed yet) | Either build+push that image, or accept the current fallback (installs everything at job start — slow) |
| `/usr/bin/python3: No module named pip` | Debian/Ubuntu strip `ensurepip` from the `python3` package on purpose | Bootstrap via PyPA's `get-pip.py` |
| `error: externally-managed-environment` (PEP 668) | Debian/Ubuntu block `pip install` outside a venv by default | `--break-system-packages` — safe here because the whole pod is thrown away after every job, it's not a persistent system to break |
| A "fixed" workflow still fails with old code, even though `git log origin/main` shows the fix is pushed | GitHub Actions' **"Re-run jobs"** button re-runs the *exact same run*, which is permanently bound to whatever commit SHA was at `origin/main` when that run was originally created — it never picks up newer commits | Use **"Run workflow"** on the workflow's own page instead (creates a brand-new run against current `HEAD`); verify by comparing the SHA printed in the run's "Checkout do código" step against `git log --oneline -1` |

Full narrative (with the actual commands and dead-ends) for all of these is in [SAGA-DA-RTX3050.md](SAGA-DA-RTX3050.md), chapters 5, 6, 7, 10 and 11 respectively.

### VS Code integration

[.vscode/launch.json](.vscode/launch.json)'s "Python: Anexar ao container (Docker debug)" attach config has [compose.debug.yaml](compose.debug.yaml) as its `preLaunchTask`; that compose file echoes the `[debugpy] pronto para anexar` marker between installing debugpy and starting it, which the task's background problem matcher watches for so VS Code doesn't try to attach before the port is actually listening. The other launch config, "Python: Arquivo atual (local, sem Docker)", just runs whatever file is open through the debugger directly — no container involved.
