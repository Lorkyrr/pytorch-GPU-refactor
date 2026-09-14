# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning project (author is a beginner) to practice PyTorch, GPU/CUDA programming, and Docker/Kubernetes. It's a single-script repo: [main.py](main.py) has two modes:

- `benchmark` (default) — GPU/CUDA/cuDNN sanity checks and micro-benchmarks (matmul, conv2d, mixed precision, a simulated training loop)
- `train` — trains a from-scratch ResNet-20 (He et al., 2015 CIFAR variant, not `torchvision.models`) on CIFAR-10

Comments and console output in `main.py` are in Portuguese; keep that convention when editing it.

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

## Architecture

### main.py layout
- `info_ambiente`, `teste_matmul`, `teste_convolucao`, `teste_precisao_mista`, `teste_treino_simulado`, `executar_benchmark` — the benchmark-mode checks, run in sequence.
- `BlocoResidual` / `ResNetCIFAR` / `resnet20_cifar` — the from-scratch ResNet-20 model.
- `cifar10_dataloaders`, `treinar_uma_epoca`, `avaliar`, `treinar_resnet_cifar10` — the train-mode pipeline (downloads CIFAR-10 into `./data` on first run, saves the best checkpoint by test accuracy to `resnet20_cifar10.pth`).
- `main()` dispatches on the positional `benchmark`/`train` argument.

### CI on a real GPU (Kubernetes + Actions Runner Controller)

CI doesn't use GitHub-hosted runners (no free GPU tier) — it runs on a **self-hosted runner backed by the author's own RTX 3050**, via Actions Runner Controller (ARC) on a local `kind` cluster. This is the part of the repo that requires reading multiple files together to understand:

- Two GitHub Actions workflow variants exist side by side, but **only one can be live at a time** because they require different Helm values applied to the same `arc-runner-set-gpu` scale-set deployment:
  - [.github/workflows/pytorch-gpu-python.yaml](.github/workflows/pytorch-gpu-python.yaml) — **current/default**. Runs `python3 main.py ...` natively inside the runner pod (no Docker). Requires [k8s/gpu-runner-values.yaml](k8s/gpu-runner-values.yaml): the runner pod's own image (built from [k8s/runner-image/Dockerfile](k8s/runner-image/Dockerfile)) bundles PyTorch, and that pod's `nvidia.com/gpu: 1` reservation is what actually trains.
  - [.github/workflows/pytorch-gpu-docker.yaml](.github/workflows/pytorch-gpu-docker.yaml) — reference/fallback. Builds [Dockerfile](Dockerfile) and runs `docker run --gpus all` on the runner, extracting results with `docker cp` (the Docker daemon doesn't share the pod's filesystem). Requires [k8s/gpu-runner-values.docker.yaml](k8s/gpu-runner-values.docker.yaml) instead, which bind-mounts the host's Docker socket into the pod — the training container then runs against the host daemon, outside the pod's own GPU reservation.
  - [.github/workflows/teste-gpu.yaml](.github/workflows/teste-gpu.yaml) — minimal `nvidia-smi` sanity check, works under either config.
- To switch which variant is live:
  ```bash
  helm upgrade --install arc-runner-set-gpu \
    oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
    -n arc-runners -f k8s/gpu-runner-values.yaml           # python (default)
    # -f k8s/gpu-runner-values.docker.yaml                 # or: docker
  ```
- The `k8s/*.yaml` files describe intended cluster state; they are applied by hand (`helm upgrade` / `kubectl apply`), not automatically from the repo:
  - [k8s/kind-gpu-config.yaml](k8s/kind-gpu-config.yaml) — passes the host RTX 3050 into the `kind` node.
  - [k8s/nvidia-device-plugin.yaml](k8s/nvidia-device-plugin.yaml) — snapshot of the NVIDIA k8s-device-plugin DaemonSet, applied via plain `kubectl create -f <url>` (no Helm/GPU Operator).
- `k8s/gpu-runner-values.yaml`'s `image:` still points at a placeholder (`<seu-registry>/pytorch-gpu-sandbox-runner:latest`) — build/push `k8s/runner-image/Dockerfile` and update that field before relying on it; until then `pytorch-gpu-python.yaml` falls back to bootstrapping pip via `get-pip.py` (the runner's Python has none, and Debian/Ubuntu also blocks `pip install` outside a venv per PEP 668, hence `--break-system-packages`) and `pip install -r requirements.txt` at job start — slow on a constrained connection since it re-downloads the CUDA wheels every run until that image exists.
- Sibling top-level scripts outside this repo ([1-create-gpu-cluster.sh](../1-create-gpu-cluster.sh), [2-setup-arc.sh](../2-setup-arc.sh), [3-teardown-cluster.sh](../3-teardown-cluster.sh), [kubernetes-gpu-arc-referencia.sh](../kubernetes-gpu-arc-referencia.sh)) create/tear down the `kind` + ARC cluster this CI setup depends on. [.vscode/tasks.json](.vscode/tasks.json) wraps them as VS Code tasks ("K8s: criar cluster GPU", "K8s: configurar ARC", "K8s: recriar cluster do zero", "K8s: derrubar cluster") — `GITHUB_TOKEN` is only prompted for by the tasks that actually need it (create/recreate), via a `promptString` input, not hardcoded anywhere.
- [.vscode/launch.json](.vscode/launch.json)'s "Python: Anexar ao container" attach config has [compose.debug.yaml](compose.debug.yaml) as its `preLaunchTask`; that compose file now echoes a `[debugpy] pronto para anexar` marker between installing debugpy and starting it, which the task's background problem matcher watches for so VS Code doesn't try to attach before the port is actually listening.
