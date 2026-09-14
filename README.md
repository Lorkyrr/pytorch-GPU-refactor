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

## Requirements

- Docker with NVIDIA Container Toolkit installed (for GPU access)
- An NVIDIA GPU with recent drivers

No local Python setup is required if you use Docker; the image already bundles PyTorch, torchvision, and torchaudio.

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

[compose.debug.yaml](compose.debug.yaml) starts the app under `debugpy`, listening on port `5678`, so you can attach a remote debugger (e.g. VS Code's "Python: Remote Attach") before execution continues:

```bash
docker compose -f compose.debug.yaml up --build
```

## Running without Docker

If you already have a CUDA-enabled PyTorch installed locally:

```bash
pip install -r requirements.txt
python main.py
```

Note: `requirements.txt` is currently empty — populate it with `pip freeze > requirements.txt` from your working environment if you want a pinned local setup.

## Project layout

- [main.py](main.py) — the environment/GPU test script (comments and output are in Portuguese)
- [Dockerfile](Dockerfile) — builds on `pytorch/pytorch:latest` and installs torchvision/torchaudio
- [compose.yaml](compose.yaml) — runs the container with GPU access
- [compose.debug.yaml](compose.debug.yaml) — runs the container with `debugpy` for remote debugging
- [TESTE_DE_AMBIENTE_LOCAL_1.txt](TESTE_DE_AMBIENTE_LOCAL_1.txt) / [TESTE_DE_AMBIENTE_LOCAL_2.txt](TESTE_DE_AMBIENTE_LOCAL_2.txt) — real output logs from two local training runs (see [Local test results](#local-test-results))