# syntax=docker/dockerfile:1
#
# Imagem do runner do ARC (arc-runner-set-gpu) com Python + PyTorch/CUDA
# já instalados, pra rodar main.py NATIVAMENTE dentro do pod do runner —
# sem Docker aninhado, sem socket do host. A GPU chega sozinha: o pod
# pede `nvidia.com/gpu: 1` (ver ../gpu-runner-values.yaml) e o
# nvidia-container-runtime do node (ver ../kind-gpu-config.yaml) injeta
# driver, /dev/nvidia*, libcuda.so e o próprio nvidia-smi no container —
# não precisa de CUDA Toolkit instalado aqui, os wheels do PyTorch já
# trazem o runtime CUDA (cuBLAS, cuDNN, NCCL etc.) embutido.
#
# Build & push (feito manualmente, não faz parte do CI). Contexto é a
# raiz do repo (precisa do requirements.txt de lá), não esta pasta:
#   docker build -t <seu-registry>/pytorch-gpu-sandbox-runner:latest \
#     -f k8s/runner-image/Dockerfile .
#   docker push <seu-registry>/pytorch-gpu-sandbox-runner:latest
#
# Depois aponte gpu-runner-values.yaml -> template.spec.containers[0].image
# pra essa tag.

FROM ghcr.io/actions/actions-runner:latest

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
# Wheels de torch com CUDA passam de 500MB cada, e numa conexão lenta o
# download de um único wheel pode ultrapassar o timeout padrão do pip
# (ver capítulo 12 do SAGA-DA-RTX3050.md) mesmo com --timeout generoso.
# --mount=type=cache mantém o cache HTTP do pip (/root/.cache/pip) entre
# tentativas de build (mesmo builder, mesmo host) — se o build falhar no
# meio (ex: um wheel específico não termina de baixar) e for repetido, os
# pacotes que já baixaram com sucesso não precisam ser baixados de novo.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages \
    --timeout 600 --retries 20 -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

USER runner
