"""Detecção e relatório do ambiente PyTorch/CUDA."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .constants import COMPUTE_CAPABILITY_MINIMA_TENSOR_CORES
from .formatting import titulo


@dataclass(frozen=True)
class InfoGPU:
    """Snapshot dos metadados da GPU 0, coletado uma única vez por execução."""

    nome: str
    memoria_total_gb: float
    compute_capability: str
    num_multiprocessadores: int
    cudnn_habilitado: bool
    versao_cudnn: int
    suporta_tensor_cores: bool


def mem_gpu_mb() -> tuple[float, float]:
    """Retorna (alocada, reservada) em MB."""
    alocada = torch.cuda.memory_allocated() / 1e6
    reservada = torch.cuda.memory_reserved() / 1e6
    return alocada, reservada


def coletar_info_gpu() -> InfoGPU:
    """Lê os metadados da GPU 0 via API do CUDA. Assume que CUDA já está disponível."""
    props = torch.cuda.get_device_properties(0)
    return InfoGPU(
        nome=torch.cuda.get_device_name(0),
        memoria_total_gb=props.total_memory / 1e9,
        compute_capability=f"{props.major}.{props.minor}",
        num_multiprocessadores=props.multi_processor_count,
        cudnn_habilitado=torch.backends.cudnn.enabled,
        versao_cudnn=torch.backends.cudnn.version(),
        suporta_tensor_cores=props.major >= COMPUTE_CAPABILITY_MINIMA_TENSOR_CORES,
    )


def info_ambiente() -> torch.device:
    """Imprime o relatório do ambiente e devolve o `device` a ser usado nos testes."""
    titulo("TESTE DE AMBIENTE LOCAL - PYTORCH + GPU NVIDIA")

    print(f"\n[INFO] Versão do PyTorch: {torch.__version__}")
    print(f"[INFO] PyTorch compilado com CUDA: {torch.version.cuda}")

    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_ok else "cpu")
    print(f"[INFO] CUDA disponível: {cuda_ok}")

    if not cuda_ok:
        print("\n[AVISO] CUDA não está disponível. O teste rodará na CPU.")
        print("Verifique se o container foi iniciado com '--gpus all' e se")
        print("a imagem base tem suporte a CUDA (ex: nvidia/cuda ou pytorch/pytorch com tag 'cuda').")
        return device

    info = coletar_info_gpu()
    print(f"[INFO] cuDNN habilitado: {info.cudnn_habilitado}")
    print(f"[INFO] Versão do cuDNN: {info.versao_cudnn}")
    print(f"[INFO] GPU detectada: {info.nome}")
    print(f"[INFO] Memória total da GPU: {info.memoria_total_gb:.2f} GB")
    print(f"[INFO] Compute Capability: {info.compute_capability}")
    print(f"[INFO] Multiprocessadores (SMs): {info.num_multiprocessadores}")
    print(f"[INFO] Suporte a Tensor Cores (FP16 acelerado): {'Sim' if info.suporta_tensor_cores else 'Não'}")

    return device
