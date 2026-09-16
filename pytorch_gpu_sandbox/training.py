"""Laços de treino/avaliação e o pipeline completo de treino da ResNet no CIFAR-10.

A precisão mista (AMP) aqui é real — usada no laço de treino de verdade via
`autocast` + `GradScaler` — e não deve ser confundida com `teste_precisao_mista`
em benchmarks.py, que é só um micro-benchmark isolado de um `matmul`.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from .constants import (
    RESNET_NUM_CLASSES_PADRAO,
    RESNET_VARIANTE_PADRAO,
    TREINO_BATCH_SIZE_PADRAO,
    TREINO_DATA_DIR_PADRAO,
    TREINO_EPOCAS_PADRAO,
    TREINO_LR_GAMMA,
    TREINO_LR_MILESTONES_FRACOES,
    TREINO_LR_PADRAO,
    TREINO_SGD_MOMENTUM,
    TREINO_SGD_WEIGHT_DECAY,
)
from .data import cifar10_dataloaders
from .formatting import secao, titulo
from .models.resnet import construir_resnet_cifar


def resolver_amp_habilitado(usar_amp: bool, device: torch.device) -> bool:
    """AMP só compensa em CUDA — a CPU não tem os Tensor Cores que justificam FP16 aqui."""
    return usar_amp and device.type == "cuda"


def treinar_uma_epoca(
    modelo: nn.Module,
    loader,
    otimizador: torch.optim.Optimizer,
    perda_fn: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_habilitado: bool,
) -> tuple[float, float]:
    modelo.train()
    perda_total, acertos, total = 0.0, 0, 0

    for entradas, rotulos in loader:
        entradas = entradas.to(device, non_blocking=True)
        rotulos = rotulos.to(device, non_blocking=True)

        otimizador.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_habilitado):
            saidas = modelo(entradas)
            perda = perda_fn(saidas, rotulos)

        scaler.scale(perda).backward()
        scaler.step(otimizador)
        scaler.update()

        perda_total += perda.item() * entradas.size(0)
        _, previstos = saidas.max(1)
        total += rotulos.size(0)
        acertos += previstos.eq(rotulos).sum().item()

    return perda_total / total, 100.0 * acertos / total


@torch.no_grad()
def avaliar(
    modelo: nn.Module,
    loader,
    perda_fn: nn.Module,
    device: torch.device,
    amp_habilitado: bool,
) -> tuple[float, float]:
    modelo.eval()
    perda_total, acertos, total = 0.0, 0, 0

    for entradas, rotulos in loader:
        entradas = entradas.to(device, non_blocking=True)
        rotulos = rotulos.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_habilitado):
            saidas = modelo(entradas)
            perda = perda_fn(saidas, rotulos)

        perda_total += perda.item() * entradas.size(0)
        _, previstos = saidas.max(1)
        total += rotulos.size(0)
        acertos += previstos.eq(rotulos).sum().item()

    return perda_total / total, 100.0 * acertos / total


def treinar_resnet_cifar10(
    device: torch.device,
    epocas: int = TREINO_EPOCAS_PADRAO,
    batch_size: int = TREINO_BATCH_SIZE_PADRAO,
    lr: float = TREINO_LR_PADRAO,
    data_dir: str = TREINO_DATA_DIR_PADRAO,
    checkpoint_path: str | None = None,
    arquitetura: str = RESNET_VARIANTE_PADRAO,
    usar_amp: bool = True,
    seed: int | None = None,
) -> None:
    titulo("TREINAMENTO: RESNET NO CIFAR-10")

    if seed is not None:
        torch.manual_seed(seed)
        print(f"[INFO] Semente fixada em: {seed}")

    if checkpoint_path is None:
        checkpoint_path = f"{arquitetura}_cifar10.pth"

    amp_habilitado = resolver_amp_habilitado(usar_amp, device)

    print(f"\n[INFO] Arquitetura: {arquitetura} | Épocas: {epocas} | Batch size: {batch_size} | LR inicial: {lr}")
    print(f"[INFO] Precisão mista (AMP): {'ativada' if amp_habilitado else 'desativada'}")
    print("[INFO] Baixando/preparando o CIFAR-10 (se necessário)...")
    loader_treino, loader_teste = cifar10_dataloaders(data_dir=data_dir, batch_size=batch_size)

    modelo = construir_resnet_cifar(arquitetura, num_classes=RESNET_NUM_CLASSES_PADRAO).to(device)
    num_parametros = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"[INFO] Parâmetros treináveis: {num_parametros:,}")

    perda_fn = nn.CrossEntropyLoss()
    otimizador = torch.optim.SGD(
        modelo.parameters(),
        lr=lr,
        momentum=TREINO_SGD_MOMENTUM,
        weight_decay=TREINO_SGD_WEIGHT_DECAY,
        nesterov=True,
    )
    milestones = [int(epocas * fracao) for fracao in TREINO_LR_MILESTONES_FRACOES]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(otimizador, milestones=milestones, gamma=TREINO_LR_GAMMA)
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_habilitado)

    melhor_acc = 0.0

    secao("Progresso do treinamento")
    for epoca in range(1, epocas + 1):
        inicio = time.time()
        perda_treino, acc_treino = treinar_uma_epoca(
            modelo, loader_treino, otimizador, perda_fn, device, scaler, amp_habilitado
        )
        perda_teste, acc_teste = avaliar(modelo, loader_teste, perda_fn, device, amp_habilitado)
        scheduler.step()
        tempo = time.time() - inicio

        print(
            f"[Época {epoca:03d}/{epocas}] "
            f"treino_loss={perda_treino:.4f} treino_acc={acc_treino:.2f}% | "
            f"teste_loss={perda_teste:.4f} teste_acc={acc_teste:.2f}% | "
            f"{tempo:.1f}s"
        )

        if acc_teste > melhor_acc:
            melhor_acc = acc_teste
            torch.save(modelo.state_dict(), checkpoint_path)

    secao("Resultado do treinamento")
    print(f" -> Melhor acurácia no teste: {melhor_acc:.2f}%")
    print(f" -> Checkpoint salvo em: {checkpoint_path}")
