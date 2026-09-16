"""ResNet para CIFAR-10 (arquitetura de He et al., 2015 — variante CIFAR, família 6n+2)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..constants import RESNET_CANAIS_INICIAIS, RESNET_NUM_CLASSES_PADRAO, RESNET_VARIANTES_BLOCOS


class BlocoResidual(nn.Module):
    """Bloco básico (3x3 -> 3x3) com atalho (shortcut) identidade ou projeção."""

    expansao = 1

    def __init__(self, canais_entrada: int, canais_saida: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(canais_entrada, canais_saida, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(canais_saida)
        self.conv2 = nn.Conv2d(canais_saida, canais_saida, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(canais_saida)
        self.relu = nn.ReLU(inplace=True)

        self.atalho = nn.Sequential()
        if stride != 1 or canais_entrada != canais_saida:
            self.atalho = nn.Sequential(
                nn.Conv2d(canais_entrada, canais_saida, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(canais_saida),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        saida = self.relu(self.bn1(self.conv1(x)))
        saida = self.bn2(self.conv2(saida))
        saida = saida + self.atalho(x)
        return self.relu(saida)


class ResNetCIFAR(nn.Module):
    """ResNet-(6n+2) para imagens 32x32, como no paper original do ResNet.

    Com blocos_por_estagio=3 obtém-se a ResNet-20; 9 -> ResNet-56; 18 -> ResNet-110.
    """

    def __init__(
        self,
        blocos_por_estagio: int = RESNET_VARIANTES_BLOCOS["resnet20"],
        num_classes: int = RESNET_NUM_CLASSES_PADRAO,
    ) -> None:
        super().__init__()
        self.canais_entrada = RESNET_CANAIS_INICIAIS

        self.conv1 = nn.Conv2d(3, RESNET_CANAIS_INICIAIS, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(RESNET_CANAIS_INICIAIS)
        self.relu = nn.ReLU(inplace=True)

        self.estagio1 = self._criar_estagio(16, blocos_por_estagio, stride=1)
        self.estagio2 = self._criar_estagio(32, blocos_por_estagio, stride=2)
        self.estagio3 = self._criar_estagio(64, blocos_por_estagio, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _criar_estagio(self, canais_saida: int, num_blocos: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocos - 1)
        camadas = []
        for s in strides:
            camadas.append(BlocoResidual(self.canais_entrada, canais_saida, s))
            self.canais_entrada = canais_saida
        return nn.Sequential(*camadas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        saida = self.relu(self.bn1(self.conv1(x)))
        saida = self.estagio1(saida)
        saida = self.estagio2(saida)
        saida = self.estagio3(saida)
        saida = self.avgpool(saida)
        saida = torch.flatten(saida, 1)
        return self.fc(saida)


def construir_resnet_cifar(variante: str, num_classes: int = RESNET_NUM_CLASSES_PADRAO) -> ResNetCIFAR:
    """Constrói uma ResNet-(6n+2) pelo nome da variante (resnet20 / resnet56 / resnet110)."""
    if variante not in RESNET_VARIANTES_BLOCOS:
        opcoes = ", ".join(sorted(RESNET_VARIANTES_BLOCOS))
        raise ValueError(f"Variante de ResNet desconhecida: {variante!r}. Opções: {opcoes}")
    return ResNetCIFAR(blocos_por_estagio=RESNET_VARIANTES_BLOCOS[variante], num_classes=num_classes)
