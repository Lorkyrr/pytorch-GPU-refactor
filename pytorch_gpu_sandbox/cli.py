"""Ponto de entrada da CLI: parseia argumentos e despacha para benchmark ou treino."""

from __future__ import annotations

import argparse

from .benchmarks import executar_benchmark
from .constants import (
    RESNET_VARIANTE_PADRAO,
    RESNET_VARIANTES_BLOCOS,
    TREINO_BATCH_SIZE_PADRAO,
    TREINO_DATA_DIR_PADRAO,
    TREINO_EPOCAS_PADRAO,
    TREINO_LR_PADRAO,
)
from .environment import info_ambiente
from .training import treinar_resnet_cifar10


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Teste de ambiente PyTorch/GPU e treinamento de ResNet no CIFAR-10"
    )
    parser.add_argument(
        "modo",
        nargs="?",
        choices=["benchmark", "train"],
        default="benchmark",
        help="'benchmark' roda os testes de GPU (padrão); 'train' treina uma ResNet no CIFAR-10",
    )
    parser.add_argument("--epochs", type=int, default=TREINO_EPOCAS_PADRAO, help="número de épocas de treino")
    parser.add_argument("--batch-size", type=int, default=TREINO_BATCH_SIZE_PADRAO, help="tamanho do lote")
    parser.add_argument("--lr", type=float, default=TREINO_LR_PADRAO, help="taxa de aprendizado inicial")
    parser.add_argument("--data-dir", type=str, default=TREINO_DATA_DIR_PADRAO, help="diretório do CIFAR-10")
    parser.add_argument(
        "--arquitetura",
        type=str,
        choices=sorted(RESNET_VARIANTES_BLOCOS),
        default=RESNET_VARIANTE_PADRAO,
        help="variante da ResNet a treinar (padrão: resnet20)",
    )
    parser.add_argument(
        "--sem-amp",
        action="store_true",
        help="desativa a precisão mista (AMP) no treino, mesmo com GPU disponível",
    )
    parser.add_argument("--seed", type=int, default=None, help="semente para reprodutibilidade (opcional)")
    return parser


def main() -> None:
    args = construir_parser().parse_args()

    device = info_ambiente()

    if args.modo == "train":
        treinar_resnet_cifar10(
            device,
            epocas=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            data_dir=args.data_dir,
            arquitetura=args.arquitetura,
            usar_amp=not args.sem_amp,
            seed=args.seed,
        )
        return

    executar_benchmark(device)


if __name__ == "__main__":
    main()
