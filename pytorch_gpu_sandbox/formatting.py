"""Utilidades de apresentação no console — só formatam texto, sem lógica de negócio."""

from __future__ import annotations

LARGURA_PADRAO = 70


def linha(char: str = "=", tamanho: int = LARGURA_PADRAO) -> None:
    print(char * tamanho)


def titulo(texto: str) -> None:
    linha()
    print(texto.center(LARGURA_PADRAO))
    linha()


def secao(texto: str) -> None:
    print("\n" + "-" * LARGURA_PADRAO)
    print(texto)
    print("-" * LARGURA_PADRAO)
