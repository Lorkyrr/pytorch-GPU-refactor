"""Métricas de desempenho, propositalmente sem dependência de torch.

Mantido separado de benchmarks.py (que faz `import torch` no topo) para que
essa lógica possa ser testada em qualquer ambiente Python puro — inclusive
o runner hospedado do GitHub, que nunca tem torch instalado de propósito
(ver .github/workflows/validacao-repositorio.yaml).
"""

from __future__ import annotations


def gflops_matmul(n: int, tempo_s: float) -> float:
    """Estimativa de GFLOPS para multiplicação de matrizes NxN."""
    flops = 2 * (n**3)
    return (flops / tempo_s) / 1e9 if tempo_s > 0 else 0.0
