"""Testes puros (sem torch) — rodam no runner hospedado do GitHub."""

from pytorch_gpu_sandbox.metrics import gflops_matmul


def test_gflops_matmul_calcula_proporcional_ao_tempo():
    # 2 * 1000^3 flops / 1s = 2e9 flops/s = 2 GFLOPS
    assert gflops_matmul(1000, 1.0) == 2.0


def test_gflops_matmul_tempo_zero_retorna_zero():
    assert gflops_matmul(1000, 0.0) == 0.0
