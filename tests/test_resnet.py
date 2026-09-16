"""Testes que dependem de torch — pulados automaticamente se torch não estiver
instalado (é o caso do runner hospedado do GitHub, de propósito: essa suíte só
roda de verdade no runner self-hosted, que já tem torch via requirements.txt)."""

import pytest

torch = pytest.importorskip("torch")

from pytorch_gpu_sandbox.constants import RESNET_VARIANTES_BLOCOS  # noqa: E402
from pytorch_gpu_sandbox.models.resnet import BlocoResidual, construir_resnet_cifar  # noqa: E402
from pytorch_gpu_sandbox.training import resolver_amp_habilitado  # noqa: E402


@pytest.mark.parametrize("variante", sorted(RESNET_VARIANTES_BLOCOS))
def test_construir_resnet_cifar_produz_logits_para_todas_as_variantes(variante):
    modelo = construir_resnet_cifar(variante, num_classes=10)
    entrada = torch.randn(2, 3, 32, 32)
    saida = modelo(entrada)
    assert saida.shape == (2, 10)


def test_construir_resnet_cifar_rejeita_variante_desconhecida():
    with pytest.raises(ValueError):
        construir_resnet_cifar("resnet-inexistente")


def test_bloco_residual_usa_atalho_de_projecao_quando_muda_canais():
    bloco = BlocoResidual(canais_entrada=16, canais_saida=32, stride=2)
    assert len(bloco.atalho) == 2  # Conv2d + BatchNorm2d


def test_bloco_residual_usa_atalho_identidade_quando_stride_e_canais_nao_mudam():
    bloco = BlocoResidual(canais_entrada=16, canais_saida=16, stride=1)
    assert len(bloco.atalho) == 0


def test_resolver_amp_habilitado_desativa_em_cpu_mesmo_se_pedido():
    assert resolver_amp_habilitado(True, torch.device("cpu")) is False


def test_resolver_amp_habilitado_respeita_flag_desligada_em_cuda():
    assert resolver_amp_habilitado(False, torch.device("cuda")) is False
