"""Constantes centralizadas do projeto — nenhum "número mágico" espalhado pelo código.

Os valores aqui (tamanhos de matriz, batch sizes, hiperparâmetros) foram
escolhidos originalmente pensando em uma RTX 3050 de 4 GB de VRAM — veja
CLAUDE.md para o raciocínio por trás de cada escolha.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Ambiente / Tensor Cores
# ----------------------------------------------------------------------
COMPUTE_CAPABILITY_MINIMA_TENSOR_CORES = 7

# ----------------------------------------------------------------------
# Teste 1: multiplicação de matrizes (cuBLAS)
# ----------------------------------------------------------------------
MATMUL_TAMANHOS = (1000, 2000, 4000, 6000)

# ----------------------------------------------------------------------
# Teste 2: convolução em lote (cuDNN)
# ----------------------------------------------------------------------
CONV_BATCH_SIZE = 16
CONV_IMAGEM_LADO = 224
CONV_REPETICOES = 20

# ----------------------------------------------------------------------
# Teste 3: precisão mista (FP16 / autocast) — micro-benchmark isolado
# ----------------------------------------------------------------------
PRECISAO_MISTA_MATRIZ_TAMANHO = 4000

# ----------------------------------------------------------------------
# Teste 4: ciclo de treino simulado
# ----------------------------------------------------------------------
TREINO_SIMULADO_BATCH_SIZE = 32
TREINO_SIMULADO_IMAGEM_LADO = 64
TREINO_SIMULADO_NUM_CLASSES = 10
TREINO_SIMULADO_EPOCAS = 30
TREINO_SIMULADO_LR = 1e-3

# ----------------------------------------------------------------------
# ResNet-CIFAR (He et al., 2015) — família ResNet-(6n+2)
# ----------------------------------------------------------------------
RESNET_CANAIS_INICIAIS = 16
RESNET_NUM_CLASSES_PADRAO = 10

# n blocos por estágio -> profundidade 6n+2 (contando o stem e a camada final)
RESNET_VARIANTES_BLOCOS = {
    "resnet20": 3,
    "resnet56": 9,
    "resnet110": 18,
}
RESNET_VARIANTE_PADRAO = "resnet20"

# ----------------------------------------------------------------------
# Dados: CIFAR-10
# ----------------------------------------------------------------------
CIFAR10_MEDIA = (0.4914, 0.4822, 0.4465)
CIFAR10_DESVIO_PADRAO = (0.2470, 0.2435, 0.2616)
CIFAR10_RANDOM_CROP_PADDING = 4
CIFAR10_NUM_WORKERS_PADRAO = 2

# ----------------------------------------------------------------------
# Treino da ResNet no CIFAR-10 — padrões da CLI
# ----------------------------------------------------------------------
TREINO_EPOCAS_PADRAO = 30
TREINO_BATCH_SIZE_PADRAO = 128
TREINO_LR_PADRAO = 0.1
TREINO_DATA_DIR_PADRAO = "./data"
TREINO_SGD_MOMENTUM = 0.9
TREINO_SGD_WEIGHT_DECAY = 5e-4
TREINO_LR_MILESTONES_FRACOES = (0.5, 0.75)
TREINO_LR_GAMMA = 0.1
