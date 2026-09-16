"""Os 4 testes de benchmark de GPU (cuBLAS, cuDNN, precisão mista, treino simulado)."""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from .constants import (
    CONV_BATCH_SIZE,
    CONV_IMAGEM_LADO,
    CONV_REPETICOES,
    MATMUL_TAMANHOS,
    PRECISAO_MISTA_MATRIZ_TAMANHO,
    TREINO_SIMULADO_BATCH_SIZE,
    TREINO_SIMULADO_EPOCAS,
    TREINO_SIMULADO_IMAGEM_LADO,
    TREINO_SIMULADO_LR,
    TREINO_SIMULADO_NUM_CLASSES,
)
from .environment import mem_gpu_mb
from .formatting import secao, titulo
from .metrics import gflops_matmul


def teste_matmul(device: torch.device) -> None:
    secao("[TESTE 1/4] Multiplicação de matrizes (cuBLAS) em múltiplas escalas")

    for n in MATMUL_TAMANHOS:
        try:
            a = torch.randn(n, n, device=device)
            b = torch.randn(n, n, device=device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            inicio = time.time()
            c = torch.mm(a, b)
            if device.type == "cuda":
                torch.cuda.synchronize()
            fim = time.time()

            tempo = fim - inicio
            gflops = gflops_matmul(n, tempo)

            print(f" -> Matriz {n}x{n}: {tempo:.4f}s | {gflops:.1f} GFLOPS")

            del a, b, c
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f" -> Matriz {n}x{n}: [FALHOU] provável falta de VRAM ({e})")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(" -> [OK] Teste de cuBLAS concluído.")


def teste_convolucao(device: torch.device) -> None:
    secao("[TESTE 2/4] Convolução 2D em lote (cuDNN)")

    imagem = torch.randn(CONV_BATCH_SIZE, 3, CONV_IMAGEM_LADO, CONV_IMAGEM_LADO, device=device)

    camada_conv = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    ).to(device)

    # Aquecimento (warm-up) — a primeira chamada ao cuDNN inclui
    # tempo de seleção de algoritmo, então não conta no benchmark
    with torch.no_grad():
        _ = camada_conv(imagem)
        if device.type == "cuda":
            torch.cuda.synchronize()

    inicio = time.time()
    with torch.no_grad():
        for _ in range(CONV_REPETICOES):
            saida = camada_conv(imagem)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fim = time.time()

    tempo_medio = (fim - inicio) / CONV_REPETICOES
    imagens_por_seg = CONV_BATCH_SIZE / tempo_medio

    print(f" -> Lote: {CONV_BATCH_SIZE} imagens de {CONV_IMAGEM_LADO}x{CONV_IMAGEM_LADO}")
    print(f" -> Tempo médio por lote: {tempo_medio * 1000:.2f} ms")
    print(f" -> Throughput: {imagens_por_seg:.1f} imagens/segundo")
    print(f" -> Formato de saída: {tuple(saida.shape)}")
    print(" -> [OK] Teste de cuDNN concluído.")


def teste_precisao_mista(device: torch.device) -> None:
    secao("[TESTE 3/4] Precisão mista (FP16 com autocast)")

    if device.type != "cuda":
        print(" -> Pulado (requer GPU).")
        return

    n = PRECISAO_MISTA_MATRIZ_TAMANHO
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    torch.cuda.synchronize()
    inicio = time.time()
    _ = torch.mm(a, b)
    torch.cuda.synchronize()
    tempo_fp32 = time.time() - inicio

    torch.cuda.synchronize()
    inicio = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()
    tempo_fp16 = time.time() - inicio

    ganho = tempo_fp32 / tempo_fp16 if tempo_fp16 > 0 else 0

    print(f" -> Matriz {n}x{n} em FP32: {tempo_fp32:.4f}s")
    print(f" -> Matriz {n}x{n} em FP16 (autocast): {tempo_fp16:.4f}s")
    print(f" -> Ganho de velocidade com FP16: {ganho:.2f}x")
    print(" -> [OK] Teste de precisão mista concluído.")

    del a, b
    torch.cuda.empty_cache()


def teste_treino_simulado(device: torch.device) -> None:
    secao("[TESTE 4/4] Ciclo de treino simulado (forward + backward)")

    modelo = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.LazyLinear(256),
        nn.ReLU(),
        nn.Linear(256, TREINO_SIMULADO_NUM_CLASSES),
    ).to(device)

    otimizador = torch.optim.Adam(modelo.parameters(), lr=TREINO_SIMULADO_LR)
    perda_fn = nn.CrossEntropyLoss()

    entrada = torch.randn(
        TREINO_SIMULADO_BATCH_SIZE, 3, TREINO_SIMULADO_IMAGEM_LADO, TREINO_SIMULADO_IMAGEM_LADO, device=device
    )
    rotulos = torch.randint(0, TREINO_SIMULADO_NUM_CLASSES, (TREINO_SIMULADO_BATCH_SIZE,), device=device)

    # Warm-up
    saida = modelo(entrada)
    perda = perda_fn(saida, rotulos)
    perda.backward()
    otimizador.step()
    otimizador.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize()

    inicio = time.time()
    for _ in range(TREINO_SIMULADO_EPOCAS):
        otimizador.zero_grad()
        saida = modelo(entrada)
        perda = perda_fn(saida, rotulos)
        perda.backward()
        otimizador.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    fim = time.time()

    tempo_medio = (fim - inicio) / TREINO_SIMULADO_EPOCAS
    amostras_por_seg = TREINO_SIMULADO_BATCH_SIZE / tempo_medio

    print(
        f" -> {TREINO_SIMULADO_EPOCAS} passos de treino, lote de {TREINO_SIMULADO_BATCH_SIZE} amostras "
        f"({TREINO_SIMULADO_IMAGEM_LADO}x{TREINO_SIMULADO_IMAGEM_LADO})"
    )
    print(f" -> Tempo médio por passo: {tempo_medio * 1000:.2f} ms")
    print(f" -> Throughput de treino: {amostras_por_seg:.1f} amostras/segundo")
    print(f" -> Perda final (loss): {perda.item():.4f}")
    print(" -> [OK] Ciclo de treino simulado concluído.")


def executar_benchmark(device: torch.device) -> None:
    if device.type == "cuda":
        alocada_antes, reservada_antes = mem_gpu_mb()
        print(f"\n[INFO] Memória GPU alocada antes dos testes: {alocada_antes:.1f} MB")
        print(f"[INFO] Memória GPU reservada antes dos testes: {reservada_antes:.1f} MB")

    teste_matmul(device)
    teste_convolucao(device)
    teste_precisao_mista(device)
    teste_treino_simulado(device)

    if device.type == "cuda":
        alocada_depois, reservada_depois = mem_gpu_mb()
        pico = torch.cuda.max_memory_allocated() / 1e6
        secao("Resumo de memória da GPU")
        print(f" -> Alocada ao final: {alocada_depois:.1f} MB")
        print(f" -> Reservada ao final: {reservada_depois:.1f} MB")
        print(f" -> Pico de alocação durante os testes: {pico:.1f} MB")

    print()
    titulo("RESULTADO FINAL: AMBIENTE CONFIGURADO CORRETAMENTE")
