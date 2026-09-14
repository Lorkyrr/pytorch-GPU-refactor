"""
TESTE COMPLETO DE AMBIENTE LOCAL - PYTORCH + GPU NVIDIA
=========================================================
Feito para validar dentro de um container Docker se o PyTorch está
enxergando corretamente a GPU (cuBLAS, cuDNN, Tensor Cores, etc).

Ajustado para hardware com VRAM limitada (ex: RTX 3050 4GB) — os
tamanhos de tensor foram escolhidos para não estourar memória, mas
ainda assim gerarem uma carga real e mensurável na placa.
"""

import argparse
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------

def linha(char="=", tamanho=70):
    print(char * tamanho)


def titulo(texto):
    linha()
    print(texto.center(70))
    linha()


def secao(texto):
    print("\n" + "-" * 70)
    print(texto)
    print("-" * 70)


def mem_gpu_mb():
    """Retorna (alocada, reservada) em MB."""
    alocada = torch.cuda.memory_allocated() / 1e6
    reservada = torch.cuda.memory_reserved() / 1e6
    return alocada, reservada


def gflops_matmul(n, tempo_s):
    """Estimativa de GFLOPS para multiplicação de matrizes NxN."""
    flops = 2 * (n ** 3)
    return (flops / tempo_s) / 1e9 if tempo_s > 0 else 0.0


# ----------------------------------------------------------------------
# Passo 1: Informações do ambiente
# ----------------------------------------------------------------------

def info_ambiente():
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

    print(f"[INFO] cuDNN habilitado: {torch.backends.cudnn.enabled}")
    print(f"[INFO] Versão do cuDNN: {torch.backends.cudnn.version()}")

    gpu_nome = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory / 1e9
    capability = f"{props.major}.{props.minor}"

    print(f"[INFO] GPU detectada: {gpu_nome}")
    print(f"[INFO] Memória total da GPU: {total_mem:.2f} GB")
    print(f"[INFO] Compute Capability: {capability}")
    print(f"[INFO] Multiprocessadores (SMs): {props.multi_processor_count}")

    # Suporte a Tensor Cores / FP16 (Compute Capability >= 7.0)
    suporta_tensor_cores = props.major >= 7
    print(f"[INFO] Suporte a Tensor Cores (FP16 acelerado): {'Sim' if suporta_tensor_cores else 'Não'}")

    return device


# ----------------------------------------------------------------------
# Teste 1: Multiplicação de matrizes (cuBLAS) em várias escalas
# ----------------------------------------------------------------------

def teste_matmul(device):
    secao("[TESTE 1/4] Multiplicação de matrizes (cuBLAS) em múltiplas escalas")

    tamanhos = [1000, 2000, 4000, 6000]  # cresce progressivamente

    for n in tamanhos:
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


# ----------------------------------------------------------------------
# Teste 2: Convolução 2D em lote (cuDNN)
# ----------------------------------------------------------------------

def teste_convolucao(device):
    secao("[TESTE 2/4] Convolução 2D em lote (cuDNN)")

    batch_size = 16
    imagem = torch.randn(batch_size, 3, 224, 224, device=device)

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

    repeticoes = 20
    inicio = time.time()
    with torch.no_grad():
        for _ in range(repeticoes):
            saida = camada_conv(imagem)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fim = time.time()

    tempo_medio = (fim - inicio) / repeticoes
    imagens_por_seg = batch_size / tempo_medio

    print(f" -> Lote: {batch_size} imagens de 224x224")
    print(f" -> Tempo médio por lote: {tempo_medio * 1000:.2f} ms")
    print(f" -> Throughput: {imagens_por_seg:.1f} imagens/segundo")
    print(f" -> Formato de saída: {tuple(saida.shape)}")
    print(" -> [OK] Teste de cuDNN concluído.")


# ----------------------------------------------------------------------
# Teste 3: Precisão mista (FP16 / AMP) — aproveita Tensor Cores
# ----------------------------------------------------------------------

def teste_precisao_mista(device):
    secao("[TESTE 3/4] Precisão mista (FP16 com autocast)")

    if device.type != "cuda":
        print(" -> Pulado (requer GPU).")
        return

    n = 4000
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    # FP32 (baseline)
    torch.cuda.synchronize()
    inicio = time.time()
    _ = torch.mm(a, b)
    torch.cuda.synchronize()
    tempo_fp32 = time.time() - inicio

    # FP16 via autocast
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


# ----------------------------------------------------------------------
# Teste 4: Ciclo de treino simulado (forward + backward + optimizer)
# ----------------------------------------------------------------------

def teste_treino_simulado(device):
    secao("[TESTE 4/4] Ciclo de treino simulado (forward + backward)")

    modelo = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.LazyLinear(256), nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)

    otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    perda_fn = nn.CrossEntropyLoss()

    batch_size = 32
    entrada = torch.randn(batch_size, 3, 64, 64, device=device)
    rotulos = torch.randint(0, 10, (batch_size,), device=device)

    # Warm-up
    saida = modelo(entrada)
    perda = perda_fn(saida, rotulos)
    perda.backward()
    otimizador.step()
    otimizador.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize()

    epocas = 30
    inicio = time.time()
    for _ in range(epocas):
        otimizador.zero_grad()
        saida = modelo(entrada)
        perda = perda_fn(saida, rotulos)
        perda.backward()
        otimizador.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    fim = time.time()

    tempo_medio = (fim - inicio) / epocas
    amostras_por_seg = batch_size / tempo_medio

    print(f" -> {epocas} passos de treino, lote de {batch_size} amostras (64x64)")
    print(f" -> Tempo médio por passo: {tempo_medio * 1000:.2f} ms")
    print(f" -> Throughput de treino: {amostras_por_seg:.1f} amostras/segundo")
    print(f" -> Perda final (loss): {perda.item():.4f}")
    print(" -> [OK] Ciclo de treino simulado concluído.")


# ----------------------------------------------------------------------
# ResNet para CIFAR-10 (arquitetura de He et al., 2015 — variante CIFAR)
# ----------------------------------------------------------------------

class BlocoResidual(nn.Module):
    """Bloco básico (3x3 -> 3x3) com atalho (shortcut) identidade ou projeção."""

    expansao = 1

    def __init__(self, canais_entrada, canais_saida, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(canais_entrada, canais_saida, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(canais_saida)
        self.conv2 = nn.Conv2d(canais_saida, canais_saida, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(canais_saida)
        self.relu = nn.ReLU(inplace=True)

        self.atalho = nn.Sequential()
        if stride != 1 or canais_entrada != canais_saida:
            self.atalho = nn.Sequential(
                nn.Conv2d(canais_entrada, canais_saida, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(canais_saida),
            )

    def forward(self, x):
        saida = self.relu(self.bn1(self.conv1(x)))
        saida = self.bn2(self.conv2(saida))
        saida = saida + self.atalho(x)
        return self.relu(saida)


class ResNetCIFAR(nn.Module):
    """ResNet-(6n+2) para imagens 32x32, como no paper original do ResNet.

    Com blocos_por_estagio=3 obtém-se a ResNet-20.
    """

    def __init__(self, blocos_por_estagio=3, num_classes=10):
        super().__init__()
        self.canais_entrada = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.estagio1 = self._criar_estagio(16, blocos_por_estagio, stride=1)
        self.estagio2 = self._criar_estagio(32, blocos_por_estagio, stride=2)
        self.estagio3 = self._criar_estagio(64, blocos_por_estagio, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _criar_estagio(self, canais_saida, num_blocos, stride):
        strides = [stride] + [1] * (num_blocos - 1)
        camadas = []
        for s in strides:
            camadas.append(BlocoResidual(self.canais_entrada, canais_saida, s))
            self.canais_entrada = canais_saida
        return nn.Sequential(*camadas)

    def forward(self, x):
        saida = self.relu(self.bn1(self.conv1(x)))
        saida = self.estagio1(saida)
        saida = self.estagio2(saida)
        saida = self.estagio3(saida)
        saida = self.avgpool(saida)
        saida = torch.flatten(saida, 1)
        return self.fc(saida)


def resnet20_cifar(num_classes=10):
    return ResNetCIFAR(blocos_por_estagio=3, num_classes=num_classes)


# ----------------------------------------------------------------------
# Dados: CIFAR-10
# ----------------------------------------------------------------------

def cifar10_dataloaders(data_dir="./data", batch_size=128, num_workers=2):
    media = (0.4914, 0.4822, 0.4465)
    desvio = (0.2470, 0.2435, 0.2616)

    transform_treino = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(media, desvio),
    ])
    transform_teste = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(media, desvio),
    ])

    conjunto_treino = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_treino)
    conjunto_teste = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_teste)

    loader_treino = DataLoader(conjunto_treino, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True)
    loader_teste = DataLoader(conjunto_teste, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)

    return loader_treino, loader_teste


# ----------------------------------------------------------------------
# Laços de treino e avaliação
# ----------------------------------------------------------------------

def treinar_uma_epoca(modelo, loader, otimizador, perda_fn, device):
    modelo.train()
    perda_total, acertos, total = 0.0, 0, 0

    for entradas, rotulos in loader:
        entradas = entradas.to(device, non_blocking=True)
        rotulos = rotulos.to(device, non_blocking=True)

        otimizador.zero_grad()
        saidas = modelo(entradas)
        perda = perda_fn(saidas, rotulos)
        perda.backward()
        otimizador.step()

        perda_total += perda.item() * entradas.size(0)
        _, previstos = saidas.max(1)
        total += rotulos.size(0)
        acertos += previstos.eq(rotulos).sum().item()

    return perda_total / total, 100.0 * acertos / total


@torch.no_grad()
def avaliar(modelo, loader, perda_fn, device):
    modelo.eval()
    perda_total, acertos, total = 0.0, 0, 0

    for entradas, rotulos in loader:
        entradas = entradas.to(device, non_blocking=True)
        rotulos = rotulos.to(device, non_blocking=True)

        saidas = modelo(entradas)
        perda = perda_fn(saidas, rotulos)

        perda_total += perda.item() * entradas.size(0)
        _, previstos = saidas.max(1)
        total += rotulos.size(0)
        acertos += previstos.eq(rotulos).sum().item()

    return perda_total / total, 100.0 * acertos / total


def treinar_resnet_cifar10(device, epocas=30, batch_size=128, lr=0.1,
                            data_dir="./data", checkpoint_path="resnet20_cifar10.pth"):
    titulo("TREINAMENTO: RESNET-20 NO CIFAR-10")

    print(f"\n[INFO] Épocas: {epocas} | Batch size: {batch_size} | LR inicial: {lr}")
    print("[INFO] Baixando/preparando o CIFAR-10 (se necessário)...")
    loader_treino, loader_teste = cifar10_dataloaders(data_dir=data_dir, batch_size=batch_size)

    modelo = resnet20_cifar(num_classes=10).to(device)
    perda_fn = nn.CrossEntropyLoss()
    otimizador = torch.optim.SGD(modelo.parameters(), lr=lr, momentum=0.9,
                                  weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        otimizador, milestones=[int(epocas * 0.5), int(epocas * 0.75)], gamma=0.1)

    melhor_acc = 0.0

    secao("Progresso do treinamento")
    for epoca in range(1, epocas + 1):
        inicio = time.time()
        perda_treino, acc_treino = treinar_uma_epoca(modelo, loader_treino, otimizador, perda_fn, device)
        perda_teste, acc_teste = avaliar(modelo, loader_teste, perda_fn, device)
        scheduler.step()
        tempo = time.time() - inicio

        print(f"[Época {epoca:03d}/{epocas}] "
              f"treino_loss={perda_treino:.4f} treino_acc={acc_treino:.2f}% | "
              f"teste_loss={perda_teste:.4f} teste_acc={acc_teste:.2f}% | "
              f"{tempo:.1f}s")

        if acc_teste > melhor_acc:
            melhor_acc = acc_teste
            torch.save(modelo.state_dict(), checkpoint_path)

    secao("Resultado do treinamento")
    print(f" -> Melhor acurácia no teste: {melhor_acc:.2f}%")
    print(f" -> Checkpoint salvo em: {checkpoint_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def executar_benchmark(device):
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


def main():
    parser = argparse.ArgumentParser(
        description="Teste de ambiente PyTorch/GPU e treinamento de ResNet no CIFAR-10")
    parser.add_argument(
        "modo", nargs="?", choices=["benchmark", "train"], default="benchmark",
        help="'benchmark' roda os testes de GPU (padrão); 'train' treina uma ResNet-20 no CIFAR-10")
    parser.add_argument("--epochs", type=int, default=30, help="número de épocas de treino")
    parser.add_argument("--batch-size", type=int, default=128, help="tamanho do lote")
    parser.add_argument("--lr", type=float, default=0.1, help="taxa de aprendizado inicial")
    parser.add_argument("--data-dir", type=str, default="./data", help="diretório do CIFAR-10")
    args = parser.parse_args()

    device = info_ambiente()

    if args.modo == "train":
        treinar_resnet_cifar10(device, epocas=args.epochs, batch_size=args.batch_size,
                                lr=args.lr, data_dir=args.data_dir)
        return

    executar_benchmark(device)


if __name__ == "__main__":
    main()