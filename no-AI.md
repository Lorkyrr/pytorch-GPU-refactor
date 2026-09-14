# Como fazer tudo isso sem IA — o roteiro na unha

> Este repositório foi construído com ajuda de um assistente de IA (Claude
> Code). Este documento é o contraponto: um roteiro pra quem quer chegar no
> **mesmo resultado** — script de benchmark, treino de ResNet-20 no CIFAR-10,
> Docker, e CI numa GPU real via Kubernetes/ARC — sem colar nada pronto de
> uma IA. Só documentação oficial, o terminal, e a paciência de ler mensagem
> de erro até o fim.

## Como usar este documento

Não é um tutorial de copiar-e-colar. Cada etapa diz **o que** você precisa
conseguir fazer, **onde** está a documentação oficial pra aprender a fazer
(não um blog de terceiros, não um vídeo — a fonte primária), e **como
verificar com seus próprios olhos** que funcionou, antes de seguir pra
próxima. Se travar, o caminho "à moda antiga" é: ler a mensagem de erro
inteira, procurar o texto exato dela (aspas, no buscador), e ler a issue ou
o tópico de fórum que aparecer — não adivinhar. As tabelas de erro em
[CLAUDE.md](CLAUDE.md) e a narrativa completa em
[SAGA-DA-RTX3050.md](SAGA-DA-RTX3050.md) existem, mas resista à tentação de
abrir elas antes de tentar resolver sozinho — elas têm a resposta pronta, o
que é exatamente o atalho que este documento tenta evitar. Use-as só como
gabarito, depois de já ter tentado.

Pré-requisito de hardware: uma GPU NVIDIA (este projeto foi testado numa
RTX 3050 Laptop de 4 GB — dá pra fazer tudo com uma placa modesta, só ajuste
os tamanhos de tensor/lote se a sua tiver menos VRAM ainda). Sem GPU NVIDIA,
as Partes 1 e 3 ainda valem a pena rodando na CPU (mais lento, mas o código
roda), a Parte 2 (CI numa GPU real) não faz sentido sem uma.

---

## Parte 1 — Ambiente local: Python, PyTorch e GPU

### 1.1 Confirme que o driver NVIDIA está funcionando

Antes de tocar em Python ou Docker, confirme que o sistema operacional já
enxerga a placa:

```bash
nvidia-smi
```

Se este comando falhar, pare aqui e resolva a instalação do driver primeiro
— nada do resto funciona sem isso. A documentação do fabricante da sua
distribuição Linux (repositório de drivers) é a fonte certa; evite instalar
o driver "na mão" baixando um `.run` do site da NVIDIA se sua distro já
empacota um, é mais fácil de manter atualizado.

### 1.2 Aprenda o básico de Docker

Fonte oficial: **<https://docs.docker.com/>** — leia especificamente as
seções "Get Started" e "Dockerfile reference". Você precisa entender, sem
precisar procurar de novo depois:

- a diferença entre uma **imagem** (o molde, imutável) e um **container**
  (uma instância rodando daquele molde);
- o que cada instrução de um `Dockerfile` faz (`FROM`, `WORKDIR`, `RUN`,
  `COPY`, `CMD`);
- o que é `docker compose` e por que ele existe (evitar digitar um `docker
  run` gigante toda vez).

Depois, instale o **NVIDIA Container Toolkit** — é ele que ensina o Docker
a expor a GPU física dentro de um container. Documentação oficial:
**<https://github.com/NVIDIA/nvidia-container-toolkit>** (o `README` do
próprio projeto tem o passo a passo de instalação por distribuição). Depois
de instalar, configure o runtime `nvidia` como padrão do Docker e reinicie o
serviço — o próprio README do projeto explica o comando
(`nvidia-ctk runtime configure`). Confirme que funcionou:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

Se aparecer a mesma saída do `nvidia-smi` do host, mas rodando **dentro** de
um container, essa etapa está pronta.

### 1.3 Aprenda PyTorch o suficiente pra escrever o benchmark

Fonte oficial: **<https://pytorch.org/docs/stable/index.html>** (referência
de API) e **<https://pytorch.org/tutorials/>** (guiado). Sem copiar nada
pronto, você precisa conseguir escrever, sozinho, um script que:

1. Imprime `torch.__version__`, `torch.version.cuda`,
   `torch.cuda.is_available()`, `torch.backends.cudnn.version()` — leia a
   documentação de `torch.cuda` e `torch.backends.cudnn` até entender pra
   que serve cada um.
2. Cria dois tensores grandes com `torch.randn(n, n, device="cuda")` e
   multiplica com `torch.mm` — meça o tempo com `time.time()` **em volta de
   `torch.cuda.synchronize()`** antes e depois (leia por que a
   sincronização é necessária: operações CUDA são assíncronas por padrão —
   documentação de `torch.cuda.synchronize`). Repita para tamanhos
   crescentes de matriz até a GPU não aguentar mais (`RuntimeError` de
   VRAM) — trate esse erro com `try/except` em vez de deixar o script
   morrer.
3. Monta um `nn.Sequential` com `Conv2d`/`ReLU`/`MaxPool2d` (documentação de
   `torch.nn`), roda um lote de imagens aleatórias por ele repetidas vezes,
   e mede o tempo médio — faça **uma passada de aquecimento não cronometrada
   antes** (pesquise por que a primeira chamada de uma camada convolucional
   no cuDNN é mais lenta que as seguintes: seleção de algoritmo).
4. Compara o mesmo `torch.mm` dentro e fora de um bloco
   `torch.autocast(device_type="cuda", dtype=torch.float16)` — documentação
   de `torch.autocast` — e calcula a razão entre os dois tempos.
5. Monta uma rede pequena, um otimizador (`torch.optim.Adam`), uma função de
   perda (`nn.CrossEntropyLoss`), e roda um laço `forward → loss.backward()
   → optimizer.step() → optimizer.zero_grad()` por algumas dezenas de
   passos, medindo tempo por passo.

Você não precisa acertar de primeira — rode, veja o erro, leia a mensagem
completa (não só a última linha), procure o nome exato da exceção na
documentação, ajuste, repita. É assim que [main.py](main.py) nasceu.

### 1.4 Implemente a ResNet-20 lendo o paper original

Não use `torchvision.models` — o objetivo aqui é entender a arquitetura, não
importar ela pronta. Leia o paper original: **He et al., "Deep Residual
Learning for Image Recognition", 2015 — <https://arxiv.org/abs/1512.03385>**
(seção 4.2, "CIFAR-10 and Analysis", é a que descreve a variante de 6n+2
camadas usada aqui). Você precisa conseguir explicar, com suas palavras,
antes de escrever uma linha de código:

- por que um bloco residual soma a entrada à saída (`x + F(x)`) em vez de só
  empilhar camadas — o problema que isso resolve (degradação do gradiente em
  redes muito profundas);
- por que o atalho (*shortcut*) às vezes precisa ser uma projeção (`Conv2d`
  1x1 + `BatchNorm2d`) em vez de identidade pura — o que muda quando o
  número de canais ou o stride não batem entre entrada e saída do bloco.

Só depois disso, escreva as classes (`BlocoResidual`, `ResNetCIFAR`) você
mesmo — a documentação de `torch.nn.Module` (como escrever `__init__` e
`forward`) é a referência pra sintaxe, o paper é a referência pra
arquitetura.

### 1.5 Dataset e loop de treino

- `torchvision.datasets.CIFAR10` (documentação em
  <https://pytorch.org/vision/stable/datasets.html>) baixa e carrega o
  dataset sozinho; a página oficial do dataset —
  **<https://www.cs.toronto.edu/~kriz/cifar.html>** — explica o que tem
  dentro (60000 imagens 32x32, 10 classes) e vale ler antes de tratar o
  download como uma caixa preta.
- `torch.optim.lr_scheduler.MultiStepLR` (documentação de
  `torch.optim.lr_scheduler`) — leia sobre *learning rate schedules* em
  geral antes de escolher os marcos (*milestones*) e o fator de decaimento;
  o paper do passo anterior também descreve o schedule que os autores
  usaram, é um bom ponto de partida pra copiar a ideia (não o código) e
  ajustar.
- Escreva o loop de uma época (percorrer o `DataLoader`, `zero_grad`,
  forward, loss, backward, step) e uma função de avaliação (mesma coisa,
  mas com `@torch.no_grad()` e sem otimizar) — isso é praticamente o mesmo
  padrão do benchmark simulado da seção 1.3, só que com dados reais em vez
  de aleatórios.
- Salve o checkpoint com `torch.save(modelo.state_dict(), caminho)` — leia
  na documentação por que salvar o `state_dict()` (só os pesos) costuma ser
  preferível a salvar o objeto do modelo inteiro (portabilidade entre
  versões de código).

Rode localmente, deixe terminar, anote a acurácia final e o tempo — é assim
que os arquivos em [testes-de-ambiente/](testes-de-ambiente/) foram gerados
(rodagens reais, sem edição).

---

## Parte 2 — Empacotar em Docker

1. Escreva um `Dockerfile` que parte de uma imagem oficial do PyTorch já
   com CUDA compilado — no Docker Hub, procure a imagem oficial
   `pytorch/pytorch` e leia a aba "Tags" pra entender como elas nomeiam
   versões (PyTorch + CUDA + cuDNN na mesma tag). Instale as dependências
   que faltarem, copie seu código, defina o comando padrão.
2. Escreva um `compose.yaml` que builda essa imagem e sobe o container com
   acesso à GPU — a sintaxe de reserva de GPU no Compose está documentada
   em <https://docs.docker.com/> (procure por "GPU support" na
   documentação do Compose). Adicione um volume que espelhe sua pasta do
   projeto pra dentro do container, senão o dataset baixado e o checkpoint
   treinado ficam presos lá dentro quando o container morre.
3. Confirme rodando `docker compose up --build` e vendo o benchmark
   completo passar, com GPU real detectada (não caindo pro aviso de CPU).

---

## Parte 3 — CI numa GPU real (a parte mais longa)

Isto é opcional e bem mais avançado — só vale a pena se você quer que o
treino rode sozinho no GitHub Actions, usando a GPU física da sua própria
máquina como runner. Assuma que isso vai te tomar alguns dias de tentativa e
erro, não uma tarde — foi o que tomou aqui também (a narrativa completa,
capítulo a capítulo, está em [SAGA-DA-RTX3050.md](SAGA-DA-RTX3050.md), mas
tente sem ela primeiro).

### 3.1 Entenda runners self-hosted

Documentação oficial do GitHub: **<https://docs.github.com/actions>**
— procure a seção sobre "self-hosted runners". A pergunta que você precisa
saber responder antes de continuar: por que os runners hospedados
(`ubuntu-latest` etc.) não servem aqui? (Resposta: não têm GPU no plano
gratuito.)

### 3.2 Primeiro workflow, sem Kubernetes ainda

Não pule direto pro cluster. Primeiro, entenda o arquivo de workflow em si:
gatilhos (`on:`), `jobs`, `steps`, a diferença entre um gatilho automático
(`push`) e um manual (`workflow_dispatch`) — tudo isso está na mesma
documentação do passo 3.1. Escreva um workflow mínimo que só roda
`nvidia-smi` (nem precisa de GPU de verdade alocada ainda — o objetivo aqui
é só entender a sintaxe do YAML).

### 3.3 Kubernetes básico

Fonte oficial: **<https://kubernetes.io/docs/home/>**. Antes de instalar
qualquer coisa, entenda (sem precisar decorar comandos ainda) os conceitos
de **node**, **pod**, **namespace**, **DaemonSet** e **resource
requests/limits** — são a seção "Concepts" da documentação. Instale o
`kubectl` (instruções na própria documentação) e confirme que a instalação
funcionou (`kubectl version --client`).

### 3.4 Um cluster local com kind

Fonte oficial: **<https://kind.sigs.k8s.io/>**. `kind` cria um cluster
Kubernetes de verdade usando containers Docker como "nodes" — leia a seção
"Quick Start" e suba um cluster simples primeiro, sem GPU nenhuma, só pra
confirmar que `kubectl get nodes` mostra o node pronto.

Depois, o desafio de verdade: fazer esse cluster enxergar a GPU física do
seu notebook. Isso exige três peças encaixando (não existe um único
comando que resolve isso sozinho):

1. O runtime `nvidia` precisa estar configurado como **padrão** no Docker
   do host (você já fez isso na seção 1.2).
2. O `containerd` **interno** do node do kind (não o Docker do host — são
   dois softwares diferentes) também precisa saber usar esse runtime. A
   documentação do `kind` explica como usar `containerdConfigPatches` na
   configuração do cluster pra isso.
3. Alguns binários e caminhos do host precisam ser trazidos pra dentro do
   node via `extraMounts` (também documentado na configuração de cluster do
   `kind`).

Tente escrever esse arquivo de configuração você mesmo, a partir só da
documentação do `kind`. Vai quebrar de umas 2-3 formas diferentes antes de
funcionar — normal. Confirme o sucesso com
`docker exec -it <nome-do-node> nvidia-smi` rodando de dentro do container
do node.

### 3.5 Anunciar a GPU como recurso do Kubernetes

Um cluster que "vê" a GPU (passo anterior) ainda não sabe que ela existe
como recurso **agendável** — pra isso existe o **device plugin** da NVIDIA.
Fonte oficial: **<https://github.com/NVIDIA/k8s-device-plugin>** — o
`README` explica o que ele faz e como instalar (é um `DaemonSet`, aplicado
com `kubectl create -f <url-do-manifest>`, sem precisar de Helm). Depois de
instalar, confirme com:

```bash
kubectl describe node <nome-do-node> | grep -A5 Capacity
```

Procure por uma linha `nvidia.com/gpu: 1` — só quando ela aparecer é que um
pod consegue *pedir* essa GPU de verdade.

### 3.6 Actions Runner Controller (ARC)

Fonte oficial: **<https://github.com/actions/actions-runner-controller>**
— leia o `README` e a documentação de instalação via Helm (procure também
pela documentação oficial do Helm em **<https://helm.sh/docs/>** se nunca
usou antes: é o gerenciador de pacotes do Kubernetes). O ARC tem duas
partes que se instalam separadamente: o **controller** (observa o GitHub) e
um ou mais **runner scale sets** (o "molde" de pod que de fato roda os
jobs). Você vai precisar de um token do GitHub com permissão de
administração no repositório — a própria documentação do ARC explica qual
escopo exatamente.

Customize o "molde" do runner set com GPU: um arquivo de `values.yaml` que
pede `resources.limits."nvidia.com/gpu": 1` no container — a sintaxe de
resource limits é a mesma do Kubernetes "puro" (documentação do passo 3.3).

Confirme escrevendo um workflow com `runs-on: <nome-do-seu-runner-set>` e
disparando manualmente pela aba Actions do GitHub — se um pod novo aparecer
(`kubectl get pods`) e o job passar, funcionou.

### 3.7 Do Docker-dentro-do-CI pro Python nativo

Uma primeira versão funcional costuma usar `docker build`/`docker run
--gpus all` dentro do job, através de um socket do Docker montado no pod —
funciona, mas é uma armadilha conceitual: o container de treino passa a
rodar fora da contabilidade do Kubernetes (o `nvidia.com/gpu: 1` reservado
pro pod não tem relação real com o que o `docker run` está usando por
baixo). Pense sozinho: por que isso é um problema, e o que teria que mudar
pra resolver? (Dica sem ser resposta: o processo que efetivamente usa a GPU
devia ser o **mesmo** processo pelo qual o Kubernetes fez a reserva.) Depois
de pensar, tente resolver construindo uma imagem de runner customizada que
já tem Python e as dependências instaladas, e chamando `python3 seu_script.py`
como um passo normal do job — sem `docker build`/`docker run` no meio.

---

## Se travar

Isso é esperado — o caminho "na antiga" é mais lento por natureza. Antes de
qualquer coisa:

1. Leia a mensagem de erro **inteira**, não só a última linha (em scripts
   com `set -e`, o erro real costuma estar mais cedo do que parece).
2. Copie o texto exato do erro (entre aspas) num buscador — a maioria
   desses erros já foi discutida em algum lugar (fórum, issue no GitHub do
   projeto em questão, Stack Overflow).
3. Prefira o **issue tracker** ou a documentação do projeto específico
   (`kind`, ARC, NVIDIA Container Toolkit, etc.) a blogs de terceiros — é
   mais provável estar atualizada.
4. Só depois de ter tentado por conta própria, [CLAUDE.md](CLAUDE.md) tem
   uma tabela condensada dos erros que este projeto especificamente já
   encontrou e como foram resolvidos, e
   [SAGA-DA-RTX3050.md](SAGA-DA-RTX3050.md) conta a história completa de
   cada um — use como gabarito, não como atalho.
