#!/bin/bash
# ==============================================================================
# SCRIPT DE REFERÊNCIA / ESTUDO
# Kubernetes (kind) + GPU NVIDIA + ARC (Actions Runner Controller)
# ==============================================================================
#
# Este arquivo NÃO deve ser rodado de ponta a ponta sem pensar — ele documenta
# a sequência de comandos usados, seção por seção, para você revisar e
# entender cada passo antes de repetir o processo em outra máquina/projeto.
#
# Algumas seções são DESTRUTIVAS (apagam o cluster atual). Leia os comentários
# antes de copiar e colar qualquer trecho.
#
# Os passos automatizáveis daqui já viraram scripts prontos nesta mesma pasta
# (1-create-gpu-cluster.sh, 2-setup-arc.sh, 3-teardown-cluster.sh) — use este
# arquivo só como material de estudo/consulta, não como algo a rodar direto.
#
# ==============================================================================


# ------------------------------------------------------------------------------
# SEÇÃO 1 — Pré-requisitos no host (fora de qualquer cluster)
# ------------------------------------------------------------------------------

# kubectl: ferramenta de linha de comando para falar com um cluster Kubernetes.
# Sozinho, não cria cluster nenhum — só é o "controle remoto".
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# kind: cria clusters Kubernetes reais dentro de containers Docker.
# Alternativa ao k3s (que roda como serviço direto no host) — preferimos o kind
# porque tivemos problemas de conexão baixando o binário do k3s via GitHub releases.
# Instalação: baixar o binário do repositório oficial kubernetes-sigs/kind e
# colocar em /usr/local/bin (verificar a versão mais recente antes de instalar).

# Helm: gerenciador de pacotes do Kubernetes (equivalente ao apt/npm, mas para
# aplicações Kubernetes). Empacota vários YAMLs (Deployments, RBAC, CRDs) em um
# "chart" único, instalável com um comando.
#
# Instalado aqui via o script oficial (não snap, não apt) — baixa o binário
# certo pra sua arquitetura e joga em /usr/local/bin:
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash


# ------------------------------------------------------------------------------
# SEÇÃO 2 — Pré-requisitos de GPU no host
# ------------------------------------------------------------------------------

# Confirma que o driver NVIDIA está funcionando
nvidia-smi

# NVIDIA Container Toolkit: ensina o Docker a passar a GPU física para dentro
# de containers.
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configura o Docker para usar o runtime nvidia como PADRÃO — isso faz com que
# QUALQUER container Docker (incluindo os "nodes" que o kind cria) já herde
# acesso à GPU automaticamente, sem precisar de --gpus all explícito.
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker

# Testa se o Docker já enxerga a GPU
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Edite /etc/nvidia-container-runtime/config.toml e garanta que esta linha
# esteja SEM o "#" na frente (ativa):
#   accept-nvidia-visible-devices-as-volume-mounts = true
# Isso permite que o mecanismo de extraMounts do kind (seção 3) funcione.
sudo nano /etc/nvidia-container-runtime/config.toml
sudo systemctl daemon-reload   # importante depois de editar arquivos de serviço
sudo systemctl restart docker


# ------------------------------------------------------------------------------
# SEÇÃO 3 — Criação do cluster kind com suporte a GPU
# ------------------------------------------------------------------------------
#
# ATENÇÃO — DESTRUTIVO: isto apaga QUALQUER cluster "kind" existente e tudo
# que estava rodando nele (namespaces, secrets, deployments — tudo).

# Arquivo de configuração do cluster: ~/kind-gpu-config.yaml
#
# kind: Cluster
# apiVersion: kind.x-k8s.io/v1alpha4
# nodes:
#   - role: control-plane
#     extraMounts:
#       # Caminho "mágico" reconhecido pelo NVIDIA Container Runtime: ao ver
#       # este destino montado, ele injeta as bibliotecas de GPU no container.
#       - hostPath: /dev/null
#         containerPath: /var/run/nvidia-container-devices/all
#       # Ferramentas do Container Toolkit (não vêm pela injeção automática,
#       # só as bibliotecas do driver vêm — por isso trazemos manualmente).
#       - hostPath: /usr/bin/nvidia-container-runtime
#         containerPath: /usr/bin/nvidia-container-runtime
#         readOnly: true
#       - hostPath: /usr/bin/nvidia-container-cli
#         containerPath: /usr/bin/nvidia-container-cli
#         readOnly: true
#       - hostPath: /usr/bin/nvidia-ctk
#         containerPath: /usr/bin/nvidia-ctk
#         readOnly: true
#       # Socket do Docker do host — necessário para os pods runner conseguirem
#       # rodar `docker build`/`docker run` (ver Seção 6).
#       - hostPath: /var/run/docker.sock
#         containerPath: /var/run/docker.sock
# containerdConfigPatches:
#   # Configura o containerd INTERNO do node (diferente do Docker do host!)
#   # para usar o runtime nvidia por padrão também. Sem isso, pods criados
#   # DENTRO do cluster (como o device plugin) não enxergam a GPU, mesmo que
#   # o "docker exec ... nvidia-smi" já funcione.
#   - |-
#     [plugins."io.containerd.grpc.v1.cri".containerd]
#       default_runtime_name = "nvidia"
#     [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
#       runtime_type = "io.containerd.runc.v2"
#       [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
#         BinaryName = "/usr/bin/nvidia-container-runtime"

kind delete cluster
kind create cluster --config ~/kind-gpu-config.yaml

# Confirma que o node existe e está pronto
kubectl get nodes

# Confirma que a GPU está visível DENTRO do container do node
# (usa o Docker do host, por isso funciona mesmo sem o device plugin)
docker exec -it kind-control-plane nvidia-smi


# ------------------------------------------------------------------------------
# SEÇÃO 4 — NVIDIA Device Plugin (avisa o Kubernetes que existe GPU alocável)
# ------------------------------------------------------------------------------
#
# IMPORTANTE: isso precisa ser reinstalado toda vez que o cluster é recriado
# (seção 3), porque o "kind delete cluster" apaga também o namespace
# kube-system original. Esqueça isso e o Scheduler vai recusar agendar
# qualquer pod que peça GPU, com o erro "Insufficient nvidia.com/gpu".

kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/deployments/static/nvidia-device-plugin.yml

# Espera o pod ficar pronto
kubectl wait --for=condition=Ready pod -l name=nvidia-device-plugin-ds -n kube-system --timeout=90s

# Confirma que o node passou a anunciar a GPU como recurso alocável
kubectl describe node kind-control-plane | grep -A10 "Capacity:"
# Deve aparecer uma linha: nvidia.com/gpu: 1


# ------------------------------------------------------------------------------
# SEÇÃO 5 — Token do GitHub e Secret no cluster
# ------------------------------------------------------------------------------
#
# NUNCA cole o token direto num arquivo versionado no Git. Use uma variável
# de ambiente definida na sessão do terminal, ou um gerenciador de senhas.
#
# Fine-grained token (github.com/settings/tokens?type=beta):
#   Repository access: selecione o repositório
#   Repository permissions -> Administration: Read and write
#   (essa é a permissão que permite ao ARC registrar/remover runners)

export GITHUB_TOKEN='cole_o_token_aqui_apenas_no_terminal'

kubectl create namespace arc-systems --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic pre-defined-secret \
  --namespace=arc-systems \
  --from-literal=github_token="$GITHUB_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -


# ------------------------------------------------------------------------------
# SEÇÃO 6 — Instalação do ARC (controller + runner sets) via Helm
# ------------------------------------------------------------------------------

# Controller: o "cérebro" que observa o GitHub e cria/destrói pods runner
# sob demanda.
helm install arc \
  --namespace arc-systems \
  --create-namespace \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller

kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=gha-rs-controller -n arc-systems --timeout=90s

# Runner set padrão (sem GPU) — troque a URL pelo seu repositório
helm install arc-runner-set \
  --namespace arc-systems \
  --set githubConfigUrl="https://github.com/SEU_USUARIO/SEU_REPOSITORIO" \
  --set githubConfigSecret=pre-defined-secret \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

# Runner set COM GPU — usa um values.yaml customizado (ver arquivo
# ~/gpu-runner-values.yaml abaixo)
#
# Conteúdo de ~/gpu-runner-values.yaml:
#
# githubConfigUrl: "https://github.com/SEU_USUARIO/SEU_REPOSITORIO"
# githubConfigSecret: pre-defined-secret
#
# template:
#   spec:
#     containers:
#       - name: runner
#         image: ghcr.io/actions/actions-runner:latest
#         command: ["/home/runner/run.sh"]
#         securityContext:
#           # A imagem do runner roda como usuário não-root por padrão, que
#           # não tem permissão para o socket do Docker montado do host.
#           # "permission denied while trying to connect to the docker API"
#           # é o erro que aparece sem isto.
#           runAsUser: 0
#         env:
#           # A imagem recusa iniciar como root sem esta variável — sem ela,
#           # o pod entra em crash loop imediato (Running ~1s, depois Error).
#           - name: RUNNER_ALLOW_RUNASROOT
#             value: "1"
#         resources:
#           limits:
#             nvidia.com/gpu: 1        # reserva a GPU exclusivamente pra este pod
#         volumeMounts:
#           - name: docker-sock
#             mountPath: /var/run/docker.sock
#     volumes:
#       - name: docker-sock
#         hostPath:
#           path: /var/run/docker.sock
#           type: Socket

helm install arc-runner-set-gpu \
  --namespace arc-systems \
  -f ~/gpu-runner-values.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

# Confirma que tudo subiu (controller + 2 listeners = 3 pods)
kubectl get pods -n arc-systems


# ------------------------------------------------------------------------------
# SEÇÃO 7 — Exemplo de workflow do GitHub Actions
# ------------------------------------------------------------------------------
#
# Salvar em: SEU_REPOSITORIO/.github/workflows/teste-gpu.yml
#
# name: Teste GPU
#
# on:
#   workflow_dispatch:
#
# jobs:
#   teste:
#     runs-on: arc-runner-set-gpu   # ou "arc-runner-set" para rodar sem GPU
#     steps:
#       - name: Checkout do código
#         uses: actions/checkout@v4
#
#       - name: Testar acesso à GPU
#         run: docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
#
# Depois de criar o arquivo: git add / git commit / git push, depois disparar
# manualmente pela aba Actions do GitHub (botão "Run workflow").


# ------------------------------------------------------------------------------
# SEÇÃO 8 — Comandos úteis de debug
# ------------------------------------------------------------------------------

# Ver todos os pods de um namespace
kubectl get pods -n arc-systems

# Ver todos os pods de todos os namespaces
kubectl get pods -A

# Acompanhar em tempo real (Ctrl+C para sair)
kubectl get pods -n arc-systems --watch

# Detalhes de um pod específico — a seção "Events" no final é a mais útil
# para descobrir POR QUE um pod está preso em Pending/ContainerCreating/Error
kubectl describe pod -n arc-systems NOME_DO_POD

# Logs de um pod (o que o processo dentro dele imprimiu)
kubectl logs -n arc-systems NOME_DO_POD

# Logs do controller do ARC especificamente (útil quando o runner set não
# consegue se conectar ao GitHub — token errado, URL errada, etc.)
kubectl logs -n arc-systems -l app.kubernetes.io/name=gha-rs-controller --tail=50

# Ver quais imagens já estão em cache dentro do node (para saber se um
# download de imagem grande já terminou)
docker exec kind-control-plane crictl images

# Logs do containerd interno do node (para depurar pulls de imagem travados
# ou erros de runtime)
docker exec kind-control-plane journalctl -u containerd --no-pager | tail -30

# Status do device plugin (se sumiu "nvidia.com/gpu" da Capacity do node,
# comece por aqui — lembre-se: ele some toda vez que o cluster é recriado)
kubectl get pods -n kube-system | grep nvidia

# Ver quais imagens já baixaram dentro do node (útil para saber se um pull
# grande, tipo a imagem do runner, já terminou)
docker exec kind-control-plane crictl images

# Ver o recurso EphemeralRunner (camada acima do pod) — o campo
# "Status.Failures" mostra timestamps de tentativas que crasharam, útil
# quando um pod entra em crash loop rápido (Running -> Error em segundos)
kubectl get ephemeralrunner -n arc-systems
kubectl describe ephemeralrunner -n arc-systems -l app.kubernetes.io/instance=arc-runner-set-gpu


# ==============================================================================
# ERROS QUE JÁ ENCONTRAMOS E COMO RESOLVEMOS (histórico útil)
# ==============================================================================
#
# 1. "field gpus not found in type v1alpha4.Node"
#    -> Campo "gpus: true" no kind config NÃO existe de verdade. O acesso à
#       GPU vem do runtime nvidia configurado como padrão no Docker do host
#       (Seção 2), não de um campo especial no YAML do kind.
#
# 2. "exec: nvidia-smi: executable file not found in $PATH" dentro do node
#    -> Faltava o mecanismo accept-nvidia-visible-devices-as-volume-mounts
#       ativo (Seção 2) + o extraMount de /var/run/nvidia-container-devices/all
#       (Seção 3).
#
# 3. Device plugin em "Error"/CrashLoopBackOff com "ERROR_LIBRARY_NOT_FOUND"
#    -> O containerd INTERNO do node (não o Docker do host) não estava
#       configurado com o runtime nvidia. Resolvido com containerdConfigPatches
#       (Seção 3) — evita ter que editar e reiniciar o containerd manualmente
#       toda vez que o cluster é recriado.
#
# 4. "0/1 nodes are available: 1 Insufficient nvidia.com/gpu"
#    -> Ou já tem outro pod segurando a única GPU, ou o device plugin caiu/
#       sumiu (comum depois de recriar o cluster) e a capacidade não está
#       mais anunciada. Ver Seção 4.
#
# 5. "MountVolume.SetUp failed for volume docker-sock: ... does not exist"
#    -> Faltava trazer /var/run/docker.sock do host para dentro do node via
#       extraMounts no kind config (Seção 3) — o pod runner monta o socket
#       relativo ao "host" dele, que é o container do node, não seu notebook.
#
# 6. "permission denied while trying to connect to the docker API"
#    -> O usuário padrão dentro do container do runner não tem permissão
#       para o socket do Docker. Resolvido com securityContext.runAsUser: 0
#       (Seção 6).
#
# 7. Pod runner com GPU entra em crash loop rápido (Running ~1s -> Error),
#    sem nenhuma mensagem clara nos logs do pod
#    -> A imagem do runner recusa iniciar como root por padrão, mesmo com
#       runAsUser: 0 setado. Precisa também da variável de ambiente
#       RUNNER_ALLOW_RUNASROOT=1 (Seção 6).
