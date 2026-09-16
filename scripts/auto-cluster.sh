#!/bin/bash
# ==============================================================================
# AUTO-RECUPERAÇÃO: DEPENDÊNCIAS + CLUSTER KIND + GPU NVIDIA + ARC
# ==============================================================================
#
# Ponto de entrada único para reconstruir o ambiente local do zero — depois de
# um crash, um reboot, ou uma reinstalação do sistema operacional. Idempotente:
# pode ser rodado quantas vezes forem necessárias, em qualquer estado inicial.
#
# Inventário de dependências para SUBIR E OPERAR O CLUSTER (não as dependências
# do projeto pytorch_gpu_sandbox em si — essas os workflows de CI já resolvem
# sozinhos, ver .github/workflows/pytorch-gpu-python.yaml). Lido dos arquivos
# reais em k8s/, scripts/ e .github/workflows/:
#   - curl                             — usado pelo próprio script pra baixar binários
#   - docker                           — roda os nodes do kind e (variante docker) o
#                                         socket usado pelo runner com GPU
#   - kubectl, kind, helm              — provisionamento do cluster (k8s/)
#   - driver NVIDIA (nvidia-smi)       — obrigatório, NUNCA instalado por este script
#   - NVIDIA Container Toolkit         — GPU dentro do node kind e dos pods runner
#   - Docker Default Runtime = nvidia
#     + accept-nvidia-visible-devices-as-volume-mounts = true — sem isso o kind
#     não repassa a GPU física pro node (ver gotcha "nvidia-smi: executable
#     file not found" no CLAUDE.md)
#   - GITHUB_TOKEN                     — registra os runner sets no seu repositório
#
# O que ele faz, em ordem:
#   FASE 0 — Diagnóstico (só leitura): confere cada item acima, o Default
#            Runtime do Docker, a config do NVIDIA Container Runtime, se já
#            existe cluster "kind" e se GITHUB_TOKEN está definido.
#   FASE 1 — Correção: instala sozinho o que for de baixo risco (binários
#            estáticos: kubectl/kind/helm). Para tudo que mexe em pacotes do
#            sistema (apt) ou reinicia o Docker, mostra a lista completa do
#            que vai fazer e PERGUNTA antes — essas mudanças afetam a máquina
#            inteira, não só este projeto.
#   FASE 2 — (Re)cria o cluster kind a partir de k8s/kind-gpu-config.yaml.
#            DESTRUTIVO: se já existir um cluster "kind", ele e tudo dentro
#            dele é apagado — o script confirma antes de seguir.
#   FASE 3 — Instala/atualiza em paralelo o NVIDIA Device Plugin e o ARC
#            (controller + runner set sem GPU + runner set com GPU).
#   FASE 4 — Verificação final: pods do ARC, GPU alocável no node.
#
# GITHUB_TOKEN — ordem de resolução:
#   1. Variável de ambiente GITHUB_TOKEN já exportada.
#   2. Arquivo scripts/token.md (mesma pasta deste script), se existir —
#      conteúdo lido como o token puro, sem mais nada no arquivo.
#   3. Se nenhum dos dois existir, pede digitado (entrada oculta) na Fase 3.
#   ATENÇÃO: scripts/token.md fica listado em .gitignore de propósito — é um
#   segredo em texto puro na sua máquina. Nunca force um "git add -f" nele.
#
# O que este script NUNCA automatiza:
#   - O driver NVIDIA em si (exige reboot, pode quebrar um driver gráfico já
#     funcionando, tem pegadinhas de secure boot) — se nvidia-smi falhar, o
#     script para e mostra a Seção 2 de kubernetes-gpu-arc-referencia.sh.
#
# Suporte a distro: os instaladores automáticos (Fase 1) usam apt-get — ou
# seja, Debian/Ubuntu, igual ao host documentado em CLAUDE.md/README.md. Em
# outra distro, o script avisa e aponta para os passos manuais equivalentes.
#
# Uso:
#   ./scripts/auto-cluster.sh [SEU_USUARIO/SEU_REPOSITORIO] [docker|python]
#   (padrão do repositório: Lorkyrr/refactored-pytorch-CTTK, variante: docker)
#
# ==============================================================================

set -e

# --- Resolve os caminhos do repositório, mesmo chamado de outro diretório ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
KIND_CONFIG="$REPO_ROOT/k8s/kind-gpu-config.yaml"
NVIDIA_RUNTIME_CONFIG="/etc/nvidia-container-runtime/config.toml"

REPO="${1:-Lorkyrr/refactored-pytorch-CTTK}"
VARIANT="${2:-docker}"

case "$VARIANT" in
  python) GPU_VALUES_TEMPLATE="$REPO_ROOT/k8s/gpu-runner-values.yaml" ;;
  docker) GPU_VALUES_TEMPLATE="$REPO_ROOT/k8s/gpu-runner-values.docker.yaml" ;;
  *)
    echo "Erro: variante '$VARIANT' inválida — use 'python' ou 'docker'."
    exit 1
    ;;
esac

if [ ! -f "$KIND_CONFIG" ] || [ ! -f "$GPU_VALUES_TEMPLATE" ]; then
  echo "Erro: não encontrei os arquivos de config em k8s/. Execute a partir do repositório clonado."
  exit 1
fi

GITHUB_URL="https://github.com/$REPO"

TOKEN_FILE="$SCRIPT_DIR/token.md"
if [ -z "$GITHUB_TOKEN" ] && [ -f "$TOKEN_FILE" ]; then
  GITHUB_TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
  if [ -n "$GITHUB_TOKEN" ]; then
    echo "GITHUB_TOKEN carregado de $TOKEN_FILE"
  fi
fi

TEM_APT=false
command -v apt-get &> /dev/null && TEM_APT=true

# ------------------------------------------------------------------------------
# Funções de instalação (só EXECUTADAS depois de confirmação na Fase 1)
# ------------------------------------------------------------------------------

instalar_pacotes_basicos() {
  echo "--- Instalando pacotes básicos via apt: ${PKGS_APT_BASICOS[*]} ---"
  sudo apt-get update
  sudo apt-get install -y "${PKGS_APT_BASICOS[@]}"
}

instalar_docker() {
  echo "--- Instalando Docker Engine via repositório oficial docker-ce ---"
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER"
  echo "Docker instalado. Se ainda não estava no grupo 'docker', saia e entre na sessão de novo pra usar sem sudo."
}

instalar_nvidia_container_toolkit() {
  echo "--- Instalando NVIDIA Container Toolkit ---"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
}

corrigir_runtime_nvidia() {
  echo "--- Configurando o Docker para usar 'nvidia' como runtime padrão ---"
  if [ "$CONFIG_TOML_OK" = false ]; then
    if grep -q "^#accept-nvidia-visible-devices-as-volume-mounts" "$NVIDIA_RUNTIME_CONFIG" 2>/dev/null; then
      sudo sed -i 's/^#accept-nvidia-visible-devices-as-volume-mounts.*/accept-nvidia-visible-devices-as-volume-mounts = true/' "$NVIDIA_RUNTIME_CONFIG"
    else
      echo 'accept-nvidia-visible-devices-as-volume-mounts = true' | sudo tee -a "$NVIDIA_RUNTIME_CONFIG" > /dev/null
    fi
  fi
  sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
  sudo systemctl daemon-reload
  sudo systemctl restart docker
}

instalar_kubectl() {
  echo "--- Instalando kubectl ---"
  local tmp_kubectl
  tmp_kubectl="$(mktemp)"
  curl -L -s -o "$tmp_kubectl" "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  sudo install -o root -g root -m 0755 "$tmp_kubectl" /usr/local/bin/kubectl
  rm -f "$tmp_kubectl"
}

instalar_kind() {
  echo "--- Instalando kind ---"
  local tmp_kind
  tmp_kind="$(mktemp)"
  curl -L -s -o "$tmp_kind" "https://kind.sigs.k8s.io/dl/latest/kind-linux-amd64"
  chmod +x "$tmp_kind"
  sudo install -o root -g root -m 0755 "$tmp_kind" /usr/local/bin/kind
  rm -f "$tmp_kind"
}

instalar_helm() {
  echo "--- Instalando Helm ---"
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
}

# ------------------------------------------------------------------------------
# FASE 0/4 — Diagnóstico de pendências (não altera nada no sistema)
# ------------------------------------------------------------------------------
echo "=== FASE 0/4: Diagnóstico de dependências e pendências ==="
echo "Repositório alvo: $REPO (variante: $VARIANT)"
echo ""

ACOES_BINARIOS=()   # instalados automaticamente, sem prompt (binário isolado em /usr/local/bin)
ACOES_SISTEMA=()    # descrição p/ exibir antes do prompt único de confirmação
FUNCOES_SISTEMA=()  # funções correspondentes, na ordem de execução
PKGS_APT_BASICOS=()
RUNTIME_OK=false
CONFIG_TOML_OK=false
CLUSTER_EXISTE=false

check() {
  local label=$1
  local test_cmd=$2
  if eval "$test_cmd" &> /dev/null; then
    echo "  [OK] $label"
    return 0
  else
    echo "  [PENDENTE] $label"
    return 1
  fi
}

check "curl" "command -v curl" || PKGS_APT_BASICOS+=("curl")

if command -v docker &> /dev/null; then
  echo "  [OK] docker"
else
  echo "  [PENDENTE] docker"
  ACOES_SISTEMA+=("Instalar Docker Engine (repositório oficial docker-ce)")
  FUNCOES_SISTEMA+=("instalar_docker")
fi

check "kubectl" "command -v kubectl" || ACOES_BINARIOS+=("instalar_kubectl")
check "kind" "command -v kind" || ACOES_BINARIOS+=("instalar_kind")
check "helm" "command -v helm" || ACOES_BINARIOS+=("instalar_helm")

# Driver NVIDIA: NUNCA automatizado — se faltar, para aqui mesmo.
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
  echo "  [OK] driver NVIDIA (nvidia-smi)"
else
  echo "  [FALHA] driver NVIDIA não encontrado ou não funcional"
  echo ""
  echo "Erro: o driver NVIDIA é pré-requisito manual e este script nunca o instala"
  echo "sozinho (exige reboot e pode conflitar com um driver gráfico já em uso)."
  echo "Siga a Seção 2 de scripts/kubernetes-gpu-arc-referencia.sh e rode este"
  echo "script de novo depois."
  exit 1
fi

if command -v nvidia-ctk &> /dev/null; then
  echo "  [OK] NVIDIA Container Toolkit (nvidia-ctk)"
else
  echo "  [PENDENTE] NVIDIA Container Toolkit (nvidia-ctk)"
  ACOES_SISTEMA+=("Instalar o NVIDIA Container Toolkit")
  FUNCOES_SISTEMA+=("instalar_nvidia_container_toolkit")
fi

# Roda mesmo se o Docker ainda não estiver instalado (será instalado antes,
# na Fase 1, respeitando a ordem de FUNCOES_SISTEMA) — senão, num sistema
# recém-formatado, o Docker seria instalado mas ficaria com o runtime padrão
# errado, e essa correção nunca seria oferecida.
DEFAULT_RUNTIME=""
if command -v docker &> /dev/null && docker info &> /dev/null; then
  DEFAULT_RUNTIME="$(docker info 2>/dev/null | grep "Default Runtime" | awk '{print $NF}')"
fi
if [ "$DEFAULT_RUNTIME" = "nvidia" ]; then
  echo "  [OK] Docker Default Runtime = nvidia"
  RUNTIME_OK=true
else
  echo "  [PENDENTE] Docker Default Runtime = ${DEFAULT_RUNTIME:-(docker ainda não instalado/ativo)} (esperado: nvidia)"
fi

if grep -q "^accept-nvidia-visible-devices-as-volume-mounts = true" "$NVIDIA_RUNTIME_CONFIG" 2>/dev/null; then
  echo "  [OK] $NVIDIA_RUNTIME_CONFIG configurado"
  CONFIG_TOML_OK=true
else
  echo "  [PENDENTE] $NVIDIA_RUNTIME_CONFIG sem 'accept-nvidia-visible-devices-as-volume-mounts = true'"
fi

if [ "$RUNTIME_OK" = false ] || [ "$CONFIG_TOML_OK" = false ]; then
  ACOES_SISTEMA+=("Configurar 'nvidia' como runtime padrão do Docker (reinicia o daemon)")
  FUNCOES_SISTEMA+=("corrigir_runtime_nvidia")
fi

if command -v kind &> /dev/null && kind get clusters 2>/dev/null | grep -qx "kind"; then
  echo "  [INFO] Cluster 'kind' já existe — será recriado (destrutivo) na Fase 2"
  CLUSTER_EXISTE=true
else
  echo "  [OK] Nenhum cluster 'kind' existente — criação limpa"
fi

if [ -n "$GITHUB_TOKEN" ]; then
  echo "  [OK] GITHUB_TOKEN definido no ambiente"
else
  echo "  [PENDENTE] GITHUB_TOKEN não definido — será solicitado interativamente na Fase 3"
fi

if [ ${#PKGS_APT_BASICOS[@]} -gt 0 ]; then
  ACOES_SISTEMA=("Instalar pacotes básicos via apt: ${PKGS_APT_BASICOS[*]}" "${ACOES_SISTEMA[@]}")
  FUNCOES_SISTEMA=("instalar_pacotes_basicos" "${FUNCOES_SISTEMA[@]}")
fi

echo ""
if [ "$TEM_APT" = false ] && { [ ${#ACOES_SISTEMA[@]} -gt 0 ] || [ ${#ACOES_BINARIOS[@]} -gt 0 ]; }; then
  echo "AVISO: esta distro não tem apt-get — a instalação automática (Fase 1) não"
  echo "é suportada aqui. Siga os passos manuais em scripts/kubernetes-gpu-arc-referencia.sh."
fi

# ------------------------------------------------------------------------------
# FASE 1/4 — Correção (automática quando seguro, com confirmação quando não é)
# ------------------------------------------------------------------------------
echo ""
echo "=== FASE 1/4: Correções ==="

if [ "$TEM_APT" = true ]; then
  for fn in "${ACOES_BINARIOS[@]}"; do
    "$fn"
  done
  [ ${#ACOES_BINARIOS[@]} -eq 0 ] && echo "Binários (kubectl/kind/helm) já OK, nada a instalar."

  if [ ${#ACOES_SISTEMA[@]} -gt 0 ]; then
    echo ""
    echo "As mudanças abaixo afetam pacotes do sistema e/ou reiniciam o Docker —"
    echo "isso impacta a máquina inteira, não só este projeto:"
    for a in "${ACOES_SISTEMA[@]}"; do
      echo "  - $a"
    done
    read -rp "Aplicar tudo isso agora? [s/N] " RESP_SISTEMA
    if [[ "$RESP_SISTEMA" =~ ^[sS]$ ]]; then
      for fn in "${FUNCOES_SISTEMA[@]}"; do
        "$fn"
      done
      echo "Correções de sistema aplicadas."
    else
      echo "Pulando — a Fase 2/3 pode falhar até isso ser corrigido manualmente."
    fi
  else
    echo "Nenhuma mudança de sistema pendente."
  fi
else
  echo "Sem apt-get — pulando instalação automática (ver aviso acima)."
fi

# ------------------------------------------------------------------------------
# FASE 2/4 — (Re)cria o cluster kind
# ------------------------------------------------------------------------------
echo ""
echo "=== FASE 2/4: Recriando o cluster kind (config: $KIND_CONFIG) ==="

if [ "$CLUSTER_EXISTE" = true ]; then
  echo "AVISO: já existe um cluster 'kind' — ele e tudo dentro dele será apagado."
  read -rp "Continuar? [s/N] " RESP_CLUSTER
  if [[ ! "$RESP_CLUSTER" =~ ^[sS]$ ]]; then
    echo "Abortado pelo usuário."
    exit 1
  fi
fi

kind delete cluster --name kind 2>/dev/null || true
kind create cluster --config "$KIND_CONFIG"

echo "Verificando o node..."
kubectl get nodes

echo "Confirmando GPU visível dentro do container do node..."
docker exec -it kind-control-plane nvidia-smi

# ------------------------------------------------------------------------------
# FASE 3/4 — Device Plugin + ARC em paralelo
# ------------------------------------------------------------------------------
echo ""
echo "=== FASE 3/4: Instalando Device Plugin e ARC em paralelo ==="

if [ -z "$GITHUB_TOKEN" ]; then
  echo -n "Digite o seu GITHUB_TOKEN (a entrada ficará oculta): "
  read -rsp "" GITHUB_TOKEN
  echo ""
fi

if [ -z "$GITHUB_TOKEN" ]; then
  echo "Erro: nenhum GITHUB_TOKEN foi fornecido."
  exit 1
fi

LOG_DEVICE_PLUGIN="$(mktemp)"
LOG_ARC="$(mktemp)"
trap 'rm -f "$LOG_DEVICE_PLUGIN" "$LOG_ARC"' EXIT

# --- Trilha A: NVIDIA Device Plugin ---
(
  set -e
  echo "=== Verificando NVIDIA Device Plugin ==="
  if ! kubectl get pods -n kube-system 2>/dev/null | grep -q nvidia-device-plugin; then
    echo "Instalando Device Plugin a partir de k8s/nvidia-device-plugin.yaml..."
    kubectl create -f "$REPO_ROOT/k8s/nvidia-device-plugin.yaml"
  fi

  echo "Aguardando Device Plugin ficar pronto..."
  kubectl wait --for=condition=Ready pod \
    -l name=nvidia-device-plugin-ds \
    -n kube-system --timeout=120s

  echo "Aguardando o Node anunciar a GPU alocável..."
  GPU_OK=false
  for i in $(seq 1 30); do
    if kubectl describe node kind-control-plane | grep -q "nvidia.com/gpu:"; then
      echo "GPU detectada com sucesso na capacidade do node."
      GPU_OK=true
      break
    fi
    sleep 1
  done

  if [ "$GPU_OK" = false ]; then
    echo "Aviso: GPU não apareceu. Reinstalando o Device Plugin..."
    kubectl delete pod -n kube-system -l name=nvidia-device-plugin-ds --ignore-not-found
    kubectl wait --for=condition=Ready pod \
      -l name=nvidia-device-plugin-ds \
      -n kube-system --timeout=120s
  fi
) > "$LOG_DEVICE_PLUGIN" 2>&1 &
PID_DEVICE_PLUGIN=$!

# --- Trilha B: ARC (Namespace, Secret, Controller e Runner Sets) ---
(
  set -e
  echo "=== Criando Namespace e Secret para o ARC ==="
  kubectl create namespace arc-systems --dry-run=client -o yaml | kubectl apply -f -

  kubectl create secret generic pre-defined-secret \
    --namespace=arc-systems \
    --from-literal=github_token="$GITHUB_TOKEN" \
    --dry-run=client -o yaml | kubectl apply -f -

  echo "=== Instalando/Atualizando Controller do ARC via Helm ==="
  helm upgrade --install arc \
    --namespace arc-systems \
    --create-namespace \
    oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller

  kubectl wait --for=condition=Ready pod \
    -l app.kubernetes.io/name=gha-rs-controller \
    -n arc-systems --timeout=90s

  echo "=== Instalando Runner Set Padrão (Sem GPU) ==="
  helm upgrade --install arc-runner-set \
    --namespace arc-systems \
    --set githubConfigUrl="$GITHUB_URL" \
    --set githubConfigSecret=pre-defined-secret \
    oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set &
  PID_NOGPU=$!

  echo "=== Gerando values da variante '$VARIANT' e instalando Runner Set com GPU ==="
  GENERATED_VALUES="$(mktemp)"
  trap 'rm -f "$GENERATED_VALUES"' EXIT
  sed "s#^githubConfigUrl:.*#githubConfigUrl: \"$GITHUB_URL\"#" "$GPU_VALUES_TEMPLATE" > "$GENERATED_VALUES"

  helm upgrade --install arc-runner-set-gpu \
    --namespace arc-systems \
    -f "$GENERATED_VALUES" \
    oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set &
  PID_GPU=$!

  wait "$PID_NOGPU"
  wait "$PID_GPU"
) > "$LOG_ARC" 2>&1 &
PID_ARC=$!

FAIL=0
wait "$PID_DEVICE_PLUGIN" || FAIL=1
wait "$PID_ARC" || FAIL=1

echo ""
echo "--- Log: NVIDIA Device Plugin ---"
cat "$LOG_DEVICE_PLUGIN"
echo ""
echo "--- Log: ARC (Controller & Runner Sets) ---"
cat "$LOG_ARC"

if [ "$FAIL" -ne 0 ]; then
  echo ""
  echo "Erro: uma das etapas de instalação falhou. Verifique os logs acima."
  exit 1
fi

# ------------------------------------------------------------------------------
# FASE 4/4 — Verificação final
# ------------------------------------------------------------------------------
echo ""
echo "=== FASE 4/4: Verificação final ==="
kubectl get pods -n arc-systems

echo ""
echo "Ambiente reconstruído com sucesso. Runners disponíveis para 'runs-on:' no GitHub Actions:"
echo "  - arc-runner-set       (execução padrão, sem GPU)"
echo "  - arc-runner-set-gpu   (execução com GPU física, variante '$VARIANT')"
