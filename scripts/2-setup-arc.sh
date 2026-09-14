#!/bin/bash
# ==============================================================================
# 2 - INSTALA DEVICE PLUGIN + ARC NO CLUSTER JÁ CRIADO
# ==============================================================================
#
# Rode depois de ./1-create-gpu-cluster.sh (ou sempre que o cluster kind for
# recriado — o "kind delete cluster" apaga o namespace kube-system original,
# então o device plugin some junto e precisa ser reinstalado).
#
# Uso:
#   export GITHUB_TOKEN='seu_token_fine-grained_aqui'
#   ./2-setup-arc.sh SEU_USUARIO/SEU_REPOSITORIO [python|docker]
#
# O 2º argumento escolhe qual variante do runner com GPU fica ativa (ver
# CLAUDE.md / README.md, seção "CI on a real GPU"):
#   python (padrão do repo, mas exige ter buildado+publicado
#           k8s/runner-image/Dockerfile e apontado esse registry em
#           k8s/gpu-runner-values.yaml — senão o pod fica em ImagePullBackOff)
#   docker (usa a imagem oficial ghcr.io/actions/actions-runner + socket do
#           Docker do host — funciona sem build nenhum)
# Por isso o padrão AQUI é "docker": roda redondo sem passos manuais extras.
# Troque pra "python" só depois de ter publicado a imagem custom.
#
# O token precisa ter, no repositório escolhido:
#   Repository permissions -> Administration: Read and write
#
# Duas trilhas independentes rodam em PARALELO (o device plugin não depende
# do ARC, e os dois runner sets só dependem do controller, não um do outro):
#   Trilha A — NVIDIA Device Plugin
#   Trilha B — namespace/secret -> controller -> runner set (sem GPU) e
#              runner set (com GPU) em paralelo entre si
# Cada trilha grava seu próprio log e só é impressa (e checada por erro)
# depois que as duas terminam — evita saída embaralhada no terminal sem abrir
# mão do ganho de tempo.
#
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

REPO="$1"
VARIANT="${2:-docker}"

if [ -z "$GITHUB_TOKEN" ]; then
  echo "Erro: defina a variável GITHUB_TOKEN antes de rodar este script."
  echo "Exemplo: export GITHUB_TOKEN='seu_token_aqui'"
  exit 1
fi

if [ -z "$REPO" ]; then
  echo "Erro: informe o repositório como argumento."
  echo "Exemplo: ./2-setup-arc.sh Lorkyrr/pytorch-gpu-sandbox"
  exit 1
fi

case "$VARIANT" in
  python) GPU_VALUES_TEMPLATE="$REPO_ROOT/k8s/gpu-runner-values.yaml" ;;
  docker) GPU_VALUES_TEMPLATE="$REPO_ROOT/k8s/gpu-runner-values.docker.yaml" ;;
  *)
    echo "Erro: variante '$VARIANT' inválida — use 'python' ou 'docker'."
    exit 1
    ;;
esac

if [ ! -f "$GPU_VALUES_TEMPLATE" ]; then
  echo "Erro: não encontrei $GPU_VALUES_TEMPLATE — rode este script de dentro do repo clonado."
  exit 1
fi

GITHUB_URL="https://github.com/$REPO"

LOG_DEVICE_PLUGIN="$(mktemp)"
LOG_ARC="$(mktemp)"
trap 'rm -f "$LOG_DEVICE_PLUGIN" "$LOG_ARC"' EXIT

# ------------------------------------------------------------------------------
# Trilha A — NVIDIA Device Plugin
# ------------------------------------------------------------------------------
(
  set -e
  echo "=== Verificando NVIDIA Device Plugin ==="
  if ! kubectl get pods -n kube-system 2>/dev/null | grep -q nvidia-device-plugin; then
    echo "Device plugin não encontrado. Instalando a partir de k8s/nvidia-device-plugin.yaml..."
    kubectl create -f "$REPO_ROOT/k8s/nvidia-device-plugin.yaml"
  fi

  echo "Aguardando device plugin ficar pronto..."
  kubectl wait --for=condition=Ready pod \
    -l name=nvidia-device-plugin-ds \
    -n kube-system --timeout=120s

  echo "Aguardando o node anunciar a GPU..."
  GPU_OK=false
  for i in $(seq 1 30); do
    if kubectl describe node kind-control-plane | grep -q "nvidia.com/gpu:"; then
      echo "GPU detectada no node."
      GPU_OK=true
      break
    fi
    sleep 1
  done

  if [ "$GPU_OK" = false ]; then
    echo "GPU não apareceu na capacidade do node. Reinstalando device plugin..."
    kubectl delete pod -n kube-system -l name=nvidia-device-plugin-ds --ignore-not-found
    kubectl wait --for=condition=Ready pod \
      -l name=nvidia-device-plugin-ds \
      -n kube-system --timeout=120s
  fi
) > "$LOG_DEVICE_PLUGIN" 2>&1 &
PID_DEVICE_PLUGIN=$!

# ------------------------------------------------------------------------------
# Trilha B — namespace/secret, controller, e os dois runner sets
# ------------------------------------------------------------------------------
(
  set -e
  echo "=== Criando namespace e secret ==="
  kubectl create namespace arc-systems --dry-run=client -o yaml | kubectl apply -f -

  kubectl create secret generic pre-defined-secret \
    --namespace=arc-systems \
    --from-literal=github_token="$GITHUB_TOKEN" \
    --dry-run=client -o yaml | kubectl apply -f -

  echo "=== Instalando/atualizando o controller do ARC ==="
  helm upgrade --install arc \
    --namespace arc-systems \
    --create-namespace \
    oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller

  kubectl wait --for=condition=Ready pod \
    -l app.kubernetes.io/name=gha-rs-controller \
    -n arc-systems --timeout=90s

  # Runner set sem GPU e runner set com GPU só dependem do controller acima,
  # não um do outro — instalados em paralelo.
  echo "=== Instalando/atualizando runner set padrão (sem GPU) ==="
  helm upgrade --install arc-runner-set \
    --namespace arc-systems \
    --set githubConfigUrl="$GITHUB_URL" \
    --set githubConfigSecret=pre-defined-secret \
    oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set &
  PID_NOGPU=$!

  # Usa o values.yaml já versionado em k8s/ como template (única fonte de
  # verdade), só substituindo o githubConfigUrl pelo repo passado como
  # argumento. Evita duplicar o YAML dentro deste script e dessincronizar das
  # duas variantes documentadas em CLAUDE.md/README.md.
  echo "=== Gerando values da variante '$VARIANT' a partir de $GPU_VALUES_TEMPLATE ==="
  GENERATED_VALUES="$(mktemp)"
  trap 'rm -f "$GENERATED_VALUES"' EXIT
  sed "s#^githubConfigUrl:.*#githubConfigUrl: \"$GITHUB_URL\"#" "$GPU_VALUES_TEMPLATE" > "$GENERATED_VALUES"

  echo "=== Instalando/atualizando runner set com GPU (variante: $VARIANT) ==="
  helm upgrade --install arc-runner-set-gpu \
    --namespace arc-systems \
    -f "$GENERATED_VALUES" \
    oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set &
  PID_GPU=$!

  wait "$PID_NOGPU"
  wait "$PID_GPU"
) > "$LOG_ARC" 2>&1 &
PID_ARC=$!

echo "Instalando device plugin e ARC (controller + runner sets) em paralelo..."
FAIL=0
wait "$PID_DEVICE_PLUGIN" || FAIL=1
wait "$PID_ARC" || FAIL=1

echo ""
echo "--- log: Device Plugin ---"
cat "$LOG_DEVICE_PLUGIN"
echo ""
echo "--- log: ARC (namespace/secret/controller/runner sets) ---"
cat "$LOG_ARC"

if [ "$FAIL" -ne 0 ]; then
  echo ""
  echo "Erro: uma das trilhas acima falhou — veja o log correspondente."
  exit 1
fi

echo ""
echo "=== Tudo pronto! ==="
kubectl get pods -n arc-systems
echo ""
echo "Runners disponíveis para usar em runs-on: no workflow:"
echo "  - arc-runner-set       (sem GPU)"
echo "  - arc-runner-set-gpu   (com GPU, variante '$VARIANT')"
