#!/bin/bash
# ==============================================================================
# 1 - CRIA O CLUSTER KIND COM SUPORTE A GPU NVIDIA
# ==============================================================================
#
# ATENÇÃO — DESTRUTIVO: apaga qualquer cluster "kind" existente e tudo que
# estava rodando nele. Rode este script sempre que precisar recriar o cluster
# do zero (ex: mudou alguma configuração de GPU/mounts).
#
# Pré-requisitos que este script NÃO instala (fazer uma vez por máquina):
#   - Docker instalado e rodando
#   - kubectl instalado (dl.k8s.io)
#   - kind instalado (kubernetes-sigs/kind)
#   - Helm instalado
#   - Driver NVIDIA + NVIDIA Container Toolkit instalados
#   - Docker configurado com runtime nvidia como padrão:
#       sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
#       sudo systemctl restart docker
#   - /etc/nvidia-container-runtime/config.toml com a linha ativa (sem #):
#       accept-nvidia-visible-devices-as-volume-mounts = true
#
# ==============================================================================

set -e

# Resolve o caminho deste script (e a raiz do repo) mesmo quando chamado de
# outro diretório — ex: a task do VS Code roda com
# "${workspaceFolder}/scripts/1-create-gpu-cluster.sh".
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
KIND_CONFIG="$REPO_ROOT/k8s/kind-gpu-config.yaml"

if [ ! -f "$KIND_CONFIG" ]; then
  echo "Erro: não encontrei $KIND_CONFIG — rode este script de dentro do repo clonado."
  exit 1
fi

# A config do cluster já é a mesma versionada em k8s/kind-gpu-config.yaml —
# usada direto daqui, sem gerar uma cópia solta em ~/, pra não haver risco de
# as duas ficarem dessincronizadas.
echo "=== Recriando o cluster kind (config: $KIND_CONFIG) ==="
kind delete cluster --name kind 2>/dev/null || true
kind create cluster --config "$KIND_CONFIG"

echo "=== Confirmando o node ==="
kubectl get nodes

echo "=== Confirmando GPU visível dentro do container do node ==="
docker exec -it kind-control-plane nvidia-smi

echo ""
echo "Cluster criado com sucesso. Próximo passo: rode ./2-setup-arc.sh SEU_USUARIO/SEU_REPOSITORIO"
