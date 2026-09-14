#!/bin/bash
# ==============================================================================
# 3 - DERRUBA O CLUSTER, LIBERANDO GPU/RAM/CPU
# ==============================================================================
#
# Roda quando você não vai usar o cluster por um tempo. Como o kind cria tudo
# como containers Docker comuns, apagar o cluster libera a GPU e os recursos
# do seu notebook imediatamente — não fica nada "preso" rodando em segundo
# plano.
#
# Nada precisa ser regerado depois: a config do cluster (k8s/kind-gpu-config.yaml)
# e os values dos runner sets (k8s/gpu-runner-values*.yaml) já ficam
# versionados no repo, prontos pra próxima vez que você rodar
# 1-create-gpu-cluster.sh + 2-setup-arc.sh.
#
# ==============================================================================

set -e

echo "=== Verificando se o cluster existe ==="
if ! kind get clusters 2>/dev/null | grep -q "^kind$"; then
  echo "Nenhum cluster 'kind' encontrado. Nada a fazer."
  exit 0
fi

echo "=== Derrubando o cluster ==="
kind delete cluster --name kind

echo "=== Confirmando que a GPU está livre no host ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo ""
echo "Cluster derrubado. RAM, CPU e GPU liberados."
echo "Para montar de novo: ./1-create-gpu-cluster.sh && export GITHUB_TOKEN='...' && ./2-setup-arc.sh SEU_USUARIO/SEU_REPOSITORIO"
