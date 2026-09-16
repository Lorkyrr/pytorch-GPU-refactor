"""
TESTE COMPLETO DE AMBIENTE LOCAL - PYTORCH + GPU NVIDIA
=========================================================
Ponto de entrada do projeto. A lógica vive no pacote `pytorch_gpu_sandbox/`;
este arquivo só existe pra manter `python main.py` funcionando como sempre
funcionou — o Dockerfile, o compose.yaml e o compose.debug.yaml chamam esse
comando diretamente.
"""

from pytorch_gpu_sandbox.cli import main

if __name__ == "__main__":
    main()
