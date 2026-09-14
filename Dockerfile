# Usa a imagem oficial do PyTorch com suporte a CUDA
FROM pytorch/pytorch:latest

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Atualiza pip e garante que torchvision e torchaudio estejam instalados
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torchvision torchaudio

# Copia os arquivos do projeto para o container
COPY . /app

# Comando padrão ao rodar o container
CMD ["python3", "main.py"]