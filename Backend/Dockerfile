FROM python:3.11-slim

# Evitar que Python genere archivos .pyc y forzar que la salida estándar no tenga búfer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema requeridas por OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivo de requerimientos
COPY requirements.txt .

# Forzar a pip a extraer paquetes pesados en disco en lugar de memoria RAM (evita EOF/OOM)
ENV TMPDIR=/var/tmp

# 1. Instalar PyTorch primero para aislar la capa más pesada
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# 2. Instalar el resto de dependencias
RUN pip install --no-cache-dir openmim
RUN grep -v "torch==" requirements.txt | grep -v "torchvision==" | grep -v "torch-directml" > requirements_gpu.txt && \
    pip install --no-cache-dir -r requirements_gpu.txt && \
    mim install --no-cache-dir "mmcv==2.1.0"

# 3. Instalar Causal Conv1d y Mamba
RUN pip install --no-cache-dir https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.1/causal_conv1d-1.1.1+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN pip install --no-cache-dir https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Copiar el resto del código del backend
COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
