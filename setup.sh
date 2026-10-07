#!/bin/bash

# Terminar inmediatamente si ocurre un error
set -e

echo "Instalando dependencias del sistema..."

# Instalar dependencias necesarias del sistema
sudo apt update
sudo apt install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    python3.12-tk \
    libgl1 \
    libglib2.0-0

echo "Dependencias del sistema instaladas."

# Crear entorno virtual con Python 3.12
echo "Creando entorno virtual..."
python3.12 -m venv venv

# Activar entorno virtual
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias de Python. Una sola versión de OpenCV: opencv-python y
# opencv-contrib-python instalan el mismo módulo cv2 y chocan.
echo "Instalando dependencias de Python..."
pip install \
    opencv-python \
    numpy \
    matplotlib \
    scipy \
    psutil \
    pillow \
    torch \
    transformers \
    pyproj \
    pandas \
    scikit-learn \
    simplekml

# El modelo de profundidad se descarga ahora, con internet: en campo el sistema
# lo carga desde el disco. También avisa si la GPU no sirve con esta versión de
# torch (las más recientes no soportan algunas tarjetas viejas).
echo "Descargando el modelo de profundidad..."
python - <<'EOF'
import sys
import numpy as np
sys.path.insert(0, "LMS/LMS_RL_ORB_GPS")
from realtime.depth_scale import DepthScaleEstimator, usable_gpu
DepthScaleEstimator(np.eye(3), device=-1)
print("Modelo de profundidad descargado.")
gpu = usable_gpu()
print(f"GPU disponible: {gpu}" if gpu else
      "AVISO: sin GPU utilizable; el modelo de profundidad correrá en la CPU (más lento).")
EOF

echo "Entorno configurado correctamente."
