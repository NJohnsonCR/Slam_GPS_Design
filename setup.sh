#!/bin/bash
#
# Deja el ambiente listo en una computadora nueva (Linux Mint 22 / Ubuntu 24.04):
# paquetes del sistema, entorno virtual con las versiones de requirements.txt,
# el modelo de profundidad descargado y la GPU revisada. Se corre una vez, con
# internet, desde la raíz del repositorio. Se puede volver a correr.

# Terminar inmediatamente si ocurre un error
set -e
cd "$(dirname "$0")"

if [ "$1" != "--sin-apt" ]; then
    echo "Instalando dependencias del sistema..."
    sudo apt update
    # adb: para traer grabaciones del teléfono y revisar la conexión por USB.
    sudo apt install -y \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        build-essential \
        python3-tk \
        libgl1 \
        libglib2.0-0t64 \
        adb
    echo "Dependencias del sistema instaladas."
fi

if ! command -v python3.12 > /dev/null; then
    echo "ERROR: no está python3.12. Linux Mint 22 y Ubuntu 24.04 lo traen;"
    echo "en versiones anteriores hay que instalarlo aparte."
    exit 1
fi

# Crear entorno virtual con Python 3.12
echo "Creando entorno virtual..."
python3.12 -m venv venv

# Activar entorno virtual
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip

echo "Instalando dependencias de Python..."
# Un entorno que ya existía puede traer opencv-contrib-python, que choca con
# opencv-python.
pip uninstall -y opencv-contrib-python opencv-python-headless opencv-contrib-python-headless \
    > /dev/null 2>&1 || true
pip install -r requirements.txt

opencv=$(pip list 2>/dev/null | grep -ci "^opencv")
if [ "$opencv" != "1" ]; then
    echo "ERROR: quedaron $opencv paquetes de OpenCV instalados; tiene que haber uno solo:"
    pip list 2>/dev/null | grep -i "^opencv"
    exit 1
fi

# El modelo de profundidad se descarga ahora, con internet: en campo el sistema
# lo carga desde el disco. También avisa si la GPU no sirve con esta versión de
# torch (las más recientes no soportan algunas tarjetas viejas).
echo "Descargando el modelo de profundidad y revisando la GPU..."
python - <<'EOF'
import sys
import numpy as np
sys.path.insert(0, "LMS/LMS_RL_ORB_GPS")
import cv2, tkinter, PIL.ImageTk, pyproj, pandas, scipy, sklearn, simplekml, torch  # noqa: F401
from realtime.depth_scale import DepthScaleEstimator, usable_gpu
DepthScaleEstimator(np.eye(3), device=-1)
print("Modelo de profundidad descargado.")
print(f"OpenCV {cv2.__version__}, torch {torch.__version__} (CUDA {torch.version.cuda})")
gpu = usable_gpu()
if gpu:
    print(f"GPU disponible: {gpu}")
elif torch.cuda.is_available():
    print("AVISO: hay GPU, pero esta versión de torch no corre en ella; el modelo de")
    print("profundidad correrá en la CPU (más lento). Ver 'Si algo falla en la laptop'")
    print("en scripts/rl/CONTEXTO.md.")
else:
    print("AVISO: sin GPU utilizable (¿driver NVIDIA instalado? probar nvidia-smi); el")
    print("modelo de profundidad correrá en la CPU (más lento).")
EOF

echo "Entorno configurado correctamente."
