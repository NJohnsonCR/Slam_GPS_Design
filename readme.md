# SLAM Visual con GPS y Reinforcement Learning

Este proyecto implementa un sistema de SLAM (Simultaneous Localization and Mapping) que integra visión monocular, datos GPS y técnicas de Reinforcement Learning para mejorar la precisión de localización. Está diseñado para ser modular, visualmente atractivo y extensible, con énfasis en la fusión de sensores y optimización mediante aprendizaje por refuerzo.

Proyecto desarrollado por Naheem Johnson Solis, estudiante del Tecnológico de Costa Rica (TEC) durante el segundo semestre de 2025, como parte del curso CE-1114 - Proyecto de Aplicación de la Ingeniería en Computadores. 

Se agradece el apoyo del profesor asesor MSc. Luis Alberto Chavarría Zamora.

## Propósito del proyecto

El proyecto fue creado con fines educativos e investigativos, como parte de un trabajo académico del Tecnológico de Costa Rica (TEC). Su objetivo es permitir a los estudiantes y desarrolladores:

- Integrar datos GPS con SLAM visual para mejorar la estimación de trayectorias.
- Aplicar técnicas de Reinforcement Learning para optimizar la fusión de sensores.
- Visualizar y comparar trayectorias estimadas con datos GPS reales.
- Evaluar el rendimiento del sistema en diferentes escenarios y condiciones.
- Facilitar la experimentación con diferentes estrategias de fusión de datos.

## Características principales

- Interfaz gráfica desarrollada en `Tkinter` con diseño moderno y responsivo.
- Integración de datos GPS con SLAM visual mediante fusión de sensores.
- Implementación de algoritmos de Reinforcement Learning para optimización.
- Ejecución con selección dinámica de video `.mp4` y archivos GPS.
- Cálculo automático de métricas de rendimiento (CPU, memoria, precisión).
- Visualización comparativa de trayectorias: SLAM puro vs GPS vs fusión.
- Exportación de resultados en formato CSV y PNG.
- Resultados organizados por fecha y tipo de experimento.

## Requisitos del sistema

- Ubuntu 22.04 o superior.
- Python 3.12 (con soporte para venv y tkinter).
- Acceso a internet para instalación de dependencias.
- Video de entrada en formato `.mp4`.
- Archivo de datos GPS (formato compatible con el sistema).

## Instalación

1. Clone este repositorio o descargue los archivos en una carpeta local:

```bash
git clone https://github.com/NJohnsonCR/Slam_GPS_Design.git
cd Slam_GPS_Design
```

2. Genere el ambiente virtual e instale las dependencias con este script:

```bash
chmod +x setup.sh
./setup.sh
```

## Ejecución

1. Active el ambiente virtual de Python, el cual contiene lo necesario para la ejecución.

```bash
source venv/bin/activate
```

2. Ejecute el archivo de la interfaz.

```bash
python3 interfaz.py
```

3. Una vez abierta la interfaz gráfica, seleccione el tipo de datos a procesar:
   
   **Opción A: Usar datos KITTI**
   - Haga clic en el botón "Seleccionar Trayectoria KITTI"
   - Navegue hasta la carpeta `kitti_data/2011_09_26/`
   - Seleccione una de las secuencias disponibles (ej: `2011_09_26_drive_0001_sync`)
   - El sistema cargará automáticamente las imágenes de la carpeta `image_02/data/` y los datos GPS de `oxts/data/`

   **Opción B: Usar datos móviles**
   - Haga clic en el botón "Seleccionar Datos Móviles"
   - Navegue hasta la carpeta `mobile_data/`
   - Seleccione la trayectoria deseada (ej: `/2025_03_11`)
   - Seleccione el archivo de video (ej: `movie.mp4`)
   - El sistema cargará automáticamente el archivo GPS correspondiente (`location.csv`)

4. Configure los parámetros del algoritmo si es necesario (ventana de configuración en la interfaz).

5. Presione el botón "Ejecutar SLAM" para iniciar el procesamiento.

6. Espere a que finalice la ejecución y revise los resultados generados en la carpeta `resultados/`.

## Manual de usuario

pendiente

### Resultados 

Los resultados se almacenan automáticamente en una jerarquía de carpetas similar a la siguiente:

```
resultados/
    └── SLAM_GPS_RL/
        └── 1245_2505_2025/             (hora-minuto_dia-mes_año)
            ├── trayectoria_slam_visual.csv
            ├── trayectoria_gps.csv
            ├── trayectoria_fusionada.csv
            ├── metricas_rendimiento.csv
            ├── comparacion_trayectorias.png
            └── graficas_rl_training.png
```