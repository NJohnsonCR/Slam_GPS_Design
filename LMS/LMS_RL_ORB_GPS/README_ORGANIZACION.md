# ORGANIZACIÓN DEL PROYECTO RL-ORB-SLAM-GPS

**Sistema Híbrido:** Reglas Heurísticas (70%) + Ajuste Fino RL (30%)

---

## ARCHIVOS ACTIVOS (SISTEMA HÍBRIDO ACTUAL)

### 1. ARCHIVO PRINCIPAL

#### `main.py` [ACTIVO]
- **Ubicación:** `LMS/LMS_RL_ORB_GPS/main.py`
- **Función:** Sistema SLAM con fusión GPS usando Reglas + RL
- **Características clave:**
  - Clase `RL_ORB_SLAM_GPS`: Hereda de `PoseGraphSLAM`
  - Método `calculate_base_weights()`: Calcula pesos base con **reglas heurísticas suaves**
  - Método `process_frame_with_gps()`: Fusiona SLAM + GPS usando reglas + RL
  - Soporte para GPS simulado con ruido (`simulate_mobile_gps=True`)
- **Uso:**
  ```bash
  # Entrenamiento
  venv/bin/python -m LMS.LMS_RL_ORB_GPS.main kitti_data/.../drive_0009_sync --kitti --train
  
  # Inferencia
  venv/bin/python -m LMS.LMS_RL_ORB_GPS.main kitti_data/.../drive_0009_sync --kitti
  ```

---

### 2. MODELO RL

#### `model/rl_agent.py` [ACTIVO]
- **Ubicación:** `LMS/LMS_RL_ORB_GPS/model/rl_agent.py`
- **Contenido:**
  - **Clase `SimpleRLAgent`:** Red neuronal mejorada (4 capas: 3→32→32→16→2)
  - **Clase `RLTrainer`:** Entrenador con replay buffer y Policy Gradient
- **Arquitectura mejorada:**
  ```
  Input (3):  [visual_conf, gps_conf, error]
     ↓
  FC1 (32):   Primera capa oculta
     ↓
  FC2 (32):   Segunda capa oculta
     ↓
  FC3 (16):   Tercera capa oculta
     ↓
  FC4 (2):    Output [w_slam, w_gps] con Softmax
  ```
- **Mejoras vs versión anterior:**
  - 4 capas (vs 2 capas antiguas)
  - 32 neuronas (vs 8 neuronas antiguas)
  - Dropout (10%) para evitar overfitting
  - ~1,600 parámetros totales

#### `model/trained_rl_agent.pth` [ACTIVO]
- **Modelo entrenado actual** con arquitectura mejorada
- Entrenado con secuencia 0009 (447 frames, ~1,400 updates)
- Mejora: 14.6% con GPS degradado (10m ruido)

#### `model/trained_rl_agent_metrics.json` [ACTIVO]
- Métricas del entrenamiento (losses, rewards, etc.)
- Generado automáticamente al entrenar

---

### 3. UTILIDADES GPS

#### `utils/gps_utils.py` [ACTIVO]
- **Funciones:**
  - `latlon_to_utm()`: Convierte coordenadas geográficas a UTM (metros)
  - `utm_to_latlon()`: Conversión inversa
- **Uso:** Procesar datos GPS de KITTI y VIRec

#### `utils/gps_filter.py` [ACTIVO]
- **Clase `GPSFilter`:**
  - Filtrado de mediciones GPS
  - Cálculo de confianza GPS basado en varianza
  - Soporte para **simulación de ruido GPS móvil**
- **Características:**
  - `add_measurement()`: Agrega medición GPS
  - `calculate_confidence()`: Calcula confianza [0, 1]
  - Parámetro `add_noise`: Simula GPS móvil con ruido gaussiano

#### `utils/medir_metricas.py` [ACTIVO]
- **Funciones:**
  - `calculate_ate()`: Calcula Absolute Trajectory Error
  - `calculate_rpe()`: Calcula Relative Pose Error
- **Uso:** Usado por `test_rl_model.py` para evaluar precisión

---

### 4. SCRIPTS DE EVALUACIÓN Y ENTRENAMIENTO

#### `scripts/test_rl_model.py` [ACTIVO - MUY IMPORTANTE]
- **Función:** Evaluar modelo entrenado con múltiples niveles de ruido GPS
- **Características:**
  - Evalúa con GPS perfecto (0m) + 3 niveles de ruido (3m, 5m, 10m)
  - Calcula ATE, RPE
  - Compara con GPS puro (sin fusión)
  - Genera gráficas comparativas
  - Guarda resultados en JSON
- **Uso:**
  ```bash
  venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.test_rl_model \
    --sequence kitti_data/.../drive_0001_sync \
    --model LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth
  ```

#### `scripts/process_mobile_video.py` [FUTURO - para VIRec]
- Para procesar videos de VIRec (GPS móvil real)
- **NO USAR AHORA** - Solo cuando tengas videos grabados
- **Uso futuro:**
  ```bash
  venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.process_mobile_video \
    video.mp4 gps_log.txt
  ```

#### `scripts/verify_model_weights.py` [ACTIVO - UTILIDAD]
- Verifica pesos del modelo entrenado
- Útil para debugging
- **Uso:**
  ```bash
  venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.verify_model_weights
  ```

---

## ESTRUCTURA ORGANIZADA ACTUAL

```
Slam_GPS_Design/
│
├── LMS/
│   └── LMS_RL_ORB_GPS/                ← Sistema RL híbrido (TODO AUTOCONTENIDO)
│       │
│       ├── main.py                     [ACTIVO] Principal (Reglas + RL)
│       │
│       ├── model/                      ← Modelo RL
│       │   ├── __init__.py
│       │   ├── rl_agent.py            [ACTIVO] Red neuronal + Trainer
│       │   ├── trained_rl_agent.pth   [ACTIVO] Modelo entrenado actual
│       │   └── trained_rl_agent_metrics.json [ACTIVO] Métricas
│       │
│       ├── utils/                      ← Utilidades GPS y métricas
│       │   ├── __init__.py
│       │   ├── gps_utils.py           [ACTIVO] Conversión lat/lon ↔ UTM
│       │   ├── gps_filter.py          [ACTIVO] Filtrado y confianza GPS
│       │   └── medir_metricas.py      [ACTIVO] Cálculo ATE/RPE
│       │
│       ├── scripts/                    ← Scripts de evaluación
│       │   ├── test_rl_model.py       [ACTIVO] Evaluador principal
│       │   ├── process_mobile_video.py [FUTURO] Para VIRec
│       │   └── verify_model_weights.py [ACTIVO] Verificador
│       │
│       ├── logs/                       ← Logs de ejecución
│       │   └── *.log                  [ACTIVO] Historial de entrenamientos
│       │
│       └── README_ORGANIZACION.md     [ACTIVO] Este archivo
│
├── resultados/LMS_RL_ORB_GPS/         ← Resultados de evaluaciones
│   └── <timestamp>/
│       ├── evaluation_results.json      [ACTIVO] Métricas
│       ├── evaluation_report.txt        [ACTIVO] Reporte legible
│       ├── evaluation_noise_*.png       [ACTIVO] Gráficas
│       └── improvement_analysis.png     [ACTIVO] Análisis
│
├── kitti_data/                         ← Datos KITTI
│
├── calcularCSV.py                      (Otros sistemas LMS)
├── interfaz.py                         (Otros sistemas LMS)
├── verify_model_weights.py             [ACTIVO] Verificador (acceso rápido)
└── venv/                               ← Entorno virtual
```

---

## FLUJO DE TRABAJO ACTUAL

### PASO 1: Entrenamiento
```bash
venv/bin/python -m LMS.LMS_RL_ORB_GPS.main \
  kitti_data/2011_09_26/2011_09_26_drive_0009_sync \
  --kitti --train --simulate_mobile_gps --gps_noise_std 5.0
```
**Output:** `LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth`

### PASO 2: Evaluación
```bash
venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.test_rl_model \
  --sequence kitti_data/2011_09_26/2011_09_26_drive_0001_sync \
  --model LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth \
  --noise_levels 0.0 3.0 5.0 10.0
```
**Output:** `resultados/LMS_RL_ORB_GPS/<timestamp>/`

### PASO 3: Verificar Modelo (Opcional)
```bash
venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.verify_model_weights
# O desde la raíz:
venv/bin/python verify_model_weights.py
```

---

## DIFERENCIAS: RL PURO vs RL HÍBRIDO

| Aspecto | RL Puro (Obsoleto) | RL Híbrido (Actual) |
|---------|-------------------|---------------------|
| **Pesos base** | RL desde cero | Reglas heurísticas |
| **Ajuste RL** | 100% (todo el peso) | ±25% (ajuste fino) |
| **Arquitectura** | 2 capas, 8 neuronas | 4 capas, 32 neuronas |
| **Datos necesarios** | 10,000+ frames | 447 frames |
| **Generalización** | Mala | Buena (reglas base) |
| **Mejora típica** | 0-5% | 6-15% |

---

## COMANDOS ÚTILES

### Ver estructura del modelo:
```bash
venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.verify_model_weights
```

### Entrenar con nueva secuencia:
```bash
venv/bin/python -m LMS.LMS_RL_ORB_GPS.main \
  kitti_data/2011_09_26/2011_09_26_drive_0005_sync \
  --kitti --train --simulate_mobile_gps --gps_noise_std 5.0
```

### Evaluar modelo entrenado:
```bash
venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.test_rl_model \
  --sequence kitti_data/2011_09_26/2011_09_26_drive_0018_sync
```

### Ver logs de entrenamiento:
```bash
tail -f LMS/LMS_RL_ORB_GPS/logs/entrenamiento_mejorado.log
```

---

## RESUMEN

**Sistema actual:** Híbrido (Reglas + RL) - TODO AUTOCONTENIDO en `LMS/LMS_RL_ORB_GPS/`

**Ventajas de la nueva organización:**
- Todo el sistema RL en una sola carpeta
- Fácil de compartir/publicar
- Estructura modular y clara
- Imports consistentes

**Archivos principales:**
- `main.py` - Sistema híbrido principal
- `model/trained_rl_agent.pth` - Modelo entrenado
- `scripts/test_rl_model.py` - Evaluador
- `utils/` - Utilidades GPS y métricas

**Última actualización:** 20 de octubre de 2025
