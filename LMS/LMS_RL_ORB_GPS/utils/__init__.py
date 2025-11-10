"""
Paquete de Utilidades para LMS_RL_ORB_GPS
Módulos organizados por funcionalidad
"""

# GPS
from .gps.gps_utils import latlon_to_utm
from .gps.gps_filter import GPSFilter

# Métricas
from .metrics.trajectory_metrics import TrajectoryMetrics
from .metrics.medir_metricas import medir_metricas, calculate_ate, calculate_rpe

# Procesamiento
from .processing.mobile_video_processor import MobileVideoProcessor
from .processing.training_augmentation import TrainingAugmentation

# Visualización
from .visualization.trajectory_plotter import TrajectoryPlotter

__all__ = [
    # GPS
    'latlon_to_utm',
    'GPSFilter',
    
    # Métricas
    'TrajectoryMetrics',
    'medir_metricas',
    'calculate_ate',
    'calculate_rpe',
    
    # Procesamiento
    'MobileVideoProcessor',
    'TrainingAugmentation',
    
    # Visualización
    'TrajectoryPlotter',
]