"""
Módulo de Procesamiento - Video móvil y aumentación de datos
"""

from .mobile_video_processor import MobileVideoProcessor
from .training_augmentation import TrainingAugmentation

__all__ = [
    'MobileVideoProcessor',
    'TrainingAugmentation',
]
