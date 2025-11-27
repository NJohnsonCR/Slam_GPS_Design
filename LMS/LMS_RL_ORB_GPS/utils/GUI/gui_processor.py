"""
Wrapper para procesar datasets desde la interfaz gráfica.
"""

import os
import sys
from typing import Optional
from io import StringIO
import contextlib

# Importar el detector
from .dataset_detector import DatasetDetector, DatasetInfo

# Importar el sistema principal
current_dir = os.path.dirname(os.path.abspath(__file__))
lms_rl_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, lms_rl_dir)

from main import RL_ORB_SLAM_GPS

class GUIProcessor:
    """Procesa datasets automáticamente desde la GUI."""
    
    @staticmethod
    def process_directory(
        base_path: str,
        max_frames: Optional[int] = None,
        training_mode: bool = False,
        augmentation_prob: float = 0.0,
        progress_callback=None
    ) -> dict:
        """
        Procesa un directorio detectando automáticamente el tipo de dataset.
        
        Args:
            base_path: Ruta al directorio seleccionado
            max_frames: Número máximo de frames (None = todos)
            training_mode: Si está en modo entrenamiento
            augmentation_prob: Probabilidad de augmentación (solo KITTI)
            progress_callback: Función callback(message: str) para reportar progreso
            
        Returns:
            dict con resultados del procesamiento
        """
        def log(message: str):
            """Helper para logging."""
            if progress_callback:
                progress_callback(message)
            else:
                print(message)
        
        # Capturador de prints para redirigir a la GUI
        class OutputCapture:
            """Captura stdout y lo redirige al callback de progreso."""
            def __init__(self, callback):
                self.callback = callback
                self.buffer = StringIO()
                
            def write(self, text):
                if text and text.strip():  # Solo enviar líneas no vacías
                    if self.callback:
                        self.callback(text.rstrip())
                    self.buffer.write(text)
                    
            def flush(self):
                pass
        
        # 1. Detectar tipo de dataset
        log(f"Analizando directorio: {base_path}")
        dataset_info = DatasetDetector.detect_dataset_type(base_path)
        
        log(f"\n{DatasetDetector.get_dataset_summary(dataset_info)}")
        
        # 2. Validar dataset
        is_valid, error_msg = DatasetDetector.validate_dataset(dataset_info)
        if not is_valid:
            return {
                'success': False,
                'error': error_msg,
                'dataset_type': dataset_info.dataset_type
            }
        
        # 3. Inicializar sistema
        log("\nInicializando RL-ORB-SLAM con GPS...")
        
        # Redirigir stdout para capturar todos los prints del sistema
        old_stdout = sys.stdout
        capture = OutputCapture(progress_callback)
        
        try:
            sys.stdout = capture
            
            system = RL_ORB_SLAM_GPS(
                training_mode=training_mode,
                simulate_mobile_gps=False
            )
            
            # 4. Procesar según tipo de dataset
            if dataset_info.dataset_type == 'kitti':
                log(f"\nProcesando secuencia KITTI: {dataset_info.sequence_path}")
                
                # Usar directamente el sequence_path que ya es el directorio correcto
                system.process_kitti_sequence(
                    sequence_path=dataset_info.sequence_path,
                    max_frames=max_frames,
                    augmentation_prob=augmentation_prob
                )
                
                result_type = "KITTI"
                
            elif dataset_info.dataset_type == 'mobile':
                log(f"\nProcesando video móvil")
                
                system.process_mobile_sequence(
                    video_path=dataset_info.video_path,
                    location_csv=dataset_info.gps_csv_path,
                    frame_timestamps_file=dataset_info.timestamps_path,
                    max_frames = max_frames
                )
                
                result_type = "Mobile"
            
            else:
                sys.stdout = old_stdout
                return {
                    'success': False,
                    'error': f"Tipo de dataset no soportado: {dataset_info.dataset_type}",
                    'dataset_type': dataset_info.dataset_type
                }
            
            log("\nProcesamiento completado exitosamente")
            
            sys.stdout = old_stdout
            
            return {
                'success': True,
                'dataset_type': dataset_info.dataset_type,
                'result_type': result_type,
                'keyframes': len(system.keyframe_poses) if hasattr(system, 'keyframe_poses') else 0
            }
            
        except Exception as e:
            import traceback
            sys.stdout = old_stdout  # Restaurar stdout antes de manejar el error
            
            error_trace = traceback.format_exc()
            log(f"\nError durante procesamiento: {e}")
            log(error_trace)
            
            return {
                'success': False,
                'error': str(e),
                'traceback': error_trace,
                'dataset_type': dataset_info.dataset_type
            }
        
        finally:
            # Asegurar que stdout siempre se restaure
            sys.stdout = old_stdout