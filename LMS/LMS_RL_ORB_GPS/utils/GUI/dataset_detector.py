"""
Utilidad para detectar automáticamente el tipo de dataset (KITTI o Mobile)
basándose en la estructura de directorios.
"""

import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class DatasetInfo:
    """Información detectada del dataset."""
    dataset_type: str  # 'kitti', 'mobile', o 'unknown'
    base_path: str
    sequence_path: Optional[str] = None
    video_path: Optional[str] = None
    gps_csv_path: Optional[str] = None
    timestamps_path: Optional[str] = None
    confidence: float = 0.0
    error_message: Optional[str] = None

class DatasetDetector:
    """Detecta automáticamente si un directorio es KITTI o Mobile dataset."""
    
    @staticmethod
    def detect_dataset_type(base_path: str) -> DatasetInfo:
        """
        Detecta el tipo de dataset basándose en la estructura de directorios.
        
        Args:
            base_path: Ruta al directorio base a analizar
            
        Returns:
            DatasetInfo con la información detectada
        """
        if not os.path.exists(base_path):
            return DatasetInfo(
                dataset_type='unknown',
                base_path=base_path,
                confidence=0.0,
                error_message=f"Directorio no existe: {base_path}"
            )
        
        # Intentar detectar KITTI
        kitti_info = DatasetDetector._detect_kitti(base_path)
        if kitti_info.dataset_type == 'kitti':
            return kitti_info
        
        # Intentar detectar Mobile
        mobile_info = DatasetDetector._detect_mobile(base_path)
        if mobile_info.dataset_type == 'mobile':
            return mobile_info
        
        # No se pudo detectar
        return DatasetInfo(
            dataset_type='unknown',
            base_path=base_path,
            confidence=0.0,
            error_message="No se detectó estructura KITTI ni Mobile"
        )
    
    @staticmethod
    def _detect_kitti(base_path: str) -> DatasetInfo:
        """Detecta si el directorio tiene estructura KITTI."""
        confidence = 0.0
        image_dir = None
        gps_dir = None
        sequence_base = None
        
        # Buscar estructura típica de KITTI
        for root, dirs, files in os.walk(base_path):
            # Buscar image_02/data
            if "image_02" in dirs:
                img_candidate = os.path.join(root, "image_02", "data")
                if os.path.exists(img_candidate):
                    png_files = [f for f in os.listdir(img_candidate) 
                                if f.endswith('.png')]
                    if png_files:
                        image_dir = img_candidate
                        # Guardar el directorio que CONTIENE image_02/
                        sequence_base = root
                        confidence += 0.5
            
            # Buscar oxts/data
            if "oxts" in dirs:
                gps_candidate = os.path.join(root, "oxts", "data")
                if os.path.exists(gps_candidate):
                    txt_files = [f for f in os.listdir(gps_candidate) 
                                if f.endswith('.txt')]
                    if txt_files:
                        gps_dir = gps_candidate
                        confidence += 0.5
            
            # Si encontramos ambos, es KITTI con alta confianza
            if image_dir and gps_dir:
                break
        
        if confidence >= 0.5 and sequence_base:
            # Retornar el directorio base que contiene image_02/ y oxts/
            # Este es el path que necesita main.py
            return DatasetInfo(
                dataset_type='kitti',
                base_path=sequence_base,  # El directorio que contiene image_02/ y oxts/
                sequence_path=sequence_base,  # Mismo path para compatibilidad
                confidence=confidence
            )
        
        return DatasetInfo(
            dataset_type='unknown',
            base_path=base_path,
            confidence=0.0
        )
    
    @staticmethod
    def _detect_mobile(base_path: str) -> DatasetInfo:
        """Detecta si el directorio tiene estructura Mobile (video + GPS CSV)."""
        confidence = 0.0
        video_path = None
        gps_csv = None
        timestamps = None
        
        # Buscar archivos característicos
        for root, dirs, files in os.walk(base_path):
            for file in files:
                full_path = os.path.join(root, file)
                
                # Buscar video MP4
                if file.endswith('.mp4') or file.endswith('.MP4'):
                    video_path = full_path
                    confidence += 0.4
                
                # Buscar location.csv
                if 'location' in file.lower() and file.endswith('.csv'):
                    gps_csv = full_path
                    confidence += 0.4
                
                # Buscar frame_timestamps.txt
                if 'timestamp' in file.lower() and file.endswith('.txt'):
                    timestamps = full_path
                    confidence += 0.2
            
            # No necesitamos recorrer todo el árbol
            if confidence >= 0.8:
                break
        
        if confidence >= 0.6:  # Al menos video + GPS
            return DatasetInfo(
                dataset_type='mobile',
                base_path=base_path,
                video_path=video_path,
                gps_csv_path=gps_csv,
                timestamps_path=timestamps,
                confidence=confidence
            )
        
        return DatasetInfo(
            dataset_type='unknown',
            base_path=base_path,
            confidence=0.0
        )
    
    @staticmethod
    def validate_dataset(info: DatasetInfo) -> Tuple[bool, Optional[str]]:
        """
        Valida que el dataset detectado tenga todos los archivos necesarios.
        
        Returns:
            (is_valid, error_message)
        """
        if info.dataset_type == 'unknown':
            return False, info.error_message or "Tipo de dataset desconocido"
        
        if info.dataset_type == 'kitti':
            if not info.sequence_path:
                return False, "No se pudo determinar la secuencia KITTI"
            
            # El sequence_path ya es la ruta completa, no concatenar
            if not os.path.exists(info.sequence_path):
                return False, f"Secuencia no existe: {info.sequence_path}"
            
            # Verificar que existan las carpetas necesarias
            image_path = os.path.join(info.sequence_path, "image_02", "data")
            oxts_path = os.path.join(info.sequence_path, "oxts", "data")
            
            if not os.path.exists(image_path):
                return False, f"No existe image_02/data en: {info.sequence_path}"
            if not os.path.exists(oxts_path):
                return False, f"No existe oxts/data en: {info.sequence_path}"
            
            return True, None
        
        if info.dataset_type == 'mobile':
            if not info.video_path:
                return False, "No se encontró archivo de video (.mp4)"
            if not info.gps_csv_path:
                return False, "No se encontró archivo GPS (location.csv)"
            if not info.timestamps_path:
                return False, "No se encontró archivo de timestamps"
            
            if not os.path.exists(info.video_path):
                return False, f"Video no existe: {info.video_path}"
            if not os.path.exists(info.gps_csv_path):
                return False, f"GPS CSV no existe: {info.gps_csv_path}"
            if not os.path.exists(info.timestamps_path):
                return False, f"Timestamps no existe: {info.timestamps_path}"
            
            return True, None
        
        return False, "Tipo de dataset no soportado"
    
    @staticmethod
    def get_dataset_summary(info: DatasetInfo) -> str:
        """Genera un resumen legible del dataset detectado."""
        if info.dataset_type == 'unknown':
            return f"Dataset desconocido\n{info.error_message or ''}"
        
        summary = f"Dataset detectado: {info.dataset_type.upper()}\n"
        summary += f"  Confianza: {info.confidence*100:.0f}%\n"
        summary += f"  Ruta base: {info.base_path}\n"
        
        if info.dataset_type == 'kitti':
            summary += f"  Secuencia: {info.sequence_path}\n"
        
        elif info.dataset_type == 'mobile':
            summary += f"  Video: {os.path.basename(info.video_path) if info.video_path else 'N/A'}\n"
            summary += f"  GPS: {os.path.basename(info.gps_csv_path) if info.gps_csv_path else 'N/A'}\n"
            summary += f"  Timestamps: {os.path.basename(info.timestamps_path) if info.timestamps_path else 'N/A'}\n"
        
        return summary