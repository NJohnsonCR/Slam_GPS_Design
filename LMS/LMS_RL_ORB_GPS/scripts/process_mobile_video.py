#!/usr/bin/env python3
"""
Script para procesar videos móviles con GPS en entrenamiento
Formato de entrada: video.mp4 + gps_log.txt
"""

import argparse
import sys
import os
import cv2
import numpy as np
from datetime import datetime

from LMS.LMS_RL_ORB_GPS.main import RL_ORB_SLAM_GPS

def load_gps_log(gps_log_path):
    """
    Carga archivo GPS del móvil
    
    Formato esperado (CSV):
    timestamp,latitude,longitude,altitude
    1634567890.123,10.123456,-84.123456,1500.5
    
    Returns:
        list: [(timestamp, lat, lon, alt), ...]
    """
    gps_data = []
    
    with open(gps_log_path, 'r') as f:
        # Saltar header si existe
        header = f.readline()
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                try:
                    timestamp = float(parts[0])
                    lat = float(parts[1])
                    lon = float(parts[2])
                    alt = float(parts[3])
                    gps_data.append((timestamp, lat, lon, alt))
                except ValueError:
                    continue
    
    print(f"OK - Cargadas {len(gps_data)} mediciones GPS")
    return gps_data

def interpolate_gps(gps_data, frame_idx, fps=30.0):
    """
    Interpola GPS para un frame específico
    
    Args:
        gps_data: Lista de (timestamp, lat, lon, alt)
        frame_idx: Índice del frame
        fps: FPS del video
    
    Returns:
        np.array: [lat, lon, alt] o None si no hay datos
    """
    if not gps_data or frame_idx >= len(gps_data):
        # Si no hay sincronización perfecta, usar índice directo
        if frame_idx < len(gps_data):
            _, lat, lon, alt = gps_data[frame_idx]
            return np.array([lat, lon, alt])
        return None
    
    # Opción 1: Mapeo directo (más simple)
    _, lat, lon, alt = gps_data[min(frame_idx, len(gps_data)-1)]
    return np.array([lat, lon, alt])

def process_mobile_video(video_path, gps_log_path, training_mode=True, model_path=None):
    """
    Procesa un video móvil con su log GPS
    """
    print("="*70)
    print("PROCESANDO VIDEO MÓVIL CON GPS")
    print("="*70)
    print(f"Video: {video_path}")
    print(f"GPS Log: {gps_log_path}")
    print(f"Modo: {'ENTRENAMIENTO' if training_mode else 'INFERENCIA'}")
    print("="*70)
    print()
    
    # Verificar archivos
    if not os.path.exists(video_path):
        print(f"ERROR - ERROR: Video no encontrado: {video_path}")
        return
    
    if not os.path.exists(gps_log_path):
        print(f"ERROR - ERROR: GPS log no encontrado: {gps_log_path}")
        return
    
    # Cargar GPS
    gps_data = load_gps_log(gps_log_path)
    if not gps_data:
        print("ERROR - ERROR: No se pudieron cargar datos GPS")
        return
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR - ERROR: No se pudo abrir el video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"OK - Video cargado: {total_frames} frames @ {fps} FPS")
    
    # Obtener parámetros de calibración de la cámara móvil
    # IMPORTANTE: Estos valores son aproximados para cámaras móviles
    # Deberías calibrar tu cámara específica para mejores resultados
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Valores típicos para cámara móvil
    fx = fy = width * 1.2  # Factor típico
    cx = width / 2.0
    cy = height / 2.0
    
    print(f"📷 Parámetros de cámara (aprox):")
    print(f"   Resolución: {width}x{height}")
    print(f"   fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"   ADVERTENCIA:  IMPORTANTE: Considera calibrar tu cámara para mejor precisión")
    print()
    
    # Crear instancia SLAM
    # NOTA: simulate_mobile_gps=False porque tu GPS YA ES ruidoso
    slam = RL_ORB_SLAM_GPS(
        fx=fx, fy=fy, cx=cx, cy=cy,
        training_mode=training_mode,
        model_path=model_path,
        simulate_mobile_gps=False,  # Tu GPS real ya tiene ruido
        gps_noise_std=0.0
    )
    
    print(f"\n{'='*60}")
    print(f"PROCESANDO {total_frames} FRAMES")
    print(f"{'='*60}\n")
    
    # Procesar frame por frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Frame {frame_idx}/{total_frames}...")
        
        # Obtener GPS para este frame
        gps_point = interpolate_gps(gps_data, frame_idx, fps)
        
        if gps_point is not None:
            # Convertir a UTM
            gps_utm = slam.gps_frame_reference(gps_point)
            
            # Agregar al filtro y calcular confianza
            slam.gps_filter.add_measurement(gps_utm)
            gps_confidence = slam.gps_filter.calculate_confidence()
            
            # Procesar frame con GPS
            slam.process_frame_with_gps(frame, gps_utm, gps_confidence)
        else:
            print(f"  ADVERTENCIA:  No hay GPS disponible para frame {frame_idx}")
            # Procesar sin GPS
            slam.process_frame_with_gps(frame, None, 0.0)
        
        frame_idx += 1
    
    cap.release()
    
    print("\n" + "="*70)
    print("PROCESAMIENTO COMPLETADO")
    print("="*70)
    
    # Guardar modelo si estamos entrenando
    if training_mode:
        model_save_path = "LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth"
        metadata = {
            'video': video_path,
            'total_frames': frame_idx,
            'total_experiences': len(slam.trainer.replay_buffer),
            'training_steps': len(slam.trainer.training_metrics['losses']),
            'mobile_video': True
        }
        slam.trainer.save_model(model_save_path, metadata)
        slam.trainer.print_training_summary()
        print(f"\nOK - Modelo guardado en: {model_save_path}")
    
    # Guardar trayectoria
    optimized_trajectory = slam.optimize_pose_graph()
    slam.save_trajectory_outputs(optimized_trajectory, video_path)
    
    print(f"\nOK - Trayectoria guardada en: resultados/LMS_RL_ORB_GPS/")

def main():
    parser = argparse.ArgumentParser(
        description='Procesar video móvil con GPS para entrenamiento/prueba de RL'
    )
    parser.add_argument(
        '--video',
        required=True,
        help='Ruta al video MP4 del móvil'
    )
    parser.add_argument(
        '--gps_log',
        required=True,
        help='Ruta al archivo GPS (CSV: timestamp,lat,lon,alt)'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Modo entrenamiento (actualiza el modelo)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='Ruta al modelo pre-entrenado (para continuar entrenamiento o inferencia)'
    )

    args = parser.parse_args()
    
    process_mobile_video(
        video_path=args.video,
        gps_log_path=args.gps_log,
        training_mode=args.train,
        model_path=args.model_path
    )

if __name__ == "__main__":
    main()
