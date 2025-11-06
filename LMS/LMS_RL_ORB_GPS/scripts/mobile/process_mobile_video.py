#!/usr/bin/env python3
"""
Script para procesar videos móviles con GPS en entrenamiento
Formato de entrada: video.mp4 + location.csv + frame_timestamps.txt
"""

import argparse
import sys
import os
import cv2
import numpy as np
from datetime import datetime
import pandas as pd

from LMS.LMS_RL_ORB_GPS.main import RL_ORB_SLAM_GPS

def load_mobile_gps_data(location_csv_path):
    """
    Carga archivo location.csv del móvil con formato:
    Time (ns), Latitude (°), Longitude (°), Altitude (m), Speed (m/s), Unix (ns)
    
    Returns:
        pd.DataFrame: DataFrame con columnas ['time_ns', 'latitude', 'longitude', 'altitude', 'speed', 'unix_ns']
    """
    try:
        # Leer CSV
        df = pd.read_csv(location_csv_path)
        
        # Verificar columnas esperadas
        expected_cols = ['Time (ns)', 'Latitude (°)', 'Longitude (°)', 'Altitude (m)']
        for col in expected_cols:
            if col not in df.columns:
                print(f"ERROR - Columna faltante: {col}")
                return None
        
        # Renombrar columnas para facilitar uso
        df_clean = pd.DataFrame({
            'time_ns': df['Time (ns)'],
            'latitude': df['Latitude (°)'],
            'longitude': df['Longitude (°)'],
            'altitude': df['Altitude (m)'],
            'speed': df['Speed (m/s)'] if 'Speed (m/s)' in df.columns else 0.0,
            'unix_ns': df['Unix (ns)'] if 'Unix (ns)' in df.columns else df['Time (ns)']
        })
        
        print(f"OK - Cargadas {len(df_clean)} mediciones GPS")
        print(f"   Rango temporal: {df_clean['time_ns'].min()} - {df_clean['time_ns'].max()} ns")
        print(f"   Coordenadas: Lat [{df_clean['latitude'].min():.6f}, {df_clean['latitude'].max():.6f}]")
        print(f"                Lon [{df_clean['longitude'].min():.6f}, {df_clean['longitude'].max():.6f}]")
        
        return df_clean
        
    except Exception as e:
        print(f"ERROR - Error al cargar GPS: {e}")
        return None

def load_frame_timestamps(timestamp_file_path):
    """
    Carga archivo frame_timestamps.txt con timestamps de cada frame
    
    Formato esperado: Un timestamp (ns) por línea
    
    Returns:
        np.array: Array de timestamps en nanosegundos
    """
    try:
        timestamps = []
        with open(timestamp_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    timestamps.append(int(line))
        
        timestamps = np.array(timestamps)
        print(f"OK - Cargados {len(timestamps)} timestamps de frames")
        print(f"   Rango: {timestamps.min()} - {timestamps.max()} ns")
        
        return timestamps
        
    except Exception as e:
        print(f"ERROR - Error al cargar timestamps: {e}")
        return None

def synchronize_gps_to_frames(gps_df, frame_timestamps):
    """
    Sincroniza mediciones GPS con timestamps de frames
    Usa interpolación para estimar GPS en cada frame
    
    Args:
        gps_df: DataFrame con datos GPS
        frame_timestamps: Array de timestamps de frames (ns)
    
    Returns:
        list: Lista de [lat, lon, alt] para cada frame (None si no hay datos)
    """
    gps_per_frame = []
    
    # Convertir a arrays para interpolación eficiente
    gps_times = gps_df['unix_ns'].values
    gps_lats = gps_df['latitude'].values
    gps_lons = gps_df['longitude'].values
    gps_alts = gps_df['altitude'].values
    
    for frame_ts in frame_timestamps:
        # Encontrar mediciones GPS más cercanas
        time_diffs = np.abs(gps_times - frame_ts)
        closest_idx = np.argmin(time_diffs)
        
        # Si la diferencia es muy grande (>5 segundos), no usar GPS
        time_diff_seconds = time_diffs[closest_idx] / 1e9
        if time_diff_seconds > 5.0:
            gps_per_frame.append(None)
            continue
        
        # Interpolación lineal si hay mediciones cercanas
        if closest_idx > 0 and closest_idx < len(gps_times) - 1:
            # Determinar si interpolar hacia adelante o atrás
            if gps_times[closest_idx] < frame_ts:
                idx1, idx2 = closest_idx, closest_idx + 1
            else:
                idx1, idx2 = closest_idx - 1, closest_idx
            
            # Factor de interpolación
            t1, t2 = gps_times[idx1], gps_times[idx2]
            if t2 - t1 > 0:
                alpha = (frame_ts - t1) / (t2 - t1)
                alpha = np.clip(alpha, 0, 1)
                
                # Interpolar lat, lon, alt
                lat = gps_lats[idx1] * (1 - alpha) + gps_lats[idx2] * alpha
                lon = gps_lons[idx1] * (1 - alpha) + gps_lons[idx2] * alpha
                alt = gps_alts[idx1] * (1 - alpha) + gps_alts[idx2] * alpha
            else:
                # Usar valor más cercano
                lat = gps_lats[closest_idx]
                lon = gps_lons[closest_idx]
                alt = gps_alts[closest_idx]
        else:
            # Usar valor más cercano
            lat = gps_lats[closest_idx]
            lon = gps_lons[closest_idx]
            alt = gps_alts[closest_idx]
        
        gps_per_frame.append(np.array([lat, lon, alt]))
    
    # Estadísticas
    valid_gps = sum(1 for g in gps_per_frame if g is not None)
    print(f"OK - Sincronización: {valid_gps}/{len(frame_timestamps)} frames con GPS válido")
    
    return gps_per_frame

def process_mobile_video(video_path, mobile_data_dir, training_mode=True, model_path=None):
    """
    Procesa un video móvil con sus datos GPS sincronizados
    
    Args:
        video_path: Ruta al video MP4
        mobile_data_dir: Directorio con location.csv y frame_timestamps.txt
        training_mode: True para entrenamiento, False para inferencia
        model_path: Ruta a modelo pre-entrenado (opcional)
    """
    print("="*70)
    print("PROCESANDO VIDEO MÓVIL CON GPS SINCRONIZADO")
    print("="*70)
    print(f"Video: {video_path}")
    print(f"Datos: {mobile_data_dir}")
    print(f"Modo: {'ENTRENAMIENTO' if training_mode else 'INFERENCIA'}")
    print("="*70)
    print()
    
    # Verificar archivos
    if not os.path.exists(video_path):
        print(f"ERROR - Video no encontrado: {video_path}")
        return
    
    location_csv = os.path.join(mobile_data_dir, 'location.csv')
    timestamps_file = os.path.join(mobile_data_dir, 'frame_timestamps.txt')
    
    if not os.path.exists(location_csv):
        print(f"ERROR - location.csv no encontrado: {location_csv}")
        return
    
    if not os.path.exists(timestamps_file):
        print(f"ERROR - frame_timestamps.txt no encontrado: {timestamps_file}")
        return
    
    # Cargar datos GPS
    print("\nCARGANDO DATOS GPS...")
    gps_df = load_mobile_gps_data(location_csv)
    if gps_df is None or len(gps_df) == 0:
        print("ERROR - No se pudieron cargar datos GPS")
        return
    
    # Cargar timestamps de frames
    print("\nCARGANDO TIMESTAMPS DE FRAMES...")
    frame_timestamps = load_frame_timestamps(timestamps_file)
    if frame_timestamps is None or len(frame_timestamps) == 0:
        print("ERROR - No se pudieron cargar timestamps de frames")
        return
    
    # Sincronizar GPS con frames
    print("\nSINCRONIZANDO GPS CON FRAMES...")
    gps_per_frame = synchronize_gps_to_frames(gps_df, frame_timestamps)
    
    # Abrir video
    print("\nABRIENDO VIDEO...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR - No se pudo abrir el video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"OK - Video cargado: {total_frames} frames @ {fps:.2f} FPS")
    print(f"   Resolución: {width}x{height}")
    
    # Verificar que coincidan frames
    if len(gps_per_frame) != total_frames:
        print(f"ADVERTENCIA - Mismatch: {len(gps_per_frame)} timestamps vs {total_frames} frames")
        print(f"            Usando min({len(gps_per_frame)}, {total_frames})")
        num_frames_to_process = min(len(gps_per_frame), total_frames)
    else:
        num_frames_to_process = total_frames
    
    # Parámetros de cámara móvil (aproximados)
    # IMPORTANTE: Deberías calibrar tu cámara para mejores resultados
    fx = fy = width * 1.2  # Factor típico para cámaras móviles
    cx = width / 2.0
    cy = height / 2.0
    
    print(f"\nPARAMETROS DE CAMARA (APROXIMADOS):")
    print(f"   fx={fx:.2f}, fy={fy:.2f}")
    print(f"   cx={cx:.2f}, cy={cy:.2f}")
    print(f"   IMPORTANTE: Considera calibrar tu cámara para mejor precisión")
    
    # Crear instancia SLAM
    slam = RL_ORB_SLAM_GPS(
        fx=fx, fy=fy, cx=cx, cy=cy,
        training_mode=training_mode,
        model_path=model_path,
        simulate_mobile_gps=False,  # GPS real ya tiene ruido
        gps_noise_std=0.0
    )
    
    print(f"\n{'='*70}")
    print(f"PROCESANDO {num_frames_to_process} FRAMES")
    print(f"{'='*70}\n")
    
    # Procesar frame por frame
    frame_idx = 0
    frames_with_gps = 0
    
    while frame_idx < num_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 50 == 0:
            print(f"Frame {frame_idx}/{num_frames_to_process}...")
        
        # Obtener GPS para este frame
        gps_point = gps_per_frame[frame_idx] if frame_idx < len(gps_per_frame) else None
        
        if gps_point is not None:
            # Convertir a UTM
            gps_utm = slam.gps_frame_reference(gps_point)
            
            # Agregar al filtro y calcular confianza
            slam.gps_filter.add_measurement(gps_utm)
            gps_confidence = slam.gps_filter.calculate_confidence()
            
            # Procesar frame con GPS
            slam.process_frame_with_gps(frame, gps_utm, gps_confidence)
            frames_with_gps += 1
        else:
            # Procesar sin GPS
            slam.process_frame_with_gps(frame, None, 0.0)
        
        frame_idx += 1
    
    cap.release()
    
    print("\n" + "="*70)
    print("PROCESAMIENTO COMPLETADO")
    print("="*70)
    print(f"Frames procesados: {frame_idx}")
    print(f"Frames con GPS: {frames_with_gps} ({frames_with_gps/frame_idx*100:.1f}%)")
    
    # Guardar modelo si estamos entrenando
    if training_mode:
        model_save_path = "LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth"
        metadata = {
            'video': os.path.basename(video_path),
            'total_frames': frame_idx,
            'frames_with_gps': frames_with_gps,
            'total_experiences': len(slam.trainer.replay_buffer),
            'training_steps': len(slam.trainer.training_metrics['losses']),
            'mobile_video': True,
            'data_source': 'mobile_recorded'
        }
        slam.trainer.save_model(model_save_path, metadata)
        slam.trainer.print_training_summary()
        print(f"\nModelo guardado en: {model_save_path}")
    
    # Guardar trayectoria
    optimized_trajectory = slam.optimize_pose_graph()
    slam.save_trajectory_outputs(optimized_trajectory, video_path)
    
    print(f"\nTrayectoria guardada en: resultados/LMS_RL_ORB_GPS/")

def main():
    parser = argparse.ArgumentParser(
        description='Procesar video móvil con GPS para entrenamiento/prueba de RL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Entrenar con datos móviles:
  python -m LMS.LMS_RL_ORB_GPS.scripts.mobile.process_mobile_video \\
      --video mobile_data/2025_03_11/movie.mp4 \\
      --data_dir mobile_data/2025_03_11 \\
      --train

  # Inferencia con modelo pre-entrenado:
  python -m LMS.LMS_RL_ORB_GPS.scripts.mobile.process_mobile_video \\
      --video mobile_data/2025_03_11/movie.mp4 \\
      --data_dir mobile_data/2025_03_11 \\
      --model_path LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth
        """
    )
    parser.add_argument(
        '--video',
        required=True,
        help='Ruta al video MP4 del móvil'
    )
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Directorio con location.csv y frame_timestamps.txt'
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
        mobile_data_dir=args.data_dir,
        training_mode=args.train,
        model_path=args.model_path
    )

if __name__ == "__main__":
    main()
