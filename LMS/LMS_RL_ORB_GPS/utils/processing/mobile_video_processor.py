"""
Módulo para procesar videos móviles y sincronizar con datos GPS.

Este módulo convierte un video continuo (MP4) en frames individuales sincronizados
con mediciones GPS, simulando el formato frame-por-frame de KITTI.
"""

import cv2
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


class MobileVideoProcessor:
    """
    Procesa videos móviles y sincroniza con datos GPS.
    
    Convierte video MP4 → frames individuales + GPS por frame
    (similar a cómo KITTI maneja image_02/*.png + oxts/*.txt)
    """
    
    def __init__(self, slam_system, video_path: str, gps_csv_path: str, 
                 timestamps_path: str, max_frames: Optional[int] = None):
        """
        Args:
            slam_system: Instancia de RL_ORB_SLAM_GPS
            video_path: Ruta al archivo de video (MP4)
            gps_csv_path: Ruta al CSV con datos GPS (location.csv)
            timestamps_path: Ruta al archivo con timestamps de frames (frame_timestamps.txt)
            max_frames: Número máximo de frames a procesar (None = todos)
        """
        self.slam_system = slam_system
        self.video_path = video_path
        self.gps_csv_path = gps_csv_path
        self.timestamps_path = timestamps_path
        self.max_frames = max_frames
        
        self.gps_df = None
        self.frame_timestamps = None
        self.gps_per_frame = None
        
        print("\n" + "="*60)
        print("MOBILE VIDEO PROCESSOR - Inicializado")
        print("="*60)
    
    def process(self):
        """
        Ejecuta el pipeline completo: carga datos, extrae frames y procesa con SLAM.
        
        Este método orquesta todo el procesamiento frame-por-frame similar a KITTI.
        """
        print("\nIniciando procesamiento de video móvil...")
        
        # 1. Cargar y sincronizar datos
        if not self.load_data():
            print("ERROR: No se pudieron cargar los datos")
            return
        
        # 2. Mostrar estadísticas
        stats = self.get_statistics()
        print("\nEstadísticas de datos:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # 3. Procesar frames uno por uno (similar a KITTI)
        print("\nProcesando frames frame-por-frame...")
        print("-" * 60)
        
        frame_count = 0
        gps_used_count = 0
        
        for frame_idx, frame, gps_data in self.get_frame_iterator(self.max_frames):
            print(f"\nFrame {frame_idx}:")
            
            # Convertir GPS a UTM si está disponible
            gps_utm = None
            gps_confidence = 0.1  # Baja confianza por defecto
            
            if gps_data is not None:
                # Procesar GPS como en KITTI (usando el método del sistema)
                gps_utm = self.slam_system.gps_frame_reference(gps_data)
                print(f"   GPS disponible: Lat={gps_data[0]:.6f}, Lon={gps_data[1]:.6f}")
                print(f"   GPS UTM: {gps_utm.round(3)}")
                
                # Calcular confianza GPS usando el filtro
                self.slam_system.gps_filter.add_measurement(gps_utm)
                gps_confidence = self.slam_system.gps_filter.calculate_confidence()
                self.slam_system.gps_filter.print_debug_info()
                
                gps_used_count += 1
            else:
                print(f"   GPS no disponible para este frame")
            
            # Procesar frame con SLAM (igual que KITTI)
            try:
                pose = self.slam_system.process_frame_with_gps(frame, gps_utm, gps_confidence)
                
                if pose is not None:
                    print(f"   Pose estimada exitosamente")
                else:
                    print(f"   No se pudo estimar pose")
                    
            except Exception as e:
                print(f"   ERROR procesando frame: {e}")
                import traceback
                traceback.print_exc()
            
            frame_count += 1
        
        # 4. Finalizar procesamiento
        print("\n" + "="*60)
        print(f"Procesamiento completado")
        print(f"   Frames procesados: {frame_count}")
        print(f"   Frames con GPS: {gps_used_count} ({gps_used_count/frame_count*100:.1f}%)")
        print("="*60 + "\n")
        
        # 5. Optimizar y guardar resultados
        print("Optimizando trayectoria...")
        optimized_trajectory = self.slam_system.optimize_pose_graph()
        
        print("Guardando resultados...")
        self.slam_system.save_trajectory_outputs(optimized_trajectory, self.video_path)
        
        print("\nProcesamiento de video móvil completado exitosamente\n")
    
    def load_data(self) -> bool:
        """
        Carga y sincroniza todos los datos necesarios.
        
        Returns:
            True si la carga fue exitosa, False en caso contrario
        """
        print("\nCargando datos...")
        
        # 1. Cargar datos GPS
        print("   Cargando GPS...")
        self.gps_df = self._load_gps_data()
        if self.gps_df is None:
            print("   ERROR: No se pudieron cargar datos GPS")
            return False
        
        # 2. Cargar timestamps de frames
        print("   Cargando timestamps de frames...")
        self.frame_timestamps = self._load_frame_timestamps()
        if self.frame_timestamps is None:
            print("   ERROR: No se pudieron cargar timestamps")
            return False
        
        # 3. Sincronizar GPS con frames
        print("   Sincronizando GPS con frames...")
        self.gps_per_frame = self._synchronize_gps_to_frames()
        
        print("\n   Datos cargados y sincronizados exitosamente\n")
        return True
    
    def get_frame_iterator(self, max_frames: Optional[int] = None):
        """
        Generador que produce frames del video con su GPS sincronizado.
        
        Yields:
            tuple: (frame_idx, frame_image, gps_data)
                - frame_idx: índice del frame
                - frame_image: imagen BGR (numpy array)
                - gps_data: [lat, lon, alt] o None si no hay GPS
        
        Args:
            max_frames: Número máximo de frames a procesar (None = todos)
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"ERROR: No se pudo abrir el video: {self.video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video abierto:")
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Duración: {total_frames/fps:.2f} segundos")
        
        # Determinar cuántos frames procesar
        process_frames = total_frames
        if max_frames:
            process_frames = min(max_frames, total_frames)
        if self.gps_per_frame is not None:
            process_frames = min(process_frames, len(self.gps_per_frame))
        
        print(f"   Procesando: {process_frames} frames\n")
        
        # Iterar sobre los frames
        frame_idx = 0
        while frame_idx < process_frames:
            ret, frame = cap.read()
            if not ret:
                print(f"   Fin del video en frame {frame_idx}")
                break
            
            # ====================================================================
            # FIX: ACTUALIZAR CONTADOR DE FRAME_ID EN EL SISTEMA SLAM
            # ====================================================================
            self.slam_system.current_frame_id = frame_idx
            # ====================================================================
            
            # Obtener GPS sincronizado para este frame
            gps_data = None
            if self.gps_per_frame is not None and frame_idx < len(self.gps_per_frame):
                gps_data = self.gps_per_frame[frame_idx]
            
            # NO SKIPPING - Procesar TODOS los frames
            yield frame_idx, frame, gps_data
            frame_idx += 1
        
        cap.release()
    
    def _load_gps_data(self) -> Optional[pd.DataFrame]:
        """
        Carga archivo location.csv del móvil.
        
        Formato esperado (flexible):
        - Timestamp[nanosecond] o Time (ns)
        - latitude[degrees] o Latitude (°)
        - longitude[degrees] o Longitude (°)
        - altitude[meters] o Altitude (m)
        
        Returns:
            DataFrame con columnas ['time_ns', 'latitude', 'longitude', 'altitude', 'speed', 'unix_ns']
        """
        try:
            df = pd.read_csv(self.gps_csv_path)
            
            # Mapeo flexible de nombres de columnas
            column_mapping = {
                'time_ns': ['Timestamp[nanosecond]', 'Time (ns)', 'timestamp', 'time'],
                'latitude': ['latitude[degrees]', 'Latitude (°)', 'lat', 'latitude'],
                'longitude': ['longitude[degrees]', 'Longitude (°)', 'lon', 'longitude'],
                'altitude': ['altitude[meters]', 'Altitude (m)', 'alt', 'altitude'],
                'speed': ['speed[meters/second]', 'Speed (m/s)', 'speed'],
                'unix_ns': ['Unix time[nanosecond]', 'Unix (ns)', 'unix_time']
            }
            
            # Encontrar columnas existentes
            df_clean_data = {}
            for target_col, possible_names in column_mapping.items():
                found = False
                for name in possible_names:
                    if name in df.columns:
                        df_clean_data[target_col] = df[name]
                        found = True
                        break
                
                # Si no se encontró la columna, usar valores por defecto
                if not found:
                    if target_col == 'speed':
                        df_clean_data[target_col] = 0.0
                    elif target_col == 'unix_ns':
                        if 'time_ns' in df_clean_data:
                            df_clean_data[target_col] = df_clean_data['time_ns']
                        else:
                            print(f"      ADVERTENCIA: No se encontró columna para {target_col}")
                            return None
                    else:
                        print(f"      ERROR: No se encontró columna para {target_col}")
                        print(f"      Columnas disponibles: {list(df.columns)}")
                        return None
            
            df_clean = pd.DataFrame(df_clean_data)
            
            print(f"      Cargadas {len(df_clean)} mediciones GPS")
            print(f"      Rango temporal: {df_clean['time_ns'].min()} - {df_clean['time_ns'].max()} ns")
            print(f"      Coordenadas: Lat [{df_clean['latitude'].min():.6f}, {df_clean['latitude'].max():.6f}]")
            print(f"                   Lon [{df_clean['longitude'].min():.6f}, {df_clean['longitude'].max():.6f}]")
            
            return df_clean
            
        except Exception as e:
            print(f"      ERROR al cargar GPS: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_frame_timestamps(self) -> Optional[np.ndarray]:
        """
        Carga archivo frame_timestamps.txt con timestamps de cada frame.
        
        Formato esperado (flexible):
        - CSV con header: "Frame timestamp[nanosec],Unix time[nanosec]"
        - O archivo de texto simple con un timestamp por línea
        
        Returns:
            Array de timestamps Unix en nanosegundos
        """
        try:
            # Intentar leer como CSV primero
            try:
                df = pd.read_csv(self.timestamps_path)
                
                # Buscar columna de Unix time
                unix_col = None
                for col_name in ['Unix time[nanosec]', 'Unix (ns)', 'unix_time', 'Unix time']:
                    if col_name in df.columns:
                        unix_col = col_name
                        break
                
                # Si no hay Unix time, usar Frame timestamp
                if unix_col is None:
                    for col_name in ['Frame timestamp[nanosecond]', 'timestamp', 'time']:
                        if col_name in df.columns:
                            unix_col = col_name
                            break
                
                if unix_col is None:
                    print(f"      No se encontró columna de timestamp. Columnas: {list(df.columns)}")
                    print(f"      Intentando usar la segunda columna...")
                    if len(df.columns) >= 2:
                        timestamps = df.iloc[:, 1].values.astype(np.int64)
                    else:
                        timestamps = df.iloc[:, 0].values.astype(np.int64)
                else:
                    timestamps = df[unix_col].values.astype(np.int64)
                
                print(f"      Cargados {len(timestamps)} timestamps (formato CSV)")
                print(f"      Rango: {timestamps.min()} - {timestamps.max()} ns")
                print(f"      Duración: {(timestamps.max() - timestamps.min()) / 1e9:.2f} segundos")
                
                return timestamps
                
            except Exception as csv_error:
                # Si falla como CSV, intentar como texto plano
                print(f"      No es CSV válido, intentando formato texto plano...")
                timestamps = []
                with open(self.timestamps_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and line.isdigit():
                            timestamps.append(int(line))
                
                if len(timestamps) == 0:
                    print(f"      ERROR: No se encontraron timestamps válidos")
                    return None
                
                timestamps = np.array(timestamps)
                print(f"      Cargados {len(timestamps)} timestamps (formato texto)")
                print(f"      Rango: {timestamps.min()} - {timestamps.max()} ns")
                
                return timestamps
            
        except Exception as e:
            print(f"      ERROR al cargar timestamps: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _synchronize_gps_to_frames(self) -> List[Optional[np.ndarray]]:
        """
        Sincroniza mediciones GPS con timestamps de frames usando interpolación.
        
        Returns:
            Lista de [lat, lon, alt] para cada frame (None si no hay datos cercanos)
        """
        gps_per_frame = []
        
        # Convertir a arrays para interpolación eficiente
        gps_times = self.gps_df['unix_ns'].values
        gps_lats = self.gps_df['latitude'].values
        gps_lons = self.gps_df['longitude'].values
        gps_alts = self.gps_df['altitude'].values
        
        matched_count = 0
        max_time_diff_ns = 5e9  # 5 segundos en nanosegundos
        
        for frame_ts in self.frame_timestamps:
            # Encontrar mediciones GPS más cercanas
            time_diffs = np.abs(gps_times - frame_ts)
            closest_idx = np.argmin(time_diffs)
            
            # Si la diferencia es muy grande (>5 segundos), no usar GPS
            if time_diffs[closest_idx] > max_time_diff_ns:
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
                # Usar valor más cercano (borde del array)
                lat = gps_lats[closest_idx]
                lon = gps_lons[closest_idx]
                alt = gps_alts[closest_idx]
            
            gps_per_frame.append(np.array([lat, lon, alt]))
            matched_count += 1
        
        # Estadísticas
        total_frames = len(self.frame_timestamps)
        print(f"      Sincronización completada:")
        print(f"         Frames con GPS: {matched_count}/{total_frames} ({matched_count/total_frames*100:.1f}%)")
        print(f"         Frames sin GPS: {total_frames - matched_count}")
        
        return gps_per_frame
    
    def get_statistics(self) -> dict:
        """
        Obtiene estadísticas sobre los datos cargados.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {}
        
        if self.frame_timestamps is not None:
            stats['total_frames'] = len(self.frame_timestamps)
        
        if self.gps_per_frame is not None:
            stats['frames_with_gps'] = sum(1 for gps in self.gps_per_frame if gps is not None)
            stats['frames_without_gps'] = sum(1 for gps in self.gps_per_frame if gps is None)
        
        if self.gps_df is not None:
            stats['total_gps_measurements'] = len(self.gps_df)
        
        return stats
