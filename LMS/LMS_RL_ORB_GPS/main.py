import sys
import os
import cv2
import numpy as np
import torch
import argparse
import csv
from datetime import datetime
import matplotlib.pyplot as plt

# Agrega la carpeta LMS_ORB_with_PG al sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'LMS_ORB_with_PG'))
from LMS.LMS_ORB_with_PG.main import PoseGraphSLAM
from LMS.LMS_RL_ORB_GPS.rl_agent import SimpleRLAgent
from LMS.LMS_RL_ORB_GPS.gps_utils import latlon_to_utm

class RL_ORB_SLAM_GPS(PoseGraphSLAM):
    def __init__(self, fx=718.856, fy=718.856, cx=607.1928, cy=185.2157):
        super().__init__(fx, fy, cx, cy)
        self.agent = SimpleRLAgent()
        self.utm_reference = None
        self.previous_gps_utm = None
        self.gps_available = False
        self.gps_history = []  # Para almacenar historial de posiciones GPS
        self.slam_history = []  # Para almacenar historial de posiciones SLAM

    def process_kitti_gps(self, oxts_data):
        lat, lon, alt = oxts_data[0], oxts_data[1], oxts_data[2]
        utm_coord = latlon_to_utm(lat, lon, alt)
        if self.utm_reference is None:
            self.utm_reference = utm_coord
        return utm_coord - self.utm_reference
    
    def process_frame(self, frame, gps_data=None):
        if gps_data is None:
            return super().process_frame(frame)
        
        gps_utm = self.process_kitti_gps(gps_data)
        return self.process_frame_with_gps(frame, gps_utm)
    
    def process_frame_with_gps(self, frame, gps_utm, gps_confidence=0.9):
        minimumMatches = 15
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb_detector.detectAndCompute(gray, None)
        print(f"  Keypoints detectados: {len(keypoints) if keypoints else 0}")

        if self.previous_keyframe_descriptors is None:
            self.previous_keyframe_image = gray
            self.previous_keyframe_keypoints = keypoints
            self.previous_keyframe_descriptors = descriptors
            self.previous_keyframe_pose = np.eye(4)
            self.keyframe_poses.append(np.eye(4))
            print("  ✓ Primer frame inicializado")
            
            # Inicializar con GPS si está disponible
            if gps_utm is not None:
                initial_pose = np.eye(4)
                initial_pose[:3, 3] = gps_utm
                self.previous_keyframe_pose = initial_pose
                self.keyframe_poses[-1] = initial_pose
                self.previous_gps_utm = gps_utm.copy()
                print("  ✓ Inicializado con posición GPS")
                
            return None

        matches = self.filter_matches_lowe_ratio(self.previous_keyframe_descriptors, descriptors)
        print(f"  Matches encontrados: {len(matches)}")

        if len(matches) > minimumMatches:
            points_prev = np.float32([self.previous_keyframe_keypoints[m.queryIdx].pt for m in matches])
            points_curr = np.float32([keypoints[m.trainIdx].pt for m in matches])

            E, mask = cv2.findEssentialMat(points_prev, points_curr, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            print(f"  E matrix shape: {E.shape if E is not None else 'None'}")
            
            if E is not None:
                try:
                    points, R, t, mask = cv2.recoverPose(E, points_prev, points_curr, self.camera_matrix)
                    
                    print(f"  recoverPose success: t={t.ravel()} | Magnitude: {np.linalg.norm(t):.6f}")
                    print(f"  R det: {np.linalg.det(R):.6f}")
                    print(f"  inliers: {np.sum(mask) if mask is not None else 'N/A'}")
                    
                    relative_pose = np.eye(4)
                    relative_pose[:3, :3] = R
                    relative_pose[:3, 3] = t.ravel()

                    # ESCALADO CON GPS
                    escala = 1.0
                    if self.previous_gps_utm is not None:
                        distancia_real = np.linalg.norm(gps_utm - self.previous_gps_utm)
                        distancia_slam = np.linalg.norm(relative_pose[:3, 3])
                
                        print(f"  Debug: dist_slam={distancia_slam:.6f}, dist_real={distancia_real:.6f}")
                        print(f"  Debug GPS: dist_real={distancia_real:.6f}, gps_diff={(gps_utm - self.previous_gps_utm)}")
                        
                        if distancia_slam > 0.0001 and distancia_real > 0.01:
                            escala = distancia_real / distancia_slam
                            relative_pose[:3, 3] = relative_pose[:3, 3] * escala
                            print(f"  ✓ ESCALADO APLICADO: {distancia_slam:.6f} → {distancia_real:.6f} m")
                        else:
                            print(f"  ✗ Escalado omitido (condición no cumplida)")
                    
                    current_pose = self.previous_keyframe_pose @ relative_pose

                    # FUSIÓN CON RL
                    visual_confidence = min(len(matches) / 50.0, 1.0)
                    error = np.linalg.norm(current_pose[:3, 3] - gps_utm)

                    # ✅ PRINT 2: VERIFICAR POSE vs GPS (antes de RL)
                    print(f"  SLAM pose: {current_pose[:3,3].round(3)} | GPS: {gps_utm.round(3)} | Error: {error:.3f} m")

                    # vector estado para funcion RL: [confianza_visual, confianza_gps, error_entre_poses]
                    state = torch.tensor([visual_confidence, gps_confidence, error], dtype=torch.float32)
                    weights = self.agent(state).detach()

                    # ✅ PRINT 3: VERIFICAR PESOS RL (después de decisión)
                    print(f"  RL weights: SLAM={weights[0]:.2f}, GPS={weights[1]:.2f}")
                    
                    # COMBINAR SLAM Y GPS SEGÚN PESOS DEL RL
                    fused_position = weights[0] * current_pose[:3, 3] + weights[1] * gps_utm
                    fused_pose = np.eye(4)
                    fused_pose[:3, 3] = fused_position.numpy()
                    fused_pose[:3, :3] = current_pose[:3, :3]

                    # Guardar historial para análisis
                    self.slam_history.append(current_pose[:3, 3].copy())
                    self.gps_history.append(gps_utm.copy())

                    # ACTUALIZAR KEYFRAME SI HAY MOVIMIENTO SIGNIFICATIVO
                    translation_magnitude = np.linalg.norm(relative_pose[:3, 3])

                    self.total_successful_frames += 1
                    self.total_tracked_matches += len(matches)
                    self.total_translation_magnitude += translation_magnitude
                    self.total_pose_estimations += 1

                    if translation_magnitude > 0.1:
                        self.keyframe_poses.append(fused_pose)
                        self.relative_transformations.append(relative_pose)
                        self.previous_keyframe_image = gray
                        self.previous_keyframe_keypoints = keypoints
                        self.previous_keyframe_descriptors = descriptors
                        self.previous_keyframe_pose = fused_pose
                        print(f"  ✓ Nuevo keyframe (movimiento: {translation_magnitude:.3f} m)")
                    else:
                        print(f"  ✓ Pose estimada, pero no keyframe (movimiento pequeño)")

                    # GUARDAR GPS PARA PRÓXIMO FRAME
                    self.previous_gps_utm = gps_utm.copy()
                    self.gps_available = True
                    
                    return fused_pose
                    
                except Exception as e:
                    print(f"  ✗ Error en recoverPose: {e}")
                    return None
            else:
                print("  ✗ No se pudo calcular matriz esencial")
                return None

        print(f"  ✗ No suficientes matches o no se pudo calcular pose")
        return None
    
    def process_video_input(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        
        while video_capture.isOpened():
            success, frame = video_capture.read()
            if not success:
                break
                
            self.process_frame(frame)
            
        video_capture.release()
        
        optimized_trajectory = self.optimize_pose_graph()
        self.save_trajectory_outputs(optimized_trajectory, video_path)
    
    def process_kitti_sequence(self, sequence_path, max_frames=None):
        frames, gps_data = load_kitti_sequence(sequence_path, max_frames)
        
        for i in range(len(frames)):
            print(f"Procesando frame {i}...")
            
            if frames[i] is None:
                print(f"  ✗ Frame {i} es None")
                continue
                
            try:
                self.process_frame(frames[i], gps_data[i])
                    
            except Exception as e:
                print(f"  ✗ Error en frame {i}: {e}")
        
        optimized_trajectory = self.optimize_pose_graph()
        self.save_trajectory_outputs(optimized_trajectory, sequence_path)

    def save_trajectory_outputs(self, trajectory, input_path):
        tipo_lms = "LMS_RL_ORB_GPS"
        timestamp = datetime.now().strftime("%H%M_%d%m_%Y")
        output_dir = os.path.join("resultados", tipo_lms, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        output_base = os.path.join(output_dir, f"trayectoria_{tipo_lms}")

        # DEBUG: Verificar la trayectoria optimizada
        print(f"=== DEBUG: Trayectoria optimizada ===")
        print(f"Tipo: {type(trajectory)}")
        print(f"Longitud: {len(trajectory) if hasattr(trajectory, '__len__') else 'N/A'}")
        
        if len(trajectory) > 0:
            print(f"Primer elemento: {trajectory[0]}")
            if hasattr(trajectory[0], 'shape'):
                print(f"Forma primer elemento: {trajectory[0].shape}")

        # EXTRAER POSICIONES SIN OPTIMIZAR (las importantes)
        try:
            # Usar los keyframes originales en lugar de la trayectoria optimizada
            if hasattr(self, 'keyframe_poses') and len(self.keyframe_poses) > 0:
                keyframe_positions = np.array([pose[:3, 3] for pose in self.keyframe_poses])
                print(f"✓ Keyframes sin optimizar: {len(keyframe_positions)} poses")
                
                # Calcular estadísticas con keyframes REALES
                x_key, y_key = keyframe_positions[:, 0], keyframe_positions[:, 1]
                x_key -= x_key[0]
                y_key -= y_key[0]
                
                # Calcular distancia total REAL
                real_total_distance = 0.0
                for i in range(1, len(keyframe_positions)):
                    pos1 = keyframe_positions[i-1][:2]  # X, Y
                    pos2 = keyframe_positions[i][:2]    # X, Y
                    real_total_distance += np.linalg.norm(pos2 - pos1)
            else:
                print("✗ No hay keyframes disponibles")
                return
                
        except Exception as e:
            print(f"✗ Error al extraer keyframes: {e}")
            return

        # Calcular estadísticas REALES (sin optimización)
        stats = {
            "keyframes": len(self.keyframe_poses),
            "dist_total": real_total_distance,
            "matches_prom": self.total_tracked_matches / max(1, self.total_pose_estimations),
            "exito_triang": self.total_successful_frames / max(1, self.total_pose_estimations),
            "mov_medio": self.total_translation_magnitude / max(1, self.total_pose_estimations)
        }

        # === PRINTS
        print("\n=== Estadísticas REALES (sin optimización) ===")
        for k, v in stats.items():
            print(f"{k}: {v}")
        print("=============================================\n")

        # Guardar keyframes REALES (los que importan)
        with open(output_base + "_reales.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["X", "Y"])
            for i in range(len(x_key)):
                writer.writerow([x_key[i], y_key[i]])

        # === PLOT PRINCIPAL: KEYFRAMES REALES ===
        plt.figure(figsize=(14, 10))
        
        # 1. KEYFRAMES REALES (lo más importante)
        plt.plot(x_key, y_key, 'b-', label='Trayectoria Real (Keyframes)', linewidth=4, 
                marker='o', markersize=8, zorder=5)
        
        # 2. SLAM por frame (poses individuales)
        if hasattr(self, 'slam_history') and len(self.slam_history) > 0:
            slam_traj = np.array(self.slam_history)
            if slam_traj.shape[1] >= 2:
                x_slam, y_slam = slam_traj[:, 0], slam_traj[:, 1]
                x_slam -= x_slam[0]; y_slam -= y_slam[0]
                plt.plot(x_slam, y_slam, 'g:', label='SLAM por frame', linewidth=2, zorder=3, alpha=0.7)
        
        # 3. GPS puro (datos brutos)
        if hasattr(self, 'gps_history') and len(self.gps_history) > 0:
            gps_traj = np.array(self.gps_history)
            if gps_traj.shape[1] >= 2:
                x_gps, y_gps = gps_traj[:, 0], gps_traj[:, 1]
                x_gps -= x_gps[0]; y_gps -= y_gps[0]
                plt.plot(x_gps, y_gps, 'r-.', label='GPS puro', linewidth=2, zorder=2, alpha=0.8)

        # 4. Trayectoria "optimizada" (solo para referencia, probablemente mala)
        try:
            if all(hasattr(pose, 'shape') and pose.shape == (4, 4) for pose in trajectory):
                opt_positions = np.array([pose[:3, 3] for pose in trajectory])
                x_opt, y_opt = opt_positions[:, 0], opt_positions[:, 1]
                x_opt -= x_opt[0]; y_opt -= y_opt[0]
                plt.plot(x_opt, y_opt, 'm--', label='Trayectoria "Optimizada"', 
                        linewidth=1, zorder=1, alpha=0.5)
        except:
            print("⚠ No se pudo graficar trayectoria optimizada (probablemente colapsada)")

        # Puntos de referencia y anotaciones
        plt.scatter(x_key[0], y_key[0], color='green', s=250, label='Inicio', zorder=6)
        plt.scatter(x_key[-1], y_key[-1], color='red', s=250, label='Fin', zorder=6)
        
        # Añadir números a los keyframes para ver el orden
        for i in range(len(x_key)):
            plt.annotate(str(i), (x_key[i], y_key[i]), xytext=(8, 8), 
                        textcoords='offset points', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        info_text = (
            f"Keyframes: {stats['keyframes']}\n"
            f"Dist. total: {stats['dist_total']:.1f} m\n"
            f"Prom. matches: {stats['matches_prom']:.1f}\n"
            f"Éxito triang: {stats['exito_triang']*100:.2f}%\n"
            f"Mov. medio: {stats['mov_medio']:.2f} m"
        )
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=11, bbox=dict(facecolor='white', alpha=0.9),
                verticalalignment='top')

        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title(f"TRAYECTORIA REAL - {os.path.basename(input_path)}", fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        
        plt.savefig(output_base + "_REAL.png", dpi=300, bbox_inches='tight')
        plt.close()

        # === PLOT SOLO KEYFRAMES REALES ===
        plt.figure(figsize=(12, 9))
        plt.plot(x_key, y_key, 'b-', linewidth=3, marker='o', markersize=8)
        plt.scatter(x_key[0], y_key[0], color='green', s=200, label='Inicio')
        plt.scatter(x_key[-1], y_key[-1], color='red', s=200, label='Fin')
        
        # Añadir flechas de dirección
        if len(x_key) > 1:
            for i in range(len(x_key)-1):
                dx = x_key[i+1] - x_key[i]
                dy = y_key[i+1] - y_key[i]
                plt.arrow(x_key[i], y_key[i], dx*0.8, dy*0.8, 
                        head_width=0.5, head_length=1.0, fc='blue', ec='blue', alpha=0.7)

        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')

        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title(f"Trayectoria Keyframes - {os.path.basename(input_path)}")
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()

        plt.savefig(output_base + "_keyframes_only.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Gráficos REALES guardados en: {output_dir}")
        print(f"✓ La trayectoria REAL tiene {len(x_key)} puntos")
        print(f"✓ Distancia total REAL: {real_total_distance:.2f} m")

    # Añade este método para diagnosticar el problema de optimización
    def optimize_pose_graph(self):
        """Método de optimización con diagnóstico"""
        print("\n=== INICIANDO OPTIMIZACIÓN ===")
        
        if not hasattr(self, 'keyframe_poses') or len(self.keyframe_poses) < 2:
            print("⚠ No hay suficientes keyframes para optimizar")
            return self.keyframe_poses if hasattr(self, 'keyframe_poses') else []
        
        # Mostrar estadísticas antes de optimizar
        original_positions = np.array([pose[:3, 3] for pose in self.keyframe_poses])
        original_variance = np.var(original_positions, axis=0)
        print(f"Posiciones originales - Media: {np.mean(original_positions, axis=0)}")
        print(f"Posiciones originales - Varianza: {original_variance}")
        
        # Llamar a la optimización original del padre
        try:
            optimized_trajectory = super().optimize_pose_graph()
            print("✓ Optimización completada")
            
            # Mostrar estadísticas después de optimizar
            if optimized_trajectory and len(optimized_trajectory) > 0:
                opt_positions = np.array([pose[:3, 3] for pose in optimized_trajectory])
                opt_variance = np.var(opt_positions, axis=0)
                print(f"Posiciones optimizadas - Media: {np.mean(opt_positions, axis=0)}")
                print(f"Posiciones optimizadas - Varianza: {opt_variance}")
                
                # Verificar si la optimización colapsó
                if np.max(opt_variance) < 0.001:  # Varianza muy pequeña
                    print("⚠ ¡ADVERTENCIA! La optimización puede haber colapsado las poses")
                    print("⚠ Usando keyframes originales en lugar de optimizados")
                    return self.keyframe_poses
                
                return optimized_trajectory
                
        except Exception as e:
            print(f"✗ Error en optimización: {e}")
            print("⚠ Usando keyframes originales")
            return self.keyframe_poses
        
        return self.keyframe_poses

    def calculate_total_distance(self, trajectory):
        """Calcula la distancia total recorrida sumando segmentos"""
        if len(trajectory) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(trajectory)):
            if hasattr(trajectory[i], "shape") and trajectory[i].shape == (4,4):
                # Es una pose 4x4
                pos1 = trajectory[i-1][:3, 3]
                pos2 = trajectory[i][:3, 3]
            elif isinstance(trajectory[i], (list, np.ndarray)) and len(trajectory[i]) >= 2:
                # Es un array 2D o 3D
                pos1 = trajectory[i-1][:2]  # Solo X, Y
                pos2 = trajectory[i][:2]    # Solo X, Y
            else:
                continue
            
            total_distance += np.linalg.norm(pos2 - pos1)
        
        return total_distance

    def plot_comparative_trajectory(self, output_base, x_opt, y_opt, input_path):
        """Grafica trayectorias comparativas si hay datos históricos"""
        plt.figure(figsize=(14, 10))
        
        # Trayectoria optimizada (principal)
        plt.plot(x_opt, y_opt, 'b-', label='Trayectoria optimizada', linewidth=3, zorder=3)
        
        # Historial SLAM (si existe)
        if hasattr(self, 'slam_history') and len(self.slam_history) > 0:
            slam_traj = np.array(self.slam_history)
            if slam_traj.shape[1] >= 2:  # Tiene al menos X, Y
                x_slam, y_slam = slam_traj[:, 0], slam_traj[:, 1]
                x_slam -= x_slam[0]; y_slam -= y_slam[0]
                plt.plot(x_slam, y_slam, 'g--', label='SLAM puro', alpha=0.6, linewidth=2, zorder=2)
        
        # Historial GPS (si existe)
        if hasattr(self, 'gps_history') and len(self.gps_history) > 0:
            gps_traj = np.array(self.gps_history)
            if gps_traj.shape[1] >= 2:  # Tiene al menos X, Y
                x_gps, y_gps = gps_traj[:, 0], gps_traj[:, 1]
                x_gps -= x_gps[0]; y_gps -= y_gps[0]
                plt.plot(x_gps, y_gps, 'r--', label='GPS puro', alpha=0.6, linewidth=2, zorder=1)
        
        # Puntos de referencia
        plt.scatter(x_opt[0], y_opt[0], color='green', s=200, label='Inicio', zorder=5)
        plt.scatter(x_opt[-1], y_opt[-1], color='red', s=200, label='Fin', zorder=5)
        
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title(f"Comparación de Trayectorias - {os.path.basename(input_path)}")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        
        plt.savefig(output_base + "_comparative.png", dpi=300, bbox_inches='tight')
    plt.close()


def load_kitti_sequence(sequence_path, max_frames=None):
    frames = []
    gps_data = []
    
    image_dir = None
    gps_dir = None
    
    for root, dirs, files in os.walk(sequence_path):
        if "image_02" in dirs and "data" in os.listdir(os.path.join(root, "image_02")):
            image_dir = os.path.join(root, "image_02", "data")
        if "oxts" in dirs and "data" in os.listdir(os.path.join(root, "oxts")):
            gps_dir = os.path.join(root, "oxts", "data")
    
    if not image_dir or not gps_dir:
        print(f"✗ Estructura de directorio KITTI no encontrada en: {sequence_path}")
        print("  Buscando carpetas: image_02/data y oxts/data")
        return frames, gps_data
    
    print(f"✓ Directorio de imágenes: {image_dir}")
    print(f"✓ Directorio de GPS: {gps_dir}")
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    gps_files = sorted([f for f in os.listdir(gps_dir) if f.endswith('.txt')])
    
    if max_frames:
        image_files = image_files[:max_frames]
        gps_files = gps_files[:max_frames]
    
    min_files = min(len(image_files), len(gps_files))
    if min_files == 0:
        print("✗ No se encontraron archivos coincidentes")
        return frames, gps_data
    
    print(f"✓ Encontrados {min_files} pares de archivos (imágenes + GPS)")
    
    for i in range(min_files):
        img_file = image_files[i]
        gps_file = gps_files[i]
        
        frame_path = os.path.join(image_dir, img_file)
        gps_path = os.path.join(gps_dir, gps_file)
        
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"✗ Error cargando frame: {frame_path}")
            continue
        
        frames.append(frame)
        
        try:
            gps_point = np.loadtxt(gps_path)
            gps_data.append(gps_point)
        except Exception as e:
            print(f"✗ Error cargando GPS {gps_path}: {e}")
            gps_data.append(None)
    
    print(f"✓ Frames cargados: {len(frames)}")
    print(f"✓ Datos GPS cargados: {len([g for g in gps_data if g is not None])}")
    return frames, gps_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SLAM con RL y GPS para videos MP4 o secuencias KITTI')
    parser.add_argument('input_path', help='Ruta al video MP4 o directorio de secuencia KITTI')
    parser.add_argument('--kitti', action='store_true', help='Forzar modo KITTI')
    parser.add_argument('--max_frames', type=int, help='Número máximo de frames a procesar')
    
    args = parser.parse_args()
    
    input_path = args.input_path
    
    is_kitti = False
    if args.kitti:
        is_kitti = True
    elif os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            if "image_02" in dirs and "oxts" in dirs:
                is_kitti = True
                break
    
    if is_kitti:
        print("Modo: Directorio KITTI")
        slam = RL_ORB_SLAM_GPS()
        slam.process_kitti_sequence(input_path, args.max_frames)
        
    elif input_path.endswith('.mp4'):
        print("Modo: Video MP4")
        slam = RL_ORB_SLAM_GPS()
        slam.process_video_input(input_path)
        
    else:
        print("Error: Formato de entrada no reconocido")
        print("Use un directorio KITTI (con image_02 y oxts) o un archivo .mp4")
        sys.exit(1)
    
    print("Procesamiento completado")