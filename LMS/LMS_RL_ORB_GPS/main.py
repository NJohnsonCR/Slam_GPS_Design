import sys
import os
import cv2
import numpy as np
import torch

# Agrega la carpeta LMS_ORB_with_PG al sys.path
from LMS.LMS_ORB_with_PG.main import PoseGraphSLAM
from LMS.LMS_RL_ORB_GPS.rl_agent import SimpleRLAgent
from LMS.LMS_RL_ORB_GPS.gps_utils import latlon_to_utm

class RL_ORB_SLAM_GPS(PoseGraphSLAM):
    def __init__(self, fx=718.856, fy=718.856, cx=607.1928, cy=185.2157):
        super().__init__(fx, fy, cx, cy)
        self.agent = SimpleRLAgent()
        self.utm_reference = None

    def process_kitti_gps(self, oxts_data):
        lat, lon, alt = oxts_data[0], oxts_data[1], oxts_data[2]
        utm_coord = latlon_to_utm(lat, lon, alt)
        if self.utm_reference is None:
            self.utm_reference = utm_coord
        return utm_coord - self.utm_reference

    def process_frame_with_gps(self, frame, gps_utm, gps_confidence=0.9):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Siempre detectar características en cada frame
        keypoints, descriptors = self.orb_detector.detectAndCompute(gray, None)
        
        # DEBUG
        print(f"  Keypoints detectados: {len(keypoints) if keypoints else 0}")
        
        # Si es el primer frame, inicializar y salir
        if self.previous_keyframe_descriptors is None:
            self.previous_keyframe_image = gray
            self.previous_keyframe_keypoints = keypoints
            self.previous_keyframe_descriptors = descriptors
            self.previous_keyframe_pose = np.eye(4)
            self.keyframe_poses.append(np.eye(4))
            print("  ✓ Primer frame inicializado")
            return None
        
        # Buscar matches con el keyframe anterior
        matches = self.filter_matches_lowe_ratio(self.previous_keyframe_descriptors, descriptors)
        print(f"  Matches encontrados: {len(matches)}")
        
        # THRESHOLD MÁS BAJO para KITTI (porque los frames están más separados)
        if len(matches) > 15:  # ← Reducido de 30 a 15
            points_prev = np.float32([self.previous_keyframe_keypoints[m.queryIdx].pt for m in matches])
            points_curr = np.float32([keypoints[m.trainIdx].pt for m in matches])
            
            E, mask = cv2.findEssentialMat(points_prev, points_curr, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, points_prev, points_curr, self.camera_matrix)
                
                # Construir pose relativa
                relative_pose = np.eye(4)
                relative_pose[:3, :3] = R
                relative_pose[:3, 3] = t.ravel()
                
                # ACTUALIZAR: Usar siempre la pose anterior para integrar
                current_pose = self.previous_keyframe_pose @ relative_pose
                
                # Fusión con RL
                visual_confidence = min(len(matches) / 50.0, 1.0)  # ← Ajustado
                error = np.linalg.norm(current_pose[:3,3] - gps_utm)
                state = torch.tensor([visual_confidence, gps_confidence, error], dtype=torch.float32)
                weights = self.agent(state).detach()
                
                fused_position = weights[0] * current_pose[:3,3] + weights[1] * gps_utm
                fused_pose = np.eye(4)
                fused_pose[:3,3] = fused_position.numpy()
                fused_pose[:3,:3] = current_pose[:3,:3]
                
                # ACTUALIZAR KEYFRAME: Ser más conservador con KITTI
                translation_magnitude = np.linalg.norm(relative_pose[:3, 3])
                if translation_magnitude > 0.1:  # ← Umbral más bajo
                    self.keyframe_poses.append(fused_pose)
                    self.relative_transformations.append(relative_pose)
                    self.previous_keyframe_image = gray
                    self.previous_keyframe_keypoints = keypoints
                    self.previous_keyframe_descriptors = descriptors
                    self.previous_keyframe_pose = fused_pose
                    print(f"  ✓ Nuevo keyframe (movimiento: {translation_magnitude:.3f})")
                else:
                    print(f"  ✓ Pose estimada, pero no keyframe (movimiento pequeño)")
                
                return fused_pose
        
        print(f"  ✗ No suficientes matches o no se pudo calcular pose")
        return None

def load_kitti_sequence(sequence_path, max_frames=50):
    frames = []
    gps_data = []
    
    for i in range(max_frames):
        frame_path = os.path.join(sequence_path, "image_02", "data", f"{i:010d}.png")
        
        if not os.path.exists(frame_path):
            print(f"✗ No existe frame {i}: {frame_path}")
            continue
            
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"✗ Error cargando frame {i}")
            continue
        
        frames.append(frame)
        
        # Cargar GPS
        gps_path = os.path.join(sequence_path, "oxts", "data", f"{i:010d}.txt")
        if os.path.exists(gps_path):
            gps_data.append(np.loadtxt(gps_path))
        else:
            print(f"✗ No existe GPS {i}")
    
    print(f"✓ Frames cargados: {len(frames)}")
    print(f"✓ Datos GPS cargados: {len(gps_data)}")
    return frames, gps_data

if __name__ == "__main__":
    # CONFIGURAR ESTA RUTA
    sequence_path = "kitti_data/2011_09_26/2011_09_26_drive_0002_extract"
    
    print("Cargando datos KITTI...")
    try:
        frames, gps_data = load_kitti_sequence(sequence_path, max_frames=50)  # Aumentado a 50 frames
        print(f"✓ Se cargaron {len(frames)} frames y {len(gps_data)} datos GPS")
    except Exception as e:
        print(f"✗ Error cargando datos: {e}")
        print("¿La ruta del dataset es correcta?")
        sys.exit(1)
    
    print("Inicializando SLAM con RL...")
    slam = RL_ORB_SLAM_GPS()
    
    print("Procesando frames...")
    trajectories = []
    
    for i in range(min(50, len(frames))):  # Procesar hasta 50 frames
        print(f"Procesando frame {i}...")
        
        if frames[i] is None:
            print(f"  ✗ Frame {i} es None")
            continue
            
        try:
            gps_utm = slam.process_kitti_gps(gps_data[i])
            print(f"  ✓ GPS convertido: {gps_utm}")
            
            pose = slam.process_frame_with_gps(frames[i], gps_utm)
            if pose is not None:
                trajectory_point = pose[:3,3]  # Guardar solo posición (x, y, z)
                trajectories.append(trajectory_point)
                print(f"  ✓ Pose fusionada: {trajectory_point}")
            else:
                print(f"  ✗ No se pudo estimar pose (pocos matches?)")
                
        except Exception as e:
            print(f"  ✗ Error en frame {i}: {e}")
    
    # Guardar trayectoria en archivo
    if trajectories:
        output_path = "trayectoria_resultados.csv"
        np.savetxt(output_path, trajectories, delimiter=",", header="x,y,z", comments="")
        print(f"✓ Trayectoria guardada en: {output_path}")
        
        # Graficar resultados simples
        import matplotlib.pyplot as plt
        trajectories_array = np.array(trajectories)
        plt.figure(figsize=(10, 6))
        plt.plot(trajectories_array[:,0], trajectories_array[:,1], 'b-', label='Trayectoria SLAM+GPS')
        plt.scatter(trajectories_array[0,0], trajectories_array[0,1], color='g', label='Inicio')
        plt.scatter(trajectories_array[-1,0], trajectories_array[-1,1], color='r', label='Fin')
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Trayectoria SLAM con RL + GPS")
        plt.legend()
        plt.grid(True)
        plt.savefig("trayectoria_plot.png")
        print(f"✓ Gráfico guardado en: trayectoria_plot.png")
        
        # Mostrar estadísticas
        print(f"\n--- ESTADÍSTICAS ---")
        print(f"Total de poses estimadas: {len(trajectories)}")
        print(f"Distancia total recorrida: {np.sum(np.linalg.norm(np.diff(trajectories_array, axis=0), axis=1)):.2f} m")
        
    else:
        print("✗ No se generó ninguna trayectoria")
    