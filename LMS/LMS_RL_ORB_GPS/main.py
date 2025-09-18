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
        self.previous_gps_utm = None # Nuevo atributo

    def process_kitti_gps(self, oxts_data):
        lat, lon, alt = oxts_data[0], oxts_data[1], oxts_data[2] # Lat, Lon, Alt como primeros tres valores de archivos oxts
        utm_coord = latlon_to_utm(lat, lon, alt) # Conversion de coordenadas de (lat/lon) a UTM (metros)
        if self.utm_reference is None:
            self.utm_reference = utm_coord # se guarda siempre el ultimo dato
        return utm_coord - self.utm_reference
    
    def process_frame_with_gps(self, frame, gps_utm, gps_confidence=0.9):
        # 1. CONVERSIÓN A GRISES Y DETECCIÓN DE FEATURES
        minimumMatches = 15  # Mínimo de matches para considerar válido
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb_detector.detectAndCompute(gray, None)
        print(f"  Keypoints detectados: {len(keypoints) if keypoints else 0}")

        # 2. INICIALIZACIÓN (primer frame)
        if self.previous_keyframe_descriptors is None:
            self.previous_keyframe_image = gray
            self.previous_keyframe_keypoints = keypoints
            self.previous_keyframe_descriptors = descriptors
            self.previous_keyframe_pose = np.eye(4)
            self.keyframe_poses.append(np.eye(4))
            print("  ✓ Primer frame inicializado")
            return None

        # Cantidad de matches entre el frame actual y el anterior
        matches = self.filter_matches_lowe_ratio(self.previous_keyframe_descriptors, descriptors)
        print(f"  Matches encontrados: {len(matches)}")

        # Se cumple condición mínima de matches
        if len(matches) > minimumMatches:

            # Coordenadas 2D de los keypoints en la imagen anterior y actual respectivamente
            points_prev = np.float32([self.previous_keyframe_keypoints[m.queryIdx].pt for m in matches])
            points_curr = np.float32([keypoints[m.trainIdx].pt for m in matches])

            # La matriz representa la relación geométrica entre dos vistas
            E, mask = cv2.findEssentialMat(points_prev, points_curr, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            # DEBUG: Verificar matriz esencial
            print(f"  E matrix shape: {E.shape if E is not None else 'None'}")
            
            if E is not None:
                try:
                    # R: rotación de la camara entre frames, t: traslación entre dos vistas, mask: inliers (matches coherentes)
                    points, R, t, mask = cv2.recoverPose(E, points_prev, points_curr, self.camera_matrix)
                    
                    # DEBUG: Verificar resultados de recoverPose
                    print(f"  recoverPose success: t={t.ravel()} | Magnitude: {np.linalg.norm(t):.6f}")
                    print(f"  R det: {np.linalg.det(R):.6f}")
                    print(f"  inliers: {np.sum(mask) if mask is not None else 'N/A'}")
                    
                    # Matriz de tranformacion 4x4 homgogenea
                    relative_pose = np.eye(4)
                    relative_pose[:3, :3] = R      # Parte de rotación
                    relative_pose[:3, 3] = t.ravel()  # Parte de traslación

                    # 7. ESCALADO CON GPS (CONVERSIÓN A METROS REALES)
                    escala = 1.0  # Valor por defecto
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
                    
                    # Pose real absoluta del frame actual   
                    current_pose = self.previous_keyframe_pose @ relative_pose

                    # 9. FUSIÓN CON RL (TU APORTE CLAVE)
                    visual_confidence = min(len(matches) / 50.0, 1.0)
                    error = np.linalg.norm(current_pose[:3, 3] - gps_utm)

                    # ✅ PRINT 2: VERIFICAR POSE vs GPS (antes de RL)
                    print(f"  SLAM pose: {current_pose[:3,3].round(3)} | GPS: {gps_utm.round(3)} | Error: {error:.3f} m")

                    # vector estado para funcion RL: [confianza_visual, confianza_gps, error_entre_poses]
                    state = torch.tensor([visual_confidence, gps_confidence, error], dtype=torch.float32)
                    weights = self.agent(state).detach()

                    # ✅ PRINT 3: VERIFICAR PESOS RL (después de decisión)
                    print(f"  RL weights: SLAM={weights[0]:.2f}, GPS={weights[1]:.2f}")
                    
                    # 10. COMBINAR SLAM Y GPS SEGÚN PESOS DEL RL
                    fused_position = weights[0] * current_pose[:3, 3] + weights[1] * gps_utm
                    fused_pose = np.eye(4)
                    fused_pose[:3, 3] = fused_position.numpy()  # Posición fusionada
                    fused_pose[:3, :3] = current_pose[:3, :3]   # Rotación del SLAM

                    # 11. ACTUALIZAR KEYFRAME SI HAY MOVIMIENTO SIGNIFICATIVO
                    translation_magnitude = np.linalg.norm(relative_pose[:3, 3])
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

                    # 12. GUARDAR GPS PARA PRÓXIMO FRAME
                    self.previous_gps_utm = gps_utm.copy()
                    
                    return fused_pose
                    
                except Exception as e:
                    print(f"  ✗ Error en recoverPose: {e}")
                    return None
            else:
                print("  ✗ No se pudo calcular matriz esencial")
                return None

        print(f"  ✗ No suficientes matches o no se pudo calcular pose")
        return None
    
   
#Todo: Poner esta funcion en un archivo utils.py
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
        frames, gps_data = load_kitti_sequence(sequence_path, max_frames= 50)  # Aumentado a 83 frames
        print(f"✓ Se cargaron {len(frames)} frames y {len(gps_data)} datos GPS")
    except Exception as e:
        print(f"✗ Error cargando datos: {e}")
        print("¿La ruta del dataset es correcta?")
        sys.exit(1)
    
    print("Inicializando SLAM con RL...")
    slam = RL_ORB_SLAM_GPS()
    
    print("Procesando frames...")
    trajectories = []
    
    for i in range(min(50, len(frames))):  # Procesar hasta 83 frames
        print(f"Procesando frame {i}...")
        
        if frames[i] is None:
            print(f"  ✗ Frame {i} es None")
            continue
            
        try:
            gps_utm = slam.process_kitti_gps(gps_data[i])
            print(f"  ✓ GPS convertido a UTM: {gps_utm}")
            
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
    