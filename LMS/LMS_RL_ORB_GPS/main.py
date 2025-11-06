import sys
import os
import cv2
import numpy as np
import torch
import argparse
import csv
import json
from datetime import datetime
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd

# Agrega el directorio raíz al sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

# Imports relativos al proyecto
from LMS.LMS_ORB_with_PG.main import PoseGraphSLAM
from model.rl_agent import SimpleRLAgent, RLTrainer
from utils.gps_utils import *
from utils.gps_filter import *
from utils.training_augmentation import TrainingAugmentation
from utils.mobile_video_processor import MobileVideoProcessor
from utils.trajectory_metrics import TrajectoryMetrics

class RL_ORB_SLAM_GPS(PoseGraphSLAM):
    def __init__(self, fx=718.856, fy=718.856, cx=607.1928, cy=185.2157, 
                 training_mode=False, model_path=None, simulate_mobile_gps=False, gps_noise_std=5.0,
                 enable_optimization=False):
        super().__init__(fx, fy, cx, cy)
        self.agent = SimpleRLAgent()
        self.trainer = RLTrainer(self.agent)
        self.training_mode = training_mode
        self.simulate_mobile_gps = simulate_mobile_gps
        self.enable_optimization = enable_optimization
        
        if model_path and os.path.exists(model_path):
            self.trainer.load_model(model_path)
        elif not training_mode:
            default_model = "LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth"
            if os.path.exists(default_model):
                print(f"Cargando modelo pre-entrenado: {default_model}")
                self.trainer.load_model(default_model)
            else:
                print("ADVERTENCIA: No se encontró modelo pre-entrenado. Usando pesos aleatorios.")
        
        self.utm_reference = None
        self.previous_gps_utm = None
        self.gps_available = False
        self.gps_history = []
        self.slam_history = []
        self.gps_keyframes = []  # GPS correspondiente a cada keyframe
        
        self.current_state = None
        self.frame_count = 0
        self.training_step_interval = 10

        self.gps_filter = GPSFilter(
            window_size=5, 
            movement_variance_threshold=0.1,
            add_noise=simulate_mobile_gps,
            noise_std=gps_noise_std
        )
        
        if simulate_mobile_gps:
            print(f"ADVERTENCIA: MODO GPS MÓVIL SIMULADO: Agregando ruido gaussiano (sigma={gps_noise_std}m)")
        
        if enable_optimization:
            print(f"Optimización del grafo: ACTIVADA (puede ser lento con muchos frames)")
        else:
            print(f"Optimización del grafo: DESACTIVADA (modo rápido, usa keyframes fusionados)")

    def gps_frame_reference(self, gps_frame_value):
        lat, lon, alt = gps_frame_value[0], gps_frame_value[1], gps_frame_value[2]
        utm_coord = latlon_to_utm(lat, lon, alt)
        if self.utm_reference is None:
            self.utm_reference = utm_coord
        return utm_coord - self.utm_reference
    
    def calculate_base_weights(self, gps_conf, visual_conf, error):
        conf_ratio = visual_conf / (gps_conf + 1e-6)
        
        gps_weight_from_conf = gps_conf
        slam_weight_from_conf = visual_conf
        
        if conf_ratio > 1.15 and visual_conf > 0.75:
            slam_boost = 0.15
            gps_reduction = 0.10
            print(f"  [BALANCE] SLAM más confiable (ratio={conf_ratio:.2f}) → Boost SLAM")
        elif conf_ratio < 0.85 and gps_conf > 0.75:
            slam_boost = -0.10
            gps_reduction = -0.15
            print(f"  [BALANCE] GPS más confiable (ratio={conf_ratio:.2f}) → Boost GPS")
        elif visual_conf > 0.75 and gps_conf > 0.75 and 0.85 <= conf_ratio <= 1.15:
            slam_boost = 0.0
            gps_reduction = 0.0
            print(f"  [BALANCE] Ambos muy confiables (ratio={conf_ratio:.2f}) → Balance natural 50-50")
        else:
            slam_boost = 0.0
            gps_reduction = 0.0
        
        slam_weight_from_conf += slam_boost
        gps_weight_from_conf += gps_reduction
        
        error_penalty_gps = 0.0
        if error > 5.0:
            error_penalty_gps = 0.3
            print(f"  [ERROR] Alto error ({error:.2f}m) → Penaliza GPS -0.30")
        elif error > 2.0:
            error_penalty_gps = 0.1 * ((error - 2.0) / 3.0)
            print(f"  [ERROR] Error moderado ({error:.2f}m) → Penaliza GPS -{error_penalty_gps:.2f}")
        
        error_bonus_gps = 0.0
        if error < 1.0 and gps_conf > 0.6:
            error_bonus_gps = 0.15
            print(f"  [ALIGN] Excelente alineación ({error:.2f}m) → Bonus GPS +0.15")
        elif error < 2.0 and gps_conf > 0.6:
            error_bonus_gps = 0.1 * (2.0 - error)
            print(f"  [ALIGN] Buena alineación ({error:.2f}m) → Bonus GPS +{error_bonus_gps:.2f}")
        
        w_gps_base = gps_weight_from_conf - error_penalty_gps + error_bonus_gps
        w_slam_base = slam_weight_from_conf
        
        total = w_slam_base + w_gps_base + 1e-6
        w_slam_base = np.clip(w_slam_base / total, 0.05, 0.95)
        w_gps_base = np.clip(w_gps_base / total, 0.05, 0.95)
        
        total = w_slam_base + w_gps_base
        w_slam_base /= total
        w_gps_base /= total
        
        return w_slam_base, w_gps_base
    
    def calculate_rl_adjustment_margin(self, visual_conf, gps_conf, error):
        conf_diff = abs(visual_conf - gps_conf)
        avg_conf = (visual_conf + gps_conf) / 2.0
        min_conf = min(visual_conf, gps_conf)
        max_conf = max(visual_conf, gps_conf)
        
        if min_conf < 0.3 and max_conf > 0.7 and conf_diff > 0.6:
            rl_margin = 0.50
            scenario = "SENSOR_EXTREMO"
            print(f"  [MARGEN] Caso EXTREMO detectado (min={min_conf:.2f}, max={max_conf:.2f}, diff={conf_diff:.2f}) → Margen amplio (±50%)")
            return rl_margin, scenario
        
        if avg_conf > 0.75 and conf_diff < 0.15 and error < 3.0:
            rl_margin = 0.10
            scenario = "ALTA_CERTEZA"
        
        elif conf_diff > 0.30:
            rl_margin = 0.25
            scenario = "SENSOR_DOMINANTE"
        
        elif avg_conf > 0.60 and error > 3.0:
            rl_margin = 0.30
            scenario = "ERROR_ALTO"
        
        elif avg_conf < 0.50:
            rl_margin = 0.40
            scenario = "INCERTIDUMBRE"
        
        else:
            rl_margin = 0.20
            scenario = "INTERMEDIO"
        
        return rl_margin, scenario
    
    def process_frame_with_gps(self, frame, gps_utm, gps_confidence):
        minimumMatches = 15
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb_detector.detectAndCompute(gray, None)
        print(f"  Keypoints detectados: {len(keypoints) if keypoints else 0}")

        if self.previous_keyframe_descriptors is None:
            self.previous_keyframe_image = gray
            self.previous_keyframe_keypoints = keypoints
            self.previous_keyframe_descriptors = descriptors

            initial_pose = np.eye(4)
            if gps_utm is not None:
                initial_pose[:3, 3] = gps_utm
                print(f"Inicializado con GPS: {gps_utm.round(3)}")
            else:
                initial_pose[:3, 3] = np.zeros(3)
                print("Inicializado en origen (sin GPS)")

            self.previous_keyframe_pose = initial_pose
            self.keyframe_poses.append(initial_pose)
            
            # Guardar GPS del primer keyframe
            if gps_utm is not None:
                self.gps_keyframes.append(gps_utm.copy())
            else:
                self.gps_keyframes.append(np.zeros(3))

            if gps_utm is not None:
                self.previous_gps_utm = gps_utm.copy()

            print("Primer frame inicializado")
            return None

        matches = self.filter_matches_lowe_ratio(self.previous_keyframe_descriptors, descriptors)
        print(f"  Matches encontrados: {len(matches)}")

        if len(matches) > minimumMatches:
            points_prev = np.float32([self.previous_keyframe_keypoints[m.queryIdx].pt for m in matches])
            points_curr = np.float32([keypoints[m.trainIdx].pt for m in matches])

            E, mask = cv2.findEssentialMat(points_prev, points_curr, self.camera_matrix,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)

            if E is not None:
                try:
                    points, R, t, mask = cv2.recoverPose(E, points_prev, points_curr, self.camera_matrix)
                    print(f"  Vector t crudo: {t.ravel().round(3)} | Magnitud: {np.linalg.norm(t):.6f}")

                    if self.previous_gps_utm is not None and gps_utm is not None:
                        movimiento_gps = gps_utm - self.previous_gps_utm
                        movimiento_gps_2d = movimiento_gps[:2]

                        if np.linalg.norm(movimiento_gps_2d) > 0.1 and np.linalg.norm(t[:2]) > 0.0001:
                            direccion_slam = t[:2].ravel() / np.linalg.norm(t[:2])
                            direccion_gps = movimiento_gps_2d / np.linalg.norm(movimiento_gps_2d)

                            angle_slam = np.arctan2(direccion_slam[1], direccion_slam[0])
                            angle_gps = np.arctan2(direccion_gps[1], direccion_gps[0])
                            delta_angle = angle_gps - angle_slam

                            R_align = np.array([
                                [np.cos(delta_angle), -np.sin(delta_angle), 0],
                                [np.sin(delta_angle),  np.cos(delta_angle), 0],
                                [0, 0, 1]
                            ])
                            t[:3] = R_align @ t[:3]
                            R = R_align @ R

                            print(f"Dirección SLAM alineada con GPS (delta_angle={np.degrees(delta_angle):.2f}°)")
                            print(f"  Vector t corregido: {t.ravel().round(3)} | Magnitud: {np.linalg.norm(t):.6f}")

                    escala = 1.0
                    if self.previous_gps_utm is not None and gps_utm is not None:
                        distancia_real = np.linalg.norm(gps_utm - self.previous_gps_utm)
                        distancia_slam = np.linalg.norm(t)
                        if distancia_slam > 0.0001 and distancia_real > 0.01:
                            escala = distancia_real / distancia_slam
                            t[:3] *= escala
                            print(f"  OK - Escalado aplicado: {distancia_slam:.6f} → {distancia_real:.6f} m (escala={escala:.3f})")

                    relative_pose = np.eye(4)
                    relative_pose[:3, :3] = R
                    relative_pose[:3, 3] = t.ravel()

                    current_pose = self.previous_keyframe_pose @ relative_pose

                    visual_confidence = min(len(matches) / 50.0, 1.0)
                    error = np.linalg.norm(current_pose[:3, 3] - gps_utm) if gps_utm is not None else 0.0
                    
                    print(f"  [CONFIANZAS] Visual={visual_confidence:.3f} (matches={len(matches)}), GPS={gps_confidence:.3f}, Error={error:.3f}m")
                    
                    fused_position = current_pose[:3, 3]
                    if gps_utm is not None:
                        w_slam_base, w_gps_base = self.calculate_base_weights(gps_confidence, visual_confidence, error)
                        
                        rl_margin, scenario = self.calculate_rl_adjustment_margin(visual_confidence, gps_confidence, error)
                        
                        state = torch.tensor([visual_confidence, gps_confidence, error], dtype=torch.float32)
                        
                        with torch.no_grad():
                            rl_weights = self.agent(state)
                            
                            rl_adjustment_slam = (rl_weights[0].item() - 0.5) * (rl_margin * 2)
                            rl_adjustment_gps = (rl_weights[1].item() - 0.5) * (rl_margin * 2)
                        
                        w_slam_final = np.clip(w_slam_base + rl_adjustment_slam, 0.05, 0.95)
                        w_gps_final = np.clip(w_gps_base + rl_adjustment_gps, 0.05, 0.95)
                        
                        total = w_slam_final + w_gps_final
                        w_slam_final /= total
                        w_gps_final /= total
                        
                        fused_position = w_slam_final * current_pose[:3, 3] + w_gps_final * gps_utm
                        
                        print(f"  [HYBRID] Base: SLAM={w_slam_base:.3f}, GPS={w_gps_base:.3f}")
                        print(f"  [RL MARGIN] Escenario={scenario}, Margen=±{rl_margin*100:.0f}%")
                        print(f"  [RL ADJ] Δ_SLAM={rl_adjustment_slam:+.3f}, Δ_GPS={rl_adjustment_gps:+.3f}")
                        print(f"  [FINAL]  SLAM={w_slam_final:.3f}, GPS={w_gps_final:.3f}, "
                              f"GPS_Conf={gps_confidence:.3f}, Error={error:.3f}m")
                        
                        if self.training_mode:
                            weights_trainable = self.agent(state)
                            
                            reward = self.trainer.calculate_reward(
                                state=state,
                                action_weights=weights_trainable,
                                slam_position=current_pose[:3, 3],
                                gps_position=gps_utm,
                                ground_truth=None
                            )
                            
                            self.trainer.store_experience(state, weights_trainable.detach(), reward, None, done=False)
                            
                            self.frame_count += 1
                            if self.frame_count % self.training_step_interval == 0:
                                train_result = self.trainer.train_step(batch_size=32)
                                if train_result:
                                    print(f"  [TRAINING] Loss: {train_result['loss']:.4f}, "
                                          f"Avg Reward: {train_result['avg_reward']:.4f}")
                            
                            print(f"  [RL TRAINING] Reward={reward:.3f}")
                    else:
                        # NO HAY GPS - Usar 100% SLAM
                        print(f"  [FINAL]  SLAM=1.000, GPS=0.000 (Sin GPS disponible)")

                    fused_pose = np.eye(4)
                    fused_pose[:3, 3] = fused_position
                    fused_pose[:3, :3] = current_pose[:3, :3]

                    self.slam_history.append(current_pose[:3, 3].copy())
                    if gps_utm is not None:
                        self.gps_history.append(gps_utm.copy())

                    translation_magnitude = np.linalg.norm(relative_pose[:3, 3])
                    if translation_magnitude > 0.1:
                        self.keyframe_poses.append(fused_pose)
                        self.relative_transformations.append(relative_pose)
                        
                        # Guardar GPS correspondiente a este keyframe
                        if gps_utm is not None:
                            self.gps_keyframes.append(gps_utm.copy())
                        else:
                            # Si no hay GPS, usar la última posición GPS conocida o ceros
                            if self.previous_gps_utm is not None:
                                self.gps_keyframes.append(self.previous_gps_utm.copy())
                            else:
                                self.gps_keyframes.append(np.zeros(3))
                        
                        self.previous_keyframe_image = gray
                        self.previous_keyframe_keypoints = keypoints
                        self.previous_keyframe_descriptors = descriptors
                        self.previous_keyframe_pose = fused_pose
                        print(f"Nuevo keyframe (movimiento={translation_magnitude:.3f} m)")
                    else:
                        print(f"Pose estimada, pero no keyframe (movimiento pequeño)")

                    if gps_utm is not None:
                        self.previous_gps_utm = gps_utm.copy()
                        self.gps_available = True

                    return fused_pose

                except Exception as e:
                    print(f"Error en recoverPose: {e}")
                    return None
            else:
                print("No se pudo calcular matriz esencial")
                return None

        print(f"No suficientes matches o no se pudo calcular pose")
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
    
    def process_kitti_sequence(self, sequence_path, max_frames=None, augmentation_prob=0.0):
        frames, gps_data = load_kitti_sequence(sequence_path, max_frames)
        
        print(f"\n{'='*60}")
        print(f"MODO: {'ENTRENAMIENTO' if self.training_mode else 'INFERENCIA'}")
        print(f"Total de frames: {len(frames)}")
        if self.training_mode and augmentation_prob > 0:
            print(f"AUMENTACIÓN ACTIVA: {augmentation_prob*100:.0f}% de probabilidad")
        print(f"{'='*60}\n")
        
        augmenter = None
        if self.training_mode and augmentation_prob > 0:
            augmenter = TrainingAugmentation(corruption_probability=augmentation_prob)
            print("Módulo de aumentación inicializado")
        
        for i in range(len(frames)):
            print(f"Procesando frame {i}...")
            
            if frames[i] is None:
                print(f"  ERROR - Frame {i} es None")
                continue
                
            try:
                gps_utm = None
                gps_confidence = 0.1

                if gps_data[i] is not None:
                    gps_utm = self.gps_frame_reference(gps_data[i])
                    print(f"  GPS UTM: {gps_utm.round(3)}")

                    self.gps_filter.add_measurement(gps_utm)
                    gps_confidence_raw = self.gps_filter.calculate_confidence()
                    
                    if augmenter is not None and np.random.rand() < augmentation_prob:
                        if hasattr(self, 'keyframe_poses') and len(self.keyframe_poses) > 0:
                            last_pose = self.keyframe_poses[-1][:3, 3]
                            error_approx = np.linalg.norm(last_pose - gps_utm) if gps_utm is not None else 0.0
                        else:
                            error_approx = 0.0
                        
                        gps_utm_aug, gps_conf_aug, visual_conf_aug, scenario = augmenter.augment_training_sample(
                            gps_utm=gps_utm,
                            gps_conf=gps_confidence_raw,
                            visual_conf=0.5,
                            error=error_approx
                        )
                        
                        gps_utm = gps_utm_aug
                        gps_confidence = gps_conf_aug
                        
                        print(f"[AUGMENT] Escenario aplicado: {scenario}")
                    else:
                        gps_confidence = gps_confidence_raw

                    self.gps_filter.print_debug_info()
                
                self.process_frame_with_gps(frames[i], gps_utm, gps_confidence)
                    
            except Exception as e:
                print(f"  ERROR - Error en frame {i}: {e}")
        
        if augmenter is not None:
            augmenter.print_statistics()
        
        if self.training_mode:
            model_save_path = "LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth"
            metadata = {
                'sequence': sequence_path,
                'total_frames': len(frames),
                'total_experiences': len(self.trainer.replay_buffer),
                'training_steps': len(self.trainer.training_metrics['losses']),
                'augmentation_prob': augmentation_prob if augmenter else 0.0
            }
            self.trainer.save_model(model_save_path, metadata)
            self.trainer.print_training_summary()
        
        optimized_trajectory = self.optimize_pose_graph()
        self.save_trajectory_outputs(optimized_trajectory, sequence_path)

    def process_mobile_sequence(self, video_path, location_csv, frame_timestamps_file, max_frames=None):
        print("\n" + "="*60)
        print("PROCESANDO VIDEO MÓVIL")
        print("="*60)
        print(f"Video: {video_path}")
        print(f"GPS: {location_csv}")
        print(f"Timestamps: {frame_timestamps_file}")
        if max_frames:
            print(f"Límite de frames: {max_frames}")
        print("="*60 + "\n")
        
        processor = MobileVideoProcessor(
            slam_system=self,
            video_path=video_path,
            gps_csv_path=location_csv,
            timestamps_path=frame_timestamps_file,
            max_frames=max_frames
        )
        
        processor.process()

    def save_trajectory_outputs(self, trajectory, input_path):
        tipo_lms = "LMS_RL_ORB_GPS"
        timestamp = datetime.now().strftime("%H%M_%d%m_%Y")
        
        # Obtener la raíz del proyecto (donde está la carpeta resultados/)
        # Este archivo está en: Slam_GPS_Design/LMS/LMS_RL_ORB_GPS/main.py
        # Necesitamos llegar a: Slam_GPS_Design/
        project_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__),  # LMS/LMS_RL_ORB_GPS/
            '..',                        # LMS/
            '..'                         # Slam_GPS_Design/
        ))
        
        # Usar la misma estructura que los otros sistemas LMS
        output_dir = os.path.join(project_root, "resultados", tipo_lms, timestamp)
        
        os.makedirs(output_dir, exist_ok=True)
        output_base = os.path.join(output_dir, f"trayectoria_{tipo_lms}")

        # ====================================================================
        # NUEVO: DETECCION AUTOMATICA DE CALIDAD DE GPS
        # ====================================================================
        print("\n" + "="*80)
        print("ANALIZANDO CALIDAD DEL GPS...")
        print("="*80)
        
        gps_quality_result = self.gps_filter.detect_gps_quality(min_samples=20)
        
        if gps_quality_result['detected']:
            # Imprimir reporte detallado
            self.gps_filter.print_quality_detection_report()
            
            # Guardar reporte en JSON
            quality_report_file = output_base + "_gps_quality_report.json"
            with open(quality_report_file, 'w') as f:
                json.dump(gps_quality_result, f, indent=4)
            print(f"Reporte de calidad GPS guardado en: {quality_report_file}")
        else:
            print(f"No se pudo detectar calidad de GPS: {gps_quality_result.get('reason', 'Razon desconocida')}")
        
        print("="*80 + "\n")
        # ====================================================================

        print(f"=== DEBUG: Trayectoria optimizada ===")
        print(f"Tipo: {type(trajectory)}")
        print(f"Longitud: {len(trajectory) if hasattr(trajectory, '__len__') else 'N/A'}")
        
        if len(trajectory) > 0:
            print(f"Primer elemento: {trajectory[0]}")
            if hasattr(trajectory[0], 'shape'):
                print(f"Forma primer elemento: {trajectory[0].shape}")

        try:
            if hasattr(self, 'keyframe_poses') and len(self.keyframe_poses) > 0:
                keyframe_positions = np.array([pose[:3, 3] for pose in self.keyframe_poses])
                print(f"Keyframes sin optimizar: {len(keyframe_positions)} poses")
                
                x_key, y_key = keyframe_positions[:, 0], keyframe_positions[:, 1]
                x_key -= x_key[0]
                y_key -= y_key[0]
                
                real_total_distance = 0.0
                for i in range(1, len(keyframe_positions)):
                    pos1 = keyframe_positions[i-1][:2]
                    pos2 = keyframe_positions[i][:2]
                    real_total_distance += np.linalg.norm(pos2 - pos1)
            else:
                print("No hay keyframes disponibles")
                return
                
        except Exception as e:
            print(f"Error al extraer keyframes: {e}")
            return

        stats = {
            "keyframes": len(self.keyframe_poses),
            "dist_total": real_total_distance,
            "matches_prom": self.total_tracked_matches / max(1, self.total_pose_estimations),
            "exito_triang": self.total_successful_frames / max(1, self.total_pose_estimations),
            "mov_medio": self.total_translation_magnitude / max(1, self.total_pose_estimations)
        }

        print("\n=== Estadísticas REALES (sin optimización) ===")
        for k, v in stats.items():
            print(f"{k}: {v}")
        print("=============================================\n")

        # Guardar CSV de trayectoria fusionada
        with open(output_base + "_reales.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["X", "Y"])
            for i in range(len(x_key)):
                writer.writerow([x_key[i], y_key[i]])

        # ====================================================================
        # COMPARACIÓN COMPLETA: Pipeline vs SLAM puro vs GPS (ground truth)
        # ====================================================================
        
        has_slam_pure = hasattr(self, 'slam_history') and len(self.slam_history) > 0
        has_gps = hasattr(self, 'gps_history') and len(self.gps_history) > 0
        has_gps_keyframes = hasattr(self, 'gps_keyframes') and len(self.gps_keyframes) > 0
        
        if has_slam_pure and has_gps and has_gps_keyframes:
            print("\n" + "="*80)
            print("COMPARACIÓN COMPLETA DE MÉTODOS")
            print("="*80)
            print(f"Pipeline Fusionado (SLAM+GPS+RL): {len(self.keyframe_poses)} keyframes")
            print(f"SLAM puro: {len(self.slam_history)} frames")
            print(f"GPS puro: {len(self.gps_history)} frames")
            print(f"GPS Ground Truth: {len(self.gps_keyframes)} keyframes")
            print("="*80 + "\n")
            
            # Método 1: Comparar usando GPS keyframes como ground truth
            print("MÉTODO 1: Comparación con GPS keyframes como ground truth")
            print("-" * 80)
            
            # Preparar trayectorias para comparación
            pipeline_traj = np.array([pose[:3, 3] for pose in self.keyframe_poses])
            slam_pure_traj = np.array(self.slam_history)
            gps_traj = np.array(self.gps_history)
            gps_keyframes_traj = np.array(self.gps_keyframes)
            
            # Método 1: Comparar usando GPS keyframes como ground truth
            print("MÉTODO 1: Comparación con GPS keyframes como ground truth")
            print("-" * 80)
            
            # Necesitamos interpolar las trayectorias SLAM puro y GPS puro
            # para tener la misma cantidad de puntos que los keyframes
            n_keyframes = len(pipeline_traj)
            
            # Interpolar SLAM puro en las posiciones de los keyframes
            if len(slam_pure_traj) > n_keyframes:
                # Submuestrear SLAM puro proporcionalmente
                indices = np.linspace(0, len(slam_pure_traj) - 1, n_keyframes).astype(int)
                slam_pure_keyframes = slam_pure_traj[indices]
            else:
                slam_pure_keyframes = slam_pure_traj
            
            # Asegurar longitudes iguales
            min_len = min(len(pipeline_traj), len(slam_pure_keyframes), len(gps_keyframes_traj))
            pipeline_traj_cut = pipeline_traj[:min_len]
            slam_pure_cut = slam_pure_keyframes[:min_len]
            gps_gt_cut = gps_keyframes_traj[:min_len]
            
            # Realizar comparación usando GPS como ground truth
            comparison_results = TrajectoryMetrics.compare_trajectories(
                pipeline_traj=pipeline_traj_cut,
                slam_traj=slam_pure_cut,
                gps_traj=gps_gt_cut,
                reference='gps'
            )
            
            # Imprimir tabla comparativa
            TrajectoryMetrics.print_comparison_table(comparison_results)
            
            # Guardar métricas en CSV
            comparison_csv = output_base + "_comparison_metrics.csv"
            TrajectoryMetrics.save_metrics_to_csv(comparison_results, comparison_csv)
            
            # Guardar en JSON también
            comparison_json = output_base + "_comparison_metrics.json"
            with open(comparison_json, 'w') as f:
                # Convertir arrays numpy a listas para JSON
                json_data = {}
                for key, data in comparison_results.items():
                    json_data[key] = {
                        'method': data['method'],
                        'ate': {k: float(v) if isinstance(v, (np.floating, np.integer)) else (v.tolist() if isinstance(v, np.ndarray) else v)
                               for k, v in data['ate'].items()},
                        'rpe': {k: float(v) if isinstance(v, (np.floating, np.integer)) else (v.tolist() if isinstance(v, np.ndarray) else v)
                               for k, v in data['rpe'].items()}
                    }
                json.dump(json_data, f, indent=4)
            print(f"Comparación guardada en JSON: {comparison_json}\n")
            
            # Generar gráfico comparativo de errores ATE
            self._plot_ate_comparison(comparison_results, output_base + "_ate_comparison.png")
            
            # Generar gráfico comparativo de errores RPE
            self._plot_rpe_comparison(comparison_results, output_base + "_rpe_comparison.png")
            
            print("="*80 + "\n")

        # Gráfico principal con todas las trayectorias
        plt.figure(figsize=(16, 12))
        
        # Pipeline fusionado (keyframes) - línea principal
        plt.plot(x_key, y_key, 'b-', label='Pipeline Fusionado (SLAM+GPS+RL)', 
                linewidth=4, marker='o', markersize=8, zorder=5)
        
        # SLAM puro por frame
        if has_slam_pure:
            slam_traj = np.array(self.slam_history)
            if slam_traj.shape[1] >= 2:
                x_slam, y_slam = slam_traj[:, 0], slam_traj[:, 1]
                x_slam -= x_slam[0]; y_slam -= y_slam[0]
                plt.plot(x_slam, y_slam, 'g--', label='SLAM Puro', 
                        linewidth=2.5, zorder=3, alpha=0.8)
        
        # GPS puro
        if has_gps:
            gps_traj = np.array(self.gps_history)
            if gps_traj.shape[1] >= 2:
                x_gps, y_gps = gps_traj[:, 0], gps_traj[:, 1]
                x_gps -= x_gps[0]; y_gps -= y_gps[0]
                plt.plot(x_gps, y_gps, 'r-.', label='GPS Puro (Ground Truth)', 
                        linewidth=2.5, zorder=4, alpha=0.9)

        # Trayectoria optimizada (si está disponible y es diferente)
        try:
            if all(hasattr(pose, 'shape') and pose.shape == (4, 4) for pose in trajectory):
                opt_positions = np.array([pose[:3, 3] for pose in trajectory])
                x_opt, y_opt = opt_positions[:, 0], opt_positions[:, 1]
                x_opt -= x_opt[0]; y_opt -= y_opt[0]
                
                # Solo graficar si es significativamente diferente
                if np.max(np.abs(x_opt - x_key)) > 0.1 or np.max(np.abs(y_opt - y_key)) > 0.1:
                    plt.plot(x_opt, y_opt, 'm:', label='Trayectoria Optimizada', 
                            linewidth=1.5, zorder=1, alpha=0.5)
        except:
            pass

        plt.scatter(x_key[0], y_key[0], color='green', s=300, marker='s', 
                   label='Inicio', zorder=10, edgecolors='black', linewidths=2)
        plt.scatter(x_key[-1], y_key[-1], color='red', s=300, marker='X', 
                   label='Fin', zorder=10, edgecolors='black', linewidths=2)
        
        # Anotaciones de keyframes
        for i in range(len(x_key)):
            plt.annotate(str(i), (x_key[i], y_key[i]), xytext=(8, 8), 
                        textcoords='offset points', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # Información de estadísticas
        info_text = (
            f"Keyframes: {stats['keyframes']}\n"
            f"Dist. total: {stats['dist_total']:.1f} m\n"
            f"Prom. matches: {stats['matches_prom']:.1f}\n"
            f"Éxito triang: {stats['exito_triang']*100:.2f}%\n"
            f"Mov. medio: {stats['mov_medio']:.2f} m"
        )
        
        # Agregar métricas de comparación si están disponibles
        if has_slam_pure and has_gps and has_gps_keyframes:
            try:
                pipeline_vs_gps = comparison_results['pipeline_vs_ref']
                slam_vs_gps = comparison_results['slam_vs_ref']
                
                info_text += f"\n\n--- Métricas vs GPS GT ---"
                info_text += f"\nPipeline ATE: {pipeline_vs_gps['ate']['rmse']:.2f}m"
                info_text += f"\nSLAM ATE: {slam_vs_gps['ate']['rmse']:.2f}m"
                
                if slam_vs_gps['ate']['rmse'] > 0:
                    improvement = ((slam_vs_gps['ate']['rmse'] - pipeline_vs_gps['ate']['rmse']) / 
                                  slam_vs_gps['ate']['rmse']) * 100
                    info_text += f"\nMejora: {improvement:.1f}%"
            except:
                pass
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=11, bbox=dict(facecolor='white', alpha=0.95, edgecolor='black', linewidth=2),
                verticalalignment='top', family='monospace')

        plt.xlabel("X Position (m)", fontsize=14, fontweight='bold')
        plt.ylabel("Y Position (m)", fontsize=14, fontweight='bold')
        plt.title(f"COMPARACIÓN COMPLETA DE TRAYECTORIAS - {os.path.basename(input_path)}", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12, loc='best', framealpha=0.95, edgecolor='black')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.axis('equal')
        plt.tight_layout()
        
        plt.savefig(output_base + "_COMPARACION_COMPLETA.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Gráfico solo de keyframes (simplificado)
        plt.figure(figsize=(12, 9))
        plt.plot(x_key, y_key, 'b-', linewidth=3, marker='o', markersize=8)
        plt.scatter(x_key[0], y_key[0], color='green', s=200, label='Inicio')
        plt.scatter(x_key[-1], y_key[-1], color='red', s=200, label='Fin')
        
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

        # Calcular métricas individuales (legacy - para compatibilidad)
        if has_gps_keyframes and len(self.gps_keyframes) == len(self.keyframe_poses):
            print("\n" + "="*60)
            print("MÉTRICAS INDIVIDUALES (Pipeline vs GPS)")
            print("="*60)
            
            estimated_traj = np.array([pose[:3, 3] for pose in self.keyframe_poses])
            ground_truth_traj = np.array(self.gps_keyframes)
            
            ate_result = TrajectoryMetrics.calculate_ate(estimated_traj, ground_truth_traj, align=True)
            rpe_result = TrajectoryMetrics.calculate_rpe(estimated_traj, ground_truth_traj, delta=1)
            
            TrajectoryMetrics.print_metrics_summary(ate_result, rpe_result)
            
            metrics_file = output_base + "_metrics.json"
            TrajectoryMetrics.save_metrics_to_json(ate_result, rpe_result, None, metrics_file)
            
            if 'errors' in ate_result and len(ate_result['errors']) > 0:
                TrajectoryMetrics.plot_ate_errors(ate_result['errors'], output_base + "_ate_errors.png")
            
            if 'errors' in rpe_result and len(rpe_result['errors']) > 0:
                TrajectoryMetrics.plot_rpe_errors(rpe_result['errors'], output_base + "_rpe_translation.png", 
                                                   metric_name="RPE Translación")
            
            print(f"Métricas individuales guardadas en: {metrics_file}")
            print("="*60 + "\n")

        print(f"\n{'='*80}")
        print(f"RESULTADOS GUARDADOS EN: {output_dir}")
        print(f"{'='*80}")
        print(f"Gráfico comparativo completo: {output_base}_COMPARACION_COMPLETA.png")
        print(f"Gráfico keyframes: {output_base}_keyframes_only.png")
        print(f"Comparación CSV: {output_base}_comparison_metrics.csv")
        print(f"Comparación JSON: {output_base}_comparison_metrics.json")
        print(f"Trayectoria CSV: {output_base}_reales.csv")
        print(f"{'='*80}\n")
        print(f"Keyframes totales: {len(x_key)}")
        print(f"Distancia total: {real_total_distance:.2f} m\n")
    
    def _plot_ate_comparison(self, comparison_results: Dict, output_file: str):
        """Genera gráfico comparativo de errores ATE entre métodos."""
        plt.figure(figsize=(14, 8))
        
        colors = {'pipeline_vs_ref': 'blue', 'slam_vs_ref': 'green', 
                 'gps_vs_ref': 'red', 'pipeline_vs_slam': 'purple'}
        
        for key, data in comparison_results.items():
            if 'errors' in data['ate'] and len(data['ate']['errors']) > 0:
                errors = data['ate']['errors']
                label = data['method']
                color = colors.get(key, 'gray')
                
                plt.plot(errors, label=f"{label} (RMSE: {data['ate']['rmse']:.3f}m)", 
                        color=color, linewidth=2, alpha=0.7)
        
        plt.xlabel('Keyframe Index', fontsize=12, fontweight='bold')
        plt.ylabel('Error Absoluto (m)', fontsize=12, fontweight='bold')
        plt.title('Comparación de ATE entre Métodos', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Gráfico comparativo ATE guardado en: {output_file}")
    
    def _plot_rpe_comparison(self, comparison_results: Dict, output_file: str):
        """Genera gráfico comparativo de errores RPE entre métodos."""
        plt.figure(figsize=(14, 8))
        
        colors = {'pipeline_vs_ref': 'blue', 'slam_vs_ref': 'green', 
                 'gps_vs_ref': 'red', 'pipeline_vs_slam': 'purple'}
        
        for key, data in comparison_results.items():
            if 'errors' in data['rpe'] and len(data['rpe']['errors']) > 0:
                errors = data['rpe']['errors']
                label = data['method']
                color = colors.get(key, 'gray')
                
                plt.plot(errors, label=f"{label} (RMSE: {data['rpe']['trans_rmse']:.3f}m)", 
                        color=color, linewidth=2, alpha=0.7)
        
        plt.xlabel('Par de Keyframes', fontsize=12, fontweight='bold')
        plt.ylabel('Error Relativo (m)', fontsize=12, fontweight='bold')
        plt.title('Comparación de RPE entre Métodos', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Gráfico comparativo RPE guardado en: {output_file}")

    def optimize_pose_graph(self):
        print("\n=== INICIANDO OPTIMIZACIÓN ===")
        
        if not hasattr(self, 'keyframe_poses') or len(self.keyframe_poses) < 2:
            print("No hay suficientes keyframes para optimizar")
            return self.keyframe_poses if hasattr(self, 'keyframe_poses') else []
        
        if not self.enable_optimization:
            print("OPTIMIZACIÓN DESACTIVADA - Usando keyframes fusionados directamente")
            print(f"   Keyframes disponibles: {len(self.keyframe_poses)}")
            return self.keyframe_poses
        
        original_positions = np.array([pose[:3, 3] for pose in self.keyframe_poses])
        original_variance = np.var(original_positions, axis=0)
        print(f"Posiciones originales - Media: {np.mean(original_positions, axis=0)}")
        print(f"Posiciones originales - Varianza: {original_variance}")
        
        try:
            optimized_trajectory = super().optimize_pose_graph()
            print("Optimización completada")
            
            if optimized_trajectory and len(optimized_trajectory) > 0:
                opt_positions = np.array([pose[:3, 3] for pose in optimized_trajectory])
                opt_variance = np.var(opt_positions, axis=0)
                print(f"Posiciones optimizadas - Media: {np.mean(opt_positions, axis=0)}")
                print(f"Posiciones optimizadas - Varianza: {opt_variance}")
                
                if np.max(opt_variance) < 0.001:
                    print("¡ADVERTENCIA! La optimización puede haber colapsado las poses")
                    print("Usando keyframes originales en lugar de optimizados")
                    return self.keyframe_poses
                
                return optimized_trajectory
                
        except Exception as e:
            print(f"Error en optimización: {e}")
            print("Usando keyframes originales")
            return self.keyframe_poses
        
        return self.keyframe_poses

    def calculate_total_distance(self, trajectory):
        if len(trajectory) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(trajectory)):
            if hasattr(trajectory[i], "shape") and trajectory[i].shape == (4,4):
                pos1 = trajectory[i-1][:3, 3]
                pos2 = trajectory[i][:3, 3]
            elif isinstance(trajectory[i], (list, np.ndarray)) and len(trajectory[i]) >= 2:
                pos1 = trajectory[i-1][:2]
                pos2 = trajectory[i][:2]
            else:
                continue
            
            total_distance += np.linalg.norm(pos2 - pos1)
        
        return total_distance

    def plot_comparative_trajectory(self, output_base, x_opt, y_opt, input_path):
        plt.figure(figsize=(14, 10))
        
        plt.plot(x_opt, y_opt, 'b-', label='Trayectoria optimizada', linewidth=3, zorder=3)
        
        if hasattr(self, 'slam_history') and len(self.slam_history) > 0:
            slam_traj = np.array(self.slam_history)
            if slam_traj.shape[1] >= 2:
                x_slam, y_slam = slam_traj[:, 0], slam_traj[:, 1]
                x_slam -= x_slam[0]; y_slam -= y_slam[0]
                plt.plot(x_slam, y_slam, 'g--', label='SLAM puro', alpha=0.6, linewidth=2, zorder=2)
        
        if hasattr(self, 'gps_history') and len(self.gps_history) > 0:
            gps_traj = np.array(self.gps_history)
            if gps_traj.shape[1] >= 2:
                x_gps, y_gps = gps_traj[:, 0], gps_traj[:, 1]
                x_gps -= x_gps[0]; y_gps -= y_gps[0]
                plt.plot(x_gps, y_gps, 'r--', label='GPS puro', alpha=0.6, linewidth=2, zorder=1)
        
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
        print(f"Estructura de directorio KITTI no encontrada en: {sequence_path}")
        print("Buscando carpetas: image_02/data y oxts/data")
        return frames, gps_data
    
    print(f"Directorio de imágenes: {image_dir}")
    print(f"Directorio de GPS: {gps_dir}")
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    gps_files = sorted([f for f in os.listdir(gps_dir) if f.endswith('.txt')])

    
    if max_frames:
        image_files = image_files[:max_frames]
        gps_files = gps_files[:max_frames]
    
    min_files = min(len(image_files), len(gps_files))
    if min_files == 0:
        print("No se encontraron archivos coincidentes")
        return frames, gps_data
    
    print(f"OK - Encontrados {min_files} pares de archivos (imágenes + GPS)")
    
    for i in range(min_files):
        img_file = image_files[i]
        gps_file = gps_files[i]
        
        frame_path = os.path.join(image_dir, img_file)
        gps_path = os.path.join(gps_dir, gps_file)
        
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error cargando frame: {frame_path}")
            continue
        
        frames.append(frame)
        
        try:
            gps_point = np.loadtxt(gps_path)
            gps_data.append(gps_point)
        except Exception as e:
            print(f"Error cargando GPS {gps_path}: {e}")
            gps_data.append(None)
    
    print(f"Frames cargados: {len(frames)}")
    print(f"Datos GPS cargados: {len([g for g in gps_data if g is not None])}")
    return frames, gps_data

def main():
    parser = argparse.ArgumentParser(description='RL-ORB-SLAM con GPS')
    
    parser.add_argument('--mode', type=str, choices=['kitti', 'mobile'], required=True,
                        help='Tipo de datos: "kitti" para KITTI dataset, "mobile" para video móvil')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Número máximo de frames a procesar')
    parser.add_argument('--visualize', action='store_true',
                        help='Mostrar visualización durante el procesamiento')
    
    parser.add_argument('--kitti-base', type=str, default='kitti_data',
                        help='Ruta base del dataset KITTI')
    parser.add_argument('--sequence', type=str, default='2011_09_26/2011_09_26_drive_0005_sync',
                        help='Secuencia KITTI a procesar')
    parser.add_argument('--augmentation-prob', type=float, default=0.0,
                        help='Probabilidad de aplicar augmentación (solo KITTI)')
    
    parser.add_argument('--video', type=str,
                        help='Ruta al video MP4 (requerido si mode=mobile)')
    parser.add_argument('--gps-csv', type=str,
                        help='Ruta al archivo location.csv con GPS (requerido si mode=mobile)')
    parser.add_argument('--timestamps', type=str,
                        help='Ruta al archivo frame_timestamps.txt (requerido si mode=mobile)')
    
    args = parser.parse_args()
    
    if args.mode == 'mobile':
        if not args.video or not args.gps_csv or not args.timestamps:
            parser.error("Modo 'mobile' requiere --video, --gps-csv y --timestamps")
        
        if not os.path.exists(args.video):
            print(f"ERROR: Video no encontrado: {args.video}")
            return
        if not os.path.exists(args.gps_csv):
            print(f"ERROR: GPS CSV no encontrado: {args.gps_csv}")
            return
        if not os.path.exists(args.timestamps):
            print(f"ERROR: Timestamps no encontrado: {args.timestamps}")
            return
    
    print("\nInicializando RL-ORB-SLAM con GPS...")
    system = RL_ORB_SLAM_GPS()
    
    if args.mode == 'kitti':
        print(f"\nModo: KITTI Dataset")
        print(f"   Secuencia: {args.sequence}")
        
        sequence_path = os.path.join(args.kitti_base, args.sequence)
        if not os.path.exists(sequence_path):
            print(f"ERROR: Secuencia KITTI no encontrada: {sequence_path}")
            return
        
        system.process_kitti_sequence(
            sequence_path=sequence_path,
            max_frames=args.max_frames,
            augmentation_prob=args.augmentation_prob
        )
    
    elif args.mode == 'mobile':
        print(f"\nModo: Video Móvil")
        print(f"   Video: {args.video}")
        print(f"   GPS: {args.gps_csv}")
        print(f"   Timestamps: {args.timestamps}")
        
        system.process_mobile_sequence(
            video_path=args.video,
            location_csv=args.gps_csv,
            frame_timestamps_file=args.timestamps,
            max_frames=args.max_frames
        )
    
    print("\nProcesamiento completado exitosamente\n")

if __name__ == "__main__":
    main()