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
from LMS.LMS_RL_ORB_GPS.model.rl_agent import SimpleRLAgent, RLTrainer
from LMS.LMS_RL_ORB_GPS.utils.gps_utils import *
from LMS.LMS_RL_ORB_GPS.utils.gps_filter import *
from LMS.LMS_RL_ORB_GPS.utils.training_augmentation import TrainingAugmentation

class RL_ORB_SLAM_GPS(PoseGraphSLAM):
    def __init__(self, fx=718.856, fy=718.856, cx=607.1928, cy=185.2157, 
                 training_mode=False, model_path=None, simulate_mobile_gps=False, gps_noise_std=5.0):
        """
        Args:
            simulate_mobile_gps: Si True, agrega ruido al GPS para simular GPS móvil
            gps_noise_std: Desviación estándar del ruido GPS en metros (default: 5.0m)
        """
        super().__init__(fx, fy, cx, cy)
        self.agent = SimpleRLAgent()
        self.trainer = RLTrainer(self.agent)
        self.training_mode = training_mode
        self.simulate_mobile_gps = simulate_mobile_gps
        
        # Intentar cargar modelo pre-entrenado
        if model_path and os.path.exists(model_path):
            self.trainer.load_model(model_path)
        elif not training_mode:
            # Buscar modelo por defecto
            default_model = "LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth"
            if os.path.exists(default_model):
                print(f"Cargando modelo pre-entrenado: {default_model}")
                self.trainer.load_model(default_model)
            else:
                print("ADVERTENCIA: No se encontró modelo pre-entrenado. Usando pesos aleatorios.")
        
        self.utm_reference = None
        self.previous_gps_utm = None
        self.gps_available = False
        self.gps_history = []  # Para almacenar historial de posiciones GPS
        self.slam_history = []  # Para almacenar historial de posiciones SLAM
        
        # Para entrenamiento
        self.current_state = None
        self.frame_count = 0
        self.training_step_interval = 10  # Entrenar cada N frames

        # GPS Filter con soporte para ruido simulado
        self.gps_filter = GPSFilter(
            window_size=5, 
            movement_variance_threshold=0.1,
            add_noise=simulate_mobile_gps,
            noise_std=gps_noise_std
        )
        
        if simulate_mobile_gps:
            print(f"ADVERTENCIA: MODO GPS MÓVIL SIMULADO: Agregando ruido gaussiano (sigma={gps_noise_std}m)")

    def gps_frame_reference(self, gps_frame_value):
        lat, lon, alt = gps_frame_value[0], gps_frame_value[1], gps_frame_value[2]
        utm_coord = latlon_to_utm(lat, lon, alt)
        if self.utm_reference is None:
            self.utm_reference = utm_coord
        return utm_coord - self.utm_reference
    
    def calculate_base_weights(self, gps_conf, visual_conf, error):
        """
        Calcula pesos base usando reglas heurísticas MEJORADAS.
        Ahora considera el ratio de confianzas para balancear mejor cuando ambos son buenos.
        
        Args:
            gps_conf: Confianza del GPS [0, 1]
            visual_conf: Confianza del SLAM visual [0, 1]
            error: Error entre SLAM y GPS en metros
        
        Returns:
            (w_slam_base, w_gps_base): Tupla con pesos base continuos
        """
        # === NUEVO: ANÁLISIS DE RATIO DE CONFIANZAS ===
        conf_ratio = visual_conf / (gps_conf + 1e-6)  # Ratio SLAM/GPS
        
        # Factor 1: Pesos base según confianzas
        gps_weight_from_conf = gps_conf
        slam_weight_from_conf = visual_conf
        
        # === NUEVO: AJUSTE SEGÚN RATIO DE CONFIANZAS ===
        # Si SLAM es significativamente más confiable que GPS
        if conf_ratio > 1.15 and visual_conf > 0.75:
            slam_boost = 0.15
            gps_reduction = 0.10
            print(f"  [BALANCE] SLAM más confiable (ratio={conf_ratio:.2f}) → Boost SLAM")
        # Si GPS es significativamente más confiable que SLAM
        elif conf_ratio < 0.85 and gps_conf > 0.75:
            slam_boost = -0.10
            gps_reduction = -0.15
            print(f"  [BALANCE] GPS más confiable (ratio={conf_ratio:.2f}) → Boost GPS")
        # CORRECCIÓN: Ambos similares y confiables → Balance equitativo (sin boost adicional)
        elif visual_conf > 0.75 and gps_conf > 0.75 and 0.85 <= conf_ratio <= 1.15:
            slam_boost = 0.0
            gps_reduction = 0.0
            print(f"  [BALANCE] Ambos muy confiables (ratio={conf_ratio:.2f}) → Balance natural 50-50")
        else:
            slam_boost = 0.0
            gps_reduction = 0.0
        
        slam_weight_from_conf += slam_boost
        gps_weight_from_conf += gps_reduction
        
        # Factor 2: Penalización GRADUAL por error alto
        error_penalty_gps = 0.0
        if error > 5.0:
            error_penalty_gps = 0.3  # Reduce GPS significativamente si error es muy alto
            print(f"  [ERROR] Alto error ({error:.2f}m) → Penaliza GPS -0.30")
        elif error > 2.0:
            # Penalización gradual entre 2m y 5m
            error_penalty_gps = 0.1 * ((error - 2.0) / 3.0)
            print(f"  [ERROR] Error moderado ({error:.2f}m) → Penaliza GPS -{error_penalty_gps:.2f}")
        
        # Factor 3: Bonificación GRADUAL por error bajo (GPS alineado con SLAM)
        error_bonus_gps = 0.0
        if error < 1.0 and gps_conf > 0.6:
            error_bonus_gps = 0.15  # Reducido de 0.2 para ser menos agresivo
            print(f"  [ALIGN] Excelente alineación ({error:.2f}m) → Bonus GPS +0.15")
        elif error < 2.0 and gps_conf > 0.6:
            # Bonificación gradual entre 1m y 2m
            error_bonus_gps = 0.1 * (2.0 - error)
            print(f"  [ALIGN] Buena alineación ({error:.2f}m) → Bonus GPS +{error_bonus_gps:.2f}")
        
        # === COMBINACIÓN DE FACTORES ===
        w_gps_base = gps_weight_from_conf - error_penalty_gps + error_bonus_gps
        w_slam_base = slam_weight_from_conf
        
        # Normalizar para que estén en rango [0.05, 0.95] (permitir casos extremos)
        total = w_slam_base + w_gps_base + 1e-6
        w_slam_base = np.clip(w_slam_base / total, 0.05, 0.95)
        w_gps_base = np.clip(w_gps_base / total, 0.05, 0.95)
        
        # Re-normalizar para que sumen exactamente 1.0
        total = w_slam_base + w_gps_base
        w_slam_base /= total
        w_gps_base /= total
        
        return w_slam_base, w_gps_base
    
    def calculate_rl_adjustment_margin(self, visual_conf, gps_conf, error):
        """
        Calcula el margen dinámico para los ajustes del RL basado en la certeza del sistema.
        
        FILOSOFÍA:
        - Alta certeza (ambos sensores confiables) → RL tiene POCO margen (reglas deciden)
        - Incertidumbre (sensores discrepan o uno es malo) → RL tiene MÁS margen (RL decide)
        - Casos extremos (uno pésimo, otro excelente) → RL tiene MÁXIMO margen (±50%)
        - Emergencia (ambos malos) → RL tiene MÁXIMO margen (RL toma control total)
        
        Args:
            visual_conf: Confianza del SLAM visual [0, 1]
            gps_conf: Confianza del GPS [0, 1]
            error: Error entre SLAM y GPS en metros
        
        Returns:
            (rl_margin, scenario): Margen para ajuste RL y nombre del escenario
        """
        conf_diff = abs(visual_conf - gps_conf)
        avg_conf = (visual_conf + gps_conf) / 2.0
        min_conf = min(visual_conf, gps_conf)
        max_conf = max(visual_conf, gps_conf)
        
        # === NUEVO: ESCENARIO 0 - SENSOR EXTREMO (PRIORIDAD MÁXIMA) ===
        # Caso EXTREMO: Uno pésimo (<0.3), otro excelente (>0.7), diferencia enorme (>0.6)
        # Ejemplo: GPS=0.10, SLAM=1.00 o GPS=1.00, SLAM=0.20
        if min_conf < 0.3 and max_conf > 0.7 and conf_diff > 0.6:
            rl_margin = 0.50  # ±50% → RL puede ajustar MUCHO
            scenario = "SENSOR_EXTREMO"
            print(f"  [MARGEN] Caso EXTREMO detectado (min={min_conf:.2f}, max={max_conf:.2f}, diff={conf_diff:.2f}) → Margen amplio (±50%)")
            return rl_margin, scenario
        
        # ESCENARIO 1: Ambos muy confiables y similares (>0.75 y diff<0.15)
        # Ejemplo: GPS=0.95, SLAM=1.00, Error=1.5m
        if avg_conf > 0.75 and conf_diff < 0.15 and error < 3.0:
            rl_margin = 0.10  # ±10% → Las reglas dominan
            scenario = "ALTA_CERTEZA"
        
        # ESCENARIO 2: Uno claramente mejor que otro (diff>0.30)
        # Ejemplo: GPS=0.95, SLAM=0.30 (pocos matches)
        elif conf_diff > 0.30:
            rl_margin = 0.25  # ±25% → RL puede ajustar moderadamente (aumentado de 0.20)
            scenario = "SENSOR_DOMINANTE"
        
        # ESCENARIO 3: Ambos confiables pero error alto (>3m)
        # Ejemplo: GPS=0.90, SLAM=0.85, pero Error=4.5m (algo está mal)
        elif avg_conf > 0.60 and error > 3.0:
            rl_margin = 0.30  # ±30% → RL puede investigar (aumentado de 0.25)
            scenario = "ERROR_ALTO"
        
        # ESCENARIO 4: Incertidumbre general (confianzas bajas)
        # Ejemplo: GPS=0.40, SLAM=0.50 (ambos regulares)
        elif avg_conf < 0.50:
            rl_margin = 0.40  # ±40% → RL toma más control (aumentado de 0.35)
            scenario = "INCERTIDUMBRE"
        
        # ESCENARIO 5: Caso intermedio
        else:
            rl_margin = 0.20  # ±20% → Balance normal (aumentado de 0.15)
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

            # INICIALIZAR CON GPS
            initial_pose = np.eye(4)
            if gps_utm is not None:
                initial_pose[:3, 3] = gps_utm
                print(f"Inicializado con GPS: {gps_utm.round(3)}")
            else:
                initial_pose[:3, 3] = np.zeros(3)
                print("Inicializado en origen (sin GPS)")

            self.previous_keyframe_pose = initial_pose
            self.keyframe_poses.append(initial_pose)

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

                    # === ALINEACIÓN DE DIRECCIÓN SLAM CON GPS ===
                    if self.previous_gps_utm is not None and gps_utm is not None:
                        movimiento_gps = gps_utm - self.previous_gps_utm
                        movimiento_gps_2d = movimiento_gps[:2]

                        if np.linalg.norm(movimiento_gps_2d) > 0.1 and np.linalg.norm(t[:2]) > 0.0001:
                            direccion_slam = t[:2].ravel() / np.linalg.norm(t[:2])
                            direccion_gps = movimiento_gps_2d / np.linalg.norm(movimiento_gps_2d)

                            # Ángulo entre direcciones
                            angle_slam = np.arctan2(direccion_slam[1], direccion_slam[0])
                            angle_gps = np.arctan2(direccion_gps[1], direccion_gps[0])
                            delta_angle = angle_gps - angle_slam

                            # Rotación 2D sobre Z
                            R_align = np.array([
                                [np.cos(delta_angle), -np.sin(delta_angle), 0],
                                [np.sin(delta_angle),  np.cos(delta_angle), 0],
                                [0, 0, 1]
                            ])
                            t[:3] = R_align @ t[:3]
                            R = R_align @ R

                            print(f"Dirección SLAM alineada con GPS (delta_angle={np.degrees(delta_angle):.2f}°)")
                            print(f"  Vector t corregido: {t.ravel().round(3)} | Magnitud: {np.linalg.norm(t):.6f}")


                    # === ESCALADO SEGÚN DISTANCIA GPS ===
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

                    # COMPOSICIÓN DE POSE
                    current_pose = self.previous_keyframe_pose @ relative_pose

                    # === FUSIÓN HÍBRIDA: REGLAS + RL ===
                    visual_confidence = min(len(matches) / 50.0, 1.0)
                    error = np.linalg.norm(current_pose[:3, 3] - gps_utm) if gps_utm is not None else 0.0
                    
                    # === DEBUG: Imprimir confianzas y error ===
                    print(f"  [CONFIANZAS] Visual={visual_confidence:.3f} (matches={len(matches)}), GPS={gps_confidence:.3f}, Error={error:.3f}m")
                    
                    # Calcular posición fusionada
                    fused_position = current_pose[:3, 3]
                    if gps_utm is not None:
                        # 1. CALCULAR PESOS BASE CON REGLAS HEURÍSTICAS
                        w_slam_base, w_gps_base = self.calculate_base_weights(gps_confidence, visual_confidence, error)
                        
                        # 2. CALCULAR MARGEN DINÁMICO PARA RL (NUEVO)
                        rl_margin, scenario = self.calculate_rl_adjustment_margin(visual_confidence, gps_confidence, error)
                        
                        # 3. APLICAR RL CON MARGEN DINÁMICO
                        state = torch.tensor([visual_confidence, gps_confidence, error], dtype=torch.float32)
                        
                        with torch.no_grad():
                            rl_weights = self.agent(state)
                            
                            # Convertir salida RL a ajuste con margen dinámico
                            # Ejemplo: si rl_margin=0.10 → ajuste de [-0.10, +0.10]
                            #          si rl_margin=0.35 → ajuste de [-0.35, +0.35]
                            rl_adjustment_slam = (rl_weights[0].item() - 0.5) * (rl_margin * 2)
                            rl_adjustment_gps = (rl_weights[1].item() - 0.5) * (rl_margin * 2)
                        
                        # 4. APLICAR AJUSTE RL A PESOS BASE
                        w_slam_final = np.clip(w_slam_base + rl_adjustment_slam, 0.05, 0.95)
                        w_gps_final = np.clip(w_gps_base + rl_adjustment_gps, 0.05, 0.95)
                        
                        # 5. NORMALIZAR (para que sumen 1.0)
                        total = w_slam_final + w_gps_final
                        w_slam_final /= total
                        w_gps_final /= total
                        
                        # 6. CALCULAR POSICIÓN FUSIONADA
                        fused_position = w_slam_final * current_pose[:3, 3] + w_gps_final * gps_utm
                        
                        # === IMPRIMIR INFORMACIÓN DETALLADA ===
                        print(f"  [HYBRID] Base: SLAM={w_slam_base:.3f}, GPS={w_gps_base:.3f}")
                        print(f"  [RL MARGIN] Escenario={scenario}, Margen=±{rl_margin*100:.0f}%")
                        print(f"  [RL ADJ] Δ_SLAM={rl_adjustment_slam:+.3f}, Δ_GPS={rl_adjustment_gps:+.3f}")
                        print(f"  [FINAL]  SLAM={w_slam_final:.3f}, GPS={w_gps_final:.3f}, "
                              f"GPS_Conf={gps_confidence:.3f}, Error={error:.3f}m")
                        
                        # === ENTRENAMIENTO RL (si está activo) ===
                        if self.training_mode:
                            # El RL aprende a ajustar finamente alrededor de las reglas
                            weights_trainable = self.agent(state)
                            
                            # Reward modificado: premiar ajustes correctos alrededor de pesos base
                            reward = self.trainer.calculate_reward(
                                state=state,
                                action_weights=weights_trainable,
                                slam_position=current_pose[:3, 3],
                                gps_position=gps_utm,
                                ground_truth=None
                            )
                            
                            # Almacenar experiencia
                            self.trainer.store_experience(state, weights_trainable.detach(), reward, None, done=False)
                            
                            # Entrenar periódicamente
                            self.frame_count += 1
                            if self.frame_count % self.training_step_interval == 0:
                                train_result = self.trainer.train_step(batch_size=32)
                                if train_result:
                                    print(f"  [TRAINING] Loss: {train_result['loss']:.4f}, "
                                          f"Avg Reward: {train_result['avg_reward']:.4f}")
                            
                            print(f"  [RL TRAINING] Reward={reward:.3f}")

                    fused_pose = np.eye(4)
                    fused_pose[:3, 3] = fused_position
                    fused_pose[:3, :3] = current_pose[:3, :3]

                    # Actualizar keyframe y historial
                    self.slam_history.append(current_pose[:3, 3].copy())
                    if gps_utm is not None:
                        self.gps_history.append(gps_utm.copy())

                    translation_magnitude = np.linalg.norm(relative_pose[:3, 3])
                    if translation_magnitude > 0.1:
                        self.keyframe_poses.append(fused_pose)
                        self.relative_transformations.append(relative_pose)
                        self.previous_keyframe_image = gray
                        self.previous_keyframe_keypoints = keypoints
                        self.previous_keyframe_descriptors = descriptors
                        self.previous_keyframe_pose = fused_pose
                        print(f"Nuevo keyframe (movimiento={translation_magnitude:.3f} m)")
                    else:
                        print(f"Pose estimada, pero no keyframe (movimiento pequeño)")

                    # Guardar GPS para próximo frame
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
        """
        Procesa secuencia KITTI con soporte para aumentación de datos.
        
        Args:
            sequence_path: Ruta a la secuencia KITTI
            max_frames: Límite de frames (None = todos)
            augmentation_prob: Probabilidad de aplicar aumentación (0.0-1.0)
                              0.0 = sin aumentación (inferencia)
                              0.3-0.6 = aumentación progresiva (entrenamiento)
        """
        frames, gps_data = load_kitti_sequence(sequence_path, max_frames)
        
        print(f"\n{'='*60}")
        print(f"MODO: {'ENTRENAMIENTO' if self.training_mode else 'INFERENCIA'}")
        print(f"Total de frames: {len(frames)}")
        if self.training_mode and augmentation_prob > 0:
            print(f"AUMENTACIÓN ACTIVA: {augmentation_prob*100:.0f}% de probabilidad")
        print(f"{'='*60}\n")
        
        # Crear módulo de aumentación si está en modo entrenamiento
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
                # Procesar datos GPS a UTM
                gps_utm = None
                gps_confidence = 0.1

                if gps_data[i] is not None:
                    gps_utm = self.gps_frame_reference(gps_data[i])
                    print(f"  GPS UTM: {gps_utm.round(3)}")

                    self.gps_filter.add_measurement(gps_utm)
                    gps_confidence_raw = self.gps_filter.calculate_confidence()
                    
                    # === APLICAR AUMENTACIÓN DE DATOS (solo en entrenamiento) ===
                    if augmenter is not None and np.random.rand() < augmentation_prob:
                        # Calcular error aproximado (usando último keyframe si existe)
                        if hasattr(self, 'keyframe_poses') and len(self.keyframe_poses) > 0:
                            last_pose = self.keyframe_poses[-1][:3, 3]
                            error_approx = np.linalg.norm(last_pose - gps_utm) if gps_utm is not None else 0.0
                        else:
                            error_approx = 0.0
                        
                        # Aumentar datos (corromper sensores artificialmente)
                        gps_utm_aug, gps_conf_aug, visual_conf_aug, scenario = augmenter.augment_training_sample(
                            gps_utm=gps_utm,
                            gps_conf=gps_confidence_raw,
                            visual_conf=0.5,  # Placeholder, se calculará en process_frame_with_gps
                            error=error_approx
                        )
                        
                        # Usar datos aumentados
                        gps_utm = gps_utm_aug
                        gps_confidence = gps_conf_aug
                        
                        print(f"  🎲 [AUGMENT] Escenario aplicado: {scenario}")
                    else:
                        # Usar datos originales sin modificar
                        gps_confidence = gps_confidence_raw

                    self.gps_filter.print_debug_info()
                
                # Usar el método correcto que acepta GPS
                self.process_frame_with_gps(frames[i], gps_utm, gps_confidence)
                    
            except Exception as e:
                print(f"  ERROR - Error en frame {i}: {e}")
        
        # Imprimir estadísticas de aumentación si se usó
        if augmenter is not None:
            augmenter.print_statistics()
        
        # Si estamos en modo entrenamiento, guardar el modelo
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
                print(f"Keyframes sin optimizar: {len(keyframe_positions)} poses")
                
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
                print("No hay keyframes disponibles")
                return
                
        except Exception as e:
            print(f"Error al extraer keyframes: {e}")
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
            print("No se pudo graficar trayectoria optimizada (probablemente colapsada)")

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

        print(f"Gráficos REALES guardados en: {output_dir}")
        print(f"La trayectoria REAL tiene {len(x_key)} puntos")
        print(f"Distancia total REAL: {real_total_distance:.2f} m")

    def optimize_pose_graph(self):
        """Método de optimización con diagnóstico"""
        print("\n=== INICIANDO OPTIMIZACIÓN ===")
        
        if not hasattr(self, 'keyframe_poses') or len(self.keyframe_poses) < 2:
            print("No hay suficientes keyframes para optimizar")
            return self.keyframe_poses if hasattr(self, 'keyframe_poses') else []
        
        # Mostrar estadísticas antes de optimizar
        original_positions = np.array([pose[:3, 3] for pose in self.keyframe_poses])
        original_variance = np.var(original_positions, axis=0)
        print(f"Posiciones originales - Media: {np.mean(original_positions, axis=0)}")
        print(f"Posiciones originales - Varianza: {original_variance}")
        
        # Llamar a la optimización original del padre
        try:
            optimized_trajectory = super().optimize_pose_graph()
            print("Optimización completada")
            
            # Mostrar estadísticas después de optimizar
            if optimized_trajectory and len(optimized_trajectory) > 0:
                opt_positions = np.array([pose[:3, 3] for pose in optimized_trajectory])
                opt_variance = np.var(opt_positions, axis=0)
                print(f"Posiciones optimizadas - Media: {np.mean(opt_positions, axis=0)}")
                print(f"Posiciones optimizadas - Varianza: {opt_variance}")
                
                # Verificar si la optimización colapsó
                if np.max(opt_variance) < 0.001:  # Varianza muy pequeña
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SLAM con RL y GPS para videos MP4 o secuencias KITTI')
    parser.add_argument('input_path', help='Ruta al video MP4 o directorio de secuencia KITTI')
    parser.add_argument('--kitti', action='store_true', help='Forzar modo KITTI')
    parser.add_argument('--max_frames', type=int, help='Número máximo de frames a procesar')
    parser.add_argument('--train', action='store_true', help='Activar modo entrenamiento')
    parser.add_argument('--model_path', type=str, help='Ruta al modelo pre-entrenado')
    parser.add_argument('--simulate_mobile_gps', action='store_true', help='Simular GPS móvil con ruido')
    parser.add_argument('--gps_noise_std', type=float, default=5.0, help='Desviación estándar del ruido GPS simulado (en metros)')
    parser.add_argument('--augmentation_prob', type=float, default=0.0, help='Probabilidad de aumentación de datos (0.0-1.0)')

    args = parser.parse_args()
    
    input_path = args.input_path
    training_mode = args.train
    model_path = args.model_path
    simulate_mobile_gps = args.simulate_mobile_gps
    gps_noise_std = args.gps_noise_std
    augmentation_prob = args.augmentation_prob
    
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
        slam = RL_ORB_SLAM_GPS(training_mode=training_mode, model_path=model_path, 
                               simulate_mobile_gps=simulate_mobile_gps, gps_noise_std=gps_noise_std)
        slam.process_kitti_sequence(input_path, args.max_frames, augmentation_prob)
        
    elif input_path.endswith('.mp4'):
        print("Modo: Video MP4")
        slam = RL_ORB_SLAM_GPS(training_mode=training_mode, model_path=model_path, 
                               simulate_mobile_gps=simulate_mobile_gps, gps_noise_std=gps_noise_std)
        slam.process_video_input(input_path)
        
    else:
        print("Error: Formato de entrada no reconocido")
        print("Use un directorio KITTI (con image_02 y oxts) o un archivo .mp4")
        sys.exit(1)
    
    print("Procesamiento completado")