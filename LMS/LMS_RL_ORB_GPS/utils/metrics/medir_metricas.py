import time
import psutil
import subprocess
import json
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

"""
medir_metricas(comando, nombre_slam=None): mide recursos del proceso.
Si nombre_slam se especifica, guarda los resultados como metricas.json
en la carpeta resultados/<nombre_slam>/<timestamp>/
"""

def medir_metricas(comando, nombre_slam=None):
    tiempo_inicio = time.time()
    proceso = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ps_proceso = psutil.Process(proceso.pid)

    uso_cpu_usuario = 0
    uso_cpu_kernel = 0
    max_memoria = 0

    try:
        while proceso.poll() is None:
            cpu_times = ps_proceso.cpu_times()
            uso_cpu_usuario = cpu_times.user
            uso_cpu_kernel = cpu_times.system
            mem_info = ps_proceso.memory_info()
            max_memoria = max(max_memoria, mem_info.rss)
            time.sleep(0.1)

        stdout, stderr = proceso.communicate()
    except psutil.NoSuchProcess:
        stdout, stderr = b"", b"Proceso no disponible"

    tiempo_total = time.time() - tiempo_inicio

    resultados = {
        "tiempo_total_seg": round(tiempo_total, 3),
        "cpu_user_seg": round(uso_cpu_usuario, 4),
        "cpu_kernel_seg": round(uso_cpu_kernel, 4),
        "memoria_max_kb": round(max_memoria / 1024, 2),
        "codigo_retorno": proceso.returncode,
        "salida": stdout.decode(errors="ignore"),
        "errores": stderr.decode(errors="ignore")
    }

    if nombre_slam:
        timestamp = datetime.now().strftime("%H%M_%d%m_%Y")
        output_dir = os.path.join("resultados", nombre_slam, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "metricas.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=4, ensure_ascii=False)

    return resultados

def calculate_ate(gt_poses, estimated_poses):
    """
    Calcula el Absolute Trajectory Error (ATE)
    
    Args:
        gt_poses: Ground truth positions (Nx3)
        estimated_poses: Estimated positions (Nx3)
    
    Returns:
        float: ATE (RMSE) en metros
    """
    if len(gt_poses) != len(estimated_poses):
        min_len = min(len(gt_poses), len(estimated_poses))
        gt_poses = gt_poses[:min_len]
        estimated_poses = estimated_poses[:min_len]
    
    # Alinear trayectorias (solo traslación, sin rotación)
    # Centrar ambas trayectorias
    gt_centered = gt_poses - np.mean(gt_poses, axis=0)
    est_centered = estimated_poses - np.mean(estimated_poses, axis=0)
    
    # Calcular RMSE
    errors = np.linalg.norm(gt_centered - est_centered, axis=1)
    ate = np.sqrt(np.mean(errors ** 2))
    
    return ate

def calculate_rpe(gt_poses, estimated_poses, delta=1):
    """
    Calcula el Relative Pose Error (RPE)
    
    Args:
        gt_poses: Ground truth positions (Nx3)
        estimated_poses: Estimated positions (Nx3)
        delta: Delta de frames para calcular poses relativas
    
    Returns:
        tuple: (rpe_trans, rpe_rot) en metros y grados
    """
    if len(gt_poses) != len(estimated_poses):
        min_len = min(len(gt_poses), len(estimated_poses))
        gt_poses = gt_poses[:min_len]
        estimated_poses = estimated_poses[:min_len]
    
    trans_errors = []
    rot_errors = []
    
    for i in range(len(gt_poses) - delta):
        # Movimiento relativo en ground truth
        gt_delta = gt_poses[i + delta] - gt_poses[i]
        gt_dist = np.linalg.norm(gt_delta)
        
        # Movimiento relativo en estimación
        est_delta = estimated_poses[i + delta] - estimated_poses[i]
        est_dist = np.linalg.norm(est_delta)
        
        # Error de traslación
        trans_error = abs(gt_dist - est_dist)
        trans_errors.append(trans_error)
        
        # Error de rotación (aproximado por el ángulo entre vectores)
        if gt_dist > 0.01 and est_dist > 0.01:  # Evitar divisiones por cero
            cos_angle = np.dot(gt_delta, est_delta) / (gt_dist * est_dist)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Por errores numéricos
            angle_error = np.arccos(cos_angle) * 180.0 / np.pi
            rot_errors.append(angle_error)
    
    rpe_trans = np.sqrt(np.mean(np.array(trans_errors) ** 2)) if trans_errors else 0.0
    rpe_rot = np.sqrt(np.mean(np.array(rot_errors) ** 2)) if rot_errors else 0.0
    
    return rpe_trans, rpe_rot

class TrajectoryMetrics:
    """Clase para calcular métricas de trayectorias SLAM"""
    
    def __init__(self):
        pass
    
    def calculate_ate(self, estimated_traj, ground_truth_traj):
        """
        Calcula el Absolute Trajectory Error (ATE) con alineación Umeyama
        
        Args:
            estimated_traj: Lista de poses estimadas (matrices 4x4)
            ground_truth_traj: Lista de poses ground truth (matrices 4x4)
        
        Returns:
            dict: Diccionario con métricas ATE
        """
        # Extraer posiciones
        est_positions = np.array([pose[:3, 3] for pose in estimated_traj])
        gt_positions = np.array([pose[:3, 3] for pose in ground_truth_traj])
        
        if len(est_positions) != len(gt_positions):
            min_len = min(len(est_positions), len(gt_positions))
            est_positions = est_positions[:min_len]
            gt_positions = gt_positions[:min_len]
        
        # Alinear trayectorias usando Umeyama
        est_aligned = self._align_trajectory_umeyama(est_positions, gt_positions)
        
        # Calcular errores
        errors = np.linalg.norm(est_aligned - gt_positions, axis=1)
        
        return {
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'mean': float(np.mean(errors)),
            'median': float(np.median(errors)),
            'std': float(np.std(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
            'errors': errors
        }
    
    def calculate_rpe(self, estimated_poses, ground_truth_poses, delta=1):
        """
        Calcula el Relative Pose Error (RPE)
        
        Args:
            estimated_poses: Lista de poses estimadas (matrices 4x4)
            ground_truth_poses: Lista de poses ground truth (matrices 4x4)
            delta: Delta de frames para calcular poses relativas
        
        Returns:
            dict: Diccionario con métricas RPE de traslación
        """
        if len(estimated_poses) != len(ground_truth_poses):
            min_len = min(len(estimated_poses), len(ground_truth_poses))
            estimated_poses = estimated_poses[:min_len]
            ground_truth_poses = ground_truth_poses[:min_len]
        
        trans_errors = []
        
        for i in range(len(estimated_poses) - delta):
            # Extraer posiciones
            gt_pos1 = ground_truth_poses[i][:3, 3]
            gt_pos2 = ground_truth_poses[i + delta][:3, 3]
            est_pos1 = estimated_poses[i][:3, 3]
            est_pos2 = estimated_poses[i + delta][:3, 3]
            
            # Movimiento relativo
            gt_delta = np.linalg.norm(gt_pos2 - gt_pos1)
            est_delta = np.linalg.norm(est_pos2 - est_pos1)
            
            # Error de traslación
            trans_error = abs(gt_delta - est_delta)
            trans_errors.append(trans_error)
        
        if not trans_errors:
            return {
                'rmse': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'errors': []
            }
        
        trans_errors = np.array(trans_errors)
        
        return {
            'rmse': float(np.sqrt(np.mean(trans_errors ** 2))),
            'mean': float(np.mean(trans_errors)),
            'median': float(np.median(trans_errors)),
            'std': float(np.std(trans_errors)),
            'min': float(np.min(trans_errors)),
            'max': float(np.max(trans_errors)),
            'errors': trans_errors
        }
    
    def _align_trajectory_umeyama(self, src, dst):
        """
        Alinea la trayectoria src con dst usando el algoritmo de Umeyama
        (alineación de similitud en 3D: rotación + traslación + escala)
        """
        # Centrar las nubes de puntos
        src_mean = np.mean(src, axis=0)
        dst_mean = np.mean(dst, axis=0)
        
        src_centered = src - src_mean
        dst_centered = dst - dst_mean
        
        # Calcular la matriz de covarianza
        H = src_centered.T @ dst_centered
        
        # SVD para encontrar la rotación
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Asegurar una rotación válida (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Calcular la escala
        src_var = np.sum(src_centered ** 2)
        scale = np.sum(S) / src_var if src_var > 0 else 1.0
        
        # Aplicar transformación
        src_aligned = scale * (src_centered @ R.T) + dst_mean
        
        return src_aligned
    
    def print_metrics_summary(self, ate_result, rpe_result, rpe_rot_result):
        """Imprime un resumen de las métricas"""
        print("\n" + "="*60)
        print("RESUMEN DE MÉTRICAS")
        print("="*60)
        
        if ate_result:
            print("\nAbsolute Trajectory Error (ATE):")
            print(f"  RMSE:   {ate_result['rmse']:.4f} m")
            print(f"  Mean:   {ate_result['mean']:.4f} m")
            print(f"  Median: {ate_result['median']:.4f} m")
            print(f"  Std:    {ate_result['std']:.4f} m")
            print(f"  Min:    {ate_result['min']:.4f} m")
            print(f"  Max:    {ate_result['max']:.4f} m")
        
        if rpe_result:
            print("\nRelative Pose Error (RPE) - Traslación:")
            print(f"  RMSE:   {rpe_result['rmse']:.4f} m")
            print(f"  Mean:   {rpe_result['mean']:.4f} m")
            print(f"  Median: {rpe_result['median']:.4f} m")
            print(f"  Std:    {rpe_result['std']:.4f} m")
        
        print("="*60 + "\n")
    
    def save_metrics_to_json(self, ate_result, rpe_result, rpe_rot_result, output_file):
        """Guarda las métricas en un archivo JSON"""
        metrics_dict = {
            'ate': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                   for k, v in ate_result.items() if k != 'errors'},
            'rpe_translation': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                               for k, v in rpe_result.items() if k != 'errors'} if rpe_result else None
        }
        
        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        
        print(f"Métricas guardadas en: {output_file}")
    
    def plot_ate_errors(self, errors, output_file):
        """Genera un gráfico de errores ATE"""
        plt.figure(figsize=(12, 6))
        plt.plot(errors, 'b-', linewidth=1.5)
        plt.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Media: {np.mean(errors):.3f}m')
        plt.xlabel('Frame')
        plt.ylabel('Error (m)')
        plt.title('Absolute Trajectory Error (ATE) por Frame')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Gráfico ATE guardado en: {output_file}")
    
    def plot_rpe_errors(self, errors, output_file, metric_name="RPE"):
        """Genera un gráfico de errores RPE"""
        if len(errors) == 0:
            print(f"No hay errores {metric_name} para graficar")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(errors, 'g-', linewidth=1.5)
        plt.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Media: {np.mean(errors):.3f}m')
        plt.xlabel('Par de Frames')
        plt.ylabel('Error (m)')
        plt.title(f'{metric_name} por Par de Frames')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Gráfico {metric_name} guardado en: {output_file}")
