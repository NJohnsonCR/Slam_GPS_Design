import time
import psutil
import subprocess
import json
import os
import numpy as np
from datetime import datetime

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
