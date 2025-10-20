#!/usr/bin/env python3
"""
Script para evaluar el modelo RL entrenado
Calcula métricas de precisión (ATE, RPE) comparando con ground truth
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

from LMS.LMS_RL_ORB_GPS.main import RL_ORB_SLAM_GPS
from LMS.LMS_RL_ORB_GPS.utils.medir_metricas import calculate_ate, calculate_rpe

def load_kitti_ground_truth(sequence_path):
    """
    Carga ground truth de KITTI
    
    Returns:
        np.array: Nx3 array de posiciones [x, y, z]
    """
    # Cargar datos OXTS (GPS/IMU)
    oxts_dir = os.path.join(sequence_path, 'oxts', 'data')
    
    if not os.path.exists(oxts_dir):
        print(f"ERROR - ERROR: Directorio OXTS no encontrado: {oxts_dir}")
        return None
    
    oxts_files = sorted([f for f in os.listdir(oxts_dir) if f.endswith('.txt')])
    
    # Convertir OXTS a poses UTM
    from LMS.LMS_RL_ORB_GPS.utils.gps_utils import latlon_to_utm
    
    poses = []
    for oxts_file in oxts_files:
        oxts_path = os.path.join(oxts_dir, oxts_file)
        with open(oxts_path, 'r') as f:
            data = f.readline().strip().split()
            if len(data) >= 3:
                lat = float(data[0])
                lon = float(data[1])
                alt = float(data[2])
                utm_pos = latlon_to_utm(lat, lon, alt)
                poses.append(utm_pos)
    
    poses = np.array(poses)
    
    # Normalizar al origen (primer punto en 0,0,0)
    if len(poses) > 0:
        poses = poses - poses[0]
    
    print(f"OK - Ground truth cargado: {len(poses)} poses")
    return poses

def evaluate_model(sequence_path, model_path, noise_levels=[0.0, 3.0, 5.0, 10.0]):
    """
    Evalúa el modelo con diferentes niveles de ruido GPS
    
    Args:
        sequence_path: Ruta a secuencia KITTI
        model_path: Ruta al modelo entrenado
        noise_levels: Lista de desviaciones estándar de ruido GPS (metros)
    
    Returns:
        dict: Resultados de evaluación
    """
    print("="*70)
    print("EVALUACIÓN DE MODELO RL-ORB-SLAM-GPS")
    print("="*70)
    print(f"Secuencia: {sequence_path}")
    print(f"Modelo: {model_path}")
    print(f"Niveles de ruido GPS: {noise_levels} metros")
    print("="*70)
    print()
    
    # Verificar archivos
    if not os.path.exists(sequence_path):
        print(f"ERROR - ERROR: Secuencia no encontrada: {sequence_path}")
        return None
    
    if not os.path.exists(model_path):
        print(f"ERROR - ERROR: Modelo no encontrado: {model_path}")
        return None
    
    # Cargar ground truth
    gt_poses = load_kitti_ground_truth(sequence_path)
    
    # Cargar imágenes
    image_dir = os.path.join(sequence_path, 'image_02', 'data')
    if not os.path.exists(image_dir):
        image_dir = os.path.join(sequence_path, 'image_00', 'data')
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    total_frames = len(image_files)
    
    print(f"OK - {total_frames} imágenes encontradas")
    print()
    
    # Parámetros de calibración KITTI
    fx, fy = 718.856, 718.856
    cx, cy = 607.1928, 185.2157
    
    results = {
        'sequence': os.path.basename(sequence_path),
        'model': model_path,
        'total_frames': total_frames,
        'noise_evaluations': []
    }
    
    # Evaluar con diferentes niveles de ruido
    for noise_std in noise_levels:
        print(f"\n{'='*70}")
        print(f"EVALUANDO CON RUIDO GPS: {noise_std} metros (σ)")
        print(f"{'='*70}\n")
        
        # Crear instancia SLAM en modo inferencia
        slam = RL_ORB_SLAM_GPS(
            fx=fx, fy=fy, cx=cx, cy=cy,
            training_mode=False,  # Modo evaluación
            model_path=model_path,
            simulate_mobile_gps=True,
            gps_noise_std=noise_std
        )
        
        # Procesar todos los frames
        import cv2
        for idx, img_file in enumerate(image_files):
            if idx % 10 == 0:
                print(f"Procesando frame {idx}/{total_frames}...")
            
            # Leer imagen
            img_path = os.path.join(image_dir, img_file)
            frame = cv2.imread(img_path)
            
            if frame is None:
                continue
            
            # GPS ground truth con ruido
            gt_position = gt_poses[min(idx, len(gt_poses)-1)]
            
            # Simular GPS ruidoso
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, 3)
                gps_noisy = gt_position + noise
            else:
                gps_noisy = gt_position.copy()
            
            # Calcular confianza GPS
            slam.gps_filter.add_measurement(gps_noisy)
            gps_confidence = slam.gps_filter.calculate_confidence()
            
            # Procesar frame
            slam.process_frame_with_gps(frame, gps_noisy, gps_confidence)
        
        # Obtener trayectoria optimizada
        optimized_trajectory = slam.optimize_pose_graph()
        
        if len(optimized_trajectory) == 0:
            print(f"ADVERTENCIA:  No se generó trayectoria para ruido {noise_std}m")
            continue
        
        # Extraer posiciones estimadas
        estimated_positions = np.array([pose[:3, 3] for pose in optimized_trajectory])
        
        # Alinear tamaños
        min_len = min(len(gt_poses), len(estimated_positions))
        gt_aligned = gt_poses[:min_len]
        est_aligned = estimated_positions[:min_len]
        
        # Calcular métricas
        ate = calculate_ate(gt_aligned, est_aligned)
        rpe_trans, rpe_rot = calculate_rpe(gt_aligned, est_aligned)
        
        # Calcular mejora vs GPS puro
        gps_positions = []
        for idx in range(min_len):
            gt_pos = gt_poses[idx]
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, 3)
                gps_positions.append(gt_pos + noise)
            else:
                gps_positions.append(gt_pos)
        gps_positions = np.array(gps_positions)
        
        ate_gps = calculate_ate(gt_aligned, gps_positions)
        improvement = ((ate_gps - ate) / ate_gps) * 100 if ate_gps > 0 else 0
        
        # Guardar resultados
        noise_result = {
            'noise_std': noise_std,
            'frames_processed': len(estimated_positions),
            'ate': float(ate),
            'rpe_trans': float(rpe_trans),
            'rpe_rot': float(rpe_rot),
            'ate_gps_only': float(ate_gps),
            'improvement_percent': float(improvement)
        }
        results['noise_evaluations'].append(noise_result)
        
        print(f"\n{'='*70}")
        print(f"RESULTADOS - Ruido GPS: {noise_std}m")
        print(f"{'='*70}")
        print(f" ATE (RL-ORB-GPS):     {ate:.3f} m")
        print(f" ATE (GPS solo):       {ate_gps:.3f} m")
        print(f" Mejora:               {improvement:.1f}%")
        print(f" RPE translación:      {rpe_trans:.3f} m")
        print(f" RPE rotación:         {rpe_rot:.3f}°")
        print(f"{'='*70}")
        
        # Guardar gráfica comparativa
        save_comparison_plot(gt_aligned, est_aligned, gps_positions, noise_std)
    
    # Guardar resultados generales
    save_results(results)
    
    return results

def save_comparison_plot(gt_poses, estimated_poses, gps_poses, noise_std):
    """Guarda gráfica comparativa de trayectorias"""
    plt.figure(figsize=(12, 8))
    
    plt.plot(gt_poses[:, 0], gt_poses[:, 2], 'g-', linewidth=2, label='Ground Truth')
    plt.plot(estimated_poses[:, 0], estimated_poses[:, 2], 'b-', linewidth=1.5, label='RL-ORB-GPS')
    plt.plot(gps_poses[:, 0], gps_poses[:, 2], 'r--', linewidth=1, alpha=0.7, label='GPS Solo')
    
    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Z (m)', fontsize=12)
    plt.title(f'Comparación de Trayectorias - Ruido GPS: {noise_std}m', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Guardar
    timestamp = datetime.now().strftime("%H%M_%d%m_%Y")
    output_dir = f"resultados/LMS_RL_ORB_GPS/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'evaluation_noise_{noise_std}m.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"OK - Gráfica guardada: {output_path}")

def save_results(results):
    """Guarda resultados en JSON y genera reporte"""
    timestamp = datetime.now().strftime("%H%M_%d%m_%Y")
    output_dir = f"resultados/LMS_RL_ORB_GPS/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar JSON
    json_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, indent=2, fp=f)
    print(f"\nOK - Resultados guardados: {json_path}")
    
    # Generar reporte de texto
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("REPORTE DE EVALUACIÓN - RL-ORB-SLAM-GPS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Secuencia: {results['sequence']}\n")
        f.write(f"Modelo: {results['model']}\n")
        f.write(f"Frames procesados: {results['total_frames']}\n\n")
        
        f.write("RESULTADOS POR NIVEL DE RUIDO GPS:\n")
        f.write("-"*70 + "\n\n")
        
        for eval_result in results['noise_evaluations']:
            f.write(f"Ruido GPS: {eval_result['noise_std']} metros (σ)\n")
            f.write(f"  ATE (RL-ORB-GPS):    {eval_result['ate']:.3f} m\n")
            f.write(f"  ATE (GPS solo):      {eval_result['ate_gps_only']:.3f} m\n")
            f.write(f"  Mejora:              {eval_result['improvement_percent']:.1f}%\n")
            f.write(f"  RPE translación:     {eval_result['rpe_trans']:.3f} m\n")
            f.write(f"  RPE rotación:        {eval_result['rpe_rot']:.3f}°\n")
            f.write("\n")
    
    print(f"OK - Reporte guardado: {report_path}")
    
    # Generar gráfica de mejora vs ruido
    plot_improvement_curve(results, output_dir)

def plot_improvement_curve(results, output_dir):
    """Genera gráfica de mejora vs nivel de ruido"""
    noise_levels = [e['noise_std'] for e in results['noise_evaluations']]
    improvements = [e['improvement_percent'] for e in results['noise_evaluations']]
    ate_values = [e['ate'] for e in results['noise_evaluations']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica 1: Mejora vs Ruido
    ax1.plot(noise_levels, improvements, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Ruido GPS (σ en metros)', fontsize=12)
    ax1.set_ylabel('Mejora respecto a GPS solo (%)', fontsize=12)
    ax1.set_title('Mejora del Modelo RL-ORB-GPS', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Gráfica 2: ATE vs Ruido
    ax2.plot(noise_levels, ate_values, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Ruido GPS (σ en metros)', fontsize=12)
    ax2.set_ylabel('ATE (metros)', fontsize=12)
    ax2.set_title('Precisión del Modelo', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'improvement_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"OK - Análisis de mejora guardado: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Evaluar modelo RL-ORB-SLAM-GPS entrenado'
    )
    parser.add_argument(
        '--sequence',
        type=str,
        default='kitti_data/2011_09_26/2011_09_26_drive_0001_sync',
        help='Ruta a secuencia KITTI para evaluación'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth',
        help='Ruta al modelo entrenado'
    )
    parser.add_argument(
        '--noise_levels',
        type=float,
        nargs='+',
        default=[0.0, 3.0, 5.0, 10.0],
        help='Niveles de ruido GPS a evaluar (en metros)'
    )
    
    args = parser.parse_args()
    
    results = evaluate_model(
        sequence_path=args.sequence,
        model_path=args.model,
        noise_levels=args.noise_levels
    )
    
    if results:
        print("\n" + "="*70)
        print("OK - EVALUACIÓN COMPLETADA")
        print("="*70)
        print("\nResultados guardados en: resultados/LMS_RL_ORB_GPS/")
        print("\nArchivos generados:")
        print("  - evaluation_results.json (métricas detalladas)")
        print("  - evaluation_report.txt (reporte legible)")
        print("  - evaluation_noise_*.png (gráficas comparativas)")
        print("  - improvement_analysis.png (análisis de mejora)")
    else:
        print("\nERROR - La evaluación falló")
        sys.exit(1)

if __name__ == "__main__":
    main()
