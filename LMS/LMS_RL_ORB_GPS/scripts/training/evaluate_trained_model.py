#!/usr/bin/env python3
"""
Script de Evaluación Completa del Modelo RL Entrenado

Evalúa el modelo recién entrenado en una secuencia NUEVA y genera:
- Comparación: GPS puro vs SLAM puro vs Fusión RL
- Métricas: ATE, RPE para cada método
- Gráficas comparativas de trayectorias
- Análisis de pesos del RL (cómo decidió en cada frame)
- Reporte detallado en texto y JSON

Uso:
    # Evaluar en secuencia 0013 (no vista en entrenamiento)
    python evaluate_trained_model.py --sequence 0013
    
    # Evaluar con ruido GPS específico
    python evaluate_trained_model.py --sequence 0018 --gps_noise 5.0
    
    # Evaluar modelo específico
    python evaluate_trained_model.py --model_path checkpoints/checkpoint_epoch2_seq0009.pth

Autor: Sistema Híbrido SLAM-GPS
Fecha: Octubre 2025
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Agregar paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from LMS.LMS_RL_ORB_GPS.main import RL_ORB_SLAM_GPS, load_kitti_sequence
from LMS.LMS_RL_ORB_GPS.utils.medir_metricas import calculate_ate, calculate_rpe


class ModelEvaluator:
    """Evaluador completo del modelo RL entrenado"""
    
    def __init__(self, model_path, sequence_path, gps_noise_std=3.0):
        """
        Args:
            model_path: Ruta al modelo entrenado
            sequence_path: Ruta a secuencia KITTI para evaluación
            gps_noise_std: Desviación estándar del ruido GPS (metros)
        """
        self.model_path = model_path
        self.sequence_path = sequence_path
        self.gps_noise_std = gps_noise_std
        
        # Resultados
        self.results = {
            'model': model_path,
            'sequence': sequence_path,
            'gps_noise_std': gps_noise_std,
            'timestamp': datetime.now().isoformat()
        }
        
        # Trayectorias
        self.traj_rl_fusion = []  # Fusión RL
        self.traj_gps_only = []   # GPS puro
        self.traj_slam_only = []  # SLAM puro
        self.gt_trajectory = []   # Ground truth
        
        # Análisis de pesos RL
        self.rl_weights_history = []  # Historial de pesos [gps_weight, slam_weight]
        self.confidence_history = []  # [gps_conf, slam_conf]
        
        # Configurar output
        self.setup_output_dir()
    
    def setup_output_dir(self):
        """Crea directorio para resultados"""
        timestamp = datetime.now().strftime("%H%M_%d%m_%Y")
        seq_name = os.path.basename(self.sequence_path)
        self.output_dir = f"resultados/LMS_RL_ORB_GPS/evaluation_{seq_name}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📂 Resultados se guardarán en: {self.output_dir}")
    
    def load_ground_truth(self):
        """Carga ground truth de KITTI (poses GPS)"""
        from LMS.LMS_RL_ORB_GPS.utils.gps_utils import latlon_to_utm
        
        oxts_dir = os.path.join(self.sequence_path, 'oxts', 'data')
        if not os.path.exists(oxts_dir):
            print(f"⚠️ Advertencia: No se encontró ground truth OXTS")
            return None
        
        oxts_files = sorted([f for f in os.listdir(oxts_dir) if f.endswith('.txt')])
        
        poses = []
        for oxts_file in oxts_files:
            oxts_path = os.path.join(oxts_dir, oxts_file)
            with open(oxts_path, 'r') as f:
                data = f.readline().strip().split()
                if len(data) >= 3:
                    lat, lon, alt = float(data[0]), float(data[1]), float(data[2])
                    utm_pos = latlon_to_utm(lat, lon, alt)
                    poses.append(utm_pos)
        
        poses = np.array(poses)
        
        # Normalizar al origen
        if len(poses) > 0:
            poses = poses - poses[0]
        
        self.gt_trajectory = poses
        print(f"✅ Ground truth cargado: {len(poses)} poses")
        return poses
    
    def evaluate(self, max_frames=None):
        """
        Ejecuta evaluación completa del modelo
        
        Args:
            max_frames: Límite de frames a procesar (None = todos)
        
        Returns:
            dict: Resultados completos de la evaluación
        """
        print("\n" + "="*80)
        print("🔍 EVALUACIÓN COMPLETA DEL MODELO RL-ORB-SLAM-GPS")
        print("="*80)
        print(f"📦 Modelo: {self.model_path}")
        print(f"📍 Secuencia: {self.sequence_path}")
        print(f"📡 Ruido GPS: {self.gps_noise_std}m (σ)")
        print(f"🎯 Frames: {max_frames or 'Todos'}")
        print("="*80 + "\n")
        
        # Cargar ground truth
        self.load_ground_truth()
        
        # Cargar secuencia
        print("📥 Cargando secuencia KITTI...")
        frames, gps_data = load_kitti_sequence(self.sequence_path, max_frames)
        print(f"✅ Cargados {len(frames)} frames\n")
        
        # === MÉTODO 1: FUSIÓN RL (modelo entrenado) ===
        print("="*80)
        print("🤖 EVALUANDO: Fusión RL (Modelo Entrenado)")
        print("="*80)
        
        slam_rl = RL_ORB_SLAM_GPS(
            fx=718.856, fy=718.856, cx=607.1928, cy=185.2157,
            training_mode=False,  # Inferencia
            model_path=self.model_path,
            simulate_mobile_gps=True,
            gps_noise_std=self.gps_noise_std
        )
        
        for i, frame in enumerate(frames):
            if i % 20 == 0:
                print(f"  Frame {i}/{len(frames)}...")
            
            if frame is None:
                continue
            
            # GPS con ruido
            gps_utm = None
            gps_conf = 0.1
            if gps_data[i] is not None:
                gps_utm = slam_rl.gps_frame_reference(gps_data[i])
                slam_rl.gps_filter.add_measurement(gps_utm)
                gps_conf = slam_rl.gps_filter.calculate_confidence()
            
            # Procesar frame
            slam_rl.process_frame_with_gps(frame, gps_utm, gps_conf)
            
            # Guardar pesos RL para análisis
            if hasattr(slam_rl, 'last_rl_weights'):
                self.rl_weights_history.append(slam_rl.last_rl_weights)
                self.confidence_history.append([gps_conf, 0.5])  # Placeholder visual_conf
        
        self.traj_rl_fusion = slam_rl.optimize_pose_graph()
        print(f"✅ Fusión RL completada: {len(self.traj_rl_fusion)} poses\n")
        
        # === MÉTODO 2: GPS PURO (sin SLAM) ===
        print("="*80)
        print("📡 EVALUANDO: GPS Puro (sin corrección SLAM)")
        print("="*80)
        
        for i, gps in enumerate(gps_data):
            if i % 20 == 0:
                print(f"  Frame {i}/{len(gps_data)}...")
            
            if gps is not None:
                gps_utm = slam_rl.gps_frame_reference(gps)
                # Agregar ruido simulado
                noise = np.random.normal(0, self.gps_noise_std, 3)
                gps_noisy = gps_utm + noise
                
                # Crear pose 4x4
                pose = np.eye(4)
                pose[:3, 3] = gps_noisy
                self.traj_gps_only.append(pose)
        
        print(f"✅ GPS puro completado: {len(self.traj_gps_only)} poses\n")
        
        # === MÉTODO 3: SLAM PURO (sin GPS) ===
        print("="*80)
        print("👁️ EVALUANDO: SLAM Puro (sin GPS)")
        print("="*80)
        
        slam_only = RL_ORB_SLAM_GPS(
            fx=718.856, fy=718.856, cx=607.1928, cy=185.2157,
            training_mode=False,
            model_path=None,  # Sin modelo RL
            simulate_mobile_gps=False
        )
        
        for i, frame in enumerate(frames):
            if i % 20 == 0:
                print(f"  Frame {i}/{len(frames)}...")
            
            if frame is None:
                continue
            
            # Procesar solo con SLAM (sin GPS)
            slam_only.process_frame_with_gps(frame, None, 0.0)
        
        self.traj_slam_only = slam_only.optimize_pose_graph()
        print(f"✅ SLAM puro completado: {len(self.traj_slam_only)} poses\n")
        
        # === CALCULAR MÉTRICAS ===
        print("="*80)
        print("📊 CALCULANDO MÉTRICAS")
        print("="*80)
        self.calculate_metrics()
        
        # === GENERAR VISUALIZACIONES ===
        print("\n" + "="*80)
        print("📈 GENERANDO VISUALIZACIONES")
        print("="*80)
        self.plot_trajectories()
        self.plot_rl_weights_analysis()
        self.plot_error_analysis()
        
        # === GUARDAR RESULTADOS ===
        print("\n" + "="*80)
        print("💾 GUARDANDO RESULTADOS")
        print("="*80)
        self.save_results()
        
        return self.results
    
    def calculate_metrics(self):
        """Calcula métricas ATE y RPE para cada método"""
        
        # Extraer posiciones
        def extract_positions(trajectory):
            if len(trajectory) == 0:
                return np.array([])
            return np.array([pose[:3, 3] for pose in trajectory])
        
        pos_rl = extract_positions(self.traj_rl_fusion)
        pos_gps = extract_positions(self.traj_gps_only)
        pos_slam = extract_positions(self.traj_slam_only)
        pos_gt = self.gt_trajectory
        
        # Alinear tamaños
        min_len = min(len(pos_gt), len(pos_rl), len(pos_gps), len(pos_slam))
        
        if min_len == 0:
            print("⚠️ Error: No hay suficientes datos para calcular métricas")
            return
        
        pos_gt = pos_gt[:min_len]
        pos_rl = pos_rl[:min_len]
        pos_gps = pos_gps[:min_len]
        pos_slam = pos_slam[:min_len]
        
        # Calcular ATE y RPE para cada método
        metrics = {}
        
        # Fusión RL
        ate_rl = calculate_ate(pos_gt, pos_rl)
        rpe_rl_t, rpe_rl_r = calculate_rpe(pos_gt, pos_rl)
        metrics['rl_fusion'] = {
            'ate': float(ate_rl),
            'rpe_translation': float(rpe_rl_t),
            'rpe_rotation': float(rpe_rl_r)
        }
        
        # GPS puro
        ate_gps = calculate_ate(pos_gt, pos_gps)
        rpe_gps_t, rpe_gps_r = calculate_rpe(pos_gt, pos_gps)
        metrics['gps_only'] = {
            'ate': float(ate_gps),
            'rpe_translation': float(rpe_gps_t),
            'rpe_rotation': float(rpe_gps_r)
        }
        
        # SLAM puro
        ate_slam = calculate_ate(pos_gt, pos_slam)
        rpe_slam_t, rpe_slam_r = calculate_rpe(pos_gt, pos_slam)
        metrics['slam_only'] = {
            'ate': float(ate_slam),
            'rpe_translation': float(rpe_slam_t),
            'rpe_rotation': float(rpe_slam_r)
        }
        
        # Calcular mejoras
        metrics['improvements'] = {
            'vs_gps': ((ate_gps - ate_rl) / ate_gps * 100) if ate_gps > 0 else 0,
            'vs_slam': ((ate_slam - ate_rl) / ate_slam * 100) if ate_slam > 0 else 0
        }
        
        self.results['metrics'] = metrics
        
        # Imprimir tabla comparativa
        print("\n📊 MÉTRICAS COMPARATIVAS:")
        print("-" * 80)
        print(f"{'Método':<20} {'ATE (m)':<15} {'RPE Trans (m)':<15} {'RPE Rot (°)':<15}")
        print("-" * 80)
        print(f"{'Fusión RL':<20} {ate_rl:<15.3f} {rpe_rl_t:<15.3f} {rpe_rl_r:<15.3f}")
        print(f"{'GPS Puro':<20} {ate_gps:<15.3f} {rpe_gps_t:<15.3f} {rpe_gps_r:<15.3f}")
        print(f"{'SLAM Puro':<20} {ate_slam:<15.3f} {rpe_slam_t:<15.3f} {rpe_slam_r:<15.3f}")
        print("-" * 80)
        print(f"{'MEJORA vs GPS:':<20} {metrics['improvements']['vs_gps']:>14.1f}%")
        print(f"{'MEJORA vs SLAM:':<20} {metrics['improvements']['vs_slam']:>14.1f}%")
        print("-" * 80)
    
    def plot_trajectories(self):
        """Genera gráfica comparativa de trayectorias"""
        
        def extract_positions(trajectory):
            if len(trajectory) == 0:
                return np.array([]), np.array([])
            positions = np.array([pose[:3, 3] for pose in trajectory])
            return positions[:, 0], positions[:, 2]  # X, Z
        
        # Extraer posiciones
        gt_x, gt_z = self.gt_trajectory[:, 0], self.gt_trajectory[:, 2]
        rl_x, rl_z = extract_positions(self.traj_rl_fusion)
        gps_x, gps_z = extract_positions(self.traj_gps_only)
        slam_x, slam_z = extract_positions(self.traj_slam_only)
        
        # Crear figura
        plt.figure(figsize=(14, 10))
        
        plt.plot(gt_x, gt_z, 'g-', linewidth=3, label='Ground Truth', alpha=0.8)
        plt.plot(rl_x, rl_z, 'b-', linewidth=2, label='Fusión RL (Modelo Entrenado)', alpha=0.9)
        plt.plot(gps_x, gps_z, 'r--', linewidth=1.5, label=f'GPS Puro (ruido {self.gps_noise_std}m)', alpha=0.6)
        plt.plot(slam_x, slam_z, 'orange', linestyle=':', linewidth=1.5, label='SLAM Puro', alpha=0.7)
        
        # Marcar inicio y fin
        plt.plot(gt_x[0], gt_z[0], 'go', markersize=12, label='Inicio', zorder=5)
        plt.plot(gt_x[-1], gt_z[-1], 'rs', markersize=12, label='Fin', zorder=5)
        
        plt.xlabel('X (metros)', fontsize=14)
        plt.ylabel('Z (metros)', fontsize=14)
        plt.title('Comparación de Trayectorias: RL vs GPS vs SLAM', fontsize=16, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Guardar
        output_path = os.path.join(self.output_dir, 'comparison_trajectories.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Trayectorias guardadas: {output_path}")
    
    def plot_rl_weights_analysis(self):
        """Analiza cómo el RL ajustó los pesos durante la evaluación"""
        
        if len(self.rl_weights_history) == 0:
            print("⚠️ No hay historial de pesos RL disponible")
            return
        
        weights = np.array(self.rl_weights_history)
        frames = np.arange(len(weights))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Gráfica 1: Evolución de pesos
        ax1.plot(frames, weights[:, 0], 'b-', linewidth=2, label='Peso GPS', alpha=0.8)
        ax1.plot(frames, weights[:, 1], 'orange', linewidth=2, label='Peso SLAM', alpha=0.8)
        ax1.set_xlabel('Frame', fontsize=12)
        ax1.set_ylabel('Peso', fontsize=12)
        ax1.set_title('Evolución de Pesos RL Durante Evaluación', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Gráfica 2: Distribución de pesos
        ax2.hist(weights[:, 0], bins=30, alpha=0.6, color='blue', label='Peso GPS', edgecolor='black')
        ax2.hist(weights[:, 1], bins=30, alpha=0.6, color='orange', label='Peso SLAM', edgecolor='black')
        ax2.set_xlabel('Valor del Peso', fontsize=12)
        ax2.set_ylabel('Frecuencia', fontsize=12)
        ax2.set_title('Distribución de Pesos RL', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Guardar
        output_path = os.path.join(self.output_dir, 'rl_weights_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Estadísticas
        print(f"✅ Análisis de pesos RL guardado: {output_path}")
        print(f"   📊 Peso GPS promedio: {weights[:, 0].mean():.3f} ± {weights[:, 0].std():.3f}")
        print(f"   📊 Peso SLAM promedio: {weights[:, 1].mean():.3f} ± {weights[:, 1].std():.3f}")
    
    def plot_error_analysis(self):
        """Analiza error a lo largo de la trayectoria"""
        
        def extract_positions(trajectory):
            if len(trajectory) == 0:
                return np.array([])
            return np.array([pose[:3, 3] for pose in trajectory])
        
        pos_gt = self.gt_trajectory
        pos_rl = extract_positions(self.traj_rl_fusion)
        pos_gps = extract_positions(self.traj_gps_only)
        pos_slam = extract_positions(self.traj_slam_only)
        
        # Alinear
        min_len = min(len(pos_gt), len(pos_rl), len(pos_gps), len(pos_slam))
        pos_gt = pos_gt[:min_len]
        pos_rl = pos_rl[:min_len]
        pos_gps = pos_gps[:min_len]
        pos_slam = pos_slam[:min_len]
        
        # Calcular errores
        errors_rl = np.linalg.norm(pos_gt - pos_rl, axis=1)
        errors_gps = np.linalg.norm(pos_gt - pos_gps, axis=1)
        errors_slam = np.linalg.norm(pos_gt - pos_slam, axis=1)
        
        frames = np.arange(min_len)
        
        # Crear figura
        plt.figure(figsize=(14, 8))
        
        plt.plot(frames, errors_rl, 'b-', linewidth=2, label='Fusión RL', alpha=0.8)
        plt.plot(frames, errors_gps, 'r--', linewidth=1.5, label='GPS Puro', alpha=0.6)
        plt.plot(frames, errors_slam, 'orange', linestyle=':', linewidth=1.5, label='SLAM Puro', alpha=0.7)
        
        plt.xlabel('Frame', fontsize=14)
        plt.ylabel('Error Euclidiano (metros)', fontsize=14)
        plt.title('Error vs Ground Truth a lo Largo de la Trayectoria', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        
        # Guardar
        output_path = os.path.join(self.output_dir, 'error_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Análisis de error guardado: {output_path}")
    
    def save_results(self):
        """Guarda resultados en JSON y genera reporte de texto"""
        
        # Guardar JSON
        json_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Resultados JSON: {json_path}")
        
        # Generar reporte de texto
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("REPORTE DE EVALUACIÓN - RL-ORB-SLAM-GPS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Modelo: {self.model_path}\n")
            f.write(f"Secuencia: {self.sequence_path}\n")
            f.write(f"Ruido GPS: {self.gps_noise_std}m\n")
            f.write(f"Fecha: {self.results['timestamp']}\n\n")
            
            if 'metrics' in self.results:
                m = self.results['metrics']
                
                f.write("MÉTRICAS COMPARATIVAS:\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Método':<20} {'ATE (m)':<15} {'RPE Trans (m)':<15} {'RPE Rot (°)':<15}\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Fusión RL':<20} {m['rl_fusion']['ate']:<15.3f} {m['rl_fusion']['rpe_translation']:<15.3f} {m['rl_fusion']['rpe_rotation']:<15.3f}\n")
                f.write(f"{'GPS Puro':<20} {m['gps_only']['ate']:<15.3f} {m['gps_only']['rpe_translation']:<15.3f} {m['gps_only']['rpe_rotation']:<15.3f}\n")
                f.write(f"{'SLAM Puro':<20} {m['slam_only']['ate']:<15.3f} {m['slam_only']['rpe_translation']:<15.3f} {m['slam_only']['rpe_rotation']:<15.3f}\n")
                f.write("-"*80 + "\n\n")
                
                f.write("MEJORAS:\n")
                f.write(f"  vs GPS Puro:  {m['improvements']['vs_gps']:>6.1f}%\n")
                f.write(f"  vs SLAM Puro: {m['improvements']['vs_slam']:>6.1f}%\n")
        
        print(f"✅ Reporte de texto: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluar modelo RL entrenado en secuencia NUEVA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Evaluar en secuencia 0013 (no vista en entrenamiento)
  python evaluate_trained_model.py --sequence 0013
  
  # Evaluar con ruido GPS específico
  python evaluate_trained_model.py --sequence 0018 --gps_noise 5.0
  
  # Evaluar modelo de checkpoint específico
  python evaluate_trained_model.py --sequence 0013 \\
      --model_path LMS/LMS_RL_ORB_GPS/model/checkpoints/checkpoint_epoch2_seq0009.pth
        """
    )
    
    parser.add_argument(
        '--sequence',
        type=str,
        required=True,
        help='Número de secuencia KITTI (ej: 0013, 0018)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth',
        help='Ruta al modelo entrenado'
    )
    
    parser.add_argument(
        '--gps_noise',
        type=float,
        default=3.0,
        help='Desviación estándar del ruido GPS en metros (default: 3.0)'
    )
    
    parser.add_argument(
        '--max_frames',
        type=int,
        default=None,
        help='Límite de frames a evaluar (default: todos)'
    )
    
    parser.add_argument(
        '--kitti_path',
        type=str,
        default='kitti_data/2011_09_26',
        help='Ruta base a secuencias KITTI'
    )
    
    args = parser.parse_args()
    
    # Construir ruta completa a secuencia
    sequence_path = os.path.join(
        args.kitti_path,
        f"2011_09_26_drive_{args.sequence}_sync"
    )
    
    # Verificar que existan
    if not os.path.exists(sequence_path):
        print(f"❌ ERROR: Secuencia no encontrada: {sequence_path}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"❌ ERROR: Modelo no encontrado: {args.model_path}")
        sys.exit(1)
    
    # Crear evaluador y ejecutar
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        sequence_path=sequence_path,
        gps_noise_std=args.gps_noise
    )
    
    results = evaluator.evaluate(max_frames=args.max_frames)
    
    # Resumen final
    print("\n" + "="*80)
    print("🎉 EVALUACIÓN COMPLETADA")
    print("="*80)
    print(f"📂 Resultados guardados en: {evaluator.output_dir}")
    print("\n📊 Archivos generados:")
    print("  ✅ evaluation_results.json      (métricas detalladas)")
    print("  ✅ evaluation_report.txt        (reporte legible)")
    print("  ✅ comparison_trajectories.png  (trayectorias comparadas)")
    print("  ✅ rl_weights_analysis.png      (análisis de pesos RL)")
    print("  ✅ error_analysis.png           (errores a lo largo del tiempo)")
    print("="*80)
    
    if 'metrics' in results:
        m = results['metrics']
        print(f"\n🏆 RESULTADO FINAL:")
        print(f"  Fusión RL ATE:  {m['rl_fusion']['ate']:.3f} m")
        print(f"  Mejora vs GPS:  {m['improvements']['vs_gps']:>6.1f}%")
        print(f"  Mejora vs SLAM: {m['improvements']['vs_slam']:>6.1f}%")


if __name__ == "__main__":
    main()
