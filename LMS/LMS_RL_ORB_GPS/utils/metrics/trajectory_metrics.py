import numpy as np
from typing import List, Dict, Tuple
import csv
import json
import matplotlib.pyplot as plt
from datetime import datetime

class TrajectoryMetrics:
    """
    Calcula métricas de evaluación de trayectorias (ATE, RPE).
    
    Referencias:
    - ATE: Absolute Trajectory Error (error global)
    - RPE: Relative Pose Error (error local frame-a-frame)
    """
    
    @staticmethod
    def align_trajectories(traj_est: np.ndarray, traj_ref: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Alinea dos trayectorias usando transformación rígida (Umeyama alignment).
        
        Args:
            traj_est: Trayectoria estimada (N x 3)
            traj_ref: Trayectoria de referencia (N x 3)
        
        Returns:
            traj_est_aligned: Trayectoria estimada alineada
            scale: Factor de escala aplicado
        """
        # Asegurar que ambas tengan la misma longitud
        n = min(len(traj_est), len(traj_ref))
        traj_est = traj_est[:n]
        traj_ref = traj_ref[:n]
        
        # Centrar las trayectorias
        mean_est = np.mean(traj_est, axis=0)
        mean_ref = np.mean(traj_ref, axis=0)
        
        traj_est_centered = traj_est - mean_est
        traj_ref_centered = traj_ref - mean_ref
        
        # Calcular matriz de covarianza
        H = traj_est_centered.T @ traj_ref_centered
        
        # SVD para obtener rotación óptima
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Asegurar rotación válida (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Calcular escala óptima (S es un array 1D de valores singulares)
        var_est = np.sum(traj_est_centered**2)
        scale = np.sum(S) / var_est if var_est > 1e-10 else 1.0
        
        # Aplicar transformación
        traj_est_aligned = scale * (traj_est_centered @ R.T) + mean_ref
        
        return traj_est_aligned, scale
    
    @staticmethod
    def calculate_ate(traj_est: np.ndarray, traj_ref: np.ndarray, align: bool = True) -> Dict[str, float]:
        """
        Calcula Absolute Trajectory Error (ATE).
        
        Args:
            traj_est: Trayectoria estimada (N x 3)
            traj_ref: Trayectoria de referencia (N x 3)
            align: Si True, alinea las trayectorias antes de calcular error
        
        Returns:
            dict con estadísticas de ATE incluyendo array de errores
        """
        if len(traj_est) != len(traj_ref):
            min_len = min(len(traj_est), len(traj_ref))
            traj_est = traj_est[:min_len]
            traj_ref = traj_ref[:min_len]
            print(f"   ADVERTENCIA: Trayectorias con longitud diferente. Usando primeros {min_len} puntos.")
        
        if align:
            traj_est_aligned, scale = TrajectoryMetrics.align_trajectories(traj_est, traj_ref)
        else:
            traj_est_aligned = traj_est
            scale = 1.0
        
        # Calcular errores punto a punto
        errors = np.linalg.norm(traj_est_aligned - traj_ref, axis=1)
        
        ate_stats = {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'median': float(np.median(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'scale': float(scale),
            'num_points': int(len(errors)),
            'errors': errors  # Agregar array de errores para gráficos
        }
        
        return ate_stats
    
    @staticmethod
    def calculate_rpe(traj_est: np.ndarray, traj_ref: np.ndarray, delta: int = 1) -> Dict[str, float]:
        """
        Calcula Relative Pose Error (RPE) para medir precisión local.
        
        Args:
            traj_est: Trayectoria estimada (N x 3)
            traj_ref: Trayectoria de referencia (N x 3)
            delta: Distancia entre frames para calcular RPE (default: 1 = consecutivos)
        
        Returns:
            dict con estadísticas de RPE incluyendo array de errores
        """
        n = min(len(traj_est), len(traj_ref))
        
        trans_errors = []
        
        for i in range(n - delta):
            # Movimiento estimado
            delta_est = traj_est[i + delta] - traj_est[i]
            
            # Movimiento de referencia
            delta_ref = traj_ref[i + delta] - traj_ref[i]
            
            # Error en el movimiento
            trans_error = np.linalg.norm(delta_est - delta_ref)
            trans_errors.append(trans_error)
        
        trans_errors = np.array(trans_errors)
        
        rpe_stats = {
            'trans_mean': float(np.mean(trans_errors)),
            'trans_std': float(np.std(trans_errors)),
            'trans_median': float(np.median(trans_errors)),
            'trans_rmse': float(np.sqrt(np.mean(trans_errors**2))),
            'num_pairs': int(len(trans_errors)),
            'delta': delta,
            'errors': trans_errors  # Agregar array de errores para gráficos
        }
        
        return rpe_stats
    
    @staticmethod
    def compare_trajectories(pipeline_traj: np.ndarray, 
                            slam_traj: np.ndarray, 
                            gps_traj: np.ndarray,
                            reference: str = 'gps') -> Dict[str, Dict[str, float]]:
        """
        Compara tres trayectorias: Pipeline fusionado, SLAM puro, GPS puro.
        
        Args:
            pipeline_traj: Trayectoria del pipeline (SLAM+GPS+RL)
            slam_traj: Trayectoria de SLAM puro
            gps_traj: Trayectoria de GPS puro
            reference: Qué usar como referencia ('gps', 'slam', o 'pipeline')
        
        Returns:
            dict con métricas comparativas
        """
        if reference == 'gps':
            ref_traj = gps_traj
            ref_name = "GPS puro (pseudo-ground truth)"
        elif reference == 'slam':
            ref_traj = slam_traj
            ref_name = "SLAM puro (referencia más precisa)"
        elif reference == 'pipeline':
            ref_traj = pipeline_traj
            ref_name = "Pipeline fusionado (referencia)"
        else:
            raise ValueError("reference debe ser 'gps', 'slam' o 'pipeline'")
        
        print(f"\n{'='*60}")
        print(f"COMPARACION DE TRAYECTORIAS")
        print(f"Referencia: {ref_name}")
        print(f"{'='*60}\n")
        
        results = {}
        
        # ATE del Pipeline vs Referencia
        if reference != 'pipeline':
            print("Calculando ATE: Pipeline fusionado vs Referencia...")
            ate_pipeline = TrajectoryMetrics.calculate_ate(pipeline_traj, ref_traj, align=True)
            results['pipeline_vs_ref'] = {
                'method': 'Pipeline (SLAM+GPS+RL)',
                'ate': ate_pipeline,
                'rpe': TrajectoryMetrics.calculate_rpe(pipeline_traj, ref_traj, delta=1)
            }
            print(f"   ATE RMSE: {ate_pipeline['rmse']:.3f}m")
        
        # ATE del SLAM puro vs Referencia (solo si SLAM no es la referencia)
        if reference != 'slam':
            print("\nCalculando ATE: SLAM puro vs Referencia...")
            ate_slam = TrajectoryMetrics.calculate_ate(slam_traj, ref_traj, align=True)
            results['slam_vs_ref'] = {
                'method': 'SLAM puro',
                'ate': ate_slam,
                'rpe': TrajectoryMetrics.calculate_rpe(slam_traj, ref_traj, delta=1)
            }
            print(f"   ATE RMSE: {ate_slam['rmse']:.3f}m")
        
        # ATE del GPS puro vs Referencia (si no es la referencia)
        if reference != 'gps':
            print("\nCalculando ATE: GPS puro vs Referencia...")
            ate_gps = TrajectoryMetrics.calculate_ate(gps_traj, ref_traj, align=True)
            results['gps_vs_ref'] = {
                'method': 'GPS puro',
                'ate': ate_gps,
                'rpe': TrajectoryMetrics.calculate_rpe(gps_traj, ref_traj, delta=1)
            }
            print(f"   ATE RMSE: {ate_gps['rmse']:.3f}m")
        
        # Comparación directa: Pipeline vs SLAM puro (siempre útil)
        if reference != 'slam':  # Solo si SLAM no es la referencia
            print("\nCalculando diferencia: Pipeline vs SLAM puro...")
            ate_diff = TrajectoryMetrics.calculate_ate(pipeline_traj, slam_traj, align=True)
            results['pipeline_vs_slam'] = {
                'method': 'Pipeline vs SLAM (diferencia directa)',
                'ate': ate_diff,
                'rpe': TrajectoryMetrics.calculate_rpe(pipeline_traj, slam_traj, delta=1)
            }
            print(f"   ATE RMSE: {ate_diff['rmse']:.3f}m")
        
        return results
    
    @staticmethod
    def print_comparison_table(results: Dict[str, Dict[str, float]]):
        """
        Imprime tabla comparativa de métricas adaptada según la referencia.
        """
        print(f"\n{'='*85}")
        print(f"{'TABLA COMPARATIVA DE METRICAS ATE/RPE':^85}")
        print(f"{'='*85}")
        
        # Detectar tipo de referencia
        reference_method = results.get('reference_method', 'unknown')
        
        # Extraer métrica del pipeline (siempre presente)
        pipeline_data = results.get('pipeline_vs_ref', {})
        pipeline_ate = pipeline_data.get('ate', {}).get('rmse', 0)
        
        # ====================================================================
        # FIX: Manejo diferente según tipo de referencia
        # ====================================================================
        if reference_method == 'gps':
            # KITTI: Mostrar tabla con columna "Mejora %"
            print(f"{'Metodo':<35} {'ATE Mean':<12} {'ATE RMSE':<12} {'RPE Mean':<12} {'Mejora':<12}")
            print(f"{'-'*85}")
            
            baseline_data = results.get('slam_vs_ref', {})
            baseline_name = "SLAM puro (sin GPS)"
            baseline_ate = baseline_data.get('ate', {}).get('rmse', 0)
            
            # Calcular mejora porcentual
            if baseline_ate > 0:
                improvement = ((baseline_ate - pipeline_ate) / baseline_ate) * 100
                mejora_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
            else:
                mejora_str = "N/A"
            
            # FILA 1: Pipeline
            pipeline_mean = pipeline_data.get('ate', {}).get('mean', 0)
            pipeline_rpe = pipeline_data.get('rpe', {}).get('trans_mean', 0)
            print(f"{'Pipeline (SLAM+GPS+RL)':<35} {pipeline_mean:>10.3f}m {pipeline_ate:>10.3f}m {pipeline_rpe:>10.3f}m  {mejora_str}")
            
            # FILA 2: Baseline (SLAM puro)
            baseline_mean = baseline_data.get('ate', {}).get('mean', 0)
            baseline_rpe = baseline_data.get('rpe', {}).get('trans_mean', 0)
            print(f"{baseline_name:<35} {baseline_mean:>10.3f}m {baseline_ate:>10.3f}m {baseline_rpe:>10.3f}m  -")
            
            # FILA 3: Diferencia directa Pipeline vs SLAM (si existe)
            if 'pipeline_vs_slam' in results:
                diff_data = results['pipeline_vs_slam']
                diff_mean = diff_data.get('ate', {}).get('mean', 0)
                diff_ate = diff_data.get('ate', {}).get('rmse', 0)
                diff_rpe = diff_data.get('rpe', {}).get('trans_mean', 0)
                
                if baseline_ate > 0:
                    diff_improvement = ((baseline_ate - diff_ate) / baseline_ate) * 100
                    diff_mejora_str = f"+{diff_improvement:.1f}%" if diff_improvement > 0 else f"{diff_improvement:.1f}%"
                else:
                    diff_mejora_str = "N/A"
                
                print(f"{'Pipeline vs SLAM (diferencia directa)':<35} {diff_mean:>10.3f}m  {diff_ate:>10.3f}m  {diff_rpe:>10.3f}m  {diff_mejora_str}")
            
        else:
            # ====================================================================
            # MÓVIL: Tabla mejorada sin 0.000m confusos
            # ====================================================================
            print(f"{'Metodo':<40} {'Error vs SLAM':<15} {'Interpretación':<20}")
            print(f"{'-'*85}")
            
            # FILA 1: Pipeline (degradación)
            pipeline_mean = pipeline_data.get('ate', {}).get('mean', 0)
            pipeline_rpe = pipeline_data.get('rpe', {}).get('trans_mean', 0)
            
            # Clasificar degradación
            if pipeline_ate < 1.0:
                interpretacion = "EXCELENTE ✓"
            elif pipeline_ate < 1.5:
                interpretacion = "BUENO ✓"
            elif pipeline_ate < 3.0:
                interpretacion = "MODERADO"
            else:
                interpretacion = "ALTO ✗"
            
            print(f"{'Pipeline (SLAM+GPS+RL)':<40} {pipeline_ate:>8.3f}m       {interpretacion:<20}")
            print(f"{'  └─ ATE Mean:':<40} {pipeline_mean:>8.3f}m")
            print(f"{'  └─ RPE Mean:':<40} {pipeline_rpe:>8.3f}m")
            print()
            print(f"{'SLAM puro (REFERENCIA PERFECTA)':<40} {'0.000m':>8}       {'(por definición)':<20}")
            print()
            
            # FILA 2: GPS para contexto (si existe)
            if 'gps_vs_ref' in results:
                gps_data = results['gps_vs_ref']
                gps_ate = gps_data.get('ate', {}).get('rmse', 0)
                gps_mean = gps_data.get('ate', {}).get('mean', 0)
                
                print(f"{'GPS consumer (PARA CONTEXTO)':<40} {gps_ate:>8.3f}m       {'ruidoso (esperado)':<20}")
                print(f"{'  └─ ATE Mean:':<40} {gps_mean:>8.3f}m")
        
        print(f"{'='*85}\n")
        
        # NOTA EXPLICATIVA ADAPTADA
        if reference_method == 'gps':
            print(f"Referencia: GPS Ground Truth")
            baseline_ate = results.get('slam_vs_ref', {}).get('ate', {}).get('rmse', 0)
            baseline_name = "SLAM puro (sin GPS)"
            print(f"Baseline: {baseline_name} (error vs GPS = {baseline_ate:.3f}m)")
            print(f"Mejora % = ((RMSE_SLAM - RMSE_Pipeline) / RMSE_SLAM) × 100")
            print(f"   + = Pipeline más cercano a GPS | - = Pipeline más alejado de GPS")
        else:
            print(f"Referencia: SLAM puro (0.000m por definición - es la referencia perfecta)")
            print(f"Degradación Pipeline = {pipeline_ate:.3f}m")
            print(f"   Objetivo: < 1.5m (EXCELENTE/BUENO) para preservar precisión visual")
            gps_error = results.get('gps_vs_ref', {}).get('ate', {}).get('rmse', 0)
            if gps_error > 0:
                print(f"   GPS consumer: {gps_error:.1f}m de error (por eso NO es ground truth)")
        
        print()
    
    @staticmethod
    def save_metrics_to_csv(results: Dict[str, Dict[str, float]], output_path: str):
        """
        Guarda métricas en archivo CSV.
        """
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Metodo', 'ATE_mean_m', 'ATE_std_m', 'ATE_median_m', 'ATE_max_m', 
                           'ATE_RMSE_m', 'RPE_mean_m', 'RPE_std_m', 'RPE_RMSE_m', 'Scale', 'Num_Points'])
            
            # Filtrar solo los resultados de métricas (ignorar metadatos)
            metrics_only = {k: v for k, v in results.items() 
                           if isinstance(v, dict) and 'method' in v and 'ate' in v}
            
            # Datos
            for key, data in metrics_only.items():
                writer.writerow([
                    data['method'],
                    f"{data['ate']['mean']:.4f}",
                    f"{data['ate']['std']:.4f}",
                    f"{data['ate']['median']:.4f}",
                    f"{data['ate']['max']:.4f}",
                    f"{data['ate']['rmse']:.4f}",
                    f"{data['rpe']['trans_mean']:.4f}",
                    f"{data['rpe']['trans_std']:.4f}",
                    f"{data['rpe']['trans_rmse']:.4f}",
                    f"{data['ate']['scale']:.4f}",
                    data['ate']['num_points']
                ])
        
        print(f"Metricas ATE/RPE guardadas en: {output_path}\n")
    
    @staticmethod
    def print_metrics_summary(ate_result: Dict, rpe_result: Dict):
        """
        Imprime un resumen legible de las métricas ATE y RPE.
        
        Args:
            ate_result: Diccionario con resultados de ATE
            rpe_result: Diccionario con resultados de RPE
        """
        print("\n" + "="*60)
        print("RESUMEN DE MÉTRICAS DE TRAYECTORIA")
        print("="*60)
        
        if ate_result:
            print("\nAbsolute Trajectory Error (ATE):")
            print(f"  RMSE:       {ate_result.get('rmse', 0):.4f} m")
            print(f"  Mean:       {ate_result.get('mean', 0):.4f} m")
            print(f"  Median:     {ate_result.get('median', 0):.4f} m")
            print(f"  Std Dev:    {ate_result.get('std', 0):.4f} m")
            print(f"  Min:        {ate_result.get('min', 0):.4f} m")
            print(f"  Max:        {ate_result.get('max', 0):.4f} m")
            print(f"  Scale:      {ate_result.get('scale', 1.0):.4f}")
            print(f"  Num Points: {ate_result.get('num_points', 0)}")
        
        if rpe_result:
            print("\nRelative Pose Error (RPE) - Traslación:")
            print(f"  RMSE:       {rpe_result.get('trans_rmse', 0):.4f} m")
            print(f"  Mean:       {rpe_result.get('trans_mean', 0):.4f} m")
            print(f"  Median:     {rpe_result.get('trans_median', 0):.4f} m")
            print(f"  Std Dev:    {rpe_result.get('trans_std', 0):.4f} m")
            print(f"  Num Pairs:  {rpe_result.get('num_pairs', 0)}")
            print(f"  Delta:      {rpe_result.get('delta', 1)}")
        
        print("="*60 + "\n")
    
    @staticmethod
    def save_metrics_to_json(ate_result: Dict, rpe_result: Dict, rpe_rot_result: Dict, output_file: str):
        """
        Guarda las métricas en un archivo JSON.
        
        Args:
            ate_result: Diccionario con resultados de ATE
            rpe_result: Diccionario con resultados de RPE translación
            rpe_rot_result: Diccionario con resultados de RPE rotación (puede ser None)
            output_file: Ruta del archivo JSON de salida
        """
        metrics_dict = {
            'timestamp': datetime.now().isoformat(),
            'ate': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in ate_result.items() if k != 'errors'},
            'rpe_translation': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in rpe_result.items() if k != 'errors'} if rpe_result else None,
            'rpe_rotation': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                            for k, v in rpe_rot_result.items() if k != 'errors'} if rpe_rot_result else None
        }
        
        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        
        print(f"Métricas guardadas en JSON: {output_file}")
    
    @staticmethod
    def plot_ate_errors(errors: np.ndarray, output_file: str):
        """
        Genera un gráfico de los errores ATE a lo largo de la trayectoria.
        
        Args:
            errors: Array con errores ATE por keyframe
            output_file: Ruta del archivo PNG de salida
        """
        if len(errors) == 0:
            print("No hay errores ATE para graficar")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(errors, 'b-', linewidth=1.5, label='Error ATE')
        plt.axhline(y=np.mean(errors), color='r', linestyle='--', 
                    label=f'Media: {np.mean(errors):.3f} m', linewidth=2)
        plt.axhline(y=np.median(errors), color='g', linestyle=':', 
                    label=f'Mediana: {np.median(errors):.3f} m', linewidth=2)
        
        plt.xlabel('Keyframe Index', fontsize=12)
        plt.ylabel('Error Absoluto (m)', fontsize=12)
        plt.title('Absolute Trajectory Error (ATE) por Keyframe', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Gráfico ATE guardado en: {output_file}")
    
    @staticmethod
    def plot_rpe_errors(errors: np.ndarray, output_file: str, metric_name: str = "RPE"):
        """
        Genera un gráfico de los errores RPE a lo largo de la trayectoria.
        
        Args:
            errors: Array con errores RPE por par de keyframes
            output_file: Ruta del archivo PNG de salida
            metric_name: Nombre de la métrica para el título
        """
        if len(errors) == 0:
            print(f"No hay errores {metric_name} para graficar")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(errors, 'g-', linewidth=1.5, label=f'Error {metric_name}')
        plt.axhline(y=np.mean(errors), color='r', linestyle='--', 
                    label=f'Media: {np.mean(errors):.3f} m', linewidth=2)
        plt.axhline(y=np.median(errors), color='orange', linestyle=':', 
                    label=f'Mediana: {np.median(errors):.3f} m', linewidth=2)
        
        plt.xlabel('Par de Keyframes', fontsize=12)
        plt.ylabel('Error Relativo (m)', fontsize=12)
        plt.title(f'{metric_name} por Par de Keyframes', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Gráfico {metric_name} guardado en: {output_file}")
