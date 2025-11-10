"""
Módulo de Visualización - Gráficos de trayectorias y métricas
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os


class TrajectoryPlotter:
    """Clase para generar gráficos de trayectorias SLAM/GPS."""
    
    @staticmethod
    def plot_ate_comparison(comparison_results: Dict, output_file: str):
        """
        Genera gráfico comparativo de errores ATE entre métodos.
        
        Args:
            comparison_results: Diccionario con resultados de comparación
            output_file: Ruta donde guardar el gráfico
        """
        plt.figure(figsize=(14, 8))
        
        colors = {
            'pipeline_vs_ref': 'blue', 
            'slam_vs_ref': 'green', 
            'gps_vs_ref': 'red', 
            'pipeline_vs_slam': 'purple'
        }
        
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
    
    @staticmethod
    def plot_rpe_comparison(comparison_results: Dict, output_file: str):
        """
        Genera gráfico comparativo de errores RPE entre métodos.
        
        Args:
            comparison_results: Diccionario con resultados de comparación
            output_file: Ruta donde guardar el gráfico
        """
        plt.figure(figsize=(14, 8))
        
        colors = {
            'pipeline_vs_ref': 'blue', 
            'slam_vs_ref': 'green', 
            'gps_vs_ref': 'red', 
            'pipeline_vs_slam': 'purple'
        }
        
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
    
    @staticmethod
    def plot_complete_comparison(
        keyframe_positions: np.ndarray,
        slam_history: Optional[List[np.ndarray]] = None,
        gps_history: Optional[List[np.ndarray]] = None,
        optimized_trajectory: Optional[List[np.ndarray]] = None,
        stats: Optional[Dict] = None,
        comparison_results: Optional[Dict] = None,
        output_file: str = "trajectory_comparison.png",
        input_path: str = ""
    ):
        """
        Genera gráfico completo comparando todas las trayectorias.
        
        Args:
            keyframe_positions: Posiciones de keyframes (N x 3)
            slam_history: Historial de posiciones SLAM puro
            gps_history: Historial de posiciones GPS
            optimized_trajectory: Trayectoria optimizada
            stats: Diccionario con estadísticas
            comparison_results: Resultados de comparación de métricas
            output_file: Ruta donde guardar el gráfico
            input_path: Nombre de la secuencia/video procesado
        """
        plt.figure(figsize=(16, 12))
        
        # Preparar coordenadas de keyframes
        x_key, y_key = keyframe_positions[:, 0], keyframe_positions[:, 1]
        x_key -= x_key[0]
        y_key -= y_key[0]
        
        # Pipeline fusionado (keyframes) - línea principal
        plt.plot(x_key, y_key, 'b-', label='Pipeline Fusionado (SLAM+GPS+RL)', 
                linewidth=4, marker='o', markersize=8, zorder=5)
        
        # SLAM puro
        if slam_history is not None and len(slam_history) > 0:
            slam_traj = np.array(slam_history)
            if slam_traj.shape[1] >= 2:
                x_slam, y_slam = slam_traj[:, 0], slam_traj[:, 1]
                x_slam -= x_slam[0]
                y_slam -= y_slam[0]
                plt.plot(x_slam, y_slam, 'g--', label='SLAM Puro', 
                        linewidth=2.5, zorder=3, alpha=0.8)
        
        # GPS puro
        if gps_history is not None and len(gps_history) > 0:
            gps_traj = np.array(gps_history)
            if gps_traj.shape[1] >= 2:
                x_gps, y_gps = gps_traj[:, 0], gps_traj[:, 1]
                x_gps -= x_gps[0]
                y_gps -= y_gps[0]
                plt.plot(x_gps, y_gps, 'r-.', label='GPS Puro (Ground Truth)', 
                        linewidth=2.5, zorder=4, alpha=0.9)
        
        # Trayectoria optimizada
        if optimized_trajectory is not None:
            try:
                if all(hasattr(pose, 'shape') and pose.shape == (4, 4) for pose in optimized_trajectory):
                    opt_positions = np.array([pose[:3, 3] for pose in optimized_trajectory])
                    x_opt, y_opt = opt_positions[:, 0], opt_positions[:, 1]
                    x_opt -= x_opt[0]
                    y_opt -= y_opt[0]
                    
                    # Solo graficar si es significativamente diferente
                    if np.max(np.abs(x_opt - x_key)) > 0.1 or np.max(np.abs(y_opt - y_key)) > 0.1:
                        plt.plot(x_opt, y_opt, 'm:', label='Trayectoria Optimizada', 
                                linewidth=1.5, zorder=1, alpha=0.5)
            except:
                pass
        
        # Marcadores de inicio y fin
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
        info_text = ""
        if stats:
            info_text = (
                f"Keyframes: {stats.get('keyframes', 'N/A')}\n"
                f"Dist. total: {stats.get('dist_total', 0):.1f} m\n"
                f"Prom. matches: {stats.get('matches_prom', 0):.1f}\n"
                f"Éxito triang: {stats.get('exito_triang', 0)*100:.2f}%\n"
                f"Mov. medio: {stats.get('mov_medio', 0):.2f} m"
            )
        
        # Agregar métricas de comparación
        if comparison_results:
            try:
                pipeline_vs_gps = comparison_results.get('pipeline_vs_ref')
                slam_vs_gps = comparison_results.get('slam_vs_ref')
                
                if pipeline_vs_gps and slam_vs_gps:
                    info_text += f"\n\n--- Métricas vs GPS GT ---"
                    info_text += f"\nPipeline ATE: {pipeline_vs_gps['ate']['rmse']:.2f}m"
                    info_text += f"\nSLAM ATE: {slam_vs_gps['ate']['rmse']:.2f}m"
                    
                    if slam_vs_gps['ate']['rmse'] > 0:
                        improvement = ((slam_vs_gps['ate']['rmse'] - pipeline_vs_gps['ate']['rmse']) / 
                                      slam_vs_gps['ate']['rmse']) * 100
                        info_text += f"\nMejora: {improvement:.1f}%"
            except:
                pass
        
        if info_text:
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
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico comparativo completo guardado en: {output_file}")
    
    @staticmethod
    def plot_keyframes_only(
        keyframe_positions: np.ndarray,
        stats: Optional[Dict] = None,
        output_file: str = "trajectory_keyframes.png",
        input_path: str = ""
    ):
        """
        Genera gráfico simplificado solo con keyframes.
        
        Args:
            keyframe_positions: Posiciones de keyframes (N x 3)
            stats: Diccionario con estadísticas
            output_file: Ruta donde guardar el gráfico
            input_path: Nombre de la secuencia/video procesado
        """
        plt.figure(figsize=(12, 9))
        
        x_key, y_key = keyframe_positions[:, 0], keyframe_positions[:, 1]
        x_key -= x_key[0]
        y_key -= y_key[0]
        
        plt.plot(x_key, y_key, 'b-', linewidth=3, marker='o', markersize=8)
        plt.scatter(x_key[0], y_key[0], color='green', s=200, label='Inicio')
        plt.scatter(x_key[-1], y_key[-1], color='red', s=200, label='Fin')
        
        # Flechas de dirección
        if len(x_key) > 1:
            for i in range(len(x_key)-1):
                dx = x_key[i+1] - x_key[i]
                dy = y_key[i+1] - y_key[i]
                plt.arrow(x_key[i], y_key[i], dx*0.8, dy*0.8, 
                        head_width=0.5, head_length=1.0, fc='blue', ec='blue', alpha=0.7)
        
        # Información de estadísticas
        if stats:
            info_text = (
                f"Keyframes: {stats.get('keyframes', 'N/A')}\n"
                f"Dist. total: {stats.get('dist_total', 0):.1f} m\n"
                f"Prom. matches: {stats.get('matches_prom', 0):.1f}\n"
                f"Éxito triang: {stats.get('exito_triang', 0)*100:.2f}%\n"
                f"Mov. medio: {stats.get('mov_medio', 0):.2f} m"
            )
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title(f"Trayectoria Keyframes - {os.path.basename(input_path)}")
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico de keyframes guardado en: {output_file}")
    
    @staticmethod
    def plot_comparative_trajectory(
        optimized_positions: np.ndarray,
        slam_history: Optional[List[np.ndarray]] = None,
        gps_history: Optional[List[np.ndarray]] = None,
        output_file: str = "trajectory_comparative.png",
        input_path: str = ""
    ):
        """
        Genera gráfico comparativo simple entre trayectorias.
        
        Args:
            optimized_positions: Posiciones optimizadas (N x 2 o N x 3)
            slam_history: Historial de posiciones SLAM puro
            gps_history: Historial de posiciones GPS
            output_file: Ruta donde guardar el gráfico
            input_path: Nombre de la secuencia/video procesado
        """
        plt.figure(figsize=(14, 10))
        
        x_opt, y_opt = optimized_positions[:, 0], optimized_positions[:, 1]
        
        plt.plot(x_opt, y_opt, 'b-', label='Trayectoria optimizada', linewidth=3, zorder=3)
        
        if slam_history is not None and len(slam_history) > 0:
            slam_traj = np.array(slam_history)
            if slam_traj.shape[1] >= 2:
                x_slam, y_slam = slam_traj[:, 0], slam_traj[:, 1]
                x_slam -= x_slam[0]
                y_slam -= y_slam[0]
                plt.plot(x_slam, y_slam, 'g--', label='SLAM puro', alpha=0.6, linewidth=2, zorder=2)
        
        if gps_history is not None and len(gps_history) > 0:
            gps_traj = np.array(gps_history)
            if gps_traj.shape[1] >= 2:
                x_gps, y_gps = gps_traj[:, 0], gps_traj[:, 1]
                x_gps -= x_gps[0]
                y_gps -= y_gps[0]
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
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico comparativo guardado en: {output_file}")
