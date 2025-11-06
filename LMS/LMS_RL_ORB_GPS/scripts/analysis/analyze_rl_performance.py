#!/usr/bin/env python3
"""
Script de Análisis Avanzado del Rendimiento del RL

Extrae métricas del log de procesamiento y genera:
1. Gráficas de evolución temporal de pesos
2. Análisis de ajustes del RL vs reglas heurísticas
3. Correlación entre confianzas, error y decisiones
4. Detección de casos problemáticos
5. Comparación antes/después del ajuste RL

Uso:
    python analyze_rl_performance.py --log /tmp/rl_analysis_100frames.log
    python analyze_rl_performance.py --log archivo.log --output ./resultados

Autor: Sistema Híbrido SLAM-GPS
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from collections import defaultdict

# Configurar estilo de gráficas
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


class RLPerformanceAnalyzer:
    """Analizador avanzado del rendimiento del sistema RL"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.frames_data = []
        self.scenarios = defaultdict(list)
        
    def parse_log(self):
        """Extrae datos estructurados del log"""
        
        print("Leyendo log file...")
        
        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        current_frame = None
        frame_data = {}
        
        for line in lines:
            # Detectar inicio de frame
            match_frame = re.search(r'Procesando frame (\d+)\.\.\.', line)
            if match_frame:
                # Guardar frame anterior si existe
                if current_frame is not None and frame_data:
                    self.frames_data.append(frame_data)
                
                current_frame = int(match_frame.group(1))
                frame_data = {'frame': current_frame}
                continue
            
            if current_frame is None:
                continue
            
            # Extraer confianzas y error
            match_conf = re.search(r'\[CONFIANZAS\] Visual=([\d.]+).*GPS=([\d.]+).*Error=([\d.]+)m', line)
            if match_conf:
                frame_data['visual_conf'] = float(match_conf.group(1))
                frame_data['gps_conf'] = float(match_conf.group(2))
                frame_data['error'] = float(match_conf.group(3))
            
            # Extraer detección de balance
            if '[BALANCE]' in line:
                if 'SLAM más confiable' in line:
                    frame_data['balance_rule'] = 'SLAM_DOMINANT'
                elif 'GPS más confiable' in line:
                    frame_data['balance_rule'] = 'GPS_DOMINANT'
                elif 'Ambos muy confiables' in line:
                    frame_data['balance_rule'] = 'BOTH_EXCELLENT'
            
            # Extraer detección de error
            if '[ERROR]' in line:
                if 'Alto error' in line:
                    frame_data['error_rule'] = 'HIGH_ERROR'
                elif 'Error moderado' in line:
                    frame_data['error_rule'] = 'MODERATE_ERROR'
            
            # Extraer alineación
            if '[ALIGN]' in line:
                if 'Excelente alineación' in line:
                    frame_data['align_rule'] = 'EXCELLENT'
                elif 'Buena alineación' in line:
                    frame_data['align_rule'] = 'GOOD'
            
            # Extraer pesos base (reglas heurísticas)
            match_base = re.search(r'\[HYBRID\] Base: SLAM=([\d.]+), GPS=([\d.]+)', line)
            if match_base:
                frame_data['w_slam_base'] = float(match_base.group(1))
                frame_data['w_gps_base'] = float(match_base.group(2))
            
            # Extraer escenario y margen RL
            match_margin = re.search(r'\[RL MARGIN\] Escenario=(\w+), Margen=±([\d.]+)%', line)
            if match_margin:
                frame_data['scenario'] = match_margin.group(1)
                frame_data['rl_margin'] = float(match_margin.group(2)) / 100.0
            
            # Extraer ajustes RL
            match_adj = re.search(r'\[RL ADJ\] Δ_SLAM=([+-][\d.]+), Δ_GPS=([+-][\d.]+)', line)
            if match_adj:
                frame_data['delta_slam_rl'] = float(match_adj.group(1))
                frame_data['delta_gps_rl'] = float(match_adj.group(2))
            
            # Extraer pesos finales
            match_final = re.search(r'\[FINAL\]\s+SLAM=([\d.]+), GPS=([\d.]+)', line)
            if match_final:
                frame_data['w_slam_final'] = float(match_final.group(1))
                frame_data['w_gps_final'] = float(match_final.group(2))
        
        # Guardar último frame
        if current_frame is not None and frame_data:
            self.frames_data.append(frame_data)
        
        print(f"Extraidos {len(self.frames_data)} frames con datos completos\n")
        
        # Clasificar por escenario
        for data in self.frames_data:
            if 'scenario' in data:
                self.scenarios[data['scenario']].append(data)
    
    def print_summary_statistics(self):
        """Imprime estadísticas generales del análisis"""
        
        print("="*80)
        print("ESTADISTICAS GENERALES DEL ANALISIS")
        print("="*80)
        print(f"\nTotal de frames analizados: {len(self.frames_data)}\n")
        
        # Distribución de escenarios
        print("DISTRIBUCION DE ESCENARIOS:")
        total = len(self.frames_data)
        for scenario, frames in sorted(self.scenarios.items(), key=lambda x: len(x[1]), reverse=True):
            count = len(frames)
            pct = (count / total * 100) if total > 0 else 0
            print(f"   {scenario:25} {count:4d} frames ({pct:5.1f}%)")
        
        print("\n" + "─"*80 + "\n")
        
        # Estadísticas de confianzas
        if self.frames_data:
            visual_confs = [f['visual_conf'] for f in self.frames_data if 'visual_conf' in f]
            gps_confs = [f['gps_conf'] for f in self.frames_data if 'gps_conf' in f]
            errors = [f['error'] for f in self.frames_data if 'error' in f]
            
            print("ESTADISTICAS DE ENTRADA:")
            print(f"   Visual Confidence: {np.mean(visual_confs):.3f} ± {np.std(visual_confs):.3f} (rango: [{np.min(visual_confs):.3f}, {np.max(visual_confs):.3f}])")
            print(f"   GPS Confidence:    {np.mean(gps_confs):.3f} ± {np.std(gps_confs):.3f} (rango: [{np.min(gps_confs):.3f}, {np.max(gps_confs):.3f}])")
            print(f"   Error SLAM-GPS:    {np.mean(errors):.3f} ± {np.std(errors):.3f}m (rango: [{np.min(errors):.3f}, {np.max(errors):.3f}]m)")
        
        print("\n" + "─"*80 + "\n")
        
        # Análisis de ajustes RL
        if self.frames_data:
            deltas_slam = [f['delta_slam_rl'] for f in self.frames_data if 'delta_slam_rl' in f]
            w_slam_base = [f['w_slam_base'] for f in self.frames_data if 'w_slam_base' in f]
            w_slam_final = [f['w_slam_final'] for f in self.frames_data if 'w_slam_final' in f]
            
            print("ANALISIS DE AJUSTES RL:")
            print(f"   Ajuste promedio RL (SLAM):   {np.mean(deltas_slam):+.3f} ± {np.std(deltas_slam):.3f}")
            print(f"   Rango de ajustes:             [{np.min(deltas_slam):+.3f}, {np.max(deltas_slam):+.3f}]")
            print(f"   Peso SLAM base (heurística):  {np.mean(w_slam_base):.3f} ± {np.std(w_slam_base):.3f}")
            print(f"   Peso SLAM final (después RL): {np.mean(w_slam_final):.3f} ± {np.std(w_slam_final):.3f}")
            
            # Calcular % de veces que RL refuerza vs contradice reglas
            reinforces = sum(1 for d in self.frames_data 
                           if 'delta_slam_rl' in d and 'w_slam_base' in d and 'w_slam_final' in d
                           and ((d['w_slam_base'] > 0.5 and d['delta_slam_rl'] > 0) or
                                (d['w_slam_base'] < 0.5 and d['delta_slam_rl'] < 0)))
            contradicts = len(deltas_slam) - reinforces
            
            print(f"\n   RL REFUERZA reglas heurísticas: {reinforces}/{len(deltas_slam)} frames ({reinforces/len(deltas_slam)*100:.1f}%)")
            print(f"   RL CONTRADICE reglas:           {contradicts}/{len(deltas_slam)} frames ({contradicts/len(deltas_slam)*100:.1f}%)")
        
        print("\n" + "="*80 + "\n")
    
    def print_scenario_analysis(self):
        """Analiza comportamiento por escenario"""
        
        print("="*80)
        print("ANALISIS DETALLADO POR ESCENARIO")
        print("="*80 + "\n")
        
        for scenario_name in ['ALTA_CERTEZA', 'SENSOR_DOMINANTE', 'ERROR_ALTO', 'INCERTIDUMBRE', 'INTERMEDIO']:
            frames = self.scenarios.get(scenario_name, [])
            if not frames:
                continue
            
            print(f"{'─'*80}")
            print(f"Escenario: {scenario_name}")
            print(f"{'─'*80}")
            print(f"   Cantidad: {len(frames)} frames\n")
            
            # Promedios
            avg_visual = np.mean([f['visual_conf'] for f in frames if 'visual_conf' in f])
            avg_gps = np.mean([f['gps_conf'] for f in frames if 'gps_conf' in f])
            avg_error = np.mean([f['error'] for f in frames if 'error' in f])
            avg_margin = np.mean([f['rl_margin'] for f in frames if 'rl_margin' in f])
            
            print(f"   Entrada promedio:")
            print(f"      Visual conf: {avg_visual:.3f}")
            print(f"      GPS conf:    {avg_gps:.3f}")
            print(f"      Error:       {avg_error:.2f}m")
            print(f"      Margen RL:   ±{avg_margin*100:.0f}%\n")
            
            # Pesos
            avg_slam_base = np.mean([f['w_slam_base'] for f in frames if 'w_slam_base' in f])
            avg_slam_final = np.mean([f['w_slam_final'] for f in frames if 'w_slam_final' in f])
            avg_delta = np.mean([f['delta_slam_rl'] for f in frames if 'delta_slam_rl' in f])
            
            print(f"   Pesos promedio:")
            print(f"      SLAM base (heurística): {avg_slam_base:.3f} ({avg_slam_base*100:.1f}%)")
            print(f"      Ajuste RL:              {avg_delta:+.3f} ({avg_delta*100:+.1f}%)")
            print(f"      SLAM final:             {avg_slam_final:.3f} ({avg_slam_final*100:.1f}%)\n")
            
            # Interpretación
            if avg_delta > 0.01:
                print(f"   RL tiende a AUMENTAR confianza en SLAM")
            elif avg_delta < -0.01:
                print(f"   RL tiende a AUMENTAR confianza en GPS")
            else:
                print(f"   RL hace ajustes MINIMOS (respeta reglas)")
            
            # Ejemplo representativo
            mid_idx = len(frames) // 2
            example = frames[mid_idx]
            print(f"\n   Ejemplo representativo (frame {example['frame']}):")
            print(f"      Visual={example.get('visual_conf', 0):.2f}, GPS={example.get('gps_conf', 0):.2f}, Error={example.get('error', 0):.1f}m")
            print(f"      Base: SLAM={example.get('w_slam_base', 0)*100:.0f}% → RL ajusta {example.get('delta_slam_rl', 0)*100:+.1f}% → Final: SLAM={example.get('w_slam_final', 0)*100:.0f}%")
            print()
    
    def detect_problematic_cases(self):
        """Detecta casos donde el RL puede estar comportándose mal"""
        
        print("="*80)
        print("DETECCION DE CASOS PROBLEMATICOS")
        print("="*80 + "\n")
        
        problematic = []
        
        for data in self.frames_data:
            issues = []
            
            # CASO 1: RL contradice fuertemente las reglas en escenario de alta certeza
            if (data.get('scenario') == 'ALTA_CERTEZA' and 
                abs(data.get('delta_slam_rl', 0)) > 0.05):
                issues.append("RL hace ajuste grande (>5%) en escenario de alta certeza")
            
            # CASO 2: Ambos sensores malos pero RL favorece uno fuertemente
            if (data.get('visual_conf', 1) < 0.5 and data.get('gps_conf', 1) < 0.5 and
                (data.get('w_slam_final', 0.5) > 0.7 or data.get('w_slam_final', 0.5) < 0.3)):
                issues.append("Ambos sensores malos pero RL decide fuertemente por uno")
            
            # CASO 3: Error muy alto pero RL favorece GPS
            if (data.get('error', 0) > 5.0 and data.get('w_gps_final', 0) > 0.6):
                issues.append("Error alto (>5m) pero RL favorece GPS")
            
            # CASO 4: GPS excelente y error bajo pero RL favorece SLAM
            if (data.get('gps_conf', 0) > 0.9 and data.get('error', 999) < 2.0 and
                data.get('w_slam_final', 0.5) > 0.6):
                issues.append("GPS excelente con error bajo pero RL favorece SLAM")
            
            if issues:
                data_copy = data.copy()
                data_copy['issues'] = issues
                problematic.append(data_copy)
        
        if not problematic:
            print("NO se detectaron casos problematicos evidentes")
            print("   El RL parece estar tomando decisiones razonables en todos los frames.\n")
        else:
            print(f"Se detectaron {len(problematic)} casos potencialmente problematicos:\n")
            
            for i, case in enumerate(problematic[:5], 1):  # Mostrar solo primeros 5
                print(f"   Caso {i} - Frame {case['frame']}:")
                print(f"      Visual={case.get('visual_conf', 0):.2f}, GPS={case.get('gps_conf', 0):.2f}, Error={case.get('error', 0):.1f}m")
                print(f"      Pesos: Base SLAM={case.get('w_slam_base', 0)*100:.0f}% → Final SLAM={case.get('w_slam_final', 0)*100:.0f}%")
                for issue in case['issues']:
                    print(f"      {issue}")
                print()
            
            if len(problematic) > 5:
                print(f"   ... y {len(problematic)-5} casos mas\n")
        
        print("="*80 + "\n")
        
        return problematic
    
    def plot_weights_evolution(self, output_dir):
        """Gráfica 1: Evolución temporal de pesos"""
        
        frames = [d['frame'] for d in self.frames_data]
        w_slam_base = [d.get('w_slam_base', np.nan) for d in self.frames_data]
        w_slam_final = [d.get('w_slam_final', np.nan) for d in self.frames_data]
        w_gps_final = [d.get('w_gps_final', np.nan) for d in self.frames_data]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
        
        # Subplot 1: Pesos SLAM (base vs final)
        ax1.plot(frames, w_slam_base, 'g--', linewidth=2, alpha=0.6, label='Base (Heurística)', marker='o', markersize=3)
        ax1.plot(frames, w_slam_final, 'b-', linewidth=2.5, alpha=0.9, label='Final (después RL)', marker='s', markersize=3)
        ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Balance 50-50')
        ax1.fill_between(frames, w_slam_base, w_slam_final, alpha=0.2, color='orange', label='Ajuste RL')
        
        ax1.set_xlabel('Frame', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Peso SLAM', fontsize=13, fontweight='bold')
        ax1.set_title('Evolución de Pesos SLAM: Heurística → RL → Final', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11, loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Subplot 2: Balance SLAM vs GPS (final)
        ax2.plot(frames, w_slam_final, 'b-', linewidth=2.5, alpha=0.9, label='SLAM', marker='s', markersize=3)
        ax2.plot(frames, w_gps_final, 'r-', linewidth=2.5, alpha=0.9, label='GPS', marker='o', markersize=3)
        ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Balance 50-50')
        ax2.fill_between(frames, 0, 1, where=np.array(w_slam_final) > np.array(w_gps_final), 
                         alpha=0.1, color='blue', label='Favorece SLAM')
        ax2.fill_between(frames, 0, 1, where=np.array(w_gps_final) > np.array(w_slam_final), 
                         alpha=0.1, color='red', label='Favorece GPS')
        
        ax2.set_xlabel('Frame', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Peso', fontsize=13, fontweight='bold')
        ax2.set_title('Balance Final: SLAM vs GPS', fontsize=15, fontweight='bold')
        ax2.legend(fontsize=11, loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'evolution_weights.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grafica 1 guardada: {output_path}")
    
    def plot_rl_adjustments(self, output_dir):
        """Gráfica 2: Análisis de ajustes del RL"""
        
        frames = [d['frame'] for d in self.frames_data]
        deltas_slam = [d.get('delta_slam_rl', 0) for d in self.frames_data]
        rl_margins = [d.get('rl_margin', 0) * 100 for d in self.frames_data]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
        
        # Subplot 1: Ajustes del RL
        colors = ['green' if d > 0 else 'red' for d in deltas_slam]
        ax1.bar(frames, np.array(deltas_slam)*100, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax1.axhline(y=5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Ajuste moderado (±5%)')
        ax1.axhline(y=-5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax1.set_xlabel('Frame', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Ajuste RL en SLAM (%)', fontsize=13, fontweight='bold')
        ax1.set_title('Ajustes del RL Frame por Frame (Verde=↑SLAM, Rojo=↑GPS)', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Subplot 2: Margen dinámico del RL
        ax2.plot(frames, rl_margins, 'purple', linewidth=2.5, alpha=0.8, marker='o', markersize=4)
        ax2.fill_between(frames, 0, rl_margins, alpha=0.3, color='purple')
        
        ax2.set_xlabel('Frame', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Margen RL (%)', fontsize=13, fontweight='bold')
        ax2.set_title('Margen Dinámico del RL (Cuánto puede ajustar)', fontsize=15, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Añadir anotaciones para diferentes niveles
        ax2.axhline(y=10, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Alta certeza (±10%)')
        ax2.axhline(y=20, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Sensor dominante (±20%)')
        ax2.axhline(y=35, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Incertidumbre (±35%)')
        ax2.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'rl_adjustments.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grafica 2 guardada: {output_path}")
    
    def plot_confidence_vs_weights(self, output_dir):
        """Gráfica 3: Correlación entre confianzas y decisiones"""
        
        visual_confs = [d.get('visual_conf', np.nan) for d in self.frames_data]
        gps_confs = [d.get('gps_conf', np.nan) for d in self.frames_data]
        w_slam_final = [d.get('w_slam_final', np.nan) for d in self.frames_data]
        errors = [d.get('error', np.nan) for d in self.frames_data]
        
        fig = plt.figure(figsize=(18, 12))
        
        # Subplot 1: Visual conf vs Peso SLAM
        ax1 = plt.subplot(2, 2, 1)
        scatter1 = ax1.scatter(visual_confs, w_slam_final, c=errors, cmap='RdYlGn_r', 
                              s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Relación lineal perfecta')
        ax1.set_xlabel('Confianza Visual', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Peso SLAM Final', fontsize=12, fontweight='bold')
        ax1.set_title('Correlación: Confianza Visual → Peso SLAM', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        plt.colorbar(scatter1, ax=ax1, label='Error (m)')
        
        # Subplot 2: GPS conf vs Peso GPS
        ax2 = plt.subplot(2, 2, 2)
        w_gps_final = [1 - w for w in w_slam_final]
        scatter2 = ax2.scatter(gps_confs, w_gps_final, c=errors, cmap='RdYlGn_r', 
                              s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Relación lineal perfecta')
        ax2.set_xlabel('Confianza GPS', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Peso GPS Final', fontsize=12, fontweight='bold')
        ax2.set_title('Correlación: Confianza GPS → Peso GPS', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.colorbar(scatter2, ax=ax2, label='Error (m)')
        
        # Subplot 3: Error vs Balance de pesos
        ax3 = plt.subplot(2, 2, 3)
        balance = [w_s - 0.5 for w_s in w_slam_final]  # >0 favorece SLAM, <0 favorece GPS
        scatter3 = ax3.scatter(errors, balance, c=visual_confs, cmap='viridis', 
                              s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax3.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.5, label='Balance 50-50')
        ax3.axvline(x=3, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Error alto (>3m)')
        ax3.set_xlabel('Error SLAM-GPS (m)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Balance (>0=SLAM, <0=GPS)', fontsize=12, fontweight='bold')
        ax3.set_title('Error vs Decisión de Fusión', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        plt.colorbar(scatter3, ax=ax3, label='Visual Conf')
        
        # Subplot 4: Ratio de confianzas vs Peso SLAM
        ax4 = plt.subplot(2, 2, 4)
        conf_ratios = [v / (g + 0.001) for v, g in zip(visual_confs, gps_confs)]
        scatter4 = ax4.scatter(conf_ratios, w_slam_final, c=errors, cmap='RdYlGn_r', 
                              s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax4.axvline(x=1, color='gray', linestyle='-', linewidth=2, alpha=0.5, label='Confianzas iguales')
        ax4.axhline(y=0.5, color='gray', linestyle='-', linewidth=2, alpha=0.5)
        ax4.set_xlabel('Ratio Conf (Visual/GPS)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Peso SLAM Final', fontsize=12, fontweight='bold')
        ax4.set_title('Ratio de Confianzas → Decisión', fontsize=13, fontweight='bold')
        ax4.set_xlim(0, 3)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        plt.colorbar(scatter4, ax=ax4, label='Error (m)')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'confidence_correlations.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grafica 3 guardada: {output_path}")
    
    def plot_scenario_comparison(self, output_dir):
        """Gráfica 4: Comparación por escenario"""
        
        scenario_names = []
        avg_margins = []
        avg_deltas = []
        avg_slam_base = []
        avg_slam_final = []
        scenario_counts = []
        
        for scenario in ['ALTA_CERTEZA', 'SENSOR_DOMINANTE', 'ERROR_ALTO', 'INCERTIDUMBRE', 'INTERMEDIO']:
            frames = self.scenarios.get(scenario, [])
            if not frames:
                continue
            
            scenario_names.append(scenario.replace('_', '\n'))
            scenario_counts.append(len(frames))
            avg_margins.append(np.mean([f.get('rl_margin', 0) for f in frames]) * 100)
            avg_deltas.append(np.mean([f.get('delta_slam_rl', 0) for f in frames]) * 100)
            avg_slam_base.append(np.mean([f.get('w_slam_base', 0) for f in frames]))
            avg_slam_final.append(np.mean([f.get('w_slam_final', 0) for f in frames]))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        x = np.arange(len(scenario_names))
        width = 0.35
        
        # Subplot 1: Pesos base vs final
        bars1 = ax1.bar(x - width/2, avg_slam_base, width, label='Base (Heurística)', 
                       alpha=0.8, color='green', edgecolor='black')
        bars2 = ax1.bar(x + width/2, avg_slam_final, width, label='Final (después RL)', 
                       alpha=0.8, color='blue', edgecolor='black')
        ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
        ax1.set_ylabel('Peso SLAM Promedio', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Escenario', fontsize=12, fontweight='bold')
        ax1.set_title('Pesos SLAM: Base vs Final por Escenario', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenario_names, fontsize=9)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1)
        
        # Subplot 2: Margen RL por escenario
        bars3 = ax2.bar(x, avg_margins, alpha=0.8, color='purple', edgecolor='black')
        ax2.set_ylabel('Margen RL Promedio (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Escenario', fontsize=12, fontweight='bold')
        ax2.set_title('Margen Dinámico del RL por Escenario', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenario_names, fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores encima de las barras
        for i, v in enumerate(avg_margins):
            ax2.text(i, v + 1, f'±{v:.0f}%', ha='center', fontweight='bold', fontsize=10)
        
        # Subplot 3: Ajustes promedio del RL
        colors = ['green' if d > 0 else 'red' for d in avg_deltas]
        bars4 = ax3.bar(x, avg_deltas, alpha=0.8, color=colors, edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax3.set_ylabel('Ajuste RL Promedio (%)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Escenario', fontsize=12, fontweight='bold')
        ax3.set_title('Ajuste Promedio del RL por Escenario', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenario_names, fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores
        for i, v in enumerate(avg_deltas):
            ax3.text(i, v + 0.2 if v > 0 else v - 0.5, f'{v:+.1f}%', 
                    ha='center', fontweight='bold', fontsize=10)
        
        # Subplot 4: Distribución de frames por escenario
        ax4.pie(scenario_counts, labels=scenario_names, autopct='%1.1f%%', 
               startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'},
               colors=plt.cm.Set3.colors[:len(scenario_names)])
        ax4.set_title(f'Distribución de Frames por Escenario (Total: {sum(scenario_counts)})', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'scenario_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grafica 4 guardada: {output_path}")
    
    def save_detailed_report(self, output_dir, problematic_cases):
        """Guarda reporte completo en texto"""
        
        report_path = os.path.join(output_dir, 'rl_performance_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("REPORTE COMPLETO DE ANÁLISIS DEL SISTEMA RL-ORB-SLAM-GPS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Archivo analizado: {self.log_file}\n")
            f.write(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total de frames: {len(self.frames_data)}\n\n")
            
            # Estadísticas generales
            if self.frames_data:
                visual_confs = [d.get('visual_conf', 0) for d in self.frames_data if 'visual_conf' in d]
                gps_confs = [d.get('gps_conf', 0) for d in self.frames_data if 'gps_conf' in d]
                errors = [d.get('error', 0) for d in self.frames_data if 'error' in d]
                deltas = [d.get('delta_slam_rl', 0) for d in self.frames_data if 'delta_slam_rl' in d]
                
                f.write("ESTADÍSTICAS GENERALES\n")
                f.write("-"*80 + "\n")
                f.write(f"Visual Confidence:  {np.mean(visual_confs):.3f} ± {np.std(visual_confs):.3f}\n")
                f.write(f"GPS Confidence:     {np.mean(gps_confs):.3f} ± {np.std(gps_confs):.3f}\n")
                f.write(f"Error SLAM-GPS:     {np.mean(errors):.3f} ± {np.std(errors):.3f}m\n")
                f.write(f"Ajuste RL promedio: {np.mean(deltas):+.3f} ± {np.std(deltas):.3f}\n\n")
            
            # Distribución por escenario
            f.write("DISTRIBUCIÓN POR ESCENARIO\n")
            f.write("-"*80 + "\n")
            for scenario, frames in sorted(self.scenarios.items(), key=lambda x: len(x[1]), reverse=True):
                count = len(frames)
                pct = (count / len(self.frames_data) * 100) if self.frames_data else 0
                f.write(f"{scenario:25} {count:4d} frames ({pct:5.1f}%)\n")
            f.write("\n")
            
            # Casos problemáticos
            f.write("CASOS PROBLEMÁTICOS DETECTADOS\n")
            f.write("-"*80 + "\n")
            if not problematic_cases:
                f.write("NO se detectaron casos problemáticos\n\n")
            else:
                f.write(f"{len(problematic_cases)} casos detectados:\n\n")
                for i, case in enumerate(problematic_cases, 1):
                    f.write(f"Caso {i} - Frame {case['frame']}:\n")
                    f.write(f"  Visual={case.get('visual_conf', 0):.2f}, GPS={case.get('gps_conf', 0):.2f}, Error={case.get('error', 0):.1f}m\n")
                    f.write(f"  Base SLAM={case.get('w_slam_base', 0)*100:.0f}% → Final SLAM={case.get('w_slam_final', 0)*100:.0f}%\n")
                    for issue in case.get('issues', []):
                        f.write(f"  - {issue}\n")
                    f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("FIN DEL REPORTE\n")
            f.write("="*80 + "\n")
        
        print(f"Reporte detallado guardado: {report_path}")
    
    def run_full_analysis(self, output_dir):
        """Ejecuta análisis completo y genera todos los outputs"""
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("INICIANDO ANALISIS COMPLETO DEL RENDIMIENTO DEL RL")
        print("="*80 + "\n")
        
        # 1. Parsear log
        self.parse_log()
        
        # 2. Estadísticas generales
        self.print_summary_statistics()
        
        # 3. Análisis por escenario
        self.print_scenario_analysis()
        
        # 4. Detectar casos problemáticos
        problematic = self.detect_problematic_cases()
        
        # 5. Generar gráficas
        print("Generando graficas...\n")
        self.plot_weights_evolution(output_dir)
        self.plot_rl_adjustments(output_dir)
        self.plot_confidence_vs_weights(output_dir)
        self.plot_scenario_comparison(output_dir)
        
        # 6. Guardar reporte
        self.save_detailed_report(output_dir, problematic)
        
        print("\n" + "="*80)
        print("ANALISIS COMPLETO FINALIZADO")
        print("="*80)
        print(f"Resultados guardados en: {output_dir}")
        print("\nArchivos generados:")
        print("  - evolution_weights.png          (evolucion temporal)")
        print("  - rl_adjustments.png             (ajustes del RL)")
        print("  - confidence_correlations.png    (correlaciones)")
        print("  - scenario_comparison.png        (comparacion por escenario)")
        print("  - rl_performance_report.txt      (reporte textual)")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analizar rendimiento del sistema RL desde logs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--log',
        type=str,
        required=True,
        help='Ruta al archivo de log'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Directorio de salida (default: auto-generado)'
    )
    
    args = parser.parse_args()
    
    # Verificar que el log existe
    if not os.path.exists(args.log):
        print(f"ERROR: Archivo de log no encontrado: {args.log}")
        return 1
    
    # Generar directorio de salida
    if args.output is None:
        timestamp = datetime.now().strftime("%H%M_%d%m_%Y")
        args.output = f"resultados/LMS_RL_ORB_GPS/analysis_{timestamp}"
    
    # Crear analizador y ejecutar
    analyzer = RLPerformanceAnalyzer(args.log)
    analyzer.run_full_analysis(args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())
