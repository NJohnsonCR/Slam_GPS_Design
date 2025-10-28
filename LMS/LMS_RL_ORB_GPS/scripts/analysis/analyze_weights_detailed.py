#!/usr/bin/env python3
"""
Script de Análisis Detallado de Pesos y Confianzas

Este script ejecuta el modelo RL entrenado y muestra FRAME POR FRAME:
- Confianzas de entrada (GPS, SLAM)
- Error SLAM-GPS
- Pesos base (reglas heurísticas)
- Pesos RL (ajustes aprendidos)
- Pesos finales (fusión)
- Comparación con casos específicos (GPS malo, SLAM malo, conflictos, etc.)

Uso:
    python analyze_weights_detailed.py --sequence 0048 --max_frames 50
    python analyze_weights_detailed.py --sequence 0013 --show_all_frames

Autor: Sistema Híbrido SLAM-GPS
Fecha: Octubre 2025
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# Agregar paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from LMS.LMS_RL_ORB_GPS.main import RL_ORB_SLAM_GPS, load_kitti_sequence


class WeightsAnalyzer:
    """Analizador detallado de pesos y confianzas del RL"""
    
    def __init__(self, model_path, sequence_path):
        self.model_path = model_path
        self.sequence_path = sequence_path
        
        # Historial de datos por frame
        self.frame_history = []
        
        # Clasificación de escenarios
        self.scenarios = {
            'gps_malo': [],
            'slam_malo': [],
            'ambos_malos': [],
            'conflicto_alto': [],
            'normal': []
        }
        
        # Configurar output
        self.setup_output_dir()
    
    def setup_output_dir(self):
        """Crea directorio para resultados"""
        timestamp = datetime.now().strftime("%H%M_%d%m_%Y")
        seq_name = os.path.basename(self.sequence_path)
        self.output_dir = f"resultados/LMS_RL_ORB_GPS/weights_analysis_{seq_name}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📂 Resultados se guardarán en: {self.output_dir}\n")
    
    def classify_scenario(self, gps_conf, visual_conf, error):
        """Clasifica el escenario del frame"""
        
        if gps_conf < 0.5 and visual_conf > 0.7:
            return 'gps_malo'
        elif visual_conf < 0.5 and gps_conf > 0.7:
            return 'slam_malo'
        elif gps_conf < 0.5 and visual_conf < 0.5:
            return 'ambos_malos'
        elif error > 8.0 and gps_conf > 0.7 and visual_conf > 0.7:
            return 'conflicto_alto'
        else:
            return 'normal'
    
    def calculate_base_weights_external(self, gps_conf, visual_conf, error):
        """
        Replica la lógica de calculate_base_weights() del main.py
        para poder analizarla externamente
        """
        
        # REGLA 1: GPS mucho mejor que SLAM
        if gps_conf > 0.7 and visual_conf < 0.5:
            return 0.35, 0.65, "GPS >> SLAM"
        
        # REGLA 2: SLAM mucho mejor que GPS
        elif visual_conf > 0.7 and gps_conf < 0.5:
            return 0.65, 0.35, "SLAM >> GPS"
        
        # REGLA 3: Error alto → desconfiar
        elif error > 5.0:
            return 0.50, 0.50, "Error alto (>5m)"
        
        # REGLA 4: Ambos buenos y error bajo
        elif gps_conf > 0.6 and visual_conf > 0.6 and error < 3.0:
            total = gps_conf + visual_conf
            w_slam = visual_conf / total
            w_gps = gps_conf / total
            return w_slam, w_gps, "Ambos buenos, balance proporcional"
        
        # REGLA 5: Default - ratio de confianzas
        else:
            conf_ratio = visual_conf / (gps_conf + 1e-6)
            
            if conf_ratio > 1.2:
                w_slam = 0.60
                rule = "SLAM ligeramente mejor (ratio>1.2)"
            elif conf_ratio < 0.8:
                w_slam = 0.40
                rule = "GPS ligeramente mejor (ratio<0.8)"
            else:
                w_slam = 0.50
                rule = "Confianzas similares"
            
            return w_slam, 1 - w_slam, rule
    
    def analyze(self, max_frames=None, show_every_n_frames=10):
        """
        Ejecuta análisis completo del modelo
        
        Args:
            max_frames: Límite de frames (None = todos)
            show_every_n_frames: Mostrar detalles cada N frames
        """
        
        print("="*80)
        print("🔍 ANÁLISIS DETALLADO DE PESOS Y CONFIANZAS")
        print("="*80)
        print(f"📦 Modelo: {self.model_path}")
        print(f"📍 Secuencia: {self.sequence_path}")
        print(f"🎯 Frames: {max_frames or 'Todos'}")
        print("="*80 + "\n")
        
        # Cargar secuencia
        print("📥 Cargando secuencia KITTI...")
        frames, gps_data = load_kitti_sequence(self.sequence_path, max_frames)
        print(f"✅ Cargados {len(frames)} frames\n")
        
        # Crear instancia SLAM con RL
        print("🤖 Inicializando modelo RL...")
        slam = RL_ORB_SLAM_GPS(
            fx=718.856, fy=718.856, cx=607.1928, cy=185.2157,
            training_mode=False,
            model_path=self.model_path,
            simulate_mobile_gps=True,
            gps_noise_std=3.0
        )
        print("✅ Modelo cargado\n")
        
        print("="*80)
        print("📊 PROCESANDO FRAMES Y ANALIZANDO PESOS")
        print("="*80 + "\n")
        
        # Procesar cada frame
        for i, frame in enumerate(frames):
            if frame is None:
                continue
            
            # Obtener GPS
            gps_utm = None
            gps_conf = 0.1
            
            if gps_data[i] is not None:
                gps_utm = slam.gps_frame_reference(gps_data[i])
                slam.gps_filter.add_measurement(gps_utm)
                gps_conf = slam.gps_filter.calculate_confidence()
            
            # Antes de procesar, capturar estado interno
            # (necesitamos acceder a visual_conf y error antes de process_frame)
            
            # Por simplicidad, procesar el frame primero
            slam.process_frame_with_gps(frame, gps_utm, gps_conf)
            
            # Capturar datos si el sistema tiene información
            if hasattr(slam, 'last_rl_weights') and slam.last_rl_weights is not None:
                w_slam_final, w_gps_final = slam.last_rl_weights
                
                # Estimar visual_conf (placeholder - en realidad viene del tracker)
                visual_conf = 0.75 if len(slam.keyframe_poses) > 0 else 0.5
                
                # Calcular error
                error = 0.0
                if gps_utm is not None and len(slam.keyframe_poses) > 0:
                    last_pose = slam.keyframe_poses[-1][:3, 3]
                    error = np.linalg.norm(last_pose - gps_utm)
                
                # Calcular pesos base (reglas heurísticas)
                w_slam_base, w_gps_base, rule = self.calculate_base_weights_external(
                    gps_conf, visual_conf, error
                )
                
                # Estimar pesos RL (inverso de la fusión)
                # final = 0.7 * rl + 0.3 * base
                # rl = (final - 0.3 * base) / 0.7
                w_slam_rl = (w_slam_final - 0.3 * w_slam_base) / 0.7
                w_gps_rl = (w_gps_final - 0.3 * w_gps_base) / 0.7
                
                # Clasificar escenario
                scenario = self.classify_scenario(gps_conf, visual_conf, error)
                
                # Guardar datos
                frame_data = {
                    'frame': i,
                    'gps_conf': gps_conf,
                    'visual_conf': visual_conf,
                    'error': error,
                    'w_slam_base': w_slam_base,
                    'w_gps_base': w_gps_base,
                    'w_slam_rl': w_slam_rl,
                    'w_gps_rl': w_gps_rl,
                    'w_slam_final': w_slam_final,
                    'w_gps_final': w_gps_final,
                    'rule': rule,
                    'scenario': scenario
                }
                
                self.frame_history.append(frame_data)
                self.scenarios[scenario].append(frame_data)
                
                # Mostrar detalles cada N frames
                if i % show_every_n_frames == 0:
                    self.print_frame_details(frame_data)
        
        print("\n" + "="*80)
        print("✅ ANÁLISIS COMPLETADO")
        print("="*80 + "\n")
        
        # Generar resúmenes
        self.print_summary()
        self.plot_weights_evolution()
        self.plot_scenarios_comparison()
        self.save_detailed_report()
    
    def print_frame_details(self, data):
        """Imprime detalles de un frame específico"""
        print(f"\n{'─'*80}")
        print(f"🎬 FRAME {data['frame']}")
        print(f"{'─'*80}")
        print(f"📊 ENTRADA:")
        print(f"   GPS conf:     {data['gps_conf']:.3f}")
        print(f"   SLAM conf:    {data['visual_conf']:.3f}")
        print(f"   Error:        {data['error']:.2f}m")
        print(f"   Escenario:    {data['scenario']}")
        print(f"\n🧮 REGLAS HEURÍSTICAS (BASE):")
        print(f"   Regla:        {data['rule']}")
        print(f"   w_slam_base = {data['w_slam_base']:.3f} ({data['w_slam_base']*100:.1f}%)")
        print(f"   w_gps_base  = {data['w_gps_base']:.3f} ({data['w_gps_base']*100:.1f}%)")
        print(f"\n🤖 AJUSTE RL:")
        print(f"   w_slam_rl   = {data['w_slam_rl']:.3f} ({(data['w_slam_rl']-data['w_slam_base'])*100:+.1f}% ajuste)")
        print(f"   w_gps_rl    = {data['w_gps_rl']:.3f} ({(data['w_gps_rl']-data['w_gps_base'])*100:+.1f}% ajuste)")
        print(f"\n✅ PESOS FINALES (70% RL + 30% Heurística):")
        print(f"   w_slam_final = {data['w_slam_final']:.3f} ({data['w_slam_final']*100:.1f}%)")
        print(f"   w_gps_final  = {data['w_gps_final']:.3f} ({data['w_gps_final']*100:.1f}%)")
        
        # Interpretación
        if data['w_slam_final'] > 0.6:
            print(f"   💡 Decisión: Confía MÁS en SLAM")
        elif data['w_gps_final'] > 0.6:
            print(f"   💡 Decisión: Confía MÁS en GPS")
        else:
            print(f"   💡 Decisión: Balance equilibrado")
    
    def print_summary(self):
        """Imprime resumen estadístico"""
        print("="*80)
        print("📊 RESUMEN ESTADÍSTICO")
        print("="*80 + "\n")
        
        total = len(self.frame_history)
        
        print(f"📈 Total de frames analizados: {total}\n")
        
        print("🎭 Distribución de escenarios:")
        for scenario, frames in self.scenarios.items():
            count = len(frames)
            pct = (count / total * 100) if total > 0 else 0
            print(f"   {scenario:20} {count:4d} frames ({pct:5.1f}%)")
        
        print("\n" + "─"*80 + "\n")
        
        # Estadísticas por escenario
        print("📊 ANÁLISIS POR ESCENARIO:\n")
        
        for scenario_name, frames in self.scenarios.items():
            if len(frames) == 0:
                continue
            
            print(f"🎯 Escenario: {scenario_name.upper()}")
            print(f"   Cantidad: {len(frames)} frames")
            
            # Promedios
            avg_slam_base = np.mean([f['w_slam_base'] for f in frames])
            avg_slam_rl = np.mean([f['w_slam_rl'] for f in frames])
            avg_slam_final = np.mean([f['w_slam_final'] for f in frames])
            
            print(f"   Peso SLAM promedio:")
            print(f"      Base (reglas):  {avg_slam_base:.3f}")
            print(f"      RL:             {avg_slam_rl:.3f} ({(avg_slam_rl-avg_slam_base)*100:+.1f}% ajuste)")
            print(f"      Final:          {avg_slam_final:.3f}")
            
            # Mostrar ejemplo representativo
            mid_idx = len(frames) // 2
            example = frames[mid_idx]
            print(f"   Ejemplo (frame {example['frame']}):")
            print(f"      GPS conf={example['gps_conf']:.2f}, SLAM conf={example['visual_conf']:.2f}, error={example['error']:.1f}m")
            print(f"      Decisión final: SLAM={example['w_slam_final']*100:.0f}%, GPS={example['w_gps_final']*100:.0f}%")
            print()
    
    def plot_weights_evolution(self):
        """Gráfica de evolución de pesos a lo largo del tiempo"""
        
        if len(self.frame_history) == 0:
            return
        
        frames = [d['frame'] for d in self.frame_history]
        w_slam_base = [d['w_slam_base'] for d in self.frame_history]
        w_slam_rl = [d['w_slam_rl'] for d in self.frame_history]
        w_slam_final = [d['w_slam_final'] for d in self.frame_history]
        
        w_gps_base = [d['w_gps_base'] for d in self.frame_history]
        w_gps_rl = [d['w_gps_rl'] for d in self.frame_history]
        w_gps_final = [d['w_gps_final'] for d in self.frame_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Gráfica 1: Pesos SLAM
        ax1.plot(frames, w_slam_base, 'g--', linewidth=1.5, alpha=0.6, label='Base (Heurística)')
        ax1.plot(frames, w_slam_rl, 'b:', linewidth=1.5, alpha=0.7, label='RL (Aprendido)')
        ax1.plot(frames, w_slam_final, 'r-', linewidth=2.5, alpha=0.9, label='Final (Fusión)')
        ax1.set_xlabel('Frame', fontsize=12)
        ax1.set_ylabel('Peso SLAM', fontsize=12)
        ax1.set_title('Evolución de Pesos SLAM: Base → RL → Final', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Gráfica 2: Pesos GPS
        ax2.plot(frames, w_gps_base, 'g--', linewidth=1.5, alpha=0.6, label='Base (Heurística)')
        ax2.plot(frames, w_gps_rl, 'orange', linestyle=':', linewidth=1.5, alpha=0.7, label='RL (Aprendido)')
        ax2.plot(frames, w_gps_final, 'purple', linewidth=2.5, alpha=0.9, label='Final (Fusión)')
        ax2.set_xlabel('Frame', fontsize=12)
        ax2.set_ylabel('Peso GPS', fontsize=12)
        ax2.set_title('Evolución de Pesos GPS: Base → RL → Final', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'weights_evolution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gráfica de evolución guardada: {output_path}")
    
    def plot_scenarios_comparison(self):
        """Gráfica comparativa de pesos por escenario"""
        
        scenario_labels = []
        slam_base_means = []
        slam_rl_means = []
        slam_final_means = []
        
        for scenario_name, frames in self.scenarios.items():
            if len(frames) == 0:
                continue
            
            scenario_labels.append(scenario_name.replace('_', '\n'))
            slam_base_means.append(np.mean([f['w_slam_base'] for f in frames]))
            slam_rl_means.append(np.mean([f['w_slam_rl'] for f in frames]))
            slam_final_means.append(np.mean([f['w_slam_final'] for f in frames]))
        
        if len(scenario_labels) == 0:
            return
        
        x = np.arange(len(scenario_labels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.bar(x - width, slam_base_means, width, label='Base (Heurística)', alpha=0.7, color='green')
        ax.bar(x, slam_rl_means, width, label='RL (Aprendido)', alpha=0.7, color='blue')
        ax.bar(x + width, slam_final_means, width, label='Final (Fusión)', alpha=0.9, color='red')
        
        ax.set_xlabel('Escenario', fontsize=13)
        ax.set_ylabel('Peso SLAM Promedio', fontsize=13)
        ax.set_title('Comparación de Pesos SLAM por Escenario', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels, fontsize=10)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        # Añadir línea en 0.5 (balance perfecto)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Balance 50-50')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'scenarios_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gráfica de escenarios guardada: {output_path}")
    
    def save_detailed_report(self):
        """Guarda reporte detallado en texto"""
        
        report_path = os.path.join(self.output_dir, 'detailed_analysis.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ANÁLISIS DETALLADO DE PESOS Y CONFIANZAS - RL-ORB-SLAM-GPS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Modelo: {self.model_path}\n")
            f.write(f"Secuencia: {self.sequence_path}\n")
            f.write(f"Fecha: {datetime.now().isoformat()}\n\n")
            
            f.write("="*80 + "\n")
            f.write("RESUMEN ESTADÍSTICO\n")
            f.write("="*80 + "\n\n")
            
            total = len(self.frame_history)
            f.write(f"Total frames analizados: {total}\n\n")
            
            f.write("Distribución de escenarios:\n")
            for scenario, frames in self.scenarios.items():
                count = len(frames)
                pct = (count / total * 100) if total > 0 else 0
                f.write(f"  {scenario:20} {count:4d} frames ({pct:5.1f}%)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("EJEMPLOS DETALLADOS POR ESCENARIO\n")
            f.write("="*80 + "\n\n")
            
            for scenario_name, frames in self.scenarios.items():
                if len(frames) == 0:
                    continue
                
                f.write(f"\n{'─'*80}\n")
                f.write(f"Escenario: {scenario_name.upper()}\n")
                f.write(f"{'─'*80}\n\n")
                
                # Mostrar 3 ejemplos
                num_examples = min(3, len(frames))
                indices = np.linspace(0, len(frames)-1, num_examples, dtype=int)
                
                for idx in indices:
                    example = frames[idx]
                    f.write(f"Ejemplo {idx+1} - Frame {example['frame']}:\n")
                    f.write(f"  Entrada:\n")
                    f.write(f"    GPS conf:     {example['gps_conf']:.3f}\n")
                    f.write(f"    SLAM conf:    {example['visual_conf']:.3f}\n")
                    f.write(f"    Error:        {example['error']:.2f}m\n")
                    f.write(f"  Reglas heurísticas:\n")
                    f.write(f"    Regla:        {example['rule']}\n")
                    f.write(f"    w_slam_base = {example['w_slam_base']:.3f}\n")
                    f.write(f"    w_gps_base  = {example['w_gps_base']:.3f}\n")
                    f.write(f"  Ajuste RL:\n")
                    f.write(f"    w_slam_rl   = {example['w_slam_rl']:.3f} ({(example['w_slam_rl']-example['w_slam_base'])*100:+.1f}%)\n")
                    f.write(f"    w_gps_rl    = {example['w_gps_rl']:.3f} ({(example['w_gps_rl']-example['w_gps_base'])*100:+.1f}%)\n")
                    f.write(f"  Pesos finales:\n")
                    f.write(f"    w_slam_final = {example['w_slam_final']:.3f} ({example['w_slam_final']*100:.1f}%)\n")
                    f.write(f"    w_gps_final  = {example['w_gps_final']:.3f} ({example['w_gps_final']*100:.1f}%)\n")
                    f.write("\n")
        
        print(f"✅ Reporte detallado guardado: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analizar pesos y confianzas del modelo RL frame por frame',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--sequence',
        type=str,
        required=True,
        help='Número de secuencia KITTI (ej: 0048)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth',
        help='Ruta al modelo entrenado'
    )
    
    parser.add_argument(
        '--max_frames',
        type=int,
        default=50,
        help='Límite de frames a analizar (default: 50)'
    )
    
    parser.add_argument(
        '--show_every',
        type=int,
        default=5,
        help='Mostrar detalles cada N frames (default: 5)'
    )
    
    parser.add_argument(
        '--kitti_path',
        type=str,
        default='kitti_data/2011_09_26',
        help='Ruta base a secuencias KITTI'
    )
    
    args = parser.parse_args()
    
    # Construir ruta a secuencia
    sequence_path = os.path.join(
        args.kitti_path,
        f"2011_09_26_drive_{args.sequence}_sync"
    )
    
    # Verificar existencia
    if not os.path.exists(sequence_path):
        print(f"❌ ERROR: Secuencia no encontrada: {sequence_path}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"❌ ERROR: Modelo no encontrado: {args.model_path}")
        sys.exit(1)
    
    # Crear analizador y ejecutar
    analyzer = WeightsAnalyzer(
        model_path=args.model_path,
        sequence_path=sequence_path
    )
    
    analyzer.analyze(
        max_frames=args.max_frames,
        show_every_n_frames=args.show_every
    )
    
    print("\n" + "="*80)
    print("🎉 ANÁLISIS COMPLETADO")
    print("="*80)
    print(f"📂 Resultados en: {analyzer.output_dir}")
    print("\n📊 Archivos generados:")
    print("  ✅ weights_evolution.png       (evolución temporal de pesos)")
    print("  ✅ scenarios_comparison.png    (comparación por escenario)")
    print("  ✅ detailed_analysis.txt       (reporte detallado)")
    print("="*80)


if __name__ == "__main__":
    main()
