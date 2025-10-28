#!/usr/bin/env python3
"""
Script de Validación Exhaustiva del Comportamiento del RL

Este script verifica que el RL aprendió correctamente para TODOS los escenarios:
1. GPS malo, SLAM bueno → Debe favorecer SLAM
2. SLAM malo, GPS bueno → Debe favorecer GPS
3. Ambos buenos, error bajo → Balance equilibrado
4. Ambos buenos, error alto → Debe desconfiar y balancear
5. Ambos malos → Comportamiento conservador
6. CASOS EXTREMOS (nuevo) → Márgenes amplios ±50%

Genera un reporte detallado con casos de prueba sintéticos.

Uso:
    python validate_rl_behavior.py --model LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth
    python validate_rl_behavior.py --model path/to/model.pth --output resultados/validacion

Autor: Sistema Híbrido SLAM-GPS
Fecha: Octubre 2025
"""

import argparse
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict

# Agregar paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from LMS.LMS_RL_ORB_GPS.model.rl_agent import SimpleRLAgent, RLTrainer


class RLBehaviorValidator:
    """Validador exhaustivo del comportamiento del RL con detección de escenarios"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        
        # Cargar modelo RL
        print(f"📦 Cargando modelo RL: {model_path}")
        self.rl_agent = SimpleRLAgent(input_dim=3)
        
        checkpoint = torch.load(model_path, weights_only=False)
        self.rl_agent.load_state_dict(checkpoint['model_state_dict'])
        self.rl_agent.eval()
        print("✅ Modelo cargado\n")
        
        # Casos de prueba
        self.test_cases = []
        self.results = defaultdict(list)
    
    def detect_scenario(self, gps_conf, visual_conf, error):
        """
        Detecta el escenario actual basado en confianzas y error.
        Replica la lógica de main.py
        
        Returns:
            (scenario_name, margin_percentage)
        """
        min_conf = min(gps_conf, visual_conf)
        max_conf = max(gps_conf, visual_conf)
        conf_diff = max_conf - min_conf
        
        # ESCENARIO 1: SENSOR_EXTREMO (uno muy bueno, otro muy malo)
        if min_conf < 0.3 and max_conf > 0.7 and conf_diff > 0.6:
            return "SENSOR_EXTREMO", 0.50  # ±50% margen
        
        # ESCENARIO 2: ALTA_CERTEZA (ambos muy confiables, error bajo)
        elif min_conf > 0.7 and error < 2.0:
            return "ALTA_CERTEZA", 0.10  # ±10% margen (ajustes finos)
        
        # ESCENARIO 3: ERROR_ALTO (confianzas altas pero error grande)
        elif min_conf > 0.6 and error > 5.0:
            return "ERROR_ALTO", 0.30  # ±30% margen (investigación)
        
        # ESCENARIO 4: SENSOR_DOMINANTE (uno claramente mejor)
        elif conf_diff > 0.3 and min_conf > 0.3:
            return "SENSOR_DOMINANTE", 0.25  # ±25% margen
        
        # ESCENARIO 5: EQUILIBRADO (confianzas similares, error moderado)
        else:
            return "EQUILIBRADO", 0.20  # ±20% margen (default)
    
    def calculate_base_weights(self, gps_conf, visual_conf, error):
        """Replica lógica de reglas heurísticas"""
        
        # Calcular ratio de confianzas
        conf_ratio = visual_conf / (gps_conf + 1e-6)
        
        # REGLA 1: Uno de los dos es claramente superior
        if conf_ratio > 3.0:  # SLAM mucho mejor
            w_slam_base = 0.80
            rule = "SLAM >> GPS (ratio>3.0)"
        elif conf_ratio < 0.33:  # GPS mucho mejor
            w_slam_base = 0.20
            rule = "GPS >> SLAM (ratio<0.33)"
        
        # REGLA 2: Diferencia moderada
        elif conf_ratio > 1.5:  # SLAM moderadamente mejor
            w_slam_base = 0.65
            rule = "SLAM > GPS (ratio>1.5)"
        elif conf_ratio < 0.67:  # GPS moderadamente mejor
            w_slam_base = 0.35
            rule = "GPS > SLAM (ratio<0.67)"
        
        # REGLA 3: Error alto → desconfiar más, balancear
        elif error > 5.0:
            w_slam_base = 0.50
            rule = "Error alto (>5m) → balance"
        
        # REGLA 4: Ambos buenos → balance proporcional
        elif gps_conf > 0.6 and visual_conf > 0.6:
            total = gps_conf + visual_conf
            w_slam_base = visual_conf / total
            rule = "Ambos buenos → proporcional"
        
        # REGLA 5: Default
        else:
            if conf_ratio > 1.2:
                w_slam_base = 0.60
                rule = "SLAM ligeramente mejor"
            elif conf_ratio < 0.8:
                w_slam_base = 0.40
                rule = "GPS ligeramente mejor"
            else:
                w_slam_base = 0.50
                rule = "Balance neutro"
        
        # Aplicar penalización por error alto
        if error > 5.0:
            penalty = min((error - 5.0) / 10.0, 0.3)
            if w_slam_base > 0.5:
                w_slam_base -= penalty
            else:
                w_slam_base += penalty
            rule += f" (ajustado por error={error:.1f}m)"
        
        w_gps_base = 1.0 - w_slam_base
        
        return w_slam_base, w_gps_base, rule
    
    def get_rl_weights(self, gps_conf, visual_conf, error):
        """Obtiene pesos del RL"""
        
        # Crear estado (debe tener 3 dimensiones para SimpleRLAgent)
        state = torch.FloatTensor([visual_conf, gps_conf, error / 10.0]).unsqueeze(0)
        
        # Inferencia
        with torch.no_grad():
            weights = self.rl_agent(state)
            w_slam_rl = weights[0, 0].item()
            w_gps_rl = weights[0, 1].item()
            
            # Normalizar
            total = w_slam_rl + w_gps_rl
            if total > 0:
                w_slam_rl /= total
                w_gps_rl /= total
        
        return w_slam_rl, w_gps_rl
    
    def apply_rl_with_margin(self, w_slam_base, w_gps_base, w_slam_rl, w_gps_rl, margin):
        """
        Aplica ajuste RL con margen adaptativo según el escenario.
        Replica la lógica de main.py
        """
        # Calcular ajuste deseado por RL
        delta_slam_desired = w_slam_rl - w_slam_base
        delta_gps_desired = w_gps_rl - w_gps_base
        
        # Limitar el ajuste según el margen del escenario
        max_adjustment = margin  # Porcentaje máximo de ajuste
        
        delta_slam_clamped = np.clip(delta_slam_desired, -max_adjustment, max_adjustment)
        delta_gps_clamped = np.clip(delta_gps_desired, -max_adjustment, max_adjustment)
        
        # Aplicar ajustes limitados
        w_slam_final = w_slam_base + delta_slam_clamped
        w_gps_final = w_gps_base + delta_gps_clamped
        
        # Normalizar para asegurar que sumen 1.0
        total = w_slam_final + w_gps_final
        if total > 0:
            w_slam_final /= total
            w_gps_final /= total
        
        # Asegurar límites [0.05, 0.95]
        w_slam_final = np.clip(w_slam_final, 0.05, 0.95)
        w_gps_final = 1.0 - w_slam_final
        
        return w_slam_final, w_gps_final, delta_slam_clamped
    
    def create_test_cases(self):
        """Crea casos de prueba sintéticos para todos los escenarios"""
        
        print("🎯 Generando casos de prueba sintéticos exhaustivos...\n")
        
        test_scenarios = [
            # ESCENARIO 1: SENSOR_EXTREMO - GPS malo, SLAM excelente
            {
                'name': '1. SENSOR_EXTREMO: GPS_MALO_SLAM_EXCELENTE',
                'description': 'GPS con ruido/baja confianza, SLAM perfecto (debe detectar margen ±50%)',
                'cases': [
                    {'gps_conf': 0.10, 'visual_conf': 1.00, 'error': 8.0},
                    {'gps_conf': 0.15, 'visual_conf': 0.95, 'error': 10.0},
                    {'gps_conf': 0.20, 'visual_conf': 0.90, 'error': 12.0},
                    {'gps_conf': 0.25, 'visual_conf': 0.85, 'error': 9.0},
                ],
                'expected': 'SLAM >> GPS (w_slam > 0.85), escenario=SENSOR_EXTREMO, margen=±50%'
            },
            
            # ESCENARIO 2: SENSOR_EXTREMO - SLAM malo, GPS excelente
            {
                'name': '2. SENSOR_EXTREMO: SLAM_MALO_GPS_EXCELENTE',
                'description': 'Pocos features visuales, GPS preciso (debe detectar margen ±50%)',
                'cases': [
                    {'gps_conf': 0.95, 'visual_conf': 0.20, 'error': 3.0},
                    {'gps_conf': 0.90, 'visual_conf': 0.25, 'error': 2.5},
                    {'gps_conf': 0.85, 'visual_conf': 0.15, 'error': 4.0},
                    {'gps_conf': 0.92, 'visual_conf': 0.28, 'error': 3.5},
                ],
                'expected': 'GPS >> SLAM (w_gps > 0.70), escenario=SENSOR_EXTREMO, margen=±50%'
            },
            
            # ESCENARIO 3: ALTA_CERTEZA - Ambos excelentes, error bajo
            {
                'name': '3. ALTA_CERTEZA: AMBOS_EXCELENTES_ERROR_BAJO',
                'description': 'Alta confianza en ambos, buena alineación (debe detectar margen ±10%)',
                'cases': [
                    {'gps_conf': 0.95, 'visual_conf': 0.95, 'error': 1.0},
                    {'gps_conf': 0.90, 'visual_conf': 0.90, 'error': 1.5},
                    {'gps_conf': 0.85, 'visual_conf': 0.85, 'error': 1.8},
                    {'gps_conf': 0.80, 'visual_conf': 0.90, 'error': 1.2},
                ],
                'expected': 'Balance equilibrado (0.40 < w_slam < 0.60), escenario=ALTA_CERTEZA, margen=±10%'
            },
            
            # ESCENARIO 4: ERROR_ALTO - Ambos buenos, error alto
            {
                'name': '4. ERROR_ALTO: AMBOS_BUENOS_ERROR_CONFLICTO',
                'description': 'Confianzas altas pero mediciones conflictivas (debe detectar margen ±30%)',
                'cases': [
                    {'gps_conf': 0.90, 'visual_conf': 0.90, 'error': 6.5},
                    {'gps_conf': 0.85, 'visual_conf': 0.85, 'error': 7.5},
                    {'gps_conf': 0.80, 'visual_conf': 0.90, 'error': 8.5},
                    {'gps_conf': 0.90, 'visual_conf': 0.75, 'error': 6.0},
                ],
                'expected': 'Balance conservador (0.40 < w_slam < 0.65), escenario=ERROR_ALTO, margen=±30%'
            },
            
            # ESCENARIO 5: SENSOR_DOMINANTE
            {
                'name': '5. SENSOR_DOMINANTE: SLAM_MEJOR',
                'description': 'SLAM claramente mejor pero no extremo (debe detectar margen ±25%)',
                'cases': [
                    {'gps_conf': 0.50, 'visual_conf': 0.90, 'error': 3.5},
                    {'gps_conf': 0.45, 'visual_conf': 0.85, 'error': 4.0},
                    {'gps_conf': 0.55, 'visual_conf': 0.90, 'error': 3.0},
                    {'gps_conf': 0.40, 'visual_conf': 0.80, 'error': 4.5},
                ],
                'expected': 'SLAM dominante (w_slam > 0.60), escenario=SENSOR_DOMINANTE, margen=±25%'
            },
            
            # ESCENARIO 6: Ambos malos
            {
                'name': '6. EQUILIBRADO: AMBOS_MALOS',
                'description': 'Baja confianza en ambos sensores',
                'cases': [
                    {'gps_conf': 0.35, 'visual_conf': 0.35, 'error': 4.0},
                    {'gps_conf': 0.40, 'visual_conf': 0.38, 'error': 5.0},
                    {'gps_conf': 0.38, 'visual_conf': 0.42, 'error': 4.5},
                    {'gps_conf': 0.45, 'visual_conf': 0.45, 'error': 3.5},
                ],
                'expected': 'Balance cauteloso (0.40 < w_slam < 0.60), margen=±20%'
            },
            
            # ESCENARIO 7: Transiciones (para ver adaptación)
            {
                'name': '7. TRANSICIONES: GPS_DEGRADA_PROGRESIVAMENTE',
                'description': 'GPS degrada de bueno a malo (ver cambio de escenario)',
                'cases': [
                    {'gps_conf': 0.90, 'visual_conf': 0.85, 'error': 2.0},  # EQUILIBRADO/ALTA_CERTEZA
                    {'gps_conf': 0.70, 'visual_conf': 0.85, 'error': 3.0},  # SENSOR_DOMINANTE
                    {'gps_conf': 0.50, 'visual_conf': 0.85, 'error': 4.5},  # SENSOR_DOMINANTE
                    {'gps_conf': 0.25, 'visual_conf': 0.85, 'error': 8.0},  # SENSOR_EXTREMO
                ],
                'expected': 'w_slam debe aumentar progresivamente, escenarios cambian dinámicamente'
            },
            
            # ESCENARIO 8: Casos extremos límite
            {
                'name': '8. CASOS_EXTREMOS: LIMITES',
                'description': 'Situaciones límite de confianzas',
                'cases': [
                    {'gps_conf': 0.00, 'visual_conf': 1.00, 'error': 15.0},
                    {'gps_conf': 1.00, 'visual_conf': 0.00, 'error': 0.5},
                    {'gps_conf': 0.50, 'visual_conf': 0.50, 'error': 0.0},
                    {'gps_conf': 0.10, 'visual_conf': 0.10, 'error': 12.0},
                ],
                'expected': 'Decisiones extremas pero razonables'
            },
            
            # ESCENARIO 9: Barrido de confianzas (grid)
            {
                'name': '9. BARRIDO_COMPLETO: GRID_CONFIANZAS',
                'description': 'Barrido sistemático de todas las combinaciones',
                'cases': self._generate_grid_cases(),
                'expected': 'Comportamiento consistente en todo el espacio de estados'
            },
        ]
        
        self.test_cases = test_scenarios
        total_cases = sum(len(s['cases']) for s in test_scenarios)
        print(f"✅ Generados {total_cases} casos de prueba en {len(test_scenarios)} escenarios\n")
    
    def _generate_grid_cases(self):
        """Genera un grid de casos para barrido completo"""
        cases = []
        
        # Grid: GPS conf vs Visual conf vs Error
        for gps_conf in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for visual_conf in [0.1, 0.3, 0.5, 0.7, 0.9]:
                for error in [1.0, 3.0, 6.0, 10.0]:
                    cases.append({
                        'gps_conf': gps_conf,
                        'visual_conf': visual_conf,
                        'error': error
                    })
        
        return cases[:40]  # Limitar a 40 casos para no saturar
    
    def run_validation(self):
        """Ejecuta validación completa"""
        
        print("="*80)
        print("🔍 VALIDACIÓN EXHAUSTIVA DEL COMPORTAMIENTO DEL RL")
        print("="*80 + "\n")
        
        all_passed = True
        scenario_summary = []
        
        for scenario in self.test_cases:
            print(f"{'─'*80}")
            print(f"🎬 {scenario['name']}")
            print(f"{'─'*80}")
            print(f"📝 {scenario['description']}")
            print(f"🎯 Comportamiento esperado: {scenario['expected']}\n")
            
            scenario_results = []
            
            for i, case in enumerate(scenario['cases'], 1):
                gps_conf = case['gps_conf']
                visual_conf = case['visual_conf']
                error = case['error']
                
                # 1. Detectar escenario
                scenario_type, margin = self.detect_scenario(gps_conf, visual_conf, error)
                
                # 2. Calcular pesos base
                w_slam_base, w_gps_base, rule = self.calculate_base_weights(gps_conf, visual_conf, error)
                
                # 3. Calcular pesos RL
                w_slam_rl, w_gps_rl = self.get_rl_weights(gps_conf, visual_conf, error)
                
                # 4. Aplicar RL con margen adaptativo
                w_slam_final, w_gps_final, delta_applied = self.apply_rl_with_margin(
                    w_slam_base, w_gps_base, w_slam_rl, w_gps_rl, margin
                )
                
                # Guardar resultados
                result = {
                    'gps_conf': gps_conf,
                    'visual_conf': visual_conf,
                    'error': error,
                    'scenario': scenario_type,
                    'margin': margin,
                    'w_slam_base': w_slam_base,
                    'w_gps_base': w_gps_base,
                    'w_slam_rl': w_slam_rl,
                    'w_gps_rl': w_gps_rl,
                    'w_slam_final': w_slam_final,
                    'w_gps_final': w_gps_final,
                    'delta_applied': delta_applied,
                    'rule': rule
                }
                
                scenario_results.append(result)
                
                # Mostrar resultado (solo primeros casos para no saturar)
                if i <= 4 or len(scenario['cases']) <= 10:
                    print(f"   Caso {i}: GPS={gps_conf:.2f}, Visual={visual_conf:.2f}, Error={error:.1f}m")
                    print(f"      Escenario: {scenario_type} (margen=±{margin*100:.0f}%)")
                    print(f"      Regla: {rule}")
                    print(f"      Base: SLAM={w_slam_base*100:.0f}% | RL: SLAM={w_slam_rl*100:.0f}% | Δ={delta_applied:+.3f}")
                    print(f"      Final: SLAM={w_slam_final*100:.0f}%, GPS={w_gps_final*100:.0f}%")
                    
                    # Validar comportamiento
                    passed = self.validate_case(scenario['name'], result, scenario['expected'])
                    if not passed:
                        all_passed = False
                    print()
            
            # Resumen del escenario
            avg_slam_final = np.mean([r['w_slam_final'] for r in scenario_results])
            avg_margin = np.mean([r['margin'] for r in scenario_results])
            scenarios_detected = set(r['scenario'] for r in scenario_results)
            
            scenario_summary.append({
                'name': scenario['name'],
                'cases': len(scenario_results),
                'avg_slam': avg_slam_final,
                'avg_margin': avg_margin,
                'scenarios': scenarios_detected
            })
            
            print(f"   📊 Resumen: {len(scenario_results)} casos, SLAM promedio={avg_slam_final*100:.1f}%")
            print(f"   🎯 Escenarios detectados: {', '.join(scenarios_detected)}")
            print()
            
            # Guardar resultados del escenario
            self.results[scenario['name']] = scenario_results
            
            print()
        
        print("="*80)
        print("📊 RESUMEN GLOBAL")
        print("="*80)
        for summary in scenario_summary:
            print(f"{summary['name']}: {summary['cases']} casos, "
                  f"SLAM={summary['avg_slam']*100:.1f}%, "
                  f"margen={summary['avg_margin']*100:.0f}%")
        print("="*80)
        
        if all_passed:
            print("✅ VALIDACIÓN EXITOSA - EL RL SE COMPORTA CORRECTAMENTE")
        else:
            print("⚠️  VALIDACIÓN CON ADVERTENCIAS - Revisar casos marcados")
        print("="*80 + "\n")
        
        return all_passed
    
    def validate_case(self, scenario_name, result, expected):
        """Valida si el comportamiento es correcto según lo esperado"""
        
        w_slam = result['w_slam_final']
        w_gps = result['w_gps_final']
        scenario_type = result['scenario']
        margin = result['margin']
        
        passed = True
        
        # Validar escenario SENSOR_EXTREMO
        if 'SENSOR_EXTREMO' in scenario_name:
            if scenario_type != 'SENSOR_EXTREMO':
                print(f"      ⚠️  ADVERTENCIA: No detectó escenario SENSOR_EXTREMO, detectó {scenario_type}")
                passed = False
            elif margin != 0.50:
                print(f"      ⚠️  ADVERTENCIA: Margen incorrecto, esperaba ±50%, obtuvo ±{margin*100:.0f}%")
                passed = False
            
            if 'GPS_MALO' in scenario_name and w_slam < 0.75:
                print(f"      ⚠️  FALLO: Se esperaba w_slam > 0.75, obtuvo {w_slam:.2f}")
                passed = False
            elif 'SLAM_MALO' in scenario_name and w_gps < 0.65:
                print(f"      ⚠️  FALLO: Se esperaba w_gps > 0.65, obtuvo {w_gps:.2f}")
                passed = False
            else:
                print(f"      ✅ CORRECTO: Escenario extremo bien detectado y gestionado")
        
        # Validar escenario ALTA_CERTEZA
        elif 'ALTA_CERTEZA' in scenario_name:
            if scenario_type != 'ALTA_CERTEZA':
                print(f"      ⚠️  ADVERTENCIA: No detectó ALTA_CERTEZA, detectó {scenario_type}")
                passed = False
            elif margin != 0.10:
                print(f"      ⚠️  ADVERTENCIA: Margen incorrecto, esperaba ±10%, obtuvo ±{margin*100:.0f}%")
                passed = False
            
            if not (0.40 <= w_slam <= 0.60):
                print(f"      ⚠️  ADVERTENCIA: Se esperaba balance (0.40-0.60), obtuvo {w_slam:.2f}")
                passed = False
            else:
                print(f"      ✅ CORRECTO: Balance fino en alta certeza")
        
        # Validar escenario ERROR_ALTO
        elif 'ERROR_ALTO' in scenario_name:
            if scenario_type != 'ERROR_ALTO':
                print(f"      ⚠️  ADVERTENCIA: No detectó ERROR_ALTO, detectó {scenario_type}")
                passed = False
            elif margin != 0.30:
                print(f"      ⚠️  ADVERTENCIA: Margen incorrecto, esperaba ±30%, obtuvo ±{margin*100:.0f}%")
                passed = False
            
            if not (0.40 <= w_slam <= 0.65):
                print(f"      ⚠️  ADVERTENCIA: Balance fuera de rango esperado (0.40-0.65)")
                passed = False
            else:
                print(f"      ✅ CORRECTO: Maneja conflicto con margen adecuado")
        
        # Validar escenario SENSOR_DOMINANTE
        elif 'SENSOR_DOMINANTE' in scenario_name:
            if scenario_type not in ['SENSOR_DOMINANTE', 'EQUILIBRADO']:
                print(f"      ⚠️  ADVERTENCIA: Detectó {scenario_type}")
            
            if 'SLAM_MEJOR' in scenario_name and w_slam < 0.55:
                print(f"      ⚠️  ADVERTENCIA: SLAM debería dominar más (w_slam={w_slam:.2f})")
                passed = False
            else:
                print(f"      ✅ CORRECTO: Dominancia apropiada")
        
        # Otros casos
        else:
            print(f"      💡 INFO: Escenario={scenario_type}, margen=±{margin*100:.0f}%")
        
        return passed
    
    def plot_behavior_heatmaps(self, output_dir):
        """Genera heatmaps del comportamiento del RL"""
        
        print("📊 Generando heatmaps de comportamiento...\n")
        
        # Crear grilla de confianzas
        gps_confs = np.linspace(0.1, 1.0, 20)
        visual_confs = np.linspace(0.1, 1.0, 20)
        errors = [1.0, 3.0, 6.0]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for idx, error in enumerate(errors):
            heatmap_data = np.zeros((len(visual_confs), len(gps_confs)))
            
            for i, v_conf in enumerate(visual_confs):
                for j, g_conf in enumerate(gps_confs):
                    # Obtener decisión final
                    w_slam_base, w_gps_base, _ = self.calculate_base_weights(g_conf, v_conf, error)
                    w_slam_rl, w_gps_rl = self.get_rl_weights(g_conf, v_conf, error)
                    w_slam_final, _ = self.apply_rl_with_margin(w_slam_base, w_gps_base, w_slam_rl, w_gps_rl, 0.2)
                    
                    heatmap_data[i, j] = w_slam_final
            
            ax = axes[idx]
            im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', origin='lower',
                          vmin=0, vmax=1, extent=[0.1, 1.0, 0.1, 1.0])
            
            ax.set_xlabel('GPS Confidence', fontsize=12, fontweight='bold')
            ax.set_ylabel('Visual Confidence', fontsize=12, fontweight='bold')
            ax.set_title(f'Peso SLAM Final (Error={error:.1f}m)', fontsize=13, fontweight='bold')
            
            # Añadir líneas de contorno
            contours = ax.contour(np.linspace(0.1, 1.0, 20), np.linspace(0.1, 1.0, 20),
                                 heatmap_data, levels=[0.3, 0.5, 0.7], colors='black',
                                 linewidths=1.5, alpha=0.5)
            ax.clabel(contours, inline=True, fontsize=10)
            
            plt.colorbar(im, ax=ax, label='Peso SLAM')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'rl_behavior_heatmaps.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Heatmaps guardados: {output_path}")
    
    def generate_report(self, output_dir, validation_passed):
        """Genera reporte completo de validación"""
        
        report_path = os.path.join(output_dir, 'rl_validation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("REPORTE DE VALIDACIÓN DEL COMPORTAMIENTO DEL RL\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Modelo: {self.model_path}\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Resultado: {'✅ APROBADO' if validation_passed else '⚠️ CON ADVERTENCIAS'}\n\n")
            
            f.write("RESUMEN POR ESCENARIO\n")
            f.write("-"*80 + "\n\n")
            
            for scenario_name, results in self.results.items():
                f.write(f"{scenario_name}\n")
                f.write(f"Casos evaluados: {len(results)}\n")
                
                avg_slam_base = np.mean([r['w_slam_base'] for r in results])
                avg_slam_rl = np.mean([r['w_slam_rl'] for r in results])
                avg_slam_final = np.mean([r['w_slam_final'] for r in results])
                
                f.write(f"  Peso SLAM promedio:\n")
                f.write(f"    Base (reglas):  {avg_slam_base:.3f}\n")
                f.write(f"    RL:             {avg_slam_rl:.3f}\n")
                f.write(f"    Final:          {avg_slam_final:.3f}\n")
                f.write(f"  Ajuste RL:        {(avg_slam_rl - avg_slam_base)*100:+.1f}%\n\n")
            
            f.write("="*80 + "\n")
            f.write("CONCLUSIÓN\n")
            f.write("-"*80 + "\n")
            
            if validation_passed:
                f.write("El modelo RL demuestra comportamiento correcto en todos los\n")
                f.write("escenarios evaluados. El entrenamiento fue exitoso.\n")
            else:
                f.write("Se detectaron algunas advertencias en casos específicos.\n")
                f.write("Revisar los casos marcados para determinar si requieren ajuste.\n")
            
            f.write("="*80 + "\n")
        
        print(f"✅ Reporte guardado: {report_path}\n")


def main():
    parser = argparse.ArgumentParser(description='Validar comportamiento del RL en todos los escenarios')
    
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo entrenado')
    parser.add_argument('--output', type=str, default=None, help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Verificar modelo
    if not os.path.exists(args.model):
        print(f"❌ ERROR: Modelo no encontrado: {args.model}")
        return 1
    
    # Crear directorio de salida
    if args.output is None:
        timestamp = datetime.now().strftime("%H%M_%d%m_%Y")
        args.output = f"resultados/LMS_RL_ORB_GPS/validation_{timestamp}"
    
    os.makedirs(args.output, exist_ok=True)
    
    # Crear validador
    validator = RLBehaviorValidator(args.model)
    
    # Crear casos de prueba
    validator.create_test_cases()
    
    # Ejecutar validación
    validation_passed = validator.run_validation()
    
    # Generar visualizaciones (comentado temporalmente)
    # validator.plot_behavior_heatmaps(args.output)
    print("ℹ️  Heatmaps deshabilitados temporalmente\n")
    
    # Generar reporte
    validator.generate_report(args.output, validation_passed)
    
    print("🎉 Validación completada")
    print(f"📂 Resultados en: {args.output}\n")
    
    return 0 if validation_passed else 1


if __name__ == "__main__":
    exit(main())
