"""
Módulo de Aumentación de Datos para Entrenamiento RL

Este módulo genera escenarios de entrenamiento diversos degradando 
artificialmente los sensores para que el RL aprenda todos los casos:
- GPS malo (ruido, baja confianza)
- SLAM malo (pocos matches, baja confianza)
- Ambos malos
- Conflicto alto (ambos buenos pero inconsistentes)

Autor: Sistema Híbrido SLAM-GPS
Fecha: Octubre 2025
"""

import numpy as np


class TrainingAugmentation:
    """
    Genera escenarios de entrenamiento diversos degradando sensores artificialmente.
    
    Estrategia:
    - 40% frames normales (KITTI original - ambos buenos)
    - 20% GPS malo
    - 15% SLAM malo
    - 10% ambos malos
    - 15% conflicto alto (ambos buenos pero inconsistentes)
    """
    
    def __init__(self, corruption_probability=0.3):
        """
        Args:
            corruption_probability: Probabilidad de aplicar aumentación (0-1)
                                   0.3 = 30% de frames serán corrompidos
        """
        self.corruption_probability = corruption_probability
        
        # CORREGIDO: Balanceado para que RL vea todos los casos por igual
        self.scenarios = ['normal', 'bad_gps', 'bad_slam', 'both_bad', 'high_conflict']
        self.scenario_probs = [0.30, 0.20, 0.20, 0.15, 0.15]  # bad_slam ahora 20% (igual que bad_gps)
        
        # Estadísticas de entrenamiento
        self.stats = {
            'total_frames': 0,
            'normal': 0,
            'bad_gps': 0,
            'bad_slam': 0,
            'both_bad': 0,
            'high_conflict': 0
        }
    
    def augment_training_sample(self, gps_utm, gps_conf, visual_conf, error):
        """
        Corrompe aleatoriamente los datos para crear escenarios difíciles.
        
        Args:
            gps_utm: Posición GPS UTM [x, y, z] (np.array)
            gps_conf: Confianza GPS original [0-1]
            visual_conf: Confianza SLAM visual original [0-1]
            error: Error entre SLAM y GPS (metros)
        
        Returns:
            tuple: (gps_utm_aug, gps_conf_aug, visual_conf_aug, scenario)
                gps_utm_aug: Posición GPS aumentada (puede tener ruido)
                gps_conf_aug: Confianza GPS modificada
                visual_conf_aug: Confianza SLAM modificada
                scenario: String con el escenario aplicado
        """
        self.stats['total_frames'] += 1
        
        # Seleccionar escenario aleatoriamente
        scenario = np.random.choice(self.scenarios, p=self.scenario_probs)
        self.stats[scenario] += 1
        
        # Copias de datos originales
        gps_utm_aug = gps_utm.copy() if gps_utm is not None else None
        gps_conf_aug = gps_conf
        visual_conf_aug = visual_conf
        
        # === ESCENARIO 1: NORMAL (40%) ===
        if scenario == 'normal':
            # No modificar nada, usar datos originales de KITTI
            pass
        
        # === ESCENARIO 2: GPS MALO (20%) ===
        elif scenario == 'bad_gps':
            # Simular GPS móvil con ruido alto
            if gps_utm_aug is not None:
                # Ruido gaussiano fuerte (±5-15m)
                noise_std = np.random.uniform(5.0, 15.0)
                noise = np.random.normal(0, noise_std, size=3)
                gps_utm_aug += noise
                
                # Ocasionalmente añadir bias sistemático (deriva GPS)
                if np.random.rand() < 0.3:
                    bias = np.random.uniform(-10, 10, size=3)
                    gps_utm_aug += bias
            
            # Reducir confianza GPS drásticamente
            gps_conf_aug = np.random.uniform(0.10, 0.40)
            
            print(f"  [AUGMENT] GPS_MALO: conf={gps_conf_aug:.2f}, ruido añadido")
        
        # === ESCENARIO 3: SLAM MALO (15%) ===
        elif scenario == 'bad_slam':
            # Simular SLAM con pocos matches o mal tracking
            # (No podemos modificar la imagen, solo la confianza)
            visual_conf_aug = np.random.uniform(0.20, 0.45)
            
            print(f"  [AUGMENT] SLAM_MALO: conf={visual_conf_aug:.2f}")
        
        # === ESCENARIO 4: AMBOS MALOS (10%) ===
        elif scenario == 'both_bad':
            # Simular situación crítica: ambos sensores fallando
            
            # GPS con ruido moderado-alto
            if gps_utm_aug is not None:
                noise_std = np.random.uniform(6.0, 12.0)
                noise = np.random.normal(0, noise_std, size=3)
                gps_utm_aug += noise
            
            # Ambas confianzas bajas
            gps_conf_aug = np.random.uniform(0.10, 0.35)
            visual_conf_aug = np.random.uniform(0.20, 0.40)
            
            print(f"  [AUGMENT] AMBOS_MALOS: gps_conf={gps_conf_aug:.2f}, "
                  f"slam_conf={visual_conf_aug:.2f}")
        
        # === ESCENARIO 5: CONFLICTO ALTO (15%) ===
        elif scenario == 'high_conflict':
            # Ambos sensores "creen" que están bien, pero no coinciden
            # Simular offset sistemático en GPS (error de calibración, multipath, etc.)
            
            if gps_utm_aug is not None:
                # Offset fijo más ruido moderado
                offset = np.random.normal(0, 6.0, size=3)
                noise = np.random.normal(0, 2.0, size=3)
                gps_utm_aug += (offset + noise)
            
            # Mantener confianzas altas (ambos "piensan" que están bien)
            gps_conf_aug = np.random.uniform(0.70, 0.95)
            visual_conf_aug = np.random.uniform(0.75, 1.00)
            
            print(f"  [AUGMENT] CONFLICTO_ALTO: ambos confiables pero inconsistentes "
                  f"(gps={gps_conf_aug:.2f}, slam={visual_conf_aug:.2f})")
        
        return gps_utm_aug, gps_conf_aug, visual_conf_aug, scenario
    
    def print_statistics(self):
        """Imprime estadísticas de aumentación durante el entrenamiento"""
        total = self.stats['total_frames']
        if total == 0:
            print("No hay estadísticas de aumentación todavía")
            return
        
        print("\n" + "="*80)
        print("ESTADÍSTICAS DE AUMENTACIÓN DE DATOS")
        print("="*80)
        print(f"Total de frames procesados: {total}")
        print(f"\nDistribución de escenarios:")
        
        for scenario in self.scenarios:
            count = self.stats[scenario]
            percentage = (count / total) * 100
            bar_length = int(percentage / 2)
            bar = '█' * bar_length
            print(f"  {scenario:<15} {count:>4} ({percentage:>5.1f}%) {bar}")
        
        print("="*80 + "\n")
    
    def reset_statistics(self):
        """Reinicia las estadísticas de aumentación"""
        for key in self.stats:
            self.stats[key] = 0
