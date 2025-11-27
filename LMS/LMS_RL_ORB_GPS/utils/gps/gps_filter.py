import numpy as np

class GPSFilter:
    def __init__(self, window_size=5, movement_variance_threshold=0.1, 
                 add_noise=False, noise_std=5.0):
        """
        Filtro de GPS con capacidad de simular GPS móvil
        
        Args:
            window_size: Tamaño de ventana para suavizado
            movement_variance_threshold: Umbral de varianza RESIDUAL para detectar ruido (no movimiento)
            add_noise: Si True, agrega ruido gaussiano para simular GPS móvil
            noise_std: Desviación estándar del ruido en metros (default: 5.0m)
        """
        self.window_size = window_size
        self.movement_variance_threshold = movement_variance_threshold
        self.add_noise = add_noise
        self.noise_std = noise_std
        
        self.measurements = []
        self.filtered_position = None
        self.last_confidence = 0.5
        self.last_residual_variance = 0.0  # Para debug
        
        # NUEVO: Para detección automática de calidad
        self.all_measurements = []  # Historial completo para análisis
        self.quality_detection_done = False
        self.is_high_precision = None
        self.quality_metrics = {}
    
    def add_measurement(self, gps_position):
        """Agrega medición GPS y opcionalmente añade ruido"""
        if self.add_noise:
            # Simular GPS móvil: agregar ruido gaussiano en X, Y
            noise = np.random.normal(0, self.noise_std, size=gps_position.shape)
            # Solo ruido horizontal (X, Y), no en Z
            noise[2] = 0.0
            noisy_position = gps_position + noise
            
            # Debug: mostrar el efecto del ruido
            if len(self.measurements) % 20 == 0:  # Cada 20 mediciones
                error = np.linalg.norm(noise[:2])
                print(f"  [GPS Móvil Simulado] Error agregado: {error:.2f}m")
            
            self.measurements.append(noisy_position)
            self.all_measurements.append(noisy_position.copy())  # NUEVO
        else:
            self.measurements.append(gps_position)
            self.all_measurements.append(gps_position.copy())  # NUEVO
        
        # Mantener solo las últimas N mediciones
        if len(self.measurements) > self.window_size:
            self.measurements.pop(0)
        
        # Calcular posición filtrada (media móvil)
        self.filtered_position = np.mean(self.measurements, axis=0)
    
    def calculate_confidence(self):
        """
        Calcula confianza basada en VARIANZA RESIDUAL (ruido), no en varianza de posiciones.
        
        Lógica:
        - Calcula el movimiento promedio (velocidad esperada)
        - Calcula desviaciones respecto a ese movimiento
        - Alta varianza residual = GPS ruidoso = Baja confianza
        - Baja varianza residual = GPS preciso = Alta confianza
        """
        if len(self.measurements) < 3:
            # No hay suficientes mediciones para calcular tendencia
            return 0.5
        
        # Convertir a array numpy
        recent_measurements = np.array(self.measurements[-self.window_size:])
        
        # Calcular diferencias consecutivas (velocidades instantáneas)
        velocities = np.diff(recent_measurements, axis=0)  # Shape: (N-1, 3)
        
        if len(velocities) < 2:
            return 0.5
        
        # Calcular velocidad promedio (movimiento esperado)
        mean_velocity = np.mean(velocities, axis=0)
        
        # Calcular desviaciones respecto al movimiento promedio
        deviations = velocities - mean_velocity
        
        # Varianza residual (solo X, Y)
        residual_variance_xy = np.var(deviations[:, :2], axis=0)
        total_residual_variance = np.sum(residual_variance_xy)
        
        self.last_residual_variance = total_residual_variance
        
        # Calcular confianza basada en varianza residual
        if self.add_noise:
            # Con GPS móvil simulado, esperamos varianza residual ~ noise_std^2
            # Confianza alta si varianza residual < noise_std^2
            expected_variance = self.noise_std ** 2
            
            # Confianza inversamente proporcional a la varianza residual
            confidence = 1.0 / (1.0 + total_residual_variance / expected_variance)
            
        else:
            # GPS preciso de KITTI (sin ruido)
            # Esperamos muy baja varianza residual (< 0.1 m^2)
            # Si la varianza es baja → Alta confianza
            # Si la varianza es alta → GPS con problemas
            
            if total_residual_variance < 0.05:
                # Varianza muy baja = GPS excelente
                confidence = 0.95
            elif total_residual_variance < 0.2:
                # Varianza baja = GPS bueno
                confidence = 0.85
            elif total_residual_variance < 0.5:
                # Varianza moderada = GPS aceptable
                confidence = 0.70
            elif total_residual_variance < 1.0:
                # Varianza alta = GPS regular
                confidence = 0.50
            else:
                # Varianza muy alta = GPS malo
                confidence = 0.30
        
        # Limitar entre 0.1 y 1.0
        confidence = np.clip(confidence, 0.1, 1.0)
        
        self.last_confidence = confidence
        return confidence
    
    def get_filtered_position(self):
        """Retorna la posición filtrada"""
        return self.filtered_position if self.filtered_position is not None else None
    
    def print_debug_info(self):
        """Imprime información de debug mejorada"""
        if len(self.measurements) > 0:
            mode = "GPS MÓVIL" if self.add_noise else "GPS PRECISO"
            
            # Clasificar calidad del GPS basado en confianza
            if self.last_confidence > 0.8:
                quality = "EXCELENTE"
            elif self.last_confidence > 0.6:
                quality = "BUENO"
            elif self.last_confidence > 0.4:
                quality = "REGULAR"
            else:
                quality = "MALO"
            
            print(f"  [Filtro GPS - {mode}] Confianza: {self.last_confidence:.3f} ({quality}), "
                  f"Varianza Residual: {self.last_residual_variance:.4f}, "
                  f"Mediciones: {len(self.measurements)}")
    
    def detect_gps_quality(self, min_samples=20):
        """
        Detecta automáticamente si el GPS es de alta precisión (KITTI-like)
        o baja precisión (móvil) basándose en características observables.
        
        Args:
            min_samples: Número mínimo de mediciones necesarias para detección
            
        Returns:
            dict con resultados de detección
        """
        if len(self.all_measurements) < min_samples:
            return {
                'detected': False,
                'reason': f'Necesita al menos {min_samples} mediciones (tiene {len(self.all_measurements)})'
            }
        
        gps_array = np.array(self.all_measurements)
        
        # ============================================================
        # METRICA 1: Jitter (variación frame-a-frame)
        # ============================================================
        # GPS de alta precisión: jitter < 0.5m
        # GPS móvil: jitter típicamente 2-10m
        velocities = np.diff(gps_array, axis=0)
        velocity_magnitudes = np.linalg.norm(velocities[:, :2], axis=1)  # Solo X, Y
        
        # Aceleración (segunda derivada)
        accelerations = np.diff(velocities, axis=0)
        acceleration_magnitudes = np.linalg.norm(accelerations[:, :2], axis=1)
        
        jitter = np.std(velocity_magnitudes)
        avg_acceleration = np.mean(acceleration_magnitudes)
        
        # ============================================================
        # METRICA 2: Magnitud de saltos repentinos
        # ============================================================
        # GPS móvil tiende a tener "saltos" grandes repentinos
        median_velocity = np.median(velocity_magnitudes)
        sudden_jumps = velocity_magnitudes > (median_velocity * 3 + 0.5)  # +0.5 para evitar división por 0
        jump_frequency = np.sum(sudden_jumps) / len(sudden_jumps)
        max_jump = np.max(velocity_magnitudes)
        
        # ============================================================
        # METRICA 3: Suavidad de trayectoria (curvatura)
        # ============================================================
        # GPS de alta precisión tiene trayectorias más suaves
        curvatures = []
        for i in range(1, len(velocities) - 1):
            v1 = velocities[i-1][:2]  # Solo X, Y
            v2 = velocities[i][:2]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0.01 and norm2 > 0.01:  # Evitar división por 0
                cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
                angle = np.arccos(cos_angle)
                curvatures.append(angle)
        
        avg_curvature = np.mean(curvatures) if len(curvatures) > 0 else 0
        
        # ============================================================
        # METRICA 4: Ratio señal/ruido
        # ============================================================
        signal = np.mean(velocity_magnitudes)  # Movimiento real
        noise = np.std(acceleration_magnitudes)  # Ruido (cambios bruscos)
        snr = signal / (noise + 1e-6)
        
        # ============================================================
        # FIX: UMBRALES AJUSTADOS PARA KITTI CON MANIOBRAS AGRESIVAS
        # ============================================================
        # Umbrales calibrados con KITTI (incluyendo secuencias con giros)
        JITTER_THRESHOLD = 0.5
        JUMP_FREQ_THRESHOLD = 0.35  # ← AUMENTADO de 0.1 a 0.35 (35%)
        MAX_JUMP_THRESHOLD = 2.5    # ← AUMENTADO de 2.0 a 2.5m
        ACCELERATION_THRESHOLD = 0.5  # ← AUMENTADO de 0.3 a 0.5
        SNR_THRESHOLD = 2.0
        
        # ============================================================
        # FIX: REGLA ESPECIAL PARA KITTI CON MANIOBRAS
        # ============================================================
        # Si SNR es alto (>5) pero jump_frequency es moderado (20-40%),
        # probablemente es KITTI con maniobras agresivas (NO GPS móvil)
        
        special_rule_applied = False
        if (snr > 5.0 and 
            jump_frequency < 0.40 and 
            jitter < 0.5):
            # Es KITTI con movimiento agresivo, NO GPS móvil
            special_rule_applied = True
            print(f"\n[GPS QUALITY] ⚙️ Regla especial aplicada:")
            print(f"  • SNR alto: {snr:.2f} > 5.0 → Señal limpia")
            print(f"  • Jitter bajo: {jitter:.3f}m < 0.5m → GPS estable")
            print(f"  • Jump frequency moderado: {jump_frequency*100:.1f}% < 40%")
            print(f"  → Clasificado como ALTA PRECISIÓN (KITTI con maniobras)")
        
        # ============================================================
        # DECISION: Alta vs Baja precisión
        # ============================================================
        
        conditions_met = 0
        total_conditions = 5
        
        # Condición 1: Jitter bajo
        if jitter < JITTER_THRESHOLD:
            conditions_met += 1
        
        # Condición 2: Jump frequency (con regla especial)
        if special_rule_applied or jump_frequency < JUMP_FREQ_THRESHOLD:
            conditions_met += 1
        
        # Condición 3: Max jump pequeño
        if max_jump < MAX_JUMP_THRESHOLD:
            conditions_met += 1
        
        # Condición 4: Aceleración suave
        if avg_acceleration < ACCELERATION_THRESHOLD:
            conditions_met += 1
        
        # Condición 5: SNR alto
        if snr > SNR_THRESHOLD:
            conditions_met += 1
        
        # Necesita al menos 4 de 5 condiciones para ser alta precisión
        is_high_precision = conditions_met >= 4
        
        quality_type = "ALTA PRECISION (tipo KITTI)" if is_high_precision else "BAJA PRECISION (tipo movil)"
        
        self.is_high_precision = is_high_precision
        self.quality_detection_done = True
        self.quality_metrics = {
            'detected': bool(True),
            'is_high_precision': bool(is_high_precision),
            'quality_type': quality_type,
            'metrics': {
                'jitter_std': float(jitter),
                'jump_frequency': float(jump_frequency),
                'max_jump': float(max_jump),
                'avg_acceleration': float(avg_acceleration),
                'avg_curvature_deg': float(np.degrees(avg_curvature)),
                'signal_noise_ratio': float(snr),
                'num_samples': int(len(self.all_measurements))
            },
            'thresholds_used': {
                'jitter_threshold': float(JITTER_THRESHOLD),
                'jump_freq_threshold': float(JUMP_FREQ_THRESHOLD),
                'max_jump_threshold': float(MAX_JUMP_THRESHOLD),
                'acceleration_threshold': float(ACCELERATION_THRESHOLD),
                'snr_threshold': float(SNR_THRESHOLD)
            },
            'special_rule_applied': bool(special_rule_applied),
            'conditions_met': f"{conditions_met}/{total_conditions}"
        }
        
        return self.quality_metrics
    
    def print_quality_detection_report(self):
        """Imprime reporte detallado de la detección de calidad de GPS."""
        if not self.quality_detection_done:
            print("Deteccion de calidad GPS aun no realizada")
            return
        
        print("\n" + "="*80)
        print("DETECCION AUTOMATICA DE CALIDAD DE GPS")
        print("="*80)
        print(f"\nResultado: {self.quality_metrics['quality_type']}")
        
        confidence_level = 'ALTA' if self.quality_metrics['metrics']['num_samples'] >= 50 else 'MEDIA'
        print(f"Confianza en deteccion: {confidence_level}")
        
        print("\n" + "-"*80)
        print("METRICAS OBSERVADAS:")
        print("-"*80)
        
        metrics = self.quality_metrics['metrics']
        thresholds = self.quality_metrics['thresholds_used']
        
        print(f"\n1. Jitter (variacion frame-a-frame):")
        print(f"   Valor medido: {metrics['jitter_std']:.3f} m")
        print(f"   Umbral: {thresholds['jitter_threshold']:.3f} m")
        status = 'BAJO (bueno)' if metrics['jitter_std'] < thresholds['jitter_threshold'] else 'ALTO (malo)'
        print(f"   Estado: {status}")
        
        print(f"\n2. Frecuencia de saltos repentinos:")
        print(f"   Valor medido: {metrics['jump_frequency']*100:.1f}% de frames")
        print(f"   Umbral: {thresholds['jump_freq_threshold']*100:.1f}%")
        status = 'BAJO (bueno)' if metrics['jump_frequency'] < thresholds['jump_freq_threshold'] else 'ALTO (malo)'
        print(f"   Estado: {status}")
        
        print(f"\n3. Magnitud de salto maximo:")
        print(f"   Valor medido: {metrics['max_jump']:.3f} m")
        print(f"   Umbral: {thresholds['max_jump_threshold']:.3f} m")
        status = 'PEQUENO (bueno)' if metrics['max_jump'] < thresholds['max_jump_threshold'] else 'GRANDE (malo)'
        print(f"   Estado: {status}")
        
        print(f"\n4. Aceleracion promedio:")
        print(f"   Valor medido: {metrics['avg_acceleration']:.3f} m/frame^2")
        print(f"   Umbral: {thresholds['acceleration_threshold']:.3f} m/frame^2")
        status = 'SUAVE (bueno)' if metrics['avg_acceleration'] < thresholds['acceleration_threshold'] else 'BRUSCO (malo)'
        print(f"   Estado: {status}")
        
        print(f"\n5. Relacion senal/ruido:")
        print(f"   Valor medido: {metrics['signal_noise_ratio']:.2f}")
        print(f"   Umbral: {thresholds['snr_threshold']:.2f}")
        status = 'ALTA (bueno)' if metrics['signal_noise_ratio'] > thresholds['snr_threshold'] else 'BAJA (malo)'
        print(f"   Estado: {status}")
        
        print("\n" + "="*80)
        print("INTERPRETACION:")
        print("="*80)
        
        if self.is_high_precision:
            print("\nGPS de ALTA PRECISION detectado")
            print("   - Puede usarse como ground truth para metricas ATE/RPE")
            print("   - Comparable con sistemas RTK-GPS profesionales")
            print("   - Precision estimada: +-5-20 cm")
        else:
            print("\nGPS de BAJA PRECISION detectado")
            print("   - NO debe usarse como ground truth absoluto")
            print("   - Solo valido para metricas relativas (suavidad, consistencia)")
            print("   - Precision estimada: +-2-15 m")
            print("\n   Recomendacion:")
            print("   - Usar metricas de CONSISTENCIA en lugar de ATE/RPE absoluto")
            print("   - Comparar Pipeline vs SLAM (ambos con mismo GPS de referencia)")
        
        print("="*80 + "\n")
    
    def get_quality_type(self):
        """Retorna el tipo de calidad detectado ('high' o 'low')"""
        if not self.quality_detection_done:
            return None
        return 'high' if self.is_high_precision else 'low'