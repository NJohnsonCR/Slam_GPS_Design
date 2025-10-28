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
        else:
            self.measurements.append(gps_position)
        
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