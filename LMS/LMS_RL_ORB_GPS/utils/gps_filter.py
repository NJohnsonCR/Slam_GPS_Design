import numpy as np

class GPSFilter:
    def __init__(self, window_size=5, movement_variance_threshold=0.1, 
                 add_noise=False, noise_std=5.0):
        """
        Filtro de GPS con capacidad de simular GPS móvil
        
        Args:
            window_size: Tamaño de ventana para suavizado
            movement_variance_threshold: Umbral de varianza para detectar movimiento
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
        """Calcula confianza basada en varianza de mediciones recientes"""
        if len(self.measurements) < 2:
            return 0.5
        
        # Calcular varianza de las mediciones
        variance = np.var(self.measurements, axis=0)
        total_variance = np.sum(variance[:2])  # Solo X, Y
        
        # Confianza inversamente proporcional a la varianza
        # Si hay mucho ruido (alta varianza), baja confianza
        if self.add_noise:
            # Con GPS móvil simulado, esperamos más varianza
            confidence = 1.0 / (1.0 + total_variance / (self.noise_std ** 2))
        else:
            # GPS preciso de KITTI
            confidence = 1.0 / (1.0 + total_variance / self.movement_variance_threshold)
        
        # Limitar entre 0.1 y 1.0
        confidence = np.clip(confidence, 0.1, 1.0)
        
        self.last_confidence = confidence
        return confidence
    
    def get_filtered_position(self):
        """Retorna la posición filtrada"""
        return self.filtered_position if self.filtered_position is not None else None
    
    def print_debug_info(self):
        """Imprime información de debug"""
        if len(self.measurements) > 0:
            variance = np.var(self.measurements, axis=0) if len(self.measurements) > 1 else np.zeros(3)
            mode = "GPS MÓVIL" if self.add_noise else "GPS PRECISO"
            print(f"  [Filtro GPS - {mode}] Confianza: {self.last_confidence:.3f}, "
                  f"Varianza XY: {np.sum(variance[:2]):.3f}, "
                  f"Mediciones: {len(self.measurements)}")