import numpy as np
from collections import deque

class GPSFilter:
    def __init__(self, window_size=5, movement_variance_threshold=0.1):
        """
        Args:
            window_size: Tamaño del buffer deslizante
            movement_variance_threshold: Umbral de varianza en movimientos (m²)
                - 0.01: Muy estricto (movimientos muy consistentes)
                - 0.1:  Moderado (para KITTI)
                - 0.5:  Relajado (tolerante a variaciones)
        """
        self.window_size = window_size
        self.movement_variance_threshold = movement_variance_threshold
        
        self.position_buffer = deque(maxlen=window_size)
        self.confidence_history = []
        self.raw_positions = []
    
    def add_measurement(self, utm_position):
        self.raw_positions.append(utm_position.copy())
        self.position_buffer.append(utm_position.copy())
        
    
    def calculate_confidence(self):

        if len(self.position_buffer) < 2:
            return 0.5  # Confianza base con pocos datos
            
        positions = np.array(self.position_buffer)
        coordinates_xy = positions[:, :2]
    
        movements = []
        for i in range(1, len(positions)):
            # Distancia entre frame i e i-1
            movement = np.linalg.norm(coordinates_xy[i] - coordinates_xy[i-1])
            movements.append(movement)

        if len(movements) < 2:
            movement_variance = 0.0
        else:
            movement_variance = np.var(movements)
        
        consistency = max(0, 1 - movement_variance / self.movement_variance_threshold)
        
        buffer_factor = len(self.position_buffer) / self.window_size
        
    
        confidence = 0.2 + 0.3 * buffer_factor + 0.5 * consistency
        
        
        confidence = np.clip(confidence, 0.1, 0.95)
        
        
        self.confidence_history.append(confidence)
        
        return confidence
    
    
    def print_debug_info(self):
        """Muestra información de debug"""
        if len(self.position_buffer) > 0:
            current_pos = self.position_buffer[-1]
            confidence = self.calculate_confidence()
            
            print(f"DEBUG - Raw: {current_pos[:2].round(2)} | "
                  f"Conf: {confidence:.3f} | "
                  f"Buffer: {len(self.position_buffer)}/{self.window_size}")