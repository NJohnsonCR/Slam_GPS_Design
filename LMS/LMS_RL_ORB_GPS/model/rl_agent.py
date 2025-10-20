import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import json

class SimpleRLAgent(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        """
        Red neuronal mejorada para ajuste fino de pesos.
        
        Arquitectura:
        - Input: 3 valores (visual_conf, gps_conf, error)
        - Hidden: 4 capas con 32 -> 32 -> 16 neuronas
        - Output: 2 valores (pesos SLAM y GPS con Softmax)
        
        Mejoras:
        - Red más profunda (4 capas vs 2)
        - Más neuronas (32 vs 8)
        - Dropout para evitar overfitting
        - Mayor capacidad de aprendizaje de patrones complejos
        """
        super(SimpleRLAgent, self).__init__()
        
        # Red más profunda con capas progresivas
        self.fc1 = nn.Linear(input_dim, hidden_dim)      # 3 -> 32
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)     # 32 -> 32
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2) # 32 -> 16
        self.fc4 = nn.Linear(hidden_dim // 2, 2)         # 16 -> 2
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # Regularización (10% dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # Capa 1: Procesar entrada
        h1 = self.relu(self.fc1(x))
        h1 = self.dropout(h1)
        
        # Capa 2: Aprender patrones complejos
        h2 = self.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        
        # Capa 3: Reducir dimensionalidad
        h3 = self.relu(self.fc3(h2))
        
        # Capa 4: Generar pesos (sin dropout en la última capa)
        weights = self.softmax(self.fc4(h3))
        
        return weights


class RLTrainer:
    """
    Entrenador para el agente RL que aprende a fusionar GPS y SLAM.
    
    Inputs del estado:
    - GPS confidence: Confiabilidad del GPS (0-1)
    - SLAM confidence: Confiabilidad del SLAM basado en matches (0-1)
    - Error: Distancia entre predicción SLAM y GPS
    
    Outputs:
    - weight_slam: Peso para SLAM (0-1)
    - weight_gps: Peso para GPS (0-1), donde weight_gps = 1 - weight_slam
    """
    
    def __init__(self, agent, learning_rate=0.001, gamma=0.95, buffer_size=10000):
        self.agent = agent
        self.optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
        self.gamma = gamma  # Factor de descuento para rewards futuros
        
        # Replay buffer para almacenar experiencias
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Métricas de entrenamiento
        self.training_metrics = {
            'losses': [],
            'rewards': [],
            'avg_weights_slam': [],
            'avg_weights_gps': []
        }
        
        # Ground truth previo para calcular rewards
        self.previous_ground_truth = None
        
    def calculate_reward(self, state, action_weights, slam_position, gps_position, ground_truth=None):
        """
        Calcula el reward basado en qué tan buena fue la decisión de fusión.
        
        NUEVA VERSIÓN MEJORADA:
        - Premia fuertemente confiar en GPS cuando es confiable (>0.7)
        - Penaliza fuertemente confiar en GPS cuando es malo (<0.3)
        - Los pesos deben reflejar las confianzas de forma proporcional
        """
        gps_confidence = state[1].item()  # state[1] es GPS confidence
        slam_confidence = state[0].item()  # state[0] es SLAM confidence
        current_error = state[2].item()    # state[2] es el error actual
        
        weight_slam = action_weights[0].item()
        weight_gps = action_weights[1].item()
        
        # Calcular posición fusionada
        if isinstance(slam_position, torch.Tensor):
            slam_position = slam_position.detach().numpy()
        if isinstance(gps_position, torch.Tensor):
            gps_position = gps_position.detach().numpy()
            
        fused_position = weight_slam * slam_position + weight_gps * gps_position
        
        reward = 0.0
        
        # === NUEVO REWARD 1: Alineación FUERTE con confianzas ===
        # Los pesos DEBEN reflejar las confianzas relativas
        total_confidence = slam_confidence + gps_confidence
        if total_confidence > 0.01:
            # Pesos ideales basados en confianzas
            ideal_weight_slam = slam_confidence / total_confidence
            ideal_weight_gps = gps_confidence / total_confidence
            
            # Error en la asignación de pesos
            weight_error_slam = abs(weight_slam - ideal_weight_slam)
            weight_error_gps = abs(weight_gps - ideal_weight_gps)
            
            # PENALIZACIÓN FUERTE por no seguir las confianzas
            alignment_penalty = (weight_error_slam + weight_error_gps) * 10.0
            reward -= alignment_penalty
        
        # === NUEVO REWARD 2: BONIFICACIÓN EXTRA por usar GPS confiable ===
        if gps_confidence > 0.7:
            # Cuanto más GPS uses cuando es confiable, mejor
            gps_usage_bonus = weight_gps * 8.0  # AUMENTADO de 1.5 a 8.0
            reward += gps_usage_bonus
            
            # PENALIZACIÓN si NO usas GPS confiable
            if weight_gps < 0.5:
                reward -= 10.0  # Penalización grande
        
        if gps_confidence > 0.8:  # GPS excelente
            # Bonificación EXTRA si confías mucho en GPS excelente
            if weight_gps > 0.6:
                reward += 5.0
        
        # === NUEVO REWARD 3: PENALIZACIÓN FUERTE por usar GPS malo ===
        if gps_confidence < 0.3:
            # Cuanto más GPS uses cuando es malo, peor
            gps_bad_penalty = weight_gps * 10.0  # AUMENTADO de 5.0 a 10.0
            reward -= gps_bad_penalty
            
            # BONIFICACIÓN si confías en SLAM cuando GPS es malo
            if weight_slam > 0.7:
                reward += 5.0
        
        # === REWARD 4: Similarmente para SLAM ===
        if slam_confidence > 0.7:
            reward += weight_slam * 5.0  # Bonificar uso de SLAM confiable
        
        if slam_confidence < 0.3 and weight_slam > 0.5:
            reward -= 8.0  # Penalizar confiar en SLAM malo
        
        # === REWARD 5: Penalización por distancia entre GPS y SLAM ===
        distance_gps_slam = np.linalg.norm(gps_position - slam_position)
        
        if distance_gps_slam > 5.0:  # Si están muy separados
            # Confiar más en el que tiene mayor confianza
            if gps_confidence > slam_confidence + 0.1:
                # GPS es más confiable
                if weight_gps > weight_slam:
                    reward += 3.0  # Bonificar decisión correcta
                else:
                    reward -= 5.0  # Penalizar decisión incorrecta
            elif slam_confidence > gps_confidence + 0.1:
                # SLAM es más confiable
                if weight_slam > weight_gps:
                    reward += 3.0
                else:
                    reward -= 5.0
        
        # === REWARD 6: Penalización por error absoluto (reducida) ===
        if current_error > 0:
            reward -= min(current_error * 0.2, 5.0)  # REDUCIDO de 0.5 a 0.2
        
        # === REWARD 7: Suavidad (OPCIONAL y MUY REDUCIDO) ===
        # SOLO aplicar si los cambios son extremos
        if hasattr(self, 'last_weights'):
            weight_change = abs(weight_slam - self.last_weights[0])
            if weight_change > 0.4:  # SOLO penalizar cambios EXTREMOS
                reward -= weight_change * 0.5  # REDUCIDO de 2.0 a 0.5
        
        self.last_weights = action_weights.detach().numpy()
        
        # === REWARD 8: Ground truth (si existe) ===
        if ground_truth is not None:
            error_fused = np.linalg.norm(fused_position - ground_truth)
            error_slam = np.linalg.norm(slam_position - ground_truth)
            error_gps = np.linalg.norm(gps_position - ground_truth)
            
            # Bonificar si la fusión es mejor que ambas fuentes
            if error_fused < min(error_slam, error_gps):
                reward += 8.0  # AUMENTADO de 5.0 a 8.0
            elif error_fused < (error_slam + error_gps) / 2:
                reward += 3.0  # AUMENTADO de 2.0 a 3.0
            
            # Penalización directa por el error (reducida)
            reward -= error_fused * 0.2  # REDUCIDO de 0.3 a 0.2
        
        return reward
    
    def store_experience(self, state, action_weights, reward, next_state, done=False):
        """Almacena una experiencia en el replay buffer"""
        self.replay_buffer.append({
            'state': state.detach().cpu(),
            'action': action_weights.detach().cpu(),
            'reward': reward,
            'next_state': next_state.detach().cpu() if next_state is not None else None,
            'done': done
        })
    
    def train_step(self, batch_size=32):
        """
        Realiza un paso de entrenamiento usando experiencias del replay buffer.
        Usa una aproximación de Policy Gradient simplificada.
        """
        if len(self.replay_buffer) < batch_size:
            return None  # No hay suficientes experiencias
        
        # Muestrear batch aleatorio
        batch = random.sample(self.replay_buffer, batch_size)
        
        total_loss = 0.0
        total_reward = 0.0
        
        for experience in batch:
            state = experience['state']
            action_weights = experience['action']
            reward = experience['reward']
            
            # Forward pass
            predicted_weights = self.agent(state)
            
            # Loss: queremos que el agente aprenda a maximizar el reward
            # Usamos negative log likelihood ponderado por el reward
            # Esto hace que acciones con rewards altos sean más probables
            
            # Calcular log probabilidad de la acción tomada
            log_prob = torch.log(predicted_weights + 1e-10)  # Evitar log(0)
            
            # Policy gradient loss
            # Queremos maximizar E[reward], equivalente a minimizar -reward * log_prob
            loss = -reward * (log_prob * action_weights).sum()
            
            # También añadimos un término de regularización para evitar distribuciones extremas
            # Queremos que el agente explore, no que siempre ponga peso 1.0 en una opción
            entropy = -(predicted_weights * log_prob).sum()
            entropy_bonus = -0.01 * entropy  # Penalización pequeña para fomentar exploración
            
            total_loss += loss + entropy_bonus
            total_reward += reward
        
        # Backward pass y optimización
        avg_loss = total_loss / batch_size
        self.optimizer.zero_grad()
        avg_loss.backward()
        
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Guardar métricas
        avg_reward = total_reward / batch_size
        self.training_metrics['losses'].append(avg_loss.item())
        self.training_metrics['rewards'].append(avg_reward)
        
        return {
            'loss': avg_loss.item(),
            'avg_reward': avg_reward
        }
    
    def save_model(self, path, metadata=None):
        """Guarda el modelo entrenado y métricas"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_metrics': self.training_metrics,
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, path)
        
        # Guardar métricas en JSON para fácil lectura
        # Convertir float32/float64 de NumPy a float de Python
        def convert_to_python_types(obj):
            """Convierte tipos de NumPy/PyTorch a tipos nativos de Python"""
            if isinstance(obj, dict):
                return {key: convert_to_python_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            else:
                return obj
        
        metrics_path = path.replace('.pth', '_metrics.json')
        metrics_serializable = convert_to_python_types(self.training_metrics)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"OK - Modelo guardado en: {path}")
        print(f"OK - Métricas guardadas en: {metrics_path}")
    
    def load_model(self, path):
        """Carga un modelo pre-entrenado"""
        if not os.path.exists(path):
            print(f"ADVERTENCIA: Modelo no encontrado en {path}, usando modelo sin entrenar")
            return False
        
        print(f"Cargando modelo desde: {path}")
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.agent.load_state_dict(checkpoint['model_state_dict'])
            print("OK - Pesos del modelo cargados")
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("OK - Estado del optimizador cargado")
        
        if 'training_metrics' in checkpoint:
            self.training_metrics = checkpoint['training_metrics']
            total_steps = len(self.training_metrics.get('losses', []))
            print(f"OK - Métricas cargadas: {total_steps} pasos de entrenamiento")
        
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
            print(f"OK - Metadata del modelo:")
            for key, value in metadata.items():
                print(f"   - {key}: {value}")
        
        print("Modelo cargado exitosamente")
        return True
    
    def print_training_summary(self):
        """Imprime un resumen del entrenamiento"""
        if not self.training_metrics['losses']:
            print("No hay datos de entrenamiento disponibles")
            return
        
        print("\n" + "="*50)
        print("RESUMEN DE ENTRENAMIENTO")
        print("="*50)
        print(f"Total de pasos de entrenamiento: {len(self.training_metrics['losses'])}")
        print(f"Pérdida promedio (últimos 100): {np.mean(self.training_metrics['losses'][-100:]):.4f}")
        print(f"Reward promedio (últimos 100): {np.mean(self.training_metrics['rewards'][-100:]):.4f}")
        print(f"Experiencias en buffer: {len(self.replay_buffer)}")
        print("="*50 + "\n")