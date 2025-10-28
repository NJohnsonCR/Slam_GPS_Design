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
        Sistema de recompensas BALANCEADO que premia decisiones inteligentes.
        
        FILOSOFÍA NUEVA (sin bias GPS):
        - Premiar cuando RL confía en el sensor correcto según confianzas
        - Penalizar cuando RL confía en el sensor equivocado
        - Recompensar balance cuando ambos son buenos
        - NO asumir que GPS siempre es correcto
        
        Args:
            state: [visual_conf, gps_conf, error]
            action_weights: Pesos predichos por RL [w_slam, w_gps]
            slam_position: Posición estimada por SLAM visual
            gps_position: Posición GPS
            ground_truth: Verdad absoluta (si existe, generalmente None)
        
        Returns:
            reward: Recompensa escalar
        """
        visual_conf = state[0].item()
        gps_conf = state[1].item()
        error = state[2].item()
        
        w_slam_rl = action_weights[0].item()
        w_gps_rl = action_weights[1].item()
        
        # Convertir a numpy si son tensores
        if isinstance(slam_position, torch.Tensor):
            slam_position = slam_position.detach().cpu().numpy()
        if isinstance(gps_position, torch.Tensor):
            gps_position = gps_position.detach().cpu().numpy()
        
        # Calcular posición fusionada
        fused_position = w_slam_rl * slam_position + w_gps_rl * gps_position
        
        # ========================================
        # COMPONENTE 1: RECOMPENSA POR DECISIÓN INTELIGENTE (AUMENTADA)
        # ========================================
        decision_reward = 0.0
        
        # Caso 1: GPS mucho mejor que SLAM → Debería preferir GPS
        if gps_conf > 0.7 and visual_conf < 0.5:
            if w_gps_rl > w_slam_rl:  # RL confía más en GPS (correcto)
                decision_reward = 5.0  # Aumentado de 2.5 a 5.0
                if np.random.rand() < 0.1:
                    print(f"    [REWARD] RL prefirió GPS correctamente (+5.0)")
            else:  # RL confía más en SLAM (incorrecto)
                decision_reward = -4.0  # Aumentado de -2.0 a -4.0
                if np.random.rand() < 0.1:
                    print(f"    [REWARD] RL ignoró GPS bueno (-4.0)")
        
        # Caso 2: SLAM mucho mejor que GPS → Debería preferir SLAM
        elif visual_conf > 0.7 and gps_conf < 0.5:
            if w_slam_rl > w_gps_rl:  # RL confía más en SLAM (correcto)
                decision_reward = 5.0  # Aumentado de 2.5 a 5.0
                if np.random.rand() < 0.1:
                    print(f"    [REWARD] RL prefirió SLAM correctamente (+5.0)")
            else:  # RL confía más en GPS (incorrecto)
                decision_reward = -4.0  # Aumentado de -2.0 a -4.0
                if np.random.rand() < 0.1:
                    print(f"    [REWARD] RL ignoró SLAM bueno (-4.0)")
        
        # Caso 3: Ambos buenos y error bajo → Debería balancear 50-50
        elif visual_conf > 0.7 and gps_conf > 0.7 and error < 2.5:
            # Calcular qué tan cerca está de 50-50
            balance_quality = 1.0 - abs(w_slam_rl - 0.5) * 2  # [0, 1], 1=perfecto
            decision_reward = 3.0 * balance_quality  # Aumentado de 2.0 a 3.0
            if np.random.rand() < 0.1:
                print(f"    [REWARD] Balance con ambos buenos (+{decision_reward:.2f})")
        
        # Caso 4: Ambos buenos pero error alto → Conflicto, preferir más confiable
        elif visual_conf > 0.7 and gps_conf > 0.7 and error > 2.5:
            # Hay conflicto, ver quién es ligeramente más confiable
            if visual_conf > gps_conf + 0.05:
                # SLAM ligeramente mejor
                if w_slam_rl > 0.55:
                    decision_reward = 2.5  # Aumentado de 1.5 a 2.5
                else:
                    decision_reward = -1.0  # Aumentado de -0.5 a -1.0
            elif gps_conf > visual_conf + 0.05:
                # GPS ligeramente mejor
                if w_gps_rl > 0.55:
                    decision_reward = 2.5  # Aumentado de 1.5 a 2.5
                else:
                    decision_reward = -1.0  # Aumentado de -0.5 a -1.0
            else:
                # Empate técnico, balance conservador
                balance_quality = 1.0 - abs(w_slam_rl - 0.5) * 2
                decision_reward = 1.5 * balance_quality  # Aumentado de 1.0 a 1.5
        
        # Caso 5: Ambos medios (0.5-0.7) → Preferir ligeramente al mejor
        elif 0.5 <= visual_conf <= 0.7 and 0.5 <= gps_conf <= 0.7:
            if visual_conf > gps_conf + 0.1:
                if w_slam_rl > 0.52:
                    decision_reward = 1.5  # Aumentado de 1.0 a 1.5
            elif gps_conf > visual_conf + 0.1:
                if w_gps_rl > 0.52:
                    decision_reward = 1.5  # Aumentado de 1.0 a 1.5
            else:
                # Balance cuando son similares
                balance_quality = 1.0 - abs(w_slam_rl - 0.5) * 2
                decision_reward = 1.2 * balance_quality  # Aumentado de 0.8 a 1.2
        
        # Caso 6: Ambos malos (<0.4) → Premiar distribución conservadora
        elif visual_conf < 0.4 and gps_conf < 0.4:
            # Cuando ambos son malos, mejor ser conservador (cercano a 50-50)
            caution_bonus = 1.0 - abs(w_slam_rl - 0.5) * 2
            decision_reward = 1.8 * caution_bonus  # Aumentado de 1.2 a 1.8
            if np.random.rand() < 0.1:
                print(f"    [REWARD] Cautela con ambos malos (+{decision_reward:.2f})")
        
        # ========================================
        # COMPONENTE 2: PENALIZACIÓN POR ERROR (si hay ground truth)
        # ========================================
        error_penalty = 0.0
        
        if ground_truth is not None:
            residual_error = np.linalg.norm(fused_position - ground_truth)
            # Penalización suave, máximo -1.5
            error_penalty = -0.3 * min(residual_error / 3.0, 5.0)
        else:
            # Sin ground truth, usar error SLAM-GPS como proxy
            if error > 5.0:
                # Gran discrepancia, penalizar ligeramente no balancear
                balance_factor = 1.0 - abs(w_slam_rl - 0.5) * 2
                error_penalty = -0.4 * (1.0 - balance_factor)
        
        # ========================================
        # COMPONENTE 3: BONUS POR ALINEACIÓN CON CONFIANZAS (AUMENTADO)
        # ========================================
        # Premiar si los pesos están alineados con las confianzas relativas
        total_conf = visual_conf + gps_conf + 1e-6
        expected_w_slam = visual_conf / total_conf
        expected_w_gps = gps_conf / total_conf
        
        # Calcular qué tan cerca está de la distribución esperada
        alignment_error = abs(w_slam_rl - expected_w_slam)
        alignment_bonus = 1.5 * (1.0 - alignment_error * 2)  # Aumentado de 0.8 a 1.5
        alignment_bonus = max(alignment_bonus, -0.8)  # Aumentado mínimo de -0.4 a -0.8
        
        # ========================================
        # COMPONENTE 4: SUAVIDAD TEMPORAL (opcional, muy reducido)
        # ========================================
        smoothness_penalty = 0.0
        
        if hasattr(self, 'last_weights'):
            weight_change = abs(w_slam_rl - self.last_weights[0])
            # Solo penalizar cambios EXTREMOS (>40%)
            if weight_change > 0.4:
                smoothness_penalty = -0.3 * (weight_change - 0.4)
        
        self.last_weights = action_weights.detach().cpu().numpy()
        
        # ========================================
        # RECOMPENSA TOTAL
        # ========================================
        total_reward = (
            decision_reward +      # Componente principal (±5.0, antes ±2.5)
            error_penalty +        # Penalización por error (≤0)
            alignment_bonus +      # Bonus por alineación (±1.5, antes ±0.8)
            smoothness_penalty     # Penalización por brusquedad (≤0)
        )
        
        # Debug ocasional
        if np.random.rand() < 0.05:  # 5% de las veces
            print(f"    [REWARD] Total={total_reward:.3f} "
                  f"(decisión={decision_reward:.2f}, error={error_penalty:.2f}, "
                  f"alineación={alignment_bonus:.2f}, suavidad={smoothness_penalty:.2f})")
        
        return total_reward
    
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
            entropy_bonus = 0.01 * entropy  # Cambiado a positivo para premiar exploración
            
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