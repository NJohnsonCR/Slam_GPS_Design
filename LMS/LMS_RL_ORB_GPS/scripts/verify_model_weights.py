#!/usr/bin/env python3
"""
Script para verificar que los pesos del modelo RL están siendo actualizados
"""

import torch
import json
import os

def verify_model_weights():
    model_path = "LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth"
    metrics_path = "LMS/LMS_RL_ORB_GPS/model/trained_rl_agent_metrics.json"
    
    print("="*70)
    print("VERIFICACIÓN DE PESOS DEL MODELO RL")
    print("="*70)
    
    # 1. Verificar si existe el modelo
    if not os.path.exists(model_path):
        print("ERROR - No se encontró modelo entrenado")
        print(f"   Esperado en: {model_path}")
        return
    
    # 2. Cargar el modelo
    print(f"\nOK - Modelo encontrado: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 3. Mostrar información del checkpoint
    print(f"\n Información del Checkpoint:")
    print(f"   - Claves disponibles: {list(checkpoint.keys())}")
    
    if 'metadata' in checkpoint:
        print(f"\n📋 Metadata:")
        for key, value in checkpoint['metadata'].items():
            print(f"   - {key}: {value}")
    
    # 4. Mostrar pesos del modelo
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\n🔢 Pesos de la Red Neuronal:")
        print(f"   Total de parámetros: {len(state_dict)}")
        
        for param_name, param_value in state_dict.items():
            print(f"\n   {param_name}:")
            print(f"      Shape: {param_value.shape}")
            print(f"      Mean: {param_value.mean().item():.6f}")
            print(f"      Std: {param_value.std().item():.6f}")
            print(f"      Min: {param_value.min().item():.6f}")
            print(f"      Max: {param_value.max().item():.6f}")
            
            # Mostrar primeros valores
            if len(param_value.shape) == 2:  # Matriz de pesos
                print(f"      Primeros valores: {param_value[0, :5].tolist()}")
            elif len(param_value.shape) == 1:  # Vector de bias
                print(f"      Primeros valores: {param_value[:5].tolist()}")
    
    # 5. Verificar métricas de entrenamiento
    if os.path.exists(metrics_path):
        print(f"\n Métricas de Entrenamiento:")
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        if 'losses' in metrics and len(metrics['losses']) > 0:
            losses = metrics['losses']
            print(f"   - Total de pasos de entrenamiento: {len(losses)}")
            print(f"   - Loss inicial: {losses[0]:.6f}")
            print(f"   - Loss final: {losses[-1]:.6f}")
            print(f"   - Reducción de loss: {((losses[0] - losses[-1])/losses[0]*100):.2f}%")
            
            if losses[-1] < losses[0]:
                print(f"   OK - El modelo SÍ está aprendiendo (loss disminuyó)")
            else:
                print(f"   ADVERTENCIA:  El modelo podría necesitar más entrenamiento")
        
        if 'rewards' in metrics and len(metrics['rewards']) > 0:
            rewards = metrics['rewards']
            print(f"\n   - Reward inicial: {rewards[0]:.6f}")
            print(f"   - Reward final: {rewards[-1]:.6f}")
            print(f"   - Mejora de reward: {((rewards[-1] - rewards[0])/abs(rewards[0])*100):.2f}%")
    
    # 6. Test de inferencia
    print(f"\n🧪 Test de Inferencia:")
    print(f"   Probando con estados de ejemplo...")
    
    from LMS.LMS_RL_ORB_GPS.model.rl_agent import SimpleRLAgent
    
    agent = SimpleRLAgent()
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    # Casos de prueba
    test_cases = [
        {"name": "GPS preciso, SLAM malo", "state": [0.3, 0.9, 2.0]},
        {"name": "GPS malo, SLAM bueno", "state": [0.9, 0.3, 5.0]},
        {"name": "Ambos buenos", "state": [0.8, 0.8, 0.5]},
        {"name": "Ambos malos", "state": [0.2, 0.2, 10.0]},
    ]
    
    for test in test_cases:
        state = torch.tensor(test['state'], dtype=torch.float32)
        with torch.no_grad():
            weights = agent(state)
        w_slam, w_gps = weights[0].item(), weights[1].item()
        print(f"\n   {test['name']}:")
        print(f"      Estado: visual={test['state'][0]:.2f}, gps={test['state'][1]:.2f}, error={test['state'][2]:.2f}")
        print(f"      Pesos: SLAM={w_slam:.3f}, GPS={w_gps:.3f}")
        
        # Validar comportamiento esperado
        if test['state'][1] > test['state'][0]:  # GPS mejor que SLAM
            if w_gps > w_slam:
                print(f"      OK - Comportamiento correcto (confía más en GPS)")
            else:
                print(f"      ADVERTENCIA:  Revisar: Debería confiar más en GPS")
        else:  # SLAM mejor que GPS
            if w_slam > w_gps:
                print(f"      OK - Comportamiento correcto (confía más en SLAM)")
            else:
                print(f"      ADVERTENCIA:  Revisar: Debería confiar más en SLAM")
    
    print("\n" + "="*70)
    print("VERIFICACIÓN COMPLETADA")
    print("="*70)

if __name__ == "__main__":
    verify_model_weights()
