#!/usr/bin/env python3
"""
Script de Entrenamiento Multi-Secuencia con Aumentación de Datos

Este script entrena el agente RL usando MÚLTIPLES secuencias KITTI con:
- Aumentación de datos (GPS malo, SLAM malo, conflictos)
- Entrenamiento progresivo (secuencias más fáciles primero)
- Métricas detalladas por secuencia
- Guardado periódico de checkpoints

Uso:
    python train_multi_sequence.py --sequences 0001 0005 0009 --epochs 3
    python train_multi_sequence.py --sequences 0001 0005 0009 0013 0018 --epochs 5 --max_frames 200

Autor: Sistema Híbrido SLAM-GPS
Fecha: Octubre 2025
"""

import argparse
import sys
import os
import json
import numpy as np
import torch
from datetime import datetime

# Agregar paths necesarios
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from LMS.LMS_RL_ORB_GPS.main import RL_ORB_SLAM_GPS, load_kitti_sequence
from LMS.LMS_RL_ORB_GPS.utils.training_augmentation import TrainingAugmentation


class MultiSequenceTrainer:
    """
    Entrenador que procesa múltiples secuencias KITTI con aumentación de datos
    """
    
    def __init__(self, kitti_base_path="kitti_data/2011_09_26", model_path=None):
        """
        Args:
            kitti_base_path: Ruta base a las secuencias KITTI
            model_path: Ruta a modelo pre-entrenado (None para empezar desde cero)
        """
        self.kitti_base_path = kitti_base_path
        self.model_path = model_path
        
        # Estadísticas globales de entrenamiento
        self.training_stats = {
            'start_time': datetime.now().isoformat(),
            'sequences_processed': [],
            'total_frames': 0,
            'total_training_steps': 0,
            'checkpoints_saved': []
        }
        
        # Configurar logging
        self.setup_logging()
    
    def setup_logging(self):
        """Configura archivo de log para el entrenamiento"""
        log_dir = "LMS/LMS_RL_ORB_GPS/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_multi_seq_{timestamp}.log")
        
        with open(self.log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ENTRENAMIENTO MULTI-SECUENCIA CON AUMENTACIÓN DE DATOS\n")
            f.write("="*80 + "\n")
            f.write(f"Inicio: {self.training_stats['start_time']}\n")
            f.write(f"Ruta base KITTI: {self.kitti_base_path}\n")
            f.write(f"Modelo inicial: {self.model_path or 'Desde cero'}\n")
            f.write("="*80 + "\n\n")
        
        print(f"Log de entrenamiento: {self.log_file}")
    
    def log(self, message, print_console=True):
        """Escribe mensaje en log y opcionalmente en consola"""
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
        
        if print_console:
            print(message)
    
    def train_on_sequence(self, sequence_number, slam, max_frames=None, augmentation_prob=0.6):
        """
        Entrena en una secuencia específica con aumentación de datos
        
        Args:
            sequence_number: Número de secuencia (ej: "0001")
            slam: Instancia de RL_ORB_SLAM_GPS
            max_frames: Límite de frames (None = todos)
            augmentation_prob: Probabilidad de aplicar aumentación (0.6 = 60%)
        
        Returns:
            dict: Estadísticas de la secuencia
        """
        sequence_path = os.path.join(
            self.kitti_base_path, 
            f"2011_09_26_drive_{sequence_number}_sync"
        )
        
        if not os.path.exists(sequence_path):
            self.log(f"ERROR: Secuencia no encontrada: {sequence_path}")
            return None
        
        self.log(f"\n{'='*80}")
        self.log(f"ENTRENANDO EN SECUENCIA: {sequence_number}")
        self.log(f"{'='*80}")
        
        # Cargar frames y GPS
        frames, gps_data = load_kitti_sequence(sequence_path, max_frames)
        
        if len(frames) == 0:
            self.log(f"ERROR: No se cargaron frames de la secuencia {sequence_number}")
            return None
        
        self.log(f"Cargados {len(frames)} frames")
        self.log(f"Probabilidad de aumentación: {augmentation_prob*100:.0f}%")
        
        # Crear módulo de aumentación
        augmenter = TrainingAugmentation(corruption_probability=augmentation_prob)
        
        # Estadísticas de la secuencia
        seq_stats = {
            'sequence': sequence_number,
            'total_frames': len(frames),
            'frames_processed': 0,
            'augmentation_applied': 0,
            'scenarios': augmenter.stats.copy()
        }
        
        # Procesar cada frame
        training_steps_before = len(slam.trainer.training_metrics['losses'])
        
        for i in range(len(frames)):
            if i % 20 == 0:
                self.log(f"  Frame {i}/{len(frames)}...", print_console=True)
            
            if frames[i] is None:
                continue
            
            try:
                # Obtener GPS original
                gps_utm = None
                gps_confidence = 0.1
                visual_confidence = 0.5  # Se calculará en process_frame
                
                if gps_data[i] is not None:
                    gps_utm = slam.gps_frame_reference(gps_data[i])
                    
                    # Agregar a filtro GPS
                    slam.gps_filter.add_measurement(gps_utm)
                    gps_confidence_raw = slam.gps_filter.calculate_confidence()
                    
                    # CORRECCIÓN 2: Calcular error ANTES de aumentación
                    # Esto permite al aumentador decidir mejor el escenario
                    error = 0.0
                    if hasattr(slam, 'keyframe_poses') and len(slam.keyframe_poses) > 0:
                        last_pose = slam.keyframe_poses[-1][:3, 3]
                        error = np.linalg.norm(last_pose - gps_utm) if gps_utm is not None else 0.0
                    
                    # === APLICAR AUMENTACIÓN DE DATOS ===
                    # CORRECCIÓN 3: SIEMPRE aumentar (no probabilísticamente)
                    # El augmenter internamente decide el escenario según las probabilidades
                    gps_utm_aug, gps_conf_aug, visual_conf_aug, scenario = augmenter.augment_training_sample(
                        gps_utm=gps_utm,
                        gps_conf=gps_confidence_raw,
                        visual_conf=visual_confidence,
                        error=error
                    )
                    
                    # Usar datos aumentados
                    gps_utm = gps_utm_aug
                    gps_confidence = gps_conf_aug
                    
                    seq_stats['augmentation_applied'] += 1
                    
                    if i % 20 == 0:
                        self.log(f"    Escenario: {scenario}", print_console=False)
                else:
                    gps_confidence = gps_confidence_raw
                
                # Procesar frame con GPS (posiblemente aumentado)
                slam.process_frame_with_gps(frames[i], gps_utm, gps_confidence)
                
                seq_stats['frames_processed'] += 1
                
            except Exception as e:
                self.log(f"  Error en frame {i}: {e}", print_console=False)
        
        # Calcular pasos de entrenamiento en esta secuencia
        training_steps_after = len(slam.trainer.training_metrics['losses'])
        seq_stats['training_steps'] = training_steps_after - training_steps_before
        seq_stats['augmentation_stats'] = augmenter.stats
        
        # Log de estadísticas de aumentación
        self.log(f"\nEstadísticas de Aumentación:")
        self.log(f"  Total frames:     {augmenter.stats['total_frames']}")
        self.log(f"  Normal:           {augmenter.stats['normal']} ({augmenter.stats['normal']/augmenter.stats['total_frames']*100:.1f}%)")
        self.log(f"  GPS malo:         {augmenter.stats['bad_gps']} ({augmenter.stats['bad_gps']/augmenter.stats['total_frames']*100:.1f}%)")
        self.log(f"  SLAM malo:        {augmenter.stats['bad_slam']} ({augmenter.stats['bad_slam']/augmenter.stats['total_frames']*100:.1f}%)")
        self.log(f"  Ambos malos:      {augmenter.stats['both_bad']} ({augmenter.stats['both_bad']/augmenter.stats['total_frames']*100:.1f}%)")
        self.log(f"  Conflicto alto:   {augmenter.stats['high_conflict']} ({augmenter.stats['high_conflict']/augmenter.stats['total_frames']*100:.1f}%)")
        
        self.log(f"\nSecuencia {sequence_number} completada:")
        self.log(f"  Frames procesados: {seq_stats['frames_processed']}")
        self.log(f"  Pasos de entrenamiento: {seq_stats['training_steps']}")
        self.log(f"  Aumentaciones aplicadas: {seq_stats['augmentation_applied']}")
        
        return seq_stats
    
    def train(self, sequence_numbers, epochs=3, max_frames=None, 
              augmentation_probs=None, save_checkpoints=True):
        """
        Entrena el modelo RL en múltiples secuencias con múltiples épocas
        
        Args:
            sequence_numbers: Lista de números de secuencia (ej: ["0001", "0005"])
            epochs: Número de épocas (pasar por todas las secuencias)
            max_frames: Límite de frames por secuencia (None = todos)
            augmentation_probs: Lista de probabilidades de aumentación por época
                                (ej: [0.3, 0.5, 0.6] para aumentar progresivamente)
            save_checkpoints: Si guardar modelo después de cada secuencia
        
        Returns:
            dict: Estadísticas finales de entrenamiento
        """
        # Configurar probabilidades de aumentación progresivas
        if augmentation_probs is None:
            # Por defecto: aumentar progresivamente la dificultad
            augmentation_probs = [0.3, 0.5, 0.6]
            if epochs > 3:
                # Extender con valores altos
                augmentation_probs.extend([0.6] * (epochs - 3))
        elif len(augmentation_probs) < epochs:
            # Rellenar con el último valor
            augmentation_probs.extend([augmentation_probs[-1]] * (epochs - len(augmentation_probs)))
        
        self.log(f"\n{'='*80}")
        self.log(f"INICIO DE ENTRENAMIENTO MULTI-SECUENCIA")
        self.log(f"{'='*80}")
        self.log(f"Secuencias: {sequence_numbers}")
        self.log(f"Épocas: {epochs}")
        self.log(f"Frames por secuencia: {max_frames or 'Todos'}")
        self.log(f"Probabilidades de aumentación: {augmentation_probs}")
        self.log(f"Guardar checkpoints: {save_checkpoints}")
        self.log(f"{'='*80}\n")
        
        # Crear instancia SLAM en modo entrenamiento
        slam = RL_ORB_SLAM_GPS(
            fx=718.856, fy=718.856, cx=607.1928, cy=185.2157,
            training_mode=True,
            model_path=self.model_path,
            simulate_mobile_gps=False  # La aumentación maneja el ruido
        )
        
        # Entrenar por épocas
        for epoch in range(epochs):
            epoch_aug_prob = augmentation_probs[epoch]
            
            self.log(f"\n{'#'*80}")
            self.log(f"ÉPOCA {epoch + 1}/{epochs} (Aumentación: {epoch_aug_prob*100:.0f}%)")
            self.log(f"{'#'*80}")
            
            epoch_stats = {
                'epoch': epoch + 1,
                'augmentation_prob': epoch_aug_prob,
                'sequences': []
            }
            
            # Entrenar en cada secuencia
            for seq_num in sequence_numbers:
                seq_stats = self.train_on_sequence(
                    sequence_number=seq_num,
                    slam=slam,
                    max_frames=max_frames,
                    augmentation_prob=epoch_aug_prob
                )
                
                if seq_stats:
                    epoch_stats['sequences'].append(seq_stats)
                    self.training_stats['sequences_processed'].append(seq_stats)
                    self.training_stats['total_frames'] += seq_stats['frames_processed']
                
                # Guardar checkpoint después de cada secuencia
                if save_checkpoints:
                    checkpoint_path = self.save_checkpoint(
                        slam=slam,
                        epoch=epoch + 1,
                        sequence=seq_num
                    )
                    self.training_stats['checkpoints_saved'].append(checkpoint_path)
            
            # Resumen de época
            total_frames_epoch = sum(s['frames_processed'] for s in epoch_stats['sequences'])
            total_steps_epoch = sum(s['training_steps'] for s in epoch_stats['sequences'])
            
            self.log(f"\nÉPOCA {epoch + 1} COMPLETADA:")
            self.log(f"  Total frames: {total_frames_epoch}")
            self.log(f"  Total pasos entrenamiento: {total_steps_epoch}")
            self.log(f"  Experiencias en buffer: {len(slam.trainer.replay_buffer)}")
        
        # Guardar modelo final
        final_model_path = "LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth"
        metadata = {
            'training_type': 'multi_sequence_augmented',
            'sequences': sequence_numbers,
            'epochs': epochs,
            'total_frames': self.training_stats['total_frames'],
            'total_experiences': len(slam.trainer.replay_buffer),
            'augmentation_probs': augmentation_probs,
            'final_training_steps': len(slam.trainer.training_metrics['losses'])
        }
        slam.trainer.save_model(final_model_path, metadata)
        
        self.log(f"\nModelo final guardado: {final_model_path}")
        
        # Guardar estadísticas completas
        self.save_training_report()
        
        # Imprimir resumen final
        slam.trainer.print_training_summary()
        
        self.log(f"\n{'='*80}")
        self.log(f"ENTRENAMIENTO COMPLETADO")
        self.log(f"{'='*80}")
        self.log(f"Total frames procesados: {self.training_stats['total_frames']}")
        self.log(f"Total checkpoints guardados: {len(self.training_stats['checkpoints_saved'])}")
        self.log(f"Experiencias finales en buffer: {len(slam.trainer.replay_buffer)}")
        self.log(f"{'='*80}\n")
        
        return self.training_stats
    
    def save_checkpoint(self, slam, epoch, sequence):
        """Guarda checkpoint intermedio del modelo"""
        checkpoint_dir = "LMS/LMS_RL_ORB_GPS/model/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_epoch{epoch}_seq{sequence}_{timestamp}.pth"
        )
        
        metadata = {
            'epoch': epoch,
            'sequence': sequence,
            'timestamp': timestamp,
            'total_experiences': len(slam.trainer.replay_buffer),
            'training_steps': len(slam.trainer.training_metrics['losses'])
        }
        
        slam.trainer.save_model(checkpoint_path, metadata)
        
        self.log(f"  Checkpoint guardado: {checkpoint_path}", print_console=False)
        
        return checkpoint_path
    
    def save_training_report(self):
        """Guarda reporte detallado del entrenamiento"""
        report_dir = "LMS/LMS_RL_ORB_GPS/logs"
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"training_report_{timestamp}.json")
        
        self.training_stats['end_time'] = datetime.now().isoformat()
        
        with open(report_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        self.log(f"Reporte de entrenamiento guardado: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar modelo RL con múltiples secuencias KITTI y aumentación de datos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Entrenamiento rápido con 3 secuencias cortas
  python train_multi_sequence.py --sequences 0001 0005 0009 --epochs 2 --max_frames 150

  # Entrenamiento completo con 5 secuencias y 3 épocas
  python train_multi_sequence.py --sequences 0001 0005 0009 0013 0018 --epochs 3

  # Continuar entrenamiento desde modelo existente
  python train_multi_sequence.py --sequences 0001 0005 --epochs 2 --model_path LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth

  # Entrenamiento con aumentación progresiva personalizada
  python train_multi_sequence.py --sequences 0001 0005 0009 --epochs 3 --augmentation 0.3 0.5 0.7
        """
    )
    
    parser.add_argument(
        '--sequences',
        nargs='+',
        required=True,
        help='Números de secuencias KITTI a procesar (ej: 0001 0005 0009)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Número de épocas de entrenamiento (default: 3)'
    )
    
    parser.add_argument(
        '--max_frames',
        type=int,
        default=None,
        help='Límite de frames por secuencia (default: todos)'
    )
    
    parser.add_argument(
        '--augmentation',
        nargs='+',
        type=float,
        default=None,
        help='Probabilidades de aumentación por época (ej: 0.3 0.5 0.6). Default: progresivo'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Ruta a modelo pre-entrenado para continuar entrenamiento'
    )
    
    parser.add_argument(
        '--kitti_path',
        type=str,
        default='kitti_data/2011_09_26',
        help='Ruta base a secuencias KITTI (default: kitti_data/2011_09_26)'
    )
    
    parser.add_argument(
        '--no_checkpoints',
        action='store_true',
        help='No guardar checkpoints intermedios (solo modelo final)'
    )
    
    args = parser.parse_args()
    
    # Crear entrenador
    trainer = MultiSequenceTrainer(
        kitti_base_path=args.kitti_path,
        model_path=args.model_path
    )
    
    # Entrenar
    stats = trainer.train(
        sequence_numbers=args.sequences,
        epochs=args.epochs,
        max_frames=args.max_frames,
        augmentation_probs=args.augmentation,
        save_checkpoints=not args.no_checkpoints
    )
    
    print("\nEntrenamiento completado exitosamente!")
    print(f"Ver log completo en: {trainer.log_file}")
    print(f"Modelo guardado en: LMS/LMS_RL_ORB_GPS/model/trained_rl_agent.pth")


if __name__ == "__main__":
    main()
