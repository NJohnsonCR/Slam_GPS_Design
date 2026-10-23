"""
Runs the classic RL-ORB-SLAM-GPS on a folder chosen in the launcher.
"""

import os
import sys
from io import StringIO
from typing import Optional

from .dataset_detector import DatasetDetector

current_dir = os.path.dirname(os.path.abspath(__file__))
lms_rl_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, lms_rl_dir)

from main import RL_ORB_SLAM_GPS


class GUIProcessor:
    """Processes a dataset folder from the launcher, detecting its type."""

    @staticmethod
    def process_directory(
        base_path: str,
        max_frames: Optional[int] = None,
        training_mode: bool = False,
        augmentation_prob: float = 0.0,
        progress_callback=None
    ) -> dict:
        """
        base_path is the folder chosen; max_frames None processes them all;
        augmentation_prob applies to KITTI only; progress_callback(message)
        gets every line of progress. Returns a dict with 'success' and either
        the result or the error.
        """
        def log(message: str):
            if progress_callback:
                progress_callback(message)
            else:
                print(message)

        class OutputCapture:
            """stdout into the progress callback: the system reports with print."""
            def __init__(self, callback):
                self.callback = callback
                self.buffer = StringIO()

            def write(self, text):
                if text and text.strip():
                    if self.callback:
                        self.callback(text.rstrip())
                    self.buffer.write(text)

            def flush(self):
                pass

        log(f"Analysing folder: {base_path}")
        dataset_info = DatasetDetector.detect_dataset_type(base_path)
        log(f"\n{DatasetDetector.get_dataset_summary(dataset_info)}")

        is_valid, error_msg = DatasetDetector.validate_dataset(dataset_info)
        if not is_valid:
            return {
                'success': False,
                'error': error_msg,
                'dataset_type': dataset_info.dataset_type
            }

        log("\nStarting RL-ORB-SLAM with GPS...")
        old_stdout = sys.stdout
        capture = OutputCapture(progress_callback)

        try:
            sys.stdout = capture
            system = RL_ORB_SLAM_GPS(
                training_mode=training_mode,
                simulate_mobile_gps=False
            )

            if dataset_info.dataset_type == 'kitti':
                log(f"\nProcessing KITTI sequence: {dataset_info.sequence_path}")
                system.process_kitti_sequence(
                    sequence_path=dataset_info.sequence_path,
                    max_frames=max_frames,
                    augmentation_prob=augmentation_prob
                )
                result_type = "KITTI"

            elif dataset_info.dataset_type == 'mobile':
                log("\nProcessing phone video")
                system.process_mobile_sequence(
                    video_path=dataset_info.video_path,
                    location_csv=dataset_info.gps_csv_path,
                    frame_timestamps_file=dataset_info.timestamps_path,
                    max_frames=3000
                )
                result_type = "Mobile"

            else:
                sys.stdout = old_stdout
                return {
                    'success': False,
                    'error': f"Dataset type not supported: {dataset_info.dataset_type}",
                    'dataset_type': dataset_info.dataset_type
                }

            log("\nProcessing finished")
            sys.stdout = old_stdout
            return {
                'success': True,
                'dataset_type': dataset_info.dataset_type,
                'result_type': result_type,
                'keyframes': len(system.keyframe_poses) if hasattr(system, 'keyframe_poses') else 0
            }

        except Exception as e:
            import traceback
            sys.stdout = old_stdout
            error_trace = traceback.format_exc()
            log(f"\nError while processing: {e}")
            log(error_trace)
            return {
                'success': False,
                'error': str(e),
                'traceback': error_trace,
                'dataset_type': dataset_info.dataset_type
            }

        finally:
            sys.stdout = old_stdout
