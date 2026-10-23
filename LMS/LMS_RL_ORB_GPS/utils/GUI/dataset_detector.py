"""
Tells from a folder's layout whether it is a KITTI sequence or a phone
recording (video + GPS CSV).
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DatasetInfo:
    """What was found in a folder."""
    dataset_type: str  # 'kitti', 'mobile' or 'unknown'
    base_path: str
    sequence_path: Optional[str] = None
    video_path: Optional[str] = None
    gps_csv_path: Optional[str] = None
    timestamps_path: Optional[str] = None
    confidence: float = 0.0
    error_message: Optional[str] = None


class DatasetDetector:
    """Detects whether a folder is a KITTI sequence or a phone recording."""

    @staticmethod
    def detect_dataset_type(base_path: str) -> DatasetInfo:
        if not os.path.exists(base_path):
            return DatasetInfo(
                dataset_type='unknown',
                base_path=base_path,
                confidence=0.0,
                error_message=f"Folder does not exist: {base_path}"
            )

        kitti_info = DatasetDetector._detect_kitti(base_path)
        if kitti_info.dataset_type == 'kitti':
            return kitti_info

        mobile_info = DatasetDetector._detect_mobile(base_path)
        if mobile_info.dataset_type == 'mobile':
            return mobile_info

        return DatasetInfo(
            dataset_type='unknown',
            base_path=base_path,
            confidence=0.0,
            error_message="Neither a KITTI nor a phone recording layout"
        )

    @staticmethod
    def _detect_kitti(base_path: str) -> DatasetInfo:
        """image_02/data with PNGs and oxts/data with TXTs, anywhere below base_path."""
        confidence = 0.0
        image_dir = None
        gps_dir = None
        sequence_base = None

        for root, dirs, files in os.walk(base_path):
            if "image_02" in dirs:
                img_candidate = os.path.join(root, "image_02", "data")
                if os.path.exists(img_candidate):
                    png_files = [f for f in os.listdir(img_candidate)
                                 if f.endswith('.png')]
                    if png_files:
                        image_dir = img_candidate
                        # main.py wants the folder that holds image_02/.
                        sequence_base = root
                        confidence += 0.5

            if "oxts" in dirs:
                gps_candidate = os.path.join(root, "oxts", "data")
                if os.path.exists(gps_candidate):
                    txt_files = [f for f in os.listdir(gps_candidate)
                                 if f.endswith('.txt')]
                    if txt_files:
                        gps_dir = gps_candidate
                        confidence += 0.5

            if image_dir and gps_dir:
                break

        if confidence >= 0.5 and sequence_base:
            return DatasetInfo(
                dataset_type='kitti',
                base_path=sequence_base,
                sequence_path=sequence_base,
                confidence=confidence
            )

        return DatasetInfo(
            dataset_type='unknown',
            base_path=base_path,
            confidence=0.0
        )

    @staticmethod
    def _detect_mobile(base_path: str) -> DatasetInfo:
        """An .mp4, a location CSV and a timestamps TXT below base_path."""
        confidence = 0.0
        video_path = None
        gps_csv = None
        timestamps = None

        for root, dirs, files in os.walk(base_path):
            for file in files:
                full_path = os.path.join(root, file)
                if file.endswith('.mp4') or file.endswith('.MP4'):
                    video_path = full_path
                    confidence += 0.4
                if 'location' in file.lower() and file.endswith('.csv'):
                    gps_csv = full_path
                    confidence += 0.4
                if 'timestamp' in file.lower() and file.endswith('.txt'):
                    timestamps = full_path
                    confidence += 0.2
            if confidence >= 0.8:
                break

        if confidence >= 0.6:  # at least video and GPS
            return DatasetInfo(
                dataset_type='mobile',
                base_path=base_path,
                video_path=video_path,
                gps_csv_path=gps_csv,
                timestamps_path=timestamps,
                confidence=confidence
            )

        return DatasetInfo(
            dataset_type='unknown',
            base_path=base_path,
            confidence=0.0
        )

    @staticmethod
    def validate_dataset(info: DatasetInfo) -> Tuple[bool, Optional[str]]:
        """(is_valid, error_message): whether every file the system reads is there."""
        if info.dataset_type == 'unknown':
            return False, info.error_message or "Unknown dataset type"

        if info.dataset_type == 'kitti':
            if not info.sequence_path:
                return False, "Could not find the KITTI sequence"
            if not os.path.exists(info.sequence_path):
                return False, f"Sequence does not exist: {info.sequence_path}"
            image_path = os.path.join(info.sequence_path, "image_02", "data")
            oxts_path = os.path.join(info.sequence_path, "oxts", "data")
            if not os.path.exists(image_path):
                return False, f"No image_02/data in: {info.sequence_path}"
            if not os.path.exists(oxts_path):
                return False, f"No oxts/data in: {info.sequence_path}"
            return True, None

        if info.dataset_type == 'mobile':
            if not info.video_path:
                return False, "No video file (.mp4)"
            if not info.gps_csv_path:
                return False, "No GPS file (location.csv)"
            if not info.timestamps_path:
                return False, "No timestamps file"
            if not os.path.exists(info.video_path):
                return False, f"Video does not exist: {info.video_path}"
            if not os.path.exists(info.gps_csv_path):
                return False, f"GPS CSV does not exist: {info.gps_csv_path}"
            if not os.path.exists(info.timestamps_path):
                return False, f"Timestamps file does not exist: {info.timestamps_path}"
            return True, None

        return False, "Dataset type not supported"

    @staticmethod
    def get_dataset_summary(info: DatasetInfo) -> str:
        if info.dataset_type == 'unknown':
            return f"Unknown dataset\n{info.error_message or ''}"

        summary = f"Dataset found: {info.dataset_type.upper()}\n"
        summary += f"  Confidence: {info.confidence*100:.0f}%\n"
        summary += f"  Base folder: {info.base_path}\n"
        if info.dataset_type == 'kitti':
            summary += f"  Sequence: {info.sequence_path}\n"
        elif info.dataset_type == 'mobile':
            summary += f"  Video: {os.path.basename(info.video_path) if info.video_path else 'N/A'}\n"
            summary += f"  GPS: {os.path.basename(info.gps_csv_path) if info.gps_csv_path else 'N/A'}\n"
            summary += f"  Timestamps: {os.path.basename(info.timestamps_path) if info.timestamps_path else 'N/A'}\n"
        return summary
