"""
Metric scale estimation from monocular depth.

WHAT IT SOLVES
    cv2.recoverPose gives the direction of motion but not its magnitude: the
    translation vector always has norm 1. That magnitude used to be taken from
    the GPS, which chained both sensors together -- when the GPS degraded, so
    did the camera.

    This module gives the camera a scale of its OWN, taken from the image:
      1. Triangulate correspondences with the unit-norm pose  -> depths in
         "VO units".
      2. A metric depth model predicts those same depths in metres.
      3. scale = median(metres / units)  ->  distance travelled in metres.

THE TRIANGULATION BASELINE IS CRITICAL
    Triangulation needs parallax. At 30 fps a vehicle advances ~14 cm between
    neighbouring frames, not enough for points 10-50 m away. Measured on real
    data: 1 frame apart gave a 2.64x bias, ~0.33 s apart gives 1.08x. That is
    why the estimator compares frames `min_baseline_s` apart, never
    consecutive ones.

OFFLINE AND LIVE
    The same class serves both: it is fed frames with timestamps and returns a
    SPEED (m/s) once it has enough baseline. Speed is smooth, so the pipeline
    can hold it between updates.
"""

import threading
from collections import deque
from typing import Optional

import cv2
import numpy as np


class DepthScaleEstimator:
    """
    Estimates the vehicle's metric speed from the image.

    Deliberately independent of the GPS: it never consumes it.
    """

    MODEL_DEFAULT = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

    def __init__(self, camera_matrix: np.ndarray,
                 model: str = MODEL_DEFAULT,
                 min_baseline_s: float = 0.33,
                 min_parallax_px: float = 1.0,
                 depth_range=(3.0, 60.0),
                 min_points: int = 20,
                 max_speed_ms: float = 40.0,
                 smooth_window: int = 7,
                 hist_len: int = 120,
                 device: Optional[int] = None,
                 orb=None, matcher=None, lowe_ratio: float = 0.75):
        self.K = camera_matrix
        self.min_baseline_s = min_baseline_s
        self.min_parallax = min_parallax_px
        self.dmin, self.dmax = depth_range
        self.min_points = min_points
        self.lowe = lowe_ratio
        # Physical plausibility filter. A vehicle does not jump from 7 to
        # 75 m/s between frames; those are triangulation errors.
        self.max_speed = max_speed_ms
        self._raw = deque(maxlen=smooth_window)
        self.n_rejected = 0

        self.orb = orb if orb is not None else cv2.ORB_create(2000)
        self.matcher = matcher if matcher is not None else cv2.BFMatcher(cv2.NORM_HAMMING)

        if device is None:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        from transformers import pipeline
        self._pipe = pipeline("depth-estimation", model=model, device=device)

        # Frame history, only deep enough to cover min_baseline_s. Each entry
        # retains a full image, so storing more costs memory without changing
        # the result: try_update() picks the MOST RECENT frame that already
        # has baseline.
        self._hist = deque(maxlen=hist_len)
        self._lock = threading.Lock()
        self._velocity = None          # m/s, last valid estimate
        self._last_t = None
        self.n_ok = 0
        self.n_fail = 0

    # ------------------------------------------------------------------ API

    @property
    def velocity(self) -> Optional[float]:
        """Last estimated speed in m/s, or None if there is none yet."""
        with self._lock:
            return self._velocity

    def push(self, t_s: float, image: np.ndarray, kp=None, des=None):
        """Register a frame. Features are computed if not supplied."""
        if kp is None or des is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            kp, des = self.orb.detectAndCompute(gray, None)
        self._hist.append((t_s, image, kp, des))

    def try_update(self) -> Optional[float]:
        """Estimate speed from the newest frame against an old enough one."""
        if len(self._hist) < 2:
            return None
        t1, img1, kp1, des1 = self._hist[-1]

        previous = None
        for entry in reversed(self._hist):
            if t1 - entry[0] >= self.min_baseline_s:
                previous = entry
                break
        if previous is None:
            return None

        t0, img0, kp0, des0 = previous
        dt = t1 - t0
        dist = self._metric_distance(img0, kp0, des0, kp1, des1)
        if dist is None or dt <= 0:
            self.n_fail += 1
            return None

        v_raw = dist / dt

        # 1) Hard physical limit. Triangulation errors produce absurd values;
        #    peaks of 75 m/s (270 km/h) were observed on real data. Rejected
        #    as impossible, without comparing against any other source.
        if not np.isfinite(v_raw) or v_raw < 0 or v_raw > self.max_speed:
            self.n_rejected += 1
            return None

        # 2) Rolling median. A hard acceleration test discards too many valid
        #    estimates (the estimator is noisy, not just the vehicle). The
        #    median suppresses remaining spikes while keeping real variation.
        self._raw.append(v_raw)
        v = float(np.median(self._raw))

        with self._lock:
            self._velocity = v
            self._last_t = t1
        self.n_ok += 1
        return v

    # -------------------------------------------------------------- internal

    def _metric_distance(self, img0, kp0, des0, kp1, des1) -> Optional[float]:
        """Metres between the two views, or None if not trustworthy."""
        if des0 is None or des1 is None or len(des0) < 2 or len(des1) < 2:
            return None

        knn = self.matcher.knnMatch(des0, des1, k=2)
        good = [m for m, n in knn if m.distance < self.lowe * n.distance]
        if len(good) <= 15:
            return None

        p0 = np.float32([kp0[m.queryIdx].pt for m in good])
        p1 = np.float32([kp1[m.trainIdx].pt for m in good])

        E, _ = cv2.findEssentialMat(p0, p1, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3):
            return None
        try:
            _, R, t, _ = cv2.recoverPose(E, p0, p1, self.K)
        except cv2.error:
            return None

        # Drop low-parallax points: they triangulate badly.
        keep = np.linalg.norm(p1 - p0, axis=1) > self.min_parallax
        if keep.sum() < self.min_points:
            return None
        a, b = p0[keep], p1[keep]

        P0 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = self.K @ np.hstack([R, t.reshape(3, 1)])
        X = cv2.triangulatePoints(P0, P1, a.T, b.T)
        w = X[3]
        ok = np.abs(w) > 1e-9
        if ok.sum() < self.min_points:
            return None
        X = X[:, ok] / w[ok]
        a = a[ok]
        z = X[2]
        ok = z > 1e-6
        z, a = z[ok], a[ok]
        if len(z) < self.min_points:
            return None

        dmap = self.depth_map(img0)
        h, wd = dmap.shape
        zm = dmap[np.clip(a[:, 1].astype(int), 0, h - 1),
                  np.clip(a[:, 0].astype(int), 0, wd - 1)]

        ok = (zm > self.dmin) & (zm < self.dmax)
        if ok.sum() < self.min_points:
            return None

        # ||t|| == 1 by construction, so the median ratio IS the metric
        # distance travelled between the two views.
        ratios = zm[ok] / z[ok]
        return float(np.median(ratios))

    def depth_map(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        out = self._pipe(Image.fromarray(rgb))
        return np.asarray(out["predicted_depth"], dtype=np.float32)
