"""
Data sources for the real-time pipeline.

Sources are interchangeable: they all deliver the same thing, so the rest of
the system never knows where the data came from. Today it comes from recorded
files; tomorrow it will come from the Android app over UDP.

    FrameSource  ->  (timestamp_ns, frame)
    GpsSource    ->  pushes (timestamp_ns, position) into a buffer

Replay runs in two modes, and the distinction matters for every measurement:

    strict         sleeps until each frame's real timestamp and drops for real.
                   Reproduces street behaviour, but is NOT deterministic.
    deterministic  delivers every frame without sleeping. Drops are computed
                   afterwards from the measured times. Reproducible, so it is
                   the mode used for development and for reported numbers.
"""

import collections
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------- GPS

class GpsBuffer:
    """
    GPS fix buffer with CAUSAL access.

    `latest_before` only returns fixes that had already arrived at the queried
    instant. This is what stops the system from using information from the
    future -- the easiest mistake to make when moving from offline to live.
    """

    def __init__(self, maxlen: int = 600):
        self._lock = threading.Lock()
        self._buf = collections.deque(maxlen=maxlen)
        self.received = 0

    def push(self, t_ns: int, pos: np.ndarray):
        with self._lock:
            self._buf.append((t_ns, pos))
            self.received += 1

    def latest_before(self, t_ns: int) -> Optional[Tuple[int, np.ndarray]]:
        """Most recent fix that had ALREADY ARRIVED at t_ns."""
        with self._lock:
            for ts, pos in reversed(self._buf):
                if ts <= t_ns:
                    return ts, pos
        return None

    def __len__(self):
        with self._lock:
            return len(self._buf)


# ------------------------------------------------------------------ frames

@dataclass
class Frame:
    t_ns: int          # capture timestamp (source clock)
    index: int         # position in the sequence
    image: np.ndarray


class ReplayFrameSource:
    """Replays a recorded video honouring the real time between frames."""

    def __init__(self, video_path: str, timestamps_ns, crop_bottom: float = 0.0,
                 strict: bool = True, max_frames: Optional[int] = None):
        self.video_path = video_path
        self.timestamps = np.asarray(timestamps_ns, dtype=np.int64)
        self.crop_bottom = crop_bottom      # bottom fraction to cut: car dashboard
        self.strict = strict
        self.max_frames = max_frames
        self._cap = None

    def __enter__(self):
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self.video_path}")
        return self

    def __exit__(self, *exc):
        if self._cap is not None:
            self._cap.release()

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        # The dashboard sits in the lower part of the frame and produces
        # correspondences that claim the vehicle did not move.
        if self.crop_bottom > 0:
            h = img.shape[0]
            img = img[: int(h * (1.0 - self.crop_bottom))]
        return img

    def frames(self):
        """Frame generator. In strict mode it blocks until real time."""
        n = len(self.timestamps)
        if self.max_frames:
            n = min(n, self.max_frames)

        t0_wall = time.perf_counter()
        t0_data = int(self.timestamps[0])

        for i in range(n):
            ok, img = self._cap.read()
            if not ok:
                break

            if self.strict:
                target = t0_wall + (int(self.timestamps[i]) - t0_data) / 1e9
                wait = target - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)

            yield Frame(t_ns=int(self.timestamps[i]), index=i,
                        image=self._preprocess(img))


class ReplayGpsSource:
    """
    Delivers GPS fixes at their real rate (1 Hz in the mobile data).

    Runs in its own thread: GPS arrives when it arrives, not when processing
    asks for it.
    """

    _idx = 0

    def __init__(self, fixes, buffer: GpsBuffer, strict: bool = True):
        self.fixes = fixes                  # list of (t_ns, np.array([x, y, z]))
        self.buffer = buffer
        self.strict = strict
        self._thread = None
        self._stop = threading.Event()

    def start(self, t0_wall: float, t0_data_ns: int):
        self._thread = threading.Thread(
            target=self._run, args=(t0_wall, t0_data_ns), daemon=True)
        self._thread.start()

    def _run(self, t0_wall: float, t0_data_ns: int):
        for t_ns, pos in self.fixes:
            if self._stop.is_set():
                return
            if self.strict:
                target = t0_wall + (t_ns - t0_data_ns) / 1e9
                wait = target - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            self.buffer.push(t_ns, pos)

    def push_all_up_to(self, t_ns: int):
        """Deterministic mode: publish every fix that would have arrived."""
        while self._idx < len(self.fixes) and self.fixes[self._idx][0] <= t_ns:
            t, pos = self.fixes[self._idx]
            self.buffer.push(t, pos)
            self._idx += 1

    def stop(self):
        self._stop.set()


# -------------------------------------------------------------------- load

def load_mobile_session(session_dir: str, latlon_to_utm):
    """
    Read a session recorded with MARS Logger (mobile_data/ format).

    Returns (frame_timestamps_ns, gps_fixes). Both are aligned through the
    'Unix time' field, which the app writes from a single clock for every
    sensor.
    """
    import os

    import pandas as pd

    ft = pd.read_csv(os.path.join(session_dir, "frame_timestamps.txt"))
    frame_t = ft["Unix time[nanosec]"].values.astype(np.int64)

    loc = pd.read_csv(os.path.join(session_dir, "location.csv"))
    origin = None
    fixes = []
    for _, r in loc.iterrows():
        utm = latlon_to_utm(r["latitude[degrees]"], r["longitude[degrees]"],
                            r["altitude[meters]"])
        if origin is None:
            origin = utm
        fixes.append((int(r["Unix time[nanosecond]"]), utm - origin))

    return frame_t, fixes
