"""
Real-time pipeline: threads, queues and latency measurement.

    [capture thread] --queue of 1, drops oldest--> [processing thread]
    [GPS thread]     --causal circular buffer---->        |
    [depth thread]   --speed (m/s), ~3 Hz-------->        |
                                                          v
                                             [metrics: latency, Hz, drops]

WHY THREADS AND NOT A SINGLE LOOP
    The camera produces a frame every 33 ms whether it is being served or not.
    A single 60 ms loop lets frames pile up in the system buffer, so each read
    returns an older one. The FPS counter still looks fine while the reported
    position falls seconds behind.

WHY THE QUEUE HOLDS EXACTLY ONE ITEM
    Queueing everything would keep throughput looking good while latency grew
    without bound. An approximate current position is worth more than an
    accurate old one, so intermediate frames are dropped on purpose. The drop
    rate is itself a metric that has to be reported.

WHY METRIC SCALE RUNS IN ITS OWN THREAD
    The depth model takes 66-150 ms per frame, five times the 33 ms budget of
    the main loop. It does not need to be in line: what it produces is a
    SPEED, and vehicle speed is smooth. Estimating it at ~3 Hz and holding it
    between updates is enough while the camera keeps supplying direction at
    30 Hz. Each frame's metric step is direction * (speed * dt).
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .sources import Frame, GpsBuffer


# ----------------------------------------------------------- communication

def put_drop_oldest(q: queue.Queue, item, counter: list):
    """Single-slot queue: an unconsumed item is discarded to make room."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
            counter[0] += 1
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            counter[0] += 1


# --------------------------------------------------------------- metrics

@dataclass
class StageTimes:
    detect: float = 0.0
    match: float = 0.0
    pose: float = 0.0

    @property
    def total(self):
        return self.detect + self.match + self.pose


@dataclass
class Metrics:
    """Per-stage times plus end-to-end latency."""

    stages: List[StageTimes] = field(default_factory=list)
    e2e_ms: List[float] = field(default_factory=list)
    pc_ms: List[float] = field(default_factory=list)      # live: arrival -> result
    gps_age_ms: List[float] = field(default_factory=list)
    depth_ms: List[float] = field(default_factory=list)
    processed: int = 0
    dropped: int = 0
    no_gps: int = 0
    no_scale: int = 0
    rejected_poses: int = 0
    t_start: float = 0.0
    t_end: float = 0.0

    def summary(self) -> dict:
        def pct(v, p):
            return float(np.percentile(v, p)) if v else float("nan")

        tot = [s.total * 1000 for s in self.stages]
        wall = max(self.t_end - self.t_start, 1e-9)
        total_frames = self.processed + self.dropped
        out = {
            "frames_procesados": self.processed,
            "frames_descartados": self.dropped,
            "tasa_descarte_%": 100.0 * self.dropped / max(total_frames, 1),
            "duracion_s": wall,
            "hz_efectivo": self.processed / wall,
            "proc_ms_p50": pct(tot, 50),
            "proc_ms_p95": pct(tot, 95),
            "proc_ms_max": max(tot) if tot else float("nan"),
            "detect_ms_p50": pct([s.detect * 1000 for s in self.stages], 50),
            "match_ms_p50": pct([s.match * 1000 for s in self.stages], 50),
            "pose_ms_p50": pct([s.pose * 1000 for s in self.stages], 50),
            "e2e_ms_p50": pct(self.e2e_ms, 50),
            "e2e_ms_p95": pct(self.e2e_ms, 95),
            "gps_age_ms_p50": pct(self.gps_age_ms, 50),
            "gps_age_ms_p95": pct(self.gps_age_ms, 95),
            "frames_sin_gps": self.no_gps,
            "poses_giro_imposible": self.rejected_poses,
            # Scale runs on another thread, so it is reported separately and
            # never counted against the per-frame budget.
            "depth_estimaciones": len(self.depth_ms),
            "depth_ms_p50": pct(self.depth_ms, 50),
            "depth_ms_p95": pct(self.depth_ms, 95),
            "frames_sin_escala": self.no_scale,
        }
        if self.pc_ms:
            out["pc_ms_p50"] = pct(self.pc_ms, 50)
            out["pc_ms_p95"] = pct(self.pc_ms, 95)
        return out


# ------------------------------------------------------------- processing

class VisualFrontEnd:
    """
    Visual front-end: ORB, matching and relative pose estimation.

    Uses the same configuration as the offline pipeline -- ORB_create(2000),
    BFMatcher Hamming, Lowe ratio 0.75, minimum 15 matches -- so both systems
    stay comparable. The T6 invariant in scripts/rl/sanity_checks.py fails
    loudly if these ever drift apart.

    mask_bottom is the lower fraction of the frame where ORB does not look for
    points: the dashboard, or the hood that reflects the scene. The frame is
    not cropped, because the depth model's metric scale changes with framing.
    """

    MIN_MATCHES = 15
    # Safety net: a car does not turn this much between frames (the gyroscope
    # peaked at 6.1 degrees, on a bump), so such a pose is a failed estimate.
    MAX_ROTATION_DEG = 10.0

    def __init__(self, camera_matrix: np.ndarray, n_features: int = 2000,
                 lowe_ratio: float = 0.75, mask_bottom: float = 0.0):
        self.K = camera_matrix
        self.orb = cv2.ORB_create(n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.lowe = lowe_ratio
        self.mask_bottom = mask_bottom
        self.n_rejected = 0                 # poses dropped by MAX_ROTATION_DEG
        self._mask = None
        self._prev_kp = None
        self._prev_des = None

    def _feature_mask(self, shape):
        """Where ORB may look for points, or None for the whole frame."""
        if self.mask_bottom <= 0:
            return None
        if self._mask is None or self._mask.shape != shape:
            # ORB already keeps its points this far from the frame border; the
            # same margin above the masked rows gives the points of a crop.
            rows = int(shape[0] * (1.0 - self.mask_bottom)) - self.orb.getEdgeThreshold()
            self._mask = np.zeros(shape, np.uint8)
            self._mask[:max(rows, 0)] = 255
        return self._mask

    def _recover_pose(self, E, p0, p1):
        """
        Pose from E, letting every point vote among its four solutions.

        By default recoverPose ignores points beyond 50 baselines. With little
        motion between frames none is left, and the tie returns the first
        solution: about half of the time, the right one turned 180 degrees
        about the baseline. The inlier count keeps the default meaning, so it
        still drops to ~0 when the camera barely moves.
        """
        _, R, t, mask, X = cv2.recoverPose(E, p0, p1, self.K, distanceThresh=1e7)
        with np.errstate(divide="ignore", invalid="ignore"):
            X = X[:3] / X[3]
            near = (mask.ravel() > 0) & (X[2] < 50) & ((R @ X + t)[2] < 50)
        return R, t.ravel(), int(np.count_nonzero(near))

    def process(self, image: np.ndarray):
        """
        Return (R, t, n_matches, n_inliers, StageTimes). ||t|| == 1, or R = I
        and t = 0 when there is no reliable pose.
        """
        st = StageTimes()

        t0 = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, self._feature_mask(gray.shape))
        st.detect = time.perf_counter() - t0

        R, t = np.eye(3), np.zeros(3)
        n_matches = n_inliers = 0

        if self._prev_des is not None and des is not None \
                and len(des) > 1 and len(self._prev_des) > 1:
            t0 = time.perf_counter()
            knn = self.matcher.knnMatch(self._prev_des, des, k=2)
            good = [m for m, n in knn if m.distance < self.lowe * n.distance]
            n_matches = len(good)
            st.match = time.perf_counter() - t0

            if n_matches > self.MIN_MATCHES:
                t0 = time.perf_counter()
                p0 = np.float32([self._prev_kp[m.queryIdx].pt for m in good])
                p1 = np.float32([kp[m.trainIdx].pt for m in good])
                E, _ = cv2.findEssentialMat(p0, p1, self.K, method=cv2.RANSAC,
                                            prob=0.999, threshold=1.0)
                if E is not None and E.shape == (3, 3):
                    try:
                        R_est, t_est, n_in = self._recover_pose(E, p0, p1)
                        angle = np.degrees(np.arccos(np.clip((np.trace(R_est) - 1) / 2, -1, 1)))
                        if angle <= self.MAX_ROTATION_DEG:
                            R, t, n_inliers = R_est, t_est, n_in
                        else:
                            self.n_rejected += 1
                    except cv2.error:
                        pass
                st.pose = time.perf_counter() - t0

        self._prev_kp, self._prev_des = kp, des
        return R, t, n_matches, n_inliers, st

    @property
    def last_features(self):
        """
        Keypoints and descriptors of the last processed frame.

        Exposed so the scale estimator can reuse them instead of running ORB
        again (~9 ms per estimation) and so both modules see identical
        features.
        """
        return self._prev_kp, self._prev_des


# ------------------------------------------------------------ metric scale

class DepthScaleWorker:
    """
    Runs the DepthScaleEstimator outside the main loop and exposes the latest
    speed.

    TWO MODES, FOR THE SAME REASON AS REPLAY (see run_replay.py)
        threaded=True   estimation runs on its own thread behind a single-slot
                        drop-oldest queue. Real behaviour: the main loop never
                        waits for the model. NOT deterministic -- which frames
                        it manages to process depends on the clock.
        threaded=False  estimation runs in line, at the same data instants.
                        Reproducible. Its cost is measured separately and is
                        NOT added to the per-frame budget, because in the real
                        system that work happens on another thread. Strict
                        mode is what validates that assumption.

    In both modes the rate is driven by DATA time, not wall-clock time, so the
    set of frames handed to the model is the same either way.
    """

    def __init__(self, estimator, rate_hz: float = 3.0, threaded: bool = True):
        self.est = estimator
        self.period = 1.0 / max(rate_hz, 1e-9)
        self.threaded = threaded
        self.times_ms: List[float] = []
        self.n_submitted = 0
        self.n_dropped = [0]
        self._last_submit = None
        self._q = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = None

    @property
    def velocity(self) -> Optional[float]:
        return self.est.velocity

    def start(self):
        if self.threaded:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def submit(self, t_s: float, image: np.ndarray, kp=None, des=None) -> bool:
        """
        Offer a frame to the estimator. False if skipped by rate limiting.

        Rate limiting is deliberate: feeding 30 frames per second to a model
        that takes 100 ms would only fill a queue.
        """
        if self._last_submit is not None and t_s - self._last_submit < self.period:
            return False
        self._last_submit = t_s
        self.n_submitted += 1

        if self.threaded:
            put_drop_oldest(self._q, (t_s, image, kp, des), self.n_dropped)
        else:
            self._estimate(t_s, image, kp, des)
        return True

    def stop(self):
        # No sentinel is pushed on purpose: the queue holds one item and put()
        # would block forever if the thread had already exited. The thread
        # notices by itself because its get() times out every 0.5 s.
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            self._estimate(*item)

    def _estimate(self, t_s, image, kp, des):
        t0 = time.perf_counter()
        self.est.push(t_s, image, kp, des)
        self.est.try_update()
        self.times_ms.append((time.perf_counter() - t0) * 1000)


# --------------------------------------------------------------- pipeline

class RealtimePipeline:
    """
    Ties source, threads and metrics together.

    strict=True   real behaviour: sleeps and drops. Not deterministic.
    strict=False  processes everything and works out AFTERWARDS what would
                  have been dropped. Reproducible; the mode for development
                  and for reported numbers.
    """

    def __init__(self, front_end: VisualFrontEnd, gps_buffer: GpsBuffer,
                 strict: bool = True, on_result=None,
                 scale_worker: Optional[DepthScaleWorker] = None):
        self.fe = front_end
        self.gps = gps_buffer
        self.strict = strict
        self.on_result = on_result
        self.scale = scale_worker
        self.metrics = Metrics()
        self._q = queue.Queue(maxsize=1)
        self._dropped = [0]
        self._stop = threading.Event()
        self._t_prev_ns = None

    def _capture_worker(self, source):
        """
        Strict mode drops the old frame: that is the real behaviour.

        Deterministic mode does NOT drop -- it enqueues blocking -- because
        the goal is to measure how long EACH frame takes. Drops are derived
        afterwards by simulate_drops() from those times.
        """
        try:
            for frame in source.frames():
                if self._stop.is_set():
                    break
                if self.strict:
                    put_drop_oldest(self._q, frame, self._dropped)
                else:
                    self._q.put(frame)          # blocks until there is room
        finally:
            self._q.put(None)                   # end sentinel

    def run(self, source, t0_data_ns: Optional[int] = None):
        """Process until the source ends. t0_data_ns is only used by replay."""
        self.metrics.t_start = time.perf_counter()
        t0_wall = self.metrics.t_start

        cap_thread = threading.Thread(
            target=self._capture_worker, args=(source,), daemon=True)
        cap_thread.start()

        if self.scale is not None:
            self.scale.start()

        while True:
            try:
                item = self._q.get(timeout=5.0)
            except queue.Empty:
                break
            if item is None:
                break

            frame: Frame = item
            R, t, n_m, n_i, st = self.fe.process(frame.image)

            # Metres travelled in THIS step, from the camera rather than the
            # GPS. Since ||t|| == 1 the metric step is just t * scale.
            scale = self._metric_step(frame)

            # CAUSAL pairing with the GPS
            fix = self.gps.latest_before(frame.t_ns)
            if fix is None:
                self.metrics.no_gps += 1
            else:
                self.metrics.gps_age_ms.append((frame.t_ns - fix[0]) / 1e6)

            # End-to-end latency: how far behind the capture instant we are.
            if frame.unix_ns is not None:
                # Live: phone capture time against the PC clock, so it
                # includes whatever offset separates the two clocks.
                now_ns = time.time_ns()
                self.metrics.e2e_ms.append((now_ns - frame.unix_ns) / 1e6)
                self.metrics.pc_ms.append((now_ns - frame.arrival_ns) / 1e6)
            elif self.strict:
                elapsed_wall = time.perf_counter() - t0_wall
                elapsed_data = (frame.t_ns - t0_data_ns) / 1e9
                self.metrics.e2e_ms.append(max(elapsed_wall - elapsed_data, 0.0) * 1000)
            else:
                self.metrics.e2e_ms.append(st.total * 1000)

            self.metrics.stages.append(st)
            self.metrics.processed += 1
            # Up to the last processed frame: a live session can end on an
            # idle timeout, and that wait is not processing time.
            self.metrics.t_end = time.perf_counter()

            if self.on_result is not None:
                self.on_result(frame, R, t, n_m, n_i, fix, scale)

        if self.metrics.processed == 0:
            self.metrics.t_end = time.perf_counter()
        self.metrics.dropped = self._dropped[0]
        self.metrics.rejected_poses = self.fe.n_rejected
        self._stop.set()
        cap_thread.join(timeout=2.0)
        if self.scale is not None:
            self.scale.stop()
            self.metrics.depth_ms = self.scale.times_ms
        return self.metrics

    def _metric_step(self, frame: Frame) -> Optional[float]:
        """
        Metres since the previous processed frame, or None if there is no
        speed yet.

        Uses the REAL dt between processed frames, not 1/30 s: with drops the
        steps are not uniform, and assuming they are would shorten the
        trajectory exactly when the system is most loaded.

        Speed is HELD between model updates (~3 Hz). That is an explicit
        constant-speed approximation over ~0.33 s, fine because vehicle speed
        changes slowly. It would not be fine for direction, which is why
        direction still comes from the camera at 30 Hz.
        """
        if self.scale is None:
            return None

        self.scale.submit(frame.t_ns / 1e9, frame.image, *self.fe.last_features)

        v = self.scale.velocity
        t_prev, self._t_prev_ns = self._t_prev_ns, frame.t_ns
        if v is None:
            # Start-up: the estimator needs triangulation baseline before it
            # can report a first speed.
            self.metrics.no_scale += 1
            return None
        if t_prev is None:
            return None
        return v * (frame.t_ns - t_prev) / 1e9

    def stop(self):
        self._stop.set()
        if self.scale is not None:
            self.scale.stop()


def simulate_drops(frame_times_ns, proc_seconds) -> dict:
    """
    Deterministic mode: given how long each frame took, work out which ones a
    single-slot queue would have dropped.

    Simulates a virtual clock: if other frames arrived while one was being
    processed, only the most recent is served and the rest are dropped.
    """
    t = np.asarray(frame_times_ns, dtype=np.float64) / 1e9
    t = t - t[0]
    d = np.asarray(proc_seconds, dtype=np.float64)

    clock = 0.0
    processed, dropped, latencies = 0, 0, []
    i = 0
    n = len(t)
    while i < n:
        if clock <= t[i]:
            clock = t[i]
        else:
            # Its moment has passed: skip to the ones already arrived and keep
            # only the most recent.
            j = i
            while j + 1 < n and t[j + 1] <= clock:
                j += 1
            dropped += (j - i)
            i = j
        start = max(clock, t[i])
        clock = start + d[min(i, len(d) - 1)]
        latencies.append((clock - t[i]) * 1000)
        processed += 1
        i += 1

    return {
        "frames_procesados": processed,
        "frames_descartados": dropped,
        "tasa_descarte_%": 100.0 * dropped / max(processed + dropped, 1),
        "e2e_ms_p50": float(np.percentile(latencies, 50)),
        "e2e_ms_p95": float(np.percentile(latencies, 95)),
        "e2e_ms_max": float(np.max(latencies)),
    }
