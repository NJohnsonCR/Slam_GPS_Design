"""
One run of the real-time pipeline, live from the phone or from a recording.

The console (run_live.py) and the window (live_window.py) drive the same
objects, in this order:

    connect()   reach the phone, or read the recording
    prepare()   camera matrix, depth model and warm-up
    run()       blocks until the source ends or stop() is called
    summary()   metrics of the finished run
    save(res)   the session folder: frames, fixes, metrics and figure
    close()     release the network

stop(), status() and latest_frame can be used from any other thread: the
window runs the session on a worker thread and polls them.

A replay session runs in real time, like the strict mode of run_replay.py, so
the video plays at its normal speed. The deterministic mode, the one used for
reported numbers, stays in run_replay.py.
"""

import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_LMS_RL = os.path.abspath(os.path.join(_THIS, ".."))
_ROOT = os.path.abspath(os.path.join(_LMS_RL, "..", ".."))
for _p in (_ROOT, _LMS_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realtime.sources import (HELLO_S23, GpsBuffer, LiveFrameSource, LiveGpsSource,
                              ReplayFrameSource, ReplayGpsSource, load_mobile_session,
                              parse_hello, read_hello)
from realtime.pipeline import RealtimePipeline, VisualFrontEnd
from realtime.run_replay import (TrackRecorder, build_scale_worker, mobile_camera_matrix,
                                 plot_trajectory, save_frames_csv)

# Same header as the phone's location.csv, so both files read the same way.
LOCATION_HEADER = ("Timestamp[nanosecond],latitude[degrees],longitude[degrees],"
                   "altitude[meters],speed[meters/second],Unix time[nanosecond]")
SKIP_START = 6          # frames the app's encoder writes before catching up


class SessionError(RuntimeError):
    """A session that cannot go on; the message is for the user."""


def warm_up(K, width, height, est=None):
    """
    Run the models once on synthetic frames before the first real one.

    The first ORB call pays a one-off initialisation (~200 ms), and so does
    the first depth inference (~0.3 s, several seconds with cold caches).
    Paid here, it does not reach the first frames as latency.
    """
    rng = np.random.default_rng(0)
    noise = cv2.GaussianBlur(
        rng.integers(0, 256, (height + 8, width + 8), dtype=np.uint8), (0, 0), 1.5)
    fe = VisualFrontEnd(K)
    # Two shifted copies, so matching and pose estimation run as well.
    for dy, dx in ((0, 0), (4, 6)):
        img = cv2.cvtColor(noise[dy:dy + height, dx:dx + width], cv2.COLOR_GRAY2BGR)
        fe.process(img)
    if est is not None:
        est.depth_map(img)


class PipelineSession:
    """What live and replay sessions share: pipeline, recording of results, status."""

    kind = ""               # prefix of the session folder

    def __init__(self, mask_bottom=0.25, scale=True, scale_hz=3.0, duration=None,
                 out="resultados/realtime"):
        self.mask_bottom = mask_bottom
        self.use_scale = scale
        self.scale_hz = scale_hz
        self.duration = duration
        self.out = out
        self.state = "sin iniciar"
        self.K = None
        self.est = self.scaler = None
        self.source = self.pipe = None
        self.recorder = TrackRecorder()
        self.latest_frame = None        # last processed Frame, for the window
        self.metrics = None
        self.out_dir = None
        self.stamp = time.strftime("%Y%m%d_%H%M%S")
        self.cancelled = threading.Event()
        self._t0 = None
        self._prev = None

    # ------------------------------------------------------------ building

    def _build_pipeline(self, K, width, height, gps_buffer):
        self.K = K
        if self.use_scale:
            self.state = "cargando el modelo de profundidad"
            self.est, self.scaler = build_scale_worker(K, self.scale_hz, threaded=True)
        self.state = "precalentando"
        warm_up(K, width, height, self.est)
        self.pipe = RealtimePipeline(VisualFrontEnd(K, mask_bottom=self.mask_bottom),
                                     gps_buffer, strict=True, on_result=self._on_result,
                                     scale_worker=self.scaler)

    def _on_result(self, frame, *result):
        self.recorder(frame, *result)
        self.latest_frame = frame

    # ------------------------------------------------------------- running

    def stop(self):
        """End the run from any thread, also before it starts."""
        self.cancelled.set()

    def _run_pipeline(self, t0_data_ns=None):
        if self.cancelled.is_set():
            raise SessionError("Sesión cancelada antes de empezar.")
        timer = threading.Timer(self.duration, self.stop) if self.duration else None
        self.state = "en marcha"
        self._t0 = time.monotonic()
        if timer is not None:
            timer.start()
        try:
            self.metrics = self.pipe.run(self.source, t0_data_ns)
        finally:
            if timer is not None:
                timer.cancel()
            self.state = "detenida"

    def _counters(self):
        """Frames delivered by the source and not sent by the phone, so far."""
        return self.pipe.metrics.processed + self.pipe.n_dropped, 0

    def status(self) -> dict:
        """
        Snapshot for a status line. Rates are per second since the previous
        call, so one caller should poll it about once a second.
        """
        st = {"state": self.state}
        if self.pipe is None or self._t0 is None:
            return st
        m = self.pipe.metrics
        received, missing = self._counters()
        now = (time.monotonic(), received, missing, m.processed)
        prev = self._prev or (self._t0, 0, 0, 0)
        self._prev = now
        span = max(now[0] - prev[0], 1e-3)
        processed = now[3] - prev[3]
        st.update({
            "elapsed_s": now[0] - self._t0,
            "video_fps": (now[1] - prev[1]) / span,
            "missing": now[2] - prev[2],
            "processed_fps": processed / span,
            "latency_ms": float(np.median(m.e2e_ms[-processed:])) if processed else None,
            # Live only: arrival -> result, on the PC clock alone.
            "pc_latency_ms": (float(np.median(m.pc_ms[-processed:]))
                              if processed and m.pc_ms else None),
            "camera_speed": self.scaler.velocity if self.scaler is not None else None,
            "stopped": bool(self.est.stopped) if self.est is not None else None,
        })
        return st

    # ------------------------------------------------------------- results

    def _summary(self) -> dict:
        res = self.metrics.summary()
        if self.scaler is not None:
            res["escala"] = {
                "enviadas": self.scaler.n_submitted,
                "descartadas_cola": self.scaler.n_dropped[0],
                "aceptadas": self.est.n_ok,
                "falladas": self.est.n_fail,
                "fuera_de_rango": self.est.n_rejected,
                "detenido": self.est.n_stopped,
            }
        return res

    def save(self, res):
        """frames.csv, gps.csv, metrics.json and the trajectory figure."""
        self.state = "guardando"
        gps_rows = self._gps_rows()
        self.out_dir = os.path.join(self.out, f"{self.kind}_{self.stamp}")
        os.makedirs(self.out_dir, exist_ok=True)
        save_frames_csv(os.path.join(self.out_dir, "frames.csv"), self.recorder.track,
                        self.metrics)
        with open(os.path.join(self.out_dir, "gps.csv"), "w") as f:
            f.write(LOCATION_HEADER + "\n")
            for row in gps_rows:
                f.write(row + "\n")
        with open(os.path.join(self.out_dir, "metrics.json"), "w") as f:
            json.dump(res, f, indent=2)
        plot_trajectory(self.recorder.track, self.out_dir, self.kind,
                        metric=self.scaler is not None)
        self.state = "terminada"


class LiveSession(PipelineSession):
    """The phone over the network: video by TCP, GPS and control by UDP."""

    kind = "live"
    IDLE_REPLY_S = 6.0          # without a SUBSCRIBED for this long, UDP is silent

    def __init__(self, ip, video_port=5000, gps_port=5001, record=False, **kw):
        super().__init__(**kw)
        self.ip = ip
        self.video_port, self.gps_port = video_port, gps_port
        self.record = record
        self.hello = self.gps = self.gps_buf = None
        self.gps_ok = False
        self.folder = None          # recording folder on the phone
        self.recording_stopped = None

    def connect(self):
        """Read the camera parameters and start the GPS link. Raises OSError."""
        self.state = "conectando con el teléfono"
        self.hello = read_hello(self.ip, self.video_port)
        self.gps_buf = GpsBuffer()
        self.gps = LiveGpsSource(self.ip, self.gps_buf, self.gps_port)
        self.gps.start()
        self.gps_ok = self.gps.wait_reply()

    def prepare(self):
        h = self.hello
        self._build_pipeline(h.camera_matrix(), h.width, h.height, self.gps_buf)
        # The video opens only now: frames sent while the model loaded would
        # reach the pipeline stale.
        self.source = LiveFrameSource(self.ip, h, self.video_port)

    def run(self):
        recording = False
        if self.record:
            if not self.gps.command("START"):
                raise SessionError("No se pudo iniciar la grabación en el teléfono. "
                                   "¿Está la app en primer plano, en la pantalla de video?")
            recording = True
            self.folder = self.gps.recording_folder
        try:
            self._run_pipeline()
        finally:
            if recording:
                self.recording_stopped = self.gps.command("STOP")

    def stop(self):
        super().stop()
        if self.source is not None:
            self.source.stop()

    def close(self):
        if self.gps is not None:
            self.gps.stop()

    @property
    def end_reason(self):
        return self.source.end_reason if self.source is not None else None

    def _counters(self):
        return self.source.n_received, self.source.n_missing

    def status(self) -> dict:
        st = super().status()
        if self.gps is None:
            return st
        fixes = self.gps.fixes
        st["gps_fixes"] = len(fixes)
        if fixes and self.source is not None and self.source.last_t_ns is not None:
            st["fix_age_s"] = (self.source.last_t_ns - fixes[-1][0]) / 1e9
            st["gps_speed"] = fixes[-1][4]
        st["recording"] = self.gps.recording_folder
        st["udp_silent"] = (self.gps.last_reply is not None
                            and time.monotonic() - self.gps.last_reply > self.IDLE_REPLY_S)
        return st

    def summary(self) -> dict:
        """Metrics of the finished run, with the phone's side of the link."""
        res = self._summary()
        src, gps = self.source, self.gps
        transit = src.transit_ms or [float("nan")]
        res["en_vivo"] = {
            "telefono": self.ip,
            "camara": self.hello._asdict(),
            "carpeta_telefono": self.folder,
            "fin": src.end_reason,
            "frames_recibidos": src.n_received,
            "frames_no_enviados": src.n_missing,
            "jpeg_invalidos": src.n_bad,
            "captura_llegada_ms_p50": float(np.percentile(transit, 50)),
            "captura_llegada_ms_p95": float(np.percentile(transit, 95)),
            "fixes_gps": len(gps.fixes),
            "udp_invalidos": gps.n_bad,
            "utm_epsg": gps.epsg,
            "utm_origen": None if gps.origin is None else gps.origin.tolist(),
            "mascara_inferior": self.mask_bottom,
        }
        return res

    def _gps_rows(self):
        return [",".join(str(v) for v in fix) for fix in self.gps.fixes]


class ReplaySession(PipelineSession):
    """A recording made by the app, played in real time through the pipeline."""

    kind = "replay"

    def __init__(self, recording, start_s=0.0, **kw):
        super().__init__(**kw)
        self.recording = recording
        self.start_s = start_s
        self.gps_src = None

    def connect(self):
        """Read the recording's timestamps and fixes. Raises OSError."""
        self.state = "leyendo la grabación"
        video = os.path.join(self.recording, "movie.mp4")
        if not os.path.exists(video):
            raise OSError(f"no hay movie.mp4 en {self.recording}")
        frame_t, fixes = load_mobile_session(self.recording)
        skip = max(SKIP_START, int(np.searchsorted(
            frame_t, frame_t[SKIP_START] + int(self.start_s * 1e9))))
        self.frame_t, self.skip = frame_t[skip:], skip
        n = len(self.frame_t)
        if self.duration is not None:
            n = int(np.searchsorted(self.frame_t, self.frame_t[0] + int(self.duration * 1e9)))
        self.n_frames = max(n, 1)
        t_from, t_to = int(self.frame_t[0]), int(self.frame_t[self.n_frames - 1])
        self.fixes = [f for f in fixes if t_from - 2_000_000_000 <= f[0] <= t_to]

        loc = pd.read_csv(os.path.join(self.recording, "location.csv"))
        self._fix_t = loc.iloc[:, 0].to_numpy(np.int64)
        self._fix_speed = loc.iloc[:, 4].to_numpy(float)
        with open(os.path.join(self.recording, "location.csv")) as f:
            rows = [r.strip() for r in f.readlines()[1:] if r.strip()]
        self._fix_rows = [r for r, t in zip(rows, self._fix_t)
                          if t_from - 2_000_000_000 <= t <= t_to]

        cap = cv2.VideoCapture(video)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def prepare(self):
        # Recordings of the app are 1280x720 and keep no optical centre: the
        # S23's HELLO gives it. Others fall back to run_replay's defaults.
        if (self.width, self.height) == (1280, 720):
            K = parse_hello(HELLO_S23).camera_matrix()
        else:
            K = mobile_camera_matrix(self.width, self.height)
        gps_buf = GpsBuffer()
        self._build_pipeline(K, self.width, self.height, gps_buf)
        self.state = f"buscando el segundo {self.start_s:.0f} de la grabación"
        self.source = ReplayFrameSource(os.path.join(self.recording, "movie.mp4"),
                                        self.frame_t, strict=True,
                                        max_frames=self.n_frames, skip_start=self.skip)
        self.source.__enter__()
        self.gps_src = ReplayGpsSource(self.fixes, gps_buf, strict=True)

    def run(self):
        t0_data = int(self.frame_t[0])
        self.gps_src.start(time.perf_counter(), t0_data)
        try:
            self._run_pipeline(t0_data)
        finally:
            self.gps_src.stop()

    def stop(self):
        super().stop()
        if self.pipe is not None:
            self.pipe.stop()

    def close(self):
        if self.source is not None:
            self.source.__exit__(None, None, None)

    @property
    def end_reason(self):
        return "sesión detenida" if self.cancelled.is_set() else "fin de la grabación"

    def status(self) -> dict:
        st = super().status()
        frame = self.latest_frame
        if frame is not None:
            i = int(np.searchsorted(self._fix_t, frame.t_ns, side="right")) - 1
            if i >= 0:
                st["gps_fixes"] = len(self.pipe.gps)
                st["fix_age_s"] = (frame.t_ns - self._fix_t[i]) / 1e9
                st["gps_speed"] = float(self._fix_speed[i])
            st["recording_s"] = (frame.t_ns - self.frame_t[0]) / 1e9 + self.start_s
        return st

    def summary(self) -> dict:
        res = self._summary()
        res["replay"] = {
            "grabacion": self.recording,
            "desde_s": self.start_s,
            "frames": self.n_frames,
            "mascara_inferior": self.mask_bottom,
            "camara": {"fx": self.K[0, 0], "fy": self.K[1, 1],
                       "cx": self.K[0, 2], "cy": self.K[1, 2]},
        }
        return res

    def _gps_rows(self):
        return self._fix_rows
