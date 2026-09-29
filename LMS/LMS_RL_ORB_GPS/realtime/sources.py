"""
Data sources for the real-time pipeline.

Sources are interchangeable: they all deliver the same thing, so the rest of
the system never knows where the data came from. The data comes either from a
recorded session (replay) or live from the Android app over the network.

    FrameSource  ->  Frame(t_ns, index, image)
    GpsSource    ->  pushes (t_ns, position) into a GpsBuffer

Replay runs in two modes, and the distinction matters for every measurement:

    strict         sleeps until each frame's real timestamp and drops for real.
                   Reproduces street behaviour, but is NOT deterministic.
    deterministic  delivers every frame without sleeping. Drops are computed
                   afterwards from the measured times. Reproducible, so it is
                   the mode used for development and for reported numbers.

Live sources always behave like strict replay. The phone is the server:

    TCP 5000  "HELLO,<w>,<h>,<fx>,<fy>,<cx>,<cy>,<fps>\\n", then per frame a
              20-byte big-endian header (uint32 JPEG length, uint64 capture
              time on the phone's boot clock in ns, uint64 capture time in
              Unix ns) followed by the JPEG.
    UDP 5001  the PC sends SUBSCRIBE every ~2 s, and START/STOP from the same
              socket. The phone answers SUBSCRIBED, STATE,... and one line per
              fix: GPS,<t_ns>,<lat>,<lon>,<alt>,<speed>,<unix_ns>.

Frames and fixes are stamped with the same boot clock, so they align directly.
"""

import collections
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
from pyproj import Transformer


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
    unix_ns: Optional[int] = None       # live only: capture time, phone wall clock
    arrival_ns: Optional[int] = None    # live only: arrival time, PC wall clock


def _crop_bottom(img: np.ndarray, fraction: float) -> np.ndarray:
    # The dashboard sits in the lower part of the frame and produces
    # correspondences that claim the vehicle did not move.
    if fraction > 0:
        img = img[: int(img.shape[0] * (1.0 - fraction))]
    return img


class ReplayFrameSource:
    """
    Replays a recorded video honouring the real time between frames.

    skip_start discards that many frames at the start of the video, which the
    phone's encoder writes before it catches up. timestamps_ns must already
    leave them out.
    """

    def __init__(self, video_path: str, timestamps_ns, crop_bottom: float = 0.0,
                 strict: bool = True, max_frames: Optional[int] = None,
                 skip_start: int = 0):
        self.video_path = video_path
        self.timestamps = np.asarray(timestamps_ns, dtype=np.int64)
        self.crop_bottom = crop_bottom      # bottom fraction to cut: car dashboard
        self.strict = strict
        self.max_frames = max_frames
        self.skip_start = skip_start
        self._cap = None

    def __enter__(self):
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self.video_path}")
        for _ in range(self.skip_start):
            self._cap.grab()
        return self

    def __exit__(self, *exc):
        if self._cap is not None:
            self._cap.release()

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
                        image=_crop_bottom(img, self.crop_bottom))


class ReplayGpsSource:
    """
    Delivers GPS fixes at their real rate.

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


# -------------------------------------------------------------------- live

_HEADER = struct.Struct(">IQQ")     # JPEG length, boot-clock ns, Unix ns


class Hello(NamedTuple):
    """Camera parameters the phone sends on connect, in transmitted pixels."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    fps: int

    def camera_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)


def parse_hello(line: str) -> Optional[Hello]:
    """Parse the HELLO line. None if malformed."""
    parts = line.strip().split(",")
    if len(parts) != 8 or parts[0] != "HELLO":
        return None
    try:
        return Hello(int(parts[1]), int(parts[2]), float(parts[3]),
                     float(parts[4]), float(parts[5]), float(parts[6]),
                     int(parts[7]))
    except ValueError:
        return None


def read_hello(ip: str, port: int = 5000, timeout: float = 5.0) -> Hello:
    """
    Connect once just to read the camera parameters, then hang up.

    The stream is opened later, when the pipeline starts reading: whatever the
    phone sent in between (loading the depth model takes seconds) would reach
    the pipeline stale.
    """
    with socket.create_connection((ip, port), timeout=timeout) as sock, \
            sock.makefile("rb") as stream:
        line = stream.readline(256).decode("ascii", errors="replace")
    hello = parse_hello(line)
    if hello is None:
        raise ConnectionError(f"saludo inválido del teléfono: {line!r}")
    return hello


class LiveFrameSource:
    """
    Video streamed by the phone over TCP.

    JPEGs are decoded here, on the capture thread, outside the per-frame
    budget. The phone never queues frames: when the network is busy it skips
    them, and those gaps are counted from the timestamps.
    """

    IDLE_TIMEOUT_S = 5.0
    MAX_JPEG_BYTES = 20_000_000     # anything larger means a corrupt header

    def __init__(self, ip: str, hello: Hello, port: int = 5000,
                 crop_bottom: float = 0.0):
        self.ip = ip
        self.port = port
        self.hello = hello
        self.crop_bottom = crop_bottom
        self.n_received = 0
        self.n_missing = 0                  # captured by the phone, never sent
        self.n_bad = 0                      # JPEGs that failed to decode
        self.transit_ms: List[float] = []   # capture -> arrival
        self.last_t_ns = None
        self.end_reason = None
        self._sock = None
        self._stopped = False

    def frames(self):
        """Frame generator. Ends on stop(), on disconnection or after 5 s idle."""
        try:
            sock = socket.create_connection((self.ip, self.port),
                                            timeout=self.IDLE_TIMEOUT_S)
        except OSError as e:
            self.end_reason = f"no se pudo conectar al video ({e})"
            return
        self._sock = sock
        interval_ns = 1e9 / self.hello.fps
        try:
            with sock.makefile("rb") as stream:
                line = stream.readline(256).decode("ascii", errors="replace")
                if parse_hello(line) != self.hello:
                    self.end_reason = "la cámara cambió entre conexiones"
                    return
                while not self._stopped:
                    head = stream.read(_HEADER.size)
                    if len(head) < _HEADER.size:
                        break
                    length, t_ns, unix_ns = _HEADER.unpack(head)
                    if length > self.MAX_JPEG_BYTES:
                        self.end_reason = "cabecera de frame inválida"
                        return
                    jpeg = stream.read(length)
                    if len(jpeg) < length:
                        break
                    arrival_ns = time.time_ns()

                    if self.last_t_ns is not None:
                        gap = round((t_ns - self.last_t_ns) / interval_ns)
                        self.n_missing += max(gap - 1, 0)
                    self.last_t_ns = t_ns
                    self.transit_ms.append((arrival_ns - unix_ns) / 1e6)

                    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        self.n_bad += 1
                        continue
                    self.n_received += 1
                    yield Frame(t_ns=t_ns, index=self.n_received - 1,
                                image=_crop_bottom(img, self.crop_bottom),
                                unix_ns=unix_ns, arrival_ns=arrival_ns)
            self.end_reason = ("sesión detenida" if self._stopped
                               else "el teléfono cerró la conexión de video")
        except socket.timeout:
            self.end_reason = "no llegan frames hace 5 s"
        except OSError:
            # stop() shuts the socket down to unblock a pending read.
            self.end_reason = ("sesión detenida" if self._stopped
                               else "se cortó la conexión de video")
        finally:
            sock.close()

    def stop(self):
        """End the stream from another thread (Ctrl+C, time limit)."""
        self._stopped = True
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class LiveGpsSource:
    """
    GPS fixes and recording control from the phone over UDP.

    One socket does everything: the phone sends to the address that last
    subscribed, so commands must leave from that same socket.
    """

    SUBSCRIBE_EVERY_S = 2.0

    def __init__(self, ip: str, buffer: GpsBuffer, port: int = 5001):
        self.phone = (ip, port)
        self.buffer = buffer
        self.fixes = []             # (t_ns, lat, lon, alt, speed, unix_ns)
        self.n_bad = 0              # unrecognised datagrams
        self.state = None           # last "STATE,..." line
        self.last_reply = None      # monotonic time of the last SUBSCRIBED
        self.epsg = None            # UTM zone, fixed by the first fix
        self.origin = None          # UTM of the first fix
        self._to_utm = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.5)
        self._state_changed = threading.Condition()
        self._stop = threading.Event()
        self._thread = None

    @property
    def recording_folder(self) -> Optional[str]:
        """Folder the phone is recording into, or None when idle."""
        state = self.state or ""
        if state.startswith("STATE,RECORDING,"):
            return state.split(",", 2)[2]
        return None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait_reply(self, timeout: float = 3.0) -> bool:
        """True once the phone has answered a SUBSCRIBE."""
        deadline = time.monotonic() + timeout
        while self.last_reply is None and time.monotonic() < deadline:
            time.sleep(0.05)
        return self.last_reply is not None

    def command(self, word: str, timeout: float = 5.0) -> bool:
        """
        Send START or STOP until the phone reports the matching state.

        UDP can lose the command or the reply. Repeating is harmless: the
        phone only acts when its state has to change.
        """
        wanted = "STATE,RECORDING" if word == "START" else "STATE,IDLE"
        deadline = time.monotonic() + timeout
        with self._state_changed:
            while time.monotonic() < deadline:
                self._send(word)
                self._state_changed.wait(0.5)
                if self.state is not None and self.state.startswith(wanted):
                    return True
        return False

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sock.close()

    def _send(self, text: str):
        try:
            self._sock.sendto(text.encode("ascii"), self.phone)
        except OSError:
            pass        # network down: the next SUBSCRIBE tries again

    def _run(self):
        last_subscribe = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_subscribe >= self.SUBSCRIBE_EVERY_S:
                self._send("SUBSCRIBE")
                last_subscribe = now
            try:
                data, _ = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                return
            self._handle(data.decode("ascii", errors="replace").strip())

    def _handle(self, msg: str):
        if msg == "SUBSCRIBED":
            self.last_reply = time.monotonic()
        elif msg.startswith("STATE,"):
            with self._state_changed:
                self.state = msg
                self._state_changed.notify_all()
        elif msg.startswith("GPS,"):
            parts = msg.split(",")
            try:
                if len(parts) != 7:
                    raise ValueError(msg)
                t_ns, unix_ns = int(parts[1]), int(parts[6])
                lat, lon, alt, speed = (float(p) for p in parts[2:6])
            except ValueError:
                self.n_bad += 1
                return
            self.fixes.append((t_ns, lat, lon, alt, speed, unix_ns))
            self.buffer.push(t_ns, self._position(lat, lon, alt))
        else:
            self.n_bad += 1

    def _position(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """Metres relative to the first fix."""
        # The zone is fixed by the first fix: switching zones mid-route (the
        # 84 W boundary crosses Costa Rica) would make positions jump.
        if self._to_utm is None:
            zone = int((lon + 180) / 6) + 1
            self.epsg = (32600 if lat >= 0 else 32700) + zone
            self._to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}",
                                                always_xy=True)
        x, y = self._to_utm.transform(lon, lat)
        utm = np.array([x, y, alt])
        if self.origin is None:
            self.origin = utm
        return utm - self.origin


# -------------------------------------------------------------------- load

def load_mobile_session(session_dir: str, latlon_to_utm):
    """
    Read a session recorded by the phone app (mobile_data/ format).

    Returns (frame_timestamps_ns, gps_fixes), both on the phone's boot clock:
    the first column of each file. The 'Unix time' column of the camera files
    is when the frame reached the app, tens of milliseconds after capture and
    with jitter, so it must not be used for alignment.
    """
    import os

    import pandas as pd

    ft = pd.read_csv(os.path.join(session_dir, "frame_timestamps.txt"))
    frame_t = ft["Frame timestamp[nanosec]"].values.astype(np.int64)

    loc = pd.read_csv(os.path.join(session_dir, "location.csv"))
    origin = None
    fixes = []
    for _, r in loc.iterrows():
        utm = latlon_to_utm(r["latitude[degrees]"], r["longitude[degrees]"],
                            r["altitude[meters]"])
        if origin is None:
            origin = utm
        fixes.append((int(r["Timestamp[nanosecond]"]), utm - origin))

    return frame_t, fixes
