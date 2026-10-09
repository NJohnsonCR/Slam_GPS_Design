"""
Phone simulator: serves a recording made by the app with the app's own
protocol, so the live client and the interface can be tested without the
phone.

    TCP 5000  HELLO line, then every frame as a 20-byte header plus its JPEG
    UDP 5001  SUBSCRIBE / START / STOP in; SUBSCRIBED, STATE, GPS and IMU lines
              out (IMU as the app will send it: each gyro_accel.csv row)

It behaves like the phone where it matters to the PC: the camera runs in real
time from the first connection, a new connection replaces the previous one,
frames are dropped (not queued) when the PC falls behind, and fixes and IMU
samples arrive when the video reaches their timestamp. --no-imu leaves the
IMU out, as the current app does. Frames carry the recording's boot
clock, so they pair with its location.csv; the Unix capture time is the
current one, so the measured latency is that of the PC.

Usage:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.phone_simulator \\
        mobile_data/2026_09_30_14_18_36 --start 380 --duration 120

    # in another terminal
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_live 127.0.0.1
"""

import argparse
import os
import socket
import struct
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

from realtime.sources import HELLO_S23

_HEADER = struct.Struct(">IQQ")
SKIP_START = 6          # frames the encoder writes before catching up


class PhoneSimulator:
    def __init__(self, session, start_s=0.0, duration_s=None, quality=80,
                 hello=HELLO_S23, video_port=5000, gps_port=5001, imu=True):
        self.session = session
        self.quality = quality
        self.hello = hello
        self.video_port, self.gps_port = video_port, gps_port

        ft = pd.read_csv(os.path.join(session, "frame_timestamps.txt"))
        t = ft.iloc[:, 0].to_numpy(np.int64)
        self.k0 = SKIP_START + int(np.searchsorted(t[SKIP_START:], t[SKIP_START] + int(start_s * 1e9)))
        k1 = len(t) if duration_s is None else int(np.searchsorted(t, t[self.k0] + int(duration_s * 1e9)))
        self.frame_t = t[:k1]
        # The app sends each fix as the same row it writes to location.csv.
        with open(os.path.join(session, "location.csv")) as f:
            self.fix_rows = [line.strip() for line in f.readlines()[1:] if line.strip()]
        self.fix_t = np.array([int(row.split(",")[0]) for row in self.fix_rows], np.int64)
        self.imu_rows = []
        imu_path = os.path.join(session, "gyro_accel.csv")
        if imu and os.path.exists(imu_path):
            with open(imu_path) as f:
                self.imu_rows = [line.strip() for line in f.readlines()[1:] if line.strip()]
        self.imu_t = np.array([int(row.split(",")[0]) for row in self.imu_rows], np.int64)

        self.n_sent = self.n_dropped = self.n_fixes = self.n_imu = 0
        self.done = threading.Event()
        self._client = None
        self._subscriber = None
        self._recording = None
        self._latest = None                 # (header + JPEG) waiting to be sent
        self._cv = threading.Condition()

    # ------------------------------------------------------------- camera

    def _camera(self, cap, udp):
        """Plays the recording in real time; each frame replaces an unsent one."""
        t0_wall, t0_data = time.perf_counter(), int(self.frame_t[self.k0])
        i_fix = int(np.searchsorted(self.fix_t, t0_data))
        i_imu = int(np.searchsorted(self.imu_t, t0_data))
        for k in range(self.k0, len(self.frame_t)):
            ok, img = cap.read()
            if not ok:
                break
            t_ns = int(self.frame_t[k])
            wait = t0_wall + (t_ns - t0_data) / 1e9 - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            data = _HEADER.pack(len(jpg), t_ns, time.time_ns()) + jpg.tobytes()
            with self._cv:
                if self._latest is not None:
                    self.n_dropped += 1
                self._latest = data
                self._cv.notify_all()
            while i_fix < len(self.fix_t) and self.fix_t[i_fix] <= t_ns:
                if self._subscriber is not None:
                    udp.sendto(f"GPS,{self.fix_rows[i_fix]}".encode("ascii"), self._subscriber)
                    self.n_fixes += 1
                i_fix += 1
            while i_imu < len(self.imu_t) and self.imu_t[i_imu] <= t_ns:
                if self._subscriber is not None:
                    udp.sendto(f"IMU,{self.imu_rows[i_imu]}".encode("ascii"), self._subscriber)
                    self.n_imu += 1
                i_imu += 1
        self.done.set()
        with self._cv:
            self._cv.notify_all()

    def _sender(self):
        """Sends the newest frame to the connected PC, like the app's streamer."""
        while not self.done.is_set():
            with self._cv:
                self._cv.wait_for(lambda: self._latest is not None or self.done.is_set())
                data, self._latest = self._latest, None
                client = self._client
            if data is None or client is None:
                continue
            try:
                client.sendall(data)
                self.n_sent += 1
            except OSError:
                with self._cv:
                    if self._client is client:
                        self._client = None
        if self._client is not None:
            self._client.close()

    # ----------------------------------------------------------- network

    def _udp_loop(self, udp):
        while not self.done.is_set():
            try:
                data, addr = udp.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                return
            word = data.decode("ascii", errors="replace").strip()
            if word == "SUBSCRIBE":
                self._subscriber = addr
                udp.sendto(b"SUBSCRIBED", addr)
            elif word == "START" and self._recording is None:
                self._recording = time.strftime("simulado_%Y_%m_%d_%H_%M_%S")
            elif word == "STOP":
                self._recording = None
            state = (f"STATE,RECORDING,{self._recording}" if self._recording
                     else "STATE,IDLE")
            udp.sendto(state.encode("ascii"), addr)

    def run(self):
        print(f"  Saltando hasta el frame {self.k0}...", flush=True)
        cap = cv2.VideoCapture(os.path.join(self.session, "movie.mp4"))
        for _ in range(self.k0):
            cap.grab()

        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.bind(("0.0.0.0", self.gps_port))
        udp.settimeout(0.2)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.video_port))
        server.listen(1)
        server.settimeout(0.5)
        threading.Thread(target=self._udp_loop, args=(udp,), daemon=True).start()
        threading.Thread(target=self._sender, daemon=True).start()
        seconds = (self.frame_t[-1] - self.frame_t[self.k0]) / 1e9
        print(f"  Listo: {seconds:.0f} s de video en los puertos {self.video_port} (video) "
              f"y {self.gps_port} (GPS). La cámara arranca con la primera conexión.", flush=True)

        camera = None
        try:
            while not self.done.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                conn.sendall((self.hello + "\n").encode("ascii"))
                with self._cv:
                    previous, self._client = self._client, conn
                if previous is not None:
                    previous.close()
                print(f"  PC conectada: {addr[0]}", flush=True)
                if camera is None:
                    camera = threading.Thread(target=self._camera, args=(cap, udp), daemon=True)
                    camera.start()
        finally:
            self.done.set()
            server.close()
            udp.close()
            cap.release()
        print(f"  Fin del video: {self.n_sent} frames enviados, {self.n_dropped} descartados, "
              f"{self.n_fixes} fixes de GPS, {self.n_imu} muestras de IMU.")


def main():
    ap = argparse.ArgumentParser(description="Simulador del teléfono: sirve una grabación con "
                                             "el protocolo de la app")
    ap.add_argument("session", help="Carpeta de una grabación de la app (mobile_data/...)")
    ap.add_argument("--start", type=float, default=0.0, help="Segundo de la grabación desde el que se sirve")
    ap.add_argument("--duration", type=float, default=None, help="Segundos a servir; sin esto, hasta el final")
    ap.add_argument("--quality", type=int, default=80, help="Calidad JPEG, como en los ajustes de la app")
    ap.add_argument("--hello", default=HELLO_S23, help="Línea HELLO con los intrínsecos de la cámara")
    ap.add_argument("--video-port", type=int, default=5000)
    ap.add_argument("--gps-port", type=int, default=5001)
    ap.add_argument("--no-imu", action="store_true",
                    help="No mandar la IMU, como la app actual")
    args = ap.parse_args()

    print("=" * 78)
    print("SIMULADOR DEL TELÉFONO")
    print("=" * 78)
    sim = PhoneSimulator(args.session, args.start, args.duration, args.quality, args.hello,
                         args.video_port, args.gps_port, imu=not args.no_imu)
    try:
        sim.run()
    except KeyboardInterrupt:
        print("\n  Simulador detenido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
