"""
Fuentes de datos para el pipeline en tiempo real.

IDEA CENTRAL
------------
La fuente es un módulo intercambiable. Todas entregan lo mismo:

    FrameSource  ->  (timestamp_ns, frame)
    GpsSource    ->  empuja (timestamp_ns, posición) a un buffer

El resto del sistema NO sabe de dónde salieron los datos. Hoy vienen de
archivos grabados; mañana vendrán de la app Android por UDP. Cambiar de una a
otra es escribir una clase nueva, no rehacer el pipeline.

MODOS DE REPLAY
---------------
  'estricto'      duerme hasta el tiempo real de cada frame y descarta de
                  verdad. Reproduce el comportamiento de la calle, pero NO es
                  determinista: dos corridas dan resultados distintos.

  'determinista'  entrega todos los frames sin dormir. Se mide cuánto tarda
                  cada uno y DESPUÉS se calcula cuáles se habrían descartado.
                  Reproducible: es el modo para desarrollar y para los números
                  del informe.
"""

import csv
import threading
import time
import collections
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------- GPS

class GpsBuffer:
    """
    Buffer de fixes GPS con acceso CAUSAL.

    `latest_before` solo devuelve mediciones que ya habían llegado en el
    instante consultado. Es lo que impide que el sistema use información del
    futuro, que es la trampa más fácil de cometer al pasar de offline a vivo.
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
        """El fix más reciente que YA HABÍA LLEGADO en t_ns."""
        with self._lock:
            for ts, pos in reversed(self._buf):
                if ts <= t_ns:
                    return ts, pos
        return None

    def __len__(self):
        with self._lock:
            return len(self._buf)


# --------------------------------------------------------------- frames

@dataclass
class Frame:
    t_ns: int          # timestamp de captura (reloj de la fuente)
    index: int         # número de frame en la secuencia
    image: np.ndarray


class ReplayFrameSource:
    """
    Reproduce un video grabado respetando los tiempos reales entre frames.

    En modo estricto duerme hasta que 'toca' cada frame, igual que si la
    cámara los estuviera entregando. En modo determinista los entrega de
    corrido.
    """

    def __init__(self, video_path: str, timestamps_ns, crop_bottom: float = 0.0,
                 strict: bool = True, max_frames: Optional[int] = None):
        self.video_path = video_path
        self.timestamps = np.asarray(timestamps_ns, dtype=np.int64)
        self.crop_bottom = crop_bottom      # fracción inferior a recortar (tablero)
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
        if self.crop_bottom > 0:
            h = img.shape[0]
            img = img[: int(h * (1.0 - self.crop_bottom))]
        return img

    def frames(self):
        """Generador de Frame. En modo estricto bloquea hasta el tiempo real."""
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
                # esperar hasta que "toque" este frame según su timestamp real
                objetivo = t0_wall + (int(self.timestamps[i]) - t0_data) / 1e9
                espera = objetivo - time.perf_counter()
                if espera > 0:
                    time.sleep(espera)

            yield Frame(t_ns=int(self.timestamps[i]), index=i,
                        image=self._preprocess(img))


class ReplayGpsSource:
    """
    Entrega los fixes de GPS a su ritmo real (1 Hz en los datos móviles).

    Corre en su propio hilo: el GPS llega cuando llega, no cuando el
    procesamiento lo pide.
    """

    def __init__(self, fixes, buffer: GpsBuffer, strict: bool = True):
        self.fixes = fixes                  # lista de (t_ns, np.array([x, y, z]))
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
                objetivo = t0_wall + (t_ns - t0_data_ns) / 1e9
                espera = objetivo - time.perf_counter()
                if espera > 0:
                    time.sleep(espera)
            self.buffer.push(t_ns, pos)

    def push_all_up_to(self, t_ns: int):
        """Modo determinista: publica todos los fixes que ya habrían llegado."""
        while self._idx < len(self.fixes) and self.fixes[self._idx][0] <= t_ns:
            t, pos = self.fixes[self._idx]
            self.buffer.push(t, pos)
            self._idx += 1

    _idx = 0

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------- carga

def load_mobile_session(session_dir: str, latlon_to_utm):
    """
    Lee una sesión grabada con MARS Logger (formato de mobile_data/).

    Devuelve (timestamps_de_frames_ns, lista_de_fixes_gps).
    Ambos se alinean por el campo 'Unix time', que la app escribe con el mismo
    reloj para todos los sensores.
    """
    import os
    import pandas as pd

    ft = pd.read_csv(os.path.join(session_dir, "frame_timestamps.txt"))
    frame_t = ft["Unix time[nanosec]"].values.astype(np.int64)

    loc = pd.read_csv(os.path.join(session_dir, "location.csv"))
    origen = None
    fixes = []
    for _, r in loc.iterrows():
        utm = latlon_to_utm(r["latitude[degrees]"], r["longitude[degrees]"],
                            r["altitude[meters]"])
        if origen is None:
            origen = utm
        fixes.append((int(r["Unix time[nanosecond]"]), utm - origen))

    return frame_t, fixes
