"""
Pipeline de tiempo real: hilos, colas y medición de latencia.

ARQUITECTURA

    [hilo captura] --cola de 1, descarta lo viejo--> [hilo procesamiento]
    [hilo GPS]     --buffer circular causal-------->        |
                                                            v
                                                    [métricas: latencia, Hz, descartes]

POR QUÉ HILOS Y NO UN SOLO BUCLE
--------------------------------
La cámara produce frames cada 33 ms la estés atendiendo o no. Con un único
bucle que tarda 60 ms por frame, los frames se acumulan en el buffer del
sistema y cada lectura devuelve uno cada vez más viejo. A los 10 segundos
estarías reportando dónde estaba el vehículo hace 5 segundos, aunque el
contador de FPS se vea bien.

Con un hilo de captura que sobrescribe siempre el más reciente, se procesa
el frame más fresco disponible y los intermedios se descartan a propósito.
Una posición actual aproximada vale más que una vieja y precisa.

LA COLA DE TAMAÑO 1 ES LO IMPORTANTE
------------------------------------
Si se encolara todo, el throughput se vería bien pero la latencia crecería sin
límite y dejaría de ser tiempo real. La tasa de descarte es, en sí misma, una
métrica que hay que reportar.
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .sources import Frame, GpsBuffer


# --------------------------------------------------------------- comunicación

def put_drop_oldest(q: queue.Queue, item, counter: list):
    """Cola de un elemento: si había uno sin consumir, se descarta."""
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


# --------------------------------------------------------------- métricas

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
    """Acumula tiempos por etapa y latencia extremo a extremo."""
    stages: List[StageTimes] = field(default_factory=list)
    e2e_ms: List[float] = field(default_factory=list)
    gps_age_ms: List[float] = field(default_factory=list)
    processed: int = 0
    dropped: int = 0
    no_gps: int = 0
    t_start: float = 0.0
    t_end: float = 0.0

    def summary(self) -> dict:
        def pct(v, p):
            return float(np.percentile(v, p)) if v else float("nan")

        tot = [s.total * 1000 for s in self.stages]
        wall = max(self.t_end - self.t_start, 1e-9)
        total_frames = self.processed + self.dropped
        return {
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
        }


# --------------------------------------------------------------- procesamiento

class VisualFrontEnd:
    """
    Front-end visual: ORB, emparejamiento y estimación de pose relativa.

    Usa exactamente la misma configuración que el pipeline offline
    (ORB_create(2000), BFMatcher Hamming, ratio de Lowe 0.75) para que los
    resultados sean comparables.
    """

    MIN_MATCHES = 15

    def __init__(self, camera_matrix: np.ndarray, n_features: int = 2000,
                 lowe_ratio: float = 0.75):
        self.K = camera_matrix
        self.orb = cv2.ORB_create(n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.lowe = lowe_ratio
        self._prev_kp = None
        self._prev_des = None

    def process(self, image: np.ndarray):
        """Devuelve (R, t, n_matches, n_inliers, StageTimes)."""
        st = StageTimes()

        t0 = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
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
                        n_in, R_est, t_est, _ = cv2.recoverPose(E, p0, p1, self.K)
                        R, t, n_inliers = R_est, t_est.ravel(), int(n_in)
                    except cv2.error:
                        pass
                st.pose = time.perf_counter() - t0

        self._prev_kp, self._prev_des = kp, des
        return R, t, n_matches, n_inliers, st


# --------------------------------------------------------------- pipeline

class RealtimePipeline:
    """
    Une fuente, hilos y métricas.

    `strict=True`  -> comportamiento real: duerme y descarta. No determinista.
    `strict=False` -> procesa todo y calcula DESPUÉS qué se habría descartado.
                      Reproducible; es el modo para desarrollar y reportar.
    """

    def __init__(self, front_end: VisualFrontEnd, gps_buffer: GpsBuffer,
                 strict: bool = True, on_result=None):
        self.fe = front_end
        self.gps = gps_buffer
        self.strict = strict
        self.on_result = on_result
        self.metrics = Metrics()
        self._q = queue.Queue(maxsize=1)
        self._dropped = [0]
        self._stop = threading.Event()

    # ---- hilo productor -------------------------------------------------
    def _capture_worker(self, source):
        """
        En modo estricto descarta lo viejo: es el comportamiento real.

        En modo determinista NO descarta — encola bloqueando — porque el
        objetivo es medir cuánto tarda CADA frame. Los descartes se calculan
        después con simulate_drops() a partir de esos tiempos.
        """
        try:
            for frame in source.frames():
                if self._stop.is_set():
                    break
                if self.strict:
                    put_drop_oldest(self._q, frame, self._dropped)
                else:
                    self._q.put(frame)          # bloquea hasta que haya lugar
        finally:
            self._q.put(None)          # centinela de fin

    # ---- consumidor -----------------------------------------------------
    def run(self, source, t0_data_ns: int):
        self.metrics.t_start = time.perf_counter()
        t0_wall = self.metrics.t_start

        cap_thread = threading.Thread(
            target=self._capture_worker, args=(source,), daemon=True)
        cap_thread.start()

        while True:
            try:
                item = self._q.get(timeout=5.0)
            except queue.Empty:
                break
            if item is None:
                break

            frame: Frame = item
            R, t, n_m, n_i, st = self.fe.process(frame.image)

            # emparejamiento CAUSAL con el GPS
            fix = self.gps.latest_before(frame.t_ns)
            if fix is None:
                self.metrics.no_gps += 1
            else:
                self.metrics.gps_age_ms.append((frame.t_ns - fix[0]) / 1e6)

            # latencia extremo a extremo: cuánto nos atrasamos respecto del
            # instante en que ese frame fue capturado
            if self.strict:
                elapsed_wall = time.perf_counter() - t0_wall
                elapsed_data = (frame.t_ns - t0_data_ns) / 1e9
                self.metrics.e2e_ms.append(max(elapsed_wall - elapsed_data, 0.0) * 1000)
            else:
                self.metrics.e2e_ms.append(st.total * 1000)

            self.metrics.stages.append(st)
            self.metrics.processed += 1

            if self.on_result is not None:
                self.on_result(frame, R, t, n_m, n_i, fix)

        self.metrics.t_end = time.perf_counter()
        self.metrics.dropped = self._dropped[0]
        self._stop.set()
        cap_thread.join(timeout=2.0)
        return self.metrics

    def stop(self):
        self._stop.set()


def simulate_drops(frame_times_ns, proc_seconds) -> dict:
    """
    Modo determinista: dado cuánto tardó cada frame, calcula cuáles se habrían
    descartado con una cola de tamaño 1.

    Se simula un reloj virtual: si al terminar un frame ya llegaron otros, solo
    se atiende el más reciente y los intermedios se descartan.
    """
    t = np.asarray(frame_times_ns, dtype=np.float64) / 1e9
    t = t - t[0]
    d = np.asarray(proc_seconds, dtype=np.float64)

    reloj = 0.0
    procesados, descartados, latencias = 0, 0, []
    i = 0
    n = len(t)
    while i < n:
        if reloj <= t[i]:
            reloj = t[i]
        else:
            # ya pasó su momento: saltar a los que ya llegaron y quedarse
            # con el más reciente
            j = i
            while j + 1 < n and t[j + 1] <= reloj:
                j += 1
            descartados += (j - i)
            i = j
        inicio = max(reloj, t[i])
        reloj = inicio + d[min(i, len(d) - 1)]
        latencias.append((reloj - t[i]) * 1000)
        procesados += 1
        i += 1

    return {
        "frames_procesados": procesados,
        "frames_descartados": descartados,
        "tasa_descarte_%": 100.0 * descartados / max(procesados + descartados, 1),
        "e2e_ms_p50": float(np.percentile(latencias, 50)),
        "e2e_ms_p95": float(np.percentile(latencias, 95)),
        "e2e_ms_max": float(np.max(latencias)),
    }
