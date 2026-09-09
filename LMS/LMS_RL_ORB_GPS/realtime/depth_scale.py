"""
Estimador de escala métrica por profundidad monocular.

QUÉ RESUELVE
------------
`cv2.recoverPose` entrega la dirección del movimiento pero no su magnitud: el
vector de traslación siempre tiene norma 1. Históricamente esa magnitud se
tomaba del GPS, lo que encadenaba ambos sensores: al degradarse el GPS se
degradaba también la cámara.

Este módulo le da a la cámara una escala PROPIA, obtenida de la imagen:

  1. Se triangulan las correspondencias con la pose de norma unitaria.
     Da profundidades en "unidades de VO".
  2. Un modelo de profundidad métrica predice esas mismas profundidades en
     metros.
  3. escala = mediana( metros / unidades )  ->  distancia recorrida en metros.

LA BASE DE TRIANGULACIÓN ES CRÍTICA
-----------------------------------
Triangular necesita paralaje. A 30 fps un vehículo avanza ~14 cm entre frames
vecinos, insuficiente para puntos a 10-50 m: el resultado es ruido. Medido
sobre datos reales, con separación de 1 frame el sesgo era 2.64x; separando
~0.33 s baja a 1.08x.

Por eso el estimador NO compara frames consecutivos, sino frames separados por
`min_baseline_s` segundos.

USO OFFLINE Y EN VIVO
---------------------
La misma clase sirve para ambos: se le entregan frames con su timestamp y
devuelve una VELOCIDAD (m/s) cuando tiene base suficiente. La velocidad es
suave, así que el pipeline puede sostenerla entre actualizaciones.
"""

import threading
from collections import deque
from typing import Optional

import cv2
import numpy as np


class DepthScaleEstimator:
    """
    Estima la velocidad métrica del vehículo a partir de la imagen.

    Es deliberadamente independiente del GPS: no lo consume en ningún momento.
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
                 device: Optional[int] = None,
                 orb=None, matcher=None, lowe_ratio: float = 0.75):
        self.K = camera_matrix
        self.min_baseline_s = min_baseline_s
        self.min_parallax = min_parallax_px
        self.dmin, self.dmax = depth_range
        self.min_points = min_points
        self.lowe = lowe_ratio
        # Filtro de plausibilidad física. Un vehículo no salta de 7 a 75 m/s
        # entre frames; esas estimaciones son errores de triangulación.
        self.max_speed = max_speed_ms
        self._raw = deque(maxlen=smooth_window)
        self.n_rechazadas = 0

        self.orb = orb if orb is not None else cv2.ORB_create(2000)
        self.matcher = matcher if matcher is not None else cv2.BFMatcher(cv2.NORM_HAMMING)

        if device is None:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        from transformers import pipeline
        self._pipe = pipeline("depth-estimation", model=model, device=device)

        # historial de frames para tener base de triangulación suficiente
        self._hist = deque(maxlen=120)
        self._lock = threading.Lock()
        self._velocity = None          # m/s, última estimación válida
        self._last_t = None
        self.n_ok = 0
        self.n_fail = 0

    # ------------------------------------------------------------------ API
    @property
    def velocity(self) -> Optional[float]:
        """Última velocidad estimada en m/s, o None si aún no hay ninguna."""
        with self._lock:
            return self._velocity

    def push(self, t_s: float, image: np.ndarray, kp=None, des=None):
        """Registra un frame. Las features se calculan si no se entregan."""
        if kp is None or des is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            kp, des = self.orb.detectAndCompute(gray, None)
        self._hist.append((t_s, image, kp, des))

    def try_update(self) -> Optional[float]:
        """
        Intenta estimar la velocidad con el frame más reciente contra uno
        suficientemente antiguo. Devuelve m/s o None.
        """
        if len(self._hist) < 2:
            return None
        t1, img1, kp1, des1 = self._hist[-1]

        anterior = None
        for entry in reversed(self._hist):
            if t1 - entry[0] >= self.min_baseline_s:
                anterior = entry
                break
        if anterior is None:
            return None

        t0, img0, kp0, des0 = anterior
        dt = t1 - t0
        dist = self._metric_distance(img0, kp0, des0, kp1, des1)
        if dist is None or dt <= 0:
            self.n_fail += 1
            return None

        v_raw = dist / dt

        # ---- 1) límite físico duro ---------------------------------------
        # Errores de triangulación producen valores absurdos: se observaron
        # picos de 75 m/s (270 km/h) en datos reales. Se descartan por
        # imposibles, sin compararlos con ninguna otra fuente.
        if not np.isfinite(v_raw) or v_raw < 0 or v_raw > self.max_speed:
            self.n_rechazadas += 1
            return None

        # ---- 2) mediana móvil --------------------------------------------
        # Un rechazo duro por aceleración descarta demasiadas estimaciones
        # legítimas (el estimador es ruidoso, no solo el vehículo). La mediana
        # suprime los picos restantes conservando la variación real.
        self._raw.append(v_raw)
        v = float(np.median(self._raw))

        with self._lock:
            self._velocity = v
            self._last_t = t1
        self.n_ok += 1
        return v

    # -------------------------------------------------------------- interno
    def _metric_distance(self, img0, kp0, des0, kp1, des1) -> Optional[float]:
        """Distancia en metros entre las dos vistas, o None si no es fiable."""
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

        # descartar puntos con poca paralaje: triangulan mal
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

        razones = zm[ok] / z[ok]
        # ||t|| = 1 por construcción, así que la mediana de razones ES la
        # distancia métrica recorrida entre las dos vistas
        return float(np.median(razones))

    def depth_map(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        out = self._pipe(Image.fromarray(rgb))
        return np.asarray(out["predicted_depth"], dtype=np.float32)
