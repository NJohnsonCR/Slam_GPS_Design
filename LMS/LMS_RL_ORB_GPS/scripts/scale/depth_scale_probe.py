"""
SONDEO - ¿Puede un modelo de profundidad métrica dar la escala de la VO?

EL PROBLEMA
-----------
`cv2.recoverPose` devuelve la DIRECCIÓN del movimiento pero no su magnitud: el
vector de traslación siempre tiene norma 1. Hace falta una referencia métrica
externa. Hoy esa referencia es el GPS, lo que encadena ambos sensores.

LA IDEA
-------
Un modelo de profundidad métrica predice, para cada píxel, su distancia EN
METROS. Con eso se puede despejar la escala:

  1. Se triangulan los puntos que ORB emparejó, usando la pose sin escala.
     Eso da su profundidad en "unidades de VO" (donde ||t|| = 1).
  2. El modelo dice cuál es la profundidad de esos mismos píxeles en metros.
  3. escala = mediana( profundidad_metros / profundidad_unidades )
  4. Como ||t|| = 1, esa escala ES la distancia recorrida en metros.

VALIDACIÓN
----------
Se compara contra la distancia real del ground truth de KITTI (OXTS). El
ground truth se usa SOLO para evaluar; el método no lo consume.

REFERENCIAS YA MEDIDAS (mismo problema, otras fuentes de escala):
    escala derivada del GPS a 1 Hz  ->  ~45 % de error mediano
    plano de tierra (implementación simple) -> ~50 % de error mediano

Uso:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.scale.depth_scale_probe \
        kitti_data/2011_09_26/2011_09_26_drive_0009_sync
"""

import os
import sys
import argparse
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))
_LMS_RL = os.path.abspath(os.path.join(_THIS, "..", ".."))
_ROOT = os.path.abspath(os.path.join(_LMS_RL, "..", ".."))
for _p in (_ROOT, _LMS_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from LMS.LMS_ORB_with_PG.main import PoseGraphSLAM
from utils.gps.gps_utils import latlon_to_utm

KITTI_FX, KITTI_FY = 718.856, 718.856
KITTI_CX, KITTI_CY = 607.1928, 185.2157

MIN_MATCHES = 15
MIN_PARALLAX_PX = 1.0      # puntos con muy poca disparidad triangulan mal
DEPTH_MIN, DEPTH_MAX = 3.0, 60.0   # rango útil del modelo en carretera
MIN_VALID_POINTS = 20


def find_dirs(seq):
    img = gps = None
    for root, dirs, _f in os.walk(seq):
        if "image_02" in dirs and "data" in os.listdir(os.path.join(root, "image_02")):
            img = os.path.join(root, "image_02", "data")
        if "oxts" in dirs and "data" in os.listdir(os.path.join(root, "oxts")):
            gps = os.path.join(root, "oxts", "data")
    return img, gps


def load_gt(gps_dir, n):
    fs = sorted(f for f in os.listdir(gps_dir) if f.endswith(".txt"))[:n]
    pos = [latlon_to_utm(*np.loadtxt(os.path.join(gps_dir, f))[:3]) for f in fs]
    pos = np.array(pos)
    return pos - pos[0]


def escala_por_profundidad(pts0, pts1, R, t, K, depth_map):
    """
    Devuelve la escala métrica implícita, o None si no hay puntos confiables.

    Se triangula con la pose de norma unitaria y se compara la profundidad
    resultante contra la que predice el modelo para los mismos píxeles.
    """
    # descartar puntos con poca paralaje: triangulan con mucho error
    parallax = np.linalg.norm(pts1 - pts0, axis=1)
    keep = parallax > MIN_PARALLAX_PX
    if keep.sum() < MIN_VALID_POINTS:
        return None, 0
    p0, p1 = pts0[keep], pts1[keep]

    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t.reshape(3, 1)])

    X = cv2.triangulatePoints(P0, P1, p0.T, p1.T)
    w = X[3]
    ok = np.abs(w) > 1e-9
    X = X[:, ok] / w[ok]
    p0 = p0[ok]

    z_vo = X[2]                       # profundidad en unidades de VO
    ok = z_vo > 1e-6
    z_vo, p0 = z_vo[ok], p0[ok]
    if len(z_vo) < MIN_VALID_POINTS:
        return None, 0

    # profundidad métrica del modelo en esos mismos píxeles
    h, wd = depth_map.shape
    cols = np.clip(p0[:, 0].astype(int), 0, wd - 1)
    rows = np.clip(p0[:, 1].astype(int), 0, h - 1)
    z_m = depth_map[rows, cols]

    ok = (z_m > DEPTH_MIN) & (z_m < DEPTH_MAX)
    if ok.sum() < MIN_VALID_POINTS:
        return None, int(ok.sum())

    razones = z_m[ok] / z_vo[ok]
    return float(np.median(razones)), int(ok.sum())


def main():
    ap = argparse.ArgumentParser(description="Sondeo de escala por profundidad monocular")
    ap.add_argument("sequence")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--step", type=int, default=1, help="Procesar 1 de cada N pares")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    ap.add_argument("--out-dir", default="resultados/rl_cache")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import pipeline

    img_dir, gps_dir = find_dirs(args.sequence)
    files = sorted(f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg")))
    if args.max_frames:
        files = files[: args.max_frames]
    n = len(files)

    gt = load_gt(gps_dir, n)
    true_step = np.linalg.norm(np.diff(gt, axis=0), axis=1)

    seq = os.path.basename(os.path.normpath(args.sequence))
    print("=" * 84)
    print(f"ESCALA POR PROFUNDIDAD MONOCULAR — {seq}  ({n} frames)")
    print("=" * 84)
    print(f"  Modelo: {args.model}")

    dev = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("depth-estimation", model=args.model, device=dev)
    print(f"  Dispositivo: {'GPU' if dev == 0 else 'CPU'}\n")

    slam = PoseGraphSLAM(fx=KITTI_FX, fy=KITTI_FY, cx=KITTI_CX, cy=KITTI_CY)
    K = slam.camera_matrix

    prev_img = prev_kp = prev_des = None
    idx, escalas, reales, n_pts = [], [], [], []
    t_depth = []

    for i, fn in enumerate(files):
        img = cv2.imread(os.path.join(img_dir, fn))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = slam.orb_detector.detectAndCompute(gray, None)

        if i > 0 and (i % args.step == 0) and des is not None and prev_des is not None:
            good = slam.filter_matches_lowe_ratio(prev_des, des)
            if len(good) > MIN_MATCHES and true_step[i - 1] > 0.05:
                p0 = np.float32([prev_kp[m.queryIdx].pt for m in good])
                p1 = np.float32([kp[m.trainIdx].pt for m in good])
                E, _ = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC,
                                            prob=0.999, threshold=1.0)
                if E is not None and E.shape == (3, 3):
                    try:
                        _, R, t, _ = cv2.recoverPose(E, p0, p1, K)
                    except cv2.error:
                        R = None
                    if R is not None:
                        t0 = time.perf_counter()
                        pil = Image.fromarray(cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB))
                        dmap = np.asarray(pipe(pil)["predicted_depth"], dtype=np.float32)
                        t_depth.append(time.perf_counter() - t0)

                        s, npt = escala_por_profundidad(p0, p1, R, t.ravel(), K, dmap)
                        if s is not None:
                            idx.append(i - 1)
                            escalas.append(s)
                            reales.append(true_step[i - 1])
                            n_pts.append(npt)

        prev_img, prev_kp, prev_des = img, kp, des
        if i % 100 == 0:
            print(f"  frame {i}/{n}")

    if len(escalas) < 10:
        print("\n  Muy pocas estimaciones válidas. El método no aplica así.")
        return 1

    escalas = np.array(escalas)
    reales = np.array(reales)
    err = 100 * np.abs(escalas - reales) / np.maximum(reales, 1e-6)
    sesgo = 100 * (np.median(escalas / np.maximum(reales, 1e-6)) - 1)

    print(f"\n{'-'*84}")
    print("RESULTADOS")
    print(f"{'-'*84}")
    print(f"  Estimaciones válidas       : {len(escalas)}/{n-1}")
    print(f"  Puntos usados por estimación: mediana {np.median(n_pts):.0f}")
    print(f"  Tiempo del modelo por frame : {np.mean(t_depth)*1000:.0f} ms")
    print()
    print(f"  ERROR DE ESCALA POR FRAME:")
    print(f"     mediana {np.median(err):5.1f} %     p90 {np.percentile(err,90):5.1f} %")
    print(f"     sesgo sistemático: {sesgo:+.1f} %")
    print()
    print("  COMPARACIÓN con las otras fuentes de escala (mismo problema):")
    print(f"     escala del GPS a 1 Hz        ~45 %")
    print(f"     plano de tierra (simple)     ~50 %")
    print(f"     PROFUNDIDAD MONOCULAR        {np.median(err):.1f} %   <-- este experimento")

    # suavizado temporal: la escala varía despacio
    for w in (5, 15):
        sm = np.array([np.median(escalas[max(0, k-w+1):k+1]) for k in range(len(escalas))])
        e = 100 * np.abs(sm - reales) / np.maximum(reales, 1e-6)
        print(f"     con mediana móvil de {w:2d}      {np.median(e):.1f} %")

    print("\n" + "=" * 84)
    m = np.median(err)
    if m < 15:
        print("  VEREDICTO: PRECISIÓN BUENA. Supera claramente al GPS y al plano de tierra.")
        print("             La cámara puede tener escala propia.")
    elif m < 30:
        print("  VEREDICTO: PRECISIÓN MODERADA, pero mejor que las alternativas.")
        print("             Vale la pena con filtrado temporal.")
    else:
        print("  VEREDICTO: NO mejora lo suficiente sobre las alternativas ya probadas.")
    print("=" * 84)

    os.makedirs(args.out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(idx, reales, lw=2, color="#F2A03D", label="distancia real (GT)")
    ax[0].plot(idx, escalas, lw=1.2, color="#4FD1C5", label="estimada por profundidad")
    ax[0].set_xlabel("frame"); ax[0].set_ylabel("desplazamiento (m)")
    ax[0].set_title("Escala estimada vs real"); ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].hist(err, bins=50, range=(0, 100), color="steelblue", edgecolor="k", alpha=0.85)
    ax[1].axvline(np.median(err), color="crimson", ls="--",
                  label=f"mediana {np.median(err):.1f}%")
    ax[1].set_xlabel("error de escala (%)"); ax[1].set_ylabel("frecuencia")
    ax[1].set_title("Distribución del error"); ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.suptitle(f"Escala por profundidad monocular — {seq}")
    fig.tight_layout()
    png = os.path.join(args.out_dir, f"{seq}_depth_scale.png")
    fig.savefig(png, dpi=130)
    print(f"\nGráfico: {png}")

    np.savez(os.path.join(args.out_dir, f"{seq}_depth_scale.npz"),
             idx=idx, escalas=escalas, reales=reales, err=err)
    return 0


if __name__ == "__main__":
    sys.exit(main())
