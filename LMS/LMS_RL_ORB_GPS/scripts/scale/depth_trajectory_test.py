"""
EXPERIMENTO DECISIVO - ¿La escala por profundidad produce mejor TRAYECTORIA?

Por qué hace falta este experimento
-----------------------------------
Ya se midió que la profundidad monocular da mejor ESCALA que las alternativas.
Pero acertar la escala y acertar el recorrido no son lo mismo: con el plano de
tierra pasó que tenía menor error de escala que el GPS (34% vs 45%) y aun así
generaba una trayectoria PEOR (9.39 m vs 9.13 m).

Este script cierra esa brecha: reconstruye la trayectoria con cada fuente de
escala y las compara contra el ground truth de KITTI.

Se corre sobre KITTI porque es el único lugar con verdad absoluta. El GPS se
degrada a 1 Hz y con ruido para imitar el régimen del teléfono.

ALINEACIÓN: rígida (rotación + traslación), SIN escala. Alinear con escala
absorbería justamente el error que se quiere medir.

Uso:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.scale.depth_trajectory_test \
        kitti_data/2011_09_26/2011_09_26_drive_0009_sync
"""

import os
import sys
import argparse

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

FX = FY = 718.856
CX, CY = 607.1928, 185.2157
GPS_PERIOD, GPS_SIGMA = 10, 5.0     # 1 Hz y 5 m de ruido: régimen del teléfono


def find_dirs(seq):
    img = gps = None
    for root, dirs, _f in os.walk(seq):
        if "image_02" in dirs and "data" in os.listdir(os.path.join(root, "image_02")):
            img = os.path.join(root, "image_02", "data")
        if "oxts" in dirs and "data" in os.listdir(os.path.join(root, "oxts")):
            gps = os.path.join(root, "oxts", "data")
    return img, gps


def align_rigid(est, ref):
    """Alineación SIN escala: solo rotación y traslación."""
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    ec, rc = est - est.mean(0), ref - ref.mean(0)
    U, S, Vt = np.linalg.svd(ec.T @ rc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return (ec @ R.T) + ref.mean(0)


def ate(est, ref):
    a = align_rigid(est, ref)
    n = min(len(a), len(ref))
    return float(np.sqrt(np.mean(np.linalg.norm(a[:n] - ref[:n], axis=1) ** 2)))


def integrate(R_rel, t_rel, scales):
    P = np.eye(4)
    pts = [np.zeros(3)]
    for i in range(len(R_rel)):
        rel = np.eye(4)
        rel[:3, :3] = R_rel[i]
        rel[:3, 3] = t_rel[i] * scales[i]
        P = P @ rel
        pts.append(P[:3, 3].copy())
    return np.array(pts)


def gps_scales(gps, n, period, smooth=5):
    """Escala derivada del GPS: el método actual del pipeline."""
    s = np.ones(n)
    dists, spans = [], []
    for k in range(n // period):
        i0, i1 = k * period, min((k + 1) * period, n)
        dists.append(float(np.linalg.norm(gps[i1] - gps[i0])))
        spans.append((i0, i1))
    for k, (i0, i1) in enumerate(spans):
        lo = max(0, k - smooth + 1)
        s[i0:i1] = float(np.mean(dists[lo:k + 1])) / max(i1 - i0, 1)
    if spans:
        s[spans[-1][1]:] = s[spans[-1][1] - 1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--gap", type=int, default=2, help="Separación para triangular")
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

    gt = np.array([latlon_to_utm(*np.loadtxt(os.path.join(gps_dir, f))[:3])
                   for f in sorted(os.listdir(gps_dir))[:n]])
    gt -= gt[0]
    true_step = np.linalg.norm(np.diff(gt, axis=0), axis=1)

    seq = os.path.basename(os.path.normpath(args.sequence))
    print("=" * 82)
    print(f"TRAYECTORIA SEGÚN LA FUENTE DE ESCALA — {seq}  ({n} frames)")
    print("=" * 82)
    print(f"  Distancia real: {true_step.sum():.0f} m")
    print(f"  GPS simulado: cada {GPS_PERIOD} frames (1 Hz), sigma={GPS_SIGMA} m")
    print(f"  Separación para triangular: {args.gap} frames\n")

    slam = PoseGraphSLAM(fx=FX, fy=FY, cx=CX, cy=CY)
    K = slam.camera_matrix
    pipe = pipeline("depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
                    device=0 if torch.cuda.is_available() else -1)

    # ---- VO frame a frame + escala por profundidad ----
    imgs, kps, dess = [], [], []
    for i, fn in enumerate(files):
        img = cv2.imread(os.path.join(img_dir, fn))
        kp, des = slam.orb_detector.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        imgs.append(img); kps.append(kp); dess.append(des)
        if i % 150 == 0:
            print(f"  ORB {i}/{n}")

    R_rel, t_rel = [], []
    for i in range(1, n):
        R, t = np.eye(3), np.zeros(3)
        if dess[i] is not None and dess[i-1] is not None:
            g = slam.filter_matches_lowe_ratio(dess[i-1], dess[i])
            if len(g) > 15:
                p0 = np.float32([kps[i-1][m.queryIdx].pt for m in g])
                p1 = np.float32([kps[i][m.trainIdx].pt for m in g])
                E, _ = cv2.findEssentialMat(p0, p1, K, cv2.RANSAC, 0.999, 1.0)
                if E is not None and E.shape == (3, 3):
                    try:
                        _, R, tt, _ = cv2.recoverPose(E, p0, p1, K)
                        t = tt.ravel()
                    except cv2.error:
                        pass
        R_rel.append(R); t_rel.append(t)
    R_rel, t_rel = np.array(R_rel), np.array(t_rel)

    print("\n  Estimando escala por profundidad...")
    depth_v = np.full(n - 1, np.nan)
    gap = args.gap
    for i in range(gap, n):
        if dess[i] is None or dess[i-gap] is None:
            continue
        g = slam.filter_matches_lowe_ratio(dess[i-gap], dess[i])
        if len(g) <= 15:
            continue
        p0 = np.float32([kps[i-gap][m.queryIdx].pt for m in g])
        p1 = np.float32([kps[i][m.trainIdx].pt for m in g])
        E, _ = cv2.findEssentialMat(p0, p1, K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3):
            continue
        try:
            _, Rm, tm, _ = cv2.recoverPose(E, p0, p1, K)
        except cv2.error:
            continue
        dm = np.asarray(pipe(Image.fromarray(cv2.cvtColor(imgs[i-gap], cv2.COLOR_BGR2RGB)))
                        ["predicted_depth"], dtype=np.float32)
        par = np.linalg.norm(p1 - p0, axis=1)
        keep = par > 1.0
        if keep.sum() < 20:
            continue
        a, b = p0[keep], p1[keep]
        P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = K @ np.hstack([Rm, tm.reshape(3, 1)])
        X = cv2.triangulatePoints(P0, P1, a.T, b.T)
        w = X[3]; ok = np.abs(w) > 1e-9
        X = X[:, ok] / w[ok]; a = a[ok]
        z = X[2]; ok = z > 1e-6; z, a = z[ok], a[ok]
        if len(z) < 20:
            continue
        h, wd = dm.shape
        zm = dm[np.clip(a[:, 1].astype(int), 0, h-1), np.clip(a[:, 0].astype(int), 0, wd-1)]
        ok = (zm > 3) & (zm < 60)
        if ok.sum() < 20:
            continue
        # distancia métrica del intervalo -> repartida entre sus frames
        depth_v[i-gap:i] = float(np.median(zm[ok] / z[ok])) / gap
        if i % 150 == 0:
            print(f"     {i}/{n}")

    # rellenar huecos e imponer positividad
    idx = np.arange(n - 1)
    good = np.isfinite(depth_v) & (depth_v > 0)
    depth_s = np.interp(idx, idx[good], depth_v[good]) if good.sum() > 5 else np.ones(n - 1)

    def medfilt(x, k):
        return np.array([np.median(x[max(0, i-k//2):i+k//2+1]) for i in range(len(x))])

    # ---- GPS degradado ----
    rng = np.random.default_rng(0)
    gps = gt.copy()
    gps[:, :2] += rng.normal(0, GPS_SIGMA, size=(n, 2))

    # ---- comparación ----
    variantes = [
        ("Ground truth (cota superior)", true_step),
        ("Derivada del GPS 1 Hz (método actual)", gps_scales(gps, n-1, GPS_PERIOD)),
        ("PROFUNDIDAD (cruda)", depth_s),
        ("PROFUNDIDAD + mediana móvil 5", medfilt(depth_s, 5)),
        ("PROFUNDIDAD + mediana móvil 15", medfilt(depth_s, 15)),
    ]

    print(f"\n{'-'*82}")
    print(f"{'fuente de escala':<40}{'ATE (m)':>10}{'error escala':>15}{'dist. total':>13}")
    print("-" * 82)
    trays = {}
    for nombre, sc in variantes:
        sc = np.asarray(sc, dtype=float)[:n-1]
        tr = integrate(R_rel, t_rel, sc)
        trays[nombre] = tr
        e = 100 * np.median(np.abs(sc - true_step) / np.maximum(true_step, 1e-6))
        print(f"{nombre:<40}{ate(tr, gt):>10.2f}{e:>14.0f}%{sc.sum():>12.0f} m")
    print("-" * 82)
    print(f"{'(distancia real)':<40}{'':>10}{'':>15}{true_step.sum():>12.0f} m")

    # ---- gráfico ----
    os.makedirs(args.out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(gt[:, 0], gt[:, 1], "k-", lw=2.5, label="Ground truth")
    for nombre, color in (("Derivada del GPS 1 Hz (método actual)", "#F2A03D"),
                          ("PROFUNDIDAD + mediana móvil 5", "#4FD1C5")):
        a = align_rigid(trays[nombre], gt)
        ax[0].plot(a[:, 0], a[:, 1], "--", lw=1.6, color=color,
                   label=f"{nombre.split('(')[0].strip()} ({ate(trays[nombre], gt):.1f} m)")
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Y (m)")
    ax[0].set_title("Trayectorias"); ax[0].axis("equal"); ax[0].grid(alpha=0.3); ax[0].legend()

    ax[1].plot(true_step, lw=2, color="k", label="real")
    ax[1].plot(medfilt(depth_s, 5), lw=1.1, color="#4FD1C5", label="profundidad (filtrada)")
    ax[1].plot(gps_scales(gps, n-1, GPS_PERIOD), lw=1.1, color="#F2A03D", label="GPS 1 Hz")
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("desplazamiento por frame (m)")
    ax[1].set_title("Escala estimada"); ax[1].grid(alpha=0.3); ax[1].legend()

    fig.suptitle(f"Trayectoria según la fuente de escala — {seq}")
    fig.tight_layout()
    png = os.path.join(args.out_dir, f"{seq}_depth_trajectory.png")
    fig.savefig(png, dpi=130)
    print(f"\nGráfico: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
