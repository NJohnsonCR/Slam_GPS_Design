"""
Probe: can a metric depth model provide the scale of monocular VO?

cv2.recoverPose returns the direction of motion but not its magnitude: the
translation always has norm 1. This probe recovers the magnitude from the image:

  1. Triangulate the ORB matches with the unit-norm pose -> depths in VO units.
  2. Read the metric depth model at those same pixels     -> depths in metres.
  3. scale = median(metres / units). Since ||t|| = 1, that is the distance
     travelled between the two frames.

KITTI ground truth (OXTS) is used only to evaluate; the method never reads it.

Besides the scale, each estimate stores observable features (matches, inliers,
triangulated points, dispersion of the depth ratios) for scale_decision.py.

Usage:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.scale.depth_scale_probe \\
        kitti_data/2011_09_26/2011_09_26_drive_0009_sync [more sequences...]
"""

import argparse
import os
import sys
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
MIN_PARALLAX_PX = 1.0              # low-parallax points triangulate badly
DEPTH_MIN, DEPTH_MAX = 3.0, 60.0   # useful range of the model on the road
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


def depth_scale(pts0, pts1, R, t, K, depth_map):
    """
    Return (scale, n_points, dispersion), or (None, ...) when unreliable.

    dispersion is the median absolute deviation of the depth ratios divided by
    their median. It needs no ground truth, so it can serve as a confidence
    signal.
    """
    parallax = np.linalg.norm(pts1 - pts0, axis=1)
    keep = parallax > MIN_PARALLAX_PX
    if keep.sum() < MIN_VALID_POINTS:
        return None, 0, np.nan
    p0, p1 = pts0[keep], pts1[keep]

    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t.reshape(3, 1)])

    X = cv2.triangulatePoints(P0, P1, p0.T, p1.T)
    w = X[3]
    ok = np.abs(w) > 1e-9
    X = X[:, ok] / w[ok]
    p0 = p0[ok]

    z_vo = X[2]                        # depth in VO units
    ok = z_vo > 1e-6
    z_vo, p0 = z_vo[ok], p0[ok]
    if len(z_vo) < MIN_VALID_POINTS:
        return None, 0, np.nan

    # Metric depth of the same pixels, according to the model.
    h, wd = depth_map.shape
    cols = np.clip(p0[:, 0].astype(int), 0, wd - 1)
    rows = np.clip(p0[:, 1].astype(int), 0, h - 1)
    z_m = depth_map[rows, cols]

    ok = (z_m > DEPTH_MIN) & (z_m < DEPTH_MAX)
    if ok.sum() < MIN_VALID_POINTS:
        return None, int(ok.sum()), np.nan

    ratios = z_m[ok] / z_vo[ok]
    median = float(np.median(ratios))
    dispersion = float(np.median(np.abs(ratios - median)) / max(abs(median), 1e-9))
    return median, int(ok.sum()), dispersion


def run_sequence(seq_path, pipe, args):
    """Process one sequence. Return its median scale error, or None."""
    img_dir, gps_dir = find_dirs(seq_path)
    files = sorted(f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg")))
    if args.max_frames:
        files = files[: args.max_frames]
    n = len(files)

    gt = load_gt(gps_dir, n)
    true_step = np.linalg.norm(np.diff(gt, axis=0), axis=1)

    seq = os.path.basename(os.path.normpath(seq_path))
    print("=" * 84)
    print(f"ESCALA POR PROFUNDIDAD MONOCULAR — {seq}  ({n} frames)")
    print("=" * 84)

    from PIL import Image

    slam = PoseGraphSLAM(fx=KITTI_FX, fy=KITTI_FY, cx=KITTI_CX, cy=KITTI_CY)
    K = slam.camera_matrix

    prev_img = prev_kp = prev_des = None
    idx, scales, true_steps, n_pts = [], [], [], []
    n_match, n_inlier, dispersions = [], [], []
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
                        n_in, R, t, _ = cv2.recoverPose(E, p0, p1, K)
                    except cv2.error:
                        R = None
                    if R is not None:
                        t0 = time.perf_counter()
                        pil = Image.fromarray(cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB))
                        dmap = np.asarray(pipe(pil)["predicted_depth"], dtype=np.float32)
                        t_depth.append(time.perf_counter() - t0)

                        s, npt, disp = depth_scale(p0, p1, R, t.ravel(), K, dmap)
                        if s is not None:
                            idx.append(i - 1)
                            scales.append(s)
                            true_steps.append(true_step[i - 1])
                            n_pts.append(npt)
                            n_match.append(len(good))
                            n_inlier.append(int(n_in))
                            dispersions.append(disp)

        prev_img, prev_kp, prev_des = img, kp, des
        if i % 200 == 0:
            print(f"  frame {i}/{n}")

    if len(scales) < 10:
        print("\n  Muy pocas estimaciones válidas. El método no aplica así.")
        return None

    scales = np.array(scales)
    true_steps = np.array(true_steps)
    err = 100 * np.abs(scales - true_steps) / np.maximum(true_steps, 1e-6)
    bias = 100 * (np.median(scales / np.maximum(true_steps, 1e-6)) - 1)

    print(f"\n{'-'*84}")
    print("RESULTADOS")
    print(f"{'-'*84}")
    print(f"  Estimaciones válidas       : {len(scales)}/{n-1}")
    print(f"  Puntos usados por estimación: mediana {np.median(n_pts):.0f}")
    print(f"  Tiempo del modelo por frame : {np.mean(t_depth)*1000:.0f} ms")
    print()
    print(f"  ERROR DE ESCALA POR FRAME:")
    print(f"     mediana {np.median(err):5.1f} %     p90 {np.percentile(err,90):5.1f} %")
    print(f"     sesgo sistemático: {bias:+.1f} %")
    print()
    print("  COMPARACIÓN con las otras fuentes de escala (mismo problema):")
    print(f"     escala del GPS a 1 Hz        ~45 %")
    print(f"     plano de tierra (simple)     ~50 %")
    print(f"     PROFUNDIDAD MONOCULAR        {np.median(err):.1f} %   <-- este experimento")

    # Scale changes slowly, so a rolling median is a fair smoother.
    for w in (5, 15):
        sm = np.array([np.median(scales[max(0, k-w+1):k+1]) for k in range(len(scales))])
        e = 100 * np.abs(sm - true_steps) / np.maximum(true_steps, 1e-6)
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
    ax[0].plot(idx, true_steps, lw=2, color="#F2A03D", label="distancia real (GT)")
    ax[0].plot(idx, scales, lw=1.2, color="#4FD1C5", label="estimada por profundidad")
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
    plt.close(fig)
    print(f"\nGráfico: {png}")

    np.savez(os.path.join(args.out_dir, f"{seq}_depth_scale.npz"),
             idx=idx, scales=scales, true_steps=true_steps, err=err,
             n_pts=n_pts, n_match=n_match, n_inlier=n_inlier,
             dispersion=dispersions)
    return float(np.median(err))


def main():
    ap = argparse.ArgumentParser(description="Sondeo de escala por profundidad monocular")
    ap.add_argument("sequence", nargs="+",
                    help="Una o varias secuencias KITTI (acepta glob del shell)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--step", type=int, default=1, help="Procesar 1 de cada N pares")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    ap.add_argument("--out-dir", default="resultados/rl_cache")
    args = ap.parse_args()

    import torch
    from transformers import pipeline

    dev = 0 if torch.cuda.is_available() else -1
    print(f"Modelo: {args.model}   Dispositivo: {'GPU' if dev == 0 else 'CPU'}\n")
    pipe = pipeline("depth-estimation", model=args.model, device=dev)

    summary = []
    for seq_path in args.sequence:
        try:
            med = run_sequence(seq_path, pipe, args)
        except Exception as e:           # one bad sequence must not abort the batch
            print(f"  ERROR en {seq_path}: {e}")
            med = None
        summary.append((os.path.basename(os.path.normpath(seq_path)), med))

    if len(summary) > 1:
        print("\n" + "=" * 84)
        print("RESUMEN DEL LOTE — error mediano de escala por secuencia")
        print("=" * 84)
        for name, med in summary:
            print(f"  {name:<34}" + (f"{med:6.1f} %" if med is not None else "   (sin datos)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
