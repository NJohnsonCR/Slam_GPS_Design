"""
Does ground-plane scale recovery work on KITTI? The method needs the camera's
height above the road, h. Instead of assuming h, it is solved for: the
homography between two frames from road points only gives the translation
over the distance to the plane, t_h = t / d; with the true distance from the
ground truth, d = distance / ||t_h||, and d is the camera's height. If the
method works, d is constant over the sequence and close to KITTI's mount
(~1.65 m). The ground truth only validates; the method itself would use a
measured h.

Uso:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.scale.ground_plane_probe \
        kitti_data/2011_09_26/2011_09_26_drive_0009_sync
"""

import os
import sys
import argparse

import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))
_LMS_RL = os.path.abspath(os.path.join(_THIS, '..', '..'))
_ROOT = os.path.abspath(os.path.join(_LMS_RL, '..', '..'))
for _p in (_ROOT, _LMS_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from LMS.LMS_ORB_with_PG.main import PoseGraphSLAM
from utils.gps.gps_utils import latlon_to_utm

# KITTI camera 02 calibration (the pipeline's values)
KITTI_FX, KITTI_FY = 718.856, 718.856
KITTI_CX, KITTI_CY = 607.1928, 185.2157
KITTI_CAM_HEIGHT = 1.65      # the mount's documented height, for reference only

GROUND_ROI_TOP = 0.60        # road points are taken below this share of the image height
MIN_GROUND_MATCHES = 25


def find_dirs(seq):
    img = gps = None
    for root, dirs, _f in os.walk(seq):
        if "image_02" in dirs and "data" in os.listdir(os.path.join(root, "image_02")):
            img = os.path.join(root, "image_02", "data")
        if "oxts" in dirs and "data" in os.listdir(os.path.join(root, "oxts")):
            gps = os.path.join(root, "oxts", "data")
    return img, gps


def load_gt(gps_dir, n):
    fs = sorted(f for f in os.listdir(gps_dir) if f.endswith('.txt'))[:n]
    pos = []
    for f in fs:
        r = np.loadtxt(os.path.join(gps_dir, f))
        pos.append(latlon_to_utm(r[0], r[1], r[2]))
    pos = np.array(pos)
    return pos - pos[0]


def pick_ground_solution(Rs, Ts, Ns):
    """
    Of decomposeHomographyMat's up to 4 solutions, the road's: KITTI's camera
    Y axis points down, so the road normal (towards the camera) has a clearly
    negative Y.
    """
    best, best_score = None, -np.inf
    for R, T, N in zip(Rs, Ts, Ns):
        n = N.ravel()
        if n[1] > 0:           # flipped normal: take the opposite
            n = -n
        score = -n[1]          # how much it points up in the camera frame
        if score > best_score:
            best_score, best = score, (R, T.ravel(), n)
    return best, best_score


def main():
    ap = argparse.ArgumentParser(description="Sondeo de escala por plano de tierra")
    ap.add_argument('sequence')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--out-dir', default='resultados/rl_cache')
    args = ap.parse_args()

    img_dir, gps_dir = find_dirs(args.sequence)
    if not img_dir:
        print("ERROR: no se encontró image_02/data")
        return 1

    files = sorted(f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg')))
    if args.max_frames:
        files = files[:args.max_frames]
    n = len(files)

    gt = load_gt(gps_dir, n)
    true_step = np.linalg.norm(np.diff(gt, axis=0), axis=1)

    slam = PoseGraphSLAM(fx=KITTI_FX, fy=KITTI_FY, cx=KITTI_CX, cy=KITTI_CY)
    K = slam.camera_matrix

    seq_name = os.path.basename(os.path.normpath(args.sequence))
    print("=" * 88)
    print(f"SONDEO DE PLANO DE TIERRA — {seq_name}  ({n} frames)")
    print("=" * 88)
    print(f"  Región del suelo: por debajo del {GROUND_ROI_TOP*100:.0f}% de la altura")
    print(f"  Altura real del montaje KITTI: {KITTI_CAM_HEIGHT} m (solo para comparar)\n")

    prev_kp = prev_des = None
    h_implied, n_ground, normals, ok_flags = [], [], [], []

    for i, fn in enumerate(files):
        frame = cv2.imread(os.path.join(img_dir, fn))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H_img = gray.shape[0]
        kp, des = slam.orb_detector.detectAndCompute(gray, None)

        if i > 0 and des is not None and prev_des is not None:
            matches = slam.filter_matches_lowe_ratio(prev_des, des)

            # only correspondences in the bottom of both images
            g = [m for m in matches
                 if prev_kp[m.queryIdx].pt[1] > GROUND_ROI_TOP * H_img
                 and kp[m.trainIdx].pt[1] > GROUND_ROI_TOP * H_img]
            n_ground.append(len(g))

            done = False
            if len(g) >= MIN_GROUND_MATCHES and true_step[i - 1] > 0.05:
                p0 = np.float32([prev_kp[m.queryIdx].pt for m in g])
                p1 = np.float32([kp[m.trainIdx].pt for m in g])
                Hm, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 2.0)

                if Hm is not None and mask is not None and mask.sum() >= MIN_GROUND_MATCHES * 0.5:
                    ret, Rs, Ts, Ns = cv2.decomposeHomographyMat(Hm, K)
                    if ret > 0:
                        (Rb, Tb, Nb), score = pick_ground_solution(Rs, Ts, Ns)
                        nt = float(np.linalg.norm(Tb))
                        if nt > 1e-6:
                            # t_h = t / d  ->  d = true distance / ||t_h||
                            h_implied.append(true_step[i - 1] / nt)
                            normals.append(Nb)
                            done = True
            ok_flags.append(done)
            if not done:
                h_implied.append(np.nan)
                normals.append([np.nan] * 3)

        prev_kp, prev_des = kp, des
        if i % 100 == 0:
            print(f"  frame {i}/{n}")

    h = np.array(h_implied, dtype=float)
    valid = h[np.isfinite(h)]
    ng = np.array(n_ground)
    N = np.array(normals, dtype=float)

    print(f"\n{'-'*88}")
    print("RESULTADOS")
    print(f"{'-'*88}")
    print(f"  Correspondencias en la región del suelo: mediana {np.median(ng):.0f}, "
          f"mín {ng.min()}, máx {ng.max()}")
    print(f"  Frames con estimación exitosa: {len(valid)}/{len(h)} "
          f"({100*len(valid)/max(len(h),1):.1f}%)")

    if len(valid) < 10:
        print("\n  RESULTADO: la geometría no se resuelve. El enfoque NO es viable así.")
        return 1

    q1, q3 = np.percentile(valid, [25, 75])
    print(f"\n  ALTURA DE CÁMARA IMPLÍCITA (debería ser constante ~{KITTI_CAM_HEIGHT} m):")
    print(f"     mediana {np.median(valid):.3f} m")
    print(f"     rango intercuartil  {q1:.3f} – {q3:.3f} m")
    print(f"     desviación relativa (IQR/mediana): {100*(q3-q1)/np.median(valid):.1f}%")

    nv = N[np.isfinite(N[:, 0])]
    if len(nv):
        print(f"\n  Normal del plano detectado (marco cámara, Y hacia abajo):")
        print(f"     media  [{nv[:,0].mean():+.3f}, {nv[:,1].mean():+.3f}, {nv[:,2].mean():+.3f}]")
        print(f"     (una normal de suelo ideal sería aproximadamente [0, -1, 0])")

    err = 100 * np.abs(valid - np.median(valid)) / np.median(valid)
    print(f"\n  Si se fijara h = mediana, el error de escala por frame sería:")
    print(f"     mediana {np.median(err):.1f}%   p90 {np.percentile(err,90):.1f}%")

    print(f"\n{'='*88}")
    consistent = (q3 - q1) / np.median(valid) < 0.25
    plausible = 0.8 < np.median(valid) < 3.0
    if consistent and plausible:
        print("  VEREDICTO: la geometría es CONSISTENTE y la altura es PLAUSIBLE.")
        print("             El enfoque merece implementarse.")
    elif plausible:
        print("  VEREDICTO: altura plausible pero MUY DISPERSA. Necesita filtrado")
        print("             temporal y mejor selección de puntos del suelo.")
    else:
        print("  VEREDICTO: la altura implícita no es plausible. Revisar la selección")
        print("             de la región del suelo o la elección de solución.")
    print(f"{'='*88}")

    # ------------------------------------------------------------------
    # The height is constant, so frames whose normal does not look like the
    # road can be dropped: only the reliable measurements are needed.
    # ------------------------------------------------------------------
    print(f"\n{'-'*88}")
    print("FILTRADO POR CALIDAD DE LA NORMAL DEL PLANO")
    print(f"{'-'*88}")
    print(f"{'exigencia sobre la normal':<34}{'frames':>9}{'mediana h':>12}{'IQR rel':>10}{'sesgo':>9}")
    print("-" * 88)

    ny = -N[:, 1]                      # how much it points up (1.0 = a perfect road)
    for thr in (0.0, 0.90, 0.97, 0.99, 0.995):
        sel = np.isfinite(h) & np.isfinite(ny) & (ny >= thr)
        if sel.sum() < 5:
            print(f"{'ny >= ' + format(thr, '.3f'):<34}{sel.sum():>9}{'—':>12}{'—':>10}{'—':>9}")
            continue
        v = h[sel]
        med = np.median(v)
        a, b = np.percentile(v, [25, 75])
        print(f"{'ny >= ' + format(thr, '.3f'):<34}{sel.sum():>9}{med:>11.3f}m"
              f"{100*(b-a)/med:>9.0f}%{100*(med-KITTI_CAM_HEIGHT)/KITTI_CAM_HEIGHT:>8.0f}%")

    np.savez(os.path.join(args.out_dir, f'{seq_name}_ground_probe.npz'),
             h_implied=h, normals=N, n_ground=ng, true_step=true_step)

    os.makedirs(args.out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(h, '.', ms=3, alpha=0.6)
    axes[0].axhline(KITTI_CAM_HEIGHT, c='crimson', ls='--', label=f'altura real ({KITTI_CAM_HEIGHT} m)')
    axes[0].axhline(np.median(valid), c='green', label=f'mediana ({np.median(valid):.2f} m)')
    axes[0].set_xlabel('frame'); axes[0].set_ylabel('altura implícita (m)')
    axes[0].set_ylim(0, 5); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title('Altura de cámara implícita por frame')

    axes[1].hist(valid, bins=60, range=(0, 5), color='steelblue', edgecolor='k', alpha=0.8)
    axes[1].axvline(KITTI_CAM_HEIGHT, c='crimson', ls='--')
    axes[1].set_xlabel('altura implícita (m)'); axes[1].set_ylabel('frecuencia')
    axes[1].set_title('Distribución'); axes[1].grid(alpha=0.3)

    fig.suptitle(f'Sondeo de plano de tierra — {seq_name}')
    fig.tight_layout()
    png = os.path.join(args.out_dir, f'{seq_name}_ground_plane_probe.png')
    fig.savefig(png, dpi=130)
    print(f"\nGráfico: {png}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
