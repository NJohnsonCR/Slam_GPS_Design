"""
PASO 1 - Extracción de "apuntes" (cache de features) de una secuencia KITTI.

Procesa el video UNA sola vez y guarda en un .npz todo lo que producen la cámara
y el GPS en cada frame. El entrenamiento posterior del RL trabaja sobre este
archivo en vez de reprocesar las imágenes, lo que baja un episodio de minutos a
milisegundos.

Lo que se guarda por frame es SOLO lo que no depende de la decisión de fusión
(el peso w). Eso es lo que hace válido el cache: la odometría visual calcula
R y t a partir de las imágenes, sin intervención del peso.

IMPORTANTE: la traslación se guarda CRUDA, tal como sale de recoverPose, antes
de cualquier alineación o escalado con GPS. Esa es la rama de odometría visual
libre que hace falta como baseline de "SLAM tradicional".

Uso:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.rl.extract_features \
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

# El proyecto usa imports relativos a dos raíces distintas: la raíz del repo
# (para LMS.*) y la carpeta LMS_RL_ORB_GPS (para utils.*). Se replican ambas.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LMS_RL_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_LMS_RL_DIR, '..', '..'))
for _p in (_PROJECT_ROOT, _LMS_RL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from LMS.LMS_ORB_with_PG.main import PoseGraphSLAM
from utils.gps.gps_utils import latlon_to_utm

# Calibración de la cámara 02 de KITTI (los mismos valores que usa main.py)
KITTI_FX, KITTI_FY = 718.856, 718.856
KITTI_CX, KITTI_CY = 607.1928, 185.2157

MIN_MATCHES = 15


def find_kitti_dirs(sequence_path):
    """Localiza image_02/data y oxts/data dentro de la secuencia."""
    image_dir = gps_dir = None
    for root, dirs, _files in os.walk(sequence_path):
        if "image_02" in dirs and "data" in os.listdir(os.path.join(root, "image_02")):
            image_dir = os.path.join(root, "image_02", "data")
        if "oxts" in dirs and "data" in os.listdir(os.path.join(root, "oxts")):
            gps_dir = os.path.join(root, "oxts", "data")
    return image_dir, gps_dir


def load_ground_truth(gps_dir, n_frames):
    """
    Lee los archivos OXTS y devuelve la posición verdadera en metros,
    relativa al primer frame (marco local ENU aproximado vía UTM).
    """
    gps_files = sorted(f for f in os.listdir(gps_dir) if f.endswith('.txt'))[:n_frames]
    positions = []
    for fname in gps_files:
        row = np.loadtxt(os.path.join(gps_dir, fname))
        lat, lon, alt = row[0], row[1], row[2]
        positions.append(latlon_to_utm(lat, lon, alt))
    positions = np.array(positions)
    return positions - positions[0]


def extract_visual_odometry(image_dir, n_frames, verbose_every=50):
    """
    Odometría visual libre, frame a frame.

    Devuelve, por cada par de frames consecutivos:
      R_vo        rotación relativa (3x3)
      t_vo        traslación relativa CRUDA, norma unitaria (3,)
      n_matches   cantidad de correspondencias tras el filtro de Lowe
      n_inliers   correspondencias que pasan la verificación geométrica
      ok          si la estimación de pose tuvo éxito

    No se aplica ninguna corrección con GPS. El algoritmo de VO es exactamente
    el del pipeline (mismo ORB, mismo matcher, mismo ratio de Lowe).
    """
    slam = PoseGraphSLAM(fx=KITTI_FX, fy=KITTI_FY, cx=KITTI_CX, cy=KITTI_CY)

    image_files = sorted(
        f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))
    )[:n_frames]

    R_all, t_all = [], []
    n_matches_all, n_inliers_all, ok_all = [], [], []

    prev_kp = prev_des = None

    for i, fname in enumerate(image_files):
        frame = cv2.imread(os.path.join(image_dir, fname))
        if frame is None:
            raise RuntimeError(f"No se pudo leer la imagen: {fname}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = slam.orb_detector.detectAndCompute(gray, None)

        if i > 0:
            R, t = np.eye(3), np.zeros(3)
            n_matches = n_inliers = 0
            ok = False

            if des is not None and prev_des is not None and len(des) > 1 and len(prev_des) > 1:
                matches = slam.filter_matches_lowe_ratio(prev_des, des)
                n_matches = len(matches)

                if n_matches > MIN_MATCHES:
                    pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                    pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])

                    E, _mask = cv2.findEssentialMat(
                        pts_prev, pts_curr, slam.camera_matrix,
                        method=cv2.RANSAC, prob=0.999, threshold=1.0
                    )
                    if E is not None and E.shape == (3, 3):
                        try:
                            n_in, R_est, t_est, _m = cv2.recoverPose(
                                E, pts_prev, pts_curr, slam.camera_matrix
                            )
                            R, t = R_est, t_est.ravel()
                            n_inliers, ok = int(n_in), True
                        except cv2.error:
                            pass

            R_all.append(R)
            t_all.append(t)
            n_matches_all.append(n_matches)
            n_inliers_all.append(n_inliers)
            ok_all.append(ok)

        prev_kp, prev_des = kp, des

        if verbose_every and i % verbose_every == 0:
            print(f"  frame {i}/{len(image_files)}")

    return {
        'R_vo': np.array(R_all),
        't_vo': np.array(t_all),
        'n_matches': np.array(n_matches_all),
        'n_inliers': np.array(n_inliers_all),
        'ok': np.array(ok_all),
        'n_frames': len(image_files),
    }


def integrate_vo(R_rel, t_rel, step_scales=None):
    """
    Encadena las poses relativas para obtener la trayectoria de VO libre.

    step_scales: si se entrega, escala cada paso (la VO monocular no puede
    recuperar la escala métrica por sí sola; es estándar tomarla del ground
    truth para evaluar la calidad de rotación y dirección por separado).
    """
    pose = np.eye(4)
    positions = [pose[:3, 3].copy()]

    for i in range(len(R_rel)):
        s = 1.0 if step_scales is None else step_scales[i]
        rel = np.eye(4)
        rel[:3, :3] = R_rel[i]
        rel[:3, 3] = t_rel[i] * s
        pose = pose @ rel
        positions.append(pose[:3, 3].copy())

    return np.array(positions)


def umeyama_align(traj_est, traj_ref):
    """
    Alineación de Umeyama con escala (misma que usa TrajectoryMetrics).
    Estándar para evaluar VO monocular, que no tiene escala absoluta.
    """
    n = min(len(traj_est), len(traj_ref))
    est, ref = traj_est[:n], traj_ref[:n]

    mu_e, mu_r = est.mean(axis=0), ref.mean(axis=0)
    ec, rc = est - mu_e, ref - mu_r

    U, S, Vt = np.linalg.svd(ec.T @ rc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    var = np.sum(ec ** 2)
    scale = np.sum(S) / var if var > 1e-10 else 1.0

    return scale * (ec @ R.T) + mu_r, scale


def ate(traj_est, traj_ref):
    """Error absoluto de trayectoria tras alineación."""
    aligned, scale = umeyama_align(traj_est, traj_ref)
    n = min(len(aligned), len(traj_ref))
    errors = np.linalg.norm(aligned[:n] - traj_ref[:n], axis=1)
    return {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'max': float(np.max(errors)),
        'scale': float(scale),
    }, aligned


def main():
    parser = argparse.ArgumentParser(description="Paso 1: cache de features de una secuencia KITTI")
    parser.add_argument('sequence', help="Ruta a la secuencia KITTI (…_sync)")
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--out-dir', default='resultados/rl_cache')
    args = parser.parse_args()

    seq_name = os.path.basename(os.path.normpath(args.sequence))
    image_dir, gps_dir = find_kitti_dirs(args.sequence)
    if not image_dir or not gps_dir:
        print(f"ERROR: no se encontró image_02/data y oxts/data en {args.sequence}")
        return 1

    n_avail = min(
        len([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]),
        len([f for f in os.listdir(gps_dir) if f.endswith('.txt')]),
    )
    n_frames = min(args.max_frames, n_avail) if args.max_frames else n_avail

    print("=" * 70)
    print(f"SECUENCIA: {seq_name}  ({n_frames} frames)")
    print("=" * 70)

    print("\n[1/3] Ground truth (OXTS -> UTM, relativo al primer frame)…")
    gt = load_ground_truth(gps_dir, n_frames)
    dist_total = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))
    print(f"  Distancia recorrida: {dist_total:.1f} m")

    print("\n[2/3] Odometría visual libre (sin corrección GPS)…")
    vo = extract_visual_odometry(image_dir, n_frames)
    n_ok = int(vo['ok'].sum())
    print(f"  Poses estimadas OK: {n_ok}/{len(vo['ok'])} ({100 * n_ok / max(len(vo['ok']), 1):.1f}%)")
    print(f"  Matches  — mediana: {np.median(vo['n_matches']):.0f}, mín: {vo['n_matches'].min()}")
    print(f"  Inliers  — mediana: {np.median(vo['n_inliers']):.0f}")

    # Escala por paso tomada del ground truth: la VO monocular no la recupera.
    step_scales = np.linalg.norm(np.diff(gt, axis=0), axis=1)

    traj_unit = integrate_vo(vo['R_vo'], vo['t_vo'])                 # sin escala
    traj_gtsc = integrate_vo(vo['R_vo'], vo['t_vo'], step_scales)    # escala del GT

    print("\n[3/3] Reconstrucción y evaluación del baseline de VO libre…")
    m_unit, aligned_unit = ate(traj_unit, gt)
    m_gtsc, aligned_gtsc = ate(traj_gtsc, gt)

    print(f"  VO sin escala      -> ATE RMSE {m_unit['rmse']:8.2f} m   (escala Umeyama {m_unit['scale']:.3f})")
    print(f"  VO con escala GT   -> ATE RMSE {m_gtsc['rmse']:8.2f} m   (escala Umeyama {m_gtsc['scale']:.3f})")

    os.makedirs(args.out_dir, exist_ok=True)
    npz_path = os.path.join(args.out_dir, f"{seq_name}_cache.npz")
    np.savez_compressed(
        npz_path,
        sequence=seq_name,
        R_vo=vo['R_vo'], t_vo=vo['t_vo'],
        n_matches=vo['n_matches'], n_inliers=vo['n_inliers'], ok=vo['ok'],
        gt=gt, step_scales=step_scales,
    )
    print(f"\n  Cache guardado: {npz_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, traj, m, title in (
        (axes[0], aligned_unit, m_unit, "VO libre SIN escala"),
        (axes[1], aligned_gtsc, m_gtsc, "VO libre CON escala del GT"),
    ):
        ax.plot(gt[:, 0], gt[:, 1], 'k-', lw=2, label='Ground truth (OXTS)')
        ax.plot(traj[:, 0], traj[:, 1], 'r--', lw=1.5, label='Odometría visual')
        ax.set_title(f"{title}\nATE RMSE = {m['rmse']:.2f} m")
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.axis('equal'); ax.grid(alpha=0.3); ax.legend()

    fig.suptitle(f"{seq_name} — baseline de odometría visual libre ({dist_total:.0f} m)")
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, f"{seq_name}_vo_baseline.png")
    fig.savefig(png_path, dpi=130)
    print(f"  Gráfico guardado: {png_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
