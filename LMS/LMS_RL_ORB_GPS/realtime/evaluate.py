"""
Evaluate a processed session against the phone's own sensors.

Reads the per-frame results of run_replay.py --save-frames (or of a live
session) and the recording made by the app, and measures:

    scale     camera speed against the GPS Doppler speed, by speed band
    rotation  VO rotation against the gyroscope, per frame and per 0.5 s
    heading   accumulated heading error, split into stopped and moving time
    shape     ATE after a rigid horizontal alignment, and drift from a common
              start anchored at the first metres of motion

The phone GPS is not ground truth (several metres of error), but outdoors it
is a fair reference for shape and distance. The gyroscope is only a measuring
instrument: it never enters the system.

Usage:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.evaluate \
        resultados/realtime/base_14_05/frames_deterministic_metric.csv \
        --recording mobile_data/2026_09_30_14_05_19
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.signal import butter, filtfilt

_THIS = os.path.dirname(os.path.abspath(__file__))
_LMS_RL = os.path.abspath(os.path.join(_THIS, ".."))
_ROOT = os.path.abspath(os.path.join(_LMS_RL, "..", ".."))
for _p in (_ROOT, _LMS_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realtime.run_replay import (COLOR_ALIGNED, COLOR_CAMERA, COLOR_GPS,
                                 camera_positions, umeyama)

STOP_SPEED = 0.5            # m/s of GPS Doppler speed below which the car is stopped
WINDOW_S = 0.5              # the gyro fit needs windows: per frame, VO noise ~ signal
SPEED_BANDS = (0, 2, 4, 6, 8, 10, 12, 14, 18, 25)
ANCHOR_SPEED = 3.0          # m/s: anchor where the car is clearly moving
ANCHOR_METRES = 20.0        # GPS course measured over this distance


# ------------------------------------------------------------------- loading

def load_frames(path):
    fr = pd.read_csv(path)
    R = fr[[f"r{i}{j}" for i in range(3) for j in range(3)]].to_numpy(float)
    return (fr["t_ns"].to_numpy(np.int64), R.reshape(-1, 3, 3),
            fr[["tx", "ty", "tz"]].to_numpy(float), fr["scale_m"].to_numpy(float))


def load_gps(rec_dir, t_from, t_to):
    """Fixes inside the processed span, in metres from its first fix."""
    loc = pd.read_csv(os.path.join(rec_dir, "location.csv"))
    t = loc.iloc[:, 0].to_numpy(np.int64)
    keep = (t >= t_from) & (t <= t_to)
    lat, lon = loc.iloc[:, 1].to_numpy()[keep], loc.iloc[:, 2].to_numpy()[keep]
    # The UTM zone is fixed by the first fix, as in LiveGpsSource.
    zone = int((lon[0] + 180) / 6) + 1
    epsg = (32600 if lat[0] >= 0 else 32700) + zone
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    xy = np.column_stack([x - x[0], y - y[0]])
    alt = loc.iloc[:, 3].to_numpy()[keep]
    return t[keep], xy, alt, loc.iloc[:, 4].to_numpy()[keep]


def load_imu(rec_dir):
    path = os.path.join(rec_dir, "gyro_accel.csv")
    if not os.path.exists(path):
        return None
    g = pd.read_csv(path)
    return (g.iloc[:, 0].to_numpy(np.int64), g.iloc[:, 1:4].to_numpy(float),
            g.iloc[:, 4:7].to_numpy(float))


# ------------------------------------------------------------------ geometry

def gyro_steps(t_imu, w, t_frames):
    """Gyro rotation vector (device axes) between consecutive frame stamps."""
    ts = (t_imu - t_imu[0]) / 1e9
    cum = np.vstack([np.zeros(3),
                     np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(ts)[:, None], axis=0)])
    tf = (t_frames - t_imu[0]) / 1e9
    at = np.column_stack([np.interp(tf, ts, cum[:, j]) for j in range(3)])
    return np.diff(at, axis=0)


def kabsch(src, dst):
    """Rotation X minimising sum |dst - X src|^2."""
    U, _, Vt = np.linalg.svd(src.T @ dst)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def fit_imu_to_camera(r_cam, phi, t_steps):
    """
    IMU -> camera axis mapping, fitted on 0.5 s windows and snapped to the
    nearest signed permutation (the phone's axes are aligned by construction).
    Returns the mapping and how far the free fit is from it, in degrees.
    """
    win = ((t_steps - t_steps[0]) / 1e9 // WINDOW_S).astype(int)
    A, B = [], []
    for k in np.unique(win):
        m = win == k
        if m.sum() >= 5:
            A.append(phi[m].sum(0))
            B.append(r_cam[m].sum(0))
    A, B = np.array(A), np.array(B)
    keep = np.ones(len(A), bool)
    for _ in range(3):
        X = kabsch(A[keep], B[keep])
        res = np.linalg.norm(B - A @ X.T, axis=1)
        keep = res < 2 * np.median(res)
    P = np.zeros((3, 3))
    for i in range(3):
        j = np.argmax(np.abs(X[i]))
        P[i, j] = np.sign(X[i, j])
    if not (np.allclose(np.abs(P).sum(0), 1) and np.isclose(np.linalg.det(P), 1)):
        return X, float("nan"), A, B
    off = float(np.degrees(np.linalg.norm(cv2.Rodrigues(P.T @ X)[0])))
    return P, off, A, B


def horizontal_frame(up):
    """2D basis of the plane normal to 'up', handed like (east, north, up)."""
    f = np.array([0.0, 0.0, 1.0]) - up[2] * up       # camera forward, levelled
    e1 = f / np.linalg.norm(f)
    return np.vstack([e1, np.cross(up, e1)])


def interp_xy(t_query, t, xy):
    return np.column_stack([np.interp(t_query, t, xy[:, j]) for j in range(xy.shape[1])])


def stopped_spans(t_s, stopped):
    """[start, end] seconds of each stretch flagged as stopped."""
    spans, start = [], None
    for ti, s in zip(t_s, stopped):
        if s and start is None:
            start = ti
        elif not s and start is not None:
            spans.append((start, ti))
            start = None
    if start is not None:
        spans.append((start, t_s[-1]))
    return spans


# ------------------------------------------------------------------ evaluation

def evaluate(frames_path, rec_dir):
    t, R, tv, s = load_frames(frames_path)
    metric = np.isfinite(s).any()
    t0 = t[0]
    ts = (t - t0) / 1e9
    tg, xy, alt, v_dop = load_gps(rec_dir, t[0], t[-1])
    tgs = (tg - t0) / 1e9
    v_gps = np.interp(ts, tgs, v_dop)
    stopped = v_gps < STOP_SPEED
    out = {"frames": len(t), "duracion_s": float(ts[-1]), "metrica": bool(metric)}

    # Rotation of the camera at each step: R maps points, so it is R^T.
    r_cam = np.zeros((len(t), 3))
    for k in range(1, len(t)):
        r_cam[k] = -cv2.Rodrigues(R[k])[0].ravel()

    imu = load_imu(rec_dir)
    up = None
    if imu is not None:
        t_imu, w, acc = imu
        ok = np.zeros(len(t), bool)
        ok[1:] = (t[:-1] >= t_imu[0]) & (t[1:] <= t_imu[-1])
        idx = np.flatnonzero(ok)
        phi = np.zeros((len(t), 3))
        phi[1:] = gyro_steps(t_imu, w, t)
        X, off, A, B = fit_imu_to_camera(r_cam[idx], phi[idx], t[idx])
        pred = phi @ X.T
        err = np.degrees(np.linalg.norm(r_cam[idx] - pred[idx], axis=1))
        err_w = np.degrees(np.linalg.norm(B - A @ X.T, axis=1))

        # Gravity (low-passed accelerometer) gives the vertical, in camera axes.
        b, a = butter(2, 0.3 / (0.5 * len(t_imu) / ((t_imu[-1] - t_imu[0]) / 1e9)))
        g = np.column_stack([filtfilt(b, a, acc[:, j]) for j in range(3)])
        up = np.column_stack([np.interp(t, t_imu, g[:, j]) for j in range(3)]) @ X.T
        up /= np.linalg.norm(up, axis=1, keepdims=True)

        yaw_err = np.zeros(len(t))
        yaw_err[idx] = np.degrees(np.sum((r_cam[idx] - pred[idx]) * up[idx], axis=1))
        heading_err = np.cumsum(yaw_err)
        out["rotacion"] = {
            "mapeo_imu_camara": X.round(3).tolist(), "ajuste_libre_deg": off,
            "error_frame_p50": float(np.median(err)), "error_frame_p90": float(np.percentile(err, 90)),
            "error_frame_p99": float(np.percentile(err, 99)),
            "rotacion_real_p50": float(np.degrees(np.median(np.linalg.norm(phi[idx], axis=1)))),
            "error_ventana_p50": float(np.median(err_w)), "error_ventana_p90": float(np.percentile(err_w, 90)),
            "rumbo_final_deg": float(heading_err[-1]),
            "rumbo_max_deg": float(np.max(np.abs(heading_err))),
            "rumbo_en_altos_deg": float(yaw_err[stopped].sum()),
            "rumbo_en_marcha_deg": float(yaw_err[~stopped].sum()),
        }
        out["_heading"] = heading_err

    # Speed: s is metres per step, so s / dt is the speed the camera used.
    if metric:
        dt = np.diff(ts, prepend=np.nan)
        v_cam = s / dt
        bands = []
        for lo, hi in zip(SPEED_BANDS[:-1], SPEED_BANDS[1:]):
            m = (v_gps >= max(lo, STOP_SPEED)) & (v_gps < hi) & np.isfinite(v_cam)
            if m.sum() >= 30:
                ratio = v_cam[m] / v_gps[m]
                bands.append((lo, hi, int(m.sum()), float(np.median(ratio)),
                              float(np.median(np.abs(ratio - 1)))))
        cam_stop = v_cam[stopped & np.isfinite(v_cam)]
        out["escala"] = {
            "bandas": bands,
            "detenido_frames": int(stopped.sum()),
            "detenido_vel_camara_p50": float(np.median(cam_stop)) if len(cam_stop) else None,
            "distancia_camara_m": float(np.nansum(s)),
            "distancia_gps_m": float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))),
            "distancia_doppler_m": float(np.sum(v_gps[1:] * np.diff(ts))),
        }
        out["_v"] = (v_cam, v_gps)

    # Shape, in the horizontal plane.
    pos = camera_positions(list(zip(R, tv)), s if metric else None)
    if up is not None:
        u0 = up[0]
    else:
        # Without IMU: the plane that best fits the route.
        u0 = np.linalg.svd(pos - pos.mean(0))[2][-1]
    H = horizontal_frame(u0)
    vo2 = pos @ H.T
    height = pos @ u0
    in_span = (tgs >= 0) & (tgs <= ts[-1])
    vo_at_fix = interp_xy(tgs[in_span], ts, vo2)
    gps_at_fix = xy[in_span]
    aligned, scale = umeyama(vo_at_fix, gps_at_fix, with_scale=not metric)
    dev = np.linalg.norm(aligned - gps_at_fix, axis=1)
    out["forma"] = {"ate_rmse_m": float(np.sqrt(np.mean(dev ** 2))),
                    "ate_mediana_m": float(np.median(dev)), "ate_max_m": float(dev.max()),
                    "altura_camara_m": [float(height.min()), float(height.max())],
                    "desnivel_gps_m": float(np.ptp(alt))}
    out["_maps"] = {"gps": xy, "aligned": aligned, "t_fix": tgs[in_span], "dev": dev}

    if metric:
        moving = np.flatnonzero(in_span & (v_dop >= ANCHOR_SPEED))
        if len(moving):
            ia = moving[0]
            dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy[ia:], axis=0), axis=1))])
            later = np.flatnonzero(dist >= ANCHOR_METRES)
            if len(later):
                ib = ia + later[0]
                pa, pb = interp_xy([tgs[ia], tgs[ib]], ts, vo2)
                course, d_vo = xy[ib] - xy[ia], pb - pa
                ang = np.arctan2(course[1], course[0]) - np.arctan2(d_vo[1], d_vo[0])
                rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
                anchored = (vo2 - pa) @ rot.T + xy[ia]
                after = in_span & (tgs >= tgs[ia])
                drift = np.linalg.norm(interp_xy(tgs[after], ts, anchored) - xy[after], axis=1)
                t_after = tgs[after] - tgs[ia]
                travelled = np.sum(np.linalg.norm(np.diff(xy[after], axis=0), axis=1))
                marks = {}
                for mark in (30, 60, 120):
                    if t_after[-1] >= mark:
                        marks[f"{mark}s"] = float(np.interp(mark, t_after, drift))
                out["forma"]["anclado_desde_s"] = float(tgs[ia])
                out["forma"]["deriva_m"] = marks
                out["forma"]["deriva_final_m"] = float(drift[-1])
                out["forma"]["deriva_final_pct"] = float(100 * drift[-1] / max(travelled, 1e-9))
                out["_maps"]["anchored"] = anchored
                out["_maps"]["drift"] = (tgs[after], drift)
    out["_t"] = ts
    out["_stops"] = stopped_spans(ts, stopped)
    return out


# ------------------------------------------------------------------ report

def print_report(r, frames_path, rec_dir):
    print("=" * 78)
    print("EVALUACIÓN CONTRA LOS SENSORES DEL TELÉFONO")
    print("=" * 78)
    print(f"  Resultados: {frames_path}")
    print(f"  Grabación:  {rec_dir}")
    print(f"  Tramo: {r['duracion_s']:.0f} s, {r['frames']} frames procesados")

    if "escala" in r:
        e = r["escala"]
        print("\nESCALA (velocidad de la cámara contra la velocidad Doppler del GPS)")
        print("  banda (m/s)   frames   cámara/GPS p50   error p50")
        for lo, hi, n, ratio, err in e["bandas"]:
            print(f"  {lo:4.0f}-{hi:<4.0f}    {n:6d}   {ratio:10.2f}       {100 * err:5.1f} %")
        if e["detenido_vel_camara_p50"] is not None:
            print(f"  Detenido según el GPS: {e['detenido_frames']} frames; la cámara "
                  f"dice {e['detenido_vel_camara_p50']:.2f} m/s (p50)")
        print(f"  Distancia: cámara {e['distancia_camara_m']:.0f} m | GPS {e['distancia_gps_m']:.0f} m "
              f"| Doppler integrado {e['distancia_doppler_m']:.0f} m")

    if "rotacion" in r:
        o = r["rotacion"]
        print("\nROTACIÓN (contra el giroscopio)")
        print(f"  Mapeo IMU -> cámara {o['mapeo_imu_camara']}; el ajuste libre "
              f"difiere {o['ajuste_libre_deg']:.1f}°")
        print(f"  Por frame:   error p50 {o['error_frame_p50']:.2f}°, p90 {o['error_frame_p90']:.2f}°, "
              f"p99 {o['error_frame_p99']:.1f}°  (rotación real p50 {o['rotacion_real_p50']:.2f}°)")
        print(f"  Cada 0.5 s:  error p50 {o['error_ventana_p50']:.2f}°, p90 {o['error_ventana_p90']:.2f}°")
        print(f"  Rumbo: error acumulado al final {o['rumbo_final_deg']:+.0f}°, máximo "
              f"{o['rumbo_max_deg']:.0f}°; aportan los altos {o['rumbo_en_altos_deg']:+.0f}° "
              f"y la marcha {o['rumbo_en_marcha_deg']:+.0f}°")

    f = r["forma"]
    print("\nFORMA (contra el GPS, en el plano horizontal)")
    print(f"  ATE con alineación {'rígida' if r['metrica'] else 'con escala'}: RMSE "
          f"{f['ate_rmse_m']:.1f} m, mediana {f['ate_mediana_m']:.1f} m, máximo {f['ate_max_m']:.1f} m")
    if "deriva_m" in f:
        marks = ", ".join(f"{k}: {v:.1f} m" for k, v in f["deriva_m"].items())
        print(f"  Deriva desde un inicio común (anclado en el segundo {f['anclado_desde_s']:.0f}): "
              f"{marks}; al final {f['deriva_final_m']:.1f} m ({f['deriva_final_pct']:.0f} % de lo recorrido)")
    print(f"  Altura de la cámara: de {f['altura_camara_m'][0]:+.1f} a {f['altura_camara_m'][1]:+.1f} m "
          f"(desnivel según el GPS: {f['desnivel_gps_m']:.0f} m)")
    print("  NOTA: el GPS del teléfono no es ground truth; al aire libre sirve de referencia.")


def plot_report(r, png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = r["_maps"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    a = ax[0, 0]
    a.plot(m["gps"][:, 0], m["gps"][:, 1], "-", lw=2, color=COLOR_GPS, label="GPS del teléfono")
    if "anchored" in m:
        a.plot(m["anchored"][:, 0], m["anchored"][:, 1], "--", lw=1.8, color=COLOR_CAMERA,
               label="Cámara, desde un inicio común")
    a.plot(m["aligned"][:, 0], m["aligned"][:, 1], ":", lw=1.8, color=COLOR_ALIGNED,
           label="Cámara, alineada (ATE)")
    a.plot(m["gps"][0, 0], m["gps"][0, 1], "o", ms=8, color="#0b0b0b", label="Inicio")
    a.set_xlabel("Este (m)"); a.set_ylabel("Norte (m)"); a.set_title("Recorrido")
    a.axis("equal"); a.grid(alpha=0.3); a.legend(loc="best", fontsize=9)

    def shade(axis):
        for i, (s0, s1) in enumerate(r["_stops"]):
            axis.axvspan(s0, s1, color="#9a9a96", alpha=0.25, lw=0,
                         label="Detenido" if i == 0 else None)

    a = ax[0, 1]
    if "_v" in r:
        v_cam, v_gps = r["_v"]
        a.plot(r["_t"], v_gps, "-", lw=2, color=COLOR_GPS, label="GPS (Doppler)")
        a.plot(r["_t"], v_cam, "--", lw=1.5, color=COLOR_CAMERA, label="Cámara (profundidad)")
        shade(a)
        a.set_ylabel("velocidad (m/s)"); a.legend(loc="best", fontsize=9)
    a.set_xlabel("tiempo (s)"); a.set_title("Velocidad"); a.grid(alpha=0.3)

    a = ax[1, 0]
    if "_heading" in r:
        a.plot(r["_t"], r["_heading"], "-", lw=2, color=COLOR_CAMERA, label="Cámara menos giroscopio")
        a.axhline(0, color="#52514e", lw=1)
        shade(a)
        a.set_ylabel("error de rumbo acumulado (°)"); a.legend(loc="best", fontsize=9)
    a.set_xlabel("tiempo (s)"); a.set_title("Rumbo contra el giroscopio"); a.grid(alpha=0.3)

    a = ax[1, 1]
    if "drift" in m:
        a.plot(*m["drift"], "--", lw=2, color=COLOR_CAMERA, label="Desde un inicio común")
    a.plot(m["t_fix"], m["dev"], ":", lw=2, color=COLOR_ALIGNED, label="Alineada (ATE)")
    shade(a)
    a.set_xlabel("tiempo (s)"); a.set_ylabel("distancia al GPS (m)")
    a.set_title("Separación respecto del GPS"); a.grid(alpha=0.3); a.legend(loc="best", fontsize=9)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Evaluación de una sesión contra los sensores del teléfono")
    ap.add_argument("frames", help="frames.csv de run_live.py o frames_*.csv de run_replay.py")
    ap.add_argument("--recording", required=True,
                    help="Carpeta de la grabación del teléfono (location.csv, gyro_accel.csv)")
    ap.add_argument("--out", default=None, help="Por defecto, la carpeta del archivo de frames")
    args = ap.parse_args()

    r = evaluate(args.frames, args.recording)
    print_report(r, args.frames, args.recording)

    out_dir = args.out or os.path.dirname(os.path.abspath(args.frames))
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "evaluation.png")
    plot_report(r, png, f"Evaluación — {os.path.basename(os.path.normpath(args.recording))}")
    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump({k: v for k, v in r.items() if not k.startswith("_")}, f, indent=2)
    print(f"\n  Gráfico: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
