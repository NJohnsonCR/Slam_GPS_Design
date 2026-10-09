"""
Evaluate a processed session against the phone's own sensors.

Reads the per-frame results of run_replay.py --save-frames (or of a live
session) and the recording made by the app, and measures:

    scale     camera speed against the GPS Doppler speed, by speed band
    rotation  VO rotation against the gyroscope, per frame and per 0.5 s
    heading   accumulated heading error, split into stopped and moving time
    shape     ATE after a rigid horizontal alignment, and drift from a common
              start anchored at the first metres of motion
    fusion    with --fusion, the filter (camera, gyroscope and GPS) against the
              fixes it had not seen yet, and during simulated GPS outages
              (--outage); --sweep compares its variants on an outage every 10 s

The phone GPS is not ground truth (several metres of error), but outdoors it
is a fair reference for shape and distance. The gyroscope measures the
camera's rotation; in the fusion it also gives the heading, so there the
reference is the GPS after each simulated outage.

Usage:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.evaluate \
        resultados/realtime/base_14_05/frames_deterministic_metric.csv \
        --recording mobile_data/2026_09_30_14_05_19

    # with the fusion, and the GPS cut for 30 s from second 120
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.evaluate \
        resultados/realtime/mask_vote_14_05/frames_deterministic_metric.csv \
        --recording mobile_data/2026_09_30_14_05_19 --fusion --outage 120 30

    # the variants of the filter, on outages of 10 to 120 s every 10 s
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.evaluate \
        resultados/realtime/full_14_05_compuerta/frames_deterministic_metric.csv \
        --recording mobile_data/2026_09_30_14_05_19 --sweep
"""

import argparse
import copy
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

_THIS = os.path.dirname(os.path.abspath(__file__))
_LMS_RL = os.path.abspath(os.path.join(_THIS, ".."))
_ROOT = os.path.abspath(os.path.join(_LMS_RL, "..", ".."))
for _p in (_ROOT, _LMS_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realtime.fusion import (GpsCameraFusion, PlanarEKF, feed_frame, gyro_inputs, run_fusion,
                             session_inputs)
from realtime.sources import utm_zone
from realtime.run_replay import (COLOR_ALIGNED, COLOR_CAMERA, COLOR_FUSED, COLOR_GPS,
                                 camera_positions, umeyama)

STOP_SPEED = 0.5            # m/s of GPS Doppler speed below which the car is stopped
WINDOW_S = 0.5              # the gyro fit needs windows: per frame, VO noise ~ signal
SPEED_BANDS = (0, 2, 4, 6, 8, 10, 12, 14, 18, 25)
ANCHOR_SPEED = 3.0          # m/s: anchor where the car is clearly moving
ANCHOR_METRES = 20.0        # GPS course measured over this distance
# The filter's variants that --sweep compares; the first is the filter as it
# was before the gyroscope. FUSION is the one used everywhere else.
VARIANTS = (
    ("cámara, sin velocidad como estado", dict(gyro=False, speed_state=False, bias=False)),
    ("cámara, con velocidad como estado", dict(gyro=False, speed_state=True, bias=False)),
    ("giroscopio, sin velocidad como estado", dict(gyro=True, speed_state=False, bias=False)),
    ("giroscopio + sesgo, sin velocidad como estado", dict(gyro=True, speed_state=False, bias=True)),
    ("giroscopio, con velocidad como estado", dict(gyro=True, speed_state=True, bias=False)),
    ("giroscopio + sesgo, con velocidad como estado", dict(gyro=True, speed_state=True, bias=True)),
)
FUSION = dict(gyro=True, speed_state=True, bias=False)
SWEEP_LENGTHS = (10, 20, 30, 60, 120)


# ------------------------------------------------------------------- loading

def load_frames(path):
    fr = pd.read_csv(path)
    R = fr[[f"r{i}{j}" for i in range(3) for j in range(3)]].to_numpy(float)
    return (fr["t_ns"].to_numpy(np.int64), R.reshape(-1, 3, 3),
            fr[["tx", "ty", "tz"]].to_numpy(float), fr["scale_m"].to_numpy(float))


def load_gps(rec_dir, t_from, t_to):
    """
    Fixes inside the processed span, in metres from its first fix. Reads the
    phone's location.csv or, from a live session folder, gps.csv (same columns).
    """
    path = os.path.join(rec_dir, "location.csv")
    if not os.path.exists(path):
        path = os.path.join(rec_dir, "gps.csv")
    loc = pd.read_csv(path)
    t = loc.iloc[:, 0].to_numpy(np.int64)
    keep = (t >= t_from) & (t <= t_to)
    lat, lon = loc.iloc[:, 1].to_numpy()[keep], loc.iloc[:, 2].to_numpy()[keep]
    x, y = utm_zone(lat[0], lon[0])[1].transform(lon, lat)
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
        # The depth estimator reports exactly 0 m/s when it sees the car stopped.
        cam_still = s == 0
        moving = v_gps > 1.0
        out["escala"] = {
            "bandas": bands,
            "detenido_frames": int(stopped.sum()),
            "detenido_vel_camara_p50": float(np.median(cam_stop)) if len(cam_stop) else None,
            "camara_detenida_con_gps_detenido_pct": float(100 * cam_still[stopped].mean()) if stopped.any() else None,
            "camara_detenida_en_marcha_pct": float(100 * cam_still[moving].mean()) if moving.any() else None,
            "metros_perdidos_por_parada_falsa": float(np.sum((v_gps * np.nan_to_num(dt))[moving & cam_still])),
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


# ------------------------------------------------------------------ fusion

def gps_alone(tg, xy, vd, i, t_query):
    """
    What the GPS alone would say after losing signal at fix i: that fix held,
    and that fix carried on at its Doppler speed along the last course.
    """
    course = xy[i] - xy[i - 1]
    u = course / max(np.linalg.norm(course), 1e-9)
    dt = (np.asarray(t_query) - tg[i]) / 1e9
    return np.repeat(xy[i][None], len(dt), axis=0), xy[i] + (vd[i] * dt)[:, None] * u


def before_each(t, est, t_query):
    """The estimate at the last frame before each instant, NaN if none."""
    k = np.searchsorted(t, t_query) - 1
    return np.where((k >= 0)[:, None], est[np.maximum(k, 0), :2], np.nan)


def fusion_inputs(frames_path, rec_dir):
    """
    What the filter is fed: frame times, the camera inputs, the fixes as
    (t_ns, xy, speed), the same fixes as arrays and the gyroscope's yaw per
    frame (None if the recording has no gyro_accel.csv).
    """
    t, R, tv, s = load_frames(frames_path)
    tg, xy, _, vd = load_gps(rec_dir, t[0], t[-1])
    imu = load_imu(rec_dir)
    gyro_yaw = None if imu is None else gyro_inputs(*imu, t)
    fixes = [(int(a), b, float(c)) for a, b, c in zip(tg, xy, vd)]
    return t, session_inputs(R, tv, s), fixes, (tg, xy, vd), gyro_yaw


def drift_from(t, est, k0, tg, xy):
    """Distance to the GPS of a run that used it only to start, at frame k0."""
    after = tg > t[k0]
    drift = np.linalg.norm(before_each(t, est, tg[after]) - xy[after], axis=1)
    t_after = (tg[after] - t[k0]) / 1e9
    travelled = np.sum(np.linalg.norm(np.diff(xy[after], axis=0), axis=1))
    return {"deriva_m": {f"{m}s": float(np.interp(m, t_after, drift))
                         for m in (30, 60, 120) if t_after[-1] >= m},
            "deriva_final_m": float(drift[-1]),
            "deriva_final_pct": float(100 * drift[-1] / max(travelled, 1e-9))}


def evaluate_fusion(frames_path, rec_dir, outages=(), modes=FUSION):
    """
    The fusion measured against fixes it had not used yet: the position it
    predicts just before each one, the visual modes from the same start (the
    GPS cut right after starting: camera alone, and camera with gyroscope)
    and each simulated outage, given as (start, length) in seconds from the
    first frame.
    """
    t, inputs, fixes, (tg, xy, vd), gyro_yaw = fusion_inputs(frames_path, rec_dir)
    ts, tgs = (t - t[0]) / 1e9, (tg - t[0]) / 1e9

    def run(**kw):
        return run_fusion(t, inputs, fixes, **{"gyro_yaw": gyro_yaw, **modes, **kw})

    fus, est, sig = run()
    ready = np.flatnonzero(np.isfinite(est[:, 0]))
    if not len(ready):
        return None
    k0 = ready[0]
    out = {"modo": dict(modes, gyro=fus.gyro), "inicio_s": float(ts[k0]),
           "s_final": float(est[-1, 3]), "s_p5_p95": np.percentile(est[ready, 3], [5, 95]).tolist()}
    if fus.n_gyro + fus.n_camera_yaw:
        out["frames_con_giroscopio_pct"] = 100 * fus.n_gyro / (fus.n_gyro + fus.n_camera_yaw)
    if "b" in fus.ekf.i:
        b = np.degrees(est[ready, 5])
        out["sesgo_final_deg_s"] = float(b[-1])
        out["sesgo_p5_p95_deg_s"] = np.percentile(b, [5, 95]).tolist()

    out["correcciones"] = {}
    for kind, label in (("pos", "posicion"), ("speed", "velocidad"), ("course", "curso"),
                        ("camera", "camara"), ("stop", "alto")):
        rows = [r for r in fus.log if r[1] == kind and r[2] != "out_of_range"]
        out["correcciones"][label] = {
            "n": len(rows),
            "rechazadas": sum(r[2] == "rejected" for r in rows),
            "forzadas": sum(r[2] == "forced" for r in rows),
            "fuera_de_rango": sum(r[1] == kind and r[2] == "out_of_range" for r in fus.log),
            "nis_medio": float(np.mean([r[3] for r in rows])) if rows else None}

    # What the course between fixes adds: the same run without it.
    _, plain, plain_sig = run(use_course=False)
    both = np.isfinite(est[:, 2]) & np.isfinite(plain[:, 2])
    diff = np.degrees(np.abs(np.angle(np.exp(1j * (est[both, 2] - plain[both, 2])))))
    out["rumbo_con_y_sin_curso"] = {
        "diferencia_p50_deg": float(np.median(diff)),
        "diferencia_p95_deg": float(np.percentile(diff, 95)),
        "diferencia_max_deg": float(diff.max()),
        "incertidumbre_p50_con_deg": float(np.degrees(np.median(sig[both, 2]))),
        "incertidumbre_p50_sin_deg": float(np.degrees(np.median(plain_sig[both, 2])))}
    pos = [r for r in fus.log if r[1] == "pos"]
    pred_err = np.array([np.linalg.norm(r[4]) for r in pos])
    out["prediccion_p50_m"] = float(np.median(pred_err))
    out["prediccion_p95_m"] = float(np.percentile(pred_err, 95))

    # The visual-only modes: the same start, and no GPS after it.
    _, cam, _ = run(visual_only=True, gyro_yaw=None)
    out["camara_sola"] = drift_from(t, cam, k0, tg, xy)
    cam_gyro = None
    if gyro_yaw is not None:
        _, cam_gyro, _ = run(visual_only=True, gyro=True)
        out["camara_giroscopio"] = drift_from(t, cam_gyro, k0, tg, xy)

    cuts, curves = [], []
    for start, length in outages:
        t_from, t_to = t[0] + int(start * 1e9), t[0] + int((start + length) * 1e9)
        i0 = np.searchsorted(tg, t_from) - 1          # last fix before the outage
        i_end = np.searchsorted(tg, t_to, side="right")   # first fix after it
        if t_from < t[k0] or i0 < 1 or i_end >= len(tg):
            print(f"  (corte en el segundo {start:.0f}: fuera del tramo con fusión, se omite)")
            continue
        _, cut, _ = run(outages=[(t_from, t_to)])
        idx = np.arange(i0 + 1, i_end + 1)
        hold, carried = gps_alone(tg, xy, vd, i0, tg[idx])
        errs = [np.linalg.norm(p - xy[idx], axis=1)
                for p in (before_each(t, cut, tg[idx]), carried, hold)]
        cuts.append({"inicio_s": float(start), "duracion_s": float((tg[i_end] - tg[i0]) / 1e9),
                     "recorrido_m": float(np.sum(np.linalg.norm(np.diff(xy[i0:i_end + 1], axis=0), axis=1))),
                     "error_fusion_m": float(errs[0][-1]),
                     "error_gps_velocidad_constante_m": float(errs[1][-1]),
                     "error_gps_congelado_m": float(errs[2][-1])})
        span = (t >= tg[i0]) & (t <= tg[i_end])
        curves.append({"t": (tg[idx] - tg[i0]) / 1e9, "errs": errs, "track": cut[span, :2]})
    out["cortes"] = cuts

    # What s should be at each fix: Doppler over the camera's speed.
    stopped = np.interp(ts, tgs, vd) < STOP_SPEED
    vcam_fix = np.interp(tg, t, np.r_[0.0, inputs[1][1:] / np.diff(ts)])
    ok = (vd >= GpsCameraFusion.MIN_SPEED) & (vcam_fix > 0)
    out["_fus"] = {"t": ts, "est": est, "sig": sig, "gps": xy, "cam": cam, "cam_gyro": cam_gyro,
                   "k0": k0, "ratio": (tgs[ok], vd[ok] / vcam_fix[ok]),
                   "pred": (np.array([(r[0] - t[0]) / 1e9 for r in pos]), pred_err),
                   "curves": curves, "stops": stopped_spans(ts, stopped)}
    return out


def sweep_outages(t, inputs, fixes, gyro_yaw=None, lengths=SWEEP_LENGTHS, every_s=10.0, **modes):
    """
    A simulated GPS outage every every_s seconds and of each length, all along
    the session: the error when the GPS returns, and that of the GPS carried
    at its last speed. Each outage continues a copy of the run with GPS from
    its start: the same filter as run_fusion(outages=...), at a fraction of
    the cost.
    """
    tg = np.array([f[0] for f in fixes], np.int64)
    xy = np.array([f[1] for f in fixes])
    vd = np.array([f[2] for f in fixes])
    if gyro_yaw is None:
        modes["gyro"] = False
    fus = GpsCameraFusion(**modes)
    found = {length: [] for length in lengths}
    every, next_t, j = int(every_s * 1e9), None, 0
    for k in range(len(t)):
        if fus.ready:
            next_t = t[k] + every if next_t is None else next_t
            if t[k] >= next_t:
                log, fus.log = fus.log, []          # the copies need no history
                for length in lengths:
                    r = _outage(copy.deepcopy(fus), k, j, next_t, length, t, inputs, gyro_yaw,
                                fixes, tg, xy, vd)
                    if r is not None:
                        found[length].append(r)
                fus.log = log
                next_t += every
        j = feed_frame(fus, k, t, inputs, gyro_yaw, fixes, j)
    return found


def _outage(fus, k, j, t_from, length, t, inputs, gyro_yaw, fixes, tg, xy, vd):
    """One outage of sweep_outages, continued from frame k; None past the last fix."""
    t_to = t_from + int(length * 1e9)
    i0 = np.searchsorted(tg, t_from) - 1                # last fix before the outage
    i_end = np.searchsorted(tg, t_to, side="right")     # first fix after it
    if i0 < 1 or i_end >= len(tg):
        return None
    while k < len(t) and t[k] < tg[i_end]:
        j = feed_frame(fus, k, t, inputs, gyro_yaw, fixes, j, [(t_from, t_to)])
        k += 1
    _, carried = gps_alone(tg, xy, vd, i0, tg[[i_end]])
    return {"inicio_s": float((t_from - t[0]) / 1e9),
            "error_m": float(np.linalg.norm(fus.ekf.x[:2] - xy[i_end])),
            "error_gps_velocidad_constante_m": float(np.linalg.norm(carried[0] - xy[i_end])),
            "recorrido_m": float(np.sum(np.linalg.norm(np.diff(xy[i0:i_end + 1], axis=0), axis=1)))}


def summarize_sweep(found):
    """Per length: p50 and p90 of both errors, and how often the filter wins."""
    out = {}
    for length, rows in found.items():
        e = np.array([r["error_m"] for r in rows])
        c = np.array([r["error_gps_velocidad_constante_m"] for r in rows])
        out[f"{length}s"] = {"n": len(rows),
                             "p50_m": float(np.median(e)), "p90_m": float(np.percentile(e, 90)),
                             "gps_velocidad_constante_p50_m": float(np.median(c)),
                             "gps_velocidad_constante_p90_m": float(np.percentile(c, 90)),
                             "gana_pct": float(100 * np.mean(e < c))}
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
            print(f"  La cámara se declara detenida en {e['camara_detenida_con_gps_detenido_pct']:.0f} % "
                  f"de esos frames y en {e['camara_detenida_en_marcha_pct']:.1f} % de los frames a más "
                  f"de 1 m/s ({e['metros_perdidos_por_parada_falsa']:.0f} m de avance real perdidos)")
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


def print_fusion_report(rf):
    m = rf["modo"]
    sensors = "la cámara, el giroscopio y el GPS" if m["gyro"] else "la cámara y el GPS"
    print(f"\nFUSIÓN (filtro de Kalman con {sensors}; velocidad como estado: "
          f"{'sí' if m['speed_state'] else 'no'}; sesgo del giroscopio: "
          f"{'sí' if m['gyro'] and m['bias'] else 'no'})")
    print(f"  Arranca en el segundo {rf['inicio_s']:.0f}, con el primer rumbo del GPS "
          f"({GpsCameraFusion.INIT_METRES:.0f} m en movimiento)")
    if m["gyro"] and "frames_con_giroscopio_pct" in rf:
        print(f"  Frames girados con el giroscopio: {rf['frames_con_giroscopio_pct']:.1f} % "
              f"(el resto, con la cámara)")
    lo, hi = rf["s_p5_p95"]
    print(f"  Factor de escala s: {rf['s_final']:.2f} al final; entre {lo:.2f} y {hi:.2f} (p5-p95)")
    if "sesgo_final_deg_s" in rf:
        lo, hi = rf["sesgo_p5_p95_deg_s"]
        print(f"  Sesgo del giroscopio: {rf['sesgo_final_deg_s']:+.3f}°/s al final; "
              f"entre {lo:+.3f} y {hi:+.3f} (p5-p95)")
    c = rf["correcciones"]
    print("  Correcciones aceptadas / rechazadas / forzadas tras dos rechazos")
    print("  (NIS medio; lo esperado es 2 en posición y 1 en las demás):")
    names = (("posición", "posicion"), ("velocidad", "velocidad"), ("curso", "curso"),
             ("cámara", "camara"), ("alto", "alto"))
    print("     " + " | ".join(
        f"{name} {c[k]['n'] - c[k]['rechazadas'] - c[k]['forzadas']} / {c[k]['rechazadas']} / "
        f"{c[k]['forzadas']}" + (f" ({c[k]['nis_medio']:.2f})" if c[k]["nis_medio"] is not None else "")
        for name, k in names if c[k]["n"]))
    out_of_range = c["velocidad"]["fuera_de_rango"] + c["camara"]["fuera_de_rango"]
    if out_of_range:
        print(f"     Velocidades de la cámara fuera de rango (implicaban s fuera de "
              f"{PlanarEKF.S_RANGE[0]}-{PlanarEKF.S_RANGE[1]}), sin usar: {out_of_range}")
    h = rf["rumbo_con_y_sin_curso"]
    print(f"  Rumbo con y sin el curso del GPS: difiere {h['diferencia_p50_deg']:.1f}° (p50), "
          f"{h['diferencia_p95_deg']:.1f}° (p95), {h['diferencia_max_deg']:.1f}° como máximo; "
          f"incertidumbre {h['incertidumbre_p50_con_deg']:.1f}° con el curso y "
          f"{h['incertidumbre_p50_sin_deg']:.1f}° sin él (p50)")
    print(f"  Distancia entre la predicción y cada fix nuevo: p50 {rf['prediccion_p50_m']:.1f} m, "
          f"p95 {rf['prediccion_p95_m']:.1f} m")
    for key, name in (("camara_sola", "Cámara sola"), ("camara_giroscopio", "Cámara y giroscopio")):
        if key in rf:
            cs = rf[key]
            marks = ", ".join(f"{k}: {v:.0f} m" for k, v in cs["deriva_m"].items())
            print(f"  {name} desde el mismo inicio, sin GPS: {marks}; al final "
                  f"{cs['deriva_final_m']:.0f} m ({cs['deriva_final_pct']:.0f} % de lo recorrido)")
    for cut in rf["cortes"]:
        print(f"  Corte de GPS en el segundo {cut['inicio_s']:.0f} ({cut['duracion_s']:.0f} s, "
              f"{cut['recorrido_m']:.0f} m recorridos). Error al volver el GPS:")
        print(f"     fusión {cut['error_fusion_m']:.1f} m | GPS a velocidad constante "
              f"{cut['error_gps_velocidad_constante_m']:.1f} m | GPS congelado "
              f"{cut['error_gps_congelado_m']:.1f} m")


def run_sweep(frames_path, rec_dir):
    """sweep_outages for every variant the recording allows (the gyro ones need gyro_accel.csv)."""
    t, inputs, fixes, _, gyro_yaw = fusion_inputs(frames_path, rec_dir)
    out = {}
    for name, modes in VARIANTS:
        if modes["gyro"] and gyro_yaw is None:
            continue
        found = sweep_outages(t, inputs, fixes, gyro_yaw if modes["gyro"] else None, **modes)
        out[name] = summarize_sweep(found)
    return out


def print_sweep(sw):
    first = next(iter(sw.values()))
    lengths = list(first)
    width = max(len(name) for name in sw) + 2
    print("\nBARRIDO DE CORTES DE GPS (uno cada 10 s, en toda la grabación; "
          "error al volver el GPS, p50 / p90)")
    print("  " + "".ljust(width) + "".join(f"| {f'corte de {L}':>17s} " for L in lengths))
    for name, rows in sw.items():
        print("  " + name.ljust(width)
              + "".join(f"| {rows[L]['p50_m']:6.1f} / {rows[L]['p90_m']:5.1f} m " for L in lengths))
    print("  " + "GPS a velocidad constante".ljust(width)
          + "".join(f"| {first[L]['gps_velocidad_constante_p50_m']:6.1f} / "
                    f"{first[L]['gps_velocidad_constante_p90_m']:5.1f} m " for L in lengths))
    print("  Cortes en que le gana a la velocidad constante:")
    for name, rows in sw.items():
        print("     " + name.ljust(width) + "   ".join(f"{L}: {rows[L]['gana_pct']:3.0f} %" for L in lengths))
    print("  Cortes por duración: " + ", ".join(f"{L}: {first[L]['n']}" for L in lengths))


def plot_fusion(rf, png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = rf["_fus"]
    ts, est, sig, k0 = f["t"], f["est"], f["sig"], f["k0"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    def shade(axis):
        for i, (s0, s1) in enumerate(f["stops"]):
            axis.axvspan(s0, s1, color="#9a9a96", alpha=0.25, lw=0,
                         label="Detenido" if i == 0 else None)

    a = ax[0, 0]
    a.plot(f["gps"][:, 0], f["gps"][:, 1], "-", lw=2, color=COLOR_GPS, label="GPS del teléfono")
    a.plot(f["cam"][:, 0], f["cam"][:, 1], "--", lw=1.6, color=COLOR_CAMERA,
           label="Cámara sola, desde el mismo inicio")
    if f["cam_gyro"] is not None:
        a.plot(f["cam_gyro"][:, 0], f["cam_gyro"][:, 1], "-.", lw=1.6, color=COLOR_ALIGNED,
               label="Cámara y giroscopio, desde el mismo inicio")
    a.plot(est[:, 0], est[:, 1], "-", lw=1.4, color=COLOR_FUSED, label="Fusión")
    for i, c in enumerate(f["curves"]):
        a.plot(c["track"][:, 0], c["track"][:, 1], ":", lw=3, color=COLOR_FUSED,
               label="Fusión durante el corte de GPS" if i == 0 else None)
        a.plot(*c["track"][0], "x", ms=10, mew=2.5, color="#0b0b0b",
               label="Empieza el corte" if i == 0 else None)
    a.plot(est[k0, 0], est[k0, 1], "o", ms=8, color="#0b0b0b", label="Inicio")
    a.set_xlabel("Este (m)"); a.set_ylabel("Norte (m)"); a.set_title("Recorrido")
    a.axis("equal"); a.grid(alpha=0.3); a.legend(loc="best", fontsize=9)

    a = ax[0, 1]
    a.plot(*f["ratio"], ".", ms=4, color=COLOR_GPS, alpha=0.6, label="Doppler / cámara, en cada fix")
    a.plot(ts, est[:, 3], "-", lw=2, color=COLOR_FUSED, label="s del filtro")
    a.fill_between(ts, est[:, 3] - sig[:, 3], est[:, 3] + sig[:, 3], color=COLOR_FUSED,
                   alpha=0.2, lw=0, label="± 1 desviación")
    shade(a)
    a.set_ylim(0, 3)
    a.set_xlabel("tiempo (s)"); a.set_ylabel("factor de escala")
    a.set_title("Escala de la cámara que aprende el filtro"); a.grid(alpha=0.3)
    a.legend(loc="upper right", fontsize=9)

    a = ax[1, 0]
    a.plot(*f["pred"], "-", lw=1.2, color=COLOR_FUSED)
    shade(a)
    a.set_xlabel("tiempo (s)"); a.set_ylabel("distancia (m)")
    a.set_title("Predicción contra cada fix nuevo (1 s de cámara)"); a.grid(alpha=0.3)

    a = ax[1, 1]
    for i, c in enumerate(f["curves"]):
        first = i == 0
        a.plot(c["t"], c["errs"][0], "-", lw=2, color=COLOR_FUSED, label="Fusión" if first else None)
        a.plot(c["t"], c["errs"][1], "--", lw=1.6, color=COLOR_GPS,
               label="GPS a velocidad constante" if first else None)
        a.plot(c["t"], c["errs"][2], ":", lw=1.6, color=COLOR_GPS,
               label="GPS congelado" if first else None)
    if f["curves"]:
        a.legend(loc="best", fontsize=9)
    else:
        a.text(0.5, 0.5, "Sin cortes simulados (--outage)", ha="center", va="center",
               transform=a.transAxes, color="#52514e")
    a.set_xlabel("segundos desde el corte"); a.set_ylabel("distancia al GPS (m)")
    a.set_title("Durante el corte de GPS"); a.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Evaluación de una sesión contra los sensores del teléfono")
    ap.add_argument("frames", help="frames.csv de run_live.py o frames_*.csv de run_replay.py")
    ap.add_argument("--recording", required=True,
                    help="Carpeta de la grabación del teléfono (location.csv, gyro_accel.csv) "
                         "o de una sesión en vivo (gps.csv; sin giroscopio no evalúa la rotación)")
    ap.add_argument("--out", default=None, help="Por defecto, la carpeta del archivo de frames")
    ap.add_argument("--fusion", action="store_true",
                    help="Evaluar también la fusión de la cámara con el GPS")
    ap.add_argument("--outage", nargs=2, type=float, action="append", default=[],
                    metavar=("INICIO", "DURACION"),
                    help="Simular un corte de GPS (segundos desde el primer frame); "
                         "se puede repetir. Implica --fusion")
    ap.add_argument("--sweep", action="store_true",
                    help="Comparar las variantes del filtro con un corte de GPS cada 10 s, "
                         "de 10, 20, 30, 60 y 120 s, en toda la grabación")
    args = ap.parse_args()

    r = evaluate(args.frames, args.recording)
    print_report(r, args.frames, args.recording)

    out_dir = args.out or os.path.dirname(os.path.abspath(args.frames))
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(os.path.normpath(args.recording))
    png = os.path.join(out_dir, "evaluation.png")
    plot_report(r, png, f"Evaluación — {name}")

    if args.fusion or args.outage:
        rf = evaluate_fusion(args.frames, args.recording, args.outage)
        if rf is None:
            print("\n  La fusión no arrancó: el GPS nunca vio el carro en movimiento.")
        else:
            print_fusion_report(rf)
            png_f = os.path.join(out_dir, "fusion.png")
            plot_fusion(rf, png_f, f"Fusión — {name}")
            r["fusion"] = {k: v for k, v in rf.items() if not k.startswith("_")}
            print(f"\n  Gráfico de la fusión: {png_f}", end="")

    if args.sweep:
        r["barrido"] = run_sweep(args.frames, args.recording)
        print_sweep(r["barrido"])

    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump({k: v for k, v in r.items() if not k.startswith("_")}, f, indent=2)
    print(f"\n  Gráfico: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
