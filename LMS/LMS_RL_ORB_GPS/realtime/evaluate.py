"""
Evaluate a processed session (run_replay.py --save-frames, or a live session)
against the phone's own sensors in its recording:

    scale     camera speed against the GPS Doppler, by speed band
    rotation  VO rotation against the gyroscope
    heading   accumulated heading error, stopped and moving
    shape     ATE after a rigid horizontal alignment, and drift from a common start
    fusion    (--fusion) the filter against the fixes it has not seen yet, and
              in simulated GPS outages (--outage; --sweep, every 10 s)

The phone's GPS is not ground truth, but outdoors it is a fair reference for
shape and distance.

    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.evaluate FRAMES_CSV --recording mobile_data/2026_09_30_14_05_19
    ... --fusion --outage 120 30     the GPS cut for 30 s from second 120
    ... --sweep                      the filter's variants on an outage every 10 s
"""

import argparse
import copy
import json
import os
import sys
import time

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

from realtime.fusion import (GpsCameraFusion, PlanarEKF, correction_counts, feed_frame,
                             gyro_inputs, run_fusion, session_inputs, wrap)
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

    out["correcciones"] = correction_counts(fus.log)

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


# ------------------------------------------------------------ four modes

# What each sensor adds: a session through the filter in four configurations
# (key, name, visual only, gyroscope). The visual ones take the GPS only to
# start, where and which way; all start at the same frame.
MODES = (("camara", "Cámara sola", True, False),
         ("camara_giroscopio", "Cámara + giroscopio", True, True),
         ("camara_gps", "Cámara + GPS", False, False),
         ("todo", "Cámara + giroscopio + GPS", False, True))
RPE_DELTAS_S = (1, 10, 30)
RPE_DISTANCE_M = 100       # also per distance, as KITTI reports it
RPE_TOL_S = 0.3            # how far from delta a pair of fixes may be
# The GPS course is a heading reference only when the car is clearly moving:
# at 1 Hz its noise turns the course of a slow chord.
RPE_HEADING_SPEED = 5.0    # m/s
JUMP_M = 2.0               # a step between frames over this (60 m/s at 30 fps) is a jump
MODE_OUTAGE_S = 30


def before_fixes(t, est, tg):
    """Position and heading just before each fix, so a mode with GPS has not used it yet."""
    k = np.searchsorted(t, tg) - 1
    ok = (k >= 0) & np.isfinite(est[np.maximum(k, 0), 0])
    pos = np.where(ok[:, None], est[np.maximum(k, 0), :2], np.nan)
    return pos, np.where(ok, est[np.maximum(k, 0), 2], np.nan)


def ate(t, est, tg, xy):
    """
    Absolute error at each fix: as the mode gives it (anchored where it
    started) and after a rigid alignment (rotation and translation, no
    scale), the usual ATE.
    """
    pos, _ = before_fixes(t, est, tg)
    ok = np.isfinite(pos[:, 0])
    err = np.linalg.norm(pos[ok] - xy[ok], axis=1)
    aligned, _ = umeyama(pos[ok], xy[ok], with_scale=False)
    err_al = np.linalg.norm(aligned - xy[ok], axis=1)
    travelled = float(np.sum(np.linalg.norm(np.diff(xy[ok], axis=0), axis=1)))
    return {"fixes": int(ok.sum()), "recorrido_m": travelled,
            "rmse_m": float(np.sqrt(np.mean(err ** 2))), "p50_m": float(np.median(err)),
            "p95_m": float(np.percentile(err, 95)), "max_m": float(err.max()),
            "final_pct_recorrido": float(100 * err[-1] / max(travelled, 1e-9)),
            "alineado_rmse_m": float(np.sqrt(np.mean(err_al ** 2))), "_err": err, "_t": tg[ok]}


def rpe(t, est, tg, xy, vd, delta_s):
    """
    Relative error over delta_s, between pairs of fixes that far apart: the
    displacement the mode gives against the GPS's and, with the car above
    RPE_HEADING_SPEED at both ends, its heading change against that of the
    GPS course (the chord from the fix before to the one after).
    """
    pos, th = before_fixes(t, est, tg)
    j = np.searchsorted(tg, tg + int(delta_s * 1e9))
    j = np.minimum(j, len(tg) - 1)
    near = np.abs((tg[j] - tg) / 1e9 - delta_s) <= RPE_TOL_S
    i = np.flatnonzero(near & np.isfinite(pos[:, 0]) & np.isfinite(pos[j, 0]))
    j = j[i]
    d_est, d_ref = pos[j] - pos[i], xy[j] - xy[i]
    e = np.linalg.norm(d_est - d_ref, axis=1)
    out = {"pares": int(len(i)), "traslacion_rmse_m": float(np.sqrt(np.mean(e ** 2))),
           "traslacion_p95_m": float(np.percentile(e, 95)),
           "traslacion_pct": float(100 * np.sum(e) / max(np.sum(np.linalg.norm(d_ref, axis=1)), 1e-9)),
           "_trans": e}
    # Course at each fix from its neighbours; the ends have none.
    course = np.full(len(tg), np.nan)
    chord = xy[2:] - xy[:-2]
    course[1:-1] = np.arctan2(chord[:, 1], chord[:, 0])
    fast = (vd[i] >= RPE_HEADING_SPEED) & (vd[j] >= RPE_HEADING_SPEED)
    ok = fast & np.isfinite(course[i]) & np.isfinite(course[j]) & np.isfinite(th[i]) & np.isfinite(th[j])
    rot = np.degrees(np.abs(wrap((th[j] - th[i]) - (course[j] - course[i]))))[ok]
    out.update({"pares_rumbo": int(ok.sum()),
                "rotacion_rmse_deg": float(np.sqrt(np.mean(rot ** 2))) if ok.any() else None,
                "_rot": rot})
    return out


def rpe_distance(t, est, tg, xy, metres=RPE_DISTANCE_M):
    """
    Relative error per distance: from each fix to the first one at least
    `metres` further along the GPS's path, the displacement the mode gives
    against the GPS's, as a share of that path.
    """
    pos, _ = before_fixes(t, est, tg)
    path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    j = np.searchsorted(path, path + metres)
    keep = (j < len(tg)) & np.isfinite(pos[:, 0])
    i = np.flatnonzero(keep & np.isfinite(pos[np.minimum(j, len(tg) - 1), 0]))
    j = j[i]
    e = np.linalg.norm((pos[j] - pos[i]) - (xy[j] - xy[i]), axis=1)
    length = path[j] - path[i]
    return {"pares": int(len(i)), "traslacion_rmse_m": float(np.sqrt(np.mean(e ** 2))),
            "traslacion_pct": float(100 * np.sum(e) / max(np.sum(length), 1e-9)), "_trans": e,
            "_len": length}


def stability(t, est, err_t, err):
    """
    Whether the track holds together: jumps between consecutive frames that no
    car could make (an update that moves the estimate at once), and how fast
    the error grows with time (m/min, least squares; about 0 when bounded).
    """
    ready = np.isfinite(est[:, 0])
    steps = np.linalg.norm(np.diff(est[ready, :2], axis=0), axis=1)
    minutes = (err_t - err_t[0]) / 60e9
    slope = float(np.polyfit(minutes, err, 1)[0]) if len(err) > 2 else None
    return {"saltos": int(np.sum(steps > JUMP_M)), "salto_max_m": float(steps.max()),
            "crecimiento_m_por_min": slope}


def four_modes(t, inputs, fixes, gyro_yaw):
    """Every metric of the four modes on one session; None if the filter never starts."""
    tg = np.array([f[0] for f in fixes], np.int64)
    xy = np.array([f[1] for f in fixes])
    vd = np.array([f[2] for f in fixes])
    out = {}
    start = None
    for key, name, visual, gyro in MODES:
        g = gyro_yaw if gyro else None
        fus, est, _ = run_fusion(t, inputs, fixes, gyro_yaw=g, visual_only=visual,
                                 speed_state=True, bias=False)
        ready = np.flatnonzero(np.isfinite(est[:, 0]))
        if not len(ready):
            return None
        start = ready[0] if start is None else start
        if ready[0] != start:
            raise RuntimeError(f"{name} arranca en otro frame ({ready[0]} contra {start})")
        after = tg > t[start]
        a = ate(t, est, tg[after], xy[after])
        r = {"nombre": name, "ate": a,
             "rpe": {**{f"{d}s": rpe(t, est, tg[after], xy[after], vd[after], d) for d in RPE_DELTAS_S},
                     f"{RPE_DISTANCE_M}m": rpe_distance(t, est, tg[after], xy[after])},
             "_est": est,
             "estabilidad": stability(t, est, tg[after], a["_err"])}
        if not visual:
            c = correction_counts(fus.log)["posicion"]
            r["estabilidad"]["posicion_rechazada_pct"] = 100 * c["rechazadas"] / max(c["n"], 1)
            r["estabilidad"]["posicion_forzada_pct"] = 100 * c["forzadas"] / max(c["n"], 1)
            found = sweep_outages(t, inputs, fixes, g, lengths=(MODE_OUTAGE_S,), speed_state=True,
                                  bias=False)
            r["cortes"] = summarize_sweep(found).get(f"{MODE_OUTAGE_S}s")
        out[key] = r
    out["_inicio_s"] = float((t[start] - t[0]) / 1e9)
    out["_t"], out["_gps"], out["_tg"] = t, xy, tg
    return out


def summarize_sweep(found):
    """
    Per length: p50 and p90 of both errors, and how often the filter wins. A
    length with no outage (longer than the session allows) is left out.
    """
    out = {}
    for length, rows in found.items():
        if not rows:
            continue
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


def cached_runs():
    """The full depth replays in the cache made by the current code: (session, entry folder)."""
    from realtime import cache
    from realtime.rl.env import load_session
    current = cache.code_hash()
    for rec, key, folder, m in cache.entries():
        o = m["opciones"]
        if (m["hash_codigo"] == current and o["scale"] == "depth" and not o["frames"]
                and not o["start"] and not o["visual_only"] and not o["no_gyro"]
                and os.path.exists(os.path.join(folder, cache.LOCATION))):
            yield load_session(folder), folder


def run_cache(out_dir):
    """The four modes on every cached session: the OE4 table, as JSON and CSV."""
    results = {}
    for s, folder in cached_runs():
        tic = time.perf_counter()
        r = four_modes(s.t, s.inputs, s.fixes, s.gyro_yaw)
        if r is None:
            print(f"  {s.name}: la fusión no arrancó, se omite")
            continue
        with open(os.path.join(folder, "cache.json")) as f:
            meta = json.load(f)
        replay = json.load(open(next(os.path.join(folder, n) for n in os.listdir(folder)
                                     if n.startswith("replay_deterministic"))))
        r["_rendimiento"] = {"equipo": meta["equipo"], "torch": meta.get("torch"),
                             "frames": replay["frames_procesados"],
                             "proc_ms_p50": replay["proc_ms_p50"], "proc_ms_p95": replay["proc_ms_p95"],
                             "profundidad_ms_p50": replay.get("depth_ms_p50"),
                             "fusion_ms_p50": replay["fusion"].get("tiempo_ms_p50")}
        results[s.name] = r
        print(f"  {s.name}: {(time.perf_counter() - tic):.0f} s", flush=True)
    print_modes(results)
    os.makedirs(out_dir, exist_ok=True)
    clean = lambda d: {k: clean(v) if isinstance(v, dict) else v for k, v in d.items()
                       if not k.startswith("_") or k in ("_inicio_s", "_rendimiento")}
    with open(os.path.join(out_dir, "modos.json"), "w") as f:
        json.dump({n: clean(r) for n, r in results.items()}, f, indent=1, default=float)
    write_modes_csv(results, os.path.join(out_dir, "modos.csv"))
    for n, r in results.items():
        plot_mode_map(n, r, os.path.join(out_dir, f"mapa_{n}.png"))
    plot_error_time(results, os.path.join(out_dir, "error_contra_tiempo.png"))
    print(f"\n  Resultados: {out_dir}/modos.json y modos.csv; figuras mapa_<sesión>.png y "
          f"error_contra_tiempo.png")
    return results


def pooled(results, key, part):
    """One mode's per-fix or per-pair errors of every session together."""
    if part == "ate":
        return np.concatenate([r[key]["ate"]["_err"] for r in results.values()])
    d, kind = part
    return np.concatenate([r[key]["rpe"][d][kind] for r in results.values()])


def print_modes(results):
    names = list(results)
    print("\n" + "=" * 100)
    print("LOS CUATRO MODOS (referencia: el GPS del teléfono, no ground truth; el modo, justo antes de cada fix)")
    print("=" * 100)
    for key, name, visual, _ in MODES:
        print(f"\n{name}{' (el GPS solo para arrancar)' if visual else ''}")
        print(f"  {'sesión':22s} {'ATE rmse':>9s} {'alineado':>9s} {'p95':>7s} {'final %':>8s} | "
              + " ".join(f"{'RPE ' + d:>9s}" for d in (f'{x}s' for x in RPE_DELTAS_S))
              + f" {'rumbo 10s':>10s} {f'{RPE_DISTANCE_M} m %':>8s} | {'saltos':>6s} {'m/min':>6s}"
              + ("" if visual else f" | {'corte 30 s p50/p90':>19s}"))
        for n in names:
            r = results[n][key]
            a, st = r["ate"], r["estabilidad"]
            rot = r["rpe"]["10s"]["rotacion_rmse_deg"]
            line = (f"  {n[-19:]:22s} {a['rmse_m']:9.1f} {a['alineado_rmse_m']:9.1f} {a['p95_m']:7.1f} "
                    f"{a['final_pct_recorrido']:8.1f} | "
                    + " ".join(f"{r['rpe'][f'{d}s']['traslacion_rmse_m']:9.2f}" for d in RPE_DELTAS_S)
                    + f" {('%.1f°' % rot) if rot is not None else '-':>10s}"
                    f" {r['rpe'][f'{RPE_DISTANCE_M}m']['traslacion_pct']:8.1f} | {st['saltos']:6d} "
                    f"{st['crecimiento_m_por_min']:6.1f}")
            if not visual and r.get("cortes"):
                line += f" | {r['cortes']['p50_m']:8.1f} / {r['cortes']['p90_m']:6.1f}"
            print(line)
        e = pooled(results, key, "ate")
        rp = [pooled(results, key, (f"{d}s", "_trans")) for d in RPE_DELTAS_S]
        rr = pooled(results, key, ("10s", "_rot"))
        dist = f"{RPE_DISTANCE_M}m"
        pct = 100 * pooled(results, key, (dist, "_trans")).sum() / pooled(results, key, (dist, "_len")).sum()
        print(f"  {'las 7 juntas':22s} {np.sqrt(np.mean(e ** 2)):9.1f} {'':9s} {np.percentile(e, 95):7.1f} "
              f"{'':8s} | " + " ".join(f"{np.sqrt(np.mean(x ** 2)):9.2f}" for x in rp)
              + f" {('%.1f°' % np.sqrt(np.mean(rr ** 2))) if len(rr) else '-':>10s} {pct:8.1f}")
    print("\nRendimiento del pipeline por sesión (replay determinista; el modelo de profundidad en línea):")
    for n in names:
        p = results[n]["_rendimiento"]
        print(f"  {n:22s} {p['equipo'][:22]:22s} procesamiento por frame p50 / p95 {p['proc_ms_p50']:5.1f} / "
              f"{p['proc_ms_p95']:5.1f} ms; profundidad p50 {p['profundidad_ms_p50']:5.0f} ms; "
              f"fusión p50 {p['fusion_ms_p50']:.2f} ms")


# Each mode's colour and dash, the project's colours plus one: the dash keeps
# them apart where colour does not (the aqua is under 3:1 on white).
COLOR_CAMERA_GPS = "#4a3aa7"
MODE_STYLE = {"camara": (COLOR_CAMERA, (0, (6, 3))),
              "camara_giroscopio": (COLOR_ALIGNED, (0, (2, 2))),
              "camara_gps": (COLOR_CAMERA_GPS, (0, (8, 2, 2, 2))),
              "todo": (COLOR_FUSED, "-")}
DETAIL_M = 150             # half-width of the map's detail panel


def plot_mode_map(name, res, png):
    """
    One session: the four modes and the GPS on the map, and a detail around
    the fix where the camera + GPS mode is furthest from the GPS, where the
    two GPS modes can be told apart.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gps = res["_gps"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    for ax, keys in ((axes[0], [m[0] for m in MODES]), (axes[1], ["camara_gps", "todo"])):
        ax.plot(gps[:, 0], gps[:, 1], "-", color=COLOR_GPS, lw=2.4, label="GPS del teléfono (referencia)")
        for key in keys:
            color, dash = MODE_STYLE[key]
            est = res[key]["_est"]
            ax.plot(est[:, 0], est[:, 1], color=color, ls=dash, lw=1.6, label=res[key]["nombre"])
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_xlabel("este (m)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    k0 = np.flatnonzero(np.isfinite(res["todo"]["_est"][:, 0]))[0]
    axes[0].plot(*res["todo"]["_est"][k0, :2], "o", color="#0b0b0b", ms=8, label="arranque")
    axes[0].set_ylabel("norte (m)")
    axes[0].set_title("Los cuatro modos desde el mismo arranque", fontsize=11, loc="left")
    axes[0].legend(fontsize=8.5, loc="best", frameon=False)

    a = res["camara_gps"]["ate"]
    t_worst = a["_t"][int(np.argmax(a["_err"]))]
    at = gps[int(np.argmin(np.abs(res["_tg"] - t_worst)))]
    ax = axes[1]
    near = np.all(np.abs(gps - at) <= DETAIL_M, axis=1)
    ax.plot(gps[near, 0], gps[near, 1], "o", color=COLOR_GPS, ms=4, label="fixes del GPS")
    ax.set_xlim(at[0] - DETAIL_M, at[0] + DETAIL_M)
    ax.set_ylim(at[1] - DETAIL_M, at[1] + DETAIL_M)
    ax.set_title(f"Detalle de {2 * DETAIL_M} m: donde Cámara + GPS más se separa del GPS "
                 f"({a['max_m']:.0f} m)", fontsize=11, loc="left")
    ax.legend(fontsize=8.5, loc="best", frameon=False)
    fig.suptitle(f"{name}: los cuatro modos (referencia: el GPS del teléfono, no ground truth)",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)


def plot_error_time(results, png):
    """The error against the GPS over time, per session and mode (10 s rolling median, log scale)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    cols = 4
    rows = (len(names) + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.1 * rows), sharey=True)
    axes = axes.ravel()
    for ax, n in zip(axes, names):
        res = results[n]
        t_start = res["_t"][0] + int(res["_inicio_s"] * 1e9)
        for key, label, _, _ in MODES:
            a = res[key]["ate"]
            minutes = (a["_t"] - t_start) / 60e9
            err = pd.Series(a["_err"]).rolling(10, center=True, min_periods=1).median()
            color, dash = MODE_STYLE[key]
            ax.plot(minutes, err, color=color, ls=dash, lw=1.6, label=label)
        ax.set_yscale("log")
        ax.set_ylim(0.3, 1e4)
        ax.set_title(n, fontsize=10, loc="left")
        ax.grid(color="#e4e3df", lw=0.6, which="major")
        ax.set_xlabel("minutos desde el arranque")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[::cols]:
        ax.set_ylabel("error contra el GPS (m)")
    legend = axes[len(names)]
    legend.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    legend.legend(handles, labels, loc="center left", fontsize=10, frameon=False,
                  title="Modo", title_fontsize=10, alignment="left")
    legend.text(0.0, 0.08, "Mediana móvil de 10 s; escala logarítmica.\nEl modo, justo antes de cada fix.",
                fontsize=8.5, color="#52514e", transform=legend.transAxes)
    for ax in axes[len(names) + 1:]:
        ax.axis("off")
    fig.suptitle("El error contra el GPS del teléfono a lo largo de cada sesión", fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)


def write_modes_csv(results, path):
    rows = []
    for n, res in results.items():
        for key, name, visual, _ in MODES:
            r = res[key]
            a, st = r["ate"], r["estabilidad"]
            row = {"sesion": n, "modo": name, "ate_rmse_m": a["rmse_m"], "ate_alineado_rmse_m": a["alineado_rmse_m"],
                   "ate_p50_m": a["p50_m"], "ate_p95_m": a["p95_m"], "ate_max_m": a["max_m"],
                   "error_final_pct_recorrido": a["final_pct_recorrido"], "recorrido_m": a["recorrido_m"]}
            for d in RPE_DELTAS_S:
                row[f"rpe_{d}s_rmse_m"] = r["rpe"][f"{d}s"]["traslacion_rmse_m"]
                row[f"rpe_{d}s_pct"] = r["rpe"][f"{d}s"]["traslacion_pct"]
            row["rpe_rumbo_10s_rmse_deg"] = r["rpe"]["10s"]["rotacion_rmse_deg"]
            row[f"rpe_{RPE_DISTANCE_M}m_rmse_m"] = r["rpe"][f"{RPE_DISTANCE_M}m"]["traslacion_rmse_m"]
            row[f"rpe_{RPE_DISTANCE_M}m_pct"] = r["rpe"][f"{RPE_DISTANCE_M}m"]["traslacion_pct"]
            row.update({"saltos": st["saltos"], "salto_max_m": st["salto_max_m"],
                        "crecimiento_m_por_min": st["crecimiento_m_por_min"],
                        "posicion_rechazada_pct": st.get("posicion_rechazada_pct"),
                        "posicion_forzada_pct": st.get("posicion_forzada_pct")})
            c = r.get("cortes") or {}
            row.update({"corte_30s_p50_m": c.get("p50_m"), "corte_30s_p90_m": c.get("p90_m")})
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description="Evaluación de una sesión contra los sensores del teléfono")
    ap.add_argument("frames", nargs="?",
                    help="frames.csv de run_live.py o frames_*.csv de run_replay.py")
    ap.add_argument("--cache", action="store_true",
                    help="Los cuatro modos (cámara sola, cámara + giroscopio, cámara + GPS, todo) "
                         "en todas las sesiones de la caché: ATE, RPE, estabilidad y rendimiento")
    ap.add_argument("--recording",
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
    if args.cache:
        run_cache(args.out or os.path.join(_ROOT, "resultados", "oe4", time.strftime("%Y%m%d_%H%M%S")))
        return 0
    if not args.frames or not args.recording:
        ap.error("hace falta el archivo de frames y --recording (o --cache)")

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
