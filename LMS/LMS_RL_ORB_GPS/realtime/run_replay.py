"""
Replay mode: run the real-time pipeline over a recorded session, honouring the
real time between frames.

It is the same system that runs live (run_live.py); only the data source
changes. It validates the architecture and measures latency WITHOUT depending
on the Android app, the network or the field hardware.

Usage:
    # deterministic mode (reproducible, for development and reporting)
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_replay \
        mobile_data/2025_03_11 --frames 900

    # strict mode (real behaviour, not deterministic)
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_replay \
        mobile_data/2025_03_11 --frames 900 --strict

    # with the camera's own metric scale (monocular depth model)
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_replay \
        mobile_data/2025_03_11 --frames 900 --scale depth --trajectory

    # a recording of the current app: intrinsics from its HELLO, no dashboard,
    # and without the first frames the encoder writes before catching up
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_replay \
        mobile_data/2026_09_29_13_01_19 --fx 867.81 --fy 868.55 \
        --cx 630.75 --cy 367.79 --mask-bottom 0 --skip-start 6

Console output stays in Spanish: it is evidence for the thesis report.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_LMS_RL = os.path.abspath(os.path.join(_THIS, ".."))
_ROOT = os.path.abspath(os.path.join(_LMS_RL, "..", ".."))
for _p in (_ROOT, _LMS_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.gps.gps_utils import latlon_to_utm
from realtime.sources import (GpsBuffer, ReplayFrameSource, ReplayGpsSource,
                              load_mobile_session)
from realtime.pipeline import (DepthScaleWorker, RealtimePipeline, VisualFrontEnd,
                               simulate_drops)

# Series colours shared by every figure, so GPS and camera keep theirs across
# charts.
COLOR_GPS = "#2a78d6"
COLOR_CAMERA = "#eb6834"
COLOR_ALIGNED = "#1baf7a"


def mobile_camera_matrix(width: int, height: int, fx: float = 899.0,
                         fy: float = None, cx: float = None, cy: float = None):
    """
    Real phone calibration. fx/fy come from movie_metadata.csv or from the
    app's HELLO; the optical centre defaults to the image centre, since the
    recording files do not store it. The offline pipeline defaults to KITTI
    values, which are wrong for this video.
    """
    fy = fx if fy is None else fy
    cx = width / 2.0 if cx is None else cx
    cy = height / 2.0 if cy is None else cy
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_positions(rel, scales=None):
    """
    Chain the relative poses into camera positions, in the first camera's frame.

    recoverPose gives the transform of POINTS from the previous camera to the
    current one (x2 = R x1 + t), so the camera itself moves by the inverse:
    rotation R^T and displacement -R^T t. Chaining R and t directly mirrors
    every turn and runs the route backwards.

    scales holds the metres of each step, or None for unit steps; a missing
    value (no speed yet) gives a null step.
    """
    P = np.eye(4)
    out = np.zeros((len(rel), 3))
    for i, (R, t) in enumerate(rel):
        s = 1.0 if scales is None else scales[i]
        if s is None or not np.isfinite(s):
            s = 0.0
        step = np.eye(4)
        step[:3, :3] = R.T
        step[:3, 3] = -R.T @ (np.asarray(t) * s)
        P = P @ step
        out[i] = P[:3, 3]
    return out


class TrackRecorder:
    """on_result callback that keeps every processed step, for plots and CSV."""

    KEYS = ("rel", "t_ns", "unix_ns", "matches", "inliers", "gps", "gps_t", "scale")

    def __init__(self):
        self.track = {key: [] for key in self.KEYS}

    def __call__(self, frame, R, t, n_m, n_i, fix, scale):
        tr = self.track
        tr["rel"].append((R.copy(), t.copy()))
        tr["t_ns"].append(frame.t_ns)
        tr["unix_ns"].append(frame.unix_ns)
        tr["matches"].append(n_m)
        tr["inliers"].append(n_i)
        tr["gps"].append(None if fix is None else fix[1].copy())
        tr["gps_t"].append(0 if fix is None else int(fix[0]))
        tr["scale"].append(scale)


FRAME_COLUMNS = (["t_ns", "unix_ns", "matches", "inliers"]
                 + [f"r{i}{j}" for i in range(3) for j in range(3)]
                 + ["tx", "ty", "tz", "scale_m",
                    "gps_t_ns", "gps_x", "gps_y", "gps_z", "e2e_ms", "pc_ms"])


def save_frames_csv(path, track, metrics):
    """
    One row per processed frame: the raw relative pose from recoverPose (the
    points convention, see camera_positions), its metres and the paired fix.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FRAME_COLUMNS)
        for k, (R, t) in enumerate(track["rel"]):
            g, s, u = track["gps"][k], track["scale"][k], track["unix_ns"][k]
            pc = metrics.pc_ms[k] if k < len(metrics.pc_ms) else None
            w.writerow([track["t_ns"][k], "" if u is None else u,
                        track["matches"][k], track["inliers"][k]]
                       + [f"{v:.9g}" for v in R.ravel()]
                       + [f"{v:.9g}" for v in t]
                       + ["" if s is None else f"{s:.6g}"]
                       + (["", "", "", ""] if g is None
                          else [track["gps_t"][k]] + [f"{v:.3f}" for v in g])
                       + [f"{metrics.e2e_ms[k]:.2f}", "" if pc is None else f"{pc:.2f}"])


def umeyama(est, ref, with_scale=True):
    """
    Umeyama alignment.

    with_scale=True   also solves for the scale factor. Standard for pure
                      monocular VO, which has no absolute scale.
    with_scale=False  rigid alignment (rotation and translation only). The
                      right choice once the trajectory is already in metres:
                      fitting the scale would correct the very error being
                      measured.
    """
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    ec, rc = est - est.mean(0), ref - ref.mean(0)
    U, S, Vt = np.linalg.svd(ec.T @ rc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    if not with_scale:
        return (ec @ R.T) + ref.mean(0), 1.0
    var = np.sum(ec ** 2)
    s = np.sum(S) / var if var > 1e-10 else 1.0
    return s * (ec @ R.T) + ref.mean(0), s


def plot_trajectory(track, out_dir, mode, metric=False):
    """
    Rebuild the route from the accumulated relative poses.

    With metric=True each step is multiplied by the metres estimated by the
    depth model and the alignment becomes RIGID.

    Interpretation warning: the mobile data has NO ground truth. The phone GPS
    is off by several metres, so comparing against it says whether the SHAPE is
    plausible, not how accurate the result is.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # VO chain. Without metric scale every step has norm 1; with it, direction
    # comes from the camera and magnitude from the depth model.
    vo = camera_positions(track["rel"], track["scale"] if metric else None)

    # Pair BY INDEX. Filtering out the Nones without keeping their position
    # shifted the whole GPS series towards the start: with 19 initial frames
    # without a fix the comparison was 19 frames out of step (~0.6 s, ~4 m at
    # this route's speed), which inflated the measured error.
    has_gps = np.array([g is not None for g in track["gps"]])
    if has_gps.sum() < 10 or len(vo) < 10:
        print("\n  (no hay suficientes datos para graficar la trayectoria)")
        return

    n = min(len(vo), len(has_gps))
    has_gps = has_gps[:n]
    vo_p = vo[:n][has_gps]
    gps = np.array([g for g in track["gps"][:n] if g is not None])

    vo_al, scale = umeyama(vo_p, gps, with_scale=not metric)
    err = np.linalg.norm(vo_al - gps, axis=1)
    dist = float(np.sum(np.linalg.norm(np.diff(gps, axis=0), axis=1)))
    dist_vo = float(np.sum(np.linalg.norm(np.diff(vo[:n], axis=0), axis=1)))

    print("\n" + "-" * 78)
    print(f"TRAYECTORIA RECONSTRUIDA  "
          f"({'escala métrica por profundidad' if metric else 'sin escala métrica'})")
    print("-" * 78)
    print(f"  Distancia recorrida según GPS : {dist:.0f} m")
    if metric:
        print(f"  Distancia recorrida según VO  : {dist_vo:.0f} m"
              f"   ({100*(dist_vo-dist)/max(dist,1e-9):+.0f}% vs GPS)")
    print(f"  Desviación VO vs GPS  RMSE    : {np.sqrt((err**2).mean()):.2f} m")
    print(f"                        mediana : {np.median(err):.2f} m")
    print(f"                        máxima  : {err.max():.2f} m")

    # Same thing, but only at the instant a new fix arrives. Between fixes the
    # GPS position freezes while the vehicle keeps moving, which is the
    # sawtooth in the deviation curve. At arrival its age is ~0, so the
    # comparison there is clean.
    t_fix = np.array([t for t, g in zip(track["gps_t"][:n], track["gps"][:n])
                      if g is not None], dtype=np.int64)
    is_new = np.ones(len(t_fix), dtype=bool)
    is_new[1:] = np.diff(t_fix) > 0
    if is_new.sum() >= 5:
        vo_f, _ = umeyama(vo_p[is_new], gps[is_new], with_scale=not metric)
        err_f = np.linalg.norm(vo_f - gps[is_new], axis=1)
        print("  Solo al llegar cada fix (sin el efecto escalera del GPS):")
        print(f"                        RMSE    : {np.sqrt((err_f**2).mean()):.2f} m"
              f"   ({int(is_new.sum())} fixes)")
        print(f"                        mediana : {np.median(err_f):.2f} m")

    if metric:
        print("  Alineación RÍGIDA (sin ajustar escala): la escala es de la cámara.")
    else:
        print(f"  Factor de escala de alineación: {scale:.4f}")
    print("  NOTA: el GPS del teléfono NO es ground truth (error de varios metros).")
    print("        Esto mide parecido de forma, no exactitud.")

    vo_label = ("Cámara con escala propia (rígida)" if metric
                else "Odometría visual (alineada)")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].plot(gps[:, 0], gps[:, 1], "-", lw=2, color=COLOR_GPS,
               label="GPS del teléfono")
    ax[0].plot(vo_al[:, 0], vo_al[:, 1], "--", lw=1.6, color=COLOR_CAMERA,
               label=vo_label)
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Y (m)")
    ax[0].set_title(f"Recorrido — {dist:.0f} m")
    ax[0].axis("equal"); ax[0].grid(alpha=0.3); ax[0].legend()

    ax[1].plot(err, lw=1.2, color=COLOR_CAMERA)
    ax[1].axhline(np.median(err), ls="--", color="gray",
                  label=f"mediana {np.median(err):.1f} m")
    ax[1].set_xlabel("keyframe"); ax[1].set_ylabel("desviación (m)")
    ax[1].set_title("Desviación entre VO y GPS")
    ax[1].grid(alpha=0.3); ax[1].legend()

    suffix = "_metric" if metric else ""
    fig.suptitle(("En vivo" if mode == "live" else f"Modo replay ({mode})")
                 + " — trayectoria reconstruida"
                 + (" con escala por profundidad" if metric else ""))
    fig.tight_layout()
    png = os.path.join(out_dir, f"trajectory_{mode}{suffix}.png")
    fig.savefig(png, dpi=130)
    print(f"  Gráfico: {png}")


def build_scale_worker(K, rate_hz, threaded):
    """
    Build the depth-based scale estimator and its worker.

    The estimator does NOT share the front-end's OpenCV objects: they run on
    different threads and cv2 gives no thread-safety guarantee. What is reused
    are the already computed keypoints and descriptors (see
    VisualFrontEnd.last_features), which are immutable data, so the estimator
    sees exactly the same features without running ORB again.
    """
    from realtime.depth_scale import DepthScaleEstimator
    print("  Escala métrica: modelo de profundidad monocular "
          f"(~{rate_hz:.0f} Hz, "
          f"{'hilo aparte' if threaded else 'en línea, determinista'})")
    print("  Cargando el modelo...", flush=True)
    est = DepthScaleEstimator(K, hist_len=12)
    worker = DepthScaleWorker(est, rate_hz=rate_hz, threaded=threaded)
    print("  Modelo listo.\n")
    return est, worker


def print_verdict(e2e_p95_ms, hz, target_ms=150.0, target_hz=10.0):
    """Verdict against the real-time objective."""
    print("\n" + "=" * 78)
    print(f"OBJETIVO PROPUESTO: latencia p95 < {target_ms:.0f} ms  "
          f"y  ≥ {target_hz:.0f} Hz sostenidos")
    print(f"  Latencia p95 : {e2e_p95_ms:7.1f} ms   "
          f"{'CUMPLE' if e2e_p95_ms < target_ms else 'NO CUMPLE'}")
    print(f"  Frecuencia   : {hz:7.1f} Hz   {'CUMPLE' if hz >= target_hz else 'NO CUMPLE'}")
    print("=" * 78)


def print_results(res, strict, est, worker):
    """Print the measurement report."""
    print("-" * 78)
    print("RESULTADOS")
    print("-" * 78)
    print(f"  Frames procesados / descartados : "
          f"{res['frames_procesados']} / {res['frames_descartados']}"
          f"  ({res['tasa_descarte_%']:.1f}% descarte)")
    print(f"  Frecuencia efectiva             : {res['hz_efectivo']:.1f} Hz")
    print()
    print(f"  Tiempo de procesamiento  p50    : {res['proc_ms_p50']:.1f} ms")
    print(f"                           p95    : {res['proc_ms_p95']:.1f} ms")
    print(f"                           máx    : {res['proc_ms_max']:.1f} ms")
    print(f"     · detección ORB       p50    : {res['detect_ms_p50']:.1f} ms")
    print(f"     · emparejamiento      p50    : {res['match_ms_p50']:.1f} ms")
    print(f"     · estimación de pose  p50    : {res['pose_ms_p50']:.1f} ms")
    print()
    print(f"  Antigüedad del fix GPS   p50    : {res['gps_age_ms_p50']:.0f} ms")
    print(f"                           p95    : {res['gps_age_ms_p95']:.0f} ms")
    print(f"  Frames sin GPS disponible       : {res['frames_sin_gps']}")
    print(f"  Poses descartadas (giro de más de {VisualFrontEnd.MAX_ROTATION_DEG:.0f}° "
          f"entre frames): {res['poses_giro_imposible']}")

    if worker is None:
        return

    print()
    print("  Escala métrica (hilo aparte, no entra en el presupuesto por frame):")
    print(f"     Estimaciones ejecutadas      : {res['depth_estimaciones']}"
          f"  de {worker.n_submitted} enviadas"
          f"  ({worker.n_dropped[0]} descartadas por cola llena)")
    print(f"     Tiempo por estimación p50/p95: "
          f"{res['depth_ms_p50']:.0f} / {res['depth_ms_p95']:.0f} ms")
    print(f"     Aceptadas / falladas / fuera de rango físico    : "
          f"{est.n_ok} / {est.n_fail} / {est.n_rejected}")
    print(f"     Frames sin escala (arranque) : {res['frames_sin_escala']}")
    v = est.velocity
    print("     Última velocidad estimada    : "
          + (f"{v:.2f} m/s" if v is not None else "—"))
    if not strict:
        print("     (en este modo el modelo corre EN LÍNEA para ser reproducible,")
        print("      así que 'frecuencia efectiva' de arriba lo incluye y baja;")
        print("      el número comparable es el p50 del bucle.)")


def main():
    ap = argparse.ArgumentParser(description="Modo replay del pipeline en tiempo real")
    ap.add_argument("session", help="Directorio de la sesión (p.ej. mobile_data/2025_03_11)")
    ap.add_argument("--video", default="movie.mp4")
    ap.add_argument("--frames", type=int, default=None, help="Máximo de frames")
    ap.add_argument("--strict", action="store_true",
                    help="Duerme y descarta de verdad (no determinista)")
    ap.add_argument("--mask-bottom", type=float, default=0.14,
                    help="Fracción inferior donde ORB no busca puntos (tablero o "
                         "capó); el modelo de profundidad ve la imagen completa")
    ap.add_argument("--fx", type=float, default=899.0)
    ap.add_argument("--fy", type=float, default=None, help="Por defecto, igual a fx")
    ap.add_argument("--cx", type=float, default=None, help="Por defecto, el centro de la imagen")
    ap.add_argument("--cy", type=float, default=None, help="Por defecto, el centro de la imagen")
    ap.add_argument("--skip-start", type=int, default=0,
                    help="Frames a descartar al inicio del video (app actual: 6, "
                         "el codificador todavía no escribe la imagen de su marca)")
    ap.add_argument("--start", type=float, default=0.0,
                    help="Segundo de la grabación desde el que se procesa")
    ap.add_argument("--out", default="resultados/realtime")
    ap.add_argument("--trajectory", action="store_true",
                    help="Reconstruir y graficar el recorrido además de medir tiempos")
    ap.add_argument("--save-frames", action="store_true",
                    help="Guardar los resultados por frame en frames_<modo>.csv, "
                         "el formato que lee evaluate.py")
    ap.add_argument("--scale", choices=["none", "depth"], default="none",
                    help="Fuente de la escala métrica. 'depth' activa el "
                         "estimador monocular en un hilo aparte")
    ap.add_argument("--scale-hz", type=float, default=3.0,
                    help="Ritmo del estimador de escala (el modelo tarda ~100 ms)")
    args = ap.parse_args()

    print("=" * 78)
    print(f"MODO REPLAY — {'ESTRICTO' if args.strict else 'DETERMINISTA'}")
    print("=" * 78)

    frame_t, fixes = load_mobile_session(args.session, latlon_to_utm)
    # Frames left out at the start: the encoder warm-up, or everything before
    # --start. The source discards the same ones.
    skip = max(args.skip_start,
               int(np.searchsorted(frame_t, frame_t[0] + int(args.start * 1e9))))
    frame_t = frame_t[skip:]
    n_total = len(frame_t) if args.frames is None else min(args.frames, len(frame_t))
    dur_s = (frame_t[n_total - 1] - frame_t[0]) / 1e9

    # Keep only the fixes inside the processed window.
    t_ini, t_fin = int(frame_t[0]), int(frame_t[n_total - 1])
    fixes = [f for f in fixes if t_ini - 2_000_000_000 <= f[0] <= t_fin]

    print(f"  Sesión:        {args.session}")
    print(f"  Frames:        {n_total}  ({dur_s:.1f} s, "
          f"{n_total/max(dur_s,1e-9):.1f} fps nominales)")
    print(f"  Fixes de GPS:  {len(fixes)} en la ventana  "
          f"({len(fixes)/max(dur_s,1e-9):.2f} Hz)")
    print(f"  Máscara inferior (tablero o capó): {args.mask_bottom*100:.0f}%")
    if skip:
        print(f"  Frames descartados al inicio: {skip}"
              + (f"  (desde el segundo {args.start:.0f})" if args.start else ""))

    video_path = os.path.join(args.session, args.video)

    # Real dimensions, for the optical centre.
    import cv2
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    K = mobile_camera_matrix(w, h, args.fx, args.fy, args.cx, args.cy)
    print(f"  Imagen:        {w}x{h}")
    print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}\n")

    gps_buf = GpsBuffer()
    fe = VisualFrontEnd(K, mask_bottom=args.mask_bottom)
    est, scaler = (build_scale_worker(K, args.scale_hz, threaded=args.strict)
                   if args.scale == "depth" else (None, None))

    # The pipeline only measures times; the recorder also keeps every step,
    # to rebuild the route and to save it.
    recorder = TrackRecorder()
    keep = args.trajectory or args.save_frames
    pipe = RealtimePipeline(fe, gps_buf, strict=args.strict,
                            on_result=recorder if keep else None,
                            scale_worker=scaler)

    t0_data = int(frame_t[0])

    with ReplayFrameSource(video_path, frame_t, strict=args.strict,
                           max_frames=n_total, skip_start=skip) as src:
        if args.strict:
            gps_src = ReplayGpsSource(fixes, gps_buf, strict=True)
            gps_src.start(time.perf_counter(), t0_data)
        else:
            # Deterministic: every fix is available up front, but access stays
            # causal through GpsBuffer.latest_before().
            for t_ns, pos in fixes:
                gps_buf.push(t_ns, pos)

        m = pipe.run(src, t0_data)

    res = m.summary()
    print_results(res, args.strict, est, scaler)

    if scaler is not None:
        res["escala"] = {
            "enviadas": scaler.n_submitted,
            "descartadas_cola": scaler.n_dropped[0],
            "aceptadas": est.n_ok,
            "falladas": est.n_fail,
            "fuera_de_rango": est.n_rejected,
        }

    if args.strict:
        print(f"\n  Latencia extremo a extremo p50  : {res['e2e_ms_p50']:.1f} ms")
        print(f"                             p95  : {res['e2e_ms_p95']:.1f} ms")
    else:
        sim = simulate_drops(frame_t[:n_total], [s.total for s in m.stages])
        print("\n  SIMULACIÓN DEL COMPORTAMIENTO REAL (a partir de los tiempos medidos):")
        print(f"     Procesados / descartados     : "
              f"{sim['frames_procesados']} / {sim['frames_descartados']}"
              f"  ({sim['tasa_descarte_%']:.1f}%)")
        print(f"     Latencia e2e  p50 / p95      : "
              f"{sim['e2e_ms_p50']:.1f} / {sim['e2e_ms_p95']:.1f} ms")
        res["simulacion"] = sim

    e2e = res["e2e_ms_p95"] if args.strict else res["simulacion"]["e2e_ms_p95"]
    hz = res["hz_efectivo"] if args.strict else 1000.0 / max(res["proc_ms_p50"], 1e-9)
    print_verdict(e2e, hz)

    os.makedirs(args.out, exist_ok=True)
    mode = "strict" if args.strict else "deterministic"
    suffix = "_metric" if scaler is not None else ""

    if args.trajectory:
        plot_trajectory(recorder.track, args.out, mode, metric=scaler is not None)
    if args.save_frames:
        path = os.path.join(args.out, f"frames_{mode}{suffix}.csv")
        save_frames_csv(path, recorder.track, m)
        print(f"  Resultados por frame: {path}")

    path = os.path.join(args.out, f"replay_{mode}{suffix}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nMétricas guardadas: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
