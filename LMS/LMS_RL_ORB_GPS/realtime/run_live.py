"""
Live mode: the real-time pipeline fed by the Android app over the network.

Same pipeline as replay; only the sources change. Live always behaves like
strict replay: real time, real drops, not deterministic.

Usage:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_live 192.168.100.108

    # also record on the phone, with metric scale from depth, for 60 s
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_live 192.168.100.108 \
        --record --scale depth --duration 60

The session ends with Ctrl+C, after --duration seconds, or when no frames
arrive for 5 s. Each one is saved in its own folder under --out.
"""

import argparse
import json
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_LMS_RL = os.path.abspath(os.path.join(_THIS, ".."))
_ROOT = os.path.abspath(os.path.join(_LMS_RL, "..", ".."))
for _p in (_ROOT, _LMS_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realtime.sources import GpsBuffer, LiveFrameSource, LiveGpsSource, read_hello
from realtime.pipeline import RealtimePipeline, VisualFrontEnd
from realtime.run_replay import (TrackRecorder, build_scale_worker, plot_trajectory,
                                 print_results, print_verdict, save_frames_csv)

# Same header as the phone's location.csv, so both files read the same way.
LOCATION_HEADER = ("Timestamp[nanosecond],latitude[degrees],longitude[degrees],"
                   "altitude[meters],speed[meters/second],Unix time[nanosecond]")


def warm_up(K, width, height, est=None):
    """
    Run the models once on synthetic frames before connecting.

    The first ORB call pays a one-off initialisation (~200 ms), and so does
    the first depth inference (~0.3 s, several seconds with cold caches).
    Paid here, it does not reach the first live frames as latency.
    """
    rng = np.random.default_rng(0)
    noise = cv2.GaussianBlur(
        rng.integers(0, 256, (height + 8, width + 8), dtype=np.uint8), (0, 0), 1.5)
    fe = VisualFrontEnd(K)
    # Two shifted copies, so matching and pose estimation run as well.
    for dy, dx in ((0, 0), (4, 6)):
        img = cv2.cvtColor(noise[dy:dy + height, dx:dx + width], cv2.COLOR_GRAY2BGR)
        fe.process(img)
    if est is not None:
        est.depth_map(img)


def status_loop(stop, src, gps, pipe, scaler):
    """One console line per second while the session runs."""
    t0 = time.monotonic()
    prev = (0, 0, 0, 0)
    while not stop.wait(1.0):
        m = pipe.metrics
        now = (src.n_received, src.n_missing, m.processed, len(gps.fixes))
        received, missing, processed, fixes = (a - b for a, b in zip(now, prev))
        prev = now

        line = (f"  {time.monotonic() - t0:4.0f} s | video {received:2d} fps, "
                f"faltan {missing:2d} | procesados {processed:2d}")
        if processed:
            line += f", latencia {np.median(m.e2e_ms[-processed:]):4.0f} ms"
        line += f" | GPS {fixes} fix"
        if gps.fixes and src.last_t_ns is not None:
            t_fix, speed = gps.fixes[-1][0], gps.fixes[-1][4]
            line += f" (hace {(src.last_t_ns - t_fix) / 1e9:3.1f} s, {speed:.1f} m/s)"
        if scaler is not None and scaler.velocity is not None:
            line += f" | cámara {scaler.velocity:.1f} m/s"
        if gps.recording_folder:
            line += " | grabando"
        if gps.last_reply is not None and time.monotonic() - gps.last_reply > 6.0:
            line += " | sin respuesta UDP"
        print(line, flush=True)


def print_live_results(res, src, gps):
    """The phone's side of the link, which only exists live."""
    sent = src.n_received + src.n_missing
    print()
    print(f"  Frames recibidos del teléfono   : {src.n_received}"
          f"  ({src.n_missing} no enviados por el teléfono, "
          f"{100 * src.n_missing / max(sent, 1):.1f}%)")
    if src.transit_ms:
        p50, p95 = np.percentile(src.transit_ms, [50, 95])
        print(f"  Captura → llegada   p50 / p95   : {p50:.0f} / {p95:.0f} ms")
    if "pc_ms_p50" in res:
        print(f"  Llegada → resultado p50 / p95   : "
              f"{res['pc_ms_p50']:.0f} / {res['pc_ms_p95']:.0f} ms   (solo la PC)")
        print(f"  Captura → resultado p50 / p95   : "
              f"{res['e2e_ms_p50']:.0f} / {res['e2e_ms_p95']:.0f} ms")
        print("     (la captura usa el reloj del teléfono y el resultado el de la PC:")
        print("      incluye el desfase entre los dos relojes)")
    n = len(gps.fixes)
    span_s = (gps.fixes[-1][0] - gps.fixes[0][0]) / 1e9 if n >= 2 else 0.0
    rate = f"{(n - 1) / span_s:.2f} Hz" if span_s > 0 else "—"
    print(f"  Fixes de GPS recibidos          : {n}  ({rate})")
    if gps.n_bad or src.n_bad:
        print(f"  Mensajes inválidos              : {gps.n_bad} UDP, {src.n_bad} JPEG")


def save_session(out_dir, track, m, res, gps):
    """Per-frame results, received fixes and metrics, to compare later."""
    os.makedirs(out_dir, exist_ok=True)
    save_frames_csv(os.path.join(out_dir, "frames.csv"), track, m)

    with open(os.path.join(out_dir, "gps.csv"), "w") as f:
        f.write(LOCATION_HEADER + "\n")
        for fix in gps.fixes:
            f.write(",".join(str(v) for v in fix) + "\n")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  Sesión guardada en: {out_dir}")


def run_session(args, hello, gps, gps_buf):
    if gps.wait_reply():
        print("  GPS:       el teléfono responde por UDP")
    else:
        print("  GPS:       el teléfono no responde por UDP; sigue solo con video")
    print()

    K = hello.camera_matrix()
    est, scaler = (build_scale_worker(K, args.scale_hz, threaded=True)
                   if args.scale == "depth" else (None, None))
    t0 = time.perf_counter()
    warm_up(K, hello.width, int(hello.height * (1.0 - args.crop_bottom)), est)
    print(f"  Precalentamiento: {time.perf_counter() - t0:.1f} s")

    src = LiveFrameSource(args.phone_ip, hello, args.video_port, args.crop_bottom)
    recorder = TrackRecorder()
    track = recorder.track
    pipe = RealtimePipeline(VisualFrontEnd(K), gps_buf, strict=True,
                            on_result=recorder, scale_worker=scaler)

    def on_sigint(*_):
        # The first Ctrl+C ends the session and still saves it; a second one
        # aborts.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        src.stop()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    stop_status = threading.Event()
    status = threading.Thread(target=status_loop, daemon=True,
                              args=(stop_status, src, gps, pipe, scaler))
    timer = threading.Timer(args.duration, src.stop) if args.duration else None
    recording = False
    folder = None
    try:
        if args.record:
            if not gps.command("START"):
                print("  No se pudo iniciar la grabación en el teléfono.")
                print("  ¿Está la app en primer plano, en la pantalla de video?")
                return 1
            recording = True
            folder = gps.recording_folder
            print(f"  Grabando en el teléfono, carpeta {folder}")
        print("  Ctrl+C para terminar.\n")
        signal.signal(signal.SIGINT, on_sigint)
        if timer is not None:
            timer.start()
        status.start()
        m = pipe.run(src)
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        stop_status.set()
        if status.is_alive():
            status.join()
        if timer is not None:
            timer.cancel()
        if recording:
            print("  Grabación detenida en el teléfono." if gps.command("STOP")
                  else "  No se pudo detener la grabación: detenerla desde el teléfono.")

    print(f"\n  Sesión terminada: {src.end_reason or 'sin frames nuevos'}.\n")
    if m.processed == 0:
        print("  No se procesó ningún frame.")
        return 1

    res = m.summary()
    print_results(res, True, est, scaler)
    print_live_results(res, src, gps)
    print_verdict(res["e2e_ms_p95"], res["hz_efectivo"])

    if scaler is not None:
        res["escala"] = {
            "enviadas": scaler.n_submitted,
            "descartadas_cola": scaler.n_dropped[0],
            "aceptadas": est.n_ok,
            "falladas": est.n_fail,
            "fuera_de_rango": est.n_rejected,
        }
    transit = src.transit_ms or [float("nan")]
    res["en_vivo"] = {
        "telefono": args.phone_ip,
        "camara": hello._asdict(),
        "carpeta_telefono": folder,
        "fin": src.end_reason,
        "frames_recibidos": src.n_received,
        "frames_no_enviados": src.n_missing,
        "jpeg_invalidos": src.n_bad,
        "captura_llegada_ms_p50": float(np.percentile(transit, 50)),
        "captura_llegada_ms_p95": float(np.percentile(transit, 95)),
        "fixes_gps": len(gps.fixes),
        "udp_invalidos": gps.n_bad,
        "utm_epsg": gps.epsg,
        "utm_origen": None if gps.origin is None else gps.origin.tolist(),
        "recorte_inferior": args.crop_bottom,
    }

    out_dir = os.path.join(args.out, f"live_{stamp}")
    save_session(out_dir, track, m, res, gps)
    plot_trajectory(track, out_dir, "live", metric=scaler is not None)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Pipeline en tiempo real con el teléfono en vivo")
    ap.add_argument("phone_ip", help="IP del teléfono (la muestra la pantalla de video de la app)")
    ap.add_argument("--video-port", type=int, default=5000)
    ap.add_argument("--gps-port", type=int, default=5001)
    ap.add_argument("--record", action="store_true",
                    help="Grabar también en el teléfono (START al empezar, STOP al terminar)")
    ap.add_argument("--duration", type=float, default=None,
                    help="Duración de la sesión en segundos; sin esto, hasta Ctrl+C")
    ap.add_argument("--crop-bottom", type=float, default=0.0,
                    help="Fracción inferior a recortar si se ve el tablero")
    ap.add_argument("--scale", choices=["none", "depth"], default="none",
                    help="Fuente de la escala métrica. 'depth' activa el "
                         "estimador monocular en un hilo aparte")
    ap.add_argument("--scale-hz", type=float, default=3.0,
                    help="Ritmo del estimador de escala (el modelo tarda ~100 ms)")
    ap.add_argument("--out", default="resultados/realtime")
    args = ap.parse_args()

    print("=" * 78)
    print("MODO EN VIVO")
    print("=" * 78)

    try:
        hello = read_hello(args.phone_ip, args.video_port)
    except OSError as e:
        print(f"  No se pudo leer la cámara del teléfono ({e}).")
        print("  Revisar: misma red WiFi, IP correcta y la app en la pantalla de video.")
        return 1
    print(f"  Teléfono:  {args.phone_ip}  (video {args.video_port}, GPS {args.gps_port})")
    print(f"  Cámara:    {hello.width}x{hello.height} a {hello.fps} fps  "
          f"fx {hello.fx:.2f}  fy {hello.fy:.2f}  cx {hello.cx:.2f}  cy {hello.cy:.2f}")
    if args.crop_bottom > 0:
        print(f"  Recorte inferior (tablero): {args.crop_bottom * 100:.0f}%")

    gps_buf = GpsBuffer()
    gps = LiveGpsSource(args.phone_ip, gps_buf, args.gps_port)
    gps.start()
    try:
        return run_session(args, hello, gps, gps_buf)
    except KeyboardInterrupt:
        print("\n  Sesión abortada.")
        return 130
    finally:
        gps.stop()


if __name__ == "__main__":
    sys.exit(main())
