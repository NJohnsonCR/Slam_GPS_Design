"""
Live mode: the real-time pipeline fed by the Android app over the network.

Same pipeline as replay; only the sources change. Live always behaves like
strict replay: real time, real drops, not deterministic. The session itself
lives in session.py, shared with the window (live_window.py); this is its
console.

Usage:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_live 192.168.100.108

    # also record on the phone, with metric scale from depth, for 60 s
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_live 192.168.100.108 \
        --record --scale depth --duration 60

The session ends with Ctrl+C, after --duration seconds, or when no frames
arrive for 5 s. Each one is saved in its own folder under --out.
"""

import argparse
import os
import signal
import sys
import threading
import time

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_LMS_RL = os.path.abspath(os.path.join(_THIS, ".."))
_ROOT = os.path.abspath(os.path.join(_LMS_RL, "..", ".."))
for _p in (_ROOT, _LMS_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realtime.run_replay import print_results, print_verdict
from realtime.session import LiveSession, SessionError


def status_line(st):
    """One console line from LiveSession.status()."""
    line = (f"  {st['elapsed_s']:4.0f} s | video {st['video_fps']:2.0f} fps, "
            f"faltan {st['missing']:2d} | procesados {st['processed_fps']:2.0f}")
    # Only the PC's share: the capture time comes from the phone's clock.
    if st["pc_latency_ms"] is not None:
        line += f", latencia en la PC {st['pc_latency_ms']:3.0f} ms"
    line += f" | GPS {st.get('gps_fixes', 0)} fix"
    if "fix_age_s" in st:
        line += f" (hace {st['fix_age_s']:3.1f} s, {st['gps_speed']:.1f} m/s)"
    if st["camera_speed"] is not None:
        line += f" | cámara {st['camera_speed']:.1f} m/s"
    if st["stopped"]:
        line += " (detenido)"
    if st.get("recording"):
        line += " | grabando"
    if st.get("udp_silent"):
        line += " | sin respuesta UDP"
    return line


def status_loop(stop, session):
    """One console line per second while the session runs."""
    while not stop.wait(1.0):
        st = session.status()
        if "elapsed_s" in st:
            print(status_line(st), flush=True)


def print_live_results(res, src, gps):
    """The phone's side of the link, which only exists live."""
    sent = src.n_received + src.n_missing
    print()
    print(f"  Frames recibidos del teléfono   : {src.n_received}"
          f"  ({src.n_missing} no enviados por el teléfono, "
          f"{100 * src.n_missing / max(sent, 1):.1f}%)")
    if "pc_ms_p50" in res:
        print(f"  Latencia en la PC   p50 / p95   : "
              f"{res['pc_ms_p50']:.0f} / {res['pc_ms_p95']:.0f} ms   (llegada → resultado)")
    if src.transit_ms:
        # The capture time comes from the phone's clock, whose offset from the
        # PC's is not measured: a reference, not the real latency.
        p50, p95 = np.percentile(src.transit_ms, [50, 95])
        print("  Referencia, desde la captura (reloj del teléfono):")
        print(f"     Captura → llegada   p50 / p95 : {p50:.0f} / {p95:.0f} ms")
        print(f"     Captura → resultado p50 / p95 : "
              f"{res['e2e_ms_p50']:.0f} / {res['e2e_ms_p95']:.0f} ms")
        print("     (incluyen el desfase entre los relojes del teléfono y de la PC,")
        print("      que cambia de un día a otro: no son la latencia real)")
    n = len(gps.fixes)
    span_s = (gps.fixes[-1][0] - gps.fixes[0][0]) / 1e9 if n >= 2 else 0.0
    rate = f"{(n - 1) / span_s:.2f} Hz" if span_s > 0 else "—"
    print(f"  Fixes de GPS recibidos          : {n}  ({rate})")
    if gps.n_bad or src.n_bad:
        print(f"  Mensajes inválidos              : {gps.n_bad} UDP, {src.n_bad} JPEG")


def run(session, args):
    """Prepare, run until Ctrl+C or the end, and report. Returns the exit code."""
    print("  GPS:       el teléfono responde por UDP" if session.gps_ok
          else "  GPS:       el teléfono no responde por UDP; sigue solo con video")
    print()
    t0 = time.perf_counter()
    session.prepare()
    print(f"  Precalentamiento: {time.perf_counter() - t0:.1f} s")

    def on_sigint(*_):
        # The first Ctrl+C ends the session and still saves it; a second one
        # aborts.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        session.stop()

    stop_status = threading.Event()
    status = threading.Thread(target=status_loop, args=(stop_status, session), daemon=True)
    try:
        if args.record:
            print("  Grabando también en el teléfono.")
        print("  Ctrl+C para terminar.\n")
        signal.signal(signal.SIGINT, on_sigint)
        status.start()
        session.run()
    except SessionError as e:
        print(f"  {e}")
        return 1
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        stop_status.set()
        if status.is_alive():
            status.join()
        if session.folder:
            print(f"  Carpeta de la grabación en el teléfono: {session.folder}")
        if session.recording_stopped is not None:
            print("  Grabación detenida en el teléfono." if session.recording_stopped
                  else "  No se pudo detener la grabación: detenerla desde el teléfono.")

    print(f"\n  Sesión terminada: {session.end_reason or 'sin frames nuevos'}.\n")
    if session.metrics.processed == 0:
        print("  No se procesó ningún frame.")
        return 1

    res = session.summary()
    print_results(res, True, session.est, session.scaler)
    print_live_results(res, session.source, session.gps)
    print_verdict(res["pc_ms_p95"], res["hz_efectivo"], what="Latencia p95 en la PC",
                  note="La latencia desde la captura no entra en el veredicto: sin medir el\n"
                       "desfase entre los relojes del teléfono y de la PC no se puede verificar.")
    session.save(res)
    print(f"\n  Sesión guardada en: {session.out_dir}")
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
    ap.add_argument("--mask-bottom", type=float, default=0.25,
                    help="Fracción inferior donde ORB no busca puntos: el capó y el "
                         "tablero con el teléfono en el carro (0 si no se ven)")
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

    session = LiveSession(args.phone_ip, args.video_port, args.gps_port, args.record,
                          mask_bottom=args.mask_bottom, scale=args.scale == "depth",
                          scale_hz=args.scale_hz, duration=args.duration, out=args.out)
    try:
        session.connect()
    except OSError as e:
        print(f"  No se pudo leer la cámara del teléfono ({e}).")
        print("  Revisar: misma red (WiFi o cable), IP correcta y la app en la pantalla de video.")
        session.close()
        return 1
    h = session.hello
    print(f"  Teléfono:  {args.phone_ip}  (video {args.video_port}, GPS {args.gps_port})")
    print(f"  Cámara:    {h.width}x{h.height} a {h.fps} fps  "
          f"fx {h.fx:.2f}  fy {h.fy:.2f}  cx {h.cx:.2f}  cy {h.cy:.2f}")
    if args.mask_bottom > 0:
        print(f"  Máscara inferior (tablero o capó): {args.mask_bottom * 100:.0f}%")

    try:
        return run(session, args)
    except KeyboardInterrupt:
        print("\n  Sesión abortada.")
        return 130
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
