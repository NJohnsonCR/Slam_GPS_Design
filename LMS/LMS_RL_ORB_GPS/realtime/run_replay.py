"""
Modo replay: ejecuta el pipeline de tiempo real sobre una sesión grabada,
respetando los tiempos reales entre frames.

Es el mismo sistema que correrá en vivo; solo cambia la fuente de datos. Sirve
para validar la arquitectura y medir latencia SIN depender de la app Android,
la red ni el hardware.

Responde con números la pregunta del OE1: ¿es viable operar en tiempo real?

Uso:
    # modo determinista (reproducible, para desarrollar y reportar)
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_replay \
        mobile_data/2025_03_11 --frames 900

    # modo estricto (comportamiento real, no determinista)
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_replay \
        mobile_data/2025_03_11 --frames 900 --estricto
"""

import argparse
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
from realtime.pipeline import (RealtimePipeline, VisualFrontEnd, simulate_drops)


# Calibración real del teléfono: fx/fy vienen de movie_metadata.csv y el centro
# óptico es el centro de la imagen. El pipeline offline usa por defecto los
# valores de KITTI, que para este video son incorrectos.
def camera_matrix_movil(width: int, height: int, fx: float = 899.0):
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fx, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _umeyama(est, ref):
    """Alineación con escala. Estándar para evaluar VO monocular, que no tiene
    escala absoluta ni conoce la orientación del mapa."""
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    ec, rc = est - est.mean(0), ref - ref.mean(0)
    U, S, Vt = np.linalg.svd(ec.T @ rc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var = np.sum(ec ** 2)
    s = np.sum(S) / var if var > 1e-10 else 1.0
    return s * (ec @ R.T) + ref.mean(0), s


def graficar_trayectoria(traza, out_dir, modo):
    """
    Reconstruye el recorrido a partir de las poses relativas acumuladas.

    ADVERTENCIA sobre la interpretación: en los datos móviles NO hay ground
    truth. El GPS del teléfono tiene varios metros de error, así que comparar
    contra él dice si la FORMA del recorrido es plausible, no cuán exacto es.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # cadena de VO libre, con paso de norma unitaria (sin escala métrica)
    P = np.eye(4)
    vo = [np.zeros(3)]
    for R, t in traza["rel"]:
        rel = np.eye(4)
        rel[:3, :3] = R
        rel[:3, 3] = t
        P = P @ rel
        vo.append(P[:3, 3].copy())
    vo = np.array(vo[1:])

    gps = np.array([g for g in traza["gps"] if g is not None])
    if len(gps) < 10 or len(vo) < 10:
        print("\n  (no hay suficientes datos para graficar la trayectoria)")
        return

    n = min(len(vo), len(gps))
    vo_al, escala = _umeyama(vo[:n], gps[:n])
    err = np.linalg.norm(vo_al - gps[:n], axis=1)
    dist = float(np.sum(np.linalg.norm(np.diff(gps[:n], axis=0), axis=1)))

    print("\n" + "-" * 78)
    print("TRAYECTORIA RECONSTRUIDA")
    print("-" * 78)
    print(f"  Distancia recorrida según GPS : {dist:.0f} m")
    print(f"  Desviación VO vs GPS  RMSE    : {np.sqrt((err**2).mean()):.2f} m")
    print(f"                        mediana : {np.median(err):.2f} m")
    print(f"                        máxima  : {err.max():.2f} m")
    print(f"  Factor de escala de alineación: {escala:.4f}")
    print("  NOTA: el GPS del teléfono NO es ground truth (error de varios metros).")
    print("        Esto mide parecido de forma, no exactitud.")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].plot(gps[:n, 0], gps[:n, 1], "-", lw=2, color="#F2A03D", label="GPS del teléfono (1 Hz)")
    ax[0].plot(vo_al[:, 0], vo_al[:, 1], "--", lw=1.6, color="#4FD1C5",
               label="Odometría visual (alineada)")
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Y (m)")
    ax[0].set_title(f"Recorrido — {dist:.0f} m")
    ax[0].axis("equal"); ax[0].grid(alpha=0.3); ax[0].legend()

    ax[1].plot(err, lw=1.2, color="#B85042")
    ax[1].axhline(np.median(err), ls="--", color="gray",
                  label=f"mediana {np.median(err):.1f} m")
    ax[1].set_xlabel("keyframe"); ax[1].set_ylabel("desviación (m)")
    ax[1].set_title("Desviación entre VO y GPS")
    ax[1].grid(alpha=0.3); ax[1].legend()

    fig.suptitle(f"Modo replay ({modo}) — trayectoria reconstruida")
    fig.tight_layout()
    png = os.path.join(out_dir, f"trayectoria_{modo}.png")
    fig.savefig(png, dpi=130)
    print(f"  Gráfico: {png}")


def main():
    ap = argparse.ArgumentParser(description="Modo replay del pipeline en tiempo real")
    ap.add_argument("session", help="Directorio de la sesión (p.ej. mobile_data/2025_03_11)")
    ap.add_argument("--video", default="movie.mp4")
    ap.add_argument("--frames", type=int, default=None, help="Máximo de frames")
    ap.add_argument("--estricto", action="store_true",
                    help="Duerme y descarta de verdad (no determinista)")
    ap.add_argument("--recorte-inferior", type=float, default=0.14,
                    help="Fracción inferior a recortar: el tablero del carro")
    ap.add_argument("--fx", type=float, default=899.0)
    ap.add_argument("--out", default="resultados/realtime")
    ap.add_argument("--trayectoria", action="store_true",
                    help="Reconstruir y graficar el recorrido además de medir tiempos")
    args = ap.parse_args()

    print("=" * 78)
    print(f"MODO REPLAY — {'ESTRICTO' if args.estricto else 'DETERMINISTA'}")
    print("=" * 78)

    frame_t, fixes = load_mobile_session(args.session, latlon_to_utm)
    n_total = len(frame_t) if args.frames is None else min(args.frames, len(frame_t))
    dur_s = (frame_t[n_total - 1] - frame_t[0]) / 1e9

    # solo los fixes que caen dentro de la ventana de frames que se procesa
    t_ini, t_fin = int(frame_t[0]), int(frame_t[n_total - 1])
    fixes = [f for f in fixes if t_ini - 2_000_000_000 <= f[0] <= t_fin]

    print(f"  Sesión:        {args.session}")
    print(f"  Frames:        {n_total}  ({dur_s:.1f} s, {n_total/max(dur_s,1e-9):.1f} fps nominales)")
    print(f"  Fixes de GPS:  {len(fixes)} en la ventana  ({len(fixes)/max(dur_s,1e-9):.2f} Hz)")
    print(f"  Recorte inferior (tablero): {args.recorte_inferior*100:.0f}%")

    video_path = os.path.join(args.session, args.video)

    # dimensiones reales tras el recorte, para el centro óptico
    import cv2
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    h = int(h_full * (1.0 - args.recorte_inferior))
    K = camera_matrix_movil(w, h_full, args.fx)      # cy respecto de la imagen completa
    print(f"  Imagen:        {w}x{h_full}  ->  {w}x{h} tras recorte")
    print(f"  fx={args.fx:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}\n")

    gps_buf = GpsBuffer()
    fe = VisualFrontEnd(K)

    # Acumulación de la trayectoria (opcional). El pipeline solo mide tiempos;
    # esto engancha un callback para además reconstruir el recorrido.
    traza = {"rel": [], "gps": [], "t_ns": []}

    def acumular(frame, R, t, n_m, n_i, fix):
        traza["rel"].append((R.copy(), t.copy()))
        traza["t_ns"].append(frame.t_ns)
        traza["gps"].append(None if fix is None else fix[1].copy())

    pipe = RealtimePipeline(fe, gps_buf, strict=args.estricto,
                            on_result=acumular if args.trayectoria else None)

    t0_data = int(frame_t[0])

    with ReplayFrameSource(video_path, frame_t, crop_bottom=args.recorte_inferior,
                           strict=args.estricto, max_frames=n_total) as src:
        if args.estricto:
            gps_src = ReplayGpsSource(fixes, gps_buf, strict=True)
            gps_src.start(time.perf_counter(), t0_data)
        else:
            # determinista: todos los fixes disponibles, pero el acceso sigue
            # siendo causal gracias a GpsBuffer.latest_before()
            for t_ns, pos in fixes:
                gps_buf.push(t_ns, pos)

        m = pipe.run(src, t0_data)

    res = m.summary()

    print("-" * 78)
    print("RESULTADOS")
    print("-" * 78)
    print(f"  Frames procesados / descartados : {res['frames_procesados']} / {res['frames_descartados']}"
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

    if args.estricto:
        print(f"\n  Latencia extremo a extremo p50  : {res['e2e_ms_p50']:.1f} ms")
        print(f"                             p95  : {res['e2e_ms_p95']:.1f} ms")
    else:
        sim = simulate_drops(frame_t[:n_total],
                             [s.total for s in m.stages])
        print("\n  SIMULACIÓN DEL COMPORTAMIENTO REAL (a partir de los tiempos medidos):")
        print(f"     Procesados / descartados     : {sim['frames_procesados']} / {sim['frames_descartados']}"
              f"  ({sim['tasa_descarte_%']:.1f}%)")
        print(f"     Latencia e2e  p50 / p95      : {sim['e2e_ms_p50']:.1f} / {sim['e2e_ms_p95']:.1f} ms")
        res["simulacion"] = sim

    # ---- veredicto contra el objetivo de tiempo real ----
    objetivo_ms, objetivo_hz = 150.0, 10.0
    e2e = res.get("simulacion", res)["e2e_ms_p95"] if not args.estricto else res["e2e_ms_p95"]
    hz = res["hz_efectivo"] if args.estricto else 1000.0 / max(res["proc_ms_p50"], 1e-9)

    print("\n" + "=" * 78)
    print(f"OBJETIVO PROPUESTO: latencia p95 < {objetivo_ms:.0f} ms  y  ≥ {objetivo_hz:.0f} Hz sostenidos")
    ok_lat, ok_hz = e2e < objetivo_ms, hz >= objetivo_hz
    print(f"  Latencia p95 : {e2e:7.1f} ms   {'CUMPLE' if ok_lat else 'NO CUMPLE'}")
    print(f"  Frecuencia   : {hz:7.1f} Hz   {'CUMPLE' if ok_hz else 'NO CUMPLE'}")
    print("=" * 78)

    os.makedirs(args.out, exist_ok=True)
    modo = "estricto" if args.estricto else "determinista"

    if args.trayectoria:
        graficar_trayectoria(traza, args.out, modo)

    path = os.path.join(args.out, f"replay_{modo}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nMétricas guardadas: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
