"""
Check what the phone sends, without the pipeline: video, GPS, IMU and the
answers to TIME, for a few seconds. If this works and the pipeline does not,
the problem is the pipeline's; if this fails too, it is the app's or the
network's.

It also checks that frames and IMU samples share the phone's boot clock: with
the offset from the quickest TIME query, both have to arrive a little after
their timestamps, never before.

Usage:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.phone_check 192.168.100.113
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.phone_check 192.168.100.113 --seconds 30
"""

import argparse
import socket
import struct
import sys
import threading
import time

import numpy as np

HEADER = struct.Struct(">IQQ")      # JPEG length, boot-clock ns, Unix ns


def read_video(ip, port, stop, frames, info):
    try:
        sock = socket.create_connection((ip, port), timeout=5)
    except OSError as e:
        info["error"] = str(e)
        return
    with sock, sock.makefile("rb") as stream:
        info["hello"] = stream.readline(256).decode("ascii", errors="replace").strip()
        while not stop.is_set():
            head = stream.read(HEADER.size)
            if len(head) < HEADER.size:
                break
            n, t_ns, unix_ns = HEADER.unpack(head)
            if len(stream.read(n)) < n:
                break
            frames.append((t_ns, unix_ns, time.time_ns()))


def read_udp(ip, port, stop, got):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.05)
    last_sub = last_time = 0.0
    while not stop.is_set():
        now = time.monotonic()
        try:
            if now - last_sub >= 2.0:
                sock.sendto(b"SUBSCRIBE", (ip, port))
                last_sub = now
            if now - last_time >= 0.1:
                sock.sendto(f"TIME,{time.time_ns()}".encode("ascii"), (ip, port))
                last_time = now
            data, _ = sock.recvfrom(2048)
        except socket.timeout:
            continue
        except OSError:
            # Unreachable phone: keep trying until the time is up.
            time.sleep(0.1)
            continue
        back = time.time_ns()
        p = data.decode("ascii", errors="replace").strip().split(",")
        try:
            if p[0] == "IMU" and len(p) == 9:
                got["imu"].append((int(p[1]), back))
            elif p[0] == "GPS" and len(p) == 7:
                got["gps"].append((int(p[1]), float(p[5])))
            elif p[0] == "TIME" and len(p) == 4:
                got["time"].append((int(p[1]), int(p[2]), int(p[3]), back))
            else:
                got["other"].add(p[0])
        except ValueError:
            got["other"].add("inválido")
    sock.close()


def main():
    ap = argparse.ArgumentParser(description="Revisa lo que manda el teléfono, sin el pipeline")
    ap.add_argument("phone_ip", help="IP del teléfono (la muestra la pantalla de video de la app)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--video-port", type=int, default=5000)
    ap.add_argument("--gps-port", type=int, default=5001)
    args = ap.parse_args()

    frames, info = [], {}
    got = {"imu": [], "gps": [], "time": [], "other": set()}
    stop = threading.Event()
    threads = [threading.Thread(target=read_video, daemon=True,
                                args=(args.phone_ip, args.video_port, stop, frames, info)),
               threading.Thread(target=read_udp, daemon=True,
                                args=(args.phone_ip, args.gps_port, stop, got))]
    print(f"  Escuchando al teléfono {args.phone_ip} durante {args.seconds:.0f} s...")
    for th in threads:
        th.start()
    time.sleep(args.seconds)
    stop.set()
    for th in threads:
        th.join(timeout=2)

    secs, ok = args.seconds, True
    print()
    if "error" in info:
        print(f"  Video : NO se pudo conectar ({info['error']}). ¿IP correcta, misma red, "
              f"app en la pantalla de video?")
        ok = False
    else:
        print(f"  Video : {len(frames) / secs:.1f} fps  ({info.get('hello', 'sin HELLO')})")
        ok &= len(frames) > 0
    imu, gps, times = got["imu"], got["gps"], got["time"]
    if not (imu or gps or times or "SUBSCRIBED" in got["other"]):
        print("  UDP   : el teléfono NO responde (ni GPS, ni IMU, ni hora). ¿App en la "
              "pantalla de video, puerto correcto?")
        print("\n  Hay problemas: ver arriba.")
        return 1
    print(f"  GPS   : {len(gps)} fixes ({len(gps) / secs:.2f} Hz)"
          + (f", el último a {gps[-1][1]:.1f} m/s" if gps else
             " -- bajo techo puede tardar; acercarse a una ventana"))
    if imu:
        gaps = np.diff([t for t, _ in imu]) / 1e6
        print(f"  IMU   : {len(imu) / secs:.1f} muestras/s, intervalo máximo "
              f"{gaps.max() if len(gaps) else 0:.0f} ms")
    else:
        print("  IMU   : NO llega (el filtro giraría con la cámara)")
        ok = False
    if not times:
        print("  Hora  : el teléfono NO responde a TIME (sin ella, la latencia desde la "
              "captura no es confiable)")
        print("\n  " + ("Todo llega." if ok else "Hay problemas: ver arriba."))
        return 0 if ok else 1

    T = np.array(times, dtype=np.int64)
    trip = (T[:, 3] - T[:, 0]) / 1e6
    mid = (T[:, 0] + T[:, 3]) // 2
    k = int(np.argmin(trip))
    boot_off = int(T[k, 2] - mid[k])        # phone boot clock minus PC wall clock
    print(f"  Hora  : {len(times)} respuestas; ida y vuelta mínima {trip[k]:.1f} ms, "
          f"mediana {np.median(trip):.1f} ms")
    print(f"          la hora Unix del teléfono va {(T[k, 1] - mid[k]) / 1e6:+.0f} ms "
          f"respecto de la PC")
    # Same clock: with the measured offset, both arrive shortly after their stamps.
    for name, rows in (("frames", [(f[0], f[2]) for f in frames]), ("IMU", imu)):
        if not rows:
            continue
        late = np.array([(back - (t - boot_off)) / 1e6 for t, back in rows])
        p50, low = np.median(late), np.min(late)
        good = low > -trip[k] and p50 < 500
        ok &= good
        print(f"  Reloj : {name} llegan {p50:.0f} ms después de su marca (mínimo {low:.0f})"
              + ("" if good else "  <- NO cuadra: ¿otro reloj?"))
    print("\n  " + ("Todo llega, en el mismo reloj." if ok else "Hay problemas: ver arriba."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
