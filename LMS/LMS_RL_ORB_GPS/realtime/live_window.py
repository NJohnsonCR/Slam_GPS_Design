"""
Window of the real-time pipeline: the video, live from the phone or from a
recording, with the session's status and a map of the GPS and the fusion,
started and stopped with buttons instead of commands.

The session (session.py) runs on a worker thread. tkinter is only touched
from the main thread, which polls the session: the newest frame about 15
times per second, the map twice and the status once per second.

The map draws the two tracks that matter while driving: the fusion (the
official track) and the GPS (its reference), in metres east and north of the
first fix, without a street background (decided on 5/10/2026). The camera
alone and the camera with the gyroscope are diagnostics: they are in the
plot saved at the end (run_replay.plot_trajectory).

Usage:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.live_window

It also opens from interfaz.py, with the "SLAM GPS con realtime" button.
"""

import json
import os
import sys
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk

_THIS = os.path.dirname(os.path.abspath(__file__))
_LMS_RL = os.path.abspath(os.path.join(_THIS, ".."))
_ROOT = os.path.abspath(os.path.join(_LMS_RL, "..", ".."))
for _p in (_ROOT, _LMS_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realtime.run_replay import COLOR_FUSED, COLOR_GPS
from realtime.session import LiveSession, ReplaySession, SessionError

VIDEO_W, VIDEO_H = 640, 360
VIDEO_MS = 66                   # ~15 frames per second on screen; processing stays at 30
MAP_H = 300
MAP_MS = 500
MASK_COLOR = "#f0b429"
LABEL = "Pipeline.TLabel"
STATUS_MS = 1000
OUT_DIR = os.path.join(_ROOT, "resultados", "realtime")
SETTINGS = os.path.join(OUT_DIR, "interfaz.json")      # last values typed, per user
DEFAULTS = {"mode": "live", "ip": "", "record": False, "recording": "",
            "start": 0.0, "scale": True, "mask": 25}


class TrackMap:
    """
    The GPS and the fusion on a canvas, in metres (east right, north up, the
    same scale on both axes), fitted to what has been driven so far.
    """

    PAD = 26
    MIN_SPAN_M = 60.0       # before the car moves, a fix's noise would fill the map
    MIN_STEP_M = 0.5        # fused points closer than this to the last kept one are skipped
    SCALE_BARS = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)

    def __init__(self, parent, width, height):
        self.w, self.h = width, height
        self.canvas = tk.Canvas(parent, width=width, height=height, bg="white",
                                highlightthickness=1, highlightbackground="#d0d0cc")
        self.reset()

    def reset(self):
        c = self.canvas
        c.delete("all")
        self.gps, self.fused = [], []
        self.n_fixes = self.n_frames = 0
        self._last_fused = None     # last fused point kept for the line
        self._now = None            # newest fused position, kept or not
        self._idle = c.create_text(self.w / 2, self.h / 2, fill="#9a9a96",
                                   text="Mapa: el GPS y la fusión, en metros")
        self._gps_line = c.create_line(0, 0, 0, 0, fill=COLOR_GPS, width=3, state="hidden")
        self._fused_line = c.create_line(0, 0, 0, 0, fill=COLOR_FUSED, width=2, state="hidden")
        self._start = c.create_oval(0, 0, 0, 0, fill="#0b0b0b", outline="", state="hidden")
        self._gps_now = c.create_oval(0, 0, 0, 0, fill=COLOR_GPS, outline="white", state="hidden")
        self._fused_now = c.create_oval(0, 0, 0, 0, fill=COLOR_FUSED, outline="white",
                                        state="hidden")
        self._bar = c.create_line(0, 0, 0, 0, width=2, fill="#0b0b0b", state="hidden")
        self._bar_text = c.create_text(0, 0, anchor="sw", font=("TkDefaultFont", 8),
                                       state="hidden")
        c.create_text(self.w - 10, 8, anchor="ne", text="N ↑", font=("TkDefaultFont", 9, "bold"))
        for row, (color, label) in enumerate(((COLOR_GPS, "GPS (referencia)"),
                                              (COLOR_FUSED, "Fusión (trazo oficial)"))):
            y = 12 + 16 * row
            c.create_line(8, y, 28, y, fill=color, width=3)
            c.create_text(34, y, anchor="w", text=label, font=("TkDefaultFont", 8))

    def add(self, gps, fused):
        """New fixes and fused rows from PipelineSession.track_since."""
        self.n_fixes += len(gps)
        self.n_frames += len(fused)
        self.gps.extend(gps)
        for p in fused:
            if p is None:
                continue
            self._now = p
            last = self._last_fused
            if last is None or (p[0] - last[0]) ** 2 + (p[1] - last[1]) ** 2 >= self.MIN_STEP_M ** 2:
                self.fused.append(p)
                self._last_fused = p
        return bool(gps or fused)

    def draw(self):
        if not self.gps:
            return
        c = self.canvas
        pts = self.gps + self.fused
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        span_x = max(max(xs) - min(xs), self.MIN_SPAN_M)
        span_y = max(max(ys) - min(ys), self.MIN_SPAN_M)
        k = min((self.w - 2 * self.PAD) / span_x, (self.h - 2 * self.PAD - 30) / span_y)
        mx, my = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        cx, cy = self.w / 2, self.h / 2 + 15

        def px(p):
            return cx + (p[0] - mx) * k, cy - (p[1] - my) * k

        def place(line, track):
            if len(track) >= 2:
                c.coords(line, *[v for p in track for v in px(p)])
                c.itemconfigure(line, state="normal")

        def dot(item, p, r):
            x, y = px(p)
            c.coords(item, x - r, y - r, x + r, y + r)
            c.itemconfigure(item, state="normal")

        c.itemconfigure(self._idle, state="hidden")
        place(self._gps_line, self.gps)
        place(self._fused_line, self.fused)
        dot(self._start, self.gps[0], 4)
        dot(self._gps_now, self.gps[-1], 5)
        if self._now is not None:
            dot(self._fused_now, self._now, 5)
        # Scale bar: the largest round length that fits in a quarter of the width.
        metres = max([b for b in self.SCALE_BARS if b * k <= self.w / 4] or [self.SCALE_BARS[0]])
        y = self.h - 10
        c.coords(self._bar, 10, y, 10 + metres * k, y)
        c.coords(self._bar_text, 10, y - 3)
        c.itemconfigure(self._bar_text, text=f"{metres} m", state="normal")
        c.itemconfigure(self._bar, state="normal")
        c.tag_raise(self._fused_now)


class PipelineWindow:
    def __init__(self, top):
        self.top = top
        self.top.title("SLAM GPS en tiempo real")
        self.top.resizable(False, False)
        self.session = None
        self.worker = None
        self.outcome = None             # (folder, summary, error) of the last run
        self.frames_shown = 0
        self._shown = None
        self._photo = None
        self._closing = False

        cfg = dict(DEFAULTS)
        try:
            with open(SETTINGS) as f:
                cfg.update(json.load(f))
        except (OSError, ValueError):
            pass
        self.mode = tk.StringVar(value=cfg["mode"])
        self.ip = tk.StringVar(value=cfg["ip"])
        self.record = tk.BooleanVar(value=cfg["record"])
        self.recording = tk.StringVar(value=cfg["recording"])
        self.start_s = tk.DoubleVar(value=cfg["start"])
        self.scale = tk.BooleanVar(value=cfg["scale"])
        self.mask = tk.IntVar(value=cfg["mask"])
        self._build()
        self._show_mode()
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self._timers = {"video": self.top.after(VIDEO_MS, self._tick_video),
                        "status": self.top.after(STATUS_MS, self._tick_status),
                        "map": self.top.after(MAP_MS, self._tick_map)}

    # -------------------------------------------------------------- layout

    def _build(self):
        # Labels get their frame's background: interfaz.py paints every TLabel white.
        style = ttk.Style(self.top)
        style.configure(LABEL, background=style.lookup("TFrame", "background"))
        left = ttk.Frame(self.top, padding=10)
        left.grid(row=0, column=0, rowspan=2, sticky="ns")

        source = ttk.LabelFrame(left, text="Fuente", padding=8)
        source.grid(row=0, column=0, sticky="ew")
        ttk.Radiobutton(source, text="Teléfono en vivo", value="live", variable=self.mode,
                        command=self._show_mode).grid(row=0, column=0, sticky="w", padx=(0, 16))
        ttk.Radiobutton(source, text="Grabación (replay)", value="replay", variable=self.mode,
                        command=self._show_mode).grid(row=0, column=1, sticky="w")

        self.live_box = ttk.Frame(source)
        ttk.Label(self.live_box, style=LABEL, text="IP del teléfono").grid(row=0, column=0, sticky="w",
                                                               padx=(0, 6))
        ttk.Entry(self.live_box, textvariable=self.ip, width=18).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(self.live_box, text="Grabar también en el teléfono",
                        variable=self.record).grid(row=1, column=0, columnspan=2, sticky="w")

        self.replay_box = ttk.Frame(source)
        ttk.Label(self.replay_box, style=LABEL, text="Grabación").grid(row=0, column=0, sticky="w",
                                                          padx=(0, 6))
        ttk.Entry(self.replay_box, textvariable=self.recording, width=24).grid(row=0, column=1)
        ttk.Button(self.replay_box, text="Elegir…",
                   command=self._choose_recording).grid(row=0, column=2, padx=(4, 0))
        ttk.Label(self.replay_box, style=LABEL, text="Desde el segundo").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(self.replay_box, from_=0, to=3600, increment=10, width=8,
                    textvariable=self.start_s).grid(row=1, column=1, sticky="w")

        options = ttk.LabelFrame(left, text="Opciones", padding=8)
        options.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(options, text="Escala métrica (modelo de profundidad)",
                        variable=self.scale).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(options, style=LABEL, text="Máscara inferior (%)").grid(row=1, column=0, sticky="w",
                                                             padx=(0, 6))
        ttk.Spinbox(options, from_=0, to=60, increment=1, width=5,
                    textvariable=self.mask).grid(row=1, column=1, sticky="w")
        ttk.Label(options, style=LABEL, text="25 con el teléfono en el tablero del carro; 0 si no se ven\n"
                                "el capó ni el tablero. La línea amarilla del video la marca.",
                  foreground="#52514e").grid(row=2, column=0, columnspan=2, sticky="w")

        buttons = ttk.Frame(left)
        buttons.grid(row=2, column=0, sticky="ew", pady=8)
        self.start_button = ttk.Button(buttons, text="Iniciar", command=self.start)
        self.start_button.grid(row=0, column=0, padx=(0, 6))
        self.stop_button = ttk.Button(buttons, text="Detener", command=self.stop,
                                      state="disabled")
        self.stop_button.grid(row=0, column=1)

        status = ttk.LabelFrame(left, text="Estado", padding=8)
        status.grid(row=3, column=0, sticky="ew")
        self.lines = {}
        for row, key in enumerate(("state", "video", "gps", "imu", "camera", "fusion", "source")):
            self.lines[key] = ttk.Label(status, style=LABEL, text="", width=60)
            self.lines[key].grid(row=row, column=0, sticky="w")
        self.lines["state"].configure(text="Sin iniciar")
        self.result = ttk.Label(left, style=LABEL, text="", width=60, wraplength=440)
        self.result.grid(row=4, column=0, sticky="w", pady=(8, 0))

        self.video = tk.Canvas(self.top, width=VIDEO_W, height=VIDEO_H, bg="#0b0b0b",
                               highlightthickness=0)
        self.video.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="n")
        self.map = TrackMap(self.top, VIDEO_W, MAP_H)
        self.map.canvas.grid(row=1, column=1, padx=(0, 10), pady=(0, 10), sticky="n")
        self._image_item = self.video.create_image(0, 0, anchor="nw")
        self._idle_text = self.video.create_text(VIDEO_W // 2, VIDEO_H // 2, text="Sin video",
                                                 fill="#9a9a96", font=("TkDefaultFont", 14))
        # Where the mask starts, to check on the video that it covers the hood.
        self._mask_line = self.video.create_line(0, 0, VIDEO_W, 0, dash=(6, 4), width=2,
                                                 fill=MASK_COLOR)
        self._mask_text = self.video.create_text(VIDEO_W - 6, 0, anchor="se", text="máscara",
                                                 fill=MASK_COLOR)
        self.mask.trace_add("write", lambda *_: self._place_mask())
        self._place_mask()
        self._inputs = [w for box in (source, self.live_box, self.replay_box, options)
                        for w in box.winfo_children() if not isinstance(w, ttk.Frame)]

    def _place_mask(self):
        try:
            fraction = self.mask.get() / 100.0
        except tk.TclError:
            return
        y = VIDEO_H * (1.0 - fraction)
        state = "normal" if 0 < fraction < 1 else "hidden"
        self.video.coords(self._mask_line, 0, y, VIDEO_W, y)
        self.video.coords(self._mask_text, VIDEO_W - 6, y - 3)
        self.video.itemconfigure(self._mask_line, state=state)
        self.video.itemconfigure(self._mask_text, state=state)

    def _show_mode(self):
        live = self.mode.get() == "live"
        (self.live_box if live else self.replay_box).grid(row=1, column=0, columnspan=2,
                                                          sticky="w", pady=(6, 0))
        (self.replay_box if live else self.live_box).grid_remove()

    def _choose_recording(self):
        folder = filedialog.askdirectory(parent=self.top, title="Elegir una grabación de la app",
                                         initialdir=os.path.join(_ROOT, "mobile_data"))
        if folder:
            self.recording.set(os.path.relpath(folder, _ROOT))

    # ------------------------------------------------------------- session

    def start(self):
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            mask = self.mask.get() / 100.0
            start_s = float(self.start_s.get())
        except (tk.TclError, ValueError):
            return self._report(None, None, "La máscara y el segundo de inicio tienen que "
                                            "ser números.")
        common = {"mask_bottom": mask, "scale": self.scale.get(), "out": OUT_DIR}
        if self.mode.get() == "live":
            if not self.ip.get().strip():
                return self._report(None, None, "Falta la IP del teléfono (la muestra la app).")
            self.session = LiveSession(self.ip.get().strip(), record=self.record.get(), **common)
        else:
            folder = os.path.join(_ROOT, self.recording.get().strip())
            if not os.path.exists(os.path.join(folder, "movie.mp4")):
                return self._report(None, None, "Esa carpeta no es una grabación de la app "
                                                "(falta movie.mp4).")
            self.session = ReplaySession(folder, start_s=start_s, **common)
        self._save_settings()
        self.outcome = None
        self.result.configure(text="", foreground="")
        self.map.reset()
        self._set_running(True)
        self.worker = threading.Thread(target=self._work, args=(self.session,), daemon=True)
        self.worker.start()

    def _work(self, session):
        """The whole session, on the worker thread. Never touches tkinter."""
        try:
            session.connect()
            session.prepare()
            session.run()
            if session.metrics is None or session.metrics.processed == 0:
                raise SessionError("No se procesó ningún frame.")
            res = session.summary()
            session.save(res)
            self.outcome = (session.out_dir, res, None)
        except OSError as e:
            self.outcome = (None, None, f"No se pudo conectar ({e}). Revisar la IP, la red "
                                        "(WiFi o cable) y que la app esté en la pantalla de video.")
        except SessionError as e:
            self.outcome = (None, None, str(e))
        except Exception as e:
            traceback.print_exc()
            self.outcome = (None, None, f"Error inesperado: {e}")
        finally:
            session.close()

    def stop(self):
        if self.session is not None:
            self.session.stop()
            self.stop_button.configure(state="disabled")
            self.lines["state"].configure(text="Deteniendo…")

    def close(self):
        """Stop and save before the window goes; nothing reopens it."""
        if self.worker is not None and self.worker.is_alive():
            self._closing = True
            self.stop()
            self.lines["state"].configure(text="Guardando la sesión antes de cerrar…")
            self.top.after(200, self.close)
            return
        for timer in self._timers.values():
            self.top.after_cancel(timer)
        self.top.destroy()

    # -------------------------------------------------------------- polling

    def _tick_video(self):
        # Rescheduled first, so an error below cannot stop the video.
        self._timers["video"] = self.top.after(VIDEO_MS, self._tick_video)
        frame = self.session.latest_frame if self.session is not None else None
        if frame is not None and frame is not self._shown:
            self._shown = frame
            small = cv2.resize(frame.image, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_AREA)
            self._photo = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), master=self.top)
            self.video.itemconfigure(self._image_item, image=self._photo)
            self.video.itemconfigure(self._idle_text, state="hidden")
            self.frames_shown += 1

    def _tick_map(self):
        self._timers["map"] = self.top.after(MAP_MS, self._tick_map)
        if self.session is not None:
            new = self.session.track_since(self.map.n_fixes, self.map.n_frames)
            if self.map.add(*new):
                self.map.draw()

    def _tick_status(self):
        self._timers["status"] = self.top.after(STATUS_MS, self._tick_status)
        if self.worker is not None and self.worker.is_alive():
            if self.session.state in ("detenida", "guardando", "terminada"):
                self.lines["state"].configure(text="Guardando la sesión…")
            else:
                self._show_status(self.session.status())
        elif self.outcome is not None:
            # Controls back before the report: if it fails, they stay usable.
            outcome, self.outcome = self.outcome, None
            self._set_running(False)
            self._report(*outcome)

    def _show_status(self, st):
        state = st["state"]
        self.lines["state"].configure(text=state[:1].upper() + state[1:])
        if "elapsed_s" not in st:
            return
        video = f"Video: {st['video_fps']:.0f} fps"
        if isinstance(self.session, LiveSession):
            video += f", faltan {st['missing']}"
        video += f" | procesados {st['processed_fps']:.0f}"
        # Live, the latency from the capture is real only once the phone's
        # clock is on the PC's; until then, only the PC's share is.
        live = isinstance(self.session, LiveSession)
        if live and not st.get("clock_synced") and st["pc_latency_ms"] is not None:
            video += f" | latencia en la PC {st['pc_latency_ms']:.0f} ms"
        elif st["latency_ms"] is not None:
            video += f" | latencia {st['latency_ms']:.0f} ms"
            if live:
                video += f" (en la PC {st['pc_latency_ms']:.0f})"
        self.lines["video"].configure(text=video)
        if "fix_age_s" in st:
            gps = (f"GPS: {st['gps_fixes']} fixes; el último hace {st['fix_age_s']:.1f} s, "
                   f"{st['gps_speed']:.1f} m/s")
        else:
            gps = "GPS: sin fixes todavía"
        if st.get("udp_silent"):
            gps += " (sin respuesta UDP)"
        self.lines["gps"].configure(text=gps)
        if st["camera_speed"] is None:
            camera = "Cámara: sin escala métrica" if self.session.est is None \
                else "Cámara: calculando la velocidad…"
        else:
            camera = f"Cámara: {st['camera_speed']:.1f} m/s"
            if st["stopped"]:
                camera += " (detenido)"
        self.lines["camera"].configure(text=camera)
        # The gyroscope only turns a frame its samples cover: on another clock
        # than the frames', they would arrive and the camera would still turn.
        hz = st["imu_hz"]
        if st["gyro"]:
            imu = f"IMU: {hz:.0f} muestras/s; el giro lo da el giroscopio"
        elif hz >= 1:
            imu = f"IMU: {hz:.0f} muestras/s, pero el giro lo da la cámara"
        else:
            imu = "IMU: no llega; el giro lo da la cámara"
        self.lines["imu"].configure(text=imu)
        sigma = st["fusion_sigma_m"]
        self.lines["fusion"].configure(
            text="Fusión: espera ver el carro avanzar 20 m a más de 3 m/s" if sigma is None
            else f"Fusión: en marcha, posición ± {sigma:.1f} m")
        if isinstance(self.session, LiveSession):
            folder = st.get("recording")
            source = f"Grabando en el teléfono: {folder}" if folder else "Sin grabar en el teléfono"
        else:
            source = f"Grabación: segundo {st.get('recording_s', 0):.0f}"
        self.lines["source"].configure(text=source)

    # -------------------------------------------------------------- helpers

    def _report(self, folder, res, error):
        if error:
            self.lines["state"].configure(text="Detenida")
            self.result.configure(text=error, foreground="#c0392b")
            return
        self.lines["state"].configure(text=f"Terminada: {self.session.end_reason}")
        live = res.get("en_vivo")
        if live and live["frames_con_reloj_medido"] < live["frames_recibidos"]:
            latency = f"latencia en la PC p95 {res['pc_ms_p95']:.0f} ms"
        elif live:
            latency = f"latencia p95 {res['e2e_ms_p95']:.0f} ms (PC {res['pc_ms_p95']:.0f})"
        else:
            latency = f"latencia p95 {res['e2e_ms_p95']:.0f} ms"
        self.lines["video"].configure(
            text=f"{res['frames_procesados']} procesados "
                 f"({res['tasa_descarte_%']:.1f} % descartados), {latency}")
        fu = res["fusion"]
        pct = fu.get("frames_con_imu_pct", 0)
        self.lines["imu"].configure(text=f"IMU: el giroscopio giró el {pct:.0f} % de los frames"
                                    if pct > 0 else "IMU: no llegó; el giro lo dio la cámara")
        self.lines["fusion"].configure(text=f"Fusión: arrancó a los {fu['arranco_s']:.0f} s"
                                       if fu["arranco"] else "Fusión: no arrancó (el GPS no vio "
                                                             "el carro avanzar)")
        if isinstance(self.session, LiveSession):
            phone = self.session.folder
            self.lines["source"].configure(text=f"Grabó en el teléfono: {phone}" if phone
                                           else "Sin grabar en el teléfono")
        self.result.configure(text=f"Sesión guardada en {os.path.relpath(folder, _ROOT)}",
                              foreground="")

    def _set_running(self, running):
        for w in self._inputs:
            w.configure(state="disabled" if running else "normal")
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def _save_settings(self):
        cfg = {"mode": self.mode.get(), "ip": self.ip.get().strip(), "record": self.record.get(),
               "recording": self.recording.get().strip(), "start": float(self.start_s.get()),
               "scale": self.scale.get(), "mask": int(self.mask.get())}
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(SETTINGS, "w") as f:
                json.dump(cfg, f, indent=2)
        except OSError:
            pass


def main():
    root = tk.Tk()
    PipelineWindow(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
