"""
Window of the real-time pipeline: the video, live from the phone or from a
recording, the session's status and a map of the GPS and the fused track.
The session runs on a worker thread; tkinter is only touched from the main
thread, which polls it.

Two tabs, built from the same panel so they cannot drift apart: "Real time"
for normal use, and "Development", which adds a simulated GPS outage and
draws, on the map, the hidden fixes, the track while the GPS was cut and the
fixed EKF beside the RL. One session at a time: the phone sends one video
stream and the depth model takes the GPU.

The texts are in English; the session's own states and errors arrive in
Spanish, as the console prints them, and are translated here (english()).

    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.live_window
"""

import json
import os
import re
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
FIXED_COLOR = "#8c8c88"         # the fixed EKF beside the RL: a comparison, recessive
OUTAGE_COLOR = "#e2b3cc"        # the fused track while the GPS was cut: a lighter fusion
MASK_COLOR = "#f0b429"
LABEL = "Pipeline.TLabel"
STATUS_MS = 1000
OUT_DIR = os.path.join(_ROOT, "resultados", "realtime")
SETTINGS = os.path.join(OUT_DIR, "interfaz.json")      # last values typed, per user
DEFAULTS = {"mode": "live", "ip": "", "record": False, "recording": "",
            "start": 0.0, "scale": True, "mask": 25, "rl": False, "cut_s": 30}
# Settings key, title, development tab.
TABS = (("tiempo_real", "Real time", False), ("desarrollo", "Development", True))

# The session's states, end reasons and errors, as the window shows them.
SESSION_TEXT = {
    "sin iniciar": "not started",
    "conectando con el teléfono": "connecting to the phone",
    "leyendo la grabación": "reading the recording",
    "cargando el modelo de profundidad": "loading the depth model",
    "precalentando": "warming up",
    "en marcha": "running",
    "detenida": "stopped",
    "guardando": "saving",
    "terminada": "finished",
    "sesión detenida": "session stopped",
    "fin de la grabación": "end of the recording",
    "la cámara cambió entre conexiones": "the camera changed between connections",
    "cabecera de frame inválida": "invalid frame header",
    "el teléfono cerró la conexión de video": "the phone closed the video connection",
    "no llegan frames hace 5 s": "no frames for 5 s",
    "se cortó la conexión de video": "the video connection dropped",
    "Sesión cancelada antes de empezar.": "Session cancelled before it started.",
    "No se procesó ningún frame.": "No frame was processed.",
    "No se pudo iniciar la grabación en el teléfono. ¿Está la app en primer plano, en la "
    "pantalla de video?": "Could not start recording on the phone. Is the app in the "
                          "foreground, on the video screen?",
}
SESSION_PATTERNS = ((r"buscando el segundo (\d+) de la grabación", r"seeking second \1 of the recording"),
                    (r"no se pudo conectar al video \((.*)\)", r"could not connect to the video (\1)"),
                    (r"No se pudo cargar la política del RL \((.*)\)\.", r"Could not load the RL policy (\1)."))


def english(text):
    """A session text in English; one not in the table is shown as it is."""
    if text in SESSION_TEXT:
        return SESSION_TEXT[text]
    for pattern, repl in SESSION_PATTERNS:
        if re.fullmatch(pattern, text or ""):
            return re.sub(pattern, repl, text)
    return text


class TrackMap:
    """
    The GPS and the fusion on a canvas, in metres (east right, north up, the
    same scale on both axes), fitted to what has been driven so far.

    With developer, the GPS line goes through every fix, also those a
    simulated outage hid from the fusion, which get a hollow circle; the
    fused track while the GPS was cut is drawn lighter, so the correction at
    its return stands out; and with the RL the fixed EKF goes under the
    fusion, dashed.
    """

    PAD = 26
    MIN_SPAN_M = 60.0       # before the car moves, a fix's noise would fill the map
    MIN_STEP_M = 0.5        # fused points closer than this to the last kept one are skipped
    SCALE_BARS = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)

    def __init__(self, parent, width, height, developer=False):
        self.w, self.h = width, height
        self.developer = developer
        self.canvas = tk.Canvas(parent, width=width, height=height, bg="white",
                                highlightthickness=1, highlightbackground="#d0d0cc")
        self.reset()

    def reset(self, rl=False):
        c = self.canvas
        c.delete("all")
        self.gps, self.hidden, self.fixed = [], [], []
        # The fused track in pieces, each (inside an outage, points, canvas line).
        self.pieces = []
        self.n_fixes = self.n_frames = 0
        self._last = {"fused": None, "fixed": None}     # last point kept for each line
        self._now = None            # newest fused position, kept or not
        self._rings = []            # one circle per hidden fix
        self._idle = c.create_text(self.w / 2, self.h / 2, fill="#9a9a96",
                                   text="Map: the GPS and the fused track, in metres")
        self._gps_line = c.create_line(0, 0, 0, 0, fill=COLOR_GPS, width=3, state="hidden")
        self._fixed_line = c.create_line(0, 0, 0, 0, fill=FIXED_COLOR, width=2, dash=(5, 3),
                                         state="hidden")
        self._start = c.create_oval(0, 0, 0, 0, fill="#0b0b0b", outline="", state="hidden")
        self._gps_now = c.create_oval(0, 0, 0, 0, fill=COLOR_GPS, outline="white", state="hidden")
        self._fused_now = c.create_oval(0, 0, 0, 0, fill=COLOR_FUSED, outline="white",
                                        state="hidden")
        self._bar = c.create_line(0, 0, 0, 0, width=2, fill="#0b0b0b", state="hidden")
        self._bar_text = c.create_text(0, 0, anchor="sw", font=("TkDefaultFont", 8),
                                       state="hidden")
        c.create_text(self.w - 10, 8, anchor="ne", text="N ↑", font=("TkDefaultFont", 9, "bold"))
        # Two columns at most, so the legend keeps to two rows.
        rows = [((8, 12), "line", COLOR_GPS, "GPS (reference)"),
                ((8, 28), "line", COLOR_FUSED,
                 "Fused with RL (official track)" if rl else "Fused (official track)")]
        if self.developer:
            rows += [((215, 12), "ring", COLOR_GPS, "Hidden GPS"),
                     ((215, 28), "line", OUTAGE_COLOR, "Fused while cut")]
            if rl:
                rows.append(((380, 12), "dash", FIXED_COLOR, "Fixed EKF, no RL"))
        for (x, y), kind, color, label in rows:
            if kind == "ring":
                c.create_oval(x + 6, y - 4, x + 14, y + 4, outline=color, width=2)
            else:
                c.create_line(x, y, x + 20, y, fill=color, width=3 if kind == "line" else 2,
                              dash=(5, 3) if kind == "dash" else ())
            c.create_text(x + 26, y, anchor="w", text=label, font=("TkDefaultFont", 8))

    def add(self, gps, fused, fixed=(), cut=()):
        """New fixes, fused rows, fixed-EKF rows and outage marks from PipelineSession.track_since."""
        self.n_fixes += len(gps)
        self.n_frames += len(fused)
        for x, y, hidden in gps:
            self.gps.append((x, y))
            if hidden:
                self.hidden.append((x, y))
        cut = list(cut) + [False] * (len(fused) - len(cut))
        for p, inside in zip(fused, cut):
            if p is not None:
                self._now = p
                self._add_fused(p, self.developer and inside)
        if self.developer:
            for p in fixed:
                if p is not None and self._far("fixed", p):
                    self.fixed.append(p)
        return bool(gps or fused)

    def _far(self, name, p):
        """Whether p is far enough from the last point kept for that line; keeps it if so."""
        last = self._last[name]
        if last is None or (p[0] - last[0]) ** 2 + (p[1] - last[1]) ** 2 >= self.MIN_STEP_M ** 2:
            self._last[name] = p
            return True
        return False

    def _add_fused(self, p, inside):
        if self.pieces and self.pieces[-1][0] == inside:
            if self._far("fused", p):
                self.pieces[-1][1].append(p)
            return
        # A new piece starts where the last one ended, so the track stays joined.
        start = [self.pieces[-1][1][-1]] if self.pieces else []
        line = self.canvas.create_line(0, 0, 0, 0, width=2, state="hidden",
                                       fill=OUTAGE_COLOR if inside else COLOR_FUSED)
        self.pieces.append((inside, start + [p], line))
        self._last["fused"] = p

    def draw(self):
        if not self.gps:
            return
        c = self.canvas
        pts = self.gps + self.fixed + [p for _, piece, _ in self.pieces for p in piece]
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
        place(self._fixed_line, self.fixed)
        for _, piece, line in self.pieces:
            place(line, piece)
        while len(self._rings) < len(self.hidden):
            self._rings.append(c.create_oval(0, 0, 0, 0, outline=COLOR_GPS, width=2))
        for ring, p in zip(self._rings, self.hidden):
            dot(ring, p, 4)
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
        # The corrected track over the one drawn while the GPS was cut.
        order = ([self._fixed_line] + [ln for inside, _, ln in self.pieces if inside]
                 + [ln for inside, _, ln in self.pieces if not inside]
                 + [self._gps_now, self._fused_now])
        for item in order:
            c.tag_raise(item)


class PipelinePanel:
    """
    One tab: its source, options, buttons, status, video and map, and the
    session it runs. developer adds the simulated outage and its drawing.
    """

    def __init__(self, window, parent, key, developer=False):
        self.window = window            # PipelineWindow: the other tab, and closing
        self.top = window.top
        self.frame = ttk.Frame(parent)
        self.key = key                  # its settings in SETTINGS
        self.developer = developer
        self.session = None
        self.worker = None
        self.outcome = None             # (folder, summary, error) of the last run
        self.frames_shown = 0
        self._shown = None
        self._photo = None

        cfg = dict(DEFAULTS)
        saved = _read_settings()
        cfg.update({k: v for k, v in saved.items() if k in DEFAULTS})   # a file from before the tabs
        cfg.update(saved.get(key, {}))
        self.mode = tk.StringVar(value=cfg["mode"])
        self.ip = tk.StringVar(value=cfg["ip"])
        self.record = tk.BooleanVar(value=cfg["record"])
        self.recording = tk.StringVar(value=cfg["recording"])
        self.start_s = tk.DoubleVar(value=cfg["start"])
        self.scale = tk.BooleanVar(value=cfg["scale"])
        self.mask = tk.IntVar(value=cfg["mask"])
        self.rl = tk.BooleanVar(value=cfg["rl"])
        self.cut_s = tk.IntVar(value=cfg["cut_s"])
        self._build()
        self._show_mode()
        self._timers = {"video": self.top.after(VIDEO_MS, self._tick_video),
                        "status": self.top.after(STATUS_MS, self._tick_status),
                        "map": self.top.after(MAP_MS, self._tick_map)}

    @property
    def running(self):
        return self.worker is not None and self.worker.is_alive()

    # -------------------------------------------------------------- layout

    def _build(self):
        left = ttk.Frame(self.frame, padding=10)
        left.grid(row=0, column=0, rowspan=2, sticky="ns")

        source = ttk.LabelFrame(left, text="Source", padding=8)
        source.grid(row=0, column=0, sticky="ew")
        ttk.Radiobutton(source, text="Live phone", value="live", variable=self.mode,
                        command=self._show_mode).grid(row=0, column=0, sticky="w", padx=(0, 16))
        ttk.Radiobutton(source, text="Recording (replay)", value="replay", variable=self.mode,
                        command=self._show_mode).grid(row=0, column=1, sticky="w")

        self.live_box = ttk.Frame(source)
        ttk.Label(self.live_box, style=LABEL, text="Phone IP").grid(row=0, column=0, sticky="w",
                                                               padx=(0, 6))
        ttk.Entry(self.live_box, textvariable=self.ip, width=18).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(self.live_box, text="Also record on the phone",
                        variable=self.record).grid(row=1, column=0, columnspan=2, sticky="w")

        self.replay_box = ttk.Frame(source)
        ttk.Label(self.replay_box, style=LABEL, text="Recording").grid(row=0, column=0, sticky="w",
                                                          padx=(0, 6))
        ttk.Entry(self.replay_box, textvariable=self.recording, width=24).grid(row=0, column=1)
        ttk.Button(self.replay_box, text="Choose…",
                   command=self._choose_recording).grid(row=0, column=2, padx=(4, 0))
        ttk.Label(self.replay_box, style=LABEL, text="From second").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(self.replay_box, from_=0, to=3600, increment=10, width=8,
                    textvariable=self.start_s).grid(row=1, column=1, sticky="w")

        options = ttk.LabelFrame(left, text="Options", padding=8)
        options.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(options, text="Metric scale (depth model)",
                        variable=self.scale).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(options, style=LABEL, text="Bottom mask (%)").grid(row=1, column=0, sticky="w",
                                                             padx=(0, 6))
        ttk.Spinbox(options, from_=0, to=60, increment=1, width=5,
                    textvariable=self.mask).grid(row=1, column=1, sticky="w")
        ttk.Label(options, style=LABEL, text="25 with the phone on the car's dashboard; 0 if neither the\n"
                                "hood nor the dashboard shows. The yellow line marks it.",
                  foreground="#52514e").grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(options, text="RL: decides how much to trust the camera when the GPS is out",
                        variable=self.rl).grid(row=3, column=0, columnspan=2, sticky="w",
                                               pady=(4, 0))

        buttons = ttk.Frame(left)
        buttons.grid(row=2, column=0, sticky="ew", pady=8)
        self.start_button = ttk.Button(buttons, text="Start", command=self.start)
        self.start_button.grid(row=0, column=0, padx=(0, 6))
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop,
                                      state="disabled")
        self.stop_button.grid(row=0, column=1)
        self.cut_button = None
        if self.developer:
            # A simulated outage, once the fusion has started.
            self.cut_button = ttk.Button(buttons, text="Cut the GPS", command=self.cut,
                                         state="disabled")
            self.cut_button.grid(row=0, column=2, padx=(18, 4))
            ttk.Spinbox(buttons, from_=5, to=120, increment=5, width=4,
                        textvariable=self.cut_s).grid(row=0, column=3)
            ttk.Label(buttons, style=LABEL, text="s").grid(row=0, column=4, padx=(2, 0))

        status = ttk.LabelFrame(left, text="Status", padding=8)
        status.grid(row=3, column=0, sticky="ew")
        self.lines = {}
        keys = ["state", "video", "gps", "imu", "camera", "fusion", "rl", "source"]
        if self.developer:
            keys.insert(7, "cut")
        for row, key in enumerate(keys):
            self.lines[key] = ttk.Label(status, style=LABEL, text="", width=60)
            self.lines[key].grid(row=row, column=0, sticky="w")
        self.lines["state"].configure(text="Not started")
        self.result = ttk.Label(left, style=LABEL, text="", width=60, wraplength=440)
        self.result.grid(row=4, column=0, sticky="w", pady=(8, 0))

        self.video = tk.Canvas(self.frame, width=VIDEO_W, height=VIDEO_H, bg="#0b0b0b",
                               highlightthickness=0)
        self.video.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="n")
        self.map = TrackMap(self.frame, VIDEO_W, MAP_H, developer=self.developer)
        self.map.canvas.grid(row=1, column=1, padx=(0, 10), pady=(0, 10), sticky="n")
        self._image_item = self.video.create_image(0, 0, anchor="nw")
        self._idle_text = self.video.create_text(VIDEO_W // 2, VIDEO_H // 2, text="No video",
                                                 fill="#9a9a96", font=("TkDefaultFont", 14))
        # Where the mask starts, to check on the video that it covers the hood.
        self._mask_line = self.video.create_line(0, 0, VIDEO_W, 0, dash=(6, 4), width=2,
                                                 fill=MASK_COLOR)
        self._mask_text = self.video.create_text(VIDEO_W - 6, 0, anchor="se", text="mask",
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
        folder = filedialog.askdirectory(parent=self.top, title="Choose a recording of the app",
                                         initialdir=os.path.join(_ROOT, "mobile_data"))
        if folder:
            self.recording.set(os.path.relpath(folder, _ROOT))

    # ------------------------------------------------------------- session

    def start(self):
        if self.running:
            return
        if self.window.busy(self):
            return self._report(None, None, "The other tab has a session running: "
                                            "stop it first (one session at a time).")
        try:
            mask = self.mask.get() / 100.0
            start_s = float(self.start_s.get())
        except (tk.TclError, ValueError):
            return self._report(None, None, "The mask and the start second have to be "
                                            "numbers.")
        common = {"mask_bottom": mask, "scale": self.scale.get(), "out": OUT_DIR,
                  "rl": self.rl.get()}
        if self.mode.get() == "live":
            if not self.ip.get().strip():
                return self._report(None, None, "The phone's IP is missing (the app shows it).")
            self.session = LiveSession(self.ip.get().strip(), record=self.record.get(), **common)
        else:
            folder = os.path.join(_ROOT, self.recording.get().strip())
            if not os.path.exists(os.path.join(folder, "movie.mp4")):
                return self._report(None, None, "That folder is not a recording of the app "
                                                "(no movie.mp4).")
            self.session = ReplaySession(folder, start_s=start_s, **common)
        self._save_settings()
        self.outcome = None
        self.result.configure(text="", foreground="")
        self.map.reset(rl=self.rl.get())
        for key in ("rl", "cut"):
            if key in self.lines:
                self.lines[key].configure(text="")
        self._set_running(True)
        self.worker = threading.Thread(target=self._work, args=(self.session,), daemon=True)
        self.worker.start()
        self.window.refresh()

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
            self.outcome = (None, None, f"Could not connect ({e}). Check the IP, the network "
                                        "(WiFi or cable) and that the app is on the video screen.")
        except SessionError as e:
            self.outcome = (None, None, english(str(e)))
        except Exception as e:
            traceback.print_exc()
            self.outcome = (None, None, f"Unexpected error: {e}")
        finally:
            session.close()

    def stop(self):
        if self.session is not None:
            self.session.stop()
            self.stop_button.configure(state="disabled")
            if self.cut_button is not None:
                self.cut_button.configure(state="disabled")
            self.lines["state"].configure(text="Stopping…")

    def cut(self):
        try:
            seconds = int(self.cut_s.get())
        except (tk.TclError, ValueError):
            seconds = 0
        if not 1 <= seconds <= 600:
            self.lines["cut"].configure(text="Simulated outage: the length is in seconds, 1 to 600")
            return
        if self.session is not None:
            self.session.cut_gps(seconds)
            # Back on once the GPS has returned (_show_status).
            self.cut_button.configure(state="disabled")
            self.lines["cut"].configure(text=f"Simulated outage: {seconds} s without GPS from now")

    def cancel_timers(self):
        for timer in self._timers.values():
            self.top.after_cancel(timer)

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
        if self.running:
            if self.session.state in ("detenida", "guardando", "terminada"):
                self.lines["state"].configure(text="Saving the session…")
            else:
                self._show_status(self.session.status())
        elif self.outcome is not None:
            # Controls back before the report: if it fails, they stay usable.
            outcome, self.outcome = self.outcome, None
            self._set_running(False)
            self.window.refresh()
            self._report(*outcome)

    def _show_status(self, st):
        state = st["state"]
        state = english(state)
        self.lines["state"].configure(text=state[:1].upper() + state[1:])
        if "elapsed_s" not in st:
            return
        video = f"Video: {st['video_fps']:.0f} fps"
        if isinstance(self.session, LiveSession):
            video += f", {st['missing']} missing"
        video += f" | processed {st['processed_fps']:.0f}"
        # Live, the latency from the capture is real only once the phone's
        # clock is on the PC's; until then, only the PC's share is.
        live = isinstance(self.session, LiveSession)
        if live and not st.get("clock_synced") and st["pc_latency_ms"] is not None:
            video += f" | latency on the PC {st['pc_latency_ms']:.0f} ms"
        elif st["latency_ms"] is not None:
            video += f" | latency {st['latency_ms']:.0f} ms"
            if live:
                video += f" (on the PC {st['pc_latency_ms']:.0f})"
        self.lines["video"].configure(text=video)
        if "fix_age_s" in st:
            gps = (f"GPS: {st['gps_fixes']} fixes; the last {st['fix_age_s']:.1f} s ago, "
                   f"{st['gps_speed']:.1f} m/s")
        else:
            gps = "GPS: no fixes yet"
        if st.get("udp_silent"):
            gps += " (no UDP reply)"
        self.lines["gps"].configure(text=gps)
        if st["camera_speed"] is None:
            camera = "Camera: no metric scale" if self.session.est is None \
                else "Camera: working out the speed…"
        else:
            camera = f"Camera: {st['camera_speed']:.1f} m/s"
            if st["stopped"]:
                camera += " (stopped)"
        self.lines["camera"].configure(text=camera)
        # The gyroscope only turns a frame its samples cover: on another clock
        # than the frames', they would arrive and the camera would still turn.
        hz = st["imu_hz"]
        if st["gyro"]:
            imu = f"IMU: {hz:.0f} samples/s; the gyroscope turns the heading"
        elif hz >= 1:
            imu = f"IMU: {hz:.0f} samples/s, but the camera turns the heading"
        else:
            imu = "IMU: not arriving; the camera turns the heading"
        self.lines["imu"].configure(text=imu)
        sigma = st["fusion_sigma_m"]
        self.lines["fusion"].configure(
            text="Fusion: waiting for the car to move 20 m above 3 m/s" if sigma is None
            else f"Fusion: running, position ± {sigma:.1f} m")
        k = st["rl_k"]
        if not self.session.use_rl:
            rl = "RL: off"
        elif k is None:
            rl = "RL: with GPS it does not decide (the fixed EKF)"
        elif k == float("inf"):
            rl = "RL: no GPS, ignores the camera"
        else:
            rl = f"RL: no GPS, trusts the camera less (k = {k:g})" if k > 1 \
                else "RL: no GPS, trusts the camera (k = 1)"
        self.lines["rl"].configure(text=rl)
        if self.developer:
            left = st["outage_left_s"]
            if left is not None:
                self.lines["cut"].configure(text=f"Simulated outage: the GPS comes back in {left:.0f} s")
            elif st["outage_back"] is not None:
                self.lines["cut"].configure(text=self._back_text(st["outage_back"]))
            # One outage at a time, and only once there is a filter to cut.
            ready = sigma is not None and left is None and self.session.state == "en marcha"
            if self.cut_button.instate(["disabled"]) == ready:
                self.cut_button.configure(state="normal" if ready else "disabled")
        if isinstance(self.session, LiveSession):
            folder = st.get("recording")
            source = f"Recording on the phone: {folder}" if folder else "Not recording on the phone"
        else:
            source = f"Recording: second {st.get('recording_s', 0):.0f}"
        self.lines["source"].configure(text=source)

    # -------------------------------------------------------------- helpers

    def _report(self, folder, res, error):
        if error:
            self.lines["state"].configure(text="Stopped" if self.session is not None
                                          else "Not started")
            self.result.configure(text=error, foreground="#c0392b")
            return
        self.lines["state"].configure(text=f"Finished: {english(self.session.end_reason)}")
        live = res.get("en_vivo")
        if live and live["frames_con_reloj_medido"] < live["frames_recibidos"]:
            latency = f"latency on the PC p95 {res['pc_ms_p95']:.0f} ms"
        elif live:
            latency = f"latency p95 {res['e2e_ms_p95']:.0f} ms (PC {res['pc_ms_p95']:.0f})"
        else:
            latency = f"latency p95 {res['e2e_ms_p95']:.0f} ms"
        self.lines["video"].configure(
            text=f"{res['frames_procesados']} processed "
                 f"({res['tasa_descarte_%']:.1f} % dropped), {latency}")
        fu = res["fusion"]
        pct = fu.get("frames_con_imu_pct", 0)
        self.lines["imu"].configure(text=f"IMU: the gyroscope turned {pct:.0f} % of the frames"
                                    if pct > 0 else "IMU: did not arrive; the camera turned the heading")
        self.lines["fusion"].configure(text=f"Fusion: started after {fu['arranco_s']:.0f} s"
                                       if fu["arranco"] else "Fusion: did not start (the GPS never "
                                                             "saw the car move)")
        rl = fu.get("rl")
        if rl:
            self.lines["rl"].configure(text=f"RL: decided {rl['decisiones']} times without GPS")
        cuts = [c for c in fu.get("cortes_simulados", []) if c["volvio_el_gps"]]
        if cuts and self.developer:
            e = cuts[-1]["error_al_volver_m"]
            self.lines["cut"].configure(text=self._back_text((e["ekf_fijo"], e["rl"]))
                                        + (f" ({len(cuts)} outages; all in metrics.json)"
                                           if len(cuts) > 1 else ""))
        if isinstance(self.session, LiveSession):
            phone = self.session.folder
            self.lines["source"].configure(text=f"Recorded on the phone: {phone}" if phone
                                           else "Not recorded on the phone")
        self.result.configure(text=f"Session saved in {os.path.relpath(folder, _ROOT)}",
                              foreground="")

    @staticmethod
    def _back_text(errors):
        """The errors at the fix the last outage ended with, (fixed EKF, RL)."""
        fixed, rl = errors
        if fixed is None:
            return "Simulated outage: it ended before the fusion started"
        return (f"When the GPS came back: fixed EKF {fixed:.1f} m off"
                + ("" if rl is None else f", RL {rl:.1f} m"))

    def _set_running(self, running):
        for w in self._inputs:
            w.configure(state="disabled" if running else "normal")
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        if not running and self.cut_button is not None:
            self.cut_button.configure(state="disabled")

    def allow_start(self, allowed):
        """Start enabled or not, from the window: the other tab may be running."""
        if not self.running:
            self.start_button.configure(state="normal" if allowed else "disabled")

    def _save_settings(self):
        try:
            cut_s = int(self.cut_s.get())
        except (tk.TclError, ValueError):
            cut_s = DEFAULTS["cut_s"]
        cfg = {"mode": self.mode.get(), "ip": self.ip.get().strip(), "record": self.record.get(),
               "recording": self.recording.get().strip(), "start": float(self.start_s.get()),
               "scale": self.scale.get(), "mask": int(self.mask.get()), "rl": self.rl.get(),
               "cut_s": cut_s}
        saved = _read_settings()
        saved[self.key] = cfg
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(SETTINGS, "w") as f:
                json.dump(saved, f, indent=2)
        except OSError:
            pass


def _read_settings() -> dict:
    try:
        with open(SETTINGS) as f:
            saved = json.load(f)
        return saved if isinstance(saved, dict) else {}
    except (OSError, ValueError):
        return {}


class PipelineWindow:
    """The window: one tab for normal use and one for development, one session at a time."""

    def __init__(self, top):
        self.top = top
        self.top.title("Real-time SLAM GPS")
        self.top.resizable(False, False)
        # Labels get their frame's background: interfaz.py paints every TLabel white.
        style = ttk.Style(self.top)
        style.configure(LABEL, background=style.lookup("TFrame", "background"))
        self.tabs = ttk.Notebook(self.top)
        self.tabs.grid(row=0, column=0)
        self.panels = []
        for key, title, developer in TABS:
            panel = PipelinePanel(self, self.tabs, key, developer)
            self.tabs.add(panel.frame, text=title)
            self.panels.append(panel)
        self.top.protocol("WM_DELETE_WINDOW", self.close)

    def busy(self, panel):
        """Whether a tab other than panel has a session running."""
        return any(p.running for p in self.panels if p is not panel)

    def refresh(self):
        for p in self.panels:
            p.allow_start(not self.busy(p))

    def close(self):
        """Stop and save before the window goes; nothing reopens it."""
        running = [p for p in self.panels if p.running]
        if running:
            for p in running:
                p.stop()
                p.lines["state"].configure(text="Saving the session before closing…")
            self.top.after(200, self.close)
            return
        for p in self.panels:
            p.cancel_timers()
        self.top.destroy()


def main():
    root = tk.Tk()
    PipelineWindow(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
