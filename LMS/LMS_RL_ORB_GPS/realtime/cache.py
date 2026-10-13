"""
Cache of deterministic replays, so a long recording is processed once and its
results travel with the repository between computers.

    resultados/cache/<recording>/<key>/
        frames_deterministic[_metric].csv
        replay_deterministic[_metric].json
        trajectory_deterministic[_metric].png
        cache.json          what the key was made of, and where it was processed

The key hashes three things, so any change misses the cache and reprocesses:
the recording's content (video, timestamps, GPS, IMU), the options that change
the results (intrinsics, mask, scale, start, frames, modes) and the code that
produces them (CODE_FILES, plus CODE_VERSION for the rest of run_replay's
processing loop). Strict replays and the window are not cached: they run in
real time and drop frames, so two runs differ.

Usage: run_replay.py uses it unless --no-cache; this module also lists the
cache:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.cache
"""

import hashlib
import json
import os
import platform
import shutil
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
CACHE_DIR = os.path.join(_ROOT, "resultados", "cache")

# The recording files the results depend on; the others (movie_metadata.csv,
# edge_epochs.txt) are not read by the replay.
RECORDING_FILES = ("frame_timestamps.txt", "location.csv", "gyro_accel.csv")
# The code that decides the numbers: frame processing, scale, fusion and the
# recording readers. A plot-only change in run_replay.py keeps the cache.
CODE_FILES = ("pipeline.py", "depth_scale.py", "fusion.py", "sources.py")
CODE_VERSION = 1            # bump when run_replay.py's processing loop changes
_CHUNK = 1 << 20


def _file_hash(path, h):
    with open(path, "rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            h.update(block)


def recording_hash(session_dir, video="movie.mp4"):
    """sha256 of the files of a recording the replay reads (~1 s for 430 MB)."""
    h = hashlib.sha256()
    for name in (video,) + RECORDING_FILES:
        path = os.path.join(session_dir, name)
        h.update(name.encode())
        if os.path.exists(path):
            _file_hash(path, h)
        else:
            h.update(b"<missing>")
    return h.hexdigest()


def code_hash():
    h = hashlib.sha256(f"v{CODE_VERSION}".encode())
    for name in CODE_FILES:
        h.update(name.encode())
        _file_hash(os.path.join(_THIS, name), h)
    return h.hexdigest()


def options(args):
    """The run_replay options that change the results, as a plain dict."""
    return {"video": args.video, "frames": args.frames, "mask_bottom": args.mask_bottom,
            "fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy,
            "skip_start": args.skip_start, "start": args.start, "scale": args.scale,
            "scale_hz": args.scale_hz, "visual_only": args.visual_only, "no_gyro": args.no_gyro}


class Entry:
    """One cached replay: where it lives and whether it is complete."""

    def __init__(self, session_dir, args):
        self.session = os.path.abspath(session_dir)
        self.name = os.path.basename(os.path.normpath(session_dir))
        self.recording = recording_hash(session_dir, args.video)
        self.code = code_hash()
        self.options = options(args)
        key = hashlib.sha256(json.dumps(
            {"recording": self.recording, "code": self.code, "options": self.options},
            sort_keys=True).encode()).hexdigest()[:16]
        self.dir = os.path.join(CACHE_DIR, self.name, key)
        self.suffix = "_metric" if args.scale != "none" else ""

    def path(self, kind):
        ext = {"frames": "csv", "replay": "json", "trajectory": "png"}[kind]
        return os.path.join(self.dir, f"{kind}_deterministic{self.suffix}.{ext}")

    @property
    def complete(self):
        return all(os.path.exists(self.path(k)) for k in ("frames", "replay", "trajectory")) \
            and os.path.exists(os.path.join(self.dir, "cache.json"))

    def restore(self, out_dir):
        """Copy the cached files to out_dir; returns the replay's metrics."""
        os.makedirs(out_dir, exist_ok=True)
        for kind in ("frames", "replay", "trajectory"):
            shutil.copy2(self.path(kind), out_dir)
        with open(self.path("replay")) as f:
            return json.load(f)

    def store(self, files):
        """Keep the given {kind: path} outputs of a finished replay."""
        os.makedirs(self.dir, exist_ok=True)
        for kind, src in files.items():
            if os.path.abspath(src) != os.path.abspath(self.path(kind)):
                shutil.copy2(src, self.path(kind))
        import cv2
        import numpy as np
        try:
            import torch
            torch_v = f"{torch.__version__} (CUDA {torch.version.cuda})"
        except ImportError:
            torch_v = None
        meta = {"grabacion": self.name, "hash_grabacion": self.recording,
                "hash_codigo": self.code, "version_codigo": CODE_VERSION,
                "opciones": self.options, "procesado": time.strftime("%Y-%m-%d %H:%M:%S"),
                "equipo": platform.node(), "opencv": cv2.__version__, "numpy": np.__version__,
                "torch": torch_v}
        with open(os.path.join(self.dir, "cache.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    if not os.path.isdir(CACHE_DIR):
        print(f"Caché vacía ({os.path.relpath(CACHE_DIR, _ROOT)} no existe).")
        return 0
    current = code_hash()
    for rec in sorted(os.listdir(CACHE_DIR)):
        for key in sorted(os.listdir(os.path.join(CACHE_DIR, rec))):
            meta_path = os.path.join(CACHE_DIR, rec, key, "cache.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                m = json.load(f)
            o = m["opciones"]
            stale = "" if m["hash_codigo"] == current else "  (código anterior: no se usa)"
            print(f"{rec}/{key}: escala {o['scale']}, máscara {o['mask_bottom']}, "
                  f"frames {o['frames'] or 'todos'}, desde {o['start']} s; "
                  f"{m['procesado']} en {m['equipo']}{stale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
