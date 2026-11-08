"""
Resource usage of a run: CPU, RAM and GPU of this process, sampled in a
background thread while the pipeline works.

    mon = ResourceMonitor().start()
    ...                                  # pipeline.run()
    res["recursos"] = mon.stop()         # p50 / p95 / max of each series

It lives outside the files cache.py hashes: it only observes the run, so a
change here must not invalidate the cached replays.

CPU is the sum over the process threads, as a percentage of one hardware
thread (400 % = four busy threads) and of the whole machine. GPU memory is
this process's, from nvidia-smi, CUDA context included; GPU utilisation is
the whole card's, so other programs on the same card count too.
"""

import os
import shutil
import subprocess
import threading

import numpy as np

try:
    import psutil
except ImportError:              # requirements.txt has it; without it, no record
    psutil = None


def _stats(values):
    if not values:
        return None
    v = np.asarray(values, dtype=float)
    return {"p50": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


class ResourceMonitor:
    """Samples every interval_s seconds between start() and stop()."""

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None
        self._cpu, self._ram, self._gpu_mem, self._gpu_util = [], [], [], []
        self._gpu_name = None
        self._gpu_error = None           # why nvidia-smi gave nothing, if it failed
        self._smi = shutil.which("nvidia-smi")
        self._proc = psutil.Process(os.getpid()) if psutil is not None else None

    def start(self):
        if self._proc is None:
            print("  AVISO: psutil no está instalado; la sesión no registra recursos.")
            return self
        self._proc.cpu_percent(None)        # the first reading only sets the origin
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        """Stop sampling and return the summary; None without psutil or samples."""
        if self._thread is None:
            return None
        self._stop.set()
        self._thread.join(timeout=5.0)
        if not self._cpu:
            return None
        n = psutil.cpu_count() or 1
        return {
            "muestras": len(self._cpu),
            "intervalo_s": self.interval_s,
            "hilos_del_equipo": n,
            "cpu_proceso_%": _stats(self._cpu),
            "cpu_equipo_%": _stats([c / n for c in self._cpu]),
            "ram_mib": _stats(self._ram),
            "gpu": self._gpu_name,
            "gpu_memoria_mib": _stats(self._gpu_mem),
            "gpu_uso_tarjeta_%": _stats(self._gpu_util),
            "gpu_aviso": self._gpu_error,
        }

    def _run(self):
        while not self._stop.wait(self.interval_s):
            try:
                self._cpu.append(self._proc.cpu_percent(None))
                self._ram.append(self._proc.memory_info().rss / 2**20)
            except psutil.Error:
                return
            self._sample_gpu()

    def _sample_gpu(self):
        """This process's GPU memory and the card's use; skipped without nvidia-smi."""
        if self._smi is None:
            return
        card = self._query("--query-gpu=name,utilization.gpu")
        if card is None:
            return
        fields = card[0].rsplit(",", 1) if card else []
        if len(fields) == 2:
            self._gpu_name = fields[0].strip()
            try:
                self._gpu_util.append(float(fields[1]))
            except ValueError:            # "[N/A]" on some laptops
                pass
        apps = self._query("--query-compute-apps=pid,used_memory") or []
        pid = str(os.getpid())
        for line in apps:
            fields = [f.strip() for f in line.split(",")]
            if len(fields) == 2 and fields[0] == pid:
                try:
                    self._gpu_mem.append(float(fields[1]))
                except ValueError:
                    pass

    def _query(self, what):
        """Lines of one nvidia-smi query; None, and no more queries, if it fails."""
        try:
            r = subprocess.run([self._smi, what, "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=2.0)
            if r.returncode == 0:
                return r.stdout.strip().splitlines()
            # Without a loaded driver nvidia-smi exits non-zero with one line of text.
            text = (r.stdout + r.stderr).strip()
            self._gpu_error = (text.splitlines()[0] if text
                               else f"nvidia-smi terminó con código {r.returncode}")
        except (OSError, subprocess.TimeoutExpired) as e:
            self._gpu_error = str(e)
        self._smi = None
        return None


def print_resources(r):
    """The resources part of the console report."""
    if not r:
        return

    def line(label, s, unit):
        if s is None:
            print(f"     {label:<32}: —")
        else:
            print(f"     {label:<32}: {s['p50']:.0f} / {s['p95']:.0f} / {s['max']:.0f} {unit}")

    print(f"\n  Recursos (p50 / p95 / máx, {r['muestras']} muestras cada {r['intervalo_s']:g} s):")
    line("CPU del proceso", r["cpu_proceso_%"], "% de un hilo")
    line(f"CPU del equipo ({r['hilos_del_equipo']} hilos)", r["cpu_equipo_%"], "%")
    line("RAM del proceso", r["ram_mib"], "MiB")
    line("Memoria de GPU del proceso", r["gpu_memoria_mib"], "MiB")
    line("Uso de la GPU (toda la tarjeta)", r["gpu_uso_tarjeta_%"], "%")
    if r.get("gpu_aviso"):
        print(f"     Sin datos de la GPU: {r['gpu_aviso']}")
