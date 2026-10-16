"""
The RL's environment: how much to trust the camera's speed while the GPS is
out, decided once a second from what the filter sees.

    episode  a cached session and a GPS outage of 10-60 s, run from a copy of
             the fixed filter taken where the outage starts, until the GPS
             comes back
    action   the camera speed's variance times k in ACTIONS; inf ignores it,
             and the filter goes on with its own speed and the gyroscope.
             Asked only after GPS_GAP_S without a fix: with the GPS there the
             camera speed adds nothing (ignoring it all session does not
             worsen the prediction of each fix), so the fixed filter decides
    state    RlState: five signals of the filter, three levels each
    reward   minus the distance (m) to each hidden fix and to the fix the GPS
             comes back with, minus SWITCH_COST per change of action; or, with
             the fixed filter's run of the same outage as baseline, how much
             closer than it, which takes out how hard each stretch is

The filter is fusion.GpsCameraFusion untouched: PolicyFusion only changes how
the camera speed's update is weighted, so with k = 1 it is the fixed EKF to
the last digit, and the cache stays valid for any policy. Everything runs on
the cache (frames.csv and location.csv), without video:

    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.rl.env
"""

import argparse
import collections
import copy
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_LMS_RL = os.path.abspath(os.path.join(_THIS, "..", ".."))
_ROOT = os.path.abspath(os.path.join(_LMS_RL, "..", ".."))
for _p in (_ROOT, _LMS_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realtime import cache
from realtime.evaluate import FUSION, load_frames, load_gps
from realtime.fusion import GpsCameraFusion, PlanarEKF, feed_frame, run_fusion, session_inputs

ACTIONS = (1.0, 4.0, np.inf)        # k: the camera speed's variance times k
SWITCH_COST = 0.1                   # m, per change of action
OUTAGE_LENGTHS = (10, 20, 30, 60)   # s
# Recorded with another app, the phone upright and another camera: kept out of training.
VALIDATION = ("2025_03_11",)


# ------------------------------------------------------------------- state

@dataclass(frozen=True)
class RlState:
    """What the policy sees at each decision, before the levels."""
    since_fix_s: float          # seconds since the last fix the filter got (into the outage)
    speed: float                # the filter's speed, m/s
    ratio_spread: float         # spread of Doppler / camera speed over the last fixes
    camera_nis: float           # mean NIS of the last camera speeds (as the fixed filter weighs them)
    stopped: float              # share of frames the stop gate called stopped, last seconds

    # Upper bounds of levels 0 and 1 for each signal; the third level is above.
    # The spread and the NIS at their terciles over the grid outages of the six
    # training sessions (15/10/2026); the others where they mean something.
    LEVELS = {"since_fix_s": (5.0, 15.0),       # just out / a while / long
              "speed": (6.0, 12.0),             # m/s
              "ratio_spread": (0.17, 0.29),
              "camera_nis": (0.28, 1.03),
              "stopped": (0.0, 0.5)}            # none / some / most
    N_STATES = 3 ** 5

    def levels(self):
        out = []
        for name, (a, b) in self.LEVELS.items():
            v = getattr(self, name)
            # Not known yet (no ratios, no camera speed): the middle level.
            out.append(1 if not np.isfinite(v) else 0 if v <= a else 1 if v <= b else 2)
        return tuple(out)

    def index(self):
        i = 0
        for level in self.levels():
            i = 3 * i + level
        return i


# ------------------------------------------------------------------ filter

class PolicyFusion(GpsCameraFusion):
    """
    The fixed filter, with the camera speed weighted by a controller: a
    function of RlState that returns k. Without one, k = 1. The signals of the
    state are kept all the time, also while the controller is not asked.
    """

    GPS_GAP_S = 1.5             # without a fix for this long, the controller decides
    RATIO_FIXES = 10            # fixes the Doppler / camera spread is taken over
    NIS_UPDATES = 5             # camera speeds the NIS is averaged over
    STOP_WINDOW_S = 5.0

    def __init__(self, controller: Optional[Callable] = None, **modes):
        super().__init__(**modes)
        self.controller = controller
        self.k = 1.0
        self.decisions = []                     # (t_ns, RlState, k)
        self._ratios = collections.deque(maxlen=self.RATIO_FIXES)
        self._nis = collections.deque(maxlen=self.NIS_UPDATES)
        self._stops = collections.deque()       # (t_ns, stopped) per frame with a camera speed
        self.n_ignored = 0

    def on_frame(self, t_ns, yaw, step, has_pose=True, has_scale=True, gyro_yaw=None):
        if has_scale:
            self._stops.append((t_ns, step == 0.0))
            while t_ns - self._stops[0][0] > self.STOP_WINDOW_S * 1e9:
                self._stops.popleft()
        super().on_frame(t_ns, yaw, step, has_pose, has_scale, gyro_yaw)

    def on_fix(self, t_ns, xy, speed):
        if self.ready and speed >= self.MIN_SPEED and self.v_cam > 0:
            self._ratios.append(speed / self.v_cam)
        super().on_fix(t_ns, xy, speed)

    def state(self) -> RlState:
        r = np.array(self._ratios)
        return RlState(
            since_fix_s=(self.t_ns - self._last_fix) / 1e9,
            speed=float(self.ekf.x[self.ekf.i["v"]]),
            ratio_spread=float(np.std(r) / np.mean(r)) if len(r) >= 3 else np.nan,
            camera_nis=float(np.mean(self._nis)) if self._nis else np.nan,
            stopped=float(np.mean([s for _, s in self._stops])) if self._stops else 0.0)

    def _camera_speed(self, hold_scale):
        # The NIS the fixed filter would see, on a copy: it keeps coming while
        # the camera is ignored.
        if self.v_cam > 0:
            probe = copy.deepcopy(self.ekf)
            _, nis = probe.update_camera_speed(self.v_cam, hold_scale=hold_scale)
            self._nis.append(nis)
        k = 1.0
        if self.controller is not None and self.t_ns - self._last_fix >= self.GPS_GAP_S * 1e9:
            st = self.state()
            k = float(self.controller(st))
            self.decisions.append((self.t_ns, st, k))
        self.k = k
        if k == 1.0:
            super()._camera_speed(hold_scale)
        elif np.isinf(k):
            self.n_ignored += 1
        else:
            # Instance attributes shadow the class's noise levels for this
            # one update: the variance (stopped or moving) times k.
            ekf, root = self.ekf, np.sqrt(k)
            ekf.CAMERA_SPEED_REL = PlanarEKF.CAMERA_SPEED_REL * root
            ekf.CAMERA_SPEED_FLOOR = PlanarEKF.CAMERA_SPEED_FLOOR * root
            ekf.STOP_SIGMA = PlanarEKF.STOP_SIGMA * root
            try:
                super()._camera_speed(hold_scale)
            finally:
                del ekf.CAMERA_SPEED_REL, ekf.CAMERA_SPEED_FLOOR, ekf.STOP_SIGMA


# --------------------------------------------------------------- sessions

@dataclass
class Session:
    name: str
    t: np.ndarray               # frame times, ns
    inputs: tuple               # session_inputs
    fixes: list                 # (t_ns, xy, Doppler speed)
    gyro_yaw: np.ndarray        # per frame, NaN where the camera turned

    @property
    def tg(self):
        return np.array([f[0] for f in self.fixes], np.int64)

    @property
    def xy(self):
        return np.array([f[1] for f in self.fixes])


def load_session(entry_dir) -> Session:
    """A cached replay as the filter's inputs: frames.csv and location.csv."""
    frames = next(os.path.join(entry_dir, f) for f in sorted(os.listdir(entry_dir))
                  if f.startswith("frames_deterministic"))
    t, R, tv, s = load_frames(frames)
    tg, xy, _, vd = load_gps(entry_dir, t[0], t[-1])
    gyro = pd.read_csv(frames, usecols=["gyro_yaw"])["gyro_yaw"].to_numpy(float)
    fixes = [(int(a), b, float(c)) for a, b, c in zip(tg, xy, vd)]
    name = os.path.basename(os.path.dirname(os.path.normpath(entry_dir)))
    return Session(name, t, session_inputs(R, tv, s), fixes, gyro)


def cached_sessions(report=print) -> List[Session]:
    """
    The sessions the RL can use: full replays with depth scale and the
    filter's modes, made by the current code, with their location.csv.
    """
    current = cache.code_hash()
    found = {}
    for rec, key, folder, m in cache.entries():
        o = m["opciones"]
        why = None
        if m["hash_codigo"] != current:
            why = "código anterior"
        elif o["scale"] != "depth" or o["frames"] or o["start"] or o["visual_only"] or o["no_gyro"]:
            why = "no es el replay completo con escala"
        elif not os.path.exists(os.path.join(folder, cache.LOCATION)):
            why = f"sin {cache.LOCATION} (python -m LMS.LMS_RL_ORB_GPS.realtime.cache --add-location)"
        if why:
            report(f"  {rec}/{key}: no se usa ({why})")
        elif rec in found:
            raise RuntimeError(f"{rec} tiene dos entradas que sirven: dejar una")
        else:
            found[rec] = folder
    return [load_session(found[rec]) for rec in sorted(found)]


# ---------------------------------------------------------------- episodes

class Snapshots:
    """
    One pass of the fixed filter over a session, keeping a copy of it every
    EVERY_S from the start: each episode begins from one, so an outage costs
    only its own frames.
    """

    EVERY_S = 1.0

    def __init__(self, session: Session):
        self.session = s = session
        self.copies = []                        # (t_ns, frame index, fix index, filter)
        fus = PolicyFusion(**FUSION)
        every, next_t, j = int(self.EVERY_S * 1e9), None, 0
        for k in range(len(s.t)):
            if fus.ready:
                next_t = s.t[k] if next_t is None else next_t
                if s.t[k] >= next_t:
                    log, fus.log = fus.log, []  # the copies need no history
                    self.copies.append((int(next_t), k, j, copy.deepcopy(fus)))
                    fus.log = log
                    next_t += every
            j = feed_frame(fus, k, s.t, s.inputs, s.gyro_yaw, s.fixes, j)
        self.start_t = self.copies[0][0] if self.copies else None


@dataclass
class Episode:
    session: str
    start_s: float              # outage start, seconds from the filter's start
    length_s: int
    errors: list                # (t_ns, metres) at each hidden fix, then at the return
    decisions: list             # (t_ns, RlState, k)
    reward: float

    @property
    def return_error(self):
        return self.errors[-1][1]


def run_episode(snaps: Snapshots, i: int, length_s: int, controller: Callable) -> Optional[Episode]:
    """
    The outage that starts at snapshot i, with the controller deciding k
    from its first second on. None when there is no fix before it or after.
    """
    s = snaps.session
    tg, xy = s.tg, s.xy
    t_from, k, j, fus = snaps.copies[i]
    t_to = t_from + int(length_s * 1e9)
    i0 = np.searchsorted(tg, t_from) - 1                # last fix before the outage
    i_end = np.searchsorted(tg, t_to, side="right")     # the fix the GPS comes back with
    if i0 < 1 or i_end >= len(tg):
        return None
    fus = copy.deepcopy(fus)
    fus.controller = controller
    targets = list(range(i0 + 1, i_end + 1))            # hidden fixes, then the return
    errors, q = [], 0
    while k < len(s.t) and q < len(targets):
        while q < len(targets) and tg[targets[q]] <= s.t[k]:
            errors.append((int(tg[targets[q]]), float(np.linalg.norm(fus.ekf.x[:2] - xy[targets[q]]))))
            q += 1
        if q == len(targets):
            break
        j = feed_frame(fus, k, s.t, s.inputs, s.gyro_yaw, s.fixes, j, [(t_from, t_to)])
        k += 1
    if q < len(targets):
        return None
    ks = [1.0] + [d[2] for d in fus.decisions]
    switches = sum(a != b for a, b in zip(ks[:-1], ks[1:]))
    reward = -sum(e for _, e in errors) - SWITCH_COST * switches
    return Episode(s.name, (t_from - snaps.start_t) / 1e9, length_s, errors, fus.decisions, reward)


def transitions(ep: Episode, baseline: Optional[dict] = None):
    """
    The episode as (state index, action index, reward, next state index or
    None at the end): each error goes to the decision in force when it was
    measured, each switch to the decision that made it. With baseline, the
    fixed filter's error at each fix of the same outage ({t_ns: metres}),
    each error counts as how much farther than the baseline's.
    """
    if not ep.decisions:
        return []
    times = np.array([d[0] for d in ep.decisions], np.int64)
    rewards = np.zeros(len(ep.decisions))
    for t_err, e in ep.errors:
        n = max(int(np.searchsorted(times, t_err, side="right")) - 1, 0)
        rewards[n] -= e - (baseline[t_err] if baseline is not None else 0.0)
    prev = 1.0
    for n, (_, _, k) in enumerate(ep.decisions):
        if k != prev:
            rewards[n] -= SWITCH_COST
        prev = k
    idx = [d[1].index() for d in ep.decisions]
    acts = [ACTIONS.index(d[2]) for d in ep.decisions]
    return [(idx[n], acts[n], rewards[n], idx[n + 1] if n + 1 < len(idx) else None)
            for n in range(len(idx))]


def grid_episodes(snaps: Snapshots, controller, lengths=OUTAGE_LENGTHS, every_s=10):
    """The evaluation outages: one every every_s from 10 s after the start, of each length."""
    step = int(round(every_s / Snapshots.EVERY_S))
    out = {L: [] for L in lengths}
    for i in range(step, len(snaps.copies), step):
        for L in lengths:
            ep = run_episode(snaps, i, L, controller)
            if ep is not None:
                out[L].append(ep)
    return out


def constant(k):
    return lambda state: k


# ------------------------------------------------------------------- check

def check_identity(session: Session):
    """PolicyFusion with k = 1 against run_fusion over the whole session: the largest difference."""
    _, ref, _ = run_fusion(session.t, session.inputs, session.fixes, gyro_yaw=session.gyro_yaw, **FUSION)
    fus = PolicyFusion(**FUSION)
    out, j = np.full((len(session.t), 2), np.nan), 0
    for k in range(len(session.t)):
        j = feed_frame(fus, k, session.t, session.inputs, session.gyro_yaw, session.fixes, j)
        if fus.ready:
            out[k] = fus.ekf.x[:2]
    ok = np.isfinite(ref[:, 0])
    if not np.array_equal(ok, np.isfinite(out[:, 0])):
        return np.inf
    return float(np.max(np.abs(out[ok] - ref[ok, :2]))) if ok.any() else 0.0


def main():
    ap = argparse.ArgumentParser(description="Revisa el entorno del RL sobre la caché")
    ap.parse_args()
    print("Sesiones de la caché:")
    sessions = cached_sessions(report=lambda m: print(m))
    for s in sessions:
        print(f"  {s.name}: {len(s.t)} frames, {len(s.fixes)} fixes"
              + ("  (validación)" if s.name in VALIDATION else ""))
    print("\nEl filtro del RL con k = 1, contra el EKF fijo (diferencia máxima de posición):")
    broken = []
    for s in sessions:
        diff = check_identity(s)
        print(f"  {s.name}: {diff:.3g} m")
        if diff != 0.0:
            broken.append(s.name)
    if broken:
        # PolicyFusion leans on GpsCameraFusion's internals: a change there shows up here.
        print(f"\n  FALLA: con k = 1 no es el EKF fijo en {', '.join(broken)}.")
        return 1
    print("\nCortes cada 10 s, error al volver el GPS (p50 / p90, m), con k fijo en todo el corte:")
    names = {1.0: "k = 1 (EKF fijo)", 4.0: "k = 4", np.inf: "sin la cámara"}
    for s in sessions:
        t0 = time.perf_counter()
        snaps = Snapshots(s)
        res = {k: grid_episodes(snaps, constant(k)) for k in ACTIONS}
        n_ep = sum(len(v) for r in res.values() for v in r.values())
        for L in OUTAGE_LENGTHS:
            errs = np.array([[ep.return_error for ep in res[k][L]] for k in ACTIONS])
            if not errs.size:
                continue
            row = " | ".join(f"{names[k]} {np.percentile(e, 50):5.1f} / {np.percentile(e, 90):5.1f}"
                             for k, e in zip(ACTIONS, errs))
            best = errs.min(axis=0)
            print(f"  {s.name} {L:2d} s ({errs.shape[1]:3d}): {row} | oráculo "
                  f"{np.percentile(best, 50):5.1f} / {np.percentile(best, 90):5.1f}")
        print(f"  ({n_ep} episodios en {time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
