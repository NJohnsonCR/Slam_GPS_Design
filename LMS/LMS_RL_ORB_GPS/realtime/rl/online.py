"""
The RL inside the pipeline: OnlineFusion with simulated GPS outages and, with
a policy, the camera's weight in them decided by it.

    OutageGps       the GPS buffer as the fusion sees it: the fixes stamped
                    inside an outage are kept from it, and listed for the map
    RlOnlineFusion  OnlineFusion over an OutageGps. With a policy its filter is
                    env.PolicyFusion, which is the fixed EKF while the GPS is
                    there, and the fixed filter runs beside it on the same
                    inputs, to compare them; without one it is OnlineFusion

At the fix the GPS comes back with after each outage, the distance from each
filter to it before it corrects them: the error the RL was evaluated with
(env.run_episode). fusion.py stays untouched, so the cache stays valid.
"""

import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_LMS_RL = os.path.abspath(os.path.join(_THIS, "..", ".."))
_ROOT = os.path.abspath(os.path.join(_LMS_RL, "..", ".."))
for _p in (_ROOT, _LMS_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realtime.evaluate import load_frames
from realtime.fusion import OnlineFusion, frame_inputs
from realtime.rl.env import PolicyFusion
from realtime.rl.policy import POLICY_FILE, QTable
from realtime.sources import GpsBuffer, ImuBuffer


def load_policy(path=POLICY_FILE) -> QTable:
    """The trained policy the pipeline uses."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no está la política del RL: {path}")
    return QTable.load(path)


class OutageGps:
    """
    A GpsBuffer as the fusion sees it with simulated outages: the fixes
    stamped inside one, (t_from_ns, t_to_ns), are kept from it.
    """

    def __init__(self, gps):
        self.gps = gps
        self.outages = []
        # Every fix seen, in time order, as (t_ns, position, speed, hidden): the
        # map still draws the hidden ones, as the reference.
        self.seen = []

    def cut(self, t_from_ns, length_s):
        self.outages.append((int(t_from_ns), int(t_from_ns) + int(round(length_s * 1e9))))

    def hidden(self, t_ns):
        # The same bounds as feed_frame's outages.
        return any(a <= t_ns <= b for a, b in self.outages)

    def between(self, after_ns, up_to_ns):
        out = []
        for fix in self.gps.between(after_ns, up_to_ns):
            hide = self.hidden(fix[0])
            # A hidden fix keeps coming back while the fusion waits past it.
            if not self.seen or fix[0] > self.seen[-1][0]:
                self.seen.append((*fix, hide))
            if not hide:
                out.append(fix)
        return out


class RlOnlineFusion(OnlineFusion):
    """
    OnlineFusion over simulated outages, and with a policy (policy.QTable)
    deciding k once the GPS has been out GPS_GAP_S. cut() can be called from
    any thread: the outage starts at the next frame. The summary gives the
    outages in seconds from origin_ns (the recording's start), or from the
    first frame without it.
    """

    def __init__(self, gps_buffer, imu_buffer=None, policy=None, origin_ns=None, **modes):
        self.outage = OutageGps(gps_buffer)
        super().__init__(self.outage, imu_buffer, **modes)
        self.policy = policy
        self.origin_ns = origin_ns
        self.fixed = None               # the fixed filter beside the policy's
        self.fixed_track = []           # its (x, y) after each frame, None before it starts
        self.in_outage = []             # per frame, whether a simulated outage hid the GPS
        if policy is not None:
            plain = self.fus
            self.fus = PolicyFusion(policy.controller(), use_course=plain.use_course,
                                    gyro=plain.gyro, speed_state=plain.speed_state,
                                    bias=plain.bias, visual_only=plain.visual_only)
            self.fixed = plain
        # Per outage the GPS came back from: (t_from_ns, t_to_ns, t_fix_ns,
        # hidden fixes, error of the fixed filter, error of the policy's).
        self.returns = []
        self._pending = None
        self._done = set()

    def cut(self, length_s):
        """A simulated outage of length_s seconds from the next frame on."""
        self._pending = float(length_s)

    def step(self, t_ns, R, t, step_m):
        pending, self._pending = self._pending, None
        if pending:
            self.outage.cut(t_ns, pending)
        # Where each filter was before this frame: the fix it may bring is
        # measured against that, as in the evaluation.
        filters = (self.fixed, self.fus) if self.fixed is not None else (self.fus,)
        before = None
        if self.outage.outages:
            before = [f.ekf.x[:2].copy() if f.ready else None for f in filters]
        n = len(self.fixes)
        self.in_outage.append(bool(self.outage.outages) and self.outage.hidden(t_ns))
        gyro_yaw, row = super().step(t_ns, R, t, step_m)
        if self.fixed is not None:
            self.fixed.on_frame(t_ns, *frame_inputs(R, t, step_m), gyro_yaw=gyro_yaw)
            for fix in self.fixes[n:]:
                self.fixed.on_fix(*fix)
            self.fixed_track.append(tuple(self.fixed.ekf.x[:2]) if self.fixed.ready else None)
        if before is not None and len(self.fixes) > n:
            self._came_back(self.fixes[n], self.fixes[n - 1][0] if n else -1, before)
        return gyro_yaw, row

    def _came_back(self, fix, prev_t, before):
        """The errors at the first fix after an outage, once per outage."""
        xy = np.asarray(fix[1], float)[:2]
        for a, b in self.outage.outages:
            if (a, b) in self._done or not prev_t < b < fix[0]:
                continue
            self._done.add((a, b))
            err = [None if p is None else float(np.linalg.norm(p - xy)) for p in before]
            hidden = sum(1 for s in self.outage.seen if s[3] and a <= s[0] <= b)
            self.returns.append((a, b, fix[0], hidden, err[0], err[1] if len(err) > 1 else None))

    @property
    def decision(self):
        """The k the policy chose last, while it decides (the GPS out); None otherwise."""
        fus = self.fus
        if self.policy is None or not fus.decisions or fus.decisions[-1][0] < fus._last_fix:
            return None
        return fus.decisions[-1][2]

    def outage_left_s(self):
        """Seconds left of the simulated outage the last frame is in, None outside one."""
        t = self.fus.t_ns
        for a, b in self.outage.outages:
            if t is not None and a <= t <= b:
                return (b - t) / 1e9
        return None

    def summary(self) -> dict:
        out = super().summary()
        if self.policy is not None:
            ks = [d[2] for d in self.fus.decisions]
            out["rl"] = {"politica": os.path.relpath(self.policy.path, _ROOT),
                         "acciones": [f"{k:g}" for k in self.policy.actions],
                         "decisiones": len(ks),
                         "por_accion": {f"{k:g}": ks.count(k) for k in self.policy.actions}}
        if self.outage.outages:
            back = {r[:2]: r for r in self.returns}
            origin = self.first_t if self.origin_ns is None else self.origin_ns
            out["cortes_simulados"] = []
            for a, b in self.outage.outages:
                r = back.get((a, b))
                out["cortes_simulados"].append({
                    "desde_s": (a - origin) / 1e9, "duracion_s": (b - a) / 1e9,
                    "volvio_el_gps": r is not None,
                    "fixes_escondidos": None if r is None else r[3],
                    "error_al_volver_m": None if r is None else {"ekf_fijo": r[4], "rl": r[5]}})
        return out


def over_frames(frames_csv, fixes, imu=None, policy=None, outages=(), origin_ns=None, **modes):
    """
    The fusion of a replay again over its saved frames (frames.csv), fed the
    same fixes and IMU samples as the pipeline was: what its fusion does with
    them, in seconds instead of the video's time. outages holds (t_from_ns,
    seconds). Returns the RlOnlineFusion after the last frame and the state
    row it gave for each (fusion.STATE_ROW, None before the start).
    """
    t, R, tv, s = load_frames(frames_csv)
    t_from, t_to = int(t[0]) - 2_000_000_000, int(t[-1])     # as run_replay keeps them
    gps = GpsBuffer(maxlen=None)
    for fix in fixes:
        if t_from <= fix[0] <= t_to:
            gps.push(*fix)
    imu_buf = None
    if imu:
        imu_buf = ImuBuffer(maxlen=None)
        for sample in imu:
            if t_from <= sample[0] <= t_to:
                imu_buf.push(*sample)
    fusion = RlOnlineFusion(gps, imu_buf, policy=policy, origin_ns=origin_ns, **modes)
    for a, length_s in outages:
        fusion.outage.cut(a, length_s)
    rows = [fusion.step(int(t[k]), R[k], tv[k], s[k])[1] for k in range(len(t))]
    return fusion, rows
