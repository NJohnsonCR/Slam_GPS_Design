"""
Camera and GPS fusion: an extended Kalman filter on the plane.

    state    [x, y, heading, s]   metres east and north of the first fix,
                                  radians from east (counter-clockwise), and
                                  the factor that corrects the depth model's
                                  scale
    predict  every frame: heading += camera yaw
                          position += s * camera step * (cos, sin)(heading)
    correct  every fix:   GPS position, Doppler speed (= s * camera speed)
                          and the course between consecutive fixes

The camera supplies how much the car turned and how far it went; the GPS
keeps both honest. Between fixes (1 s in the car) the camera carries the
track, and during a GPS outage it carries it alone with the last s.

The car is taken to move along its heading, never sideways or in reverse, so
the per-frame direction of the VO translation (noisy, ~13 degrees after JPEG)
is not used. Without IMU the vertical comes from the mount: the camera's y
axis, within ~3 degrees of gravity with the phone on the dashboard.
"""

import collections

import cv2
import numpy as np

UP_CAMERA = np.array([0.0, -1.0, 0.0])      # OpenCV camera axes: y points down


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def frame_inputs(R, t, step_m, up=UP_CAMERA):
    """
    Yaw (radians, + = left turn) and metres of one frame, from the raw
    recoverPose output and the depth model's step. Also whether there was a
    pose: without one the yaw is 0 but unknown.
    """
    # R maps points from the previous camera to the current one; the camera
    # turns by R^T, whose rotation vector is minus that of R.
    yaw = -float(cv2.Rodrigues(np.asarray(R, float))[0].ravel() @ up)
    step = float(step_m) if step_m is not None and np.isfinite(step_m) else 0.0
    return yaw, step, bool(np.any(np.asarray(t) != 0))


class PlanarEKF:
    """The filter itself. Noise levels were measured on the car recordings."""

    GPS_SIGMA_M = 4.0               # phone fix outdoors
    DOPPLER_SIGMA = 0.3             # m/s
    # The depth model's speed against Doppler: a fast relative noise plus a
    # drift with scene and speed, which s has to follow.
    CAMERA_SPEED_REL = 0.2
    SCALE_RW = 0.1                  # relative, per sqrt(s)
    COURSE_SIGMA_M = 0.5            # error between consecutive fixes (their steps match Doppler to 0.3 m)
    YAW_RW = np.radians(2.2)        # rad/sqrt(s): camera heading against the gyroscope
    MISSED_YAW = np.radians(2.0)    # a frame without pose
    STEP_REL = 0.1                  # per frame; the slow part of the step error goes into s
    GATE_2D, GATE_1D = 13.8, 10.8   # chi-square at 99.9 %
    # Physical range of s. Measured 0.3-3.3; beyond it the camera speed is a
    # failure of the depth model, not a scale.
    S_RANGE = (0.2, 5.0)

    def __init__(self, x, y, heading, s=1.0, heading_sigma=np.radians(10.0), s_sigma=0.5):
        self.x = np.array([x, y, heading, s], float)
        self.P = np.diag([self.GPS_SIGMA_M ** 2, self.GPS_SIGMA_M ** 2,
                          heading_sigma ** 2, s_sigma ** 2])
        self.innovation = None          # of the last update: measured minus predicted

    def predict(self, yaw, step, dt, has_pose=True):
        x, y, th, s = self.x
        mid = th + 0.5 * yaw
        c, sn = np.cos(mid), np.sin(mid)
        self.x = np.array([x + s * step * c, y + s * step * sn, wrap(th + yaw), s])

        F = np.eye(4)
        F[0, 2], F[0, 3] = -s * step * sn, step * c
        F[1, 2], F[1, 3] = s * step * c, step * sn
        # Noise of the two inputs, step and yaw, mapped onto the state.
        G = np.array([[s * c, -0.5 * s * step * sn],
                      [s * sn, 0.5 * s * step * c],
                      [0.0, 1.0],
                      [0.0, 0.0]])
        yaw_var = self.YAW_RW ** 2 * dt + (0.0 if has_pose else self.MISSED_YAW ** 2)
        Q = G @ np.diag([(self.STEP_REL * step) ** 2, yaw_var]) @ G.T
        Q[3, 3] += (self.SCALE_RW * s) ** 2 * dt
        self.P = F @ self.P @ F.T + Q

    def _update(self, innov, H, Rm, gate, force=False):
        """
        EKF update with an innovation gate. Returns ("ok", "rejected" or
        "forced", NIS).

        force accepts a measurement beyond the gate, after inflating the
        covariance until it fits: when several in a row disagree, the
        prediction is what went wrong.
        """
        self.innovation = innov
        S = H @ self.P @ H.T + Rm
        nis = float(innov @ np.linalg.solve(S, innov))
        status = "ok"
        if nis > gate:
            if not force:
                return "rejected", nis
            status = "forced"
            self.P *= nis / gate
            S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.x[2] = wrap(self.x[2])
        self.x[3] = np.clip(self.x[3], *self.S_RANGE)
        A = np.eye(4) - K @ H
        self.P = A @ self.P @ A.T + K @ Rm @ K.T      # Joseph form, stays symmetric
        return status, nis

    def update_position(self, xy, lag_s, v_cam, force=False):
        """
        A fix taken lag_s seconds before the current state; the car's motion
        since then is taken back out of the prediction.
        """
        _, _, th, s = self.x
        u = np.array([np.cos(th), np.sin(th)])
        back = lag_s * v_cam
        h = self.x[:2] - s * back * u
        H = np.zeros((2, 4))
        H[:, :2] = np.eye(2)
        H[:, 2] = -s * back * np.array([-u[1], u[0]])
        H[:, 3] = -back * u
        return self._update(np.asarray(xy) - h, H, np.eye(2) * self.GPS_SIGMA_M ** 2,
                            self.GATE_2D, force)

    def update_speed(self, v_gps, v_cam, force=False):
        H = np.array([[0.0, 0.0, 0.0, v_cam]])
        var = self.DOPPLER_SIGMA ** 2 + (self.x[3] * self.CAMERA_SPEED_REL * v_cam) ** 2
        return self._update(np.array([v_gps - self.x[3] * v_cam]), H,
                            np.array([[var]]), self.GATE_1D, force)

    def update_course(self, course, turned_since, chord_m, force=False):
        """
        Course of the chord between two fixes. It is the heading at the middle
        of the interval, so the yaw turned since then is added back.
        """
        H = np.array([[0.0, 0.0, 1.0, 0.0]])
        var = 2 * (self.COURSE_SIGMA_M / chord_m) ** 2
        innov = wrap(course - (self.x[2] - turned_since))
        return self._update(np.array([innov]), H, np.array([[var]]), self.GATE_1D, force)


class GpsCameraFusion:
    """
    Feeds a PlanarEKF in time order: frames predict, fixes correct. Starts
    once the GPS has seen the car move INIT_METRES, which gives a first
    heading.
    """

    MIN_SPEED = 2.0         # m/s: below this the camera/Doppler ratio means nothing
    MIN_CHORD_M = 3.0       # shortest chord whose course is used
    INIT_SPEED = 3.0        # m/s
    INIT_METRES = 20.0
    HISTORY_S = 30.0
    MAX_REJECTED = 2        # a GPS jump is taken as such at most twice in a row

    def __init__(self, use_course=True):
        self.use_course = use_course
        self.ekf = None
        self.t_ns = None
        self.v_cam = 0.0
        self.yaw_total = 0.0
        self._yaw_hist = collections.deque()      # (t_ns, accumulated yaw)
        self._prev_fix = None                     # (t_ns, xy)
        self._anchor = None
        self._rejected = {"pos": 0, "speed": 0, "course": 0}
        self.log = []                             # (t_ns, kind, status, nis, innovation)

    @property
    def ready(self):
        return self.ekf is not None

    def on_frame(self, t_ns, yaw, step, has_pose=True):
        dt = 0.0 if self.t_ns is None else (t_ns - self.t_ns) / 1e9
        if dt > 0:
            self.v_cam = step / dt
        self.t_ns = t_ns
        self.yaw_total += yaw
        self._yaw_hist.append((t_ns, self.yaw_total))
        while t_ns - self._yaw_hist[0][0] > self.HISTORY_S * 1e9:
            self._yaw_hist.popleft()
        if self.ekf is not None:
            self.ekf.predict(yaw, step, dt, has_pose)

    def on_fix(self, t_ns, xy, speed):
        xy = np.asarray(xy, float)[:2]
        prev, self._prev_fix = self._prev_fix, (t_ns, xy)
        if self.t_ns is None:
            return                  # no frame yet: nothing to correct
        if self.ekf is None:
            self._try_start(t_ns, xy, speed)
            return

        self._apply("pos", self.ekf.update_position, xy, (self.t_ns - t_ns) / 1e9, self.v_cam)

        # The camera speed is a rolling median, ~1 s behind Doppler; s absorbs
        # that delay too, which is what the prediction needs.
        if speed >= self.MIN_SPEED and self.v_cam > 0:
            lo, hi = PlanarEKF.S_RANGE
            if lo <= speed / self.v_cam <= hi:
                self._apply("speed", self.ekf.update_speed, speed, self.v_cam)
            else:
                # Never forced: no scale explains it.
                self.log.append((self.t_ns, "speed", "out_of_range", np.nan, None))

        if self.use_course and prev is not None and (t_ns - prev[0]) < 1.5e9:
            chord = xy - prev[1]
            dist = float(np.hypot(*chord))
            if dist >= self.MIN_CHORD_M:
                turned = self.yaw_total - self._yaw_at((t_ns + prev[0]) // 2)
                self._apply("course", self.ekf.update_course,
                            np.arctan2(chord[1], chord[0]), turned, dist)

    def _apply(self, kind, update, *args):
        """Run one update, forcing it after MAX_REJECTED rejections in a row."""
        status, nis = update(*args, force=self._rejected[kind] >= self.MAX_REJECTED)
        self._rejected[kind] = self._rejected[kind] + 1 if status == "rejected" else 0
        self.log.append((self.t_ns, kind, status, nis, self.ekf.innovation))

    def _try_start(self, t_ns, xy, speed):
        if speed < self.INIT_SPEED:
            self._anchor = None
            return
        if self._anchor is None:
            self._anchor = (t_ns, xy)
            return
        t_a, xy_a = self._anchor
        chord = xy - xy_a
        if np.hypot(*chord) < self.INIT_METRES:
            return
        heading = np.arctan2(chord[1], chord[0])
        heading += self.yaw_total - self._yaw_at((t_a + t_ns) // 2)
        self.ekf = PlanarEKF(xy[0], xy[1], wrap(heading))

    def _yaw_at(self, t_ns):
        ts, ys = zip(*self._yaw_hist)
        return float(np.interp(t_ns, ts, ys))


def session_inputs(R, tv, steps):
    """frame_inputs for every frame of a session, as (yaw, step, has_pose) arrays."""
    rows = [frame_inputs(R[k], tv[k], steps[k]) for k in range(len(R))]
    yaw, step, has_pose = (np.array(c) for c in zip(*rows))
    return yaw, step, has_pose


def run_fusion(t_ns, inputs, fixes, outages=(), use_course=True):
    """
    Run the fusion over a processed session.

    inputs comes from session_inputs. fixes holds (t_ns, xy, speed) in time
    order; a fix inside any outage (t_from_ns, t_to_ns) is not delivered.
    use_course=False leaves out the course between fixes, to measure what it
    adds. Returns the fusion object, the state (x, y, heading, s) after every
    frame and its standard deviations, NaN before the start.
    """
    fus = GpsCameraFusion(use_course)
    out = np.full((len(t_ns), 4), np.nan)
    sig = np.full((len(t_ns), 4), np.nan)
    yaw, step, has_pose = inputs
    j = 0
    for k in range(len(t_ns)):
        fus.on_frame(int(t_ns[k]), yaw[k], step[k], has_pose[k])
        # A fix is available from the first frame after its timestamp, as in
        # GpsBuffer.latest_before.
        while j < len(fixes) and fixes[j][0] <= t_ns[k]:
            tf = fixes[j][0]
            if not any(a <= tf <= b for a, b in outages):
                fus.on_fix(*fixes[j])
            j += 1
        if fus.ready:
            out[k] = fus.ekf.x
            sig[k] = np.sqrt(np.diag(fus.ekf.P))
    return fus, out, sig
