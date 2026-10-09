"""
Camera, gyroscope and GPS fusion: an extended Kalman filter on the plane.

    state    [x, y, heading, s, v, b]   metres east and north of the first fix,
                                        radians from east (counter-clockwise),
                                        the factor that corrects the depth
                                        model's scale, the speed (m/s) and the
                                        gyroscope bias (rad/s); v and b are
                                        optional
    predict  every frame: heading += gyroscope yaw - b * dt (camera yaw when
                          there are no gyro samples)
                          position += v * dt * (cos, sin)(heading)
    correct  every fix:   GPS position, Doppler speed (= v) and the course
                          between consecutive fixes
             every second: the camera speed (= v / s), or v = 0 when the stop
                          gate sees the car stopped

The gyroscope carries the heading and the camera the speed; the GPS keeps
both honest. During a GPS outage the camera speed can be rejected, and then
the filter goes on with the speed it had. Without the speed state the camera
step drives the prediction directly (position += s * step), as the first
version of the filter did; the evaluation compares both.

The car is taken to move along its heading, never sideways or in reverse, so
the per-frame direction of the VO translation (noisy, ~13 degrees after JPEG)
is not used. The camera yaw needs the vertical from the mount: the camera's y
axis, within ~3 degrees of gravity with the phone on the dashboard. The
gyroscope takes it from the accelerometer.
"""

import bisect
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
    pose (without one the yaw is 0 but unknown) and whether the depth model
    gave a speed (exactly 0 when the stop gate sees the car stopped).
    """
    # R maps points from the previous camera to the current one; the camera
    # turns by R^T, whose rotation vector is minus that of R.
    yaw = -float(cv2.Rodrigues(np.asarray(R, float))[0].ravel() @ up)
    has_scale = step_m is not None and bool(np.isfinite(step_m))
    step = float(step_m) if has_scale else 0.0
    return yaw, step, bool(np.any(np.asarray(t) != 0)), has_scale


class GyroYaw:
    """
    Gyroscope samples turned into yaw: the turn about the vertical between two
    frames. The vertical is a slow low-pass of the accelerometer, where the
    car's own accelerations average out. Fed sample by sample, as the live
    pipeline receives them; a frame a little beyond the last sample is
    extrapolated with the last rate, and the next frame takes up the
    difference, so the total turn stays exact.
    """

    GRAVITY_TAU_S = 20.0
    MAX_GAP_S = 0.25            # a longer gap leaves its frames to the camera
    MAX_AHEAD_S = 0.1           # how far past the last sample a frame can be
    HISTORY_S = 5.0

    def __init__(self):
        self._g = None              # low-passed accelerometer, m/s^2
        self._w = None              # last rate vector, rad/s
        self._rate = 0.0            # last rate about the vertical
        self._t = []                # sample times, ns
        self._yaw = []              # yaw accumulated up to each sample
        self._gaps = collections.deque()
        self._taken = None          # (t_ns, accumulated yaw) handed out last

    def add(self, t_ns, w, acc):
        w, acc = np.asarray(w, float), np.asarray(acc, float)
        if self._g is None:
            self._g, self._w = acc, w
            self._t.append(t_ns)
            self._yaw.append(0.0)
            return
        dt = (t_ns - self._t[-1]) / 1e9
        if dt <= 0:
            return
        if dt > self.MAX_GAP_S:
            self._gaps.append((self._t[-1], t_ns))
        self._g = self._g + min(dt / self.GRAVITY_TAU_S, 1.0) * (acc - self._g)
        up = self._g / np.linalg.norm(self._g)
        self._yaw.append(self._yaw[-1] + 0.5 * float((self._w + w) @ up) * dt)
        self._t.append(t_ns)
        self._w, self._rate = w, float(w @ up)
        if len(self._t) > 1000 and t_ns - self._t[500] > self.HISTORY_S * 1e9:
            del self._t[:500], self._yaw[:500]
        while self._gaps and t_ns - self._gaps[0][1] > self.HISTORY_S * 1e9:
            self._gaps.popleft()

    def _at(self, t_ns):
        if not self._t or t_ns < self._t[0]:
            return None
        if t_ns >= self._t[-1]:
            ahead = (t_ns - self._t[-1]) / 1e9
            return self._yaw[-1] + self._rate * ahead if ahead <= self.MAX_AHEAD_S else None
        k = bisect.bisect_right(self._t, t_ns)
        t0, t1 = self._t[k - 1], self._t[k]
        return self._yaw[k - 1] + (self._yaw[k] - self._yaw[k - 1]) * (t_ns - t0) / (t1 - t0)

    def take(self, t_ns):
        """
        Yaw turned since the previous call (radians), or None when the
        samples do not cover that interval: then the frame uses the camera.
        """
        yaw = self._at(t_ns)
        prev, self._taken = self._taken, (None if yaw is None else (t_ns, yaw))
        if yaw is None or prev is None:
            return None
        if any(a < t_ns and b > prev[0] for a, b in self._gaps):
            return None
        return yaw - prev[1]


class PlanarEKF:
    """The filter itself. Noise levels were measured on the car recordings."""

    GPS_SIGMA_M = 4.0               # phone fix outdoors
    DOPPLER_SIGMA = 0.3             # m/s
    # The depth model's speed against Doppler: a fast relative noise plus a
    # drift with scene and speed, which s has to follow. Below ~2 m/s its
    # error is about half the speed.
    CAMERA_SPEED_REL = 0.2
    CAMERA_SPEED_FLOOR = 0.5        # m/s
    SCALE_RW = 0.1                  # relative, per sqrt(s)
    COURSE_SIGMA_M = 0.5            # error between consecutive fixes (their steps match Doppler to 0.3 m)
    YAW_RW = np.radians(2.2)        # rad/sqrt(s): camera heading against the gyroscope
    MISSED_YAW = np.radians(2.0)    # a frame without pose
    STEP_REL = 0.1                  # per frame; the slow part of the step error goes into s
    # Gyroscope heading against the GPS course: an upper bound, since the
    # course's own slow errors are in it too. Its bias drifts by up to
    # ~0.07 deg/s between two-minute windows.
    GYRO_RW = np.radians(0.5)       # rad/sqrt(s)
    BIAS_SIGMA = np.radians(0.07)   # rad/s, at the start
    BIAS_RW = np.radians(0.003)     # rad/s per sqrt(s)
    ACCEL_RW = 1.0                  # m/s per sqrt(s): Doppler, from 1 to 10 s apart
    STOP_SIGMA = 0.5                # m/s: Doppler while the stop gate says stopped (p95 ~1)
    # A held scale differs from the one the next stretch needs by ~18 % (median
    # 16-19 % over 10 s outages): that share of the distance driven since it
    # was held is uncertain along the way.
    HELD_SCALE_ERR = 0.18
    GATE_2D, GATE_1D = 13.8, 10.8   # chi-square at 99.9 %
    # Physical range of s. Measured 0.3-3.3; beyond it the camera speed is a
    # failure of the depth model, not a scale.
    S_RANGE = (0.2, 5.0)

    def __init__(self, x, y, heading, speed=None, bias=False, s=1.0,
                 heading_sigma=np.radians(10.0), s_sigma=0.5):
        """speed (m/s) adds the speed state; bias adds the gyroscope bias."""
        names = ["x", "y", "th", "s"] + (["v"] if speed is not None else []) + (["b"] if bias else [])
        self.i = {n: k for k, n in enumerate(names)}
        state = [x, y, heading, s] + ([speed] if speed is not None else []) + ([0.0] if bias else [])
        sig = [self.GPS_SIGMA_M, self.GPS_SIGMA_M, heading_sigma, s_sigma]
        sig += ([self.DOPPLER_SIGMA] if speed is not None else []) + ([self.BIAS_SIGMA] if bias else [])
        self.x = np.array(state, float)
        self.P = np.diag(np.square(sig))
        self.innovation = None          # of the last update: measured minus predicted
        self._held_m = 0.0              # metres driven with the scale held

    @property
    def speed_state(self):
        return "v" in self.i

    def predict(self, dt, yaw, yaw_var, gyro=False, step=0.0, hold_scale=False):
        """
        One frame: dt seconds, the yaw and its variance (from the gyroscope or
        the camera) and, without the speed state, the camera step in metres.
        hold_scale keeps s from drifting (see update_camera_speed); the error
        of the held scale then grows the uncertainty along the way, in
        proportion to the distance driven, so the filter's sigma follows the
        real error when the GPS comes back.
        """
        n, i = len(self.x), self.i
        th, s = self.x[2], self.x[3]
        F, Q = np.eye(n), np.zeros((n, n))
        unbias = gyro and "b" in i
        if unbias:
            yaw -= self.x[i["b"]] * dt
        mid = th + 0.5 * yaw
        c, sn = np.cos(mid), np.sin(mid)
        d = self.x[i["v"]] * dt if self.speed_state else s * step

        if self.speed_state:
            F[0, i["v"]], F[1, i["v"]] = dt * c, dt * sn
            Q[i["v"], i["v"]] = self.ACCEL_RW ** 2 * dt
        else:
            F[0, 3], F[1, 3] = step * c, step * sn
            g = np.zeros(n)
            g[:2] = s * c, s * sn
            Q += (self.STEP_REL * step) ** 2 * np.outer(g, g)
        F[0, 2], F[1, 2] = -d * sn, d * c
        if unbias:
            F[2, i["b"]] = -dt
            F[0, i["b"]], F[1, i["b"]] = 0.5 * d * dt * sn, -0.5 * d * dt * c
            Q[i["b"], i["b"]] = self.BIAS_RW ** 2 * dt
        # The yaw noise moves the heading and, through the mid-step heading,
        # the position.
        g = np.zeros(n)
        g[:3] = -0.5 * d * sn, 0.5 * d * c, 1.0
        Q += yaw_var * np.outer(g, g)
        if hold_scale and self.speed_state:
            # Variance of HELD_SCALE_ERR times the distance held, which grows
            # by this much in this step.
            d0, d1 = self._held_m, self._held_m + abs(d)
            Q[:2, :2] += self.HELD_SCALE_ERR ** 2 * (d1 ** 2 - d0 ** 2) * np.outer([c, sn], [c, sn])
            self._held_m = d1
        else:
            Q[3, 3] += (self.SCALE_RW * s) ** 2 * dt
            self._held_m = 0.0

        self.x[0] += d * c
        self.x[1] += d * sn
        self.x[2] = wrap(th + yaw)
        self.P = F @ self.P @ F.T + Q

    def _update(self, innov, H, Rm, gate, force=False, keep=()):
        """
        EKF update with an innovation gate. Returns ("ok", "rejected" or
        "forced", NIS).

        force accepts a measurement beyond the gate, after inflating the
        covariance until it fits: when several in a row disagree, the
        prediction is what went wrong. keep lists state indices the update
        must not change.
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
        K[list(keep)] = 0.0
        self.x = self.x + K @ innov
        self.x[2] = wrap(self.x[2])
        self.x[3] = np.clip(self.x[3], *self.S_RANGE)
        A = np.eye(len(self.x)) - K @ H
        self.P = A @ self.P @ A.T + K @ Rm @ K.T      # Joseph form, stays symmetric
        return status, nis

    def position_innovation(self, xy, lag_s, v_cam):
        """
        Fix minus prediction, for a fix taken lag_s seconds before the current
        state: the car's motion since then is taken back out of the
        prediction. Also returns the measurement Jacobian.
        """
        th, s = self.x[2], self.x[3]
        u = np.array([np.cos(th), np.sin(th)])
        H = np.zeros((2, len(self.x)))
        if self.speed_state:
            back = lag_s * self.x[self.i["v"]]
            H[:, self.i["v"]] = -lag_s * u
        else:
            back = lag_s * s * v_cam
            H[:, 3] = -lag_s * v_cam * u
        H[:, :2] = np.eye(2)
        H[:, 2] = -back * np.array([-u[1], u[0]])
        return np.asarray(xy) - (self.x[:2] - back * u), H

    def update_position(self, xy, lag_s, v_cam, force=False):
        innov, H = self.position_innovation(xy, lag_s, v_cam)
        return self._update(innov, H, np.eye(2) * self.GPS_SIGMA_M ** 2, self.GATE_2D, force)

    def update_speed(self, v_gps, v_cam, force=False):
        """Doppler: the speed itself, or s times the camera speed without it."""
        H = np.zeros((1, len(self.x)))
        if self.speed_state:
            H[0, self.i["v"]] = 1.0
            innov, var = v_gps - self.x[self.i["v"]], self.DOPPLER_SIGMA ** 2
        else:
            H[0, 3] = v_cam
            innov = v_gps - self.x[3] * v_cam
            var = self.DOPPLER_SIGMA ** 2 + (self.x[3] * self.CAMERA_SPEED_REL * v_cam) ** 2
        return self._update(np.array([innov]), H, np.array([[var]]), self.GATE_1D, force)

    def update_camera_speed(self, v_cam, force=False, hold_scale=False):
        """
        With the speed state: the camera measures v / s, and 0 when stopped.

        Without GPS, s cannot be told apart from v: hold_scale freezes it, so
        the camera moves only the speed, with the last scale. Otherwise every
        speed change the camera sees would leak into s.
        """
        i = self.i
        v, s = self.x[i["v"]], self.x[3]
        H = np.zeros((1, len(self.x)))
        if v_cam == 0:
            H[0, i["v"]] = 1.0
            innov, var = -v, self.STOP_SIGMA ** 2
        else:
            H[0, i["v"]] = 1.0 / s
            H[0, 3] = 0.0 if hold_scale else -v / s ** 2
            innov = v_cam - v / s
            var = (self.CAMERA_SPEED_REL * v_cam) ** 2 + self.CAMERA_SPEED_FLOOR ** 2
        return self._update(np.array([innov]), H, np.array([[var]]), self.GATE_1D, force,
                            keep=(3,) if hold_scale else ())

    def update_course(self, course, turned_since, chord_m, force=False):
        """
        Course of the chord between two fixes. It is the heading at the middle
        of the interval, so the yaw turned since then is added back.
        """
        H = np.zeros((1, len(self.x)))
        H[0, 2] = 1.0
        var = 2 * (self.COURSE_SIGMA_M / chord_m) ** 2
        innov = wrap(course - (self.x[2] - turned_since))
        return self._update(np.array([innov]), H, np.array([[var]]), self.GATE_1D, force)


class GpsCameraFusion:
    """
    Feeds a PlanarEKF in time order: frames predict, fixes correct. Starts
    once the GPS has seen the car move INIT_METRES, which gives a first
    heading.

    gyro uses the gyroscope's yaw when a frame has it; speed_state and bias
    add those states. All three off is the first version of the filter. The
    bias is off by default: estimated from the course it did not lower the
    error of 10 to 120 s outages on the car recordings.
    visual_only uses the GPS only to start (where, and which way): after that
    the camera, and the gyroscope if on, carry the track with the depth
    model's own scale: the visual-only baseline the full filter is compared
    against.
    """

    MIN_SPEED = 2.0         # m/s: below this the camera/Doppler ratio means nothing
    MIN_CHORD_M = 3.0       # shortest chord whose course is used
    INIT_SPEED = 3.0        # m/s
    INIT_METRES = 20.0
    HISTORY_S = 30.0
    MAX_REJECTED = 2        # a GPS jump is taken as such at most twice in a row
    # ...unless it is farther than the car could have driven, at ~144 km/h,
    # since the last fix that agreed with the prediction: then it is never
    # forced, however long it lasts.
    V_MAX = 40.0            # m/s
    CAMERA_PERIOD_S = 1.0   # the camera speed as a measurement, at the rate its noise was measured
    # Without a fix for this long, s is held: only the GPS sees the scale
    # change. Between the camera and Doppler it changes ~50 % within 5 s and
    # then hardly more, so what the camera sees next is the car's speed.
    SCALE_HOLD_S = 2.0

    def __init__(self, use_course=True, gyro=True, speed_state=True, bias=False, visual_only=False):
        self.use_course = use_course
        self.gyro, self.speed_state, self.bias = gyro, speed_state, bias
        self.visual_only = visual_only
        self.ekf = None
        self.t_ns = None
        self.v_cam = 0.0
        self.yaw_total = 0.0
        self._yaw_hist = collections.deque()      # (t_ns, accumulated yaw)
        self._prev_fix = None                     # (t_ns, xy)
        self._anchor = None
        self._next_camera = None
        self._last_fix = None                     # t_ns of the last fix delivered
        self._last_agreed = None                  # t_ns of the last fix that agreed with the prediction
        self._rejected = {"pos": 0, "speed": 0, "course": 0}
        self.n_gyro = self.n_camera_yaw = 0       # frames turned by each sensor
        self.log = []                             # (t_ns, kind, status, nis, innovation)

    @property
    def ready(self):
        return self.ekf is not None

    def on_frame(self, t_ns, yaw, step, has_pose=True, has_scale=True, gyro_yaw=None):
        dt = 0.0 if self.t_ns is None else (t_ns - self.t_ns) / 1e9
        if dt > 0:
            self.v_cam = step / dt
        self.t_ns = t_ns
        from_gyro = self.gyro and gyro_yaw is not None and np.isfinite(gyro_yaw)
        if from_gyro:
            yaw, yaw_var = gyro_yaw, PlanarEKF.GYRO_RW ** 2 * dt
        else:
            yaw_var = PlanarEKF.YAW_RW ** 2 * dt + (0.0 if has_pose else PlanarEKF.MISSED_YAW ** 2)
        self.yaw_total += yaw
        self._yaw_hist.append((t_ns, self.yaw_total))
        while t_ns - self._yaw_hist[0][0] > self.HISTORY_S * 1e9:
            self._yaw_hist.popleft()
        if self.ekf is None:
            return
        if from_gyro:
            self.n_gyro += 1
        else:
            self.n_camera_yaw += 1
        hold = self.visual_only or t_ns - self._last_fix >= self.SCALE_HOLD_S * 1e9
        self.ekf.predict(dt, yaw, yaw_var, from_gyro, step, hold_scale=hold)
        if self.speed_state and has_scale and t_ns >= self._next_camera:
            self._next_camera = t_ns + int(self.CAMERA_PERIOD_S * 1e9)
            self._camera_speed(hold)

    def _camera_speed(self, hold_scale):
        """
        The camera speed as a measurement, never forced: when it disagrees the
        filter keeps the speed it had. A speed that implies s out of range is
        not even tried (the depth model failing, not a scale).
        """
        v = self.ekf.x[self.ekf.i["v"]]
        kind = "stop" if self.v_cam == 0 else "camera"
        if kind == "camera" and v >= self.MIN_SPEED:
            lo, hi = PlanarEKF.S_RANGE
            if not lo <= v / self.v_cam <= hi:
                self.log.append((self.t_ns, kind, "out_of_range", np.nan, None))
                return
        status, nis = self.ekf.update_camera_speed(self.v_cam, hold_scale=hold_scale)
        self.log.append((self.t_ns, kind, status, nis, self.ekf.innovation))

    def on_fix(self, t_ns, xy, speed):
        xy = np.asarray(xy, float)[:2]
        prev, self._prev_fix = self._prev_fix, (t_ns, xy)
        self._last_fix = t_ns
        if self.t_ns is None:
            return                  # no frame yet: nothing to correct
        if self.ekf is None:
            self._try_start(t_ns, xy, speed)
            return
        if self.visual_only:
            return

        lag = (self.t_ns - t_ns) / 1e9
        jump = np.linalg.norm(self.ekf.position_innovation(xy, lag, self.v_cam)[0])
        agrees = jump <= 3 * PlanarEKF.GPS_SIGMA_M
        reach = self.V_MAX * (self.t_ns - self._last_agreed) / 1e9 + 3 * PlanarEKF.GPS_SIGMA_M
        status = self._apply("pos", self.ekf.update_position, xy, lag, self.v_cam,
                             may_force=jump <= reach)
        # A fix taken with a large innovation (after an outage) may be the wrong
        # one, so only a fix that agreed with the prediction moves the reach.
        if status != "rejected" and agrees:
            self._last_agreed = self.t_ns

        if self.speed_state:
            self._apply("speed", self.ekf.update_speed, speed, self.v_cam)
        elif speed >= self.MIN_SPEED and self.v_cam > 0:
            # The camera speed is a rolling median, ~1 s behind Doppler; s
            # absorbs that delay too, which is what the prediction needs.
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

    def _apply(self, kind, update, *args, may_force=True):
        """Run one update, forcing it after MAX_REJECTED rejections in a row. Returns its status."""
        force = may_force and self._rejected[kind] >= self.MAX_REJECTED
        status, nis = update(*args, force=force)
        self._rejected[kind] = self._rejected[kind] + 1 if status == "rejected" else 0
        self.log.append((self.t_ns, kind, status, nis, self.ekf.innovation))
        return status

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
        if self.visual_only:
            speed = self.v_cam
        self.ekf = PlanarEKF(xy[0], xy[1], wrap(heading),
                             speed=speed if self.speed_state else None, bias=self.gyro and self.bias)
        self._next_camera = self._last_agreed = self.t_ns

    def _yaw_at(self, t_ns):
        ts, ys = zip(*self._yaw_hist)
        return float(np.interp(t_ns, ts, ys))


def session_inputs(R, tv, steps):
    """frame_inputs for every frame of a session, as (yaw, step, has_pose, has_scale) arrays."""
    rows = [frame_inputs(R[k], tv[k], steps[k]) for k in range(len(R))]
    return tuple(np.array(c) for c in zip(*rows))


def gyro_inputs(t_imu, w, acc, t_frames):
    """
    The gyroscope's yaw for every frame of a session, NaN where it does not
    cover the frame. Samples are fed up to each frame's timestamp, as they
    would have arrived live.
    """
    gy = GyroYaw()
    out = np.full(len(t_frames), np.nan)
    j = 0
    for k, t in enumerate(t_frames):
        while j < len(t_imu) and t_imu[j] <= t:
            gy.add(int(t_imu[j]), w[j], acc[j])
            j += 1
        yaw = gy.take(int(t))
        if yaw is not None:
            out[k] = yaw
    return out


# Columns of the state returned by run_fusion; v and b are NaN without them.
STATE = ("x", "y", "th", "s", "v", "b")


def feed_frame(fus, k, t_ns, inputs, gyro_yaw, fixes, j, outages=()):
    """
    Frame k into the fusion, then the fixes up to its timestamp except those
    inside an outage. Returns the index of the next fix.
    """
    yaw, step, has_pose, has_scale = inputs
    fus.on_frame(int(t_ns[k]), yaw[k], step[k], has_pose[k], has_scale[k],
                 None if gyro_yaw is None else gyro_yaw[k])
    # A fix is available from the first frame after its timestamp, as in
    # GpsBuffer.latest_before.
    while j < len(fixes) and fixes[j][0] <= t_ns[k]:
        tf = fixes[j][0]
        if not any(a <= tf <= b for a, b in outages):
            fus.on_fix(*fixes[j])
        j += 1
    return j


def run_fusion(t_ns, inputs, fixes, outages=(), use_course=True, gyro_yaw=None, **modes):
    """
    Run the fusion over a processed session.

    inputs comes from session_inputs and gyro_yaw from gyro_inputs (None: no
    gyroscope). fixes holds (t_ns, xy, speed) in time order; a fix inside any
    outage (t_from_ns, t_to_ns) is not delivered. use_course=False leaves out
    the course between fixes, to measure what it adds; modes are those of
    GpsCameraFusion. Returns the fusion object, the state (STATE columns)
    after every frame and its standard deviations, NaN before the start.
    """
    if gyro_yaw is None:
        modes["gyro"] = False
    fus = GpsCameraFusion(use_course, **modes)
    out = np.full((len(t_ns), len(STATE)), np.nan)
    sig = np.full((len(t_ns), len(STATE)), np.nan)
    j = 0
    for k in range(len(t_ns)):
        j = feed_frame(fus, k, t_ns, inputs, gyro_yaw, fixes, j, outages)
        if fus.ready:
            cols = [STATE.index(n) for n in fus.ekf.i]
            out[k, cols] = fus.ekf.x
            sig[k, cols] = np.sqrt(np.diag(fus.ekf.P))
    return fus, out, sig
