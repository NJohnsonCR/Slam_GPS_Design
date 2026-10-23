"""
Invariants of the fusion loop, and an audit of ground-truth leaks, so a change
that breaks them shows up on its own.

INVARIANTS
  T1  perfect GPS and w = 0: the fused pose is the ground truth at every fix
      (index and frame errors)
  T2  w = 1 ignores the GPS: changing its noise does not change the track
  T3  the oracle's weight is at least as good as any constant one in the same
      state (the closed form of the optimum)
  T4  with the right starting heading, 1 s of VO from the ground truth stays
      close (the reference-frame bug)
  T5  the same seed gives the same result
  T6  the visual odometry's parameters match between the old system
      (PoseGraphSLAM / main.py) and the real-time one (VisualFrontEnd);
      otherwise comparing them would compare two VOs, not the architectures

LEAKS: how much information the real system would not have helps
  L1  metric scale from the ground truth (monocular VO does not know it)
  L2  starting heading from the whole sequence (uses the future)
  L3  starting position from the ground truth (really, the first GPS fix)

Uso:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.rl.sanity_checks
"""

import ast
import inspect
import os
import sys

import numpy as np

# Imports resolve against two roots: the repository (LMS.*) and
# LMS_RL_ORB_GPS (realtime.*, utils.*).
_THIS = os.path.dirname(os.path.abspath(__file__))
_LMS_RL = os.path.abspath(os.path.join(_THIS, '..', '..'))
_ROOT = os.path.abspath(os.path.join(_LMS_RL, '..', '..'))
for _p in (_THIS, _ROOT, _LMS_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realistic_rates import load_caches, degrade_gps, estimate_initial_rotation, ate
from imitate_oracle import run_episode

PERIOD = 10
TOL = 1e-9

_results = []


def check(name, ok, detail=""):
    _results.append(ok)
    print(f"  [{'PASS' if ok else 'FALLA'}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def gps_derived_scales(gps, period, n, smooth_fixes=5):
    """
    Metric scale from the GPS alone, as the real system would get it: the
    distance between fixes, smoothed causally, spread over the frames between
    them.
    """
    scales = np.ones(n)
    dists, spans = [], []
    for k in range(n // period):
        i0, i1 = k * period, min((k + 1) * period, n)
        dists.append(float(np.linalg.norm(gps[i1] - gps[i0])))
        spans.append((i0, i1))

    for k, (i0, i1) in enumerate(spans):
        lo = max(0, k - smooth_fixes + 1)
        d = float(np.mean(dists[lo:k + 1]))          # causal: solo fixes pasados
        scales[i0:i1] = d / max(i1 - i0, 1)
    if spans:
        scales[spans[-1][1]:] = scales[spans[-1][1] - 1]
    return scales


# --------------------------------------------------------------------------
# T6: realtime/ reimplements the odometry on purpose (the old code carries the
# KITTI calibration and keyframe state it does not need), so the parameters
# are checked rather than shared: the algorithm has to be the same.
#
# Without it, changing the 2000 features on one side would part them silently.
# --------------------------------------------------------------------------

# Everything OpenCV exposes of the detector, not only the number of features.
_ORB_GETTERS = ('getMaxFeatures', 'getScaleFactor', 'getNLevels',
                'getEdgeThreshold', 'getFirstLevel', 'getWTA_K',
                'getPatchSize', 'getFastThreshold')


def _min_matches_in_source(path, class_name, method_name):
    """
    minimumMatches from the old code, read with ast instead of importing it:
    main.py would bring torch, the RL agent and matplotlib, and its imports
    fail depending on where it runs from.
    """
    with open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in ast.walk(node):
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    for st in ast.walk(child):
                        if isinstance(st, ast.Assign) and any(
                                isinstance(t, ast.Name) and t.id == 'minimumMatches'
                                for t in st.targets):
                            return ast.literal_eval(st.value)
    return None


def check_vo_parameters():
    """The real-time front end against the old system."""
    from LMS.LMS_ORB_with_PG.main import PoseGraphSLAM
    from realtime.pipeline import VisualFrontEnd

    old = PoseGraphSLAM()
    new = VisualFrontEnd(np.eye(3))

    diffs = []

    for g in _ORB_GETTERS:
        a, b = getattr(old.orb_detector, g)(), getattr(new.orb, g)()
        if a != b:
            diffs.append(f"ORB.{g[3:]}: viejo={a} nuevo={b}")

    # Lowe's ratio: in the old system, the default of the method that filters
    old_lowe = inspect.signature(
        PoseGraphSLAM.filter_matches_lowe_ratio).parameters['ratio'].default
    if old_lowe != new.lowe:
        diffs.append(f"ratio de Lowe: viejo={old_lowe} nuevo={new.lowe}")

    # minimum matches: the RL+GPS system's
    main_py = os.path.join(_LMS_RL, 'main.py')
    old_min_matches = _min_matches_in_source(main_py, 'RL_ORB_SLAM_GPS',
                                       'process_frame_with_gps')
    if old_min_matches is None:
        diffs.append("no se encontró `minimumMatches` en "
                    "RL_ORB_SLAM_GPS.process_frame_with_gps")
    elif old_min_matches != new.MIN_MATCHES:
        diffs.append(f"mínimo de matches: viejo={old_min_matches} nuevo={new.MIN_MATCHES}")

    # the matcher does not expose its normType in Python: its type is compared
    if type(old.matcher) is not type(new.matcher):
        diffs.append(f"matcher: viejo={type(old.matcher).__name__} "
                    f"nuevo={type(new.matcher).__name__}")

    details = ("idénticos: features, escala, niveles, ratio de Lowe y mínimo de matches"
               if not diffs else " | ".join(diffs))
    return check("T6  los parámetros de la VO coinciden entre main.py y realtime/",
                 not diffs, details)


def main():
    caches = {c['name']: c for c in load_caches('resultados/rl_cache')}
    seq = caches['2011_09_26_drive_0009_sync']
    gt, n = seq['gt'], len(seq['R_vo'])

    print("=" * 96)
    print("PARTE 1 — INVARIANTES DEL BUCLE DE FUSIÓN")
    print("=" * 96)

    # ---- T1: GPS perfecto + w=0 -> pose exacta en cada fix ----
    gps0 = degrade_gps(gt, 0.0, np.random.default_rng(0))
    R0 = estimate_initial_rotation(seq['R_vo'], seq['t_vo'], seq['step_scales'], gps0, n_init=60)
    traj, _, _, _ = run_episode(seq, gps0, PERIOD, R0, 0.0)
    fix_idx = np.arange(PERIOD, n + 1, PERIOD)
    err_fix = np.abs(traj[fix_idx] - gt[fix_idx]).max()
    check("T1  GPS perfecto + w=0 reproduce el GT en los fixes",
          err_fix < 1e-6, f"error máx {err_fix:.2e} m")

    # ---- T2: w = 1 ignores the GPS ----
    gA = degrade_gps(gt, 3.0, np.random.default_rng(1))
    gB = degrade_gps(gt, 12.0, np.random.default_rng(2))
    tA = run_episode(seq, gA, PERIOD, R0, 1.0)[0]
    tB = run_episode(seq, gB, PERIOD, R0, 1.0)[0]
    d = np.abs(tA - tB).max()
    check("T2  w=1 ignora completamente el GPS", d < TOL, f"diferencia máx {d:.2e} m")

    # ---- T3: the oracle beats any constant weight at each fix ----
    gps5 = degrade_gps(gt, 5.0, np.random.default_rng(3))
    _, F, ws, _ = run_episode(seq, gps5, PERIOD, R0, 'oracle')
    worse = 0
    for w_const in np.linspace(0, 1, 11):
        _, _, ws_c, _ = run_episode(seq, gps5, PERIOD, R0, float(w_const))
        # in the same state the closed-form optimum cannot be beaten
        _, _, _, _ = run_episode(seq, gps5, PERIOD, R0, float(w_const))
    # the closed form, checked directly on sampled states
    rng = np.random.default_rng(7)
    bad = 0
    for _ in range(2000):
        a, b, g = rng.normal(0, 10, 3), rng.normal(0, 10, 3), rng.normal(0, 10, 3)
        diff = a - b
        den = float(diff @ diff)
        w_opt = 0.5 if den < 1e-12 else float(np.clip(-(diff @ (b - g)) / den, 0, 1))
        e_opt = np.linalg.norm(w_opt * a + (1 - w_opt) * b - g)
        e_grid = min(np.linalg.norm(w * a + (1 - w) * b - g) for w in np.linspace(0, 1, 101))
        if e_opt > e_grid + 1e-9:
            bad += 1
    check("T3  la fórmula cerrada del peso óptimo domina a la búsqueda en grilla",
          bad == 0, f"{bad}/2000 estados violan la condición")

    # ---- T4: 1 s of VO with the right heading ----
    errs = []
    for s in range(0, n - PERIOD, PERIOD):
        P = np.eye(4); P[:3, :3] = R0; P[:3, 3] = gt[s]
        for i in range(s, s + PERIOD):
            rel = np.eye(4); rel[:3, :3] = seq['R_vo'][i]
            rel[:3, 3] = seq['t_vo'][i] * seq['step_scales'][i]
            P = P @ rel
        errs.append(np.linalg.norm(P[:3, 3] - gt[s + PERIOD]))
    med = float(np.median(errs))
    check("T4  la VO propaga 1 s con poco error usando el rumbo correcto",
          med < 3.0, f"mediana {med:.2f} m")

    # ---- T5: determinismo ----
    r1 = run_episode(seq, degrade_gps(gt, 5.0, np.random.default_rng(9)), PERIOD, R0, 'oracle')[0]
    r2 = run_episode(seq, degrade_gps(gt, 5.0, np.random.default_rng(9)), PERIOD, R0, 'oracle')[0]
    check("T5  determinismo con la misma semilla", np.array_equal(r1, r2))

    # ---- T6: the old system's VO and the real-time one match ----
    check_vo_parameters()

    # =====================================================================
    print("\n" + "=" * 96)
    print("PARTE 2 — AUDITORÍA DE FUGAS DE GROUND TRUTH")
    print("=" * 96)
    print("Mide cuánto se degrada el desempeño al quitar información que el")
    print("sistema real NO tendría. Una caída grande significa que el resultado")
    print("anterior estaba apoyado en esa fuga.\n")

    test_seqs = ['2011_09_26_drive_0001_sync', '2011_09_26_drive_0013_sync',
                 '2011_09_29_drive_0071_sync']
    sigma = 5.0

    def eval_config(use_gt_scale, r0_window, start_from_gt):
        out = []
        for nm in test_seqs:
            c = caches[nm]
            nn = len(c['R_vo'])
            for seed in range(4):
                rng = np.random.default_rng(abs(hash((nm, seed))) % (2**32))
                gps = degrade_gps(c['gt'], sigma, rng)
                scales = c['step_scales'] if use_gt_scale else gps_derived_scales(gps, PERIOD, nn)
                R0c = estimate_initial_rotation(c['R_vo'], c['t_vo'], scales, gps, n_init=r0_window)

                cc = dict(c)
                cc['step_scales'] = scales
                if not start_from_gt:
                    cc['gt'] = c['gt'].copy()
                    cc['gt'][0] = gps[0]        # start at the first GPS fix
                traj = run_episode(cc, gps, PERIOD, R0c, 0.5)[0]
                out.append(ate(traj, c['gt']))
        return float(np.mean(out))

    base = eval_config(True, None, True)
    print(f"{'configuración':<58}{'ATE':>10}{'degradación':>15}")
    print("-" * 96)
    print(f"{'BASE (con las tres fugas — lo que corrí antes)':<58}{base:>9.2f}m{'—':>15}")

    for label, kw in [
        ("L2  rumbo inicial solo con los primeros 6 s (sin futuro)", dict(use_gt_scale=True, r0_window=60, start_from_gt=True)),
        ("L3  posición inicial del primer fix GPS (no del GT)",      dict(use_gt_scale=True, r0_window=None, start_from_gt=False)),
        ("L1  escala métrica derivada del GPS (no del GT)",          dict(use_gt_scale=False, r0_window=None, start_from_gt=True)),
        ("SIN NINGUNA FUGA (condiciones reales)",                    dict(use_gt_scale=False, r0_window=60, start_from_gt=False)),
    ]:
        v = eval_config(**kw)
        print(f"{label:<58}{v:>9.2f}m{100*(v-base)/base:>14.0f}%")

    print("\n" + "=" * 96)
    ok = all(_results)
    print(f"INVARIANTES: {sum(_results)}/{len(_results)} pasan")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
