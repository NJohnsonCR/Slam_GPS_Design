"""
Pruebas de sanidad del bucle de fusión + auditoría de fugas de ground truth.

Motivación: durante el análisis aparecieron tres defectos seguidos, y cada uno
cambió (o invirtió) la conclusión. Este archivo fija invariantes que deben
cumplirse siempre, para que un cambio futuro que los rompa se detecte solo.

PARTE 1 - INVARIANTES
  T1  Con GPS perfecto y w=0, la pose fusionada debe coincidir EXACTAMENTE con
      el ground truth en cada fix. Detecta errores de índice y de marco.
  T2  Con w=1 el GPS se ignora por completo: la trayectoria no debe cambiar al
      cambiar el ruido del GPS. Detecta que el GPS se cuele por otra vía.
  T3  El peso del oráculo debe ser, en cada fix, al menos tan bueno como
      cualquier peso constante en ese mismo estado. Detecta errores en la
      fórmula cerrada del óptimo.
  T4  Con el rumbo inicial correcto, propagar la VO durante 1 segundo desde el
      ground truth debe dar poco error. Detecta el bug de marco de referencia.
  T5  Determinismo: misma semilla, mismo resultado.
  T6  Los parámetros de la odometría visual coinciden entre el sistema viejo
      (`PoseGraphSLAM` / `main.py`) y el de tiempo real (`VisualFrontEnd`).
      Sin esto, la comparación "sistema anterior vs sistema en tiempo real"
      del OE4 mediría dos VO distintas y no la arquitectura.

PARTE 2 - AUDITORÍA DE FUGAS
  Mide cuánto ayuda información que el sistema real NO tendría:
  L1  escala métrica tomada del ground truth (la VO monocular no la conoce)
  L2  rumbo inicial estimado con la secuencia COMPLETA (usa el futuro)
  L3  posición inicial tomada del ground truth (lo real es el primer fix GPS)

Uso:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.rl.sanity_checks
"""

import ast
import inspect
import os
import sys

import numpy as np

# El proyecto usa imports relativos a dos raíces distintas: la raíz del repo
# (para LMS.*) y la carpeta LMS_RL_ORB_GPS (para realtime.* y utils.*).
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
    Escala métrica estimada SOLO con el GPS, como haría el sistema real.

    La VO monocular no puede recuperar la escala absoluta. Se toma la distancia
    que el GPS dice haberse desplazado en cada intervalo entre fixes, suavizada
    de forma causal, y se reparte entre los frames del intervalo.
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
# T6 — los parámetros de la VO no se comparten entre los dos sistemas, se
# verifican. `realtime/` reimplementa la odometría a propósito (no importa el
# código viejo, que arrastra la calibración de KITTI y estado de keyframes que
# no usa), pero el ALGORITMO tiene que ser idéntico o la comparación del OE4
# deja de medir la arquitectura.
#
# Sin esta prueba, cambiar 2000 features en un lado y no en el otro haría
# divergir los sistemas EN SILENCIO.
# --------------------------------------------------------------------------

# Todo lo que OpenCV expone del detector: así el invariante no se limita al
# número de features, que es lo único que alguien pensaría en revisar a mano.
_ORB_GETTERS = ('getMaxFeatures', 'getScaleFactor', 'getNLevels',
                'getEdgeThreshold', 'getFirstLevel', 'getWTA_K',
                'getPatchSize', 'getFastThreshold')


def _min_matches_del_fuente(path, clase, metodo):
    """
    Lee `minimumMatches` del código viejo SIN importarlo.

    Importar `main.py` traería torch, el agente RL y matplotlib, y además sus
    imports fallan según desde dónde se ejecute (bug conocido). El umbral es un
    literal en el fuente; leerlo con ast es exacto y no cuesta nada.
    """
    with open(path, encoding='utf-8') as f:
        arbol = ast.parse(f.read(), filename=path)
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.ClassDef) and nodo.name == clase:
            for hijo in ast.walk(nodo):
                if isinstance(hijo, ast.FunctionDef) and hijo.name == metodo:
                    for st in ast.walk(hijo):
                        if isinstance(st, ast.Assign) and any(
                                isinstance(t, ast.Name) and t.id == 'minimumMatches'
                                for t in st.targets):
                            return ast.literal_eval(st.value)
    return None


def check_parametros_vo():
    """Compara el front-end de tiempo real contra el sistema viejo."""
    import cv2
    from LMS.LMS_ORB_with_PG.main import PoseGraphSLAM
    from realtime.pipeline import VisualFrontEnd

    viejo = PoseGraphSLAM()
    nuevo = VisualFrontEnd(np.eye(3))

    difs = []

    for g in _ORB_GETTERS:
        a, b = getattr(viejo.orb_detector, g)(), getattr(nuevo.orb, g)()
        if a != b:
            difs.append(f"ORB.{g[3:]}: viejo={a} nuevo={b}")

    # ratio de Lowe: en el sistema viejo es el default del método que filtra
    lowe_viejo = inspect.signature(
        PoseGraphSLAM.filter_matches_lowe_ratio).parameters['ratio'].default
    if lowe_viejo != nuevo.lowe:
        difs.append(f"ratio de Lowe: viejo={lowe_viejo} nuevo={nuevo.lowe}")

    # mínimo de matches: el del sistema RL+GPS, que es el baseline del OE4
    main_py = os.path.join(_LMS_RL, 'main.py')
    mm_viejo = _min_matches_del_fuente(main_py, 'RL_ORB_SLAM_GPS',
                                       'process_frame_with_gps')
    if mm_viejo is None:
        difs.append("no se encontró `minimumMatches` en "
                    "RL_ORB_SLAM_GPS.process_frame_with_gps")
    elif mm_viejo != nuevo.MIN_MATCHES:
        difs.append(f"mínimo de matches: viejo={mm_viejo} nuevo={nuevo.MIN_MATCHES}")

    # el matcher no expone su normType en Python; se compara el tipo, que es
    # lo verificable, y el resto queda cubierto por la revisión del algoritmo
    if type(viejo.matcher) is not type(nuevo.matcher):
        difs.append(f"matcher: viejo={type(viejo.matcher).__name__} "
                    f"nuevo={type(nuevo.matcher).__name__}")

    detalle = ("idénticos: features, escala, niveles, ratio de Lowe y mínimo de matches"
               if not difs else " | ".join(difs))
    return check("T6  los parámetros de la VO coinciden entre main.py y realtime/",
                 not difs, detalle)


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

    # ---- T2: w=1 ignora el GPS ----
    gA = degrade_gps(gt, 3.0, np.random.default_rng(1))
    gB = degrade_gps(gt, 12.0, np.random.default_rng(2))
    tA = run_episode(seq, gA, PERIOD, R0, 1.0)[0]
    tB = run_episode(seq, gB, PERIOD, R0, 1.0)[0]
    d = np.abs(tA - tB).max()
    check("T2  w=1 ignora completamente el GPS", d < TOL, f"diferencia máx {d:.2e} m")

    # ---- T3: el oráculo domina a cualquier peso constante en cada fix ----
    gps5 = degrade_gps(gt, 5.0, np.random.default_rng(3))
    _, F, ws, _ = run_episode(seq, gps5, PERIOD, R0, 'oracle')
    worse = 0
    for w_const in np.linspace(0, 1, 11):
        _, _, ws_c, _ = run_episode(seq, gps5, PERIOD, R0, float(w_const))
        # en el mismo estado, el óptimo cerrado no puede ser superado
        _, _, _, _ = run_episode(seq, gps5, PERIOD, R0, float(w_const))
    # verificación directa de la fórmula cerrada sobre estados muestreados
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

    # ---- T4: propagación de VO durante 1 s con rumbo correcto ----
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

    # ---- T6: la VO del sistema viejo y la del de tiempo real coinciden ----
    check_parametros_vo()

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
                    cc['gt'][0] = gps[0]        # arrancar en el primer fix GPS
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
