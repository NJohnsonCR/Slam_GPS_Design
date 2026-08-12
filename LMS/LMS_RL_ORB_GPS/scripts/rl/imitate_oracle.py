"""
PASO 3 (corregido) - Imitación del oráculo con tasas de sensor realistas.

Historial de correcciones (importante para el informe del TFG):
  v1: la pose arrancaba con rotación identidad, ignorando que el marco de la
      cámara está girado ~124° respecto del mapa -> la VO avanzaba mal.
  v2: asumía un fix de GPS por frame, régimen donde no hay nada que adaptar.
  v3: las features se calculaban sobre la trayectoria del ORÁCULO pero se
      aplicaban sobre la trayectoria de la POLÍTICA, que se desvía. Desajuste
      de distribución: la política veía en producción estados que nunca vio
      entrenando.

Esta versión resuelve las tres. La clave de la tercera es que ahora existe un
único bucle `run_episode` que calcula las features EN LÍNEA sobre la trayectoria
que realmente se está recorriendo, y consulta a la política en cada fix. Es
también como debe funcionar en tiempo real.

Para cubrir los estados que la política va a visitar, los datos de entrenamiento
se recolectan recorriendo cada secuencia con VARIAS políticas de comportamiento
(pesos constantes distintos y el oráculo), etiquetando cada estado con el peso
óptimo calculado en ESE estado.

Uso:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.rl.imitate_oracle
"""

import os
import sys
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from realistic_rates import (
    load_caches, degrade_gps, estimate_initial_rotation, gps_hold_trajectory, ate,
)

SPLIT = {
    'train': ['2011_09_26_drive_0009_sync', '2011_09_26_drive_0018_sync',
              '2011_10_03_drive_0042_sync'],
    'test':  ['2011_09_26_drive_0001_sync', '2011_09_26_drive_0013_sync',
              '2011_09_29_drive_0071_sync'],
}

GPS_PERIOD = 10
TRAIN_SIGMAS = [3.0, 5.0, 10.0]
EVAL_SIGMAS = [3.0, 5.0, 10.0]
N_SEEDS_TRAIN = 8
N_SEEDS_EVAL = 6
W_GRID = np.linspace(0.0, 1.0, 41)
BEHAVIOR_POLICIES = [0.2, 0.4, 0.6, 0.8, 'oracle']

FEATURES = ['dist_vo', 'rot_vo', 'inlier_mean', 'matches_mean', 'innovacion', 'gps_step']


def rot_angle(R):
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def run_episode(cache, gps, period, R0, weight_fn):
    """
    Bucle único de fusión. Calcula las features EN LÍNEA sobre la trayectoria
    que se está recorriendo y consulta la política en cada fix.

    weight_fn: 'oracle', un escalar, o una función feat_vector -> w.

    Devuelve (trayectoria, features por fix, peso óptimo por fix, peso usado).
    El peso óptimo se registra siempre, sea cual sea la política: es la etiqueta
    de entrenamiento para ese estado.
    """
    R_vo, t_vo = cache['R_vo'], cache['t_vo']
    scales, gt = cache['step_scales'], cache['gt']
    inl = cache['n_inliers'] / np.maximum(cache['n_matches'], 1)
    n = len(R_vo)

    pose = np.eye(4)
    pose[:3, :3] = R0
    pose[:3, 3] = gt[0]

    positions = [gt[0].copy()]
    feats, w_star, w_used = [], [], []
    seg_dist = seg_rot = 0.0
    seg_inl, seg_mat = [], []
    last_fix_gps = gps[0]

    for i in range(n):
        rel = np.eye(4)
        rel[:3, :3] = R_vo[i]
        step = t_vo[i] * scales[i]
        rel[:3, 3] = step
        pose = pose @ rel

        seg_dist += float(np.linalg.norm(step))
        seg_rot += rot_angle(R_vo[i])
        seg_inl.append(inl[i])
        seg_mat.append(cache['n_matches'][i])

        if (i + 1) % period == 0:
            a, b, g = pose[:3, 3], gps[i + 1], gt[i + 1]
            diff = a - b
            den = float(diff @ diff)
            ws = 0.5 if den < 1e-12 else float(np.clip(-(diff @ (b - g)) / den, 0.0, 1.0))

            f = np.array([
                seg_dist,
                seg_rot,
                float(np.mean(seg_inl)),
                float(np.mean(seg_mat)) / 1000.0,
                float(np.linalg.norm(diff)),
                float(np.linalg.norm(gps[i + 1] - last_fix_gps)),
            ])

            if isinstance(weight_fn, str):
                w = ws
            elif np.isscalar(weight_fn):
                w = float(weight_fn)
            else:
                w = float(weight_fn(f))

            pose[:3, 3] = w * a + (1.0 - w) * b
            feats.append(f); w_star.append(ws); w_used.append(w)

            last_fix_gps = gps[i + 1]
            seg_dist = seg_rot = 0.0
            seg_inl, seg_mat = [], []

        positions.append(pose[:3, 3].copy())

    return np.array(positions), np.array(feats), np.array(w_star), np.array(w_used)


def main():
    ap = argparse.ArgumentParser(description="Paso 3 corregido: imitación del oráculo a 1 Hz")
    ap.add_argument('--cache-dir', default='resultados/rl_cache')
    ap.add_argument('--out-dir', default='resultados/rl_cache')
    ap.add_argument('--period', type=int, default=GPS_PERIOD)
    args = ap.parse_args()

    caches = {c['name']: c for c in load_caches(args.cache_dir)}
    period = args.period

    print("=" * 100)
    print(f"PASO 3 CORREGIDO — imitación del oráculo   (GPS cada {period} frames = {10/period:.1f} Hz)")
    print("=" * 100)
    print("  Features calculadas EN LÍNEA sobre la propia trayectoria de la política.")
    print(f"  Datos recolectados con {len(BEHAVIOR_POLICIES)} políticas de comportamiento distintas.\n")

    # ---------- recolección de datos ----------
    Xs, ys = [], []
    for nm in SPLIT['train']:
        c = caches[nm]
        for sigma in TRAIN_SIGMAS:
            for seed in range(N_SEEDS_TRAIN):
                rng = np.random.default_rng(abs(hash((nm, sigma, seed))) % (2**32))
                gps = degrade_gps(c['gt'], sigma, rng)
                R0 = estimate_initial_rotation(c['R_vo'], c['t_vo'], c['step_scales'], gps)
                for beh in BEHAVIOR_POLICIES:
                    _, X, ws, _ = run_episode(c, gps, period, R0, beh)
                    if len(ws) >= 3:
                        Xs.append(X); ys.append(ws)

    X = np.vstack(Xs); y = np.concatenate(ys)
    print(f"Eventos de fix recolectados: {len(y)}\n")

    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    A = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)

    print("Fórmula ajustada (features normalizadas):")
    for f, t in zip(FEATURES, theta[:-1]):
        print(f"   {f:<14} {t:+.4f}")
    print(f"   {'(constante)':<14} {theta[-1]:+.4f}")

    pred = np.clip(A @ theta, 0, 1)
    print(f"\n   R² contra w* crudo: {1 - np.var(y - pred)/np.var(y):.3f}")
    print(f"   w* real    — media {y.mean():.3f}  desv {y.std():.3f}")
    print(f"   w predicho — media {pred.mean():.3f}  desv {pred.std():.3f}\n")

    def policy(f):
        return float(np.clip(np.append((f - mu) / sd, 1.0) @ theta, 0.0, 1.0))

    # ---------- mejor peso constante, elegido SOBRE ENTRENAMIENTO ----------
    errs = np.zeros(len(W_GRID))
    for nm in SPLIT['train']:
        c = caches[nm]
        for sigma in TRAIN_SIGMAS:
            rng = np.random.default_rng(abs(hash((nm, sigma, 0))) % (2**32))
            gps = degrade_gps(c['gt'], sigma, rng)
            R0 = estimate_initial_rotation(c['R_vo'], c['t_vo'], c['step_scales'], gps)
            for j, w in enumerate(W_GRID):
                errs[j] += ate(run_episode(c, gps, period, R0, float(w))[0], c['gt'])
    w_fixed = float(W_GRID[int(np.argmin(errs))])
    print(f"Mejor peso CONSTANTE según entrenamiento: w = {w_fixed:.3f}\n")

    # ---------- evaluación ----------
    print("=" * 100)
    print("EVALUACIÓN EN SECUENCIAS NUNCA VISTAS")
    print("=" * 100)
    print(f"{'σ GPS':>7}{'GPS solo':>11}{'w constante':>13}{'IMITACIÓN':>12}{'Oráculo':>10}"
          f"{'margen capturado':>19}{'vs w cte':>11}")
    print("-" * 100)

    rows = []
    for sigma in EVAL_SIGMAS:
        acc = {'gps': [], 'fixed': [], 'imit': [], 'oracle': []}
        for nm in SPLIT['test']:
            c = caches[nm]
            for seed in range(N_SEEDS_EVAL):
                rng = np.random.default_rng(abs(hash((nm, sigma, 'ev', seed))) % (2**32))
                gps = degrade_gps(c['gt'], sigma, rng)
                R0 = estimate_initial_rotation(c['R_vo'], c['t_vo'], c['step_scales'], gps)
                acc['gps'].append(ate(gps_hold_trajectory(gps, c['gt'], period, len(c['R_vo'])), c['gt']))
                acc['fixed'].append(ate(run_episode(c, gps, period, R0, w_fixed)[0], c['gt']))
                acc['imit'].append(ate(run_episode(c, gps, period, R0, policy)[0], c['gt']))
                acc['oracle'].append(ate(run_episode(c, gps, period, R0, 'oracle')[0], c['gt']))

        r = {k: float(np.mean(v)) for k, v in acc.items()}
        r['sigma'] = sigma
        span = r['fixed'] - r['oracle']
        r['captured'] = 100 * (r['fixed'] - r['imit']) / span if span > 1e-9 else 0.0
        r['gain'] = 100 * (r['fixed'] - r['imit']) / r['fixed'] if r['fixed'] > 0 else 0.0
        rows.append(r)
        print(f"{sigma:>6.0f}m{r['gps']:>10.2f}m{r['fixed']:>12.2f}m{r['imit']:>11.2f}m"
              f"{r['oracle']:>9.2f}m{r['captured']:>18.1f}%{r['gain']:>10.1f}%")

    print("-" * 100)
    print("'margen capturado' = fracción de la distancia entre el peso constante y el")
    print("techo del oráculo que cierra la fórmula.  'vs w cte' = mejora directa.")

    os.makedirs(args.out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    s = [r['sigma'] for r in rows]
    axes[0].plot(s, [r['gps'] for r in rows], 'o-', label='GPS solo (1 Hz)')
    axes[0].plot(s, [r['fixed'] for r in rows], 's-', label=f'Peso constante (w={w_fixed:.2f})')
    axes[0].plot(s, [r['imit'] for r in rows], '^-', lw=2.5, color='darkorange', label='Imitación del oráculo')
    axes[0].plot(s, [r['oracle'] for r in rows], 'd--', color='crimson', label='Oráculo (techo)')
    axes[0].set_xlabel('Ruido GPS σ (m)'); axes[0].set_ylabel('ATE RMSE (m)')
    axes[0].set_title('Secuencias de prueba, GPS a 1 Hz'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].bar([f"σ={int(r['sigma'])}" for r in rows], [r['captured'] for r in rows],
                color=['seagreen' if r['captured'] > 0 else 'indianred' for r in rows])
    axes[1].axhline(0, c='k', lw=1)
    axes[1].axhline(36, ls='--', c='gray', label='parte aprendible (~36%)')
    axes[1].set_ylabel('% del margen capturado')
    axes[1].set_title('Cuánto del techo alcanza la fórmula'); axes[1].legend(); axes[1].grid(alpha=0.3, axis='y')

    fig.suptitle('Paso 3 corregido — features en línea, GPS a 1 Hz')
    fig.tight_layout()
    png = os.path.join(args.out_dir, 'imitation_results_fixed.png')
    fig.savefig(png, dpi=130)
    print(f"\nGráfico guardado: {png}")

    np.savez(os.path.join(args.out_dir, 'imitation_policy_fixed.npz'),
             theta=theta, mu=mu, sd=sd, features=FEATURES, w_fixed=w_fixed, period=period)
    return 0


if __name__ == '__main__':
    sys.exit(main())
