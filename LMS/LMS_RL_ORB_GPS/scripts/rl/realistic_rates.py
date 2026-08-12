"""
PASO 2b - Oráculo bajo tasas de sensor REALISTAS.

El paso 3 dio negativo, pero el experimento estaba mal planteado: se asumía un
fix de GPS en CADA frame. En ese régimen la pose se reancla constantemente, la
odometría visual nunca acumula deriva, y por lo tanto el peso óptimo es casi
constante. No había nada que adaptar por construcción.

En el despliegue real el GPS llega a ~1 Hz y la cámara corre a 10-30 Hz: entre
fix y fix la VO navega sola y acumula deriva. Este script mide si en ESE régimen
aparece una señal consistente y aprendible.

Se responden tres preguntas separadas:

  A) ¿Aparece margen para un peso adaptativo al bajar la tasa de GPS?
  B) ¿La conclusión depende de la calidad de la cámara?
  C) ¿Cuánto del peso óptimo es PREDECIBLE y cuánto es azar del ruido GPS?

(C) es la decisiva. El peso perfecto instantáneo depende del ruido concreto que
le tocó a ese fix, que nadie puede predecir. Se descompone su varianza en:
    - varianza ENTRE eventos de fix (se repite entre semillas -> predecible)
    - varianza DENTRO de un evento (cambia con cada semilla -> azar)
Solo la primera es aprendible. Si es despreciable, ninguna política sirve.

Uso:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.rl.realistic_rates
"""

import os
import sys
import glob
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# KITTI graba a 10 Hz -> el período se expresa en frames.
#   1 frame  = 10 Hz  (lo que asumía el experimento anterior)
#  10 frames = 1 Hz   (GPS realista de teléfono)
#  30 frames = 0.33 Hz (GPS degradado / cañón urbano)
GPS_PERIODS = [1, 5, 10, 20, 30]
SIGMAS = [3.0, 5.0, 10.0]
VO_DEGRADATIONS = [0.0, 0.25, 0.5, 1.0]   # grados extra de deriva de rumbo por frame
N_SEEDS = 6
W_GRID = np.linspace(0.0, 1.0, 21)


def degrade_gps(gt, sigma, rng):
    noisy = gt.copy()
    if sigma > 0:
        noisy[:, :2] += rng.normal(0.0, sigma, size=(len(gt), 2))
    return noisy


def degrade_vo(R_vo, rot_sigma_deg, rng):
    """
    Simula una cámara/algoritmo peor: agrega deriva de rumbo aleatoria por paso.
    Con rot_sigma_deg = 0 se usa la VO real tal cual salió de las imágenes.
    """
    if rot_sigma_deg <= 0:
        return R_vo
    out = np.empty_like(R_vo)
    angs = rng.normal(0.0, np.deg2rad(rot_sigma_deg), size=len(R_vo))
    for i, a in enumerate(angs):
        c, s = np.cos(a), np.sin(a)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        out[i] = Rz @ R_vo[i]
    return out


def estimate_initial_rotation(R_vo, t_vo, scales, ref_traj, n_init=None):
    """
    Estima la rotación mundo <- cámara al arrancar.

    La VO entrega el movimiento en el marco de la cámara, que en KITTI está
    rotado más de 100° respecto del marco UTM. Sin esta alineación la VO avanza
    en direcciones equivocadas y toda la fusión pierde sentido.

    Un sistema real resuelve esto igual: usa los primeros desplazamientos del
    GPS para fijar el rumbo. Por eso `ref_traj` es el GPS (ruidoso, observable)
    y no el ground truth.
    """
    n = len(R_vo) if n_init is None else min(n_init, len(R_vo))
    P = np.eye(4)
    pts = [np.zeros(3)]
    for i in range(n):
        rel = np.eye(4)
        rel[:3, :3] = R_vo[i]
        rel[:3, 3] = t_vo[i] * scales[i]
        P = P @ rel
        pts.append(P[:3, 3].copy())

    est = np.array(pts)
    ref = ref_traj[:len(est)]
    ec = est - est.mean(axis=0)
    rc = ref - ref.mean(axis=0)

    U, S, Vt = np.linalg.svd(ec.T @ rc)
    R0 = Vt.T @ U.T
    if np.linalg.det(R0) < 0:
        Vt[-1, :] *= -1
        R0 = Vt.T @ U.T
    return R0


def replay(R_vo, t_vo, scales, gps, gt, period, w_policy, R0=None):
    """
    Bucle de fusión con GPS intermitente.

    Entre fixes la pose avanza SOLO con la VO (acumulando deriva). En cada fix
    se mezcla la pose acumulada con la medición de GPS usando el peso w.

    w_policy: 'oracle', un escalar, o un vector con un peso por evento de fix.
    R0: rotación inicial mundo <- cámara (ver estimate_initial_rotation).
    Devuelve la trayectoria completa y los pesos usados en cada fix.
    """
    n = len(R_vo)
    pose = np.eye(4)
    if R0 is not None:
        pose[:3, :3] = R0
    pose[:3, 3] = gt[0]

    positions = [gt[0].copy()]
    weights, fix_idx = [], []
    k = 0

    for i in range(n):
        rel = np.eye(4)
        rel[:3, :3] = R_vo[i]
        rel[:3, 3] = t_vo[i] * scales[i]
        pose = pose @ rel

        if (i + 1) % period == 0:            # llegó un fix de GPS
            a, b, g = pose[:3, 3], gps[i + 1], gt[i + 1]
            diff = a - b
            den = float(diff @ diff)

            if isinstance(w_policy, str):    # 'oracle'
                w = 0.5 if den < 1e-12 else float(np.clip(-(diff @ (b - g)) / den, 0.0, 1.0))
            elif np.isscalar(w_policy):
                w = float(w_policy)
            else:
                w = float(w_policy[min(k, len(w_policy) - 1)])

            pose[:3, 3] = w * a + (1.0 - w) * b
            weights.append(w)
            fix_idx.append(i)
            k += 1

        positions.append(pose[:3, 3].copy())

    return np.array(positions), np.array(weights), np.array(fix_idx)


def gps_hold_trajectory(gps, gt, period, n):
    """GPS solo, manteniendo el último fix entre mediciones (1 Hz real)."""
    out = np.empty((n + 1, 3))
    last = gps[0].copy()
    out[0] = last
    for i in range(n):
        if (i + 1) % period == 0:
            last = gps[i + 1].copy()
        out[i + 1] = last
    return out


def ate(traj, gt):
    n = min(len(traj), len(gt))
    return float(np.sqrt(np.mean(np.linalg.norm(traj[:n] - gt[:n], axis=1) ** 2)))


def load_caches(cache_dir, min_steps=50):
    out = []
    for f in sorted(glob.glob(os.path.join(cache_dir, '*_cache.npz'))):
        d = np.load(f, allow_pickle=True)
        if len(d['R_vo']) < min_steps:
            continue
        out.append({'name': str(d['sequence']), 'R_vo': d['R_vo'], 't_vo': d['t_vo'],
                    'step_scales': d['step_scales'], 'gt': d['gt'],
                    'n_matches': d['n_matches'], 'n_inliers': d['n_inliers']})
    return out


def evaluate(caches, period, sigma, vo_deg, n_seeds=N_SEEDS):
    """Devuelve ATE medio de cada estrategia y la matriz de pesos óptimos por semilla."""
    res = {'gps': [], 'vo': [], 'fixed': [], 'oracle': [], 'w_best': []}
    w_by_seq = {}

    for c in caches:
        n = len(c['R_vo'])
        per_seed_w = []
        errs_grid = np.zeros(len(W_GRID))

        for seed in range(n_seeds):
            rng = np.random.default_rng(abs(hash((c['name'], period, sigma, vo_deg, seed))) % (2**32))
            gps = degrade_gps(c['gt'], sigma, rng)
            R = degrade_vo(c['R_vo'], vo_deg, rng)

            # Rumbo inicial estimado con el GPS ruidoso, como haría el sistema real
            R0 = estimate_initial_rotation(R, c['t_vo'], c['step_scales'], gps)

            res['gps'].append(ate(gps_hold_trajectory(gps, c['gt'], period, n), c['gt']))
            res['vo'].append(ate(replay(R, c['t_vo'], c['step_scales'], gps, c['gt'],
                                        period, 1.0, R0)[0], c['gt']))

            traj_o, w_o, _ = replay(R, c['t_vo'], c['step_scales'], gps, c['gt'], period, 'oracle', R0)
            res['oracle'].append(ate(traj_o, c['gt']))
            per_seed_w.append(w_o)

            for j, w in enumerate(W_GRID):
                errs_grid[j] += ate(replay(R, c['t_vo'], c['step_scales'], gps,
                                           c['gt'], period, float(w), R0)[0], c['gt'])

        j = int(np.argmin(errs_grid))
        res['fixed'].append(errs_grid[j] / n_seeds)
        res['w_best'].append(W_GRID[j])

        m = min(len(w) for w in per_seed_w)
        w_by_seq[c['name']] = np.array([w[:m] for w in per_seed_w])   # (semillas, eventos)

    return {k: float(np.mean(v)) for k, v in res.items()}, w_by_seq


def variance_decomposition(w_by_seq):
    """
    Separa la varianza del peso óptimo en parte predecible y parte de azar.

    predecible : varianza ENTRE eventos de fix de la media sobre semillas
    azar       : varianza DENTRO de cada evento, entre semillas
    """
    pred, noise = [], []
    for w in w_by_seq.values():                  # (semillas, eventos)
        if w.shape[1] < 5:
            continue
        pred.append(np.var(w.mean(axis=0)))      # varía entre eventos
        noise.append(np.mean(np.var(w, axis=0))) # varía entre semillas
    if not pred:
        return 0.0, 0.0, 0.0
    p, q = float(np.mean(pred)), float(np.mean(noise))
    return p, q, 100.0 * p / (p + q) if (p + q) > 1e-12 else 0.0


def main():
    ap = argparse.ArgumentParser(description="Oráculo con tasas de sensor realistas")
    ap.add_argument('--cache-dir', default='resultados/rl_cache')
    ap.add_argument('--out-dir', default='resultados/rl_cache')
    args = ap.parse_args()

    caches = load_caches(args.cache_dir)
    print("=" * 104)
    print("ORÁCULO CON TASAS DE SENSOR REALISTAS")
    print("=" * 104)
    print(f"Secuencias: {len(caches)}   |   KITTI graba a 10 Hz")
    print("Período de GPS:  1 frame = 10 Hz  |  10 frames = 1 Hz  |  30 frames = 0.33 Hz\n")

    # ---------------- A) barrido de tasa de GPS ----------------
    print("=" * 104)
    print("A) ¿APARECE MARGEN AL BAJAR LA TASA DE GPS?   (VO real, sin degradar)")
    print("=" * 104)
    print(f"{'período':>9}{'tasa':>9}{'σ':>6}{'GPS solo':>11}{'VO sola':>11}"
          f"{'mejor w fijo':>14}{'(w)':>7}{'oráculo':>11}{'margen':>9}{'predecible':>12}")
    print("-" * 104)

    rows_a = []
    for period in GPS_PERIODS:
        for sigma in SIGMAS:
            r, w_by_seq = evaluate(caches, period, sigma, 0.0)
            _, _, pred_pct = variance_decomposition(w_by_seq)
            margin = 100 * (r['fixed'] - r['oracle']) / r['fixed'] if r['fixed'] > 0 else 0
            rows_a.append({'period': period, 'sigma': sigma, 'margin': margin,
                           'pred': pred_pct, **r})
            print(f"{period:>9}{10/period:>8.1f}Hz{sigma:>6.0f}{r['gps']:>10.2f}m{r['vo']:>10.2f}m"
                  f"{r['fixed']:>13.2f}m{r['w_best']:>7.2f}{r['oracle']:>10.2f}m"
                  f"{margin:>8.1f}%{pred_pct:>11.1f}%")

    # ---------------- B) barrido de calidad de VO ----------------
    print("\n" + "=" * 104)
    print("B) ¿DEPENDE DE LA CALIDAD DE LA CÁMARA?   (GPS a 1 Hz = 10 frames, σ=5m)")
    print("=" * 104)
    print(f"{'deriva extra':>14}{'GPS solo':>11}{'VO sola':>11}{'mejor w fijo':>14}"
          f"{'(w)':>7}{'oráculo':>11}{'margen':>9}{'predecible':>12}")
    print("-" * 104)

    rows_b = []
    for deg in VO_DEGRADATIONS:
        r, w_by_seq = evaluate(caches, 10, 5.0, deg)
        _, _, pred_pct = variance_decomposition(w_by_seq)
        margin = 100 * (r['fixed'] - r['oracle']) / r['fixed'] if r['fixed'] > 0 else 0
        rows_b.append({'deg': deg, 'margin': margin, 'pred': pred_pct, **r})
        label = "VO real" if deg == 0 else f"+{deg:.2f}°/frame"
        print(f"{label:>14}{r['gps']:>10.2f}m{r['vo']:>10.2f}m{r['fixed']:>13.2f}m"
              f"{r['w_best']:>7.2f}{r['oracle']:>10.2f}m{margin:>8.1f}%{pred_pct:>11.1f}%")

    # ---------------- C) descomposición de varianza ----------------
    print("\n" + "=" * 104)
    print("C) ¿CUÁNTO DEL PESO ÓPTIMO ES APRENDIBLE?   (σ=5m, VO real)")
    print("=" * 104)
    print("El peso perfecto depende del ruido concreto de cada fix, que es impredecible.")
    print("Solo la parte que se REPITE entre distintas realizaciones de ruido es aprendible.\n")
    print(f"{'período':>9}{'tasa':>9}{'var predecible':>17}{'var por azar':>15}{'% aprendible':>15}")
    print("-" * 104)
    for period in GPS_PERIODS:
        _, w_by_seq = evaluate(caches, period, 5.0, 0.0)
        p, q, pct = variance_decomposition(w_by_seq)
        print(f"{period:>9}{10/period:>8.1f}Hz{p:>17.4f}{q:>15.4f}{pct:>14.1f}%")

    # ---------------- gráfico ----------------
    os.makedirs(args.out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for sigma in SIGMAS:
        sub = [r for r in rows_a if r['sigma'] == sigma]
        axes[0].plot([10/r['period'] for r in sub], [r['margin'] for r in sub],
                     'o-', label=f'σ={sigma:.0f}m')
    axes[0].axhline(30, ls='--', c='green', label='seguir (>30%)')
    axes[0].axhline(15, ls='--', c='red', label='parar (<15%)')
    axes[0].set_xscale('log'); axes[0].invert_xaxis()
    axes[0].set_xlabel('tasa de GPS (Hz)  — menor hacia la derecha')
    axes[0].set_ylabel('margen sobre el mejor peso fijo (%)')
    axes[0].set_title('A) Margen vs tasa de GPS'); axes[0].legend(); axes[0].grid(alpha=0.3)

    for sigma in SIGMAS:
        sub = [r for r in rows_a if r['sigma'] == sigma]
        axes[1].plot([10/r['period'] for r in sub], [r['pred'] for r in sub],
                     's-', label=f'σ={sigma:.0f}m')
    axes[1].set_xscale('log'); axes[1].invert_xaxis()
    axes[1].set_xlabel('tasa de GPS (Hz)  — menor hacia la derecha')
    axes[1].set_ylabel('% de la señal que es predecible')
    axes[1].set_title('C) Fracción aprendible del peso óptimo')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot([r['deg'] for r in rows_b], [r['margin'] for r in rows_b],
                 'o-', color='purple', label='margen')
    axes[2].plot([r['deg'] for r in rows_b], [r['pred'] for r in rows_b],
                 's--', color='teal', label='% predecible')
    axes[2].set_xlabel('deriva de rumbo extra (°/frame)')
    axes[2].set_ylabel('%')
    axes[2].set_title('B) Robustez a la calidad de la cámara')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle('Oráculo con tasas de sensor realistas (GPS intermitente)')
    fig.tight_layout()
    png = os.path.join(args.out_dir, 'realistic_rates.png')
    fig.savefig(png, dpi=130)
    print(f"\nGráfico guardado: {png}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
