"""
PASO 2 - Análisis del oráculo: ¿vale la pena un peso adaptativo?

Antes de entrenar nada, este script responde la pregunta que decide todo el
enfoque de RL:

    ¿Existe alguna política de ponderación variable en el tiempo que sea
    significativamente mejor que el MEJOR peso constante posible?

Para eso hace trampa a propósito: como en el cache está la posición verdadera,
se puede calcular en cada paso cuál habría sido el peso PERFECTO. Eso da el
techo de desempeño alcanzable por cualquier política. Si el techo apenas supera
al mejor peso constante, ninguna política adaptativa (RL incluido) va a aportar
y conviene saberlo ahora.

El peso óptimo por paso tiene solución cerrada. Fusionando
    p(w) = w*a + (1-w)*b
minimizar ||p(w) - g||^2 respecto de w da
    w* = -(a-b)·(b-g) / ||a-b||^2      recortado a [0,1]

DECISIONES DE DISEÑO (explícitas, afectan la lectura de los resultados):

  1. El GPS de KITTI (OXTS) es casi perfecto. Se usa como ground truth y se le
     inyecta ruido para simular el GPS que entra al sistema. Sin degradar, la
     respuesta óptima sería siempre "confiá en el GPS" y no habría nada que
     aprender.

  2. La VO se propaga con la escala real del ground truth. Esto separa la
     pregunta de la fusión de la pregunta de la estimación de escala, y le da
     a la rama visual su mejor caso: la comparación resulta CONSERVADORA para
     el RL.

  3. El oráculo es "codicioso" (elige el mejor w en cada paso sin mirar el
     futuro). Como la pose fusionada es el ancla del paso siguiente, no es
     necesariamente el óptimo global, pero es una referencia fuerte y barata.

Uso:
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.rl.oracle_analysis
"""

import os
import sys
import glob
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NOISE_LEVELS = [0.0, 1.0, 3.0, 5.0, 10.0]
N_SEEDS = 5
W_GRID = np.linspace(0.0, 1.0, 101)


def degrade_gps(gt, sigma, rng):
    """GPS simulado: ruido gaussiano en X e Y sobre la posición verdadera."""
    noisy = gt.copy()
    if sigma > 0:
        noisy[:, :2] += rng.normal(0.0, sigma, size=(len(gt), 2))
    return noisy


def replay(R_vo, t_vo, step_scales, gps, gt, w_policy, align_to_gps=False):
    """
    Reproduce el bucle de fusión del pipeline sobre el cache.

    w_policy: escalar (peso constante) o 'oracle' (peso perfecto por paso).

    Devuelve la trayectoria fusionada, los pesos usados y las features
    observables en cada paso (las que un agente real podría ver).
    """
    n = len(R_vo)
    pose = np.eye(4)
    pose[:3, 3] = gt[0]

    positions = [gt[0].copy()]
    weights = []
    innovations = []

    for i in range(n):
        rel = np.eye(4)
        rel[:3, :3] = R_vo[i]
        t = t_vo[i] * step_scales[i]

        if align_to_gps:
            # Replica la rotación de t hacia la dirección del GPS que hace el
            # pipeline actual (main.py:338-344). Solo para medir su efecto.
            d_gps = gps[i + 1][:2] - gps[i][:2]
            if np.linalg.norm(d_gps) > 0.1 and np.linalg.norm(t[:2]) > 1e-4:
                ang = np.arctan2(d_gps[1], d_gps[0]) - np.arctan2(t[1], t[0])
                c, s = np.cos(ang), np.sin(ang)
                t = np.array([c * t[0] - s * t[1], s * t[0] + c * t[1], t[2]])

        rel[:3, 3] = t
        pose_vo = pose @ rel

        a = pose_vo[:3, 3]          # estimación de la cámara
        b = gps[i + 1]              # estimación del GPS
        g = gt[i + 1]               # posición verdadera

        diff = a - b
        denom = float(diff @ diff)
        if w_policy == 'oracle':
            w = 0.5 if denom < 1e-12 else float(np.clip(-(diff @ (b - g)) / denom, 0.0, 1.0))
        else:
            w = float(w_policy)

        fused = w * a + (1.0 - w) * b

        pose = pose_vo.copy()
        pose[:3, 3] = fused

        positions.append(fused.copy())
        weights.append(w)
        innovations.append(np.linalg.norm(diff))

    return np.array(positions), np.array(weights), np.array(innovations)


def ate_rmse(traj, gt):
    """Error absoluto sin alineación: ambas están en el marco absoluto del GPS."""
    n = min(len(traj), len(gt))
    return float(np.sqrt(np.mean(np.linalg.norm(traj[:n] - gt[:n], axis=1) ** 2)))


def load_caches(cache_dir, min_frames):
    caches = []
    for f in sorted(glob.glob(os.path.join(cache_dir, '*_cache.npz'))):
        d = np.load(f, allow_pickle=True)
        if len(d['R_vo']) < min_frames:
            print(f"  (omitida {os.path.basename(f)}: solo {len(d['R_vo'])} pasos)")
            continue
        caches.append({
            'name': str(d['sequence']),
            'R_vo': d['R_vo'], 't_vo': d['t_vo'],
            'step_scales': d['step_scales'], 'gt': d['gt'],
            'n_matches': d['n_matches'], 'n_inliers': d['n_inliers'],
        })
    return caches


def main():
    ap = argparse.ArgumentParser(description="Paso 2: análisis del oráculo")
    ap.add_argument('--cache-dir', default='resultados/rl_cache')
    ap.add_argument('--out-dir', default='resultados/rl_cache')
    ap.add_argument('--min-frames', type=int, default=50)
    ap.add_argument('--align-gps', action='store_true',
                    help="Replicar la rotación de t hacia el GPS del pipeline actual")
    args = ap.parse_args()

    print("=" * 100)
    print("ANÁLISIS DEL ORÁCULO — ¿tiene sentido un peso adaptativo?")
    print("=" * 100)

    caches = load_caches(args.cache_dir, args.min_frames)
    if not caches:
        print("ERROR: no hay caches. Corré primero extract_features.py")
        return 1
    print(f"\nSecuencias: {len(caches)}  ({', '.join(c['name'].replace('2011_','') for c in caches)})\n")

    rows = []
    detail = {}

    for sigma in NOISE_LEVELS:
        acc = {'oracle': [], 'best_fixed': [], 'gps': [], 'vo': [], 'w_best': []}

        for c in caches:
            per_seed = {'oracle': [], 'best_fixed': [], 'gps': [], 'vo': [], 'w_best': []}

            for seed in range(N_SEEDS):
                rng = np.random.default_rng(1000 * seed + int(sigma * 10))
                gps = degrade_gps(c['gt'], sigma, rng)

                traj_o, w_o, innov = replay(c['R_vo'], c['t_vo'], c['step_scales'],
                                            gps, c['gt'], 'oracle', args.align_gps)
                per_seed['oracle'].append(ate_rmse(traj_o, c['gt']))

                # Mejor peso CONSTANTE, elegido con ventaja (mirando el resultado).
                # Es la vara honesta: el RL debe superar esto, no a la heurística.
                errs = [ate_rmse(replay(c['R_vo'], c['t_vo'], c['step_scales'],
                                        gps, c['gt'], w, args.align_gps)[0], c['gt'])
                        for w in W_GRID]
                k = int(np.argmin(errs))
                per_seed['best_fixed'].append(errs[k])
                per_seed['w_best'].append(W_GRID[k])
                per_seed['gps'].append(errs[0])    # w=0  -> GPS puro
                per_seed['vo'].append(errs[-1])    # w=1  -> VO propagada sola

                if sigma == 5.0 and seed == 0:
                    detail[c['name']] = {
                        'w_oracle': w_o, 'innov': innov,
                        'inlier_ratio': c['n_inliers'] / np.maximum(c['n_matches'], 1),
                        'n_matches': c['n_matches'],
                        'step': c['step_scales'],
                    }

            for k2 in acc:
                acc[k2].append(float(np.mean(per_seed[k2])))

        rows.append({
            'sigma': sigma,
            'oracle': float(np.mean(acc['oracle'])),
            'best_fixed': float(np.mean(acc['best_fixed'])),
            'gps': float(np.mean(acc['gps'])),
            'vo': float(np.mean(acc['vo'])),
            'w_best': float(np.mean(acc['w_best'])),
        })

    print(f"{'σ GPS':>7}{'GPS puro':>11}{'VO sola':>10}{'Mejor w fijo':>14}{'(w)':>7}"
          f"{'ORÁCULO':>11}{'margen':>10}")
    print("-" * 100)
    for r in rows:
        margin = 100 * (r['best_fixed'] - r['oracle']) / r['best_fixed'] if r['best_fixed'] > 0 else 0
        print(f"{r['sigma']:>6.0f}m{r['gps']:>10.2f}m{r['vo']:>9.2f}m"
              f"{r['best_fixed']:>13.2f}m{r['w_best']:>7.2f}{r['oracle']:>10.2f}m{margin:>9.1f}%")

    print("\n" + "-" * 100)
    print("El 'margen' es cuánto mejoraría un peso PERFECTO variable respecto del")
    print("MEJOR peso constante. Es el techo de lo que cualquier política adaptativa")
    print("—RL incluido— podría capturar.")
    print("   < 15 %  -> no hay nada que aprender, replantear el enfoque")
    print("   > 30 %  -> hay margen real, seguir con el entrenamiento")

    # ---- ¿el peso perfecto es constante o varía? ----
    print("\n" + "=" * 100)
    print("¿EL PESO PERFECTO VARÍA EN EL TIEMPO?  (σ = 5 m)")
    print("=" * 100)
    print(f"{'secuencia':<30}{'media':>8}{'desv':>8}{'p10':>8}{'p90':>8}{'% en extremos':>15}")
    print("-" * 100)
    for name, d in detail.items():
        w = d['w_oracle']
        extremes = 100 * np.mean((w < 0.02) | (w > 0.98))
        print(f"{name.replace('2011_',''):<30}{w.mean():>8.3f}{w.std():>8.3f}"
              f"{np.percentile(w,10):>8.3f}{np.percentile(w,90):>8.3f}{extremes:>14.0f}%")

    # ---- ¿qué variables predicen el peso perfecto? ----
    print("\n" + "=" * 100)
    print("CORRELACIÓN DEL PESO PERFECTO CON CADA VARIABLE CANDIDATA  (σ = 5 m)")
    print("=" * 100)
    feat_names = ['inlier_ratio', 'n_matches', 'innovacion |a-b|', 'movimiento/frame', 'indice de frame']
    all_w, all_f = [], []
    for name, d in detail.items():
        n = len(d['w_oracle'])
        F = np.column_stack([
            d['inlier_ratio'][:n], d['n_matches'][:n], d['innov'][:n],
            d['step'][:n], np.arange(n) / n,
        ])
        all_w.append(d['w_oracle']); all_f.append(F)
    W = np.concatenate(all_w); F = np.vstack(all_f)

    print(f"{'variable':<22}{'correlación':>14}   interpretación")
    print("-" * 100)
    for j, fn in enumerate(feat_names):
        col = F[:, j]
        c = 0.0 if col.std() < 1e-12 else float(np.corrcoef(col, W)[0, 1])
        verdict = ("SIN INFORMACIÓN (constante)" if col.std() < 1e-12
                   else "informativa" if abs(c) > 0.15
                   else "débil")
        print(f"{fn:<22}{c:>14.3f}   {verdict}")

    # ---- gráfico ----
    os.makedirs(args.out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    s = [r['sigma'] for r in rows]
    axes[0].plot(s, [r['gps'] for r in rows], 'o-', label='GPS puro (w=0)')
    axes[0].plot(s, [r['vo'] for r in rows], 's-', label='VO sola (w=1)')
    axes[0].plot(s, [r['best_fixed'] for r in rows], '^-', label='Mejor peso constante')
    axes[0].plot(s, [r['oracle'] for r in rows], 'd-', lw=2.5, color='crimson', label='ORÁCULO (techo)')
    axes[0].set_xlabel('Ruido GPS σ (m)'); axes[0].set_ylabel('ATE RMSE (m)')
    axes[0].set_title('Desempeño vs calidad del GPS'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].hist(W, bins=50, color='steelblue', edgecolor='k', alpha=0.8)
    axes[1].set_xlabel('peso perfecto w*'); axes[1].set_ylabel('frecuencia')
    axes[1].set_title(f'Distribución del peso perfecto (σ=5m)\nmedia={W.mean():.3f}  desv={W.std():.3f}')
    axes[1].grid(alpha=0.3)

    axes[2].plot(s, [100*(r['best_fixed']-r['oracle'])/r['best_fixed'] if r['best_fixed']>0 else 0
                     for r in rows], 'o-', lw=2.5, color='darkgreen')
    axes[2].axhline(30, ls='--', c='green', label='seguir (>30%)')
    axes[2].axhline(15, ls='--', c='red', label='parar (<15%)')
    axes[2].set_xlabel('Ruido GPS σ (m)'); axes[2].set_ylabel('margen sobre el mejor peso fijo (%)')
    axes[2].set_title('Margen disponible para adaptatividad'); axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle('Paso 2 — Análisis del oráculo' + (' [con alineación GPS]' if args.align_gps else ''))
    fig.tight_layout()
    suffix = '_aligned' if args.align_gps else ''
    png = os.path.join(args.out_dir, f'oracle_analysis{suffix}.png')
    fig.savefig(png, dpi=130)
    print(f"\nGráfico guardado: {png}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
