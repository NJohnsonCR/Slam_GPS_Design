"""
Is choosing the scale source per frame worth learning?

Two scale sources compete on every KITTI frame: the depth model and the scale
derived from GPS positions. The script answers two questions in order:

  1. Is there margin? A per-frame oracle that always picks the better source
     is compared with the best constant choice.
  2. Is it learnable? Policies built on observable features are fitted with
     leave-one-sequence-out validation, so they are always scored on a
     sequence they never saw.

Ground truth is used only to score. Every feature is observable live.

KITTI's GPS is navigation grade, so it is degraded with Gaussian noise to
create the conflict. A phone GPS fails differently (biased, correlated jumps),
which limits how far the result transfers.

Usage (after depth_scale_probe.py has produced the per-sequence files):
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.scale.scale_decision
"""

import glob
import os
import sys
import zlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))
_LMS_RL = os.path.abspath(os.path.join(_THIS, "..", ".."))
_ROOT = os.path.abspath(os.path.join(_LMS_RL, "..", ".."))
_RL = os.path.join(_LMS_RL, "scripts", "rl")
for _p in (_ROOT, _LMS_RL, _RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realistic_rates import degrade_gps          # noqa: E402
from sanity_checks import gps_derived_scales     # noqa: E402

PERIOD = 10          # 1 Hz GPS over 10 Hz KITTI frames
SIGMAS = (1.0, 3.0, 5.0, 10.0)
SEEDS = range(4)
FEATURES = ("n_pts", "n_match", "n_inlier", "inlier_ratio", "dispersion", "v_gps")

MIN_MARGIN_PCT = 10.0    # oracle gain over the best constant needed to call it margin
MIN_CAPTURE_PCT = 25.0   # share of that margin a validated policy must recover


# ------------------------------------------------------------------ loading

def load_sequences(cache_dir="resultados/rl_cache"):
    """Join the depth probe output with the per-sequence feature cache."""
    out = []
    for f in sorted(glob.glob(os.path.join(cache_dir, "*_depth_scale.npz"))):
        seq = os.path.basename(f).replace("_depth_scale.npz", "")
        cache = os.path.join(cache_dir, f"{seq}_cache.npz")
        if not os.path.exists(cache):
            print(f"  (sin cache de features para {seq}, se omite)")
            continue
        d = np.load(f)
        if "dispersion" not in d:
            print(f"  ({seq} es de una corrida vieja sin features, se omite)")
            continue
        c = np.load(cache, allow_pickle=True)
        out.append({
            "seq": seq,
            "idx": d["idx"].astype(int),
            "depth_step": d["scales"],
            "true_step": d["true_steps"],
            "n_pts": d["n_pts"].astype(float),
            "n_match": d["n_match"].astype(float),
            "n_inlier": d["n_inlier"].astype(float),
            "dispersion": d["dispersion"].astype(float),
            "gt": c["gt"],
        })
    return out


def source_errors(s, sigma, seed):
    """Absolute step error of each source (m) plus the observable features."""
    # str hash() is randomized per process; crc32 keeps runs reproducible.
    rng = np.random.default_rng(
        zlib.crc32(f"{s['seq']}|{sigma}|{seed}".encode()) % (2 ** 32))
    gps = degrade_gps(s["gt"], sigma, rng)
    n_steps = len(s["gt"]) - 1
    gps_steps = gps_derived_scales(gps, PERIOD, n_steps)

    ok = s["idx"] < n_steps
    idx = s["idx"][ok]
    true_step = s["true_step"][ok]

    err_depth = np.abs(s["depth_step"][ok] - true_step)
    err_gps = np.abs(gps_steps[idx] - true_step)

    # Speed as the GPS itself reports it, which the live system also has.
    v_gps = gps_steps[idx] * 10.0

    feats = np.column_stack([
        s["n_pts"][ok], s["n_match"][ok], s["n_inlier"][ok],
        s["n_inlier"][ok] / np.maximum(s["n_match"][ok], 1.0),
        s["dispersion"][ok], v_gps,
    ])
    return err_depth, err_gps, feats


# ------------------------------------------------------------------- margin

def pooled_errors(data, sigma):
    """Errors and features of every sequence and seed, stacked."""
    E_d, E_g, X, S = [], [], [], []
    for s in data:
        for seed in SEEDS:
            ed, eg, f = source_errors(s, sigma, seed)
            E_d.append(ed); E_g.append(eg); X.append(f)
            S.append(np.full(len(ed), s["seq"]))
    E_d = np.concatenate(E_d); E_g = np.concatenate(E_g)
    X = np.vstack(X); S = np.concatenate(S)

    finite = np.isfinite(E_d) & np.isfinite(E_g) & np.isfinite(X).all(1)
    return E_d[finite], E_g[finite], X[finite], S[finite]


def margin_summary(E_d, E_g):
    depth_only, gps_only = E_d.mean(), E_g.mean()
    oracle = np.minimum(E_d, E_g).mean()
    best_const = min(depth_only, gps_only)
    margin = 100.0 * (best_const - oracle) / max(best_const, 1e-9)
    return depth_only, gps_only, oracle, best_const, margin


# ---------------------------------------------------------------- policies

def logistic_policy(E_d, E_g, X, S):
    """Logistic regression on all features, leave-one-sequence-out."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = (E_d < E_g).astype(int)          # 1 = depth is the better source
    pred = np.zeros(len(y), dtype=int)

    for seq in np.unique(S):
        test = S == seq
        train = ~test
        if y[train].min() == y[train].max():
            pred[test] = y[train][0]
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[train], y[train])
        pred[test] = clf.predict(X[test])

    err = np.where(pred == 1, E_d, E_g).mean()
    return err, 100.0 * (pred == y).mean(), y


def threshold_policy(E_d, E_g, X, S, feature="v_gps"):
    """
    Single threshold on one feature, leave-one-sequence-out.

    With seven sequences from different domains, a multi-feature classifier
    has room to memorize the training domain; one parameter does not.
    """
    v = X[:, FEATURES.index(feature)]
    pred = np.zeros(len(E_d), dtype=bool)

    for seq in np.unique(S):
        test = S == seq
        train = ~test
        best_u, best_e = np.median(v[train]), np.inf
        for u in np.percentile(v[train], np.arange(5, 100, 5)):
            e = np.where(v[train] < u, E_d[train], E_g[train]).mean()
            if e < best_e:
                best_u, best_e = u, e
        pred[test] = v[test] < best_u

    err = np.where(pred, E_d, E_g).mean()
    return err, 100.0 * (pred == (E_d < E_g)).mean()


# --------------------------------------------------------------- breakdowns

def per_sequence(data, sigma):
    """Which source wins in each sequence; a pooled margin can hide one outlier."""
    rows = []
    for s in data:
        ed, eg, X = [], [], []
        for seed in SEEDS:
            a, b, f = source_errors(s, sigma, seed)
            ed.append(a); eg.append(b); X.append(f)
        ed = np.concatenate(ed); eg = np.concatenate(eg); X = np.vstack(X)
        fin = np.isfinite(ed) & np.isfinite(eg) & np.isfinite(X).all(1)
        ed, eg, X = ed[fin], eg[fin], X[fin]
        rows.append((s["seq"], ed.mean(), eg.mean(),
                     np.median(X[:, FEATURES.index("v_gps")]),
                     100.0 * (ed < eg).mean()))
    return rows


def feature_correlations(E_d, E_g, X):
    """Correlation of each feature with the depth source's advantage."""
    advantage = E_g - E_d
    out = []
    for j, name in enumerate(FEATURES):
        x = X[:, j]
        c = np.corrcoef(x, advantage)[0, 1] if np.std(x) > 1e-12 else 0.0
        out.append((name, c))
    return out


# --------------------------------------------------------------------- main

def main():
    print("=" * 88)
    print("PUERTA DE DECISIÓN DEL OE3 — ¿es aprendible elegir la fuente de escala?")
    print("=" * 88)

    data = load_sequences()
    if len(data) < 3:
        print(f"\n  Solo {len(data)} secuencias con datos. Corré primero:")
        print("  venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.scale.depth_scale_probe \\")
        print("      kitti_data/2011_09_26/*_sync")
        return 1

    print(f"\n  Secuencias: {len(data)}  "
          f"({', '.join(s['seq'].replace('2011_', '').replace('_sync', '') for s in data)})")
    print(f"  GPS degradado a sigma = {SIGMAS} m, {len(SEEDS)} semillas cada uno")
    print(f"  Features observables: {', '.join(FEATURES)}")

    rows = []
    print("\n" + "=" * 88)
    print("1. ¿HAY MARGEN?  (error medio de escala por paso, en metros)")
    print("=" * 88)
    print(f"{'sigma GPS':>10}{'solo depth':>13}{'solo GPS':>11}{'mejor cte':>12}"
          f"{'oráculo':>10}{'margen':>10}")
    print("-" * 88)
    for sg in SIGMAS:
        E_d, E_g, X, S = pooled_errors(data, sg)
        d, g, oracle, const, margin = margin_summary(E_d, E_g)
        print(f"{sg:>9.0f}m{d:>12.3f}{g:>11.3f}{const:>12.3f}{oracle:>10.3f}{margin:>9.1f}%")
        rows.append((sg, E_d, E_g, X, S, d, g, oracle, const, margin))

    print("\n" + "=" * 88)
    print("2. ¿ES APRENDIBLE?  (validación dejando una secuencia afuera)")
    print("=" * 88)
    print(f"{'sigma GPS':>10}{'margen':>10}{'regresión 6 feats':>19}{'captura':>10}"
          f"{'regla 1 umbral':>16}{'captura':>10}")
    print("-" * 88)
    verdicts = []
    for (sg, E_d, E_g, X, S, d, g, oracle, const, margin) in rows:
        err_lr, _, _ = logistic_policy(E_d, E_g, X, S)
        err_rule, _ = threshold_policy(E_d, E_g, X, S)
        cap_lr = 100.0 * (const - err_lr) / max(const - oracle, 1e-9)
        cap_rule = 100.0 * (const - err_rule) / max(const - oracle, 1e-9)
        print(f"{sg:>9.0f}m{margin:>9.1f}%{err_lr:>19.3f}{cap_lr:>9.1f}%"
              f"{err_rule:>16.3f}{cap_rule:>9.1f}%")
        verdicts.append((sg, margin, cap_lr, cap_rule))
    print("\n  Captura negativa = la política es PEOR que quedarse con la mejor constante.")

    print("\n" + "=" * 88)
    print("3. ¿DE DÓNDE SALE EL MARGEN?  (por secuencia, sigma = 5 m)")
    print("=" * 88)
    print(f"{'secuencia':<28}{'err depth':>11}{'err GPS':>10}{'gana':>9}"
          f"{'v mediana':>12}{'% frames depth mejor':>22}")
    print("-" * 88)
    for seq, ed, eg, v, pct in per_sequence(data, 5.0):
        print(f"{seq.replace('2011_', '').replace('_sync', ''):<28}"
              f"{ed:>11.3f}{eg:>10.3f}{'depth' if ed < eg else 'GPS':>9}"
              f"{v:>11.1f}m/s{pct:>21.1f}%")

    print("\n" + "=" * 88)
    print("4. CORRELACIÓN DE CADA FEATURE CON LA VENTAJA DE LA PROFUNDIDAD")
    print("=" * 88)
    _, E_d, E_g, X, S, *_ = rows[len(rows) // 2]
    for name, c in feature_correlations(E_d, E_g, X):
        print(f"  {name:<16}{c:+.3f}  {'#' * int(abs(c) * 40)}")
    print("  (|correlación| < 0.1 = la feature no dice nada)")

    print("\n" + "=" * 88)
    print("VEREDICTO")
    print("=" * 88)
    has_margin = any(m >= MIN_MARGIN_PCT for _, m, _, _ in verdicts)
    rule_works = any(m >= MIN_MARGIN_PCT and r >= MIN_CAPTURE_PCT
                     for _, m, _, r in verdicts)
    lr_works = any(m >= MIN_MARGIN_PCT and l >= MIN_CAPTURE_PCT
                   for _, m, l, _ in verdicts)

    if not has_margin:
        print(f"  NO HAY MARGEN (ninguna sigma supera {MIN_MARGIN_PCT:.0f}%).")
        print("  Elegir la fuente de escala frame a frame no tiene nada que ganar.")
    elif rule_works and not lr_works:
        print("  HAY MARGEN, Y LO CAPTURA UNA REGLA DE UN SOLO UMBRAL.")
        print("  La regresión con 6 features es PEOR que no decidir: con 7 secuencias")
        print("  de dominios distintos, sobra capacidad para memorizar el dominio.")
        print()
        print("  CONSECUENCIA PARA EL OE3: el baseline a superar ya no es 'la mejor")
        print("  constante' sino la regla de umbral. Ese es el techo real que un")
        print("  agente de RL tendría que batir, y hay que declararlo así.")
    elif not rule_works:
        print(f"  HAY MARGEN pero NADA lo captura ({MIN_CAPTURE_PCT:.0f}% mínimo).")
        print("  Opciones: más datos, mejores features, o replantear el OE3.")
    else:
        print("  HAY MARGEN Y ES APRENDIBLE. El OE3 tiene sustancia.")
    print("=" * 88)

    _plot(rows, verdicts)
    return 0


def _plot(rows, verdicts, out="resultados/rl_cache/oe3_decision_gate.png"):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sg = [r[0] for r in rows]
    ax[0].plot(sg, [r[5] for r in rows], "o-", label="solo profundidad", color="#4FD1C5")
    ax[0].plot(sg, [r[6] for r in rows], "s-", label="solo GPS", color="#F2A03D")
    ax[0].plot(sg, [r[7] for r in rows], "^--", label="oráculo por frame", color="#38A169")
    ax[0].set_xlabel("ruido del GPS σ (m)"); ax[0].set_ylabel("error de escala (m/paso)")
    ax[0].set_title("Cuánto se puede ganar eligiendo bien")
    ax[0].grid(alpha=.3); ax[0].legend()

    ax[1].bar([str(int(s)) for s, _, _, _ in verdicts], [r for _, _, _, r in verdicts],
              color="#B85042")
    ax[1].axhline(MIN_CAPTURE_PCT, ls="--", color="gray",
                  label=f"puerta: {MIN_CAPTURE_PCT:.0f}%")
    ax[1].set_xlabel("ruido del GPS σ (m)")
    ax[1].set_ylabel("% del margen capturado")
    ax[1].set_title("Captura de la regla de un umbral (validada)")
    ax[1].grid(alpha=.3); ax[1].legend()

    fig.suptitle("Puerta de decisión del OE3 — elegir la fuente de escala")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nGráfico: {out}")


if __name__ == "__main__":
    sys.exit(main())
