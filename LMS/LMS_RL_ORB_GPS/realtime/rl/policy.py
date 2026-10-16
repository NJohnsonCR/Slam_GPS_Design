"""
The RL's policy: tabular Q-learning over the outages of env.py, evaluated
leaving one session out, against the two baselines it has to be measured
with: the speed rule (ignore the camera above a speed) and the oracle (the
best constant k of each outage).

The protocol was fixed before the first training (CONTEXTO.md §4.30), and the
evaluation follows it as written:

    sessions   the cached ones but VALIDATION train and are tested in turn;
               VALIDATION is tested with the policy trained on all of them
    training   EPISODES episodes per policy, an outage of 10-60 s from a random
               second of a training session; epsilon-greedy from EPS_START to
               EPS_END; alpha = max(1 / visits, ALPHA_MIN); no discount (the
               episode ends when the GPS comes back); reward relative to the
               fixed filter; SEEDS policies per test
    tested on  the grid outages (one every 10 s, of 10/20/30/60 s) of the
               session left out, all test sessions pooled
    criterion  the policy takes at least CRITERION_CAPTURE of the margin from
               the fixed filter to the oracle in the p90 of the 20 and of the
               30 s outages, and with good GPS the prediction of each fix
               moves no more than GOOD_GPS_TOL_M in the p95

    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.rl.policy
    venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.rl.policy --sessions 14_05 14_18 --seeds 0 --check
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_LMS_RL = os.path.abspath(os.path.join(_THIS, "..", ".."))
_ROOT = os.path.abspath(os.path.join(_LMS_RL, "..", ".."))
for _p in (_ROOT, _LMS_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realtime.evaluate import FUSION
from realtime.fusion import feed_frame
from realtime.rl import env as E

EPISODES = 20_000
EPS_START, EPS_END = 0.5, 0.05
ALPHA_MIN = 0.02
GAMMA = 1.0
TRAIN_LENGTHS_S = (10, 60)          # outage lengths drawn in training, whole seconds
SEEDS = (0, 1, 2)
# The speed rule: ignore the camera above a speed, one threshold for every
# outage (when it starts nobody knows how long it will last).
RULE_THRESHOLDS = tuple(float(v) for v in range(0, 21, 2)) + (np.inf,)
CRITERION_LENGTHS = (20, 30)
CRITERION_CAPTURE = 0.25
GOOD_GPS_TOL_M = 0.2
POLICY_FILE = os.path.join(_THIS, "q_table.npz")
OUT_DIR = os.path.join(_ROOT, "resultados", "rl")


# ------------------------------------------------------------------- table

class QTable:
    """The policy: a value per state and action; on a tie, k = 1 (the fixed filter)."""

    def __init__(self):
        self.q = np.zeros((E.RlState.N_STATES, len(E.ACTIONS)))
        self.n = np.zeros(self.q.shape, np.int64)

    def best(self, s):
        return int(np.argmax(self.q[s]))        # the first of the ties: k = 1

    def controller(self, rng=None, eps=0.0):
        def act(state):
            if eps and rng.random() < eps:
                return E.ACTIONS[rng.integers(len(E.ACTIONS))]
            return E.ACTIONS[self.best(state.index())]
        return act

    def update(self, trans):
        """One episode's transitions, from the last back, so its outcome reaches its start."""
        for s, a, r, s2 in reversed(trans):
            target = r + (GAMMA * self.q[s2].max() if s2 is not None else 0.0)
            self.n[s, a] += 1
            self.q[s, a] += max(1.0 / self.n[s, a], ALPHA_MIN) * (target - self.q[s, a])

    def save(self, path, **meta):
        np.savez(path, q=self.q, n=self.n, actions=np.array(E.ACTIONS),
                 levels=json.dumps(E.RlState.LEVELS), meta=json.dumps(meta))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        if json.loads(str(d["levels"])) != {k: list(v) for k, v in E.RlState.LEVELS.items()} \
                or list(d["actions"]) != list(E.ACTIONS):
            raise ValueError(f"{path} se entrenó con otros niveles o acciones")
        t = cls()
        t.q, t.n = d["q"], d["n"]
        return t


# ----------------------------------------------------------------- sessions

class Bench:
    """A session's snapshots, and the fixed filter's error at each fix of the outages from them."""

    def __init__(self, session):
        self.name = session.name
        self.snaps = E.Snapshots(session)
        self._base = {}

    def baseline(self, i, length_s):
        """{t_ns: metres} of the fixed filter in outage (i, length_s)."""
        # Up to the GPS's return, the fixed filter does the same in any longer
        # outage from the same second: the longest one serves all.
        longest = TRAIN_LENGTHS_S[1]
        if (i, longest) not in self._base:
            ep = E.run_episode(self.snaps, i, longest, E.constant(1.0))
            self._base[(i, longest)] = None if ep is None else dict(ep.errors)
        if self._base[(i, longest)] is not None:
            return self._base[(i, longest)]
        if (i, length_s) not in self._base:
            self._base[(i, length_s)] = dict(E.run_episode(self.snaps, i, length_s,
                                                          E.constant(1.0)).errors)
        return self._base[(i, length_s)]


BENCHES = {}        # name -> Bench, built before the workers fork


def make_controller(spec, table=None):
    kind, value = spec
    if kind == "k":
        return E.constant(value)
    if kind == "rule":
        return lambda st: np.inf if st.speed > value else 1.0
    return table.controller()


def grid_errors(name, spec, table=None):
    """Return error of every grid outage of a session, {length: array}, and the RL's actions."""
    eps = E.grid_episodes(BENCHES[name].snaps, make_controller(spec, table))
    errors = {L: [ep.return_error for ep in eps[L]] for L in E.OUTAGE_LENGTHS}
    actions = [d[2] for L in E.OUTAGE_LENGTHS for ep in eps[L] for d in ep.decisions]
    return errors, actions


def good_gps(name, table=None):
    """The prediction against each fix over the whole session, without outages (p50, p95)."""
    s = BENCHES[name].snaps.session
    fus, j = E.PolicyFusion(controller=None if table is None else table.controller(), **FUSION), 0
    for k in range(len(s.t)):
        j = feed_frame(fus, k, s.t, s.inputs, s.gyro_yaw, s.fixes, j)
    e = [float(np.linalg.norm(r[4])) for r in fus.log if r[1] == "pos" and r[4] is not None]
    return float(np.percentile(e, 50)), float(np.percentile(e, 95)), len(fus.decisions)


# ---------------------------------------------------------------- training

def train(names, seed, episodes=EPISODES):
    """A policy trained on the outages of the given sessions; also its learning curve."""
    rng = np.random.default_rng(seed)
    table = QTable()
    benches = [BENCHES[n] for n in names]
    weight = np.array([len(b.snaps.copies) for b in benches], float)
    weight /= weight.sum()
    curve, recent, done = [], [], 0
    while done < episodes:
        eps = EPS_START + (EPS_END - EPS_START) * done / max(episodes - 1, 1)
        b = benches[rng.choice(len(benches), p=weight)]
        i = int(rng.integers(len(b.snaps.copies)))
        length = int(rng.integers(TRAIN_LENGTHS_S[0], TRAIN_LENGTHS_S[1] + 1))
        ep = E.run_episode(b.snaps, i, length, table.controller(rng, eps))
        if ep is None or not ep.decisions:
            continue
        trans = E.transitions(ep, b.baseline(i, length))
        table.update(trans)
        recent.append(sum(t[2] for t in trans))
        done += 1
        if done % 1000 == 0:
            # The mean is pulled by the few outages where ignoring the camera
            # saves hundreds of metres; the median shows the typical one.
            curve.append((float(np.mean(recent)), float(np.median(recent))))
            recent = []
    return table, curve


# --------------------------------------------------------------------- jobs

def job(task):
    """One unit of work for the pool; returns (task, result)."""
    kind = task[0]
    t0 = time.perf_counter()
    if kind == "grid":                              # a constant k or a rule, on one session
        _, name, spec = task
        out = grid_errors(name, spec)[0]
    elif kind == "rl":                              # train without the test session, test on it
        _, test, train_names, seed, save = task
        table, curve = train(train_names, seed)
        errors, actions = grid_errors(test, ("rl", None), table)
        out = {"errors": errors, "curve": curve, "good_gps": good_gps(test, table),
               "actions": {f"{k:g}": int(np.sum(np.array(actions) == k)) for k in E.ACTIONS},
               "states_visited": int(np.sum(table.n.sum(axis=1) > 0))}
        if save:
            table.save(save, trained_on=list(train_names), seed=seed, episodes=EPISODES)
    elif kind == "fixed_gps":
        _, name = task
        out = good_gps(name)
    return task, out, time.perf_counter() - t0


def pooled(per_session, names, L):
    return np.concatenate([np.asarray(per_session[n][L], float) for n in names])


def p(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Entrena y evalúa la política del RL dejando una sesión afuera")
    ap.add_argument("--sessions", nargs="*", help="Solo estas sesiones (basta un pedazo del nombre)")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--check", action="store_true",
                    help="Prueba de funcionamiento: no guarda la política ni se reporta")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    t_all = time.perf_counter()
    sessions = E.cached_sessions(report=print)
    if args.sessions:
        sessions = [s for s in sessions if any(p in s.name for p in args.sessions)]
    train_pool = [s.name for s in sessions if s.name not in E.VALIDATION]
    validation = [s.name for s in sessions if s.name in E.VALIDATION]
    if len(train_pool) < 2:
        print("Hacen falta al menos dos sesiones para dejar una afuera.")
        return 1
    print(f"Entrenamiento y prueba: {', '.join(train_pool)}")
    print(f"Validación: {', '.join(validation) or 'ninguna'}")
    if args.check:
        print("PRUEBA DE FUNCIONAMIENTO: estos números no se reportan.")
    for s in sessions:
        BENCHES[s.name] = Bench(s)
    print(f"Copias del filtro listas ({time.perf_counter() - t_all:.0f} s); "
          f"{EPISODES} episodios por política, semillas {args.seeds}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_DIR, ("prueba_" if args.check else "") + stamp)
    os.makedirs(out_dir, exist_ok=True)
    tasks = [("grid", n, ("k", k)) for n in train_pool + validation for k in E.ACTIONS]
    tasks += [("grid", n, ("rule", thr)) for n in train_pool + validation
              for thr in RULE_THRESHOLDS if np.isfinite(thr)]
    tasks += [("fixed_gps", n) for n in train_pool + validation]
    for test in train_pool:
        others = tuple(n for n in train_pool if n != test)
        tasks += [("rl", test, others, seed, None) for seed in args.seeds]
    for v in validation:
        for seed in args.seeds:
            save = None if args.check or seed != args.seeds[0] else POLICY_FILE
            tasks.append(("rl", v, tuple(train_pool), seed, save))
    # The long ones first, so the pool does not end waiting on one.
    tasks.sort(key=lambda t: t[0] != "rl")

    results, trained = {}, {}           # trained: (test session, seed) -> its policy's results
    with mp.get_context("fork").Pool(args.workers) as pool:
        for n, (task, out, dt) in enumerate(pool.imap_unordered(job, tasks), 1):
            results[task] = out
            if task[0] == "rl":
                trained[(task[1], task[3])] = out
                print(f"  [{n}/{len(tasks)}] política probada en {task[1]}, semilla {task[3]}: "
                      f"{dt / 60:.1f} min", flush=True)

    const = {k: {n: results[("grid", n, ("k", k))] for n in train_pool + validation} for k in E.ACTIONS}
    rule_runs = {thr: {n: results[("grid", n, ("rule", thr))] for n in train_pool + validation}
                 for thr in RULE_THRESHOLDS if np.isfinite(thr)}
    rule_runs[np.inf] = const[1.0]          # never ignoring it is the fixed filter

    def fit_rule(names):
        """The threshold with the lowest mean p90 of the criterion outages on these sessions."""
        score = {thr: np.mean([p(pooled(rule_runs[thr], names, L), 90) for L in CRITERION_LENGTHS])
                 for thr in RULE_THRESHOLDS}
        best = min(score.values())
        return max(thr for thr, v in score.items() if v == best)   # ties: the higher threshold

    report = {"protocolo": {"episodios": EPISODES, "eps": [EPS_START, EPS_END], "alpha_min": ALPHA_MIN,
                            "gamma": GAMMA, "largos_entrenamiento_s": TRAIN_LENGTHS_S,
                            "semillas": args.seeds, "umbrales_regla": [str(t) for t in RULE_THRESHOLDS],
                            "niveles": {k: list(v) for k, v in E.RlState.LEVELS.items()},
                            "prueba_de_funcionamiento": args.check},
              "sesiones": {}}
    rl_err = {seed: {} for seed in args.seeds}
    rule_err, oracle_err = {}, {}
    print("\nPor sesión de prueba (p90 al volver el GPS, m; la regla y el RL entrenados sin ella):")
    for test in train_pool + validation:
        others = [n for n in train_pool if n != test]
        thr = fit_rule(others)
        rule_err[test] = rule_runs[thr][test]
        oracle_err[test] = {L: np.min([const[k][test][L] for k in E.ACTIONS], axis=0)
                            for L in E.OUTAGE_LENGTHS}
        runs = [trained[(test, seed)] for seed in args.seeds]
        for seed, r in zip(args.seeds, runs):
            rl_err[seed][test] = r["errors"]
        fixed_gps = results[("fixed_gps", test)]
        rows = {}
        for L in E.OUTAGE_LENGTHS:
            rows[L] = {"cortes": len(const[1.0][test][L]),
                       "ekf_fijo": [p(const[1.0][test][L], 50), p(const[1.0][test][L], 90)],
                       "regla": [p(rule_err[test][L], 50), p(rule_err[test][L], 90)],
                       "rl": [[p(r["errors"][L], 50), p(r["errors"][L], 90)] for r in runs],
                       "oraculo": [p(oracle_err[test][L], 50), p(oracle_err[test][L], 90)]}
        report["sesiones"][test] = {
            "umbral_regla": str(thr), "cortes": rows,
            "gps_bueno_ekf_fijo": fixed_gps, "gps_bueno_rl": [r["good_gps"] for r in runs],
            "acciones_rl": [r["actions"] for r in runs], "curva": [r["curve"] for r in runs],
            "estados_visitados": [r["states_visited"] for r in runs]}
        line = " | ".join(
            f"{L} s: fijo {rows[L]['ekf_fijo'][1]:5.1f}, regla {rows[L]['regla'][1]:5.1f}, "
            f"RL {np.mean([x[1] for x in rows[L]['rl']]):5.1f}, oráculo {rows[L]['oraculo'][1]:5.1f}"
            for L in CRITERION_LENGTHS)
        print(f"  {test}{' (validación)' if test in validation else ''}: {line}; umbral de la regla {thr:g} m/s")

    def summary(names, label):
        print(f"\n{label} (p50 / p90 al volver el GPS, m; RL: media de {len(args.seeds)} semillas):")
        out = {}
        for L in E.OUTAGE_LENGTHS:
            f, o, r = pooled(const[1.0], names, L), pooled(oracle_err, names, L), pooled(rule_err, names, L)
            rl = [pooled(rl_err[seed], names, L) for seed in args.seeds]
            rl90 = [p(x, 90) for x in rl]
            margin = p(f, 90) - p(o, 90)
            # Below a metre of margin a share of it means nothing.
            cap = lambda v: (p(f, 90) - v) / margin if margin >= 1.0 else float("nan")
            out[L] = {"cortes": len(f), "ekf_fijo": [p(f, 50), p(f, 90)], "oraculo": [p(o, 50), p(o, 90)],
                      "regla": [p(r, 50), p(r, 90)], "rl_p50": [p(x, 50) for x in rl], "rl_p90": rl90,
                      "captura_regla": cap(p(r, 90)), "captura_rl": cap(float(np.mean(rl90))),
                      "captura_rl_por_semilla": [cap(v) for v in rl90]}
            c = out[L]
            share = lambda v: "sin margen" if np.isnan(v) else f"{100 * v:+.0f} %"
            print(f"  {L:2d} s ({c['cortes']} cortes): EKF fijo {c['ekf_fijo'][0]:5.1f} / {c['ekf_fijo'][1]:5.1f} | "
                  f"regla {c['regla'][0]:5.1f} / {c['regla'][1]:5.1f} ({share(c['captura_regla'])}) | "
                  f"RL {np.mean(c['rl_p50']):5.1f} / {np.mean(rl90):5.1f} ({share(c['captura_rl'])}; "
                  f"semillas {', '.join(share(v) for v in c['captura_rl_por_semilla'])}) | "
                  f"oráculo {c['oraculo'][0]:5.1f} / {c['oraculo'][1]:5.1f}")
        return out

    report["prueba"] = summary(train_pool, "Sesiones de prueba, todas juntas")
    if validation:
        report["validacion"] = summary(validation, "Validación")
    worst_gps = max(abs(r[1] - report["sesiones"][n]["gps_bueno_ekf_fijo"][1])
                    for n in train_pool for r in report["sesiones"][n]["gps_bueno_rl"])
    captures = [report["prueba"][L]["captura_rl"] for L in CRITERION_LENGTHS]
    ok = all(c >= CRITERION_CAPTURE for c in captures) and worst_gps <= GOOD_GPS_TOL_M
    report["criterio"] = {"captura_20_30": captures, "gps_bueno_peor_diferencia_p95_m": worst_gps, "cumple": ok}
    print(f"\nCon el GPS bueno, la predicción de cada fix: p95 a lo sumo {worst_gps:.2f} m del EKF fijo")
    print(f"Criterio (≥ {100 * CRITERION_CAPTURE:.0f} % del margen a 20 y 30 s, GPS bueno ± {GOOD_GPS_TOL_M} m): "
          f"captura {', '.join(f'{100 * c:.0f} %' for c in captures)} -> "
          f"{'CUMPLE' if ok else 'NO CUMPLE'}" + ("  (prueba de funcionamiento)" if args.check else ""))

    with open(os.path.join(out_dir, "evaluacion.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(f"\nResultados en {os.path.relpath(out_dir, _ROOT)}  ({(time.perf_counter() - t_all) / 60:.0f} min)")
    if not args.check and validation:
        print(f"Política entrenada con las {len(train_pool)} sesiones: {os.path.relpath(POLICY_FILE, _ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
