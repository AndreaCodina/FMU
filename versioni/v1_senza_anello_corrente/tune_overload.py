"""Taratura nel dominio del tempo sui test in sovraccarico (obiettivo: energia scaricata).

Ottimizza le curve di corrente massima vs velocità (limite di tensione del drive, in trazione e
in rigenerazione) e la banda del riferimento di velocità, minimizzando lo scarto sull'energia E = int(vG * F) e sull'inseguimento di
posizione, su un insieme di test in sovraccarico + alcuni test normali di controllo.

Uso:
    python tune_overload.py            # ottimizza e salva in data/identified_params.json
    python tune_overload.py --eval     # valuta solo i parametri attuali
    python tune_overload.py --ctrl     # ottimizza i parametri del controllore (+ Tt anti-windup)
"""
from __future__ import annotations

import argparse
import copy
import json
from multiprocessing import Pool

import numpy as np
from scipy import optimize

from data_io import list_tests
from params import IDENTIFIED_FILE, AMDParams
from validation import energy_metric, frf, metrics, model_frf_linear, prepare_test, run_test

OVERLOAD = [
    "Test35_Seno_f2.2_Flim21_GVmax40.5.csv", "Test4_Sisma1320_f1.4_Flim21_GVmax60.csv",
    "Test39_Seno1050_f1.48_Flim21_GVmax60.csv", "Test38_Seno780_f1.1_Flim21_GVmax25.csv",
    "Test41_Seno1320_f1_Flim21_GVmax40.5.csv", "Test42_Seno1320_f1.18_Flim21_GVmax60.csv",
    "Test38_Seno1050_f0.82_Flim21_GVmax25.csv", "Test3_SismaSLC31050_f1.75_Flim21_GVmax60.csv",
    "Test34_Seno_f1.7_Flim21_GVmax25.csv", "Test40_Seno780_f1.5_Flim21_GVmax40.5.csv",
    "Test40_Seno1050_f1.2_Flim21_GVmax40.5.csv",
]
CONTROL = [
    "Test12_Seno1320_f1.4_Flim21_GVmax60.csv", "Test7_Seno1320_f1.18_Flim21_GVmax40.5.csv",
    "Test43_Seno1320_f0.75_Flim21_GVmax25.csv", "Test12_Seno1050_f1.75_Flim21_GVmax60.csv",
    "Test39_Seno780_f2_Flim21_GVmax60.csv", "Test8_Sisma1320SLC2_f1.4_Flim21_GVmax60.csv",
]
SMOOTH = ("vlim_mot_v1", "vlim_mot_vc", "vlim_mot_n", "vlim_gen_v1", "vlim_gen_vc", "vlim_gen_n")
V_FLAT = 2.0   # corrente piena (200 A) garantita fino a questa velocità
V_MOT = (2.2, 2.5, 2.8, 3.2, 3.6, 4.0, 4.3)
V_GEN = (2.6, 3.0, 3.4, 3.8, 4.2, 4.6)

_TESTS = {}


def _init(names):
    allf = {f.name: f for k in ("CVT", "Sismi") for f in list_tests()[k]}
    for n in names:
        _TESTS[n] = prepare_test(allf[n])


def tables_from_x(x):
    im = np.minimum.accumulate(np.clip(x[:len(V_MOT)], 0, 200))
    ig = np.minimum.accumulate(np.clip(x[len(V_MOT):], 0, 200))
    return ((0.0, V_FLAT, *V_MOT, 5.0), (200.0, 200.0, *map(float, im), 0.0),
            (0.0, *V_GEN, 5.5), (200.0, *map(float, ig), 0.0))


def apply(p: AMDParams, x) -> AMDParams:
    p = copy.deepcopy(p)
    vm, im, vg, ig = tables_from_x(x)
    p.drive.vlim_v_mot, p.drive.vlim_I_mot, p.drive.vlim_v_gen, p.drive.vlim_I_gen = vm, im, vg, ig
    return p


def _eval_one(args):
    name, p = args
    tc = _TESTS[name]
    sim = run_test(tc, p)
    em = energy_metric(tc, sim)
    return name, em["err_pct"], metrics(tc, sim)["x"]["nrmse"] * 100


def evaluate(pool, p, names):
    return {n: (e, xe) for n, e, xe in pool.map(_eval_one, [(n, p) for n in names])}


def cost(res):
    e = np.array([min(abs(v[0]), 150.0) for v in res.values()])
    xe = np.array([min(v[1], 50.0) for v in res.values()])
    return float(np.mean(e) + 0.5 * np.mean(xe) + 2.0 * np.max(e) / len(e))


def x_from_params(p: AMDParams):
    im = np.interp(V_MOT, p.drive.vlim_v_mot, p.drive.vlim_I_mot)
    ig = np.interp(V_GEN, p.drive.vlim_v_gen, p.drive.vlim_I_gen)
    return np.r_[im, ig]


def report(res, title):
    print(f"--- {title}")
    for n, (e, xe) in res.items():
        flag = "OVL" if n in OVERLOAD else "ctl"
        print(f"  [{flag}] {n:52s} E {e:+8.2f}%   x {xe:6.2f}%")
    print(f"  costo {cost(res):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--maxfev", type=int, default=400)
    ap.add_argument("--ctrl", action="store_true")
    a = ap.parse_args()
    if a.ctrl:
        return tune_controller(a.maxfev)
    names = OVERLOAD + CONTROL
    base = AMDParams.default()
    smooth = base.drive.vlim_mode == "smooth"
    with Pool(min(16, len(names)), initializer=_init, initargs=(names,)) as pool:
        if smooth:
            x0 = np.array([getattr(base.drive, k) for k in SMOOTH])
        else:
            x0 = x_from_params(base)
        res0 = evaluate(pool, base, names)
        report(res0, "parametri attuali")
        if a.eval:
            return
        best = [x0, cost(res0)]
        n = [0]

        def mk(x):
            if smooth:
                p = copy.deepcopy(base)
                for k, v in zip(SMOOTH, x):
                    setattr(p.drive, k, float(v))
                return p
            return apply(base, x)

        def J(z):
            x = x0 * np.exp(z)
            c = cost(evaluate(pool, mk(x), names))
            n[0] += 1
            if c < best[1]:
                best[:] = [x, c]
                print(f"  eval {n[0]}: nuovo migliore {c:.3f}", flush=True)
            return c

        optimize.minimize(J, np.zeros(len(x0)), method="Nelder-Mead",
                          options=dict(maxfev=a.maxfev, xatol=1e-3, fatol=1e-3,
                                       initial_simplex=np.vstack([np.zeros(len(x0)),
                                                                  0.15 * np.eye(len(x0))])))
        pb = mk(best[0])
        report(evaluate(pool, pb, names), "parametri ottimizzati")
    d = json.loads(IDENTIFIED_FILE.read_text()) if IDENTIFIED_FILE.exists() else {}
    if smooth:
        d.setdefault("drive", {}).update({k: float(getattr(pb.drive, k)) for k in SMOOTH})
        print("curva limite: " + ", ".join(f"{k}={getattr(pb.drive, k):.3f}" for k in SMOOTH))
    else:
        vm, im, vg, ig = tables_from_x(best[0])
        d.setdefault("drive", {}).update(vlim_v_mot=vm, vlim_I_mot=im, vlim_v_gen=vg, vlim_I_gen=ig)
    IDENTIFIED_FILE.write_text(json.dumps(d, indent=2))
    print(f"Salvato in {IDENTIFIED_FILE}")


CTRL = ("Kp_pos", "K_ff", "kp_scale", "ki_scale", "T_delay", "aw_Tt")


def _frf_data():
    from data_io import load_twincat_csv, mass_from_path
    out = []
    for f in list_tests()["FRF"]:
        if "Alti" in f.name or "Test2" in f.name:
            continue
        d = load_twincat_csv(f)
        sw = d["bSweepActive"].values > 0.5
        fr, H, _ = frf(d["rSetPosition"].values[sw], d["rActualPosition"].values[sw])
        k = (fr > 0.3) & (fr < 28)
        out.append((mass_from_path(f), fr[k], H[k]))
    return out


def frf_cost(p, data):
    c = 0.0
    for M, f, H in data:
        e = np.log(model_frf_linear(p, M, f) / H)
        c += float(np.mean(e.real ** 2 + e.imag ** 2))
    return c / len(data)


def tune_controller(maxfev):
    names = OVERLOAD + CONTROL
    base = AMDParams.default()
    data = _frf_data()
    x0 = np.array([getattr(base.drive, k) for k in CTRL])
    frf0 = frf_cost(base, data)

    def mk(z):
        p = copy.deepcopy(base)
        for k, v in zip(CTRL, x0 * np.exp(z)):
            setattr(p.drive, k, float(v))
        return p

    with Pool(min(16, len(names)), initializer=_init, initargs=(names,)) as pool:
        res0 = evaluate(pool, base, names)
        report(res0, "parametri attuali")
        best = [np.zeros(len(x0)), cost(res0) + 20 * max(0.0, frf_cost(base, data) - frf0)]
        n = [0]

        def J(z):
            p = mk(z)
            # penalità se la FRF lineare peggiora rispetto al fit attuale
            c = cost(evaluate(pool, p, names)) + 200 * max(0.0, frf_cost(p, data) - 1.15 * frf0)
            n[0] += 1
            if c < best[1]:
                best[:] = [np.array(z), c]
                print(f"  eval {n[0]}: nuovo migliore {c:.3f}  " +
                      ", ".join(f"{k}={v:.4g}" for k, v in zip(CTRL, x0 * np.exp(z))), flush=True)
            return c

        optimize.minimize(J, np.zeros(len(x0)), method="Nelder-Mead",
                          options=dict(maxfev=maxfev, xatol=1e-3, fatol=1e-3,
                                       initial_simplex=np.vstack([np.zeros(len(x0)),
                                                                  0.15 * np.eye(len(x0))])))
        pb = mk(best[0])
        report(evaluate(pool, pb, names), "parametri ottimizzati")
    print(f"FRF cost: {frf0:.4f} -> {frf_cost(pb, data):.4f}")
    d = json.loads(IDENTIFIED_FILE.read_text()) if IDENTIFIED_FILE.exists() else {}
    d.setdefault("drive", {}).update({k: float(getattr(pb.drive, k)) for k in CTRL})
    IDENTIFIED_FILE.write_text(json.dumps(d, indent=2))
    print(f"Salvato in {IDENTIFIED_FILE}")


if __name__ == "__main__":
    main()
