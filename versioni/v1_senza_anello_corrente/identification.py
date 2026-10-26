"""Identificazione dei parametri del modello dai test sperimentali.

Uso:
    python identification.py            # esegue tutti i passi e salva data/identified_params.json
    python identification.py --no-save  # solo stampa

Passi:
  1. attriti e kf medio (test CVT, regressione Iq ~ accelerazione; massa = massa mobile nominale)
  2. curva di saturazione kf_eff(|Iq|)                 (test CVT + sismi)
  3. (solo indicativo) fit lineare equivalente P+FF / PI sulle FRF; i guadagni usati dal modello
     vengono dai parametri del drive (struttura AMC) — vedi params.DriveParams
  4. modello di potenza assorbita dal bus (diPowerOutput)
  5. modello batteria (scala R interna, ramo RC) su iVoltageDCLink con potenza misurata
La curva limite di tensione I(v) (params.DriveParams.vlim_*) è stata ricavata
dall'inviluppo misurato e ottimizzata nel tempo sui test di sovraccarico; non viene
rieseguita qui perché richiede diversi minuti.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import optimize, signal

from battery import BatteryBus
from data_io import DT_SCOPE, list_tests, load_twincat_csv, mass_from_path
from params import IDENTIFIED_FILE, AMDParams, BatteryParams, MotorParams

G = 9.81
LAG = -3         # allineamento [campioni] tra Iq loggata e accelerazione (da cross-correlazione)
_B, _A = signal.butter(2, 20 / 500)


def _kin(d):
    x = d["rActualPosition"].values
    v = np.gradient(signal.filtfilt(_B, _A, x), DT_SCOPE)
    a = np.gradient(v, DT_SCOPE)
    i = np.roll(signal.filtfilt(_B, _A, d["rCurrentIq"].values), LAG)
    return v, a, i


def _good(f):
    return not any(s in f.name for s in ("NoOk", "Fail", "2Stringhe"))


# --------------------------------------------------------------------------- 1
def step_mass_friction(tests):
    """Iq = (M+m0)/kf * a + Fc/kf * sign(v) + c/kf * v  (fuori saturazione, |Iq| < 185 A)."""
    per_mass = {}
    for f in tests["CVT"]:
        if not _good(f):
            continue
        M = mass_from_path(f)
        d = load_twincat_csv(f)
        v, a, i = _kin(d)
        on = d["bActiveControlSystem"].values > 0.5
        k = on & (np.abs(i) > 3) & (np.abs(i) < 185)
        if k.sum() < 500:
            continue
        X = np.c_[a[k], np.sign(v[k]), v[k]]
        sol, *_ = np.linalg.lstsq(X, i[k], rcond=None)
        per_mass.setdefault(M, []).append(sol)
    Ms = np.array(sorted(per_mass))
    S = np.array([np.median(np.array(per_mass[M])[:, 0]) for M in Ms])
    # la massa mobile nominale comprende già forcer e carrello: Iq/a = M/kf (retta per l'origine)
    kf = float(np.sum(Ms ** 2) / np.sum(Ms * S))
    m0 = 0.0
    Fc = np.array([np.median(np.array(per_mass[M])[:, 1]) * kf for M in Ms])
    c = np.array([np.median(np.array(per_mass[M])[:, 2]) * kf for M in Ms])
    mu = float(np.median(Fc / ((Ms + m0) * G)))
    print(f"[1] kf medio = {kf:.1f} N/A, "
          f"mu = {mu:.4f}, c = {np.median(c):.0f} N s/m")
    return dict(kf=float(kf), m0=float(m0), mu=mu, c=float(max(np.median(c), 0)))


# --------------------------------------------------------------------------- 2
def step_kf_saturation(tests, base):
    edges = np.arange(0, 230, 20)
    vals = [[] for _ in edges]
    for f in tests["CVT"][::2] + tests["Sismi"]:
        if "2Stringhe" in f.name:
            continue
        M = mass_from_path(f) + base["m0"]
        d = load_twincat_csv(f)
        v, a, i = _kin(d)
        on = d["bActiveControlSystem"].values > 0.5
        F = M * a + base["mu"] * M * G * np.sign(v) + base["c"] * v
        k = on & (np.abs(i) > 20) & (np.abs(d["rCurrentIq"].values) < 199.5)
        idx = np.digitize(np.abs(i[k]), edges) - 1
        for j, val in zip(idx, F[k] / i[k]):
            vals[j].append(val)
    I_pts, kf_pts = [], []
    for j, e in enumerate(edges):
        if len(vals[j]) > 100:
            I_pts.append(e + 10)
            kf_pts.append(np.median(vals[j]))
    kf0 = kf_pts[0]
    fac = np.minimum(np.array(kf_pts) / kf0, 1.0)
    fac = np.minimum.accumulate(fac)
    I_tab = [0.0] + [float(x) for x in I_pts] + [300.0]
    fac_tab = [1.0] + [float(round(x, 3)) for x in fac] + [float(round(fac[-1] * 0.9, 3))]
    print(f"[2] kf_eff: {kf0:.1f} N/A a bassa corrente -> {kf_pts[-1]:.1f} N/A a ~{I_pts[-1]:.0f} A")
    return dict(kf=float(kf0), kf_sat_I=I_tab, kf_sat_fac=fac_tab)


# --------------------------------------------------------------------------- 3
def step_frf(tests, kf, m0, c, x0):
    data = []
    for f in tests["FRF"]:
        if "Alti" in f.name or "Test2" in f.name:
            continue
        M = mass_from_path(f)
        d = load_twincat_csv(f)
        sw = d["bSweepActive"].values > 0.5
        x, r = d["rActualPosition"].values[sw], d["rSetPosition"].values[sw]
        fr, Pxy = signal.csd(r, x, 1 / DT_SCOPE, nperseg=8192)
        _, Prr = signal.welch(r, 1 / DT_SCOPE, nperseg=8192)
        k = (fr > 0.3) & (fr < 28)
        data.append((M, fr[k], (Pxy / Prr)[k]))

    def model(p, M, f):
        Kpp, Kff, Kpv, Kiv, Td = p
        s = 2j * np.pi * f
        Gp = kf / ((M + m0) * s ** 2 + c * s)
        Cv = (Kpv + Kiv / s) * np.exp(-s * Td)
        return Gp * Cv * (Kpp + Kff * s) / (1 + Gp * Cv * (Kpp + s))

    def res(p):
        out = []
        for M, f, H in data:
            e = np.log(model(p, M, f) / H)
            out += [e.real, e.imag]
        return np.concatenate(out)

    sol = optimize.least_squares(res, x0, bounds=([0] * 5, [2000, 2, 1e4, 1e6, 0.05]))
    names = ("Kp_pos", "K_ff", "Kp_vel", "Ki_vel", "T_delay")
    out = {n: float(v) for n, v in zip(names, sol.x)}
    print("[3] drive: " + ", ".join(f"{n}={v:.4g}" for n, v in out.items()))
    return out


# --------------------------------------------------------------------------- 4
def step_power(tests, motor: MotorParams):
    A, Y = [], []
    for f in tests["CVT"][::4] + tests["Sismi"]:
        d = load_twincat_csv(f)
        V = d["iVoltageDCLink"].values
        on = (d["bActiveControlSystem"].values > 0.5) & (V > 650)
        v, _, _ = _kin(d)
        I = d["rCurrentIq"].values
        F = np.array([motor.force(x) for x in I])
        Pm = F * v
        A.append(np.c_[np.clip(Pm, 0, None), np.clip(Pm, None, 0), 3 * I ** 2, np.abs(v)][on])
        Y.append(d["diPowerOutput"].values[on])
    A, Y = np.vstack(A), np.concatenate(Y)
    sol, *_ = np.linalg.lstsq(A, Y, rcond=None)
    out = dict(k_mot=float(sol[0]), k_gen=float(sol[1]), P_v=float(sol[3]))
    print(f"[4] potenza: k_mot={sol[0]:.3f}, k_gen={sol[1]:.3f}, R_eq={sol[2]:.3f} ohm, P_v={sol[3]:.0f} W s/m"
          f"  (RMSE {np.sqrt(np.mean((A @ sol - Y) ** 2)) / 1e3:.2f} kW)")
    return out, float(sol[2])


# --------------------------------------------------------------------------- 5
def _batt_case(f):
    d = load_twincat_csv(f)
    V = d["iVoltageDCLink"].values.astype(float)
    P = d["diPowerOutput"].values.astype(float)
    rel = sum(d[c].values for c in d.columns if c.startswith("bRelayStrings"))
    i0 = int(np.argmax(V > 720))
    i1 = len(V) - int(np.argmax(rel[::-1] > 0.5))
    return V[i0:i1], P[i0:i1], int(max(rel.max(), 1))


def sim_battery(bp: BatteryParams, V, P, ns):
    bp.n_strings = ns
    b = BatteryBus(bp)
    b.reset(float(np.median(V[30:60])))
    return np.array([b.step(p, DT_SCOPE) for p in P])


def step_battery(tests):
    files = [f for f in tests["CVT"] if "GVmax60" in f.name or "GVmax40" in f.name][::4] + tests["Sismi"][::2]
    cases = [_batt_case(f) for f in files]

    def cost(x):
        bp = BatteryParams(R_scale=x[0], R1=x[1], tau1=x[2], tau_rise=x[3])
        if min(x) <= 0:
            return 1e6
        return float(np.mean([np.sqrt(np.mean((sim_battery(bp, V, P, ns) - V) ** 2)) for V, P, ns in cases]))

    x0 = [1.0, 0.35, 3.0, 2.0]
    c0 = cost(x0)
    sol = optimize.minimize(cost, x0, method="Nelder-Mead", options=dict(maxfev=80, xatol=1e-3, fatol=0.02))
    out = dict(R_scale=float(sol.x[0]), R1=float(sol.x[1]), tau1=float(sol.x[2]), tau_rise=float(sol.x[3]))
    print(f"[5] batteria: RMSE Vbus {c0:.2f} V (parametri MATLAB) -> {sol.fun:.2f} V con " +
          ", ".join(f"{k}={v:.3g}" for k, v in out.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    tests = list_tests()
    p = AMDParams.default(use_identified=False)

    base = step_mass_friction(tests)
    sat = step_kf_saturation(tests, base)
    motor = MotorParams(kf=sat["kf"], kf_sat_I=tuple(sat["kf_sat_I"]), kf_sat_fac=tuple(sat["kf_sat_fac"]),
                        R_ph=p.motor.R_ph, L_ph=p.motor.L_ph)
    # struttura e guadagni del drive: dai parametri reali (AMC, menu 3/4/39) + fit nel tempo sulle
    # sweep (scala del guadagno di velocità e ritardi). Il fit lineare qui sotto resta solo indicativo.
    step_frf(tests, sat["kf"], base["m0"], base["c"],
             [108.6, 0.41, 650.0, 3200.0, 0.0042])
    drv = {}
    pw, _ = step_power(tests, motor)
    bat = step_battery(tests)

    result = {
        "mech": dict(mass_extra=0.0, mu_coulomb=base["mu"], c_visc=base["c"]),
        "motor": dict(kf=sat["kf"], kf_sat_I=sat["kf_sat_I"], kf_sat_fac=sat["kf_sat_fac"]),
        "drive": {**drv, **pw},
        "batt": bat,
    }
    print(json.dumps(result, indent=2))
    if not args.no_save:
        IDENTIFIED_FILE.parent.mkdir(exist_ok=True)
        # unisce ai valori già presenti (es. curve limite tarate con tune_overload.py)
        old = json.loads(IDENTIFIED_FILE.read_text()) if IDENTIFIED_FILE.exists() else {}
        for grp, vals in result.items():
            old.setdefault(grp, {}).update(vals)
        IDENTIFIED_FILE.write_text(json.dumps(old, indent=2))
        print(f"Salvato {IDENTIFIED_FILE}")


if __name__ == "__main__":
    main()
