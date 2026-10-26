"""Confronto modello / test sperimentali (CVT, sismi, sweep FRF)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

from amd_model import AMDModel
from data_io import DT_SCOPE, load_twincat_csv, mass_from_path
from force_energy import energy, energy_error, ground_velocity, plc_force
from params import AMDParams


# guadagni dell'anello di velocità usati nelle prove (Pr 03.010 [s/rad], Pr 03.011 [s²/rad]);
# le masse non elencate usano quelli di params.DriveParams (0.0735 / 4.80).
# Nota 510 kg: indicati 0.06 / 3.2, ma sweep "GainGiusti" e test CVT 510 kg tornano molto meglio
# con 0.0735 / 4.80 (sweep "GainAlti" ~0.09-0.11). Per usarli: TEST_SPEED_GAINS = {510.0: (0.06, 3.2)}
TEST_SPEED_GAINS: dict[float, tuple[float, float]] = {}


@dataclass
class TestCase:
    path: Path
    name: str
    mass: float
    n_strings: int
    t: np.ndarray
    x_ref: np.ndarray
    meas: dict = field(default_factory=dict)
    speed_gains: tuple | None = None

    @property
    def V0(self) -> float:
        return float(np.median(self.meas["Vbus"][:20]))


def prepare_test(path: str | Path, pre_s: float = 0.2) -> TestCase:
    """Taglia il test dal momento in cui il bus è carico (>720 V, come in MATLAB) a fine prova."""
    path = Path(path)
    d = load_twincat_csv(path)
    V = d["iVoltageDCLink"].values
    rel = sum(d[c].values for c in d.columns if c.startswith("bRelayStrings"))\
        if any(c.startswith("bRelayStrings") for c in d.columns) else np.ones(len(d))
    i0 = int(np.argmax(V > 720))
    if "bSweepActive" in d and d["bSweepActive"].max() > 0 and "Sweep" in path.name:
        sw = np.flatnonzero(d["bSweepActive"].values > 0.5)
        i0 = max(i0, sw[0] - int(pre_s / DT_SCOPE))
        i1 = sw[-1] + int(1.0 / DT_SCOPE)
    else:
        on = np.flatnonzero(rel > 0.5)
        i1 = on[-1] if len(on) else len(d)
    i1 = min(i1, len(d))
    sl = slice(i0, i1)
    n_str = int(max(round(float(np.max(rel[sl]))), 1)) if "Sweep" not in path.name else 1
    meas = {
        "x": d["rActualPosition"].values[sl],
        "Iq": d["rCurrentIq"].values[sl],
        "Vbus": V[sl].astype(float),
        "P_drive": d["diPowerOutput"].values[sl].astype(float),
        "F_actual": d["rActualForce"].values[sl] if "rActualForce" in d else np.zeros(i1 - i0),
        "acc_ground": d["rAccGround"].values[sl] if "rAccGround" in d else np.zeros(i1 - i0),
    }
    x_ref = d["rSetPosition"].values[sl]
    t = np.arange(len(x_ref)) * DT_SCOPE
    mass = mass_from_path(path) or 1320.0
    return TestCase(path, path.name, mass, n_str, t, x_ref, meas, TEST_SPEED_GAINS.get(mass))


def run_test(tc: TestCase, params: AMDParams | None = None, progress=None) -> dict:
    p = copy.deepcopy(params or AMDParams.default())
    p.mech.mass = tc.mass
    p.batt.n_strings = tc.n_strings
    if tc.speed_gains is not None:
        p.drive.speed_kp_drive, p.drive.speed_ki_drive = tc.speed_gains
    p.sim.dt = DT_SCOPE
    m = AMDModel(p)
    return m.simulate(tc.x_ref, x0=float(tc.meas["x"][0]), V0=tc.V0, precharged=True,
                      progress=progress)


def energy_metric(tc: TestCase, sim: dict) -> dict:
    """Energia scaricata: E = int(vG * F). Reale da rActualForce, FMU dalla posizione simulata
    con lo stesso calcolo del PLC."""
    vg = ground_velocity(tc.meas["acc_ground"])
    F_sim = sim["F_actual"] if "F_actual" in sim else plc_force(sim["x"], tc.mass)
    E_meas = energy(vg, tc.meas["F_actual"])
    E_sim = energy(vg, F_sim)
    E_rec = energy(vg, plc_force(tc.meas["x"], tc.mass))   # stesso calcolo sulla posizione misurata
    return dict(vg=vg, F_sim=F_sim, E_meas=E_meas, E_sim=E_sim,
                err_pct=energy_error(E_meas, E_sim), err_rec_pct=energy_error(E_meas, E_rec),
                F_rmse=float(np.sqrt(np.mean((F_sim - tc.meas["F_actual"]) ** 2))))


def metrics(tc: TestCase, sim: dict) -> dict:
    out = {}
    for k in ("x", "Iq", "Vbus", "P_drive"):
        e = sim[k] - tc.meas[k]
        rng = np.ptp(tc.meas[k]) or 1.0
        out[k] = dict(rmse=float(np.sqrt(np.mean(e ** 2))), nrmse=float(np.sqrt(np.mean(e ** 2)) / rng))
    return out


def frf(x_in: np.ndarray, y_out: np.ndarray, fs: float = 1 / DT_SCOPE, nperseg: int = 8192):
    f, Pxy = signal.csd(x_in, y_out, fs, nperseg=nperseg)
    _, Pxx = signal.welch(x_in, fs, nperseg=nperseg)
    _, coh = signal.coherence(x_in, y_out, fs, nperseg=nperseg)
    return f, Pxy / Pxx, coh


def model_frf_linear(params: AMDParams, mass: float, f: np.ndarray) -> np.ndarray:
    """FRF analitica (regime lineare, senza limiti) SetPosition -> ActualPosition."""
    dr, mo, me = params.drive, params.motor, params.mech
    s = 2j * np.pi * np.asarray(f, dtype=float)
    s = np.where(s == 0, 1e-9j, s)
    G = mo.kf / ((mass + me.mass_extra) * s ** 2 + me.c_visc * s)
    kpv, kiv = dr.speed_gains()
    Cv = (kpv + kiv / s) * np.exp(-s * dr.T_delay) / (1 + s * dr.i_filter)
    if dr.cl_enabled:
        Cv = Cv / (1 + s * dr.cl_tau)
    H = G * Cv * (dr.Kp_pos + dr.K_ff * s) / (1 + G * Cv * (dr.Kp_pos + s))
    # nota: ignora il limite dell'uscita P di posizione (valido solo per errori molto piccoli)
    return H * np.exp(-s * dr.T_ref)
