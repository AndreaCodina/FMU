"""Forza scaricata e metrica energetica.

* Forza scaricata come la calcola il PLC (rActualForce): identificata sui test come
      F = -M_nom * LP(d2x/dt2),  LP = Butterworth 2° ordine 20 Hz causale, ritardo 2 campioni
  (errore relativo ~0.3% rispetto a rActualForce registrata).
* Velocità del suolo: integrale di rAccGround + passa-alto Butterworth (filtfilt).
* Energia: E(t) = integrale di vG * F; il confronto si fa sul valore finale.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from data_io import DT_SCOPE

PLC_FORCE_FC = 20.0      # [Hz]
PLC_FORCE_ORDER = 2
PLC_FORCE_DELAY = 2      # [campioni]
VG_HP_FC = 0.1           # passa-alto per la velocità del suolo [Hz]
VG_HP_ORDER = 2


def plc_force(x: np.ndarray, mass_nom: float, dt: float = DT_SCOPE) -> np.ndarray:
    """Ricostruisce rActualForce dalla posizione (stesso calcolo del PLC)."""
    acc = np.r_[0.0, 0.0, np.diff(x, 2)] / dt ** 2
    b, a = signal.butter(PLC_FORCE_ORDER, PLC_FORCE_FC * 2 * dt)
    acc = signal.lfilter(b, a, acc)
    acc = np.r_[np.zeros(PLC_FORCE_DELAY), acc[:-PLC_FORCE_DELAY]] if PLC_FORCE_DELAY else acc
    return -mass_nom * acc


def ground_velocity(acc_ground: np.ndarray, dt: float = DT_SCOPE) -> np.ndarray:
    v = np.cumsum(acc_ground) * dt
    b, a = signal.butter(VG_HP_ORDER, VG_HP_FC * 2 * dt, "high")
    return signal.filtfilt(b, a, v)


def energy(vg: np.ndarray, force: np.ndarray, dt: float = DT_SCOPE) -> np.ndarray:
    return np.cumsum(vg * force) * dt


def energy_error(E_ref: np.ndarray, E_sim: np.ndarray) -> float:
    """Scarto percentuale sul valore finale di energia."""
    return 100.0 * (E_sim[-1] - E_ref[-1]) / max(abs(E_ref[-1]), 1e-9)


class PLCForceFilter:
    """Versione causale, passo-passo, di plc_force (per il modello / la FMU)."""

    def __init__(self, mass_nom: float, dt: float = DT_SCOPE, x0: float = 0.0):
        self.m = mass_nom
        self.dt = dt
        self.b, self.a = (c.tolist() for c in signal.butter(PLC_FORCE_ORDER, PLC_FORCE_FC * 2 * dt))
        self.xh = [x0, x0]                       # x[k-1], x[k-2]
        self.n = 0
        self.u = [0.0] * len(self.b)             # ingressi del filtro
        self.y = [0.0] * len(self.a)             # uscite del filtro
        self.buf = [0.0] * PLC_FORCE_DELAY

    def step(self, x: float) -> float:
        acc = (x - 2 * self.xh[0] + self.xh[1]) / self.dt ** 2 if self.n >= 2 else 0.0
        self.n += 1
        self.xh = [x, self.xh[0]]
        self.u = [acc] + self.u[:-1]
        yk = sum(bi * ui for bi, ui in zip(self.b, self.u)) - sum(
            ai * yi for ai, yi in zip(self.a[1:], self.y[:-1]))
        self.y = [yk] + self.y[:-1]
        if self.buf:
            self.buf.append(yk)
            yk = self.buf.pop(0)
        return -self.m * yk
