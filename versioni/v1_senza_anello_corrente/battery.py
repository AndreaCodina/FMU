"""Pacco batterie + bus DC del drive (porting del modello MATLAB 'main_CVT_Dissipazione...').

Modello per stringa:
    Vbus = Voc(Ah) - V1 - V2 - Rp * Ib
    Rp   = R0(|Ib|) * f(dI/dt) + R_cavi   (discesa istantanea, risalita lenta)
    V1,V2: rami RC di polarizzazione
La corrente si ricava dalla potenza richiesta dal drive risolvendo P = Vbus * Ib.
Sopra V_brake il chopper scarica l'eccesso sulla resistenza di frenatura; la potenza
rigenerativa sotto soglia rientra nelle batterie.

Precarica: con il solo contattore di precarica chiuso il DC link (C_dc) si carica
attraverso R_precharge; con il contattore principale chiuso il bus è collegato
direttamente al pacco.
"""
from __future__ import annotations

import bisect
import math

import numpy as np
from scipy import signal

from params import DATA_DIR, BatteryParams

CURVE_FILE = DATA_DIR / "ovonic_6S_1200mAh_scarica_1ohm.csv"


class VocCurve:
    """Curva tensione a vuoto del pacco in funzione della capacità scaricata (per stringa)."""

    def __init__(self, p: BatteryParams):
        raw = np.loadtxt(CURVE_FILE, delimiter=",", skiprows=1)
        cap, v = raw[:, 0] * p.capacity_scale, raw[:, 1] * p.n_series + p.voc_offset
        dc = float(np.mean(np.diff(cap)))
        grid = np.arange(0.0, cap[-1], dc)
        vg = np.interp(grid, cap, v)
        b, a = signal.butter(2, 0.02)  # come MATLAB: flp = 0.01*fsamp -> Wn = 0.02
        vf = signal.filtfilt(b, a, vg)
        self.dc = dc
        self.cap = grid
        self.v = vf.tolist()
        self.n = len(self.v)
        self.cap_max = float(grid[-1])

    def __call__(self, ah: float) -> float:
        x = ah / self.dc
        if x <= 0:
            return self.v[0]
        i = int(x)
        if i >= self.n - 1:
            return self.v[-1]
        f = x - i
        return self.v[i] * (1 - f) + self.v[i + 1] * f

    def ah_from_voltage(self, v0: float) -> float:
        arr = np.asarray(self.v)
        idx = int(np.argmin(np.abs(arr - v0)))
        return idx * self.dc


class BatteryBus:
    def __init__(self, p: BatteryParams):
        self.p = p
        self.voc = VocCurve(p)
        self._R_I = list(p.R_I)
        self._R_val = [r * p.R_scale for r in p.R_val]
        self._thr = list(p.dIdt_thr)
        self._fac = list(p.dIdt_fac)
        self.reset(740.0, precharged=True)

    # ------------------------------------------------------------------ utility
    def _R0(self, i: float) -> float:
        xs, ys = self._R_I, self._R_val
        if i <= xs[0]:
            return ys[0]
        if i >= xs[-1]:
            return ys[-1]
        k = bisect.bisect_right(xs, i) - 1
        f = (i - xs[k]) / (xs[k + 1] - xs[k])
        return ys[k] * (1 - f) + ys[k + 1] * f

    def _fac_dIdt(self, didt: float) -> float:
        k = bisect.bisect_right(self._thr, didt)
        return self._fac[min(max(k, 1), len(self._fac)) - 1] if didt > 0 else self._fac[0]

    # ------------------------------------------------------------------ API
    def reset(self, V0: float, precharged: bool = True):
        self.ah = self.voc.ah_from_voltage(V0)
        self.V1 = 0.0
        self.V2 = 0.0
        self.Ib = 0.0          # corrente per stringa (positiva in scarica)
        self._Ib_prev = 0.0
        self.Rp = 2.0
        self.didt_f = 0.0
        self.Vdc = V0 if precharged else 0.0
        self.Vbus = self.Vdc
        self.P_brake = 0.0
        self.main_closed = precharged
        self.pre_closed = False

    def soc(self) -> float:
        return max(0.0, 1.0 - self.ah / self.voc.cap_max)

    def step(self, P_drive: float, dt: float, main: bool = True, pre: bool = False) -> float:
        """Avanza di dt con potenza richiesta dal drive P_drive [W]; restituisce Vbus [V]."""
        p = self.p
        ns = max(int(p.n_strings), 1)
        Voc = self.voc(self.ah)
        Vd = Voc - self.V1 - self.V2

        # resistenza dinamica (logica MATLAB)
        ib_abs = abs(self.Ib)
        alpha_d = (2 * math.pi * p.fc_dIdt * dt) / (1 + 2 * math.pi * p.fc_dIdt * dt)
        didt_raw = abs(self.Ib - self._Ib_prev) / dt
        self._Ib_prev = self.Ib
        self.didt_f += alpha_d * (didt_raw - self.didt_f)
        Rp_nom = self._R0(ib_abs) * self._fac_dIdt(self.didt_f) + p.R_cable
        if Rp_nom < self.Rp:
            Rp = Rp_nom
        else:
            Rp = self.Rp + (dt / p.tau_rise) * (Rp_nom - self.Rp)
        self.Rp = Rp

        self.P_brake = 0.0
        if main:
            P = P_drive / ns
            disc = max(Vd * Vd - 4.0 * P * Rp, p.disc_lim)
            Ib = (Vd - math.sqrt(disc)) / (2.0 * Rp)
            Vbus = Vd - Ib * Rp
            if Vbus > p.V_brake:
                # chopper attivo: la resistenza di frenatura (R_brake) tiene il bus a V_brake
                # finché la sua potenza massima V^2/R basta, altrimenti la tensione sale oltre
                Vbus = p.V_brake
                if p.R_brake > 0:
                    a_ = ns / Rp + 1.0 / p.R_brake
                    b_ = ns * Vd / Rp
                    disc_b = b_ * b_ - 4.0 * a_ * P_drive
                    if disc_b > 0:
                        Vbus = max(p.V_brake, (b_ + math.sqrt(disc_b)) / (2.0 * a_))
                Ib = (Vd - Vbus) / Rp
            self.P_brake = max(Vbus * Ib * ns - P_drive, 0.0) if Vbus >= p.V_brake else 0.0
            self.Vdc = Vbus
        else:
            # DC link alimentato (eventualmente) tramite resistenza di precarica
            V = self.Vdc
            Ib_tot = (Vd - V) / (p.R_precharge + Rp / ns) if pre else 0.0
            I_load = P_drive / max(V, 1.0)
            dV = (Ib_tot - I_load - V / p.R_bleed) / p.C_dc
            # integrazione esatta del termine RC per stabilità con dt grande
            if pre:
                tau = p.R_precharge * p.C_dc
                V_inf = Vd - (I_load + V / p.R_bleed) * p.R_precharge
                V = V_inf + (V - V_inf) * math.exp(-dt / tau)
            else:
                V = V + dV * dt
            V = min(max(V, 0.0), p.V_brake)
            self.Vdc = V
            Vbus = V
            Ib = Ib_tot / ns

        self.Ib = Ib
        self.Vbus = Vbus
        # integrazione stati
        self.ah = min(max(self.ah + Ib * dt / 3600.0, 0.0), self.voc.cap_max)
        self.V1 += (Ib / (p.tau1 / p.R1) - self.V1 / p.tau1) * dt if p.R1 > 0 else 0.0
        if p.R2 > 0:
            self.V2 += (Ib / (p.tau2 / p.R2) - self.V2 / p.tau2) * dt
        self.Voc = Voc
        return Vbus

    @property
    def I_batt(self) -> float:
        return self.Ib * max(int(self.p.n_strings), 1)
