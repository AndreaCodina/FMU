"""Modello completo AMD: SetPosition -> ActualPosition.

Catena simulata (passo interno h = dt / n_sub):

  x_ref ──► [anello posizione P + FF] ─v_ref─► [anello velocità PI] ─Iq_cmd─► [ritardo T_delay]
        ──► [limite corrente 200 A] ──► [limite tensione: f.c.e.m. + R I + wL I <= Vbus]
        ──► F = kf * Iq ──► [massa + attriti + bumper] ──► x, v
  P_dc = f(F v, R I^2, |v|) ──► [pacco batterie + bus DC + chopper 780 V] ──► Vbus (retroazione)

L'interfaccia `reset()` / `do_step()` ricalca quella di una FMU co-simulation: il
wrapper FMU (fmu_wrapper.py) la chiama direttamente.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from battery import BatteryBus
from force_energy import PLCForceFilter
from params import AMDParams, bumper_coefficients

OUTPUT_NAMES = [
    "x", "v", "a", "x_ref", "Iq_cmd", "Iq", "I_lim", "F_motor", "F_bumper",
    "Vbus", "I_batt", "Voc", "SOC", "P_drive", "P_brake", "P_mech",
    "F_actual", "sat_current", "sat_voltage", "bumper_contact", "drive_enabled",
]


class FracDelay:
    """Ritardo puro di d campioni (anche frazionario, interpolazione lineare)."""

    def __init__(self, d: float, x0: float = 0.0):
        self.n = int(math.floor(d))
        self.f = d - self.n
        self.buf = deque([x0] * (self.n + 1), maxlen=self.n + 1)

    def step(self, u: float) -> float:
        newer = self.buf[1] if self.n > 0 else u      # u[k-n]
        y = (1.0 - self.f) * newer + self.f * self.buf[0]   # buf[0] = u[k-n-1]
        self.buf.append(u)
        return y


class AMDModel:
    def __init__(self, params: AMDParams | None = None):
        self.p = params or AMDParams.default()
        self.batt = BatteryBus(self.p.batt)
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, x0: float = 0.0, V0: float | None = None, precharged: bool | None = None):
        p = self.p
        self.h = p.sim.dt / max(int(p.sim.n_sub), 1)
        self.k_b, self.d_b = bumper_coefficients(p.mech)
        self.m = p.mech.m_tot
        self.kpv, self.kiv = p.drive.speed_gains()
        self.delay = FracDelay(p.drive.T_delay / self.h, 0.0) if p.drive.T_delay > 0 else None
        self.ref_buf = FracDelay(p.drive.T_ref / p.sim.dt, x0) if p.drive.T_ref > 0 else None
        self.x = x0
        self.v = 0.0
        self.a = 0.0
        self.integ = 0.0
        self.iq_f = 0.0
        self.iq_act = 0.0
        self.x_ref_prev = x0
        self.vbus_f = None
        self.vff_f = 0.0
        self.f_plc = PLCForceFilter(p.mech.mass, p.sim.dt, x0)
        self.t = 0.0
        pre = p.sim.bus_precharged if precharged is None else precharged
        self.batt.reset(p.sim.V0 if V0 is None else V0, precharged=pre)
        self.main_closed = pre
        self.pre_closed = False
        self.out = dict.fromkeys(OUTPUT_NAMES, 0.0)
        self.out["Vbus"] = self.batt.Vbus

    # ------------------------------------------------------------------ fisica
    def current_limit(self, v: float, vbus: float, sign_i: float) -> float:
        """Massima |Iq| disponibile per limite di corrente e di tensione (Id = 0)."""
        mo, dr = self.p.motor, self.p.drive
        if dr.vlim_mode == "smooth":
            v_eff = abs(v) * (dr.vlim_Vref / max(vbus, 1.0)) ** dr.vlim_alpha
            if sign_i * v >= 0:
                if dr.vlim_mot_form == "envelope":
                    # inviluppo misurato: I = I_max (1 + (v/vk)^(m p))^(-1/p) (coda ~ v^-m)
                    return dr.I_max * (1.0 + (v_eff / dr.vlim_mot_vk) ** (dr.vlim_mot_m * dr.vlim_mot_p))                         ** (-1.0 / dr.vlim_mot_p)
                v1, vc, n = dr.vlim_mot_v1, dr.vlim_mot_vc, dr.vlim_mot_n
            else:
                v1, vc, n = dr.vlim_gen_v1, dr.vlim_gen_vc, dr.vlim_gen_n
            if v_eff <= v1:
                return dr.I_max
            return dr.I_max / (1.0 + ((v_eff - v1) / vc) ** n)
        if dr.vlim_mode == "table":
            v_eff = abs(v) * dr.vlim_Vref / max(vbus, 1.0)
            if sign_i * v >= 0:
                i_v = float(np.interp(v_eff, dr.vlim_v_mot, dr.vlim_I_mot))
            else:
                i_v = float(np.interp(v_eff, dr.vlim_v_gen, dr.vlim_I_gen))
            return min(dr.I_max, i_v)
        Vm = dr.mod_index * vbus / math.sqrt(6.0)          # tensione di fase rms disponibile
        R, ke = mo.R_ph, mo.ke
        X = (math.pi * abs(v) / mo.pole_pitch) * mo.L_ph   # reattanza wL
        e = ke * abs(v)
        s = 1.0 if sign_i * v >= 0 else -1.0                # motore (+) o freno (-)
        A = R * R + X * X
        B = 2.0 * e * R * s
        C = e * e - Vm * Vm
        disc = B * B - 4 * A * C
        if disc < 0:
            return 0.0
        i_v = (-B + math.sqrt(disc)) / (2 * A)
        return min(dr.I_max, max(i_v, 0.0))

    def bumper_force(self, x: float, v: float) -> float:
        s = self.p.mech.stroke
        if x > s:
            return -max(self.k_b * (x - s) + self.d_b * v, 0.0)
        if x < -s:
            return max(self.k_b * (-s - x) - self.d_b * v, 0.0)
        return 0.0

    def drive_power(self, F: float, v: float, iq: float, enabled: bool) -> float:
        if not enabled:
            return 0.0
        dr, mo = self.p.drive, self.p.motor
        pm = F * v
        pm = dr.k_mot * pm if pm > 0 else dr.k_gen * pm
        return pm + 3.0 * mo.R_ph * iq * iq + dr.P_v * abs(v) + dr.P_0

    # ------------------------------------------------------------------ step
    def do_step(self, x_ref: float, enable: bool = True, main_contactor: bool | None = None,
                precharge_contactor: bool | None = None) -> dict:
        """Avanza di un passo dt con riferimento di posizione x_ref [m]."""
        p = self.p
        dr, mo, me = p.drive, p.motor, p.mech
        dt, h, n = p.sim.dt, self.h, max(int(p.sim.n_sub), 1)
        x_ref_in = x_ref
        if self.ref_buf is not None:
            x_ref = self.ref_buf.step(x_ref_in)
        main = self.main_closed if main_contactor is None else main_contactor
        pre = self.pre_closed if precharge_contactor is None else precharge_contactor
        self.main_closed, self.pre_closed = main, pre

        vff_raw = (x_ref - self.x_ref_prev) / dt
        if dr.v_ref_filter > 0:
            self.vff_f += (dt / (dr.v_ref_filter + dt)) * (vff_raw - self.vff_f)
        else:
            self.vff_f = vff_raw
        xr0 = self.x_ref_prev
        sat_i = sat_v = contact = False
        Pd_acc = Pb_acc = Pm_acc = 0.0
        F_b = iq = iq_cmd = ilim = ilim_out = 0.0
        vbus = self.batt.Vbus
        if self.vbus_f is None:
            self.vbus_f = vbus
        a_vb = h / (dr.vbus_filter + h) if dr.vbus_filter > 0 else 1.0
        enabled = enable and vbus > dr.V_undervolt

        for k in range(n):
            xr = xr0 + (x_ref - xr0) * (k + 1) / n
            if enabled:
                v_pos = dr.Kp_pos * (xr - self.x)
                if dr.pos_speed_clamp > 0:   # Pr 13.012: limite uscita P dell'anello di posizione
                    v_pos = max(-dr.pos_speed_clamp, min(dr.pos_speed_clamp, v_pos))
                v_ref = v_pos + dr.K_ff * self.vff_f
                if dr.v_ref_band > 0:
                    # in sovraccarico il drive reale insegue la velocità del riferimento: v_ref resta
                    # entro +/- v_ref_band dalla derivata del riferimento (identificato dai test)
                    v_ref = max(vff_raw - dr.v_ref_band, min(vff_raw + dr.v_ref_band, v_ref))
                v_ref = max(-dr.v_max, min(dr.v_max, v_ref))
                ev = v_ref - self.v
                iq_cmd = self.kpv * ev + self.integ
            else:
                ev = 0.0
                iq_cmd = 0.0
                self.integ = 0.0
            if self.delay is not None:
                iq_del = self.delay.step(iq_cmd)
            else:
                iq_del = iq_cmd
            pre_v = False
            if dr.vlim_before_filter and dr.i_filter > 0:
                # limite di tensione applicato prima del filtro di corrente: in campo di tensione la
                # corrente non segue un taglio istantaneo ma la dinamica dell'anello di corrente,
                # quindi i passaggi limite <-> comando del regolatore sono raccordati
                lim_v = self.current_limit(self.v, self.vbus_f, iq_del)
                pre_v = abs(iq_del) > lim_v and lim_v < dr.I_max - 1e-9
                iq_del = max(-lim_v, min(lim_v, iq_del))
            if dr.i_filter > 0:      # Pr 04.012: filtro del 1° ordine sul riferimento di corrente
                if dr.clamp_before_filter:   # uscita del regolatore di velocità già limitata
                    iq_del = max(-dr.I_max, min(dr.I_max, iq_del))
                self.iq_f += h / (dr.i_filter + h) * (iq_del - self.iq_f)
                iq_del = self.iq_f
            if dr.vlim_before_filter and dr.i_filter > 0:
                ilim = dr.I_max                       # dopo il filtro resta solo il limite del drive
                ilim_out = self.current_limit(self.v, self.vbus_f, iq_del)
            else:
                ilim = ilim_out = self.current_limit(self.v, self.vbus_f, iq_del)
            iq = max(-ilim, min(ilim, iq_del))
            if dr.cl_enabled:
                # anello di corrente + dinamica elettrica: la corrente insegue il riferimento con
                # costante cl_tau, ma la sua velocità di variazione è limitata dalla tensione
                # disponibile: L di/dt = V - R i - E, con |V| <= Vbus/sqrt(6) * cl_vscale
                Vm = dr.cl_vscale * self.vbus_f / math.sqrt(6.0)
                e_bemf = mo.ke * self.v + mo.R_ph * self.iq_act
                di = (iq - self.iq_act) / dr.cl_tau
                di = max((-Vm - e_bemf) / mo.L_ph, min((Vm - e_bemf) / mo.L_ph, di))
                self.iq_act += di * h
                self.iq_act = max(-dr.I_max, min(dr.I_max, self.iq_act))
                iq = self.iq_act
            saturated = abs(iq_del) > ilim or pre_v
            if saturated:
                sat_i |= abs(iq_del) > ilim
                sat_v |= pre_v or ilim < dr.I_max - 1e-9
            if enabled and dr.antiwindup == "tracking":
                # back-calculation a inseguimento: l'integrale viene riportato verso il valore
                # che mantiene il comando al limite attuale (corrente o tensione) con costante Tt
                lim_now = self.current_limit(self.v, self.vbus_f, iq_cmd)
                sat_cmd = max(-lim_now, min(lim_now, iq_cmd))
                self.integ += (self.kiv * ev + (sat_cmd - iq_cmd) / dr.aw_Tt) * h
            elif enabled and dr.antiwindup == "backcalc":
                # in limite di corrente: back-calculation (l'uscita PI resta al limite);
                # in limite di tensione: integrale libero ma limitato a +/- I_max
                self.integ += self.kiv * ev * h
                if ilim >= dr.I_max - 1e-9:
                    p_term = self.kpv * ev
                    self.integ = max(-dr.I_max - p_term, min(dr.I_max - p_term, self.integ))
                else:
                    self.integ = max(-dr.I_max, min(dr.I_max, self.integ))
            elif enabled and not (saturated and ev * iq_del > 0):
                # integrazione condizionata: non integra se saturo nella direzione dell'errore
                self.integ += self.kiv * ev * h
            if not enabled:
                iq = 0.0
                self.iq_act = 0.0

            F_m = mo.force(iq) * mo.pos_factor(self.x)
            F_b = self.bumper_force(self.x, self.v)
            contact |= F_b != 0.0
            F_fr = me.c_visc * self.v + me.f_coulomb * math.tanh(self.v / me.v_eps)
            self.a = (F_m + F_b - F_fr) / self.m
            self.v += self.a * h
            self.x += self.v * h

            Pd = self.drive_power(F_m, self.v, iq, enabled)
            vbus = self.batt.step(Pd, h, main=main, pre=pre)
            self.vbus_f += a_vb * (vbus - self.vbus_f)
            enabled = enable and vbus > dr.V_undervolt
            Pd_acc += Pd
            Pb_acc += self.batt.P_brake
            Pm_acc += F_m * self.v

        self.x_ref_prev = x_ref
        self.t += dt
        o = self.out
        o.update(
            x=self.x, v=self.v, a=self.a, x_ref=x_ref_in, Iq_cmd=iq_cmd, Iq=iq, I_lim=ilim_out,
            F_motor=mo.force(iq) * mo.pos_factor(self.x), F_bumper=F_b, Vbus=vbus, I_batt=self.batt.I_batt,
            Voc=self.batt.Voc, SOC=self.batt.soc(), P_drive=Pd_acc / n, P_brake=Pb_acc / n,
            P_mech=Pm_acc / n, F_actual=self.f_plc.step(self.x), sat_current=float(sat_i), sat_voltage=float(sat_v),
            bumper_contact=float(contact), drive_enabled=float(enabled),
        )
        return o

    # ------------------------------------------------------------------ batch
    def simulate(self, x_ref: np.ndarray, enable: np.ndarray | None = None,
                 main: np.ndarray | None = None, pre: np.ndarray | None = None,
                 x0: float | None = None, V0: float | None = None,
                 precharged: bool | None = None, progress=None) -> dict[str, np.ndarray]:
        """Simula un intero profilo campionato a dt. Restituisce un dict di array."""
        N = len(x_ref)
        self.reset(x0=float(x_ref[0]) if x0 is None else x0, V0=V0, precharged=precharged)
        res = {k: np.empty(N) for k in OUTPUT_NAMES}
        for i in range(N):
            o = self.do_step(float(x_ref[i]),
                             enable=True if enable is None else bool(enable[i]),
                             main_contactor=None if main is None else bool(main[i]),
                             precharge_contactor=None if pre is None else bool(pre[i]))
            for k in OUTPUT_NAMES:
                res[k][i] = o[k]
            if progress is not None and i % 2000 == 0:
                progress(i / N)
        res["t"] = np.arange(N) * self.p.sim.dt
        return res
