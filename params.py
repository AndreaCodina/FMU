"""Parametri del modello AMD (motore lineare + drive Unidrive M701 + pacco batterie).

Tutti i parametri sono raccolti in dataclass modificabili (da GUI, da script o, in
futuro, come parametri dell'FMU). I valori di default derivano da datasheet,
dal modello MATLAB 'main_CVT_Dissipazione...' e dall'identificazione sui test
sperimentali (vedi identification.py, che può sovrascriverli salvando
'data/identified_params.json').
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
IDENTIFIED_FILE = DATA_DIR / "identified_params.json"


@dataclass
class MechParams:
    mass: float = 1320.0          # massa mobile nominale AMD [kg] (modulabile)
    mass_extra: float = 0.0       # massa aggiuntiva [kg] (0: la massa mobile comprende già il forcer)
    stroke: float = 0.95          # semicorsa: posizione di contatto bumper (+/-) [m] (modulabile)
    c_visc: float = 250.0         # attrito viscoso equivalente [N s/m]
    mu_coulomb: float = 0.013     # coeff. attrito guide: F_c = mu * m_tot * g [-]
    v_eps: float = 0.01           # regolarizzazione segno velocità [m/s]
    # bumper di fine corsa: taratura sul dato sperimentale 100 kN @ 4 m/s, 1320 kg
    bumper_F_ref: float = 100e3   # forza di picco misurata [N]
    bumper_v_ref: float = 4.0     # velocità di impatto del test [m/s]
    bumper_m_ref: float = 1320.0  # massa del test [kg]
    bumper_e: float = 0.3         # coefficiente di restituzione [-]
    bumper_k: float = 0.0         # rigidezza [N/m]  (0 = calcolata dalla taratura)
    bumper_d: float = 0.0         # smorzamento [N s/m] (0 = calcolato da e)

    @property
    def m_tot(self) -> float:
        return self.mass + self.mass_extra

    @property
    def f_coulomb(self) -> float:
        return self.mu_coulomb * self.m_tot * 9.81


@dataclass
class MotorParams:
    name: str = "Hiwin LMFA64L"
    kf: float = 150.0             # costante di forza a bassa corrente, identificata [N/A] (datasheet 171.4, errato)
    R_ph: float = 0.43            # resistenza di fase (autotune drive 0.441, fit potenza 0.427) [ohm]
    L_ph: float = 3.45e-3         # induttanza di fase (autotune Ld/Lq 3.48/3.42 mH) [H]
    pole_pitch: float = 0.023     # passo polare tau [m] (2tau = 46 mm)
    ke_ph: float = 0.0            # cost. f.c.e.m. fase rms [V/(m/s)] (0 = kf/3)
    # saturazione magnetica: kf_eff = kf * fattore(|Iq|), identificata dai test CVT/sismi
    # (a 200 A la forza misurata è ~22.2 kN per tutte le masse -> kf_eff ~ 111 N/A)
    kf_sat_I: tuple = (0.0, 30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0, 190.0, 200.0, 300.0)
    # dipendenza di kf dalla posizione (statori non uniformi), dai test: F_reale/F_modello
    # ~1.01 per x<0 e 0.95-0.98 per x>0 (tupla vuota = disattivata)
    kf_pos_x: tuple = (-0.825, -0.675, -0.525, -0.375, -0.225, -0.075, 0.075, 0.225, 0.375, 0.525, 0.675, 0.825)
    kf_pos_fac: tuple = (1.001, 1.008, 1.010, 1.012, 1.012, 1.005, 0.982, 0.977, 0.967, 0.954, 0.951, 0.947)
    kf_sat_fac: tuple = (1.0, 1.0, 0.988, 0.979, 0.959, 0.931, 0.895, 0.853, 0.811, 0.758, 0.741, 0.667)

    @property
    def ke(self) -> float:
        return self.ke_ph if self.ke_ph > 0 else self.kf / 3.0

    def pos_factor(self, x: float) -> float:
        xs, ys = self.kf_pos_x, self.kf_pos_fac
        if not xs:
            return 1.0
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        k = 0
        while xs[k + 1] < x:
            k += 1
        return ys[k] + (ys[k + 1] - ys[k]) * (x - xs[k]) / (xs[k + 1] - xs[k])

    def force(self, iq: float) -> float:
        a = abs(iq)
        xs, ys = self.kf_sat_I, self.kf_sat_fac
        if a <= xs[1]:
            fac = ys[1]
        elif a >= xs[-1]:
            fac = ys[-1]
        else:
            k = 1
            while xs[k + 1] < a:
                k += 1
            fac = ys[k] + (ys[k + 1] - ys[k]) * (a - xs[k]) / (xs[k + 1] - xs[k])
        return self.kf * fac * iq


MOTOR_PRESETS = {
    "Hiwin LMFA64L": dict(name="Hiwin LMFA64L", kf=150.0, R_ph=0.43, L_ph=3.45e-3,
                          pole_pitch=0.023, ke_ph=0.0),
    "Siemens 1FN3900-4WC00": dict(name="Siemens 1FN3900-4WC00", kf=112.0, R_ph=0.239 * 1.3,
                                  L_ph=3.54e-3, pole_pitch=0.023, ke_ph=37.5),
}


@dataclass
class DriveParams:
    """Unidrive M701 RFC-S con AMC (menu 31-41): posizione P (limitata) + feed-forward di velocità
    -> anello velocità PI -> filtro riferimento di corrente -> limiti di corrente/tensione."""
    I_max: float = 200.0          # limite di corrente (picco drive) [A]
    Kp_pos: float = 25.0          # Kp anello posizione AMC (Pr 39.007) [1/s]
    K_ff: float = 1.0             # feed-forward di velocità AMC (Pr 39.010) [-]
    # guadagni anello di velocità nelle unità del drive (Pr 03.010 [s/rad], Pr 03.011 [s²/rad])
    speed_kp_drive: float = 0.0735
    speed_ki_drive: float = 4.80
    # conversione unità drive -> modello, identificata sulle sweep 780/1050/1320 kg
    kp_scale: float = 12024.5     # [A/(m/s)] per [s/rad]   (0.0735 s/rad -> 883.8 A/(m/s))
    ki_scale: float = 11894.4     # [A/m] per [s²/rad]      (4.80 s²/rad -> 57093 A/m)
    # guadagni diretti in A, usati solo se speed_kp_drive <= 0
    Kp_vel: float = 883.8         # P anello velocità [A/(m/s)]
    Ki_vel: float = 57093.0       # I anello velocità [A/m]
    T_delay: float = 0.9e-3       # ritardo residuo nell'anello [s] (identificato sulle sweep)
    T_ref: float = 4.0e-3         # ritardo riferimento PLC -> AMC (EtherCAT + interpolazione) [s]
    i_filter: float = 4.0e-3      # filtro 1° ordine riferimento di corrente (Pr 04.012) [s]
    clamp_before_filter: bool = False  # limita a I_max il comando prima del filtro di corrente
    vlim_before_filter: bool = True   # limite di tensione prima del filtro di corrente (passaggi raccordati)
    # anello di corrente con dinamica elettrica (False = modello v1, corrente = riferimento)
    cl_enabled: bool = False
    cl_tau: float = 0.5e-3        # costante di tempo dell'anello di corrente [s]
    cl_vscale: float = 1.0        # scala della tensione disponibile Vbus/sqrt(6) [-]
    v_max: float = 4.6            # limite velocità uscita AMC (Pr 39.011 = 6000 rpm, 46 mm/giro) [m/s]
    pos_speed_clamp: float = 0.07665  # limite uscita P posizione AMC (Pr 39.009 = 109.2 UU/ms) [m/s]
    v_ref_band: float = 0.0       # (fenomenologico, non più necessario con la struttura AMC) [m/s]
    antiwindup: str = "tracking"  # "tracking" (default) | "backcalc" | "conditional"
    aw_Tt: float = 0.02           # costante di tempo anti-windup "tracking" [s]
    v_ref_filter: float = 0.0     # costante di tempo filtro derivata setpoint [s]
    mod_index: float = 1.0        # sfruttamento tensione bus (Vfase_rms_max = m*Vdc/sqrt(6))
    # limite di tensione: 'smooth' = curva liscia (sotto); 'table' = tabella I(v), scalata con Vbus;
    #                     'physical' = modello dq con Id=0 (ke, R, L)
    vlim_mode: str = "smooth"
    # curva liscia: I = I_max per v <= v1, poi I_max / (1 + ((v - v1) / vc)^n)  (v scalata con Vbus)
    # trazione, forma "envelope" (opzionale, non adottata): inviluppo misurato della corrente
    vlim_mot_form: str = "smooth"
    vlim_mot_vk: float = 2.441    # velocità di ginocchio [m/s]
    vlim_mot_m: float = 3.27      # pendenza della coda: I ~ v^-m
    vlim_mot_p: float = 9.5       # nitidezza del ginocchio
    # trazione, forma "smooth" (attiva): I = I_max / (1 + ((v - v1)/vc)^n)
    vlim_mot_v1: float = 2.0      # trazione: velocità di inizio calo [m/s]
    vlim_mot_vc: float = 0.5      # trazione: velocità (oltre v1) a cui la corrente è dimezzata [m/s]
    vlim_mot_n: float = 2.2       # trazione: ripidità del calo [-]
    vlim_gen_v1: float = 3.4      # frenata: velocità di inizio calo [m/s]
    vlim_gen_vc: float = 1.1      # frenata
    vlim_gen_n: float = 2.0       # frenata
    vbus_filter: float = 5e-3     # filtro misura Vbus usata per il limite di tensione [s]
    vlim_Vref: float = 660.0      # tensione di bus a cui sono riferite le tabelle [V]
    vlim_alpha: float = 1.0       # sensibilità del limite alla Vbus: v_eff = v * (Vref/Vbus)^alpha
    vlim_v_mot: tuple = (0.0, 2.2, 2.6, 3.0, 3.4, 3.8, 4.4)
    vlim_I_mot: tuple = (200.0, 200.0, 150.0, 80.0, 50.0, 35.0, 0.0)
    vlim_v_gen: tuple = (0.0, 3.0, 3.5, 4.0, 4.5, 5.5)
    vlim_I_gen: tuple = (200.0, 200.0, 170.0, 140.0, 100.0, 0.0)
    V_undervolt: float = 330.0    # soglia di sottotensione (Pr 06.065) [V]
    # modello potenza assorbita dal bus (fit su diPowerOutput)
    k_mot: float = 1.125          # P_dc = k_mot * P_mecc (motore)
    k_gen: float = 0.878          # P_dc = k_gen * P_mecc (rigenerazione)
    P_v: float = 109.0            # perdite proporzionali a |v| [W/(m/s)]
    P_0: float = 0.0              # perdite costanti a drive abilitato [W]

    def speed_gains(self) -> tuple[float, float]:
        """Guadagni effettivi (Kp [A/(m/s)], Ki [A/m]) dell'anello di velocità."""
        if self.speed_kp_drive > 0:
            return self.speed_kp_drive * self.kp_scale, self.speed_ki_drive * self.ki_scale
        return self.Kp_vel, self.Ki_vel


@dataclass
class BatteryParams:
    n_series: int = 32            # batterie 6S in serie per stringa
    n_strings: int = 1            # stringhe in parallelo
    capacity_scale: float = 1.0   # scala sull'asse capacità della curva (1200 mAh)
    voc_offset: float = 0.0       # offset sulla curva Voc del pacco [V]
    # resistenza di stringa vs corrente (MATLAB: Corr_str_caratt / Res_str_caratt)
    R_I: tuple = (0, 50, 100, 150, 200, 250, 300, 1000)
    R_val: tuple = (1.4, 1.34, 1.08, 0.95, 0.9, 0.85, 0.85, 0.8)
    R_scale: float = 1.0
    # fattore riduzione R in funzione di dI/dt
    dIdt_thr: tuple = (0, 2000, 3500, 7000, 10000, 12000, 1e12)
    dIdt_fac: tuple = (1.0, 0.95, 0.85, 0.75, 0.75, 0.73, 0.70)
    fc_dIdt: float = 50.0         # filtro derivata corrente [Hz]
    tau_rise: float = 2.0         # risalita lenta della resistenza [s]
    # rami RC (polarizzazione)
    R1: float = 0.35
    tau1: float = 3.0
    R2: float = 0.0               # 0 = ramo disattivato
    tau2: float = 0.02
    # cavi batteria-drive
    cable_len: float = 2.0        # [m]
    cable_area: float = 16e-6     # [m^2]
    rho_cu: float = 1.7e-8
    # bus DC
    V_brake: float = 780.0        # soglia IGBT di frenatura (Pr 06.073/06.074) [V]
    R_brake: float = 8.0          # resistenza di frenatura (Pr 10.061) [ohm], 0 = ideale
    C_dc: float = 0.9e-3          # capacità DC link (stimata dalla precarica) [F]
    R_precharge: float = 47.0     # resistenza di precarica [ohm]
    R_bleed: float = 20e3         # scarica naturale DC link [ohm]
    disc_lim: float = 5e4

    @property
    def R_cable(self) -> float:
        return 2.0 * self.rho_cu * self.cable_len / self.cable_area


@dataclass
class SimParams:
    dt: float = 1e-3              # passo di comunicazione (= campionamento dati) [s]
    n_sub: int = 4                # sotto-passi interni per passo
    V0: float = 740.0             # tensione a vuoto iniziale del pacco [V]
    bus_precharged: bool = True   # True: bus già carico all'avvio


@dataclass
class AMDParams:
    mech: MechParams = field(default_factory=MechParams)
    motor: MotorParams = field(default_factory=MotorParams)
    drive: DriveParams = field(default_factory=DriveParams)
    batt: BatteryParams = field(default_factory=BatteryParams)
    sim: SimParams = field(default_factory=SimParams)

    def to_dict(self) -> dict:
        return asdict(self)

    def update(self, d: dict) -> "AMDParams":
        for grp, vals in d.items():
            obj = getattr(self, grp, None)
            if obj is None:
                continue
            names = {f.name for f in fields(obj)}
            for k, v in vals.items():
                if k in names:
                    if isinstance(getattr(obj, k), tuple):
                        v = tuple(v)
                    setattr(obj, k, v)
        return self

    @classmethod
    def default(cls, use_identified: bool = True) -> "AMDParams":
        p = cls()
        if use_identified and IDENTIFIED_FILE.exists():
            p.update(json.loads(IDENTIFIED_FILE.read_text()))
        return p


def bumper_coefficients(m: MechParams) -> tuple[float, float]:
    """Rigidezza e smorzamento del bumper (molla-smorzatore lineare, forza solo di compressione).

    k è tarato in modo che una massa m_ref che impatta a v_ref generi la forza di picco F_ref
    con il coefficiente di restituzione richiesto (bisezione su simulazione dell'urto).
    """
    e = min(max(m.bumper_e, 1e-3), 0.999)
    zeta = -math.log(e) / math.sqrt(math.pi ** 2 + math.log(e) ** 2)
    if m.bumper_k > 0:
        k = m.bumper_k
        d = m.bumper_d if m.bumper_d > 0 else 2 * zeta * math.sqrt(k * m.bumper_m_ref)
        return k, d

    def peak(k):
        d = 2 * zeta * math.sqrt(k * m.bumper_m_ref)
        x, v, fmax = 0.0, m.bumper_v_ref, 0.0
        h = 2 * math.pi / math.sqrt(k / m.bumper_m_ref) / 4000
        while True:
            f = max(k * x + d * v, 0.0)
            fmax = max(fmax, f)
            v -= f / m.bumper_m_ref * h
            x += v * h
            if x < 0 or (v < 0 and f == 0):
                return fmax

    lo, hi = 1e3, 1e9
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if peak(mid) > m.bumper_F_ref:
            hi = mid
        else:
            lo = mid
    k = math.sqrt(lo * hi)
    d = m.bumper_d if m.bumper_d > 0 else 2 * zeta * math.sqrt(k * m.bumper_m_ref)
    return k, d
