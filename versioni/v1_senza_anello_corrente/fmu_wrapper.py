"""Wrapper FMI 2.0 Co-Simulation del modello AMD (basato su pythonfmu).

Generazione della FMU (dalla cartella amd_fmu):
    python build_fmu.py
che produce 'AMD_Actuator.fmu'. La FMU contiene i sorgenti Python del modello e
richiede un interprete Python con numpy/scipy sulla macchina che la esegue
(limite di pythonfmu, accettabile per il prototipo).

Ingressi:   x_ref [m], enable, precharge_contactor, main_contactor
Uscite:     posizione, velocità, corrente, forze, tensione bus, corrente batteria, SOC, ...
Parametri:  massa, corsa, n° stringhe, tensione iniziale, limite corrente, ...
"""
from __future__ import annotations

import sys
from pathlib import Path

from pythonfmu import Boolean, Fmi2Causality, Fmi2Slave, Fmi2Variability, Real

# i moduli del modello vengono copiati nelle 'resources' della FMU
_RES = Path(__file__).resolve().parent
if str(_RES) not in sys.path:
    sys.path.insert(0, str(_RES))


class AMD_Actuator(Fmi2Slave):
    author = "ISAAC s.r.l."
    description = "AMD: motore lineare + Unidrive M701 (RFC-S) + pacco batterie Ovonic 32x6S"

    # parametri esposti: nome FMU -> (gruppo, attributo)
    PARAMS = {
        "mass": ("mech", "mass"),
        "stroke": ("mech", "stroke"),
        "bumper_e": ("mech", "bumper_e"),
        "mu_coulomb": ("mech", "mu_coulomb"),
        "kf": ("motor", "kf"),
        "I_max": ("drive", "I_max"),
        "Kp_pos": ("drive", "Kp_pos"),
        "K_ff": ("drive", "K_ff"),
        "Kp_speed_Pr03010": ("drive", "speed_kp_drive"),
        "Ki_speed_Pr03011": ("drive", "speed_ki_drive"),
        "T_delay": ("drive", "T_delay"),
        "V_brake": ("batt", "V_brake"),
        "R_scale_batt": ("batt", "R_scale"),
        "V0": ("sim", "V0"),
    }
    OUTPUTS = ["x", "v", "a", "Iq", "I_lim", "F_motor", "F_bumper", "F_actual", "Vbus", "I_batt", "Voc",
               "SOC", "P_drive", "P_brake", "P_mech"]
    FLAGS = ["sat_current", "sat_voltage", "bumper_contact", "drive_enabled"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        res = Path(self.resources) if self.resources else _RES
        if str(res) in sys.path:
            sys.path.remove(str(res))
        sys.path.insert(0, str(res))
        # ogni istanza deve usare i moduli della propria cartella 'resources'
        for mod in ("params", "battery", "amd_model", "data_io", "force_energy"):
            sys.modules.pop(mod, None)
        from params import AMDParams

        self._p = AMDParams.default()
        for name, (grp, attr) in self.PARAMS.items():
            setattr(self, name, float(getattr(getattr(self._p, grp), attr)))
            self.register_variable(Real(name, causality=Fmi2Causality.parameter,
                                        variability=Fmi2Variability.tunable))
        self.n_strings = float(self._p.batt.n_strings)
        self.register_variable(Real("n_strings", causality=Fmi2Causality.parameter,
                                    variability=Fmi2Variability.tunable))
        self.bus_precharged = True
        self.register_variable(Boolean("bus_precharged", causality=Fmi2Causality.parameter,
                                       variability=Fmi2Variability.tunable))

        self.x_ref = 0.0
        self.enable = True
        self.precharge_contactor = False
        self.main_contactor = True
        self.register_variable(Real("x_ref", causality=Fmi2Causality.input))
        self.register_variable(Boolean("enable", causality=Fmi2Causality.input,
                                       variability=Fmi2Variability.discrete))
        self.register_variable(Boolean("precharge_contactor", causality=Fmi2Causality.input,
                                       variability=Fmi2Variability.discrete))
        self.register_variable(Boolean("main_contactor", causality=Fmi2Causality.input,
                                       variability=Fmi2Variability.discrete))

        for o in self.OUTPUTS:
            setattr(self, o, 0.0)
            self.register_variable(Real(o, causality=Fmi2Causality.output))
        for o in self.FLAGS:
            setattr(self, o, False)
            self.register_variable(Boolean(o, causality=Fmi2Causality.output,
                                           variability=Fmi2Variability.discrete))
        self._model = None

    def _build(self):
        from amd_model import AMDModel

        for name, (grp, attr) in self.PARAMS.items():
            setattr(getattr(self._p, grp), attr, float(getattr(self, name)))
        self._p.batt.n_strings = int(round(self.n_strings))
        self._p.sim.bus_precharged = bool(self.bus_precharged)
        self._model = AMDModel(self._p)
        self._model.reset(x0=float(self.x_ref))
        self._xr_prev = float(self.x_ref)
        self.Vbus = float(self._model.batt.Vbus)
        self.Voc = float(self._model.batt.voc(self._model.batt.ah))
        self.SOC = float(self._model.batt.soc())
        self.x = float(self.x_ref)
        if not self.bus_precharged:
            self.main_contactor = False

    def exit_initialization_mode(self):
        self._build()

    def do_step(self, current_time: float, step_size: float) -> bool:
        if self._model is None:
            self._build()
        m = self._model
        dt = m.p.sim.dt
        n = max(int(round(step_size / dt)), 1)
        if abs(n * dt - step_size) > 1e-9:
            # passo diverso da 1 ms: adatto il passo interno del modello
            m.p.sim.dt = step_size / n
            m.h = m.p.sim.dt / max(int(m.p.sim.n_sub), 1)
        xr0, xr1 = self._xr_prev, float(self.x_ref)
        self._xr_prev = xr1
        for k in range(n):
            # riferimento interpolato linearmente all'interno del passo di comunicazione
            o = m.do_step(xr0 + (xr1 - xr0) * (k + 1) / n, enable=bool(self.enable),
                          main_contactor=bool(self.main_contactor),
                          precharge_contactor=bool(self.precharge_contactor))
        for k in self.OUTPUTS:
            setattr(self, k, float(o[k]))
        for k in self.FLAGS:
            setattr(self, k, bool(o[k] > 0.5))
        return True
