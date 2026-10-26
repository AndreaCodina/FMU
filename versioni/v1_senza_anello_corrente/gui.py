"""Interfaccia interattiva per il modello AMD.

Avvio:  python gui.py

Schede:
  * Simulazione  - profilo di SetPosition generato (seno, sweep, gradino, rampa, da file)
  * Validazione  - confronto modello / test sperimentale selezionato
  * FRF          - confronto FRF sperimentale / modello lineare per le prove sweep
Il pannello a sinistra contiene tutti i parametri del modello (massa e corsa in testa).
"""
from __future__ import annotations

import copy
import json
import sys
import traceback
from dataclasses import fields
from pathlib import Path

import numpy as np
from scipy import signal
import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402

import references as refs  # noqa: E402
from amd_model import AMDModel  # noqa: E402
from data_io import DEFAULT_DATA_DIR, list_tests, mass_from_path, load_twincat_csv  # noqa: E402
from params import MOTOR_PRESETS, AMDParams, bumper_coefficients  # noqa: E402
from validation import energy_metric, frf, metrics, model_frf_linear, prepare_test, run_test  # noqa: E402

# etichette e range dei parametri mostrati nel pannello: (gruppo, nome) -> (etichetta, min, max, decimali)
PARAM_UI = {
    ("mech", "mass"): ("Massa mobile [kg]", 50, 5000, 1),
    ("mech", "stroke"): ("Semicorsa / contatto bumper [m]", 0.05, 3, 3),
    ("mech", "mu_coulomb"): ("Coeff. attrito guide μ [-]", 0, 0.2, 4),
    ("mech", "c_visc"): ("Attrito viscoso [N·s/m]", 0, 20000, 0),
    ("mech", "bumper_e"): ("Bumper: coeff. restituzione [-]", 0.01, 0.99, 2),
    ("mech", "bumper_F_ref"): ("Bumper: forza picco di taratura [N]", 1e3, 1e6, 0),
    ("mech", "bumper_v_ref"): ("Bumper: velocità di taratura [m/s]", 0.1, 10, 2),
    ("mech", "bumper_m_ref"): ("Bumper: massa di taratura [kg]", 50, 5000, 0),
    ("motor", "kf"): ("kf bassa corrente [N/A]", 10, 500, 1),
    ("motor", "R_ph"): ("R fase [Ω]", 0.01, 5, 3),
    ("motor", "L_ph"): ("L fase [H]", 1e-4, 0.05, 5),
    ("motor", "ke_ph"): ("ke fase rms [V/(m/s)] (0=kf/3)", 0, 300, 1),
    ("drive", "I_max"): ("Limite corrente [A]", 1, 400, 1),
    ("drive", "Kp_pos"): ("Kp posizione AMC [1/s]", 0, 5000, 2),
    ("drive", "pos_speed_clamp"): ("Limite uscita P posizione [m/s]", 0, 10, 5),
    ("drive", "i_filter"): ("Filtro riferimento corrente [s]", 0, 0.05, 4),
    ("drive", "v_max"): ("Velocità massima [m/s]", 0.1, 10, 2),
    ("drive", "K_ff"): ("Feed-forward velocità [-]", 0, 2, 4),
    ("drive", "speed_kp_drive"): ("Kp velocità Pr 03.010 [s/rad]", 0, 10, 4),
    ("drive", "speed_ki_drive"): ("Ki velocità Pr 03.011 [s²/rad]", 0, 1000, 3),
    ("drive", "T_delay"): ("Ritardo anello [s]", 0, 0.05, 5),
    ("drive", "T_ref"): ("Ritardo riferimento [s]", 0, 0.05, 4),
    ("drive", "aw_Tt"): ("Anti-windup: costante Tt [s]", 0.001, 1, 4),
    ("drive", "v_ref_band"): ("Banda v_ref attorno a v riferimento [m/s]", 0, 10, 3),
    ("drive", "vlim_Vref"): ("Limite tensione: V riferimento [V]", 100, 1000, 0),
    ("drive", "k_mot"): ("Potenza: fattore motore [-]", 0.5, 2, 3),
    ("drive", "k_gen"): ("Potenza: fattore rigenerazione [-]", 0, 1.5, 3),
    ("drive", "P_v"): ("Potenza: perdite ∝ |v| [W·s/m]", 0, 1e4, 1),
    ("drive", "V_undervolt"): ("Soglia sottotensione [V]", 0, 800, 0),
    ("batt", "n_strings"): ("Stringhe in parallelo [-]", 1, 8, 0),
    ("batt", "n_series"): ("Batterie 6S in serie [-]", 1, 64, 0),
    ("batt", "R_scale"): ("Scala resistenza interna [-]", 0.1, 5, 3),
    ("batt", "R1"): ("R ramo RC lento [Ω]", 0, 5, 3),
    ("batt", "tau1"): ("τ ramo RC lento [s]", 0.01, 100, 2),
    ("batt", "cable_len"): ("Lunghezza cavi batteria-drive [m]", 0, 50, 2),
    ("batt", "cable_area"): ("Sezione cavi [m²]", 1e-6, 1e-3, 7),
    ("batt", "V_brake"): ("Soglia chopper frenatura [V]", 600, 900, 0),
    ("batt", "R_brake"): ("Resistenza di frenatura [Ω]", 0, 100, 2),
    ("batt", "C_dc"): ("Capacità DC link [F]", 1e-5, 0.1, 5),
    ("batt", "R_precharge"): ("Resistenza precarica [Ω]", 1, 1000, 1),
    ("sim", "V0"): ("Tensione iniziale pacco (a vuoto) [V]", 400, 820, 1),
    ("sim", "n_sub"): ("Sotto-passi per ms [-]", 1, 20, 0),
}
COMBO_UI = {
    ("drive", "vlim_mode"): ("Limite di tensione", ["table", "physical"]),
    ("drive", "antiwindup"): ("Anti-windup", ["tracking", "backcalc", "conditional"]),
}
GROUP_TITLES = {"mech": "Meccanica / bumper", "motor": "Motore", "drive": "Drive Unidrive M701",
                "batt": "Batterie / bus DC", "sim": "Simulazione"}


# ---------------------------------------------------------------------------- utility Qt
class Worker(QtCore.QThread):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(float)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(self.progress.emit))
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())


class PlotArea(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(10, 8), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas)

    def axes(self, nrows, ncols, sharex=True):
        self.fig.clear()
        ax = self.fig.subplots(nrows, ncols, sharex=sharex, squeeze=False)
        for a in ax.flat:
            a.grid(alpha=0.3)
        return ax

    def draw(self):
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------- pannello parametri
class ParamPanel(QtWidgets.QScrollArea):
    changed = QtCore.Signal()

    def __init__(self, params: AMDParams):
        super().__init__()
        self.setWidgetResizable(True)
        self.params = params
        self.widgets = {}
        self.touched = set()   # parametri modificati dall'utente (gli altri restano a piena precisione)
        inner = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(inner)

        motor_box = QtWidgets.QHBoxLayout()
        motor_box.addWidget(QtWidgets.QLabel("Motore:"))
        self.motor_combo = QtWidgets.QComboBox()
        self.motor_combo.addItems(list(MOTOR_PRESETS))
        self.motor_combo.setCurrentText(params.motor.name)
        self.motor_combo.currentTextChanged.connect(self._motor_preset)
        motor_box.addWidget(self.motor_combo, 1)
        lay.addLayout(motor_box)

        for grp in ("mech", "motor", "drive", "batt", "sim"):
            box = QtWidgets.QGroupBox(GROUP_TITLES[grp])
            form = QtWidgets.QFormLayout(box)
            obj = getattr(params, grp)
            for f in fields(obj):
                key = (grp, f.name)
                if key in PARAM_UI:
                    label, lo, hi, dec = PARAM_UI[key]
                    if isinstance(getattr(obj, f.name), int) and not isinstance(getattr(obj, f.name), bool):
                        w = QtWidgets.QSpinBox()
                        w.setRange(int(lo), int(hi))
                    else:
                        w = QtWidgets.QDoubleSpinBox()
                        w.setDecimals(dec)
                        w.setRange(lo, hi)
                        w.setSingleStep(10 ** -dec if dec > 0 else 1)
                    w.setKeyboardTracking(False)
                    w.valueChanged.connect(self.changed)
                elif key in COMBO_UI:
                    label, items = COMBO_UI[key]
                    w = QtWidgets.QComboBox()
                    w.addItems(items)
                    w.currentTextChanged.connect(self.changed)
                elif key == ("sim", "bus_precharged"):
                    label = "Bus già carico all'avvio"
                    w = QtWidgets.QCheckBox()
                    w.toggled.connect(self.changed)
                else:
                    continue
                form.addRow(label, w)
                self.widgets[key] = w
                for sig in ("valueChanged", "currentTextChanged", "toggled"):
                    if hasattr(w, sig):
                        getattr(w, sig).connect(lambda *_, k=key: self.touched.add(k))
            if grp == "mech":
                self.bumper_lbl = QtWidgets.QLabel()
                self.bumper_lbl.setStyleSheet("color: gray")
                form.addRow(self.bumper_lbl)
            lay.addWidget(box)

        btns = QtWidgets.QHBoxLayout()
        for txt, fn in (("Default", self._defaults), ("Salva…", self._save), ("Carica…", self._load)):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            btns.addWidget(b)
        lay.addLayout(btns)
        lay.addStretch(1)
        self.setWidget(inner)
        self.set_params(params)
        self.changed.connect(self._update_bumper_label)

    def set_params(self, p: AMDParams):
        self.params = p
        self.touched = set()
        for (grp, name), w in self.widgets.items():
            v = getattr(getattr(p, grp), name)
            w.blockSignals(True)
            if isinstance(w, QtWidgets.QComboBox):
                w.setCurrentText(str(v))
            elif isinstance(w, QtWidgets.QCheckBox):
                w.setChecked(bool(v))
            else:
                w.setValue(v)
            w.blockSignals(False)
        self._update_bumper_label()

    def get_params(self) -> AMDParams:
        p = copy.deepcopy(self.params)
        for (grp, name), w in self.widgets.items():
            if (grp, name) not in self.touched:
                continue
            obj = getattr(p, grp)
            if isinstance(w, QtWidgets.QComboBox):
                setattr(obj, name, w.currentText())
            elif isinstance(w, QtWidgets.QCheckBox):
                setattr(obj, name, w.isChecked())
            elif isinstance(w, QtWidgets.QSpinBox):
                setattr(obj, name, int(w.value()))
            else:
                setattr(obj, name, float(w.value()))
        p.motor.name = self.motor_combo.currentText()
        return p

    def set_value(self, grp, name, value):
        w = self.widgets.get((grp, name))
        if w is not None:
            w.setValue(value)

    def _update_bumper_label(self):
        try:
            k, d = bumper_coefficients(self.get_params().mech)
            self.bumper_lbl.setText(f"Bumper calcolato: k = {k:,.0f} N/m, c = {d:,.0f} N·s/m")
        except Exception:  # noqa: BLE001
            pass

    def _motor_preset(self, name):
        p = self.get_params()
        p.motor = type(p.motor)(**{**p.motor.__dict__, **MOTOR_PRESETS[name]})
        self.set_params(p)
        self.changed.emit()

    def _defaults(self):
        self.set_params(AMDParams.default())
        self.motor_combo.setCurrentText(self.params.motor.name)
        self.changed.emit()

    def _save(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Salva parametri", "parametri.json", "JSON (*.json)")
        if fn:
            Path(fn).write_text(json.dumps(self.get_params().to_dict(), indent=2))

    def _load(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Carica parametri", "", "JSON (*.json)")
        if fn:
            p = AMDParams.default(use_identified=False).update(json.loads(Path(fn).read_text()))
            self.set_params(p)
            self.changed.emit()


# ---------------------------------------------------------------------------- scheda simulazione
PROFILES = ["Seno in forza", "Sweep in forza", "Seno (spostamento)", "Sweep (spostamento)", "Gradino",
            "Rampa (urto bumper)", "Da file CSV"]
# campi visibili per ciascun profilo
PROFILE_FIELDS = {
    "Seno in forza": ("famp", "freq", "dur", "ramp"),
    "Sweep in forza": ("famp", "freq", "f1", "dur", "ramp"),
    "Seno (spostamento)": ("amp", "freq", "dur", "ramp"),
    "Sweep (spostamento)": ("amp", "freq", "f1", "dur"),
    "Gradino": ("amp", "dur"),
    "Rampa (urto bumper)": ("amp", "speed", "dur"),
    "Da file CSV": ("file", "unit", "dt1"),
}


class SimTab(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.file = None
        self.last = None
        self._file_data = None
        lay = QtWidgets.QVBoxLayout(self)

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("<b>Profilo SetPosition</b>"))
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(PROFILES)
        self.kind.currentTextChanged.connect(self._update_fields)
        row1.addWidget(self.kind)
        self.fields = {}

        def add(key, label, w):
            box = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(box)
            hl.setContentsMargins(6, 0, 6, 0)
            if label:
                hl.addWidget(QtWidgets.QLabel(label))
            hl.addWidget(w)
            row1.addWidget(box)
            self.fields[key] = box
            return w

        self.famp = add("famp", "Ampiezza forza", self._spin(15.0, 0.1, 60, 2, " kN"))
        self.amp = add("amp", "Ampiezza / pos. finale", self._spin(0.4, 0.001, 3, 3, " m"))
        self.freq = add("freq", "Frequenza (f0)", self._spin(1.0, 0.05, 50, 2, " Hz"))
        self.f1 = add("f1", "f1", self._spin(10.0, 0.05, 50, 2, " Hz"))
        self.speed = add("speed", "Velocità", self._spin(3.0, 0.1, 6, 2, " m/s"))
        self.dur = add("dur", "Durata", self._spin(10.0, 0.1, 600, 1, " s"))
        self.ramp = add("ramp", "Rampa in/out", self._spin(1.0, 0.0, 30, 1, " s"))
        self.file_btn = add("file", "", QtWidgets.QPushButton("Carica CSV…"))
        self.file_btn.clicked.connect(self._pick_file)
        self.unit = QtWidgets.QComboBox()
        self.unit.addItems(["m", "mm"])
        add("unit", "Unità posizione", self.unit)
        self.dt1 = add("dt1", "Passo (se 1 colonna)", self._spin(1.0, 0.01, 1000, 2, " ms"))
        self.unit.currentTextChanged.connect(lambda *_: self._load_file())
        self.dt1.valueChanged.connect(lambda *_: self._load_file())
        row1.addStretch(1)
        lay.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        self.file_lbl = QtWidgets.QLabel("")
        self.file_lbl.setStyleSheet("color: gray")
        self.file_lbl.setWordWrap(True)
        row2.addWidget(self.file_lbl, 1)
        self.precharge = QtWidgets.QCheckBox("Simula precarica bus (1 s)")
        row2.addWidget(self.precharge)
        self.run_btn = QtWidgets.QPushButton("▶ Simula")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 4px 18px")
        self.run_btn.clicked.connect(self.run)
        row2.addWidget(self.run_btn)
        self.exp_btn = QtWidgets.QPushButton("Esporta risultati CSV…")
        self.exp_btn.clicked.connect(self._export)
        self.exp_btn.setEnabled(False)
        row2.addWidget(self.exp_btn)
        lay.addLayout(row2)

        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.plot = PlotArea()
        lay.addWidget(self.plot, 1)
        self.auto = QtWidgets.QCheckBox("Ricalcola automaticamente al cambio parametri")
        lay.addWidget(self.auto)
        self._update_fields(self.kind.currentText())

    @staticmethod
    def _spin(v, lo, hi, dec, suffix=""):
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(dec)
        w.setValue(v)
        w.setSuffix(suffix)
        w.setKeyboardTracking(False)
        return w

    def _update_fields(self, kind):
        show = PROFILE_FIELDS.get(kind, ())
        for k, box in self.fields.items():
            box.setVisible(k in show)
        self.file_lbl.setVisible(kind == "Da file CSV")

    def _pick_file(self):
        start = str(Path(self.file).parent) if self.file else str(DEFAULT_DATA_DIR)
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Set position da CSV (file di test TwinCAT o CSV tempo;posizione)", start,
            "CSV / testo (*.csv *.txt);;Tutti i file (*)")
        if fn:
            self.load_file(fn)

    def load_file(self, fn):
        self.file = fn
        self.kind.setCurrentText("Da file CSV")
        self._load_file()

    def _load_file(self):
        if not self.file:
            return
        try:
            t, xr, meas = refs.from_csv(self.file, self.unit.currentText(), self.dt1.value() * 1e-3)
        except Exception as e:  # noqa: BLE001
            self._file_data = None
            self.file_lbl.setText(f"Errore nel file: {e}")
            return
        self._file_data = (t, xr, meas)
        kind = "file di test TwinCAT (rSetPosition)" if refs.is_twincat_csv(self.file) else "CSV"
        extra = " + posizione misurata per confronto" if "x" in meas else ""
        self.file_lbl.setText(f"{Path(self.file).name} — {kind}{extra}: {len(t)} campioni a 1 ms, "
                              f"durata {t[-1]:.2f} s, SetPosition da {xr.min():+.3f} a {xr.max():+.3f} m")
        ax = self.plot.axes(1, 1)
        ax[0, 0].plot(t, xr, "k", lw=1, label="SetPosition dal file")
        if "x" in meas:
            ax[0, 0].plot(t, meas["x"], lw=0.8, alpha=0.7, label="Posizione misurata")
        ax[0, 0].set_xlabel("Tempo [s]")
        ax[0, 0].set_ylabel("Posizione [m]")
        ax[0, 0].legend(loc="upper right", fontsize=8)
        ax[0, 0].set_title("Anteprima — premi ▶ Simula")
        self.plot.draw()

    def reference(self):
        k = self.kind.currentText()
        a, f, T = self.amp.value(), self.freq.value(), self.dur.value()
        mass = self.main.params.get_params().mech.mass
        if k == "Seno in forza":
            t, x, F = refs.force_sine(self.famp.value() * 1e3, f, T, mass, t_ramp=self.ramp.value())
            return t, x, {"F_target": F}
        if k == "Sweep in forza":
            t, x, F = refs.force_sweep(self.famp.value() * 1e3, f, self.f1.value(), T, mass,
                                       t_ramp=self.ramp.value())
            return t, x, {"F_target": F}
        if k == "Seno (spostamento)":
            t, x = refs.sine(a, f, T, t_ramp=self.ramp.value())
            return t, x, {}
        if k == "Sweep (spostamento)":
            t, x = refs.sweep(a, f, self.f1.value(), T)
            return t, x, {}
        if k == "Gradino":
            t, x = refs.step(a, T)
            return t, x, {}
        if k.startswith("Rampa"):
            t, x = refs.ramp_to(a, self.speed.value(), T)
            return t, x, {}
        if self._file_data is None:
            raise ValueError("Carica prima un file CSV con il pulsante «Carica CSV…»")
        return self._file_data

    def run(self):
        p = self.main.params.get_params()
        try:
            t, xr, meas = self.reference()
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Riferimento", str(e))
            return
        pre_on = self.precharge.isChecked()
        x0 = float(meas["x"][0]) if "x" in meas else float(xr[0])

        def job(progress):
            m = AMDModel(p)
            if pre_on:
                # sequenza: precarica chiusa a 0.1 s, contattore principale a 0.6 s, drive abilitato a 0.8 s
                n_pre = int(1.0 / p.sim.dt)
                xr_full = np.r_[np.full(n_pre, xr[0]), xr]
                tt = np.arange(len(xr_full)) * p.sim.dt
                pre = (tt >= 0.1) & (tt < 0.65)
                main = tt >= 0.6
                en = tt >= 0.8
                s = m.simulate(xr_full, enable=en, main=main, pre=pre, precharged=False, x0=x0,
                               progress=progress)
                s["_meas"] = {k: np.r_[np.full(n_pre, v[0]), v] for k, v in meas.items()}
            else:
                s = m.simulate(xr, x0=x0, progress=progress)
                s["_meas"] = meas
            return s

        self.main.run_job(job, self.show)

    def show(self, s):
        self.last = s
        self.exp_btn.setEnabled(True)
        t = s["t"]
        meas = s.get("_meas", {})
        ax = self.plot.axes(4, 2)
        a = ax[0, 0]
        a.plot(t, s["x_ref"], "k--", lw=1, label="SetPosition")
        if "x" in meas:
            a.plot(t, meas["x"][:len(t)], lw=1, alpha=0.8, label="Posizione misurata")
        a.plot(t, s["x"], lw=1.2, label="ActualPosition (modello)")
        stroke = self.main.params.get_params().mech.stroke
        a.axhline(stroke, color="r", lw=.6, ls=":")
        a.axhline(-stroke, color="r", lw=.6, ls=":", label="bumper")
        a.set_ylabel("Posizione [m]")
        a.legend(loc="upper right", fontsize=8)
        a = ax[0, 1]
        a.plot(t, (s["x_ref"] - s["x"]) * 1e3, lw=1, label="SetPosition − modello")
        if "x" in meas:
            a.plot(t, (s["x"] - meas["x"][:len(t)]) * 1e3, lw=1, label="modello − misura")
        a.set_ylabel("Errore [mm]")
        a.legend(loc="upper right", fontsize=8)
        a = ax[1, 0]
        a.plot(t, s["Iq"], lw=1, label="Iq")
        a.plot(t, s["I_lim"], "r:", lw=.8, label="limite (corrente / tensione)")
        a.plot(t, -s["I_lim"], "r:", lw=.8)
        a.set_ylabel("Corrente [A]")
        a.legend(loc="upper right", fontsize=8)
        a = ax[1, 1]
        if "F_target" in meas:
            a.plot(t, meas["F_target"][:len(t)] / 1e3, "k--", lw=0.9, label="Forza richiesta")
        a.plot(t, s["F_actual"] / 1e3, lw=1, label="Forza scaricata (come rActualForce)")
        if np.abs(s["F_bumper"]).max() > 0:
            a.plot(t, s["F_bumper"] / 1e3, lw=1, label="Bumper")
        a.set_ylabel("Forza [kN]")
        a.legend(loc="upper right", fontsize=8)
        ax[2, 0].plot(t, s["v"], lw=1)
        ax[2, 0].set_ylabel("Velocità [m/s]")
        a = ax[2, 1]
        a.plot(t, s["Vbus"], lw=1, label="Vbus")
        a.plot(t, s["Voc"], lw=.8, label="Voc pacco")
        a.set_ylabel("Tensione [V]")
        a.legend(loc="lower right", fontsize=8)
        a = ax[3, 0]
        a.plot(t, s["P_drive"] / 1e3, lw=1, label="Drive (bus DC)")
        a.plot(t, s["P_brake"] / 1e3, lw=1, label="Res. frenatura")
        a.set_ylabel("Potenza [kW]")
        a.legend(loc="upper right", fontsize=8)
        a = ax[3, 1]
        a.plot(t, s["I_batt"], lw=1)
        a.set_ylabel("Corrente batteria [A]")
        a2 = a.twinx()
        a2.plot(t, 100 * s["SOC"], "g", lw=.8)
        a2.set_ylabel("SOC [%]", color="g")
        for a in ax[-1]:
            a.set_xlabel("Tempo [s]")
        self.plot.draw()
        dt = t[1] - t[0] if len(t) > 1 else 1e-3
        e_brake = s["P_brake"].sum() * dt
        err = np.abs(s["x_ref"] - s["x"]).max() * 1e3
        txt = (f"max |x| {np.abs(s['x']).max():.3f} m · errore max vs SetPosition {err:.1f} mm · "
               f"max |v| {np.abs(s['v']).max():.2f} m/s · max |F| {np.abs(s['F_actual']).max() / 1e3:.1f} kN · "
               f"Vbus min/max {s['Vbus'].min():.0f}/{s['Vbus'].max():.0f} V · "
               f"saturazione corrente {100 * s['sat_current'].mean():.1f}% / tensione "
               f"{100 * s['sat_voltage'].mean():.1f}% · energia frenatura {e_brake / 1e3:.2f} kJ")
        if np.abs(s["F_bumper"]).max() > 0:
            txt += f" · <b>URTO BUMPER: {np.abs(s['F_bumper']).max() / 1e3:.0f} kN</b>"
        # fattibilità del profilo richiesto: forza (M·a_ref) e velocità rispetto ai limiti del sistema
        p = self.main.params.get_params()
        b, a_ = signal.butter(2, 30 * 2 * dt)
        v_ref = np.gradient(s["x_ref"], dt)
        acc_ref = signal.filtfilt(b, a_, np.gradient(v_ref, dt)) if len(t) > 20 else np.gradient(v_ref, dt)
        F_req = p.mech.m_tot * np.abs(acc_ref).max()
        F_av = p.motor.force(p.drive.I_max)
        # velocità oltre la quale la corrente disponibile in trazione scende sotto il 60%
        v_lim = next((v for v, i in zip(p.drive.vlim_v_mot, p.drive.vlim_I_mot) if i < p.drive.I_max * 0.6),
                     p.drive.v_max)
        feas = (f"<br>Profilo richiesto: forza max {F_req / 1e3:.1f} kN (disponibile ≈ {F_av / 1e3:.1f} kN a "
                f"{p.drive.I_max:.0f} A), velocità max {np.abs(v_ref).max():.2f} m/s")
        x_req = np.abs(s["x_ref"]).max()
        feas += f", spostamento max {x_req:.3f} m (corsa ±{p.mech.stroke:.2f} m)"
        if F_req > F_av or np.abs(v_ref).max() > v_lim or x_req > p.mech.stroke:
            feas = ("<span style='color:#c0392b'><b>" + feas[4:] + " → oltre i limiti dell'attuatore: "
                    "il sistema non può seguirlo (saturazione)</b></span>")
            feas = "<br>" + feas
        txt += feas
        if "F_target" in meas:
            Ft = meas["F_target"][:len(t)]
            Fa = s["F_actual"]
            # ritardo della forza ottenuta (correlazione incrociata) e rapporto di ampiezza (valori RMS)
            lags = np.arange(0, int(0.05 / dt))
            cc = [np.dot(Ft[:len(Ft) - L], Fa[L:]) for L in lags]
            lag_ms = lags[int(np.argmax(cc))] * dt * 1e3
            ratio = np.std(Fa) / max(np.std(Ft), 1e-9)
            txt += (f"<br>Forza: richiesta max {np.abs(Ft).max() / 1e3:.1f} kN, ottenuta max "
                    f"{np.abs(Fa).max() / 1e3:.1f} kN · ampiezza ottenuta/richiesta {100 * ratio:.1f}% · "
                    f"ritardo {lag_ms:.0f} ms (≈13 ms filtro di calcolo della forza del PLC + ≈5 ms drive e comunicazione)")
        if "x" in meas:
            e = s["x"] - meas["x"][:len(t)]
            txt += (f"<br>Confronto con la posizione misurata: RMSE {np.sqrt(np.mean(e ** 2)) * 1e3:.2f} mm, "
                    f"max {np.abs(e).max() * 1e3:.1f} mm")
        self.info.setText(txt)

    def _export(self):
        if self.last is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Esporta risultati", "risultati_simulazione.csv",
                                                      "CSV (*.csv)")
        if fn:
            self.export_csv(fn)
            self.main.statusBar().showMessage(f"Salvato {fn}", 4000)

    def export_csv(self, fn):
        import pandas as pd
        s = self.last
        cols = ["t", "x_ref", "x", "v", "a", "Iq", "I_lim", "F_actual", "F_motor", "F_bumper", "Vbus",
                "Voc", "I_batt", "SOC", "P_drive", "P_brake", "sat_current", "sat_voltage", "bumper_contact"]
        df = pd.DataFrame({c: s[c] for c in cols})
        if "x" in s.get("_meas", {}):
            df.insert(3, "x_misurata", s["_meas"]["x"][:len(df)])
        df.to_csv(fn, sep=";", decimal=",", index=False)


# ---------------------------------------------------------------------------- scheda validazione
class ValTab(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Test"])
        self.tree.itemDoubleClicked.connect(lambda *_: self.run())
        left.addWidget(self.tree, 1)
        self.use_test = QtWidgets.QCheckBox("Imposta massa e n° stringhe dal test")
        self.use_test.setChecked(True)
        left.addWidget(self.use_test)
        b = QtWidgets.QPushButton("▶ Confronta modello / test")
        b.clicked.connect(self.run)
        left.addWidget(b)
        dir_btn = QtWidgets.QPushButton("Cartella dati…")
        dir_btn.clicked.connect(self._pick_dir)
        left.addWidget(dir_btn)
        w = QtWidgets.QWidget()
        w.setLayout(left)
        w.setMaximumWidth(380)
        lay.addWidget(w)
        right = QtWidgets.QVBoxLayout()
        self.info = QtWidgets.QLabel("Doppio click su un test per confrontarlo con il modello")
        self.info.setWordWrap(True)
        right.addWidget(self.info)
        self.plot = PlotArea()
        right.addWidget(self.plot, 1)
        lay.addLayout(right, 1)
        self.data_dir = DEFAULT_DATA_DIR
        self.populate()

    def _pick_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Cartella '02_File FMU per claude'", str(self.data_dir))
        if d:
            self.data_dir = Path(d)
            self.populate()
            self.main.frf_tab.populate(self.data_dir)

    def populate(self):
        self.tree.clear()
        try:
            tests = list_tests(self.data_dir)
        except Exception:  # noqa: BLE001
            return
        for cat in ("CVT", "Sismi", "FRF"):
            top = QtWidgets.QTreeWidgetItem([cat])
            by_mass = {}
            for f in tests[cat]:
                m = mass_from_path(f)
                if m not in by_mass:
                    by_mass[m] = QtWidgets.QTreeWidgetItem([f"{m:.0f} kg"])
                    top.addChild(by_mass[m])
                it = QtWidgets.QTreeWidgetItem([f.name])
                it.setData(0, QtCore.Qt.UserRole, str(f))
                by_mass[m].addChild(it)
            self.tree.addTopLevelItem(top)

    def run(self):
        it = self.tree.currentItem()
        path = it.data(0, QtCore.Qt.UserRole) if it else None
        if not path:
            return
        p = self.main.params.get_params()
        use_test = self.use_test.isChecked()

        def job(progress):
            tc = prepare_test(path)
            if not use_test:
                tc.mass, tc.n_strings = p.mech.mass, p.batt.n_strings
            sim = run_test(tc, p, progress=progress)
            return tc, sim

        def show(res):
            tc, sim = res
            if use_test:
                self.main.params.set_value("mech", "mass", tc.mass)
                self.main.params.set_value("batt", "n_strings", tc.n_strings)
            m = metrics(tc, sim)
            em = energy_metric(tc, sim)
            ax = self.plot.axes(6, 1)
            for a, k, lab, sc in zip(ax[:, 0], ("x", "Iq", "F_actual", "Vbus", "P_drive"),
                                     ("Posizione [m]", "Iq [A]", "Forza scaricata [kN]", "Vbus [V]",
                                      "Potenza drive [kW]"),
                                     (1, 1, 1e-3, 1, 1e-3)):
                a.plot(tc.t, tc.meas[k] * sc, lw=1, label="Misura")
                a.plot(tc.t, sim[k] * sc, lw=1, label="Modello")
                a.set_ylabel(lab)
            ax[5, 0].plot(tc.t, em["E_meas"] / 1e3, lw=1, label="Reale (rActualForce)")
            ax[5, 0].plot(tc.t, em["E_sim"] / 1e3, lw=1, label="Modello")
            ax[5, 0].set_ylabel("Energia ∫vG·F [kJ]")
            ax[5, 0].legend(loc="upper left", fontsize=8)
            ax[0, 0].plot(tc.t, tc.x_ref, "k--", lw=.7, label="SetPosition")
            ax[0, 0].legend(loc="upper right", fontsize=8)
            ax[-1, 0].set_xlabel("Tempo [s]")
            self.plot.draw()
            self.info.setText(
                f"<b>{tc.name}</b> — massa {tc.mass:.0f} kg, stringhe {tc.n_strings}, V0 {tc.V0:.0f} V<br>"
                f"RMSE posizione {m['x']['rmse'] * 1e3:.2f} mm ({m['x']['nrmse'] * 100:.2f}% del range) — "
                f"Iq {m['Iq']['rmse']:.1f} A — Vbus {m['Vbus']['rmse']:.1f} V — "
                f"Potenza {m['P_drive']['rmse'] / 1e3:.2f} kW — Forza {em['F_rmse'] / 1e3:.2f} kN<br>"
                f"<b>Energia finale: reale {em['E_meas'][-1] / 1e3:.3f} kJ, modello {em['E_sim'][-1] / 1e3:.3f} kJ "
                f"→ scarto {em['err_pct']:+.2f}%</b>")

        self.main.run_job(job, show)


# ---------------------------------------------------------------------------- scheda FRF
class FrfTab(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QtWidgets.QVBoxLayout(self)
        ctl = QtWidgets.QHBoxLayout()
        self.combo = QtWidgets.QComboBox()
        ctl.addWidget(QtWidgets.QLabel("Prova sweep:"))
        ctl.addWidget(self.combo, 1)
        self.sim_chk = QtWidgets.QCheckBox("anche FRF da simulazione nel tempo (non lineare)")
        self.sim_chk.setChecked(True)
        ctl.addWidget(self.sim_chk)
        b = QtWidgets.QPushButton("▶ Calcola FRF")
        b.clicked.connect(self.run)
        ctl.addWidget(b)
        lay.addLayout(ctl)
        self.plot = PlotArea()
        lay.addWidget(self.plot, 1)
        self.populate(DEFAULT_DATA_DIR)

    def populate(self, data_dir):
        self.combo.clear()
        try:
            for f in list_tests(data_dir)["FRF"]:
                self.combo.addItem(f.name, str(f))
        except Exception:  # noqa: BLE001
            pass

    def run(self):
        path = self.combo.currentData()
        if not path:
            return
        p = self.main.params.get_params()
        do_sim = self.sim_chk.isChecked()

        def job(progress):
            d = load_twincat_csv(path)
            sw = d["bSweepActive"].values > 0.5
            r, x = d["rSetPosition"].values[sw], d["rActualPosition"].values[sw]
            f, H, coh = frf(r, x)
            mass = mass_from_path(path)
            Hm = model_frf_linear(p, mass, f)
            Hs = None
            if do_sim:
                pp = copy.deepcopy(p)
                pp.mech.mass = mass
                s = AMDModel(pp).simulate(r, x0=float(x[0]), progress=progress)
                _, Hs, _ = frf(r, s["x"])
            return f, H, coh, Hm, Hs, mass

        def show(res):
            f, H, coh, Hm, Hs, mass = res
            k = (f > 0.2) & (f < 40)
            ax = self.plot.axes(3, 1)
            ax[0, 0].plot(f[k], np.abs(H[k]), label="Sperimentale")
            ax[0, 0].plot(f[k], np.abs(Hm[k]), ":", label="Modello lineare (senza limiti)")
            ax[1, 0].plot(f[k], np.angle(H[k], deg=True))
            ax[1, 0].plot(f[k], np.angle(Hm[k], deg=True), ":")
            if Hs is not None:
                ax[0, 0].plot(f[k], np.abs(Hs[k]), "--", label="Simulazione nel tempo")
                ax[1, 0].plot(f[k], np.angle(Hs[k], deg=True), "--")
            ax[2, 0].plot(f[k], coh[k], "k", lw=.8)
            ax[0, 0].set_ylabel("|ActualPos / SetPos| [-]")
            ax[1, 0].set_ylabel("Fase [°]")
            ax[2, 0].set_ylabel("Coerenza [-]")
            ax[2, 0].set_xlabel("Frequenza [Hz]")
            ax[0, 0].set_title(f"FRF massa {mass:.0f} kg")
            ax[0, 0].legend()
            for a in ax[:, 0]:
                a.set_xscale("log")
            self.plot.draw()

        self.main.run_job(job, show)


# ---------------------------------------------------------------------------- finestra principale
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AMD — modello attuatore (motore lineare + Unidrive M701 + batterie)")
        self.resize(1600, 950)
        self.params = ParamPanel(AMDParams.default())
        self.params.setMinimumWidth(380)
        self.tabs = QtWidgets.QTabWidget()
        self.sim_tab = SimTab(self)
        self.val_tab = ValTab(self)
        self.frf_tab = FrfTab(self)
        self.tabs.addTab(self.sim_tab, "Simulazione")
        self.tabs.addTab(self.val_tab, "Validazione test")
        self.tabs.addTab(self.frf_tab, "FRF")
        split = QtWidgets.QSplitter()
        split.addWidget(self.params)
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(250)
        self.statusBar().addPermanentWidget(self.progress)
        self.worker = None
        self.params.changed.connect(self._auto)

    def _auto(self):
        if self.tabs.currentWidget() is self.sim_tab and self.sim_tab.auto.isChecked():
            self.sim_tab.run()

    def run_job(self, fn, on_done):
        if self.worker is not None and self.worker.isRunning():
            self.statusBar().showMessage("Simulazione già in corso…", 2000)
            return
        self.statusBar().showMessage("Calcolo in corso…")
        self.progress.setValue(0)
        self.worker = Worker(fn)
        self.worker.progress.connect(lambda v: self.progress.setValue(int(v * 100)))

        def ok(res):
            self.progress.setValue(100)
            self.statusBar().showMessage("Fatto", 3000)
            on_done(res)

        self.worker.done.connect(ok)
        self.worker.failed.connect(self._error)
        self.worker.start()

    def _error(self, msg):
        self.statusBar().showMessage("Errore", 3000)
        QtWidgets.QMessageBox.critical(self, "Errore", msg)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
