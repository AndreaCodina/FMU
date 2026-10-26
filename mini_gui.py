"""Mini interfaccia interattiva: cursori + grafici, ricalcolo al rilascio del cursore.

Avvio:  python mini_gui.py

Seno in forza scaricata (ampiezza in kN) oppure in spostamento (ampiezza in m), con rampa di 1 s,
per massa, corsa, frequenza, durata e n° di stringhe scelti. Mostra posizione, forza scaricata (come rActualForce),
corrente con limite e tensione di bus.
"""
from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

import references as refs
from amd_model import AMDModel
from params import AMDParams

BASE = AMDParams.default()

fig = plt.figure(figsize=(13, 8.5))
fig.canvas.manager.set_window_title("AMD - mini simulatore")
gs = fig.add_gridspec(4, 1, left=0.07, right=0.98, top=0.93, bottom=0.36, hspace=0.12)
ax = [fig.add_subplot(gs[i]) for i in range(4)]
for a in ax[1:]:
    a.sharex(ax[0])
for a in ax[:-1]:
    a.tick_params(labelbottom=False)
info = fig.text(0.07, 0.955, "", fontsize=10)

SLIDERS = [  # nome, etichetta, min, max, valore iniziale, passo
    ("mass", "Massa [kg]", 300, 2500, BASE.mech.mass, 10),
    ("stroke", "Corsa ± [m]", 0.2, 1.5, BASE.mech.stroke, 0.01),
    ("famp", "Ampiezza forza [kN]", 0.5, 30, 15, 0.5),
    ("amp", "Ampiezza spost. [m]", 0.01, 1.2, 0.4, 0.01),
    ("freq", "Frequenza [Hz]", 0.2, 10, 1.0, 0.05),
    ("dur", "Durata [s]", 2, 30, 8, 0.5),
    ("strings", "Stringhe batteria", 1, 2, BASE.batt.n_strings, 1),
]
sl = {}
for i, (key, lab, lo, hi, v0, step) in enumerate(SLIDERS):
    sax = fig.add_axes([0.16, 0.30 - i * 0.038, 0.62, 0.024])
    sl[key] = Slider(sax, lab, lo, hi, valinit=v0, valstep=step)

btn = Button(fig.add_axes([0.83, 0.08, 0.12, 0.06]), "Simula")
mode = RadioButtons(fig.add_axes([0.83, 0.16, 0.14, 0.12]), ("Seno in forza", "Seno in spostamento"))
for s in sl.values():
    s.label.set_fontsize(9)


def simulate(_=None):
    info.set_text("Calcolo in corso…")
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    p = copy.deepcopy(BASE)
    p.mech.mass = float(sl["mass"].val)
    p.mech.stroke = float(sl["stroke"].val)
    p.batt.n_strings = int(sl["strings"].val)
    F_t = None
    if mode.value_selected == "Seno in forza":
        t, xr, F_t = refs.force_sine(float(sl["famp"].val) * 1e3, float(sl["freq"].val),
                                     float(sl["dur"].val), p.mech.mass)
    else:
        t, xr = refs.sine(float(sl["amp"].val), float(sl["freq"].val), float(sl["dur"].val))
    s = AMDModel(p).simulate(xr)

    for a in ax:
        a.clear()
        a.grid(alpha=0.3)
    ax[0].plot(t, xr, "k--", lw=0.9, label="SetPosition")
    ax[0].plot(t, s["x"], lw=1.2, label="ActualPosition")
    for sg in (1, -1):
        ax[0].axhline(sg * p.mech.stroke, color="r", ls=":", lw=0.8)
    ax[0].set_ylabel("Posizione [m]")
    ax[0].legend(loc="upper right", fontsize=8)
    if F_t is not None:
        ax[1].plot(t, F_t / 1e3, "k--", lw=0.9, label="Forza richiesta")
    ax[1].plot(t, s["F_actual"] / 1e3, lw=1, label="Forza scaricata (come rActualForce)")
    if np.abs(s["F_bumper"]).max() > 0:
        ax[1].plot(t, s["F_bumper"] / 1e3, lw=1, label="Bumper")
    ax[1].set_ylabel("Forza [kN]")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(t, s["Iq"], lw=1, label="Iq")
    ax[2].plot(t, s["I_lim"], "r:", lw=0.8, label="limite corrente/tensione")
    ax[2].plot(t, -s["I_lim"], "r:", lw=0.8)
    ax[2].set_ylabel("Corrente [A]")
    ax[2].legend(loc="upper right", fontsize=8)
    ax[3].plot(t, s["Vbus"], lw=1)
    ax[3].set_ylabel("Vbus [V]")
    ax[3].set_xlabel("Tempo [s]")

    err = np.abs(s["x"] - xr).max() * 1e3
    info.set_text(
        f"max |x| {np.abs(s['x']).max():.3f} m   errore max {err:.0f} mm   "
        f"max |v| {np.abs(s['v']).max():.2f} m/s   max |F| {np.abs(s['F_actual']).max() / 1e3:.1f} kN   "
        f"saturazione corrente {100 * s['sat_current'].mean():.0f}%   "
        f"Vbus min {s['Vbus'].min():.0f} V   "
        + (f"URTO BUMPER {np.abs(s['F_bumper']).max() / 1e3:.0f} kN" if s["bumper_contact"].max() else ""))
    fig.canvas.draw_idle()


btn.on_clicked(simulate)

# ricalcolo automatico 0.4 s dopo l'ultimo movimento di un cursore
_timer = fig.canvas.new_timer(interval=400)
_timer.single_shot = True
_timer.add_callback(simulate)


def _restart_timer(_):
    _timer.stop()
    _timer.start()


for s in sl.values():
    s.on_changed(_restart_timer)
mode.on_clicked(_restart_timer)
simulate()

if __name__ == "__main__":
    plt.show()
