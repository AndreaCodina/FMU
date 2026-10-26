"""Valida il modello su tutti i test sperimentali e salva una tabella riassuntiva.

Uso:
    python run_validation.py                 # tutti i test CVT + sismi
    python run_validation.py --filter 1320   # solo i file che contengono '1320'
    python run_validation.py --plots         # salva anche i grafici in results/plots
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import DEFAULT_DATA_DIR, list_tests
from params import AMDParams
from validation import energy_metric, metrics, prepare_test, run_test

OUT_DIR = Path(__file__).resolve().parent / "results"


def save_plot(tc, sim, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for a, k, lab in zip(ax, ("x", "Iq", "Vbus", "P_drive"),
                         ("Posizione [m]", "Iq [A]", "Vbus [V]", "Potenza drive [W]")):
        a.plot(tc.t, tc.meas[k], label="Misura", lw=1)
        a.plot(tc.t, sim[k], label="Modello", lw=1)
        a.set_ylabel(lab)
        a.grid(alpha=.3)
    ax[0].plot(tc.t, tc.x_ref, "k--", lw=.7, label="SetPosition")
    ax[0].legend(loc="upper right")
    ax[-1].set_xlabel("Tempo [s]")
    fig.suptitle(tc.name)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--filter", default="")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    tests = list_tests(args.data)
    files = [f for f in tests["CVT"] + tests["Sismi"] if args.filter in str(f)]
    OUT_DIR.mkdir(exist_ok=True)
    if args.plots:
        (OUT_DIR / "plots").mkdir(exist_ok=True)
    params = AMDParams.default()
    rows = []
    for i, f in enumerate(files, 1):
        t0 = time.time()
        tc = prepare_test(f)
        sim = run_test(tc, params)
        m = metrics(tc, sim)
        em = energy_metric(tc, sim)
        rows.append(dict(
            file=f.name, categoria=f.parent.name, massa=tc.mass, stringhe=tc.n_strings,
            durata_s=round(tc.t[-1], 1),
            x_rmse_mm=m["x"]["rmse"] * 1e3, x_nrmse_pct=m["x"]["nrmse"] * 100,
            Iq_rmse_A=m["Iq"]["rmse"], Vbus_rmse_V=m["Vbus"]["rmse"],
            P_rmse_kW=m["P_drive"]["rmse"] / 1e3,
            F_rmse_kN=em["F_rmse"] / 1e3,
            E_mis_kJ=em["E_meas"][-1] / 1e3, E_sim_kJ=em["E_sim"][-1] / 1e3,
            E_err_pct=em["err_pct"], E_err_ricostr_pct=em["err_rec_pct"],
            Vbus_min_mis=tc.meas["Vbus"].min(), Vbus_min_sim=sim["Vbus"].min(),
            sat_corrente_pct=100 * sim["sat_current"].mean(),
            bumper=bool(sim["bumper_contact"].max() > 0),
        ))
        print(f"[{i}/{len(files)}] {f.name:55s} x {rows[-1]['x_nrmse_pct']:5.2f}%  "
              f"Iq {rows[-1]['Iq_rmse_A']:5.1f} A  Vbus {rows[-1]['Vbus_rmse_V']:5.1f} V  "
              f"E {rows[-1]['E_err_pct']:+6.2f}%  "
              f"({time.time() - t0:.1f}s)", flush=True)
        if args.plots:
            save_plot(tc, sim, OUT_DIR / "plots" / (f.stem + ".png"))
    df = pd.DataFrame(rows)
    out = OUT_DIR / "validazione.csv"
    df.to_csv(out, index=False, sep=";", decimal=",")
    print("\nMediane per massa:")
    print(df.groupby("massa")[["x_nrmse_pct", "Iq_rmse_A", "Vbus_rmse_V", "P_rmse_kW", "E_err_pct"]].median().round(2))
    ok = (df["E_err_pct"].abs() <= 1).mean() * 100
    print(f"\nTest con |errore energia| <= 1%: {ok:.0f}%   (mediana |err| {df['E_err_pct'].abs().median():.2f}%)")
    print(f"\nTabella salvata in {out}")


if __name__ == "__main__":
    main()
