"""Costruisce la FMU (FMI 2.0 Co-Simulation) 'AMD_Actuator.fmu' e la verifica con FMPy.

Uso:
    python build_fmu.py            # build + test rapido
    python build_fmu.py --no-test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_FILES = ["params.py", "battery.py", "amd_model.py", "data_io.py", "force_energy.py", "data"]


def build() -> Path:
    from pythonfmu.builder import FmuBuilder

    out = FmuBuilder.build_FMU(HERE / "fmu_wrapper.py", dest=HERE,
                               project_files=[HERE / f for f in MODEL_FILES])
    print(f"FMU creata: {out}")
    return Path(out)


def test(fmu: Path):
    from fmpy import simulate_fmu

    t = np.arange(0, 3.0, 1e-3)
    xr = 0.3 * np.sin(2 * np.pi * 1.0 * t) * np.clip(t / 0.5, 0, 1)
    inputs = np.array([(ti, xi, True, False, True) for ti, xi in zip(t, xr)],
                      dtype=[("time", float), ("x_ref", float), ("enable", bool),
                             ("precharge_contactor", bool), ("main_contactor", bool)])
    res = simulate_fmu(str(fmu), start_time=0, stop_time=2.999, step_size=1e-3, output_interval=1e-3,
                       input=inputs, output=["x", "Iq", "Vbus", "SOC"], start_values={"mass": 1320.0})
    err = np.max(np.abs(res["x"] - np.interp(res["time"], t, xr)))
    print(f"Test FMPy: max |x - x_ref| = {err * 1e3:.1f} mm, Vbus min {res['Vbus'].min():.0f} V, "
          f"Iq max {np.abs(res['Iq']).max():.0f} A")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-test", action="store_true")
    a = ap.parse_args()
    f = build()
    if not a.no_test:
        test(f)
