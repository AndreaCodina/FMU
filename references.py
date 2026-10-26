"""Generatori di profili SetPosition per la simulazione."""
from __future__ import annotations

import numpy as np

from data_io import DT_SCOPE, load_twincat_csv


def _ramp_window(t: np.ndarray, t_ramp: float, T: float) -> np.ndarray:
    """Finestra con rampa coseno in ingresso e uscita (evita gradini di velocità)."""
    w = np.ones_like(t)
    if t_ramp > 0:
        a = t < t_ramp
        w[a] = 0.5 - 0.5 * np.cos(np.pi * t[a] / t_ramp)
        b = t > T - t_ramp
        w[b] = 0.5 - 0.5 * np.cos(np.pi * (T - t[b]) / t_ramp)
    return w


def sine(amp: float, freq: float, duration: float, t_ramp: float = 1.0,
         dt: float = DT_SCOPE, t_pre: float = 0.5):
    t = np.arange(0, duration + t_pre, dt)
    tt = np.clip(t - t_pre, 0, None)
    x = amp * np.sin(2 * np.pi * freq * tt) * _ramp_window(tt, t_ramp, duration)
    x[t < t_pre] = 0.0
    return t, x


def sweep(amp: float, f0: float, f1: float, duration: float, dt: float = DT_SCOPE,
          t_pre: float = 0.5):
    t = np.arange(0, duration + t_pre, dt)
    tt = np.clip(t - t_pre, 0, None)
    k = (f1 - f0) / duration
    phase = 2 * np.pi * (f0 * tt + 0.5 * k * tt ** 2)
    x = amp * np.sin(phase) * _ramp_window(tt, min(1.0, duration / 10), duration)
    x[t < t_pre] = 0.0
    return t, x


def _cos_window(t: np.ndarray, t_ramp: float, T: float) -> np.ndarray:
    w = np.ones_like(t)
    if t_ramp > 0:
        a = t < t_ramp
        w[a] = 0.5 - 0.5 * np.cos(np.pi * t[a] / t_ramp)
        b = t > T - t_ramp
        w[b] = 0.5 - 0.5 * np.cos(np.pi * np.clip(T - t[b], 0, None) / t_ramp)
    w[(t < 0) | (t > T)] = 0.0
    return w


def force_to_position(F: np.ndarray, mass: float, dt: float = DT_SCOPE, f_cut: float = 0.1) -> np.ndarray:
    """SetPosition che produce la forza scaricata F (convenzione PLC: F = -M * x'').

    Doppia integrazione nel dominio della frequenza, X(f) = F(f) / (M (2 pi f)^2), con passa-alto
    dolce sotto f_cut (evita derive di posizione). Il segnale viene esteso con zeri per evitare
    effetti di periodicità.
    """
    n = len(F)
    N = int(2 ** np.ceil(np.log2(2 * n)))
    Fz = np.zeros(N)
    Fz[:n] = F
    f = np.fft.rfftfreq(N, dt)
    S = np.fft.rfft(Fz)
    w = 2 * np.pi * f
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.where(f > 0, S / (mass * w ** 2), 0.0)
    hp = (f / f_cut) ** 4 / (1 + (f / f_cut) ** 4)       # passa-alto del 4° ordine (fase zero)
    x = np.fft.irfft(X * hp, N)[:n]
    return x - x[0]


def force_sine(F_amp: float, freq: float, duration: float, mass: float, t_ramp: float = 1.0,
               dt: float = DT_SCOPE, t_pre: float = 0.5, t_post: float = 1.0):
    """Seno in forza scaricata (convenzione PLC: F = -M x'').

    x(t) = F_amp / (M w^2) * win(t) * sin(w t): a regime la forza è esattamente F_amp sin(w t),
    la posizione parte e torna a zero senza derive. Restituisce (t, x_ref, F_rif) con F_rif la
    forza implicata dal riferimento (-M x_ref'').
    """
    t = np.arange(0, t_pre + duration + t_post, dt)
    tt = t - t_pre
    w = 2 * np.pi * freq
    x = F_amp / (mass * w ** 2) * _cos_window(tt, t_ramp, duration) * np.sin(w * tt)
    return t, x, _implied_force(x, mass, dt)


def force_sweep(F_amp: float, f0: float, f1: float, duration: float, mass: float,
                t_ramp: float = 1.0, dt: float = DT_SCOPE, t_pre: float = 0.5, t_post: float = 1.0):
    """Sweep lineare in forza scaricata da f0 a f1 (stessa costruzione di force_sine con w(t))."""
    t = np.arange(0, t_pre + duration + t_post, dt)
    tt = np.clip(t - t_pre, 0, duration)
    k = (f1 - f0) / duration
    fi = f0 + k * tt
    phase = 2 * np.pi * (f0 * tt + 0.5 * k * tt ** 2)
    x = F_amp / (mass * (2 * np.pi * fi) ** 2) * _cos_window(t - t_pre, t_ramp, duration) * np.sin(phase)
    return t, x, _implied_force(x, mass, dt)


def _implied_force(x: np.ndarray, mass: float, dt: float) -> np.ndarray:
    return -mass * np.gradient(np.gradient(x, dt), dt)


def step(amp: float, duration: float, t_step: float = 0.5, dt: float = DT_SCOPE):
    t = np.arange(0, duration, dt)
    return t, np.where(t >= t_step, amp, 0.0)


def ramp_to(target: float, speed: float, duration: float, t_start: float = 0.5,
            dt: float = DT_SCOPE):
    """Rampa a velocità costante fino a 'target' (utile per provare l'urto sul bumper)."""
    t = np.arange(0, duration, dt)
    x = np.clip((t - t_start) * speed * np.sign(target), -abs(target), abs(target))
    x[t < t_start] = 0.0
    return t, x


def from_test_file(path: str):
    d = load_twincat_csv(path)
    x = d["rSetPosition"].values
    return np.arange(len(x)) * DT_SCOPE, x


def is_twincat_csv(path: str) -> bool:
    with open(path, encoding="latin1") as fh:
        head = [fh.readline() for _ in range(8)]
    return any(h.startswith("Name;") for h in head)


_TIME_NAMES = ("t", "time", "tempo", "time_s", "t_s", "tempo_s", "s")
_POS_NAMES = ("x", "x_ref", "setpos", "set_position", "setposition", "rsetposition", "posizione",
              "position", "pos", "riferimento", "ref")


def _read_table(path: str):
    """Legge un CSV numerico provando separatori e separatore decimale."""
    import pandas as pd

    best = None
    for sep in (";", ",", "\t", r"\s+"):
        for dec in (".", ","):
            if sep == "," and dec == ",":
                continue
            try:
                df = pd.read_csv(path, sep=sep, decimal=dec, engine="python", header=None,
                                 skip_blank_lines=True, comment="#")
            except Exception:  # noqa: BLE001
                continue
            num = df.apply(pd.to_numeric, errors="coerce")
            # eventuale riga di intestazione: prima riga tutta non numerica
            header = None
            if num.iloc[0].isna().all() and len(df) > 1:
                header = [str(c).strip().lower() for c in df.iloc[0]]
                num = num.iloc[1:]
            num = num.dropna(axis=1, how="all").dropna(axis=0, how="any")
            if num.shape[0] < 2 or num.shape[1] < 1:
                continue
            score = num.shape[0] * num.shape[1]
            if best is None or score > best[0]:
                cols = None
                if header is not None:
                    cols = [header[i] if i < len(header) else str(i) for i in num.columns]
                best = (score, num.to_numpy(dtype=float), cols)
    if best is None:
        raise ValueError("Formato CSV non riconosciuto: servono una o due colonne numeriche "
                         "(tempo [s] e posizione, oppure solo posizione)")
    return best[1], best[2]


def from_csv(path: str, pos_unit: str = "m", dt_single: float = DT_SCOPE, dt: float = DT_SCOPE):
    """Set position da CSV qualsiasi.

    * File TwinCAT Scope (come i test sperimentali): usa rSetPosition; se presente restituisce anche
      la posizione misurata rActualPosition.
    * CSV semplice: colonne [tempo_s, posizione] (anche con intestazione) oppure una sola colonna
      di posizione campionata a dt_single. Posizione in m o mm (pos_unit). Ricampionato a dt.

    Restituisce (t, x_ref, misure) con misure = dict (eventualmente vuoto).
    """
    if is_twincat_csv(path):
        d = load_twincat_csv(path)
        meas = {}
        if "rActualPosition" in d:
            meas["x"] = d["rActualPosition"].values
        x = d["rSetPosition"].values
        return np.arange(len(x)) * DT_SCOPE, x, meas

    data, cols = _read_table(path)
    scale = 1e-3 if pos_unit == "mm" else 1.0
    if data.shape[1] == 1:
        x = data[:, 0] * scale
        t_raw = np.arange(len(x)) * dt_single
    else:
        it, ix = 0, 1
        if cols:
            for i, c in enumerate(cols):
                if c in _TIME_NAMES:
                    it = i
            for i, c in enumerate(cols):
                if c in _POS_NAMES and i != it:
                    ix = i
                    break
            else:
                ix = 1 if it == 0 else 0
        t_raw, x = data[:, it], data[:, ix] * scale
        order = np.argsort(t_raw, kind="stable")
        t_raw, x = t_raw[order], x[order]
        keep = np.r_[True, np.diff(t_raw) > 0]
        t_raw, x = t_raw[keep], x[keep]
        if len(t_raw) < 2:
            raise ValueError("La colonna del tempo deve essere crescente")
    t = np.arange(0.0, t_raw[-1] - t_raw[0] + dt / 2, dt)
    return t, np.interp(t + t_raw[0], t_raw, x), {}
