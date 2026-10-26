"""Lettura dei file CSV esportati da TwinCAT Scope (formato 'Name;var;Name;var...').

Le colonne vengono rinominate togliendo gli indici di array ([0], [0][0]) così che
i nomi siano uguali tra le varie campagne di test (es. 'rActualPosition').
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

DT_SCOPE = 1e-3  # periodo di campionamento dei test [s]

# cartella dati di default (sibling di '03_Directory claude code')
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "02_File FMU per claude"


def _clean_name(name: str) -> str:
    return re.sub(r"(\[\d+\])+$", "", name.strip())


def load_twincat_csv(path: str | os.PathLike) -> pd.DataFrame:
    """Carica un CSV TwinCAT Scope e restituisce un DataFrame con colonna 't' [s]."""
    path = str(path)
    with open(path, encoding="latin1") as fh:
        header_idx = None
        for i, line in enumerate(fh):
            if line.startswith("Name;") and i > 2:
                header_idx = i
                names = line.strip().split(";")[1::2]
                break
    if header_idx is None:
        raise ValueError(f"Header non trovato in {path}")
    df = pd.read_csv(path, sep=";", skiprows=header_idx + 2, header=None,
                     encoding="latin1", low_memory=False)
    df = df.iloc[:, 1::2]
    df = df.iloc[:, : len(names)]
    cols = [_clean_name(n) for n in names[: df.shape[1]]]
    # nomi duplicati dopo la pulizia (es. bRelayStrings[0]/[1]) -> tengo l'indice
    seen = {}
    final = []
    for raw, c in zip(names, cols):
        if cols.count(c) > 1:
            c = raw.strip()
        seen[c] = seen.get(c, 0) + 1
        final.append(c)
    df.columns = final
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all").reset_index(drop=True)
    df = df.ffill().fillna(0.0)
    df.insert(0, "t", np.arange(len(df)) * DT_SCOPE)
    return df


def list_tests(data_dir: str | os.PathLike = DEFAULT_DATA_DIR) -> dict[str, list[Path]]:
    """Restituisce i CSV sperimentali raggruppati per categoria (FRF, CVT, sismi)."""
    base = Path(data_dir) / "Test sperimentali"
    out: dict[str, list[Path]] = {"FRF": [], "CVT": [], "Sismi": []}
    for p in sorted(base.rglob("*.csv")):
        parent = p.parent.name.lower()
        if "frf" in parent or "sweep" in parent:
            out["FRF"].append(p)
        elif "sism" in parent:
            out["Sismi"].append(p)
        else:
            out["CVT"].append(p)
    return out


def mass_from_path(path: str | os.PathLike) -> float | None:
    """Estrae la massa dal nome cartella 'NN_Massa XXX kg'."""
    m = re.search(r"Massa\s*(\d+)\s*kg", str(path))
    return float(m.group(1)) if m else None
