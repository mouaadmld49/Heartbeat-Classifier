"""Fonctions de lecture des enregistrements MIT-BIH avec WFDB."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from .constants import AAMI_MAP


@dataclass
class ECGRecord:
    record_id: str
    signal: np.ndarray      # signal 1D en mV (float32)
    fs: int                 # fréquence d'échantillonnage
    lead: str               # dérivation effectivement utilisée
    sig_names: list         # toutes les dérivations du record
    units: str


def load_record(record_id, data_dir, lead="MLII") -> ECGRecord:
    """Lit un record et extrait la dérivation demandée PAR SON NOM.

    Le canal 0 n'est pas toujours MLII (ex. record 114), d'où la recherche
    par nom. Si la dérivation est absente (102, 104), on prend le canal 0.
    """
    path = str(Path(data_dir) / str(record_id))
    rec = wfdb.rdrecord(path)  # p_signal : signal physique en mV
    if lead in rec.sig_name:
        idx = rec.sig_name.index(lead)
    else:
        idx = 0
        print(f"[WARN] record {record_id} : {lead} absente {rec.sig_name}, "
              f"utilisation de {rec.sig_name[0]}")
    return ECGRecord(
        record_id=str(record_id),
        signal=rec.p_signal[:, idx].astype(np.float32),
        fs=rec.fs,
        lead=rec.sig_name[idx],
        sig_names=list(rec.sig_name),
        units=rec.units[idx],
    )


def load_annotations(record_id, data_dir) -> pd.DataFrame:
    """Lit le fichier .atr : une ligne par annotation (battement ou non)."""
    path = str(Path(data_dir) / str(record_id))
    ann = wfdb.rdann(path, "atr")
    return pd.DataFrame({
        "record_id": str(record_id),
        "sample": ann.sample,     # indice d'échantillon (proche du pic R)
        "symbol": ann.symbol,     # symbole d'annotation original
    })


def build_beat_table(record_ids, data_dir):
    """Retourne (toutes les annotations, table des battements avec classe AAMI)."""
    all_ann = pd.concat([load_annotations(r, data_dir) for r in record_ids],
                        ignore_index=True)
    beats = all_ann[all_ann["symbol"].isin(AAMI_MAP)].copy()
    beats["aami"] = beats["symbol"].map(AAMI_MAP)
    return all_ann, beats.reset_index(drop=True)