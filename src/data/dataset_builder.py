"""Construction du dataset de battements segmentés (traçable, sans fuite)."""
import numpy as np
import pandas as pd

from ..preprocessing.filters import clean_ecg
from ..preprocessing.normalization import robust_scale_record
from ..preprocessing.segmentation import extract_windows, rr_features
from .constants import AAMI_MAP
from .loader import load_annotations, load_record


def process_record(rid, data_dir, cfg):
    """Un record -> (fenêtres X, table de métadonnées, journal de comptage)."""
    rec = load_record(rid, data_dir, lead=cfg["lead"])
    if rec.lead != cfg["lead"]:
        raise ValueError(f"Record {rid} : dérivation {rec.lead} != {cfg['lead']}")
    if rec.fs != cfg["fs"]:
        raise ValueError(f"Record {rid} : fs = {rec.fs} != {cfg['fs']}")

    flt = cfg["filter"]
    x = clean_ecg(rec.signal, rec.fs, flt["low_hz"], flt["high_hz"],
                  flt["order"], flt["notch_hz"])
    if cfg["normalization"] == "record_robust":
        x = robust_scale_record(x)
    elif cfg["normalization"] != "none":
        raise ValueError("normalization doit être 'none' ou 'record_robust'")

    ann = load_annotations(rid, data_dir)
    beats = ann[ann["symbol"].isin(AAMI_MAP)].reset_index(drop=True)
    beats["aami"] = beats["symbol"].map(AAMI_MAP)

    # RR calculés sur TOUS les battements, avant tout filtrage de classe
    for k, v in rr_features(beats["sample"].to_numpy(), rec.fs,
                            cfg["rr_local_n"]).items():
        beats[k] = v
    n_annotated = len(beats)

    beats = beats[beats["aami"].isin(cfg["classes"])].reset_index(drop=True)
    n_dropped_class = n_annotated - len(beats)

    X, keep = extract_windows(x, beats["sample"].to_numpy(),
                              cfg["window"]["pre"], cfg["window"]["post"])
    beats = beats[keep].reset_index(drop=True)

    log = {"record_id": str(rid), "n_annotated_beats": n_annotated,
           "n_dropped_class": n_dropped_class,
           "n_dropped_boundary": int((~keep).sum()), "n_kept": len(beats)}
    return X, beats, log


def build_dataset(record_ids, data_dir, cfg):
    """Retourne (X float32 (N, L), table meta, tableau des comptages)."""
    Xs, metas, logs = [], [], []
    for rid in record_ids:
        X, meta, log = process_record(rid, data_dir, cfg)
        Xs.append(X)
        metas.append(meta)
        logs.append(log)
    X = np.concatenate(Xs).astype(np.float32)
    meta = pd.concat(metas, ignore_index=True)
    cls2id = {c: i for i, c in enumerate(cfg["classes"])}
    meta["label"] = meta["aami"].map(cls2id).astype(np.int8)
    return X, meta, pd.DataFrame(logs)