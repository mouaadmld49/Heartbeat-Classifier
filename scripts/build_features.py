"""Construit la table de features : python scripts/build_features.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.features.extract import extract_features, feature_groups


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "features.yaml").read_text(encoding="utf-8"))
    pcfg = yaml.safe_load((ROOT / "configs" / "preprocessing.yaml").read_text(encoding="utf-8"))
    assert cfg["pre"] == pcfg["window"]["pre"], "features.pre doit égaler window.pre"
    assert cfg["fs"] == pcfg["fs"]

    d = np.load(ROOT / cfg["input"]["dataset"])
    bands = {k: tuple(v) for k, v in cfg["freq_bands_hz"].items()}
    feats = extract_features(
        d["X"], d["rr_prev"], d["rr_next"], d["rr_local"], pre=cfg["pre"], fs=cfg["fs"],
        bands=bands, half_search=cfg["qrs"]["half_search"],
        slope_half=cfg["qrs"]["slope_half"], slope_frac=cfg["qrs"]["slope_frac"])

    meta = pd.DataFrame({"record_id": d["record_id"], "sample": d["sample"],
                         "symbol": d["symbol"], "label": d["y"], "split": d["split"]})
    out = pd.concat([meta, feats], axis=1)

    path = ROOT / cfg["output"]["features"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, compression="gzip")

    print("Table :", out.shape, "->", path)
    print("Groupes :", {g: len(c) for g, c in feature_groups(feats.columns).items()})
    nan = feats.isna().sum()
    print("NaN par colonne (non nuls) :\n", nan[nan > 0])
    print("inf :", int(np.isinf(feats.to_numpy()).sum()))


if __name__ == "__main__":
    main()