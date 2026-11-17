"""Construit le dataset segmenté : python scripts/build_dataset.py"""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy
import wfdb
import yaml

from src.data.constants import ALL_RECORDS, DS1, DS2, PACED_RECORDS
from src.data.dataset_builder import build_dataset
from src.data.splits import assign_splits, choose_val_records


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    data_dir = ROOT / "data" / "raw" / "mitdb"
    records = [r for r in ALL_RECORDS if r not in PACED_RECORDS]

    X, meta, log = build_dataset(records, data_dir, cfg)

    classes = cfg["classes"]
    counts = (pd.crosstab(meta["record_id"], meta["aami"])
              .reindex(columns=classes, fill_value=0))
    sp = cfg["split"]
    val_records = choose_val_records(counts, DS1, classes, sp["val_n_records"],
                                     sp["val_fraction"], sp["n_trials"], cfg["seed"])
    meta["split"] = assign_splits(meta["record_id"], DS1, DS2, val_records)

    out = ROOT / cfg["output"]["dataset"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=X, y=meta["label"].to_numpy(np.int8),
        record_id=meta["record_id"].to_numpy(str),
        sample=meta["sample"].to_numpy(np.int32),
        symbol=meta["symbol"].to_numpy(str),
        rr_prev=meta["rr_prev"].to_numpy(np.float32),
        rr_next=meta["rr_next"].to_numpy(np.float32),
        rr_local=meta["rr_local"].to_numpy(np.float32),
        split=meta["split"].to_numpy(str))
    meta.to_csv(ROOT / cfg["output"]["meta"], index=False)

    split_counts = (pd.crosstab(meta["split"], meta["aami"])
                    .reindex(index=["train", "val", "test"], columns=classes,
                             fill_value=0))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "wfdb": wfdb.__version__},
        "X_shape": list(X.shape),
        "X_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
        "val_records": val_records,
        "counts_per_split": split_counts.astype(int).to_dict("index"),
        "dropped": {"class_excluded": int(log["n_dropped_class"].sum()),
                    "boundary": int(log["n_dropped_boundary"].sum())},
        "per_record_log": log.to_dict("records"),
    }
    (ROOT / cfg["output"]["manifest"]).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("X :", X.shape, X.dtype)
    print("Battements annotés :", int(log["n_annotated_beats"].sum()),
          "| exclus (classe) :", int(log["n_dropped_class"].sum()),
          "| exclus (bords) :", int(log["n_dropped_boundary"].sum()))
    print("Records de validation :", val_records)
    print(split_counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "preprocessing.yaml"))
    main(ap.parse_args().config)