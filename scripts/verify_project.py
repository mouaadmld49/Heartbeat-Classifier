"""Vérifie la cohérence de bout en bout : python scripts/verify_project.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

P = yaml.safe_load((ROOT / "configs" / "preprocessing.yaml").read_text(encoding="utf-8"))
F = yaml.safe_load((ROOT / "configs" / "features.yaml").read_text(encoding="utf-8"))
ok = True


def check(name, cond):
    global ok
    print(f"  [{'OK' if cond else 'KO'}] {name}")
    ok &= bool(cond)


print("Fichiers")
for p in ["data/raw/mitdb/100.dat", P["output"]["dataset"], P["output"]["meta"],
          P["output"]["manifest"], F["output"]["features"]]:
    check(p, (ROOT / p).exists())

d = np.load(ROOT / P["output"]["dataset"])
feats = pd.read_csv(ROOT / F["output"]["features"], dtype={"record_id": str})

print("Dataset")
check("aucun NaN/inf dans X", np.isfinite(d["X"]).all())
check("X en float32, L = pre + post",
      d["X"].dtype == np.float32 and d["X"].shape[1] == P["window"]["pre"] + P["window"]["post"])
check("labels dans {0..3}", set(np.unique(d["y"])) == {0, 1, 2, 3})
rec, sp = d["record_id"], d["split"]
check("aucun record partagé train/val/test",
      not (set(rec[sp == "train"]) & set(rec[sp == "val"]))
      and not (set(rec[sp == "train"]) & set(rec[sp == "test"]))
      and not (set(rec[sp == "val"]) & set(rec[sp == "test"])))

print("Features")
check("même nombre de lignes que X", len(feats) == len(d["X"]))
check("record_id/sample alignés avec le dataset",
      (feats.record_id.to_numpy() == rec).all() and (feats["sample"].to_numpy() == d["sample"]).all())
check("labels/split identiques", (feats.label.to_numpy() == d["y"]).all()
      and (feats.split.to_numpy() == sp).all())
cols = [c for c in feats.columns if c not in ("record_id", "sample", "symbol", "label", "split")]
nan_cols = [c for c in cols if feats[c].isna().any()]
check("NaN uniquement dans rr_*", all(c.startswith("rr_") for c in nan_cols))

print("Pytest")
r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT)
check("tous les tests passent", r.returncode == 0)

print("\nRESULTAT GLOBAL :", "OK" if ok else "ECHEC")
sys.exit(0 if ok else 1)