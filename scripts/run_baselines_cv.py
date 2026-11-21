"""Baselines : validation croisée par records sur DS1 UNIQUEMENT.
python scripts/run_baselines_cv.py [--models logreg rf svm xgb] [--skip-ablation]"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import classification_summary
from src.models.baselines import (MODEL_NAMES, feature_sets, fit_pipeline,
                                  make_group_folds, make_pipeline)

META = ["record_id", "sample", "symbol", "label", "split"]


def run_cv(name, params, weighted, X, y, folds, seed):
    """Prédictions out-of-fold et Macro-F1 (N,S,V) de chaque pli."""
    oof = np.full(len(y), -1, dtype=np.int64)
    per_fold = []
    for tr, te in folds:
        pipe = make_pipeline(name, params, weighted, seed)
        fit_pipeline(pipe, name, X[tr], y[tr], weighted)
        oof[te] = pipe.predict(X[te])
        per_fold.append(classification_summary(y[te], oof[te])["macro_f1_nsv"])
    return oof, per_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "baselines.yaml").read_text(encoding="utf-8"))
    seed = cfg["seed"]
    df = pd.read_csv(ROOT / cfg["features"], dtype={"record_id": str})
    df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)   # DS1 seulement
    assert "test" not in set(df["split"])
    cols = [c for c in df.columns if c not in META]
    fsets = feature_sets(cols)
    y = df["label"].to_numpy(np.int64)
    groups = df["record_id"].to_numpy(str)
    print(f"DS1 : {len(df)} battements, {len(set(groups))} records, {len(cols)} features")

    folds = make_group_folds(y, groups, cfg["cv"]["n_splits"], seed)
    print("\nComposition des plis (le pli contenant 208 contient presque tous les F) :")
    print(pd.DataFrame([{"pli": k, "records": len(set(groups[te])),
                         **dict(zip("NSVF", np.bincount(y[te], minlength=4)))}
                        for k, (_, te) in enumerate(folds)]).to_string(index=False))

    exps = [(m, w, "all") for m in args.models for w in (False, True)]
    if not args.skip_ablation:
        exps += [(m, True, fs) for m in args.models for fs in ("no_rr", "rr_only", "morph_rr")]

    rows, oofs = [], {}

    def record(name, weighted, fs, cv_kind, oof, per_fold, secs):
        s = classification_summary(y, oof)
        rows.append({"model": name, "weighting": "balanced" if weighted else "none",
                     "features": fs, "n_features": len(fsets[fs]), "cv": cv_kind, **s,
                     "f1nsv_fold_mean": float(np.mean(per_fold)),
                     "f1nsv_fold_std": float(np.std(per_fold)), "seconds": round(secs, 1)})
        oofs[f"{name}__{'balanced' if weighted else 'none'}__{fs}__{cv_kind}"] = oof
        print(f"  {name:7s} {'balanced' if weighted else 'none':8s} {fs:9s} {cv_kind:6s} "
              f"macroF1_NSV={s['macro_f1_nsv']:.3f} | recall N/S/V/F = "
              f"{s['recall_N']:.3f}/{s['recall_S']:.3f}/{s['recall_V']:.3f}/{s['recall_F']:.3f} "
              f"| {secs:.0f}s")

    print("\nValidation croisée par records :")
    for name, weighted, fs in exps:
        t0 = time.time()
        X = df[fsets[fs]].to_numpy(np.float32)
        oof, pf = run_cv(name, cfg["models"][name], weighted, X, y, folds, seed)
        record(name, weighted, fs, "group", oof, pf, time.time() - t0)

    if "rf" in args.models:
        print("\nContrôle de fuite (H3) : Random Forest, CV ALEATOIRE par battement :")
        t0 = time.time()
        X = df[fsets["all"]].to_numpy(np.float32)
        rnd = list(StratifiedKFold(cfg["cv"]["n_splits"], shuffle=True,
                                   random_state=seed).split(X, y))
        oof, pf = run_cv("rf", cfg["models"]["rf"], True, X, y, rnd, seed)
        record("rf", True, "all", "random", oof, pf, time.time() - t0)

    out = ROOT / cfg["output"]["tables"]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "baselines_cv.csv", index=False)
    np.savez_compressed(out / "baselines_oof.npz", y=y, record_id=groups, **oofs)
    print("\nRésultats ->", out / "baselines_cv.csv")


if __name__ == "__main__":
    main()