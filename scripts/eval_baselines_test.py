"""Évaluation finale des baselines, UNE FOIS (hyperparamètres fixes).
python scripts/eval_baselines_test.py [--direction ds1_to_ds2|ds2_to_ds1] [--force]"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import classification_summary
from src.models.baselines import MODEL_NAMES, feature_sets, fit_pipeline, make_pipeline

META = ["record_id", "sample", "symbol", "label", "split"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="ds1_to_ds2", choices=["ds1_to_ds2", "ds2_to_ds1"])
    ap.add_argument("--force", action="store_true", help="réécrire un résultat existant")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "baselines.yaml").read_text(encoding="utf-8"))
    out_dir = ROOT / cfg["output"]["tables"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"baselines_test_{args.direction}.csv"
    if out_csv.exists() and not args.force:
        sys.exit(f"{out_csv.name} existe déjà : le test ne se rejoue pas (--force pour l'écraser).")

    df = pd.read_csv(ROOT / cfg["features"], dtype={"record_id": str})
    cols = [c for c in df.columns if c not in META]
    fsets = feature_sets(cols)
    is_ds1 = df["split"].isin(["train", "val"])
    tr_df, te_df = (df[is_ds1], df[~is_ds1]) if args.direction == "ds1_to_ds2" else (df[~is_ds1], df[is_ds1])
    assert not set(tr_df.record_id) & set(te_df.record_id), "fuite : record partagé"
    ytr, yte = tr_df["label"].to_numpy(np.int64), te_df["label"].to_numpy(np.int64)
    rec = te_df["record_id"].to_numpy(str)
    f_mask = yte == 3
    dominant = pd.Series(rec[f_mask]).value_counts().idxmax()   # record qui porte presque tous les F
    print(f"{args.direction} : train {len(tr_df)} | test {len(te_df)} | record F dominant : {dominant}")

    rows, preds = [], {}
    for name in MODEL_NAMES:
        for weighted, fs in [(False, "all"), (True, "all"), (True, "no_rr")]:
            pipe = make_pipeline(name, cfg["models"][name], weighted, cfg["seed"])
            fit_pipeline(pipe, name, tr_df[fsets[fs]].to_numpy(np.float32), ytr, weighted)
            pred = pipe.predict(te_df[fsets[fs]].to_numpy(np.float32))
            s = classification_summary(yte, pred)
            m = f_mask & (rec != dominant)
            s["recall_F_sans_dominant"] = float((pred[m] == 3).mean()) if m.any() else np.nan
            label = "balanced" if weighted else "none"
            rows.append({"model": name, "weighting": label, "features": fs, "direction": args.direction, **s})
            preds[f"{name}__{label}__{fs}"] = pred
            byrec = (pd.DataFrame({"rec": rec[f_mask], "ok": pred[f_mask] == 3})
                     .groupby("rec")["ok"].agg(["sum", "count"]))
            print(f"{name:7s} {label:8s} {fs:6s} macroF1={s['macro_f1']:.3f} | recall N/S/V/F = "
                  f"{s['recall_N']:.3f}/{s['recall_S']:.3f}/{s['recall_V']:.3f}/{s['recall_F']:.3f} | "
                  f"F hors {dominant} = {s['recall_F_sans_dominant']:.3f} | "
                  f"F par record (bien classés/total) : "
                  + ", ".join(f"{r}:{int(v['sum'])}/{int(v['count'])}" for r, v in byrec.iterrows()))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    np.savez_compressed(out_dir / f"baselines_test_{args.direction}_pred.npz",
                        y=yte, record_id=rec, **preds)
    print("\nRésultats ->", out_csv)


if __name__ == "__main__":
    main()