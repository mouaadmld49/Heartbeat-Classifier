"""Tests d'intégrité du dataset segmenté."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.constants import AAMI_MAP

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "preprocessing.yaml").read_text(encoding="utf-8"))
DATASET = ROOT / CFG["output"]["dataset"]
RAW = ROOT / "data" / "raw" / "mitdb"
need_dataset = pytest.mark.skipif(not DATASET.exists(), reason="dataset non construit")


@need_dataset
def test_arrays_have_same_length_and_expected_window():
    d = np.load(DATASET)
    n = len(d["X"])
    assert all(len(d[k]) == n for k in d.files)
    assert d["X"].shape[1] == CFG["window"]["pre"] + CFG["window"]["post"]


@need_dataset
def test_labels_match_symbol_mapping():
    d = np.load(DATASET)
    classes = CFG["classes"]
    expected = np.array([classes.index(AAMI_MAP[s]) for s in d["symbol"]])
    assert np.array_equal(expected, d["y"])


@need_dataset
def test_no_duplicate_beats_and_sorted_samples():
    d = np.load(DATASET)
    df = pd.DataFrame({"r": d["record_id"], "s": d["sample"]})
    assert not df.duplicated().any()
    assert df.groupby("r")["s"].apply(lambda s: s.is_monotonic_increasing).all()


@need_dataset
def test_manifest_counts_match_dataset():
    m = json.loads((ROOT / CFG["output"]["manifest"]).read_text(encoding="utf-8"))
    d = np.load(DATASET)
    classes = CFG["classes"]
    for split, row in m["counts_per_split"].items():
        for cls, n in row.items():
            mask = (d["split"] == split) & (d["y"] == classes.index(cls))
            assert int(mask.sum()) == n


@need_dataset
@pytest.mark.skipif(not (RAW / "100.dat").exists(), reason="données brutes absentes")
def test_reprocessing_record_100_gives_identical_windows():
    from src.data.dataset_builder import process_record
    X, meta, _ = process_record("100", RAW, CFG)
    d = np.load(DATASET)
    m = d["record_id"] == "100"
    assert np.array_equal(d["sample"][m], meta["sample"].to_numpy())
    assert np.allclose(d["X"][m], X, atol=1e-6)