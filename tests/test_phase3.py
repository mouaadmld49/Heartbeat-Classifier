"""Tests de la Phase 3 : python -m pytest tests -v"""
from pathlib import Path

import numpy as np
import pytest

from src.data.constants import DS1, DS2, PACED_RECORDS
from src.preprocessing.filters import clean_ecg
from src.preprocessing.normalization import GlobalScaler
from src.preprocessing.segmentation import extract_windows, rr_features

DATASET = Path(__file__).resolve().parents[1] / "data" / "processed" / "mitdb_beats_v1.npz"


def test_ds1_ds2_disjoint_and_exclude_paced():
    assert not set(DS1) & set(DS2)
    assert not (set(DS1) | set(DS2)) & set(PACED_RECORDS)


def test_filter_removes_baseline_wander():
    fs = 360
    t = np.arange(0, 30, 1 / fs)
    x = (2.0 * np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.sin(2 * np.pi * 10 * t))
    y = clean_ecg(x.astype(np.float32), fs)
    mid = slice(5 * fs, -5 * fs)          # on ignore les effets de bord
    # Sans dérive, il reste le sinus 10 Hz : écart-type ~ 0.1/sqrt(2) = 0.071
    assert 0.05 < y[mid].std() < 0.09


def test_windows_shape_and_boundaries():
    x = np.arange(1000, dtype=np.float32)
    W, keep = extract_windows(x, np.array([10, 100, 500, 995]), pre=90, post=110)
    assert W.shape == (2, 200)
    assert keep.tolist() == [False, True, True, False]
    assert W[0, 90] == 100                # le pic est à l'indice `pre`


def test_rr_features():
    rr = rr_features(np.array([0, 360, 720, 1080]), fs=360)
    assert np.isnan(rr["rr_prev"][0]) and np.isnan(rr["rr_next"][-1])
    assert np.allclose(rr["rr_prev"][1:], 1.0)
    assert np.allclose(rr["rr_next"][:-1], 1.0)


def test_scaler_uses_train_statistics_only():
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, (100, 50))
    test = rng.normal(10, 1, (100, 50))
    sc = GlobalScaler("zscore").fit(train)
    assert abs(sc.transform(train).mean()) < 1e-4
    assert sc.transform(test).mean() > 5   # le test n'est PAS recentré


@pytest.mark.skipif(not DATASET.exists(), reason="dataset non construit")
def test_split_is_by_record_without_leakage():
    d = np.load(DATASET)
    rec, split = d["record_id"], d["split"]
    s = {k: set(rec[split == k]) for k in ("train", "val", "test")}
    assert not (s["train"] & s["val"]) and not (s["train"] & s["test"])
    assert not (s["val"] & s["test"])
    assert s["test"] <= set(DS2)
    assert (s["train"] | s["val"]) <= set(DS1)
    assert not np.isnan(d["X"]).any()