"""Tests de la Phase 4 : python -m pytest tests -v"""
import numpy as np
import pandas as pd

from src.features.extract import (extract_features, feature_groups,
                                  frequency_features, morphological_features)

FS, PRE, L = 360, 90, 200


def gaussian_beat(sigma):
    t = np.arange(L) - PRE
    return np.exp(-0.5 * (t / sigma) ** 2)


def test_wider_qrs_gives_larger_width():
    X = np.stack([gaussian_beat(4), gaussian_beat(12)])
    w = morphological_features(X, PRE, FS)["m_qrs_width_ms"].to_numpy()
    assert w[1] > 2 * w[0]


def test_r_amplitude_keeps_sign():
    X = np.stack([gaussian_beat(4), -gaussian_beat(4)])
    amp = morphological_features(X, PRE, FS)["m_r_amp"].to_numpy()
    assert amp[0] > 0.9 and amp[1] < -0.9


def test_dominant_frequency_of_a_sinusoid():
    t = np.arange(L) / FS
    X = np.sin(2 * np.pi * 10 * t)[None, :]
    dom = frequency_features(X, FS)["f_dom_freq"].iloc[0]
    assert abs(dom - 10) <= 2.0            # résolution = 1,8 Hz


def test_extract_features_shape_groups_and_no_nan():
    rng = np.random.default_rng(0)
    n = 50
    X = rng.normal(size=(n, L)).astype(np.float32)
    rr = np.full(n, 0.8, dtype=np.float32)
    df = extract_features(X, rr, rr, rr, pre=PRE, fs=FS)
    assert len(df) == n and df.shape[1] == 36
    assert not df.isna().any().any() and np.isfinite(df.to_numpy()).all()
    groups = feature_groups(df.columns)
    assert sum(len(v) for v in groups.values()) == df.shape[1]
    assert np.allclose(df["rr_ratio_prev_local"], 1.0, atol=1e-5)


def test_flat_window_does_not_crash():
    X = np.zeros((2, L), dtype=np.float32)
    rr = np.ones(2, dtype=np.float32)
    df = extract_features(X, rr, rr, rr, pre=PRE, fs=FS)
    assert np.isfinite(df.to_numpy()).all()
    assert (df["m_qrs_width_ms"] == 0).all()