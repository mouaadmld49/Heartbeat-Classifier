"""Segmentation en fenêtres autour des pics R et features rythmiques."""
import numpy as np
import pandas as pd


def extract_windows(x, peaks, pre, post):
    """Fenêtres [s - pre, s + post) autour de chaque pic.

    Retourne (fenêtres (n, pre+post), masque des pics conservés).
    Les battements trop près des bords du record sont exclus.
    Le pic se trouve à l'indice `pre` de chaque fenêtre.
    """
    peaks = np.asarray(peaks, dtype=np.int64)
    keep = (peaks - pre >= 0) & (peaks + post <= len(x))
    idx = peaks[keep][:, None] + np.arange(-pre, post)[None, :]
    return np.asarray(x)[idx], keep


def rr_features(samples, fs, n_local=10):
    """Intervalles RR (secondes) à partir des positions de TOUS les battements.

    rr_prev  : intervalle avec le battement précédent (NaN pour le 1er)
    rr_next  : intervalle avec le suivant (NaN pour le dernier)
    rr_local : moyenne des n_local RR précédant rr_prev (rythme de fond)
    """
    s = np.asarray(samples, dtype=np.float64)
    if len(s) < 2:
        nan = np.full(len(s), np.nan, dtype=np.float32)
        return {"rr_prev": nan, "rr_next": nan.copy(), "rr_local": nan.copy()}
    rr = np.diff(s) / fs
    rr_prev = np.r_[np.nan, rr]
    rr_next = np.r_[rr, np.nan]
    rr_local = (pd.Series(rr_prev).shift(1)
                .rolling(n_local, min_periods=1).mean().to_numpy())
    return {"rr_prev": rr_prev.astype(np.float32),
            "rr_next": rr_next.astype(np.float32),
            "rr_local": rr_local.astype(np.float32)}