"""Extraction de caractéristiques par battement : morphologie, statistiques,
fréquentiel et rythme. Aucun paramètre n'est appris (donc pas de fuite)."""
import numpy as np
import pandas as pd
from scipy import signal, stats

EPS = 1e-8
DEFAULT_BANDS = {"0p5_5": (0.5, 5), "5_10": (5, 10), "10_15": (10, 15),
                 "15_25": (15, 25), "25_40": (25, 40)}


def morphological_features(X, pre=90, fs=360, half_search=10,
                           slope_half=36, slope_frac=0.25):
    """Features de forme (unités d'IQR, car le signal est normalisé par record)."""
    X = np.asarray(X, dtype=np.float64)
    rows = np.arange(len(X))

    # Pic dominant (en valeur absolue) autour de l'annotation, avec son signe
    seg = X[:, pre - half_search: pre + half_search + 1]
    k = np.abs(seg).argmax(axis=1)
    r_amp = seg[rows, k]
    r_offset = k - half_search

    # Largeur de pente : étendue des points dont |dx/dt| > slope_frac * pente max
    slope = np.abs(np.gradient(X, axis=1)) * fs
    zone = slope[:, pre - slope_half: pre + slope_half + 1]
    max_slope = zone.max(axis=1)
    above = zone > (slope_frac * max_slope)[:, None]
    first = above.argmax(axis=1)
    last = zone.shape[1] - 1 - above[:, ::-1].argmax(axis=1)
    width_ms = np.where(max_slope > EPS, (last - first + 1) / fs * 1000.0, 0.0)

    qrs = X[:, pre - 20: pre + 21]           # zone du complexe QRS (±56 ms)
    pre_zone = X[:, 10: pre - 30]            # proxy de l'onde P
    post_zone = X[:, pre + 40:]              # proxy de l'onde T

    def signed_peak(Z):                      # valeur du point de |x| maximal
        return Z[np.arange(len(Z)), np.abs(Z).argmax(axis=1)]

    return pd.DataFrame({
        "m_r_amp": r_amp, "m_r_offset": r_offset,
        "m_qrs_pos": qrs.max(axis=1), "m_qrs_neg": qrs.min(axis=1),
        "m_qrs_width_ms": width_ms, "m_max_slope": max_slope,
        "m_qrs_area": np.abs(qrs).sum(axis=1) / fs,
        "m_pre_peak": signed_peak(pre_zone), "m_post_peak": signed_peak(post_zone),
    })


def statistical_features(X):
    """Statistiques descriptives de la fenêtre."""
    X = np.asarray(X, dtype=np.float64)
    return pd.DataFrame({
        "s_mean": X.mean(axis=1), "s_std": X.std(axis=1), "s_var": X.var(axis=1),
        "s_min": X.min(axis=1), "s_max": X.max(axis=1), "s_ptp": np.ptp(X, axis=1),
        # fenêtre plate : skew/kurtosis valent NaN, on les remplace par 0
        "s_skew": np.nan_to_num(stats.skew(X, axis=1), nan=0.0),
        "s_kurtosis": np.nan_to_num(stats.kurtosis(X, axis=1), nan=0.0),
        "s_energy": (X ** 2).mean(axis=1),
    })


def frequency_features(X, fs=360, bands=None):
    """PSD (périodogramme à fenêtre de Hann, via FFT) et résumés spectraux."""
    bands = bands or DEFAULT_BANDS
    f, P = signal.periodogram(np.asarray(X, dtype=np.float64), fs=fs,
                              window="hann", detrend="constant", axis=1)
    total = P.sum(axis=1) + EPS
    out = {}
    for name, (lo, hi) in bands.items():
        bp = P[:, (f >= lo) & (f < hi)].sum(axis=1)
        out[f"f_bp_{name}"] = bp               # puissance absolue
        out[f"f_rbp_{name}"] = bp / total      # puissance relative
    out["f_dom_freq"] = f[P.argmax(axis=1)]    # fréquence dominante
    out["f_centroid"] = (P * f).sum(axis=1) / total
    pn = P / total[:, None]
    out["f_entropy"] = -(pn * np.log(pn + EPS)).sum(axis=1)
    return pd.DataFrame(out)


def rhythm_features(rr_prev, rr_next, rr_local):
    """Contexte rythmique (secondes). NaN possibles en début/fin de record."""
    rp, rn, rl = (np.asarray(a, dtype=np.float64) for a in (rr_prev, rr_next, rr_local))
    return pd.DataFrame({
        "rr_prev": rp, "rr_next": rn, "rr_local": rl,
        "rr_ratio_prev_local": rp / (rl + EPS),   # <1 : battement prématuré
        "rr_ratio_next_prev": rn / (rp + EPS),    # >1 : pause compensatrice
    })


def extract_features(X, rr_prev, rr_next, rr_local, pre=90, fs=360,
                     bands=None, half_search=10, slope_half=36, slope_frac=0.25):
    """Table de features (float32), une ligne par battement."""
    parts = [
        morphological_features(X, pre, fs, half_search, slope_half, slope_frac),
        statistical_features(X),
        frequency_features(X, fs, bands),
        rhythm_features(rr_prev, rr_next, rr_local),
    ]
    return pd.concat(parts, axis=1).astype(np.float32)


def feature_groups(columns):
    """Regroupe les colonnes par famille (pour les ablations des Phases 6 et 8)."""
    prefixes = {"morph": "m_", "stat": "s_", "freq": "f_", "rr": "rr_"}
    return {g: [c for c in columns if c.startswith(p)] for g, p in prefixes.items()}