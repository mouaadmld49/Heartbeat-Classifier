"""Détection des pics R (inspirée de Pan & Tompkins, 1985), seuil adaptatif."""
import numpy as np
import pandas as pd
from scipy import signal


def qrs_energy(x, fs=360):
    """Retourne (signal filtré 5-15 Hz, énergie intégrée sur 150 ms)."""
    x = np.asarray(x, dtype=np.float64)
    sos = signal.butter(2, [5, 15], btype="bandpass", fs=fs, output="sos")
    f = signal.sosfiltfilt(sos, x)
    energy = np.gradient(f) ** 2                      # dérivée puis carré
    w = int(0.15 * fs)
    mwi = np.convolve(energy, np.ones(w) / w, mode="same")
    return f, mwi


def detect_r_peaks(x, fs=360, refractory_s=0.25, rel_thr=0.15,
                   k_neighbors=10, floor_ratio=0.03):
    """Pics R détectés avec un seuil LOCAL.

    1) candidats = pics de l'énergie au-dessus d'un plancher très bas
       (floor_ratio x 99,5e percentile), séparés d'au moins refractory_s ;
    2) chaque candidat est conservé si sa hauteur dépasse rel_thr x médiane
       des hauteurs de ses 2*k_neighbors+1 voisins (le seuil s'adapte à
       l'amplitude locale, donc aux QRS petits comme grands).
    rel_thr et floor_ratio sont des hyperparamètres : à choisir sur DS1.
    """
    f, mwi = qrs_energy(x, fs)
    floor = floor_ratio * np.percentile(mwi, 99.5)
    cand, props = signal.find_peaks(mwi, height=floor,
                                    distance=int(refractory_s * fs))
    if len(cand) == 0:
        return np.array([], dtype=np.int64)
    h = props["peak_heights"]
    local = (pd.Series(h).rolling(2 * k_neighbors + 1, center=True, min_periods=1)
             .median().to_numpy())
    cand = cand[h >= rel_thr * local]

    r = int(0.075 * fs)                               # affinage autour du candidat
    peaks = []
    for c in cand:
        lo, hi = max(0, c - r), min(len(f), c + r + 1)
        peaks.append(lo + int(np.argmax(np.abs(f[lo:hi]))))
    return np.unique(peaks)


def evaluate_detection(ref, det, tol):
    """Apparie détections et références (tol en échantillons). Retourne (TP, FP, FN)."""
    ref, det = np.asarray(ref), np.asarray(det)
    used = np.zeros(len(det), dtype=bool)
    tp = 0
    for r in ref:
        j = np.searchsorted(det, r)
        cands = [k for k in (j - 1, j)
                 if 0 <= k < len(det) and not used[k] and abs(det[k] - r) <= tol]
        if cands:
            k = min(cands, key=lambda k: abs(det[k] - r))
            used[k] = True
            tp += 1
    return tp, len(det) - tp, len(ref) - tp