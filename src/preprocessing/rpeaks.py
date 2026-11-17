"""Détection des pics R (inspirée de Pan & Tompkins, 1985) et évaluation."""
import numpy as np
from scipy import signal


def detect_r_peaks(x, fs=360, refractory_s=0.25, thr_ratio=0.2):
    """Retourne les indices des pics R détectés.

    thr_ratio est un seuil heuristique (fraction du 99,5e percentile de
    l'énergie intégrée) : à ajuster si la sensibilité ou le PPV sont faibles.
    """
    x = np.asarray(x, dtype=np.float64)
    sos = signal.butter(2, [5, 15], btype="bandpass", fs=fs, output="sos")
    f = signal.sosfiltfilt(sos, x)               # isole l'énergie du QRS
    energy = np.gradient(f) ** 2                 # dérivée puis carré
    w = int(0.15 * fs)
    mwi = np.convolve(energy, np.ones(w) / w, mode="same")  # intégration
    thr = thr_ratio * np.percentile(mwi, 99.5)
    cand, _ = signal.find_peaks(mwi, height=thr, distance=int(refractory_s * fs))

    r = int(0.075 * fs)                          # affinage autour du candidat
    peaks = []
    for c in cand:
        lo, hi = max(0, c - r), min(len(f), c + r + 1)
        peaks.append(lo + int(np.argmax(np.abs(f[lo:hi]))))
    return np.unique(peaks)


def evaluate_detection(ref, det, tol):
    """Apparie détections et références (tolérance tol, en échantillons).

    Retourne (TP, FP, FN). Chaque détection ne peut servir qu'une fois.
    """
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