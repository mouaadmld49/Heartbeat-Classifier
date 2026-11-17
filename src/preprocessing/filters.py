"""Filtrage du signal ECG (Butterworth, phase nulle)."""
import numpy as np
from scipy import signal


def design_bandpass(low_hz, high_hz, fs, order=4):
    """Passe-bande Butterworth en sections d'ordre 2 (SOS), stable numériquement."""
    return signal.butter(order, [low_hz, high_hz], btype="bandpass",
                         fs=fs, output="sos")


def bandpass(x, fs, low_hz=0.5, high_hz=40.0, order=4):
    """Filtrage avant-arrière (sosfiltfilt) : aucun décalage de phase."""
    sos = design_bandpass(low_hz, high_hz, fs, order)
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def notch(x, fs, f0=60.0, q=30.0):
    """Coupe-bande étroit à f0 (largeur de bande = f0 / q)."""
    b, a = signal.iirnotch(f0, q, fs=fs)
    return signal.filtfilt(b, a, np.asarray(x, dtype=np.float64))


def clean_ecg(x, fs, low_hz=0.5, high_hz=40.0, order=4, notch_hz=None):
    """Pipeline de nettoyage : passe-bande, puis notch optionnel."""
    y = bandpass(x, fs, low_hz, high_hz, order)
    if notch_hz:
        y = notch(y, fs, notch_hz)
    return y.astype(np.float32)