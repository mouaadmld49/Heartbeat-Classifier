import numpy as np
from src.preprocessing.rpeaks import detect_r_peaks, evaluate_detection


def test_detector_adapts_to_local_amplitude():
    fs, n = 360, 40 * 360
    t = np.arange(n)
    centers = np.arange(2 * fs, n - 2 * fs, int(0.8 * fs))
    x = np.zeros(n)
    for c in centers:
        amp = 1.0 if c < n // 2 else 0.3          # amplitude divisée par ~3 en 2e moitié
        x += amp * np.exp(-0.5 * ((t - c) / 6.0) ** 2)
    tp, fp, fn = evaluate_detection(centers, detect_r_peaks(x, fs), tol=int(0.15 * fs))
    assert tp >= len(centers) - 1 and fp <= 1