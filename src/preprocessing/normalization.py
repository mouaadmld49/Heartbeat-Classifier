"""Normalisation : par record (sans fit) ou globale (fit sur le train uniquement)."""
import json

import numpy as np

EPS = 1e-8


def robust_scale_record(x):
    """(x - médiane) / IQR calculés sur le signal du record seul (pas de fuite)."""
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    q75, q25 = np.percentile(x, [75, 25])
    return ((x - med) / (q75 - q25 + EPS)).astype(np.float32)


class GlobalScaler:
    """Scaler global. fit() DOIT être appelé sur le train uniquement."""
    METHODS = ("zscore", "minmax", "robust")

    def __init__(self, method="robust"):
        if method not in self.METHODS:
            raise ValueError(f"method doit être dans {self.METHODS}")
        self.method = method
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.method == "zscore":
            center, scale = X.mean(), X.std()
        elif self.method == "minmax":
            center, scale = X.min(), X.max() - X.min()
        else:
            q25, q50, q75 = np.percentile(X, [25, 50, 75])
            center, scale = q50, q75 - q25
        self.center_, self.scale_ = float(center), max(float(scale), EPS)
        return self

    def transform(self, X):
        if self.center_ is None:
            raise RuntimeError("Appelez fit() sur le train avant transform().")
        X = np.asarray(X, dtype=np.float64)
        return ((X - self.center_) / self.scale_).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"method": self.method, "center": self.center_,
                       "scale": self.scale_}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        s = cls(d["method"])
        s.center_, s.scale_ = d["center"], d["scale"]
        return s