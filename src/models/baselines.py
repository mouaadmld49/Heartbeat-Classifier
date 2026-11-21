"""Baselines ML : Pipeline (imputation -> standardisation -> modèle).
Tout ce qui est appris l'est dans fit(), donc sur le train passé à fit()."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ..features.extract import feature_groups

MODEL_NAMES = ("logreg", "rf", "svm", "xgb")


class CappedSVC(ClassifierMixin, BaseEstimator):
    """SVM RBF entraîné sur au plus `cap_per_class` exemples par classe.

    Le plafonnement a lieu DANS fit() (donc sur le train du pli uniquement).
    Raison : le coût d'un SVM RBF croît au moins quadratiquement avec n.
    """

    def __init__(self, cap_per_class=4000, C=1.0, gamma="scale",
                 class_weight=None, seed=42):
        self.cap_per_class = cap_per_class
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.seed = seed

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.default_rng(self.seed)
        keep = []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            if len(idx) > self.cap_per_class:
                idx = rng.choice(idx, size=self.cap_per_class, replace=False)
            keep.append(idx)
        keep = np.concatenate(keep)
        self.svc_ = SVC(C=self.C, gamma=self.gamma, class_weight=self.class_weight)
        self.svc_.fit(X[keep], y[keep])
        self.classes_ = self.svc_.classes_
        return self

    def predict(self, X):
        return self.svc_.predict(np.asarray(X))


def make_pipeline(name, params, weighted, seed=42):
    """Pipeline non ajusté. weighted=True : poids de classe 'balanced'."""
    cw = "balanced" if weighted else None
    if name == "logreg":
        clf = LogisticRegression(**params, class_weight=cw)
    elif name == "rf":
        clf = RandomForestClassifier(**params, class_weight=cw, random_state=seed)
    elif name == "svm":
        clf = CappedSVC(**params, class_weight=cw, seed=seed)
    elif name == "xgb":
        clf = XGBClassifier(**params, tree_method="hist", random_state=seed)
    else:
        raise ValueError(f"modèle inconnu : {name}")
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("clf", clf)])


def fit_pipeline(pipe, name, X, y, weighted, n_classes=4):
    """Ajuste le pipeline. XGBoost reçoit les poids via sample_weight."""
    present = set(np.unique(y).tolist())
    if present != set(range(n_classes)):
        raise ValueError(f"Classes présentes dans l'entraînement : {sorted(present)}")
    if name == "xgb" and weighted:
        pipe.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    else:
        pipe.fit(X, y)
    return pipe


def make_group_folds(y, groups, n_splits=5, seed=42):
    """Plis stratifiés où un record n'est jamais dans le train ET le test d'un même pli."""
    y, groups = np.asarray(y), np.asarray(groups)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(y)), y, groups))
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te]), "fuite : record partagé"
    return folds


def feature_sets(columns):
    """Jeux de features pour l'ablation de H4 (colonnes de features uniquement)."""
    g = feature_groups(columns)
    base = g["morph"] + g["stat"] + g["freq"]
    return {"all": list(columns), "no_rr": base,
            "rr_only": g["rr"], "morph_rr": g["morph"] + g["rr"]}