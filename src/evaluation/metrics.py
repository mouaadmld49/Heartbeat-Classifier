"""Métriques de classification (N, S, V, F)."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

CLASSES = ["N", "S", "V", "F"]


def classification_summary(y_true, y_pred, classes=CLASSES):
    """Dictionnaire de métriques globales et par classe.

    macro_f1     : moyenne des F1 des 4 classes
    macro_f1_nsv : moyenne des F1 de N, S, V (critère de sélection, F exclu)
    Une prédiction F sur un battement N/S/V reste une erreur pour sa vraie classe.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_f1_nsv": f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }
    for i, c in enumerate(classes):
        out[f"recall_{c}"], out[f"precision_{c}"] = r[i], p[i]
        out[f"f1_{c}"], out[f"support_{c}"] = f[i], int(s[i])
    return out


def confusion(y_true, y_pred, n_classes=4):
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))