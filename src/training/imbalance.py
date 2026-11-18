"""Outils contre le déséquilibre. Les étiquettes reçues doivent être celles du TRAIN."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


def class_counts(y, n_classes):
    return np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes)


def compute_class_weights(y_train, n_classes, method="balanced", beta=0.999):
    """Poids par classe, normalisés pour que le poids moyen par échantillon vaille 1.

    balanced : N / (C * n_c)      sqrt : 1/sqrt(n_c)      effective : (1-beta)/(1-beta^n_c)
    """
    n = class_counts(y_train, n_classes).astype(np.float64)
    if (n == 0).any():
        raise ValueError(f"Classe absente du train : effectifs = {n.tolist()}")
    if method == "balanced":
        w = 1.0 / n
    elif method == "sqrt":
        w = 1.0 / np.sqrt(n)
    elif method == "effective":
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
    else:
        raise ValueError("method doit être balanced, sqrt ou effective")
    w = w * n.sum() / (w * n).sum()            # poids moyen par échantillon = 1
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    """FL = -(1 - p_t)^gamma * log p_t, avec poids de classe optionnels.

    Normalisation identique à CrossEntropyLoss(weight=...) : division par la
    somme des poids des cibles. Avec gamma = 0 on retrouve la cross-entropy.
    """

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        logpt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -((1.0 - logpt.exp()) ** self.gamma) * logpt
        if self.weight is not None:
            w = self.weight[target]
            return (loss * w).sum() / w.sum()
        return loss.mean()


def oversample_indices(y, rng, target_per_class=None):
    """Indices (RELATIFS au tableau y passé, donc au train) avec duplication des minoritaires."""
    y = np.asarray(y)
    groups = [np.where(y == c)[0] for c in np.unique(y)]
    target = target_per_class or max(len(g) for g in groups)
    out = []
    for g in groups:
        out.append(g)
        if len(g) < target:
            out.append(rng.choice(g, size=target - len(g), replace=True))
    out = np.concatenate(out)
    rng.shuffle(out)
    return out


def undersample_indices(y, rng, max_per_class):
    """Indices (relatifs à y) en plafonnant chaque classe à max_per_class."""
    y = np.asarray(y)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out


def make_weighted_sampler(y_train, power=1.0):
    """Sampler à tirage avec remise : proba ∝ (1/n_c)^power.

    power = 1 : classes équilibrées en espérance ; power = 0.5 : rééquilibrage partiel.
    À ne PAS combiner avec des poids de classe dans la perte.
    """
    y = np.asarray(y_train, dtype=np.int64)
    n = np.bincount(y).astype(np.float64)
    w = (1.0 / n[y]) ** power
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=len(y), replacement=True)