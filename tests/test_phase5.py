"""Tests de la Phase 5 : python -m pytest tests -v"""
import numpy as np
import torch
import torch.nn as nn

from src.training.augmentation import augment_batch
from src.training.imbalance import (FocalLoss, class_counts, compute_class_weights,
                                    make_weighted_sampler, oversample_indices,
                                    undersample_indices)


def test_balanced_weights_formula_and_mean_one():
    y = np.array([0] * 90 + [1] * 10)
    w = compute_class_weights(y, 2).numpy()
    assert np.allclose(w, [100 / (2 * 90), 100 / (2 * 10)], atol=1e-5)
    assert np.isclose((w * class_counts(y, 2)).sum() / len(y), 1.0)


def test_all_methods_have_unit_mean_weight():
    y = np.array([0] * 900 + [1] * 80 + [2] * 20)
    for m in ("balanced", "sqrt", "effective"):
        w = compute_class_weights(y, 3, method=m).numpy()
        assert np.isclose((w * class_counts(y, 3)).sum() / len(y), 1.0, atol=1e-5)
        assert w[2] > w[1] > w[0]           # plus rare -> plus lourd


def test_missing_class_raises():
    try:
        compute_class_weights(np.array([0, 0, 1]), 3)
        assert False
    except ValueError:
        pass


def test_weighted_ce_normalisation_matches_manual():
    torch.manual_seed(0)
    logits, t = torch.randn(32, 4), torch.randint(0, 4, (32,))
    w = torch.tensor([0.3, 13.0, 3.0, 26.0])
    ce = nn.CrossEntropyLoss(weight=w)(logits, t)
    logp = torch.log_softmax(logits, 1)[torch.arange(32), t]
    manual = -(w[t] * logp).sum() / w[t].sum()
    assert torch.isclose(ce, manual, atol=1e-6)


def test_focal_gamma0_equals_cross_entropy():
    torch.manual_seed(1)
    logits, t = torch.randn(64, 4), torch.randint(0, 4, (64,))
    w = torch.tensor([0.3, 13.0, 3.0, 26.0])
    assert torch.isclose(FocalLoss(0.0)(logits, t), nn.CrossEntropyLoss()(logits, t), atol=1e-6)
    assert torch.isclose(FocalLoss(0.0, w)(logits, t), nn.CrossEntropyLoss(weight=w)(logits, t), atol=1e-6)


def test_focal_downweights_easy_examples():
    easy, hard = torch.tensor([[6.0, 0.0]]), torch.tensor([[0.2, 0.0]])
    t = torch.tensor([0])
    ratio_easy = FocalLoss(2.0)(easy, t) / nn.CrossEntropyLoss()(easy, t)
    ratio_hard = FocalLoss(2.0)(hard, t) / nn.CrossEntropyLoss()(hard, t)
    assert ratio_easy < 0.01 and 0.1 < ratio_hard < 0.5   # facile : quasi annulé ; difficile : conservé
    assert ratio_easy < ratio_hard


def test_resampling_indices_stay_inside_train_and_balance():
    rng = np.random.default_rng(0)
    y_train = np.array([0] * 100 + [1] * 10 + [2] * 5)
    o = oversample_indices(y_train, rng)
    assert o.min() >= 0 and o.max() < len(y_train)
    assert np.all(np.bincount(y_train[o]) == 100)
    u = undersample_indices(y_train, rng, max_per_class=8)
    assert np.array_equal(np.bincount(y_train[u]), [8, 8, 5])


def test_weighted_sampler_balances_in_expectation():
    torch.manual_seed(0)
    y = np.array([0] * 3000 + [1] * 100 + [2] * 50)
    drawn = np.array(list(make_weighted_sampler(y)))
    frac = np.bincount(y[drawn], minlength=3) / len(drawn)
    assert np.all(np.abs(frac - 1 / 3) < 0.05)


def test_augmentation_identity_when_disabled():
    x = torch.randn(8, 1, 200)
    out = augment_batch(x, max_shift=0, scale_range=(1, 1), noise_std=0, wander_amp=0)
    assert torch.allclose(out, x)


def test_augmentation_shift_is_bounded_and_shape_kept():
    torch.manual_seed(0)
    x = torch.zeros(64, 1, 200)
    x[:, 0, 90] = 1.0                                   # pic R à l'indice 90
    out = augment_batch(x, max_shift=10, scale_range=(1, 1), noise_std=0, wander_amp=0)
    assert out.shape == x.shape
    pos = out[:, 0, :].argmax(dim=1)
    assert ((pos >= 80) & (pos <= 100)).all()


def test_only_classes_leaves_other_classes_untouched():
    torch.manual_seed(0)
    x, y = torch.randn(40, 1, 200), torch.tensor([0, 1] * 20)
    out = augment_batch(x, y, only_classes=[1])
    assert torch.equal(out[y == 0], x[y == 0])
    assert not torch.equal(out[y == 1], x[y == 1])