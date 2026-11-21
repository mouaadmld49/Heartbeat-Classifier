"""Tests de la Phase 6 : python -m pytest tests -v"""
import numpy as np
import pytest

from src.evaluation.metrics import classification_summary
from src.models.baselines import (CappedSVC, feature_sets, fit_pipeline,
                                  make_group_folds, make_pipeline)

PARAMS = {
    "logreg": {"C": 1.0, "max_iter": 200},
    "rf": {"n_estimators": 10, "min_samples_leaf": 1, "n_jobs": 1},
    "svm": {"C": 1.0, "gamma": "scale", "cap_per_class": 50},
    "xgb": {"n_estimators": 5, "max_depth": 3, "learning_rate": 0.3,
            "subsample": 1.0, "colsample_bytree": 1.0},
}


def toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 4
    X = rng.normal(size=(n, 6))
    X[:, 0] += y                                  # signal simple
    X[rng.random(X.shape) < 0.05] = np.nan        # valeurs manquantes
    return X, y


def test_group_folds_have_no_record_overlap():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 30).astype(str)
    y = rng.integers(0, 4, size=len(groups))
    folds = make_group_folds(y, groups, n_splits=5, seed=1)
    all_test = np.concatenate([te for _, te in folds])
    assert len(all_test) == len(np.unique(all_test)) == len(y)   # chaque battement testé une fois
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])


def test_feature_sets_ablation_contents():
    cols = ["m_a", "s_a", "f_a", "rr_prev", "rr_next"]
    fs = feature_sets(cols)
    assert fs["all"] == cols
    assert not any(c.startswith("rr_") for c in fs["no_rr"])
    assert fs["rr_only"] == ["rr_prev", "rr_next"]
    assert set(fs["morph_rr"]) == {"m_a", "rr_prev", "rr_next"}


def test_metrics_always_majority_matches_formula():
    y = np.array([0] * 90 + [1] * 4 + [2] * 4 + [3] * 2)
    s = classification_summary(y, np.zeros_like(y))
    assert np.isclose(s["accuracy"], 0.9)
    assert np.isclose(s["macro_f1"], (2 * 0.9 * 1.0 / 1.9) / 4, atol=1e-6)
    assert s["recall_S"] == 0 and s["recall_V"] == 0 and s["recall_F"] == 0


def test_recall_and_support_per_class():
    s = classification_summary([0, 0, 1, 1, 2, 3], [0, 1, 1, 1, 2, 0])
    assert (s["recall_N"], s["recall_S"], s["recall_V"], s["recall_F"]) == (0.5, 1.0, 1.0, 0.0)
    assert s["support_N"] == 2 and s["support_F"] == 1


def test_fit_pipeline_rejects_missing_class():
    X, y = toy()
    keep = y != 3
    with pytest.raises(ValueError):
        fit_pipeline(make_pipeline("logreg", PARAMS["logreg"], True), "logreg", X[keep], y[keep], True)


@pytest.mark.parametrize("name", ["logreg", "rf", "svm", "xgb"])
@pytest.mark.parametrize("weighted", [False, True])
def test_pipelines_handle_nan_and_predict(name, weighted):
    X, y = toy()
    pipe = make_pipeline(name, PARAMS[name], weighted)
    fit_pipeline(pipe, name, X, y, weighted)
    pred = pipe.predict(X)
    assert pred.shape == (len(y),) and set(np.unique(pred)) <= {0, 1, 2, 3}


def test_capped_svc_limits_training_size():
    X, y = toy(400)
    X = np.nan_to_num(X)
    m = CappedSVC(cap_per_class=30).fit(X, y)
    assert len(m.svc_.support_) <= 4 * 30
    assert list(m.classes_) == [0, 1, 2, 3]