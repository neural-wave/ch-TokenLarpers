from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pvdetect.model import load_model, save_model
from pvdetect.selection import choose_kappa, choose_threshold


def test_threshold_tie_goes_closest_to_half():
    truth = pd.DataFrame(
        {
            "customer_id": list("abcd"),
            "pv_installed": [True, True, False, False],
            "pv_installation_date": pd.to_datetime(["2023-01-01", "2023-06-01", None, None]),
            "pv_capacity_kwp": [5.0, 8.0, np.nan, np.nan],
        }
    )
    oof = pd.DataFrame(
        {
            "customer_id": list("abcd"),
            "prob": [0.30, 0.35, 0.10, 0.20],
            "pv_installation_date": pd.to_datetime(["2023-01-01", "2023-06-01", "2023-01-01", "2023-01-01"]),
            "pv_capacity_kwp": [5.0, 8.0, 4.0, 4.0],
        }
    )
    assert choose_threshold(oof, truth) == 0.30


def test_kappa_corrects_systematic_overestimate():
    true = np.array([4.0, 6.0, 10.0])
    assert choose_kappa(true * 1.1, true) == 0.90
    assert choose_kappa(true, true) == 1.0


def test_old_schema_rejected(tmp_path):
    p = tmp_path / "model.json"
    p.write_text(json.dumps({"version": 1}))
    with pytest.raises(ValueError, match="unsupported model version: 1"):
        load_model(p)


def test_missing_sibling_artifact_rejected(tmp_path):
    from pvdetect.model import CURRENT_VERSION

    p = tmp_path / "model.json"
    p.write_text(json.dumps({"version": CURRENT_VERSION, "classifier": {"catboost_file": "catboost_clf.cbm"}}))
    with pytest.raises(FileNotFoundError, match="model artifact not found"):
        load_model(p)


def test_catboost_roundtrip_identical(tmp_path):
    from catboost import CatBoostClassifier

    from pvdetect.procedure import CB_PARAMS

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    X[rng.random(X.shape) < 0.1] = np.nan
    y = (np.nan_to_num(X[:, 0]) > 0).astype(int)
    a = CatBoostClassifier(auto_class_weights="Balanced", **CB_PARAMS).fit(X, y)
    b = CatBoostClassifier(auto_class_weights="Balanced", **CB_PARAMS).fit(X, y)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
    from pvdetect.model import CURRENT_VERSION

    model = {"version": CURRENT_VERSION, "classifier": {"catboost_file": "catboost_clf.cbm"}, "_objects": {"catboost_clf": a}}
    save_model(model, tmp_path / "model.json")
    loaded = load_model(tmp_path / "model.json")
    np.testing.assert_array_equal(a.predict_proba(X), loaded["_objects"]["catboost_clf"].predict_proba(X))
