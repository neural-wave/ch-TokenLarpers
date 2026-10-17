from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pvdetect.model import load_model, save_model
from pvdetect.procedure import _select
from pvdetect.selection import choose_kappa, choose_threshold


def _inner_oof(ridge_factor_pos: float, ridge_factor_low: float) -> pd.DataFrame:
    """Hand-built pooled inner OOF frame: two confident positives (dates 40 days off, ridge capacity
    `ridge_factor_pos`× true), one low-probability positive (exact date, ridge `ridge_factor_low`× true)
    and two negatives scored just above it. CatBoost scores equal logreg (no margin) and no CatBoost
    capacity, so only the threshold, κ and the rescoring matter."""
    t_date = pd.to_datetime(["2023-05-01", "2023-06-01", "2023-07-01", None, None])
    pred_date = pd.to_datetime(["2023-06-10", "2023-07-11", "2023-07-01", "2023-01-01", "2023-01-01"])
    true_k = np.array([8.0, 10.0, 6.0, np.nan, np.nan])
    factor = np.array([ridge_factor_pos, ridge_factor_pos, ridge_factor_low, 1.0, 1.0])
    prob = np.array([0.90, 0.90, 0.40, 0.42, 0.44])
    return pd.DataFrame(
        {
            "p_logreg": prob,
            "p_catboost": prob,
            "kwp_ridge": np.nan_to_num(true_k, nan=5.0) * factor,
            "kwp_catboost": np.nan,
            "regime": "event",
            "date_weak": pred_date,
            "date_first": pred_date,
            "t_pv": [True, True, True, False, False],
            "t_date": t_date,
            "t_kwp": true_k,
        },
        index=pd.Index(list("abcde"), name="customer_id"),
    )


def test_threshold_is_rescored_on_final_capacity_outputs():
    """Ridge (κ=1) favours including the low-probability positive (t=0.40); after κ=0.80 the confident
    positives earn capacity credit, and including it plus two false positives no longer pays (t=0.50)."""
    sel = _select(_inner_oof(1.5, 1.0))
    assert sel["threshold_initial"] == 0.40
    assert sel["kappa"] == 0.80
    assert sel["threshold"] == 0.50
    # S = 2·Σ_TP(70+15d+15p) / (N_pred_pos + N_true_pos)
    assert sel["inner_score_initial"] == pytest.approx(2 * (70 + 70 + 100) / (5 + 3))
    assert sel["inner_score"] == pytest.approx(2 * (79 + 79) / (2 + 3))


def test_threshold_unchanged_when_final_outputs_agree():
    sel = _select(_inner_oof(1.0, 1.0))
    assert sel["kappa"] == 1.0
    assert sel["threshold"] == sel["threshold_initial"]
    assert sel["inner_score"] == pytest.approx(sel["inner_score_initial"])


def test_final_threshold_score_uses_submitted_capacity_bounds():
    oof = _inner_oof(1.5, 1.0)
    oof.loc[oof.t_pv, "t_kwp"] = 500.
    oof.loc[oof.t_pv, "kwp_ridge"] = [750., 750., 500.]
    selected = _select(oof)
    assert selected["threshold"] == .5
    # The two detected customers have exact clipped capacities, but date reward 0.
    assert selected["inner_score"] == pytest.approx(2 * (85 + 85) / (2 + 3))


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
