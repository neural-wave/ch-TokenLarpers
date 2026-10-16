"""Training procedure: every data-driven choice happens inside `fit` (selection policy, Tier A)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pvdetect.features import CAPACITY_FEATURES, FEATURE_NAMES
from pvdetect.model import CURRENT_VERSION, fit_logreg, fit_ridge, standardize_params
from pvdetect.pipeline import FeatureTable, baseline_feature_frame, capacity_matrix


def _labels_for(labels: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    return labels.set_index("customer_id").loc[ids]


def _constant_linear(X: np.ndarray, intercept: float) -> dict:
    p = standardize_params(X)
    p.update(coef=[0.0] * X.shape[1], intercept=float(intercept))
    return p


def export_enabled_for(table: FeatureTable, ids: list[str]) -> bool:
    """Signed-data test: export features only when enough customers have ≥1% negative readings."""
    signed = sum(1 for c in ids if table.customers.get(c) is not None and table.customers[c].neg_frac >= 0.01)
    return signed >= max(2, 0.1 * len(ids))


def fit(table: FeatureTable, labels: pd.DataFrame, train_ids: list[str], *, inner: bool = True) -> dict:
    lab = _labels_for(labels, train_ids)
    F, _ = baseline_feature_frame(table, train_ids)
    y = lab["pv_installed"].to_numpy(bool)
    X = F[list(FEATURE_NAMES)].to_numpy(float)
    if y.all() or not y.any():
        clf = _constant_linear(X, 50.0 if y.all() else -50.0)
    else:
        clf = fit_logreg(X, y, C=1.0)
    kwp = lab["pv_capacity_kwp"].to_numpy(float)
    pos = y & np.isfinite(kwp) & (kwp > 0)
    fallback = float(np.median(kwp[pos])) if pos.any() else 5.0
    Xc = capacity_matrix(F, list(CAPACITY_FEATURES))
    ridge = fit_ridge(Xc[pos], np.log(kwp[pos])) if pos.sum() >= 3 else _constant_linear(Xc, np.log(fallback))
    return {
        "version": CURRENT_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "capacity_feature_names": list(CAPACITY_FEATURES),
        "export_enabled": False,
        "classifier": {"kind": "logreg", "threshold": 0.5, "logreg": clf, "catboost_file": None},
        "capacity": {
            "kind": "ridge",
            "ridge": ridge,
            "ridge_export": None,
            "catboost_file": None,
            "kappa": 1.0,
            "fallback_kwp": fallback,
        },
        "date": {"offset_event": 0},
        "selection": {"classifier": "logreg", "capacity": "ridge", "threshold": 0.5, "kappa": 1.0, "offset_event": 0},
        "extensions": {"enabled": False},
    }
