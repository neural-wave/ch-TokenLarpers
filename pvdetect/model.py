"""Model artifact (JSON + sibling files) and numpy-only linear model evaluation."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

CURRENT_VERSION = 2


def standardize_params(X: np.ndarray) -> dict:
    """Median imputation (0 for all-NaN columns) followed by z-scoring (zero scale -> 1)."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        impute = np.nanmedian(X, axis=0) if X.size else np.zeros(X.shape[1])
    impute = np.where(np.isfinite(impute), impute, 0.0)
    Xi = np.where(np.isnan(X), impute, X)
    mean = Xi.mean(axis=0) if len(Xi) else np.zeros(X.shape[1])
    scale = Xi.std(axis=0) if len(Xi) else np.ones(X.shape[1])
    scale = np.where(scale > 0, scale, 1.0)
    return {"impute": impute.tolist(), "mean": mean.tolist(), "scale": scale.tolist()}


def standardize(params: dict, X: np.ndarray) -> np.ndarray:
    imp = np.asarray(params["impute"])
    Xi = np.where(np.isnan(X), imp, X)
    return (Xi - np.asarray(params["mean"])) / np.asarray(params["scale"])


def linear_score(params: dict, X: np.ndarray) -> np.ndarray:
    return standardize(params, X) @ np.asarray(params["coef"]) + float(params["intercept"])


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def fit_logreg(X: np.ndarray, y: np.ndarray, C: float, sample_weight: np.ndarray | None = None) -> dict:
    from sklearn.linear_model import LogisticRegression

    p = standardize_params(X)
    est = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    est.fit(standardize(p, X), y.astype(int), sample_weight=sample_weight)
    p.update(coef=est.coef_[0].tolist(), intercept=float(est.intercept_[0]))
    return p


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict:
    from sklearn.linear_model import Ridge

    p = standardize_params(X)
    est = Ridge(alpha=alpha)
    est.fit(standardize(p, X), y)
    p.update(coef=est.coef_.tolist(), intercept=float(est.intercept_))
    return p


def save_model(model: dict, path: str | Path) -> None:
    """Write JSON; objects under `_objects` (CatBoost) are saved as sibling files named in the JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    objects = model.get("_objects", {})
    for name, obj in objects.items():
        obj.save_model(str(path.parent / f"{name}.cbm"), format="cbm")
    data = {k: v for k, v in model.items() if k != "_objects"}
    path.write_text(json.dumps(data, indent=2))


def referenced_files(model: dict) -> list[str]:
    files = []
    for section in ("classifier", "capacity"):
        f = (model.get(section) or {}).get("catboost_file")
        if f:
            files.append(f)
    return files


def load_model(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model artifact not found: {path}")
    model = json.loads(path.read_text())
    v = model.get("version")
    if v != CURRENT_VERSION:
        raise ValueError(f"unsupported model version: {v}")
    objects = {}
    for f in referenced_files(model):
        fp = path.parent / f
        if not fp.exists():
            raise FileNotFoundError(f"model artifact not found: {fp}")
        name = Path(f).stem
        from catboost import CatBoostClassifier, CatBoostRegressor

        obj = CatBoostClassifier() if name.endswith("clf") else CatBoostRegressor()
        obj.load_model(str(fp))
        objects[name] = obj
    model["_objects"] = objects
    return model
