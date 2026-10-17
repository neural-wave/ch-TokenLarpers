"""Model artifact (JSON + sibling files) and numpy-only linear model evaluation."""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from pathlib import Path, PurePosixPath
import numpy as np

CURRENT_VERSION = 3


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


def fit_logreg(X: np.ndarray, y: np.ndarray, C: float) -> dict:
    from sklearn.linear_model import LogisticRegression

    p = standardize_params(X)
    est = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    est.fit(standardize(p, X), y.astype(int))
    p.update(coef=est.coef_[0].tolist(), intercept=float(est.intercept_[0]))
    return p


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict:
    from sklearn.linear_model import Ridge

    p = standardize_params(X)
    est = Ridge(alpha=alpha)
    est.fit(standardize(p, X), y)
    p.update(coef=est.coef_.tolist(), intercept=float(est.intercept_))
    return p


def _safe_reference(name: str) -> str:
    p = PurePosixPath(name)
    if (
        p.is_absolute()
        or ".." in p.parts
        or "\\" in name
        or any(c in name for c in "\r\n\x00")
        or not name
        or p.as_posix() != name
    ):
        raise ValueError(f"unsafe model artifact reference: {name!r}")
    return name


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(model: dict, path: str | Path) -> None:
    """Write JSON and only the exact sibling files named by the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    objects = model.get("_objects", {})
    for name, obj in objects.items():
        if name not in {"catboost_clf", "catboost_cap"} or not hasattr(obj, "save_model"):
            raise ValueError(f"unexpected model object: {name}")
        obj.save_model(str(path.parent / f"{name}.cbm"), format="cbm")
    embedding = model.get("embedding")
    if embedding is not None:
        source = model.get("_encoder_source")
        if not source:
            raise ValueError("selected embedding model has no encoder source")
        refs = embedding.get("encoder_files")
        if not isinstance(refs, dict) or set(refs) != {"weights", "metadata"}:
            raise ValueError("embedding encoder_files must name weights and metadata")
        if len(set(refs.values())) != 2:
            raise ValueError("duplicate encoder artifact reference")
        for key in ("weights", "metadata"):
            ref = _safe_reference(refs[key])
            if ref not in {"encoder/encoder.pt", "encoder/encoder.json"}:
                raise ValueError("encoder references must be encoder/encoder.pt and encoder/encoder.json")
            name = Path(ref).name
            src = Path(source) / name
            if not src.is_file():
                raise FileNotFoundError(f"encoder artifact not found: {src}")
            dst = path.parent / ref
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        hashes = embedding.setdefault("encoder_hashes", {})
        hashes.clear()
        hashes.update({ref: _sha256(path.parent / ref) for ref in refs.values()})
    data = {k: v for k, v in model.items() if not k.startswith("_")}
    path.write_text(json.dumps(data, indent=2))


def referenced_files(model: dict) -> list[str]:
    files = []
    for section in ("classifier", "capacity"):
        f = (model.get(section) or {}).get("catboost_file")
        if f:
            files.append(_safe_reference(f))
    embedding = model.get("embedding")
    if embedding is not None:
        refs = embedding.get("encoder_files")
        if not isinstance(refs, dict) or set(refs) != {"weights", "metadata"}:
            raise ValueError("embedding encoder_files must name weights and metadata")
        files.extend(_safe_reference(refs[k]) for k in ("weights", "metadata"))
    if len(files) != len(set(files)):
        raise ValueError("duplicate model artifact reference")
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
        if not fp.is_file() or fp.is_symlink():
            raise FileNotFoundError(f"model artifact not found: {fp}")
        name = Path(f).stem
        if name in {"encoder", ""}:
            continue
        from catboost import CatBoostClassifier, CatBoostRegressor

        obj = CatBoostClassifier() if name.endswith("clf") else CatBoostRegressor()
        obj.load_model(str(fp))
        objects[name] = obj
    embedding = model.get("embedding")
    if embedding is not None:
        refs = embedding.get("encoder_files", {})
        if refs != {"weights": "encoder/encoder.pt", "metadata": "encoder/encoder.json"}:
            raise ValueError("encoder references must be encoder/encoder.pt and encoder/encoder.json")
        hashes = embedding.get("encoder_hashes", {})
        for key in ("weights", "metadata"):
            name = _safe_reference(refs[key])
            fp = path.parent / name
            expected = hashes.get(name)
            if not fp.is_file() or fp.is_symlink() or not expected or _sha256(fp) != expected:
                raise ValueError(f"encoder artifact hash mismatch: {name}")
        from pvdetect.embedding import FrozenEncoder
        enc_dir = path.parent / "encoder"
        if not enc_dir.is_dir() or enc_dir.is_symlink():
            raise FileNotFoundError(f"model artifact not found: {enc_dir}")
        model["_embedding_encoder"] = FrozenEncoder.load(enc_dir)
    model["_objects"] = objects
    return model
